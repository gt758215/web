#!/usr/bin/env python2
# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.

import argparse
from collections import Counter
import logging
import math
import os
import Queue
import random
import re
import shutil
import sys
import threading
import time
from datetime import datetime

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import h5py
import lmdb
import numpy as np
import PIL.Image

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import digits.config  # noqa
from digits import utils, log  # noqa

# Import digits.config first to set the path to Caffe
import digits.tools.tf.caffe_tf_io as caffe_io  # noqa
import digits.tools.tf.caffe_tf_pb2 as caffe_pb2  # noqa

if digits.config.config_value('tensorflow')['enabled']:
    import tensorflow as tf
else:
    tf = None

logger = logging.getLogger('digits.tools.create_db')


class Error(Exception):
    pass


class BadInputFileError(Error):
    """Input file is empty"""
    pass


class ParseLineError(Error):
    """Failed to parse a line in the input file"""
    pass


class LoadError(Error):
    """Failed to load image[s]"""
    pass


class WriteError(Error):
    """Failed to write image[s]"""
    pass


class Hdf5DatasetExtendError(Error):
    """Failed to extend an hdf5 dataset"""
    pass


class DbWriter(object):
    """
    Abstract class for writing to databases
    """

    def __init__(self, output_dir, image_height, image_width, image_channels):
        self._dir = output_dir
        os.makedirs(output_dir)
        self._image_height = image_height
        self._image_width = image_width
        self._image_channels = image_channels
        self._count = 0

    def write_batch(self, batch):
        raise NotImplementedError

    def count(self):
        return self._count


class LmdbWriter(DbWriter):
    # TODO
    pass


class Hdf5Writer(DbWriter):
    """
    A class for writing to HDF5 files
    """
    LIST_FILENAME = 'list.txt'
    DTYPE = 'float32'

    def __init__(self, **kwargs):
        """
        Keyword arguments:
        compression -- the type of dataset compression
        dset_limit -- the dataset size limit
        """
        self._compression = kwargs.pop('compression', None)
        self._dset_limit = kwargs.pop('dset_limit', None)
        super(Hdf5Writer, self).__init__(**kwargs)
        self._db = None

        if self._dset_limit is not None:
            self._max_count = self._dset_limit / (
                self._image_height * self._image_width * self._image_channels)
        else:
            self._max_count = None

    def write_batch(self, batch):
        # convert batch to numpy arrays
        if batch[0][0].ndim == 2:
            # add channel axis for grayscale images
            data_batch = np.array([i[0][..., np.newaxis] for i in batch])
        else:
            data_batch = np.array([i[0] for i in batch])
        # Transpose to (channels, height, width)
        data_batch = data_batch.transpose((0, 3, 1, 2))
        label_batch = np.array([i[1] for i in batch])

        # first batch
        if self._db is None:
            self._create_new_file(len(batch))
            self._db['data'][:] = data_batch
            self._db['label'][:] = label_batch
            self._count += len(batch)
            return

        current_count = self._db['data'].len()

        # will fit in current dataset
        if current_count + len(batch) <= self._max_count:
            self._db['data'].resize(current_count + len(batch), axis=0)
            self._db['label'].resize(current_count + len(batch), axis=0)
            self._db['data'][-len(batch):] = data_batch
            self._db['label'][-len(batch):] = label_batch
            self._count += len(batch)
            return

        # calculate how many will fit in current dataset
        split = self._max_count - current_count

        if split > 0:
            # put what we can into the current dataset
            self._db['data'].resize(self._max_count, axis=0)
            self._db['label'].resize(self._max_count, axis=0)
            self._db['data'][-split:] = data_batch[:split]
            self._db['label'][-split:] = label_batch[:split]
            self._count += split

        self._create_new_file(len(batch) - split)
        self._db['data'][:] = data_batch[split:]
        self._db['label'][:] = label_batch[split:]
        self._count += len(batch) - split

    def _create_new_file(self, initial_count):
        assert self._max_count is None or initial_count <= self._max_count, \
            'Your batch size is too large for your dataset limit - %d vs %d' % \
            (initial_count, self._max_count)

        # close the old file
        if self._db is not None:
            self._db.close()
            mode = 'a'
        else:
            mode = 'w'

        # get the filename
        filename = self._new_filename()
        logger.info('Creating HDF5 database at "%s" ...' %
                    os.path.join(*filename.split(os.sep)[-2:]))

        # update the list
        with open(self._list_filename(), mode) as outfile:
            outfile.write('%s\n' % filename)

        # create the new file
        self._db = h5py.File(os.path.join(self._dir, filename), 'w')

        # initialize the datasets
        self._db.create_dataset('data',
                                (initial_count, self._image_channels,
                                 self._image_height, self._image_width),
                                maxshape=(self._max_count, self._image_channels,
                                          self._image_height, self._image_width),
                                chunks=True, compression=self._compression, dtype=self.DTYPE)
        self._db.create_dataset('label',
                                (initial_count,),
                                maxshape=(self._max_count,),
                                chunks=True, compression=self._compression, dtype=self.DTYPE)

    def _list_filename(self):
        return os.path.join(self._dir, self.LIST_FILENAME)

    def _new_filename(self):
        return '%s.h5' % self.count()


def _find_datadir_labels(input_file):
    """
    Search for subdirection under train or validation foler as labels name
    input_file:
        /imagenet/train/n01440764/n01440764_8834.JPEG 0
        /imagenet/train/n15075141/n15075141_45683.JPEG 999
    labels:
        [n01440764, n15075141]
    """
    with open(input_file) as infile:
        line = infile.readline().strip()
        if not line:
            raise ParseLineError
        match = re.match(r'(.+)(\btrain\b|\bvalidation\b)', line)
        if match is None:
            raise ParseLineError
        data_dir = match.group(1) + match.group(2)

    subdirs = []
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        for filename in os.listdir(data_dir):
            subdir = os.path.join(data_dir, filename)
            if os.path.isdir(subdir):
                subdirs.append(filename)
    else:
        logger.error('folder does not exist')
        return False
    subdirs.sort()

    filenames = []
    labels = []
    texts = []
    # leave label index 0 empty as a background class
    label_index = 1
    for text in subdirs:
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            logger.info('Finished finding files in %d of %d classes.' % (
                        label_index, len(labels)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    logger.info('Found %d JPEG files across %d labels inside %s.' %
                (len(filenames), len(subdirs), data_dir))

    return filenames, texts, labels


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        # force use CPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _is_png(filename):
    return filename.endswith('.png')


def _process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        logger.info('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _convert_to_example(filename, image_buffer, label, text, height, width):
    """Build an Example proto for an example.
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards, output_dir,
                               image_count, image_width, image_height):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            try:
                image_buffer, height, width = _process_image(filename, coder)
            except Exception as e:
                logger.warning(e)
                logger.warning('SKIPPED: Unexpected error while decoding %s.' % filename)
                continue

            # can't resize without decode
            #if (height != image_height or width != image_width):
            #    image_buffer = tf.image.resize_images(
            #        image_buffer,
            #        [image_height, image_width],
            #        tf.image.ResizeMethod.BILINEAR,
            #        align_corners=False)
            example = _convert_to_example(filename, image_buffer, label,
                                          text, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            image_count[thread_index] = counter

            if not counter % 1000:
                logger.info('%s [thread %d]: Processed %d of %d images in thread batch.' %
                            (datetime.now(), thread_index, counter, num_files_in_thread))

        writer.close()
        logger.info('%s [thread %d]: Wrote %d images to %s' %
                    (datetime.now(), thread_index, shard_counter, output_file))
        shard_counter = 0
        logger.info('%s [thread %d]: Wrote %d images to %d shards.' %
                    (datetime.now(), thread_index, counter, num_files_in_thread))


def create_tfrecords_db(input_file, output_dir,
              image_width, image_height, image_channels,
              backend,
              resize_mode=None,
              image_folder=None,
              shuffle=True,
              mean_files=None,
              delete_files=False,
              **kwargs):
    """ find labels and convert to tfrecords
    """
    num_threads = 2
    #num_shards = 2

    os.makedirs(output_dir)

    filenames, texts, labels =_find_datadir_labels(input_file)
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    num_shards = len(filenames) // 10000
    if num_shards = 0:
      num_shards = 1

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    logger.info('Launching %d threads for spacings: %s' % (num_threads, ranges))

    coord = tf.train.Coordinator()
    coder = ImageCoder()
    threads = []
    image_count = [0] * num_threads
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, "shard", filenames,
                texts, labels, num_shards, output_dir, image_count,
                image_width, image_height)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    wait_time = time.time()
    while sum(image_count) < len(filenames):
        if time.time() - wait_time > 2:
            logger.debug('Processed %d/%d' % (sum(image_count), len(filenames)))
            wait_time = time.time()
        time.sleep(0.2)

    # Wait for all the threads to terminate.
    coord.join(threads)
    logger.info('%s images written to database' % len(filenames))


def create_db(input_file, output_dir,
              image_width, image_height, image_channels,
              backend,
              resize_mode=None,
              image_folder=None,
              shuffle=True,
              mean_files=None,
              delete_files=False,
              **kwargs):
    """
    Create a database of images from a list of image paths
    Raises exceptions on errors

    Arguments:
    input_file -- a textfile containing labelled image paths
    output_dir -- the location to store the created database
    image_width -- image resize width
    image_height -- image resize height
    image_channels -- image channels
    backend -- the DB format (lmdb/hdf5)

    Keyword arguments:
    resize_mode -- passed to utils.image.resize_image()
    shuffle -- if True, shuffle the images in the list before creating
    mean_files -- a list of mean files to save
    delete_files -- if True, delete raw images after creation of database
    """
    # Validate arguments

    if not os.path.exists(input_file):
        raise ValueError('input_file does not exist')
    if os.path.exists(output_dir):
        logger.warning('removing existing database')
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            os.remove(output_dir)
    if image_width <= 0:
        raise ValueError('invalid image width')
    if image_height <= 0:
        raise ValueError('invalid image height')
    if image_channels not in [1, 3]:
        raise ValueError('invalid number of channels')
    if resize_mode not in [None, 'crop', 'squash', 'fill', 'half_crop']:
        raise ValueError('invalid resize_mode')
    if image_folder is not None and not os.path.exists(image_folder):
        raise ValueError('image_folder does not exist')
    if mean_files:
        for mean_file in mean_files:
            if os.path.exists(mean_file):
                logger.warning('overwriting existing mean file "%s"!' % mean_file)
            else:
                dirname = os.path.dirname(mean_file)
                if not dirname:
                    dirname = '.'
                if not os.path.exists(dirname):
                    raise ValueError('Cannot save mean file at "%s"' % mean_file)
    compute_mean = bool(mean_files)

    # Load lines from input_file into a load_queue

    load_queue = Queue.Queue()
    image_count = _fill_load_queue(input_file, load_queue, shuffle)

    # Start some load threads

    batch_size = _calculate_batch_size(image_count,
                                       bool(backend == 'hdf5'), kwargs.get('hdf5_dset_limit'),
                                       image_channels, image_height, image_width)
    num_threads = _calculate_num_threads(batch_size, shuffle)
    write_queue = Queue.Queue(2 * batch_size)
    summary_queue = Queue.Queue()

    for _ in xrange(num_threads):
        p = threading.Thread(target=_load_thread,
                             args=(load_queue, write_queue, summary_queue,
                                   image_width, image_height, image_channels,
                                   resize_mode, image_folder, compute_mean),
                             kwargs={'backend': backend,
                                     'encoding': kwargs.get('encoding', None)},
                             )
        p.daemon = True
        p.start()

    start = time.time()

    if backend == 'lmdb':
        _create_lmdb(image_count, write_queue, batch_size, output_dir,
                     summary_queue, num_threads,
                     mean_files, **kwargs)
    elif backend == 'hdf5':
        _create_hdf5(image_count, write_queue, batch_size, output_dir,
                     image_width, image_height, image_channels,
                     summary_queue, num_threads,
                     mean_files, **kwargs)
    elif backend == 'tfrecords':
        _create_tfrecords(image_count, write_queue, batch_size, output_dir,
                          summary_queue, num_threads,
                          mean_files, **kwargs)
    else:
        raise ValueError('invalid backend')

    if delete_files:
        # delete files
        deleted_files = 0
        distribution = Counter()
        with open(input_file) as infile:
            for line in infile:
                try:
                    # delete file
                    [path, label] = _parse_line(line, distribution)
                    os.remove(path)
                    deleted_files += 1
                except ParseLineError:
                    pass
                logger.info("Deleted " + str(deleted_files) + " files")
        logger.info('Database created after %d seconds.' % (time.time() - start))


def _create_tfrecords(image_count, write_queue, batch_size, output_dir,
                      summary_queue, num_threads,
                      mean_files=None,
                      encoding=None,
                      lmdb_map_size=None,
                      **kwargs):
    """
    Creates the TFRecords database(s)
    """
    LIST_FILENAME = 'list.txt'

    if not tf:
        raise ValueError("Can't create TFRecords as support for Tensorflow "
                         "is not enabled.")

    wait_time = time.time()
    threads_done = 0
    images_loaded = 0
    images_written = 0
    image_sum = None
    compute_mean = bool(mean_files)

    os.makedirs(output_dir)

    # We need shards to achieve good mixing properties because TFRecords
    # is a sequential/streaming reader, and has no random access.

    num_shards = 2 if image_count < 100000 else 128

    writers = []
    with open(os.path.join(output_dir, LIST_FILENAME), 'w') as outfile:
        for shard_id in xrange(num_shards):
            shard_name = 'SHARD_%03d.tfrecords' % (shard_id)
            filename = os.path.join(output_dir, shard_name)
            writers.append(tf.python_io.TFRecordWriter(filename))
            outfile.write('%s\n' % (filename))

    shard_id = 0
    while (threads_done < num_threads) or not write_queue.empty():

        # Send update every 2 seconds
        if time.time() - wait_time > 2:
            logger.debug('Processed %d/%d' % (images_written, image_count))
            wait_time = time.time()

        processed_something = False

        if not summary_queue.empty():
            result_count, result_sum = summary_queue.get()
            images_loaded += result_count
            # Update total_image_sum
            if compute_mean and result_count > 0 and result_sum is not None:
                if image_sum is None:
                    image_sum = result_sum
                else:
                    image_sum += result_sum
            threads_done += 1
            processed_something = True

        if not write_queue.empty():
            writers[shard_id].write(write_queue.get())
            shard_id += 1
            if shard_id >= num_shards:
                shard_id = 0
            images_written += 1
            processed_something = True

        if not processed_something:
            time.sleep(0.2)

    if images_loaded == 0:
        raise LoadError('no images loaded from input file')
    logger.debug('%s images loaded' % images_loaded)

    if images_written == 0:
        raise WriteError('no images written to database')
    logger.info('%s images written to database' % images_written)

    if compute_mean:
        _save_means(image_sum, images_written, mean_files)

    for writer in writers:
        writer.close()


def _create_lmdb(image_count, write_queue, batch_size, output_dir,
                 summary_queue, num_threads,
                 mean_files=None,
                 encoding=None,
                 lmdb_map_size=None,
                 **kwargs):
    """
    Create an LMDB

    Keyword arguments:
    encoding -- image encoding format
    lmdb_map_size -- the initial LMDB map size
    """
    wait_time = time.time()
    threads_done = 0
    images_loaded = 0
    images_written = 0
    image_sum = None
    batch = []
    compute_mean = bool(mean_files)
    keys = []

    db = lmdb.open(output_dir,
                   map_size=lmdb_map_size,
                   map_async=True,
                   max_dbs=0)

    while (threads_done < num_threads) or not write_queue.empty():

        # Send update every 2 seconds
        if time.time() - wait_time > 2:
            logger.debug('Processed %d/%d' % (images_written, image_count))
            wait_time = time.time()

        processed_something = False

        if not summary_queue.empty():
            result_count, result_sum = summary_queue.get()
            images_loaded += result_count
            # Update total_image_sum
            if compute_mean and result_count > 0 and result_sum is not None:
                if image_sum is None:
                    image_sum = result_sum
                else:
                    image_sum += result_sum
            threads_done += 1
            processed_something = True

        if not write_queue.empty():
            datum = write_queue.get()
            batch.append(datum)

            if len(batch) == batch_size:
                _write_batch_lmdb(db, batch, keys, images_written)
                images_written += len(batch)
                batch = []
            processed_something = True

        if not processed_something:
            time.sleep(0.2)

    if len(batch) > 0:
        _write_batch_lmdb(db, batch, keys, images_written)
        images_written += len(batch)

    # Keys Saver
    import cPickle as pickle
    keys.sort()
    pickle.dump(keys, open(output_dir + '/keys.mdb', "wb"), protocol=True)

    if images_loaded == 0:
        raise LoadError('no images loaded from input file')
    logger.debug('%s images loaded' % images_loaded)

    if images_written == 0:
        raise WriteError('no images written to database')
    logger.info('%s images written to database' % images_written)

    if compute_mean:
        _save_means(image_sum, images_written, mean_files)

    db.close()


def _create_hdf5(image_count, write_queue, batch_size, output_dir,
                 image_width, image_height, image_channels,
                 summary_queue, num_threads,
                 mean_files=None,
                 compression=None,
                 hdf5_dset_limit=None,
                 **kwargs):
    """
    Create an HDF5 file

    Keyword arguments:
    compression -- dataset compression format
    """
    wait_time = time.time()
    threads_done = 0
    images_loaded = 0
    images_written = 0
    image_sum = None
    batch = []
    compute_mean = bool(mean_files)

    writer = Hdf5Writer(
        output_dir=output_dir,
        image_height=image_height,
        image_width=image_width,
        image_channels=image_channels,
        dset_limit=hdf5_dset_limit,
        compression=compression,
    )

    while (threads_done < num_threads) or not write_queue.empty():

        # Send update every 2 seconds
        if time.time() - wait_time > 2:
            logger.debug('Processed %d/%d' % (images_written, image_count))
            wait_time = time.time()

        processed_something = False

        if not summary_queue.empty():
            result_count, result_sum = summary_queue.get()
            images_loaded += result_count
            # Update total_image_sum
            if compute_mean and result_count > 0 and result_sum is not None:
                if image_sum is None:
                    image_sum = result_sum
                else:
                    image_sum += result_sum
            threads_done += 1
            processed_something = True

        if not write_queue.empty():
            batch.append(write_queue.get())

            if len(batch) == batch_size:
                writer.write_batch(batch)
                images_written += len(batch)
                batch = []
            processed_something = True

        if not processed_something:
            time.sleep(0.2)

    if len(batch) > 0:
        writer.write_batch(batch)
        images_written += len(batch)

    assert images_written == writer.count()

    if images_loaded == 0:
        raise LoadError('no images loaded from input file')
    logger.debug('%s images loaded' % images_loaded)

    if images_written == 0:
        raise WriteError('no images written to database')
    logger.info('%s images written to database' % images_written)

    if compute_mean:
        _save_means(image_sum, images_written, mean_files)


def _fill_load_queue(filename, queue, shuffle):
    """
    Fill the queue with data from the input file
    Print the category distribution
    Returns the number of lines added to the queue

    NOTE: This can be slow on a large input file, but we need the total image
        count in order to report the progress, so we might as well read it all
    """
    total_lines = 0
    valid_lines = 0
    distribution = Counter()

    with open(filename) as infile:
        if shuffle:
            lines = infile.readlines()  # less memory efficient
            random.shuffle(lines)
            for line in lines:
                total_lines += 1
                try:
                    result = _parse_line(line, distribution)
                    valid_lines += 1
                    queue.put(result)
                except ParseLineError:
                    pass
        else:
            for line in infile:  # more memory efficient
                total_lines += 1
                try:
                    result = _parse_line(line, distribution)
                    valid_lines += 1
                    queue.put(result)
                except ParseLineError:
                    pass

    logger.debug('%s total lines in file' % total_lines)
    if valid_lines == 0:
        raise BadInputFileError('No valid lines in input file')
    logger.info('%s valid lines in file' % valid_lines)

    for key in sorted(distribution):
        logger.debug('Category %s has %d images.' % (key, distribution[key]))

    return valid_lines


def _parse_line(line, distribution):
    """
    Parse a line in the input file into (path, label)
    """
    line = line.strip()
    if not line:
        raise ParseLineError

    # Expect format - [/]path/to/file.jpg 123
    match = re.match(r'(.+)\s+(\d+)\s*$', line)
    if match is None:
        raise ParseLineError

    path = match.group(1)
    label = int(match.group(2))

    distribution[label] += 1

    return path, label


def _calculate_batch_size(image_count, is_hdf5=False, hdf5_dset_limit=None,
                          image_channels=None, image_height=None, image_width=None):
    """
    Calculates an appropriate batch size for creating this database
    """
    if is_hdf5 and hdf5_dset_limit is not None:
        return min(100, image_count, hdf5_dset_limit / (image_channels * image_height * image_width))
    else:
        return min(100, image_count)


def _calculate_num_threads(batch_size, shuffle):
    """
    Calculates an appropriate number of threads for creating this database
    """
    if shuffle:
        return min(10, int(round(math.sqrt(batch_size))))
    else:
        # XXX This is the only way to preserve order for now
        # This obviously hurts performance considerably
        return 1


def _load_thread(load_queue, write_queue, summary_queue,
                 image_width, image_height, image_channels,
                 resize_mode, image_folder, compute_mean,
                 backend=None, encoding=None):
    """
    Consumes items in load_queue
    Produces items to write_queue
    Stores cumulative results in summary_queue
    """
    images_added = 0
    if compute_mean:
        image_sum = _initial_image_sum(image_width, image_height, image_channels)
    else:
        image_sum = None

    while not load_queue.empty():
        try:
            path, label = load_queue.get(True, 0.05)
        except Queue.Empty:
            continue

        # prepend path with image_folder, if appropriate
        if not utils.is_url(path) and image_folder and not os.path.isabs(path):
            path = os.path.join(image_folder, path)

        try:
            image = utils.image.load_image(path)
        except utils.errors.LoadImageError as e:
            logger.warning('[%s %s] %s: %s' % (path, label, type(e).__name__, e))
            continue

        image = utils.image.resize_image(image,
                                         image_height, image_width,
                                         channels=image_channels,
                                         resize_mode=resize_mode,
                                         )

        if compute_mean:
            image_sum += image

        if backend == 'lmdb':
            datum = _array_to_datum(image, label, encoding)
            write_queue.put(datum)
        elif backend == 'tfrecords':
            tf_example = _array_to_tf_feature(image, label, encoding)
            write_queue.put(tf_example)
        else:
            write_queue.put((image, label))

        images_added += 1

    summary_queue.put((images_added, image_sum))


def _initial_image_sum(width, height, channels):
    """
    Returns an array of zeros that will be used to store the accumulated sum of images
    """
    if channels == 1:
        return np.zeros((height, width), np.float64)
    else:
        return np.zeros((height, width, channels), np.float64)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _array_to_tf_feature(image, label, encoding):
    """
    Creates a tensorflow Example from a numpy.ndarray
    if not encoding:
        image_raw = image.tostring()
        encoding_id = 0
    else:
        s = StringIO()
        if encoding == 'png':
            PIL.Image.fromarray(image).save(s, format='PNG')
            encoding_id = 1
        elif encoding == 'jpg':
            PIL.Image.fromarray(image).save(s, format='JPEG', quality=90)
            encoding_id = 2
        else:
            raise ValueError('Invalid encoding type')
        image_raw = s.getvalue()
    """
    encoding_id = 0
    depth = image.shape[2] if len(image.shape) > 2 else 1

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'height': _int64_feature(image.shape[0]),
                'width': _int64_feature(image.shape[1]),
                'depth': _int64_feature(depth),
                'label': _int64_feature(label),
                #'image_raw': _bytes_feature(image_raw),
                'image_raw': _float_array_feature(image.flatten()),
                'encoding':  _int64_feature(encoding_id),
                # @TODO(tzaman) - add bitdepth flag?
            }
        ))
    return example.SerializeToString()


def _array_to_datum(image, label, encoding):
    """
    Create a caffe Datum from a numpy.ndarray
    """
    if not encoding:
        # Transform to caffe's format requirements
        if image.ndim == 3:
            # Transpose to (channels, height, width)
            image = image.transpose((2, 0, 1))
            if image.shape[0] == 3:
                # channel swap
                # XXX see issue #59
                image = image[[2, 1, 0], ...]
        elif image.ndim == 2:
            # Add a channels axis
            image = image[np.newaxis, :, :]
        else:
            raise Exception('Image has unrecognized shape: "%s"' % image.shape)
        datum = caffe_io.array_to_datum(image, label)
    else:
        datum = caffe_pb2.Datum()
        if image.ndim == 3:
            datum.channels = image.shape[2]
        else:
            datum.channels = 1
        datum.height = image.shape[0]
        datum.width = image.shape[1]
        datum.label = label

        s = StringIO()
        if encoding == 'png':
            PIL.Image.fromarray(image).save(s, format='PNG')
        elif encoding == 'jpg':
            PIL.Image.fromarray(image).save(s, format='JPEG', quality=90)
        else:
            raise ValueError('Invalid encoding type')
        datum.data = s.getvalue()
        datum.encoded = True
    return datum


def _write_batch_lmdb(db, batch, keys, image_count):
    """
    Write a batch to an LMDB database
    """
    try:
        with db.begin(write=True) as lmdb_txn:
            for i, datum in enumerate(batch):
                key = '%08d_%d' % (image_count + i, datum.label)
                lmdb_txn.put(key, datum.SerializeToString())
                keys.append(key)

    except lmdb.MapFullError:
        # double the map_size
        curr_limit = db.info()['map_size']
        new_limit = curr_limit * 2
        try:
            db.set_mapsize(new_limit)  # double it
        except AttributeError as e:
            version = tuple(int(x) for x in lmdb.__version__.split('.'))
            if version < (0, 87):
                raise Error('py-lmdb is out of date (%s vs 0.87)' % lmdb.__version__)
            else:
                raise e
        # try again
        _write_batch_lmdb(db, batch, keys, image_count)


def _save_means(image_sum, image_count, mean_files):
    """
    Save mean[s] to file
    """
    mean = np.around(image_sum / image_count).astype(np.uint8)
    for mean_file in mean_files:
        if mean_file.lower().endswith('.npy'):
            np.save(mean_file, mean)
        elif mean_file.lower().endswith('.binaryproto'):
            data = mean
            # Transform to caffe's format requirements
            if data.ndim == 3:
                # Transpose to (channels, height, width)
                data = data.transpose((2, 0, 1))
                if data.shape[0] == 3:
                    # channel swap
                    # XXX see issue #59
                    data = data[[2, 1, 0], ...]
            elif mean.ndim == 2:
                # Add a channels axis
                data = data[np.newaxis, :, :]

            blob = caffe_pb2.BlobProto()
            blob.num = 1
            blob.channels, blob.height, blob.width = data.shape
            blob.data.extend(data.astype(float).flat)

            with open(mean_file, 'wb') as outfile:
                outfile.write(blob.SerializeToString())
        elif mean_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = PIL.Image.fromarray(mean)
            image.save(mean_file)
        else:
            logger.warning('Unrecognized file extension for mean file: "%s"' % mean_file)
            continue

        logger.info('Mean saved at "%s"' % mean_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create-Db tool - DIGITS')

    # Positional arguments

    parser.add_argument('input_file',
                        help='An input file of labeled images')
    parser.add_argument('output_dir',
                        help='Path to the output database')
    parser.add_argument('width',
                        type=int,
                        help='width of resized images'
                        )
    parser.add_argument('height',
                        type=int,
                        help='height of resized images'
                        )

    # Optional arguments

    parser.add_argument('-c', '--channels',
                        type=int,
                        default=3,
                        help='channels of resized images (1 for grayscale, 3 for color [default])'
                        )
    parser.add_argument('-r', '--resize_mode',
                        help='resize mode for images (must be "crop", "squash" [default], "fill" or "half_crop")'
                        )
    parser.add_argument('-m', '--mean_file', action='append',
                        help="location to output the image mean (doesn't save mean if not specified)")
    parser.add_argument('-f', '--image_folder',
                        help='folder containing the images (if the paths in input_file are not absolute)')
    parser.add_argument('-s', '--shuffle',
                        action='store_true',
                        help='Shuffle images before saving'
                        )
    parser.add_argument('-e', '--encoding',
                        help='Image encoding format (jpg/png)'
                        )
    parser.add_argument('-C', '--compression',
                        help='Database compression format (gzip)'
                        )
    parser.add_argument('-b', '--backend',
                        default='lmdb',
                        help='The database backend - lmdb[default], hdf5 or tfrecords')
    parser.add_argument('--lmdb_map_size',
                        type=int,
                        help='The initial map size for LMDB (in MB)')
    parser.add_argument('--hdf5_dset_limit',
                        type=int,
                        default=2**31,
                        help='The size limit for HDF5 datasets')
    parser.add_argument('--delete_files',
                        action='store_true',
                        help='Specifies whether to keep files after creation of dataset')

    args = vars(parser.parse_args())

    if args['lmdb_map_size']:
        # convert from MB to B
        args['lmdb_map_size'] <<= 20

    try:
        if args['backend'] == 'tfrecords':
            create_tfrecords_db(args['input_file'], args['output_dir'],
                  args['width'], args['height'], args['channels'],
                  args['backend'],
                  resize_mode=args['resize_mode'],
                  image_folder=args['image_folder'],
                  shuffle=args['shuffle'],
                  mean_files=args['mean_file'],
                  encoding=args['encoding'],
                  compression=args['compression'],
                  delete_files=args['delete_files'])
            exit(0)

        create_db(args['input_file'], args['output_dir'],
                  args['width'], args['height'], args['channels'],
                  args['backend'],
                  resize_mode=args['resize_mode'],
                  image_folder=args['image_folder'],
                  shuffle=args['shuffle'],
                  mean_files=args['mean_file'],
                  encoding=args['encoding'],
                  compression=args['compression'],
                  lmdb_map_size=args['lmdb_map_size'],
                  hdf5_dset_limit=args['hdf5_dset_limit'],
                  delete_files=args['delete_files']
                  )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
