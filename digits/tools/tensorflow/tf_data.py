# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# This document should comply with PEP-8 Style Guide
# Linter: pylint

"""
Interface for data loading for Tensorflow.
Data loading is done through a data loading factory,that will setup
the correct functions for the respective backends.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import logging
import magic
import math
import numpy as np
import os
import tensorflow as tf

# Local imports
import caffe_tf_pb2
import utils as digits

import multiprocessing

# Constants
MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.4
MAX_ABSOLUTE_EXAMPLES_IN_QUEUE = 8192  # The queue size cannot exceed this number
#PREFETCH_BATCH = 32  # The queue size cannot exceed this number
NUM_THREADS_DATA_LOADER = 16
NUM_THREADS_DATA_READER = 16
NUM_THREADS_DATA_BATCHER = 8
LOG_MEAN_FILE = False  # Logs the mean file as loaded in TF to TB

# Supported extensions for Loaders
DB_EXTENSIONS = {
    'hdf5': ['.H5', '.HDF5'],
    'lmdb': ['.MDB', '.LMDB'],
    'tfrecords': ['.TFRECORDS'],
    'filelist': ['.TXT'],
    'file': ['.JPG', '.JPEG', '.PNG'],
    'gangrid': ['.GAN'],
}

LIST_DELIMITER = ' '  # For the FILELIST format

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def get_backend_of_source(db_path):
    """
    Takes a path as argument and infers the format of the data.
    If a directory is provided, it looks for the existance of an extension
    in the entire directory in an order of a priority of dbs (hdf5, lmdb, filelist, file)
    Args:
        db_path: path to a file or directory
    Returns:
        backend: the backend type
    """

    # If a directory is given, we include all its contents. Otherwise it's just the one file.
    if os.path.isdir(db_path):
        files_in_path = [fn for fn in os.listdir(db_path) if not fn.startswith('.')]
    else:
        files_in_path = [db_path]

    # Keep the below priority ordering
    for db_fmt in ['hdf5', 'lmdb', 'tfrecords', 'filelist', 'file', 'gangrid']:
        ext_list = DB_EXTENSIONS[db_fmt]
        for ext in ext_list:
            if any(ext in os.path.splitext(fn)[1].upper() for fn in files_in_path):
                return db_fmt

    logging.error("Cannot infer backend from db_path (%s)." % (db_path))
    exit(-1)


class MeanLoader(object):
    """
    Loads in a mean file for tensorflow. This is done through using a constant
    variable. It needs to be loaded first, after which the constant tf op
    can be retrieved through a function, and can be accounted for.

    """
    def __init__(self, mean_file_path, subtraction_type, bitdepth):
        self._mean_file_path = mean_file_path
        self._subtraction_type = subtraction_type
        self._bitdepth = bitdepth
        self.tf_mean_image = None
        self.load_mean()

    def load_mean(self):
        """
        The mean is loaded in the graph through a tf.constant for maximum efficiency. This is first
        done only once through a numpy array that defines the value of the constant.
        All pre-processing of the mean file is done before the definition of the tf.constant
        to make sure these operations are not repeated in the graph
        """

        file_extension = os.path.splitext(self._mean_file_path)[1].upper()

        if file_extension == '.BINARYPROTO':
            blob = caffe_tf_pb2.BlobProto()
            with open(self._mean_file_path, 'rb') as infile:
                blob.ParseFromString(infile.read())
            data = np.array(blob.data, dtype="float32").reshape(blob.channels, blob.height, blob.width)
            if blob.channels == 3:
                # converting from BGR to RGB
                data = data[[2, 1, 0], ...]  # channel swap
                # convert to (height, width, channels)
                data = data.transpose((1, 2, 0))
            elif blob.channels == 1:
                # convert to (height, width)
                data = data[0]
            else:
                logging.error('Unknown amount of channels (%d) in mean file (%s)' %
                              (blob.channels, self._mean_file_path))
                exit(-1)
        # elif file_extension in IMG_FILE_EXT:
        #     img = Image.open(self._mean_file_path)
        #     img.load()
        #     data = np.asarray(img, dtype="float32")
        else:
            logging.error('Failed loading mean file: Unsupported extension (%s)' % (file_extension))
            exit(-1)

        if (self._subtraction_type == 'image') or (self._subtraction_type == 'pixel'):
            if self._subtraction_type == 'pixel':
                data = data.mean(axis=(0, 1))
                data = np.reshape(data, (1, 1, -1))
            elif len(data.shape) != 3:
                # Explicitly add channel dim
                data = data[:, :, None]

            # return data in original pixel scale
            self.tf_mean_image = tf.constant(data, name='Const_Mean_Image')

        else:
            logging.error('Unsupported mean subtraction type (%s)' % (self._subtraction_type))
            exit(-1)

    def subtract_mean_op(self, tf_graph):
        """
        Places mean subtraction on top of the tensorflow graph supplied, returns the added op
        Args:
            tf_graph: the graph the subtraction of the mean should placed upon
        Returns:
            The graph with the mean subtraction placed on top of it
        """
        return (tf_graph - self.tf_mean_image)


class LoaderFactory(object):
    """
    A factory for data loading. It sets up a subclass with data loading
    done with the respective backend. Its output is a tensorflow queue op
    that is used to load in data, with optionally some minor postprocessing ops.
    """
    def __init__(self):
        self.croplen = None
        self.nclasses = None
        self.mean_loader = None
        self.backend = None
        self.db_path = None
        self.batch_x = None
        self.batch_y = None
        self.batch_k = None
        self.stage = None
        self._seed = None
        self.unencoded_data_format = 'hwc'
        self.unencoded_channel_scheme = 'rgb'
        self.summaries = None
        self.aug_dict = {}
        self.gpus = 0
        self.batch2gpu_cpu = min(NUM_THREADS_DATA_LOADER, multiprocessing.cpu_count())
        self.reader_cpu = min(NUM_THREADS_DATA_LOADER, multiprocessing.cpu_count())
        self.aug_cpu = min(NUM_THREADS_DATA_LOADER, multiprocessing.cpu_count()) if not self.is_inference else 1

        # @TODO(tzaman) rewrite this factory again
        pass

    @staticmethod
    def set_source(db_path, is_inference=False):
        """
        Returns the correct backend.
        """
        backend = get_backend_of_source(db_path)
        loader = None
        if backend == 'lmdb':
            loader = LmdbLoader()
        elif backend == 'hdf5':
            loader = Hdf5Loader()
        elif backend == 'file' or backend == 'filelist':
            loader = FileListLoader()
        elif backend == 'tfrecords':
            loader = TFRecordsLoader()
        elif backend == 'gangrid':
            loader = GanGridLoader()
        else:
            logging.error("Backend (%s) not implemented" % (backend))
            exit(-1)
        loader.backend = backend
        loader.db_path = db_path
        loader.is_inference = is_inference
        return loader

    def setup(self, labels_db_path, shuffle, bitdepth, batch_size, num_epochs=None, seed=None):
        with tf.device('/cpu:0'):
            self.labels_db_path = labels_db_path

            self.shuffle = shuffle
            self.bitdepth = bitdepth
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self._seed = seed

            if self.labels_db_path:
                self.labels_db = LoaderFactory.set_source(self.labels_db_path)
                self.labels_db.bitdepth = self.bitdepth
                self.labels_db.stage = self.stage
                self.labels_db.initialize()

            self.initialize()
            logging.info("Found %s images in db %s ", self.get_total(), self.db_path)

    def get_key_index(self, key):
        return self.keys.index(key)

    def set_augmentation(self, mean_loader, aug_dict={}):
        with tf.device('/cpu:0'):
            self.mean_loader = mean_loader
            self.aug_dict = aug_dict

    def get_shape(self):
        input_shape = [self.height, self.width, self.channels]
        # update input_shape if crop length specified
        # this is necessary as the input_shape is provided
        # below to the user-defined function that defines the network
        if self.croplen > 0:
            input_shape[0] = self.croplen
            input_shape[1] = self.croplen
        return input_shape

    def get_total(self):
        return self.total

    def reshape_decode(self, data, shape):
        if self.float_data:  # @TODO(tzaman): this is LMDB specific - Make generic!
            data = tf.reshape(data, shape)
            data = digits.chw_to_hwc(data)
        else:
            # Decode image of any time option might come: https://github.com/tensorflow/tensorflow/issues/4009
            # Distinguish between mime types
            if self.data_encoded:
                if self.data_mime == 'image/png':
                    data = tf.image.decode_png(data, dtype=self.image_dtype, name='image_decoder')
                elif self.data_mime == 'image/jpeg':
                    data = tf.image.decode_jpeg(data, name='image_decoder')
                else:
                    logging.error('Unsupported mime type (%s); cannot be decoded' % (self.data_mime))
                    exit(-1)
            else:
                if self.backend == 'lmdb':
                    data = tf.decode_raw(data, self.image_dtype, name='raw_decoder')

                # if data is in CHW, set the shape and convert to HWC
                if self.unencoded_data_format == 'chw':
                    data = tf.reshape(data, [shape[0], shape[1], shape[2]])
                    data = digits.chw_to_hwc(data)
                else:  # 'hwc'
                    data = tf.reshape(data, shape)

                if (self.channels == 3) and self.unencoded_channel_scheme == 'bgr':
                    data = digits.bgr_to_rgb(data)

            # Convert to float
            data = tf.to_float(data)
            # data = tf.image.convert_image_dtype(data, tf.float32) # normalize to [0:1) range
        return data

    def load_to_gpu(self, k, d, l):
            d_g = []
            l_g = []
            dev = range(self.gpus)

            for i in dev:
                with tf.device("/gpu:%d" % i):
                    d_g.append(tf.identity(d[i]))
                    l_g.append(tf.identity(l[i]))
            d_2 = tf.stack(d_g, 0)
            l_2 = tf.stack(l_g, 0)
            return k, d_2, l_2

    def create_input_pipeline(self):
        """
        This function returns part of the graph that does data loading, and
        includes a queueing, optional data decoding and optional post-processing
        like data augmentation or mean subtraction.

        Args:
            None.
        Produces:
            batch_x: Input data batch
            batch_y: Label data batch
            batch_k: A list of keys (strings) from which the batch originated
        Returns:
            None.
        """

        def reshape_and_aug(k, d, ds, l, ls):
            ds = tf.reshape(ds, [3])
            d = self.reshape_decode(d, ds)

            if self.labels_db_path:
                ls = tf.reshape(ls, [3])
                l = self.labels_db.reshape_decode(l, ls)
            else:
                if self.stage != digits.STAGE_INF:
                    l = tf.reshape(l, [])

            # Mean Subtraction
            if self.mean_loader:
                with tf.name_scope('mean_subtraction'):
                    d = self.mean_loader.subtract_mean_op(d)
                    if LOG_MEAN_FILE:
                        expanded_data = tf.expand_dims(self.mean_loader.tf_mean_image, 0)
                        self.summaries.append(tf.summary.image('mean_image', expanded_data, max_outputs=1))

            # (Random) Cropping
            if self.croplen:
                with tf.name_scope('cropping'):
                    if self.stage == digits.STAGE_TRAIN:
                        d = tf.random_crop(
                            d,
                            [self.croplen, self.croplen, self.channels],
                            seed=self._seed
                        )
                    else:  # Validation or Inference
                        d = tf.image.resize_image_with_crop_or_pad(d, self.croplen, self.croplen)

            # Data Augmentation
            if self.aug_dict:
                with tf.name_scope('augmentation'):
                    flipflag = self.aug_dict['aug_flip']
                    if flipflag == 'fliplr' or flipflag == 'fliplrud':
                        d = tf.image.random_flip_left_right(d, seed=self._seed)
                    if flipflag == 'flipud' or flipflag == 'fliplrud':
                        d = tf.image.random_flip_up_down(d, seed=self._seed)

                    noise_std = self.aug_dict['aug_noise']
                    if noise_std > 0.:
                        # Note the tf.random_normal requires a static shape
                        d = tf.add(
                                d,
                                tf.random_normal(
                                    self.get_shape(),
                                    mean=0.0,
                                    stddev=noise_std,
                                    dtype=tf.float32,
                                    seed=self._seed,
                                    name='AWGN'
                                )
                            )

                    contrast_fact = self.aug_dict['aug_contrast']
                    if contrast_fact > 0:
                        d = tf.image.random_contrast(
                                d,
                                lower=1.-contrast_fact,
                                upper=1.+contrast_fact,
                                seed=self._seed
                            )

                    # @TODO(tzaman): rewrite the below HSV stuff entirely in a TF PR to be done in one single operation
                    aug_hsv = self.aug_dict['aug_HSV']
                    if aug_hsv['h'] > 0.:
                        d = tf.image.random_hue(d, aug_hsv['h'], seed=self._seed)
                    if aug_hsv['s'] > 0.:
                        d = tf.image.random_saturation(
                                d,
                                1 - aug_hsv['s'],
                                1 + aug_hsv['s'],
                                seed=self._seed
                            )
                    if aug_hsv['v'] > 0.:
                        # closely resembles V - temporary until rewritten
                        d = tf.image.random_brightness(d, aug_hsv['v'], seed=self._seed)

                    # @TODO(tzaman) whitening is so invasive that we need a way to add it to the val/inf too in a
                    # portable manner, like the mean file : how? If we don't find a way, don't use whitening.
                    aug_whitening = self.aug_dict['aug_whitening']
                    if aug_whitening:
                        # Subtract off its own mean and divide by the standard deviation of its own the pixels.
                        with tf.name_scope('whitening'):
                            d = tf.image.per_image_standardization(d)  # N.B. also converts to float

            if self.stage != digits.STAGE_INF:
                return k, d, l
            else:
                return k, d

        max_queue_capacity = min(math.ceil(self.total * MIN_FRACTION_OF_EXAMPLES_IN_QUEUE),
                                 MAX_ABSOLUTE_EXAMPLES_IN_QUEUE)
        prefetch_batches = max_queue_capacity//self.batch_size
        key_queue = self.get_queue()
        dataset = self.get_single_data(key_queue)
        dataset = dataset.map(reshape_and_aug, num_parallel_calls=self.aug_cpu)
        if self.shuffle:
            # shuffled data range = 5 * batch_size
            dataset = dataset.shuffle(buffer_size=5*self.batch_size, seed=self._seed)

        dataset = dataset.repeat(self.num_epochs)
        # batch of one gpu
        dataset = dataset.batch(self.batch_size//(self.gpus if self.gpus else 1))
        # prefetch batches of all gpus
        dataset = dataset.prefetch(prefetch_batches*self.gpus)

        if(self.gpus > 0):
            # batch all gpus to one
            dataset = dataset.batch(self.gpus)
            # unpack batches of all gpus, and direct each batch tensor on it's gpu
            dataset = dataset.map(self.load_to_gpu, num_parallel_calls=self.batch2gpu_cpu)

        # now prefetch store batches on different gpus
        dataset = dataset.prefetch(prefetch_batches)

        #iterator = dataset.make_one_shot_iterator()
        iterator = dataset.make_initializable_iterator()
        self.iterator = iterator
        self.init_op = iterator.initializer
        next_batch = iterator.get_next()

        self.batch_k = next_batch[0]  # Key
        self.batch_x = next_batch[1]  # Input
        if len(next_batch) == 3:
            # There's a label (unlike during inferencing)
            self.batch_y = next_batch[2]  # Output (label)


class LmdbLoader(LoaderFactory):
    """ Loads files from lmbd files as used in Caffe
    """
    def __init__(self):
        pass

    def initialize(self):
        try:
            import lmdb
        except ImportError:
            logging.error("Attempt to create LMDB Loader but lmdb is not installed.")
            exit(-1)

        self.unencoded_data_format = 'chw'
        self.unencoded_channel_scheme = 'bgr'

        # Set up the data loader
        self.lmdb_env = lmdb.open(self.db_path, readonly=True, lock=False)
        self.lmdb_txn = self.lmdb_env.begin(buffers=False)
        self.total = self.lmdb_txn.stat()['entries']

        # Keys Saver
        import cPickle as pickle

        key_path = self.db_path + '/keys.mdb'
        if os.path.isfile(key_path):
            self.keys = pickle.load(open(key_path, "rb"))
        else:
            self.keys = [key for key, _ in self.lmdb_txn.cursor()]
            pickle.dump(self.keys, open(key_path, "wb"), protocol=True)

        # Read the first entry to get some info
        lmdb_val = self.lmdb_txn.get(self.keys[0])
        datum = caffe_tf_pb2.Datum()
        datum.ParseFromString(lmdb_val)

        self.channels = datum.channels
        self.width = datum.width
        self.height = datum.height
        self.data_encoded = datum.encoded
        self.float_data = datum.float_data

        if self.data_encoded:
            # Obtain mime-type
            self.data_mime = magic.from_buffer(datum.data, mime=True)

        if not self.float_data:
            if self.bitdepth == 8:
                self.image_dtype = tf.uint8
            else:
                if self.data_mime == 'image/jpeg':
                    logging.error("Tensorflow does not support 16 bit jpeg decoding.")
                    exit(-1)
                self.image_dtype = tf.uint16

    def get_queue(self):
        return tf.data.Dataset.from_tensor_slices(self.keys)

    def get_tf_data_type(self):
        """Returns the type of the data, in tf format.
            It takes in account byte-data or floating point data.
            It also takes in account the possible seperate lmdb label db.
        Returns:
            The tensorflow-datatype of the data
        """
        return tf.float32 if self.float_data else tf.string

    def get_tf_label_type(self):
        """Returns the type of the label, in tf format.
            It takes in account byte-data or floating point data.
            It also takes in account the possible seperate lmdb label db.
        Returns:
            The tensorflow-datatype of the label
        """
        if self.labels_db_path:
            return self.labels_db.get_tf_data_type()
        else:
            # No seperate db, return scalar label
            return tf.int64

    def generate_data_op(self):
        """Generates and returns an op that fetches a single sample of data.

        Args:
            self:

        Returns:
            A python function that is inserted as an op
        """
        def get_data_and_shape(lmdb_txn, key):
            val = lmdb_txn.get(key)
            datum = caffe_tf_pb2.Datum()
            datum.ParseFromString(val)
            shape = np.array([datum.channels, datum.height, datum.width], dtype=np.int32)
            if datum.float_data:
                data = np.asarray(datum.float_data, dtype='float32')
            else:
                data = datum.data
            label = np.asarray([datum.label], dtype=np.int64)  # scalar label
            return data, shape, label

        def get_data_op(key):
            """Fetches a sample of data and its label from lmdb. If a seperate label database
               exists, it will also load it from the seperate db inside this function. This is
               done the data and its label are loaded at the same time, avoiding multiple queues
               and race conditions.

            Args:
                self: the current lmdb instance

            Returns:
                single_data: One sample of training data
                single_data_shape: The shape of the preceeding training data
                single_label: The label that is the reference value describing the data
                single_label_shape: The shape of the preceeding label data
            """
            single_data, single_data_shape, single_label = get_data_and_shape(self.lmdb_txn, key)
            single_label_shape = np.array([], dtype=np.int32)
            if self.labels_db_path:
                single_label, single_label_shape, _ = get_data_and_shape(self.labels_db.lmdb_txn, key)
            return key, single_data, [single_data_shape], single_label, [single_label_shape]
        return get_data_op

    def get_single_data(self, key_queue):
        """
        Returns:
            key, single_data, single_data_shape, single_label, single_label_shape
        """
        py_func_return_type = [tf.string, self.get_tf_data_type(), tf.int32, self.get_tf_label_type(), tf.int32]
        return key_queue.map(
            lambda key: tf.py_func(self.generate_data_op(), [key], py_func_return_type),
            num_parallel_calls=self.reader_cpu
        )

    def __del__(self):
        self.lmdb_env.close()


class FileListLoader(LoaderFactory):
    """ The FileListLoader loads files from a list of string(s) pointing to (a) file(s).
    These files are then retrieved by their string and loaded according to their extension.
    """
    def __init__(self):
        pass

    def initialize(self):
        self.float_data = False
        self.data_encoded = True

        if self.backend == 'file':
            # Single file
            self.total = 1
            self.keys = [self.db_path]
            first_file_path = self.db_path
        elif self.backend == 'filelist':
            # Single file with a list of files
            with open(self.db_path) as f:
                self.keys = f.readlines()

            # Retain only the images in the list
            self.keys = [key.split(LIST_DELIMITER)[0].rstrip() for key in self.keys]

            if len(self.keys) > 0:
                # Assume the first entry in the line is a pointer to the file path
                first_file_path = self.keys[0]
            else:
                logging.error('Filelist (%s) contains no lines.' % (self.db_path))
                exit(-1)
        else:
            logging.error('Unsupported backend in FileListLoader (%s)' % (self.backend))
            exit(-1)

        self.total = len(self.keys)

        # Check first file for statistics
        im = Image.open(first_file_path)
        self.width, self.height = im.size
        self.channels = 1 if im.mode == 'L' else 3  # @TODO(tzaman): allow more channels

        self.data_mime = magic.from_file(first_file_path, mime=True)

        if self.bitdepth == 8:
            self.image_dtype = tf.uint8
        else:
            if self.data_mime == 'image/jpeg':
                logging.error("Tensorflow does not support 16 bit jpeg decoding.")
                exit(-1)
            self.image_dtype = tf.uint16

        #self.reader = tf.WholeFileReader()
        self.reader = None

    def get_queue(self):
        return tf.data.Dataset.from_tensor_slices(self.keys)

    def get_single_data(self, key_queue):
        """
        Returns:
            key, single_data, single_data_shape, single_label, single_label_shape
        """
        def func(key):
            value = tf.read_file(key)
            shape = np.array([self.width, self.height, self.channels], dtype=np.int32)  # @TODO: this is not dynamic
            shape = tf.convert_to_tensor(shape, tf.int32)
            return key, value, shape  # @TODO(tzaman) - Note: will only work for inferencing stage!
        return key_queue.map(func, num_parallel_calls=self.reader_cpu)

class TFRecordsLoader(LoaderFactory):
    """ The TFRecordsLoader connects directly into the tensorflow graph.
    It uses TFRecords, the 'standard' tensorflow data format.
    """
    def __init__(self):
        pass

    def initialize(self):
        self.float_data = False  # For now only strings
        self.unencoded_data_format = 'hwc'
        self.unencoded_channel_scheme = 'rgb'
        self.reader = None
        if self.bitdepth == 8:
            self.image_dtype = tf.uint8
        else:
            self.image_dtype = tf.uint16

        # Count all the records @TODO(tzaman): account for shards!
        # Loop the records in path @TODO(tzaman) get this from a txt?
        # self.db_path += '/test.tfrecords' # @TODO(tzaman) this is a hack

        self.shard_paths = []
        self.keys = []
        list_db_files = os.path.join(self.db_path, 'list.txt')
        self.total = 0
        if os.path.exists(list_db_files):
            files = [os.path.join(self.db_path, f) for f in open(list_db_files, 'r').read().splitlines()]
        else:
            files = [self.db_path]
        for shard_path in files:
            # Account for the relative path format in list.txt
            record_iter = tf.python_io.tf_record_iterator(shard_path)
            pos = 0
            for r in record_iter:
                self.total += 1
                pos = pos + 16 + len(r)
                self.keys.append(shard_path + ':' + pos)

            # no record if pos==0
            if pos != 0:
                self.shard_paths.append(shard_path)
        #self.keys = ['%s:0' % p for p in self.shard_paths]
        if not self.total:
            raise ValueError('Database or shard contains no records (%s)' % (self.db_path))
        # Use last record read to extract some preliminary data that is sometimes needed or useful
        example_proto = tf.train.Example()
        example_proto.ParseFromString(r)

        # @TODO(tzaman) - bitdepth flag?
        self.channels = example_proto.features.feature['depth'].int64_list.value[0]
        self.height = example_proto.features.feature['height'].int64_list.value[0]
        self.width = example_proto.features.feature['width'].int64_list.value[0]
        data_encoding_id = example_proto.features.feature['encoding'].int64_list.value[0]
        if data_encoding_id:
            self.data_encoded = True
            self.data_mime = 'image/png' if data_encoding_id == 1 else 'image/jpeg'
        else:
            self.data_encoded = False

    def get_queue(self):
        return tf.data.TFRecordDataset(self.shard_paths)

    def get_single_data(self, key_queue):
        """
        Returns:
            key, single_data, single_data_shape, single_label, single_label_shape
        """
        def parser(record):
            features = tf.parse_single_example(
                record,
                features={
                    'image_raw': tf.FixedLenFeature([self.height, self.width, self.channels], tf.float32),
                    'label': tf.FixedLenFeature([], tf.int64),
                    }
            )
            d = features['image_raw']
            ds = np.array([self.height, self.width, self.channels], dtype=np.int32)  # @TODO: this is not dynamic
            ds = tf.convert_to_tensor(ds, tf.int32)
            l = features['label']  # l = tf.cast(features['label'], tf.int32)
            ls = np.array([], dtype=np.int32)  # @TODO: this is not dynamic
            ls = tf.convert_to_tensor(ls, tf.int32)
            return d, ds, l, ls

        tf_db = key_queue.map(parser, num_parallel_calls=self.reader_cpu)
        key_db = tf.data.Dataset.from_tensor_slices(self.keys)
        return tf.data.Dataset.zip((key_db, tf_db))

class Hdf5Loader(LoaderFactory):

    def __init__(self):
        pass

    def initialize(self):
        try:
            import h5py
        except ImportError:
            logging.error("Attempt to create HDF5 Loader but h5py is not installed.")
            exit(-1)

        self.data_encoded = False
        self.float_data = True  # Always stored as float32
        self.keys = None  # Not using keys

        self.h5dbs = []
        self.h5dbs_endrange = []
        list_db_files = self.db_path + '/list.txt'
        self.total = 0
        with open(list_db_files) as f:
            for line in f:
                # Account for the relative path format in list.txt
                fn = self.db_path + '/' + os.path.basename(line.strip())
                db = h5py.File(fn)
                self.check_hdf5_db(db)
                self.total += len(db['data'])
                self.h5dbs_endrange.append(self.total)
                self.h5dbs.append(db)

        # Read the first file to get shape information
        self.channels, self.height, self.width = self.h5dbs[0]['data'][0].shape

    def check_hdf5_db(self, db):
        # Make sure we have data and labels in the db
        if "data" not in db or "label" not in db:
            logging.error("The HDF5 loader requires both a 'data' and 'label' group in the HDF5 root.")
            exit(-1)

        if len(db['data']) != len(db['label']):
            logging.error("HDF5 data and label amount mismatch (%d/%d)" % (len(db['data']), len(db['label'])))
            exit(-1)

        if len(db['data']) == 0:
            logging.error("HDF5 database contains no data.")
            exit(-1)

    def get_queue(self):
        return tf.data.Dataset.range(self.total)

    def get_tf_data_type(self):
        """Returns the type of the data, in tf format.
            It takes in account byte-data or floating point data.
            It also takes in account the possible seperate lmdb label db.
        Returns:
            The tensorflow-datatype of the data
        """
        return tf.float32 if self.float_data else tf.string

    def get_tf_label_type(self):
        """Returns the type of the label, in tf format.
            It takes in account byte-data or floating point data.
            It also takes in account the possible seperate lmdb label db.
        Returns:
            The tensorflow-datatype of the label
        """
        if self.labels_db_path:
            return self.labels_db.get_tf_data_type()
        else:
            # No seperate db, return scalar label
            return tf.int64

    def get_data_and_shape(self, sample_key):
        """ Gets a sample across multiple hdf5 databases
        """
        prev_end_range = 0
        for i, end_range in enumerate(self.h5dbs_endrange):
            if sample_key < end_range:
                key_within_db = sample_key-prev_end_range
                data = self.h5dbs[i]['data'][key_within_db]
                shape = np.asarray(data.shape, dtype=np.int32)
                label = self.h5dbs[i]['label'][key_within_db].astype(np.int64)
                return data, shape, label
            prev_end_range = end_range

        logging.error("Out of range")  # @TODO(tzaman) out of range error
        exit(-1)

    def generate_data_op(self):
        """Generates and returns an op that fetches a single sample of data.
        Returns:
            A python function that is inserted as an op
        """
        def get_data_op(key):
            """Fetches a sample of data and its label from db. If a seperate label database
               exists, it will also load it from the seperate db inside this function. This is
               done the data and its label are loaded at the same time, avoiding multiple queues
               and race conditions.
            Args:
                key: integer key id
            Returns:
                single_data: One sample of training data
                single_data_shape: The shape of the preceeding training data
                single_label: The label that is the reference value describing the data
                single_label_shape: The shape of the preceeding label data
            """
            single_data, single_data_shape, single_label = self.get_data_and_shape(key)
            single_label_shape = np.array([], dtype=np.int32)
            if self.labels_db_path:
                single_label, single_label_shape, _ = self.labels_db.get_data_and_shape(key)
            return key, single_data, [single_data_shape], single_label, [single_label_shape]
        return get_data_op

    def get_single_data(self, key_queue):
        """
        Returns:
            key, single_data, single_data_shape, single_label, single_label_shape
        """
        py_func_return_type = [tf.int64, self.get_tf_data_type(), tf.int32, self.get_tf_label_type(), tf.int32]
        return key_queue.map(
            lambda key: tf.py_func(self.generate_data_op(), [key], py_func_return_type),
            num_parallel_calls=self.reader_cpu
        )


    def __del__(self):
        for db in self.h5dbs:
            db.close()


class GanGridLoader(LoaderFactory):
    """
    The GanGridLoader generates data for a GAN.
    """
    def __init__(self):
        pass

    def initialize(self):
        self.float_data = False  # For now only strings
        self.keys = None  # Not using keys
        self.unencoded_data_format = 'hwc'
        self.unencoded_channel_scheme = 'rgb'
        self.reader = None
        self.image_dtype = tf.float32

        self.channels = 1
        self.height = 1
        self.width = 100
        self.data_encoded = False

        self.total = 100000

    def get_queue(self):
        return tf.data.Dataset.range(self.total)


    def get_single_data(self, key_queue):
        """
        Returns:
            key, single_data, single_data_shape, single_label, single_label_shape
        """
        def func(key):
            key = tf.to_int32(key)
            d = keys
            ds = np.array([1, 1, 1], dtype=np.int32)
            ds = tf.convert_to_tensor(ds, dtype=tf.int32)
            return key, d, ds, None, None
        return key_queue.map(func, num_parallel_calls=self.reader_cpu)
