import os
import tensorflow as tf

class DataLoader(object):
  def __init__(
      self,
      height,
      width,
      batch_size,
      num_splits,
      cpu_device,
      gpu_devices,
      data_type,
      train_db,
      val_db,
      labels_file,
      verbosity):
    self.height = height
    self.width = width
    self.batch_size = batch_size
    self.num_splits = num_splits
    self.cpu_device = cpu_device
    self.gpu_devices = gpu_devices
    self.data_type = data_type
    self.train_db = train_db
    self.val_db = val_db
    self.verbosity = verbosity
    self.total_train = 0
    if os.path.isfile(os.path.join(self.train_db, "list.txt")):
      self.total_train = sum(1 for line in open(os.path.join(self.train_db, "list.txt")))
    else:
      file_pattern = os.path.join(self.train_db, 'train-*-of-*')
      files = tf.gfile.Glob(file_pattern)
      for shard_file in files:
        record_iter = tf.python_io.tf_record_iterator(shard_file)
        for r in record_iter:
          self.total_train += 1
    self.total_val = 0
    if os.path.isfile(os.path.join(self.val_db, "list.txt")):
      self.total_val = sum(1 for line in open(os.path.join(self.val_db, "list.txt")))
    else:
      file_pattern = os.path.join(self.val_db, 'validation-*-of-*')
      files = tf.gfile.Glob(file_pattern)
      for shard_file in files:
        record_iter = tf.python_io.tf_record_iterator(shard_file)
        for r in record_iter:
          self.total_val += 1
    self.labels_file = labels_file
    self.labels = [line for line in open(self.labels_file) if line.strip()]
    self.num_classes = len(self.labels)
    self.depth = 3

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return self.total_train
    elif subset == 'validation':
      return self.total_val
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

  def normalized_image(self, images):
    # Rescale from [0, 255] to [0, 2]
    images = tf.multiply(images, 1. / 127.5)
    # Rescale to [-1, 1]
    return tf.subtract(images, 1.0)

  def parse_fn(self, example):
    feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
    }
    features = tf.parse_single_example(example, feature_map)
    image = features['image/encoded']
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    image_decoded = tf.image.decode_jpeg(
      image, channels=3, dct_method='INTEGER_FAST')
    if self.verbosity >= 3:
      tf.summary.image(
          'original_image', tf.expand_dims(image_decoded, 0))
    # resize image
    image_resize_method = tf.image.ResizeMethod.BILINEAR
    resized_image = tf.image.resize_images(
        image_decoded, [self.height, self.width],
        image_resize_method,
        align_corners=False)
    normalized = self.normalized_image(resized_image)
    image = tf.cast(normalized, self.data_type)
    return image, label

  def get_inputs(self, subset):
    with tf.device(self.cpu_device):
      batch_size_per_split = self.batch_size // self.num_splits
      # start read files
      if subset == 'train':
        glob_pattern = os.path.join(self.train_db, "train-*-of-*")
      else:
        glob_pattern = os.path.join(self.val_db, "validation-*-of-*")
      files = tf.data.Dataset.list_files(glob_pattern)
      dataset = files.apply(tf.contrib.data.parallel_interleave(
          tf.data.TFRecordDataset, cycle_length=self.num_splits))
      dataset = dataset.repeat()
      dataset = dataset.apply(tf.contrib.data.map_and_batch(
          map_func=self.parse_fn,
          batch_size=batch_size_per_split,
          num_parallel_batches=self.num_splits))
      dataset = dataset.prefetch(buffer_size=10000)
      ds_iterator = dataset.make_one_shot_iterator()
    # build final results for split.
    images = [[] for _ in range(self.num_splits)]
    labels = [[] for _ in range(self.num_splits)]
    for d in range(len(self.gpu_devices)):
      with tf.device(self.gpu_devices[d]):
        images_d, labels_d = ds_iterator.get_next()
        images[d] = tf.reshape(
          images_d, shape=[batch_size_per_split, self.height, self.width, 3])
        labels[d] = tf.reshape(labels_d, [batch_size_per_split])
    return images, labels


