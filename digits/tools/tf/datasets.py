# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Benchmark dataset utilities.
"""

from abc import abstractmethod
import os

import numpy as np
from six.moves import cPickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
import preprocessing

IMAGENET_NUM_TRAIN_IMAGES = 1281167
IMAGENET_NUM_VAL_IMAGES = 50000


class Dataset(object):
  """Abstract class for cnn benchmarks dataset."""

  def __init__(self, name, height=None, width=None, depth=None, data_dir=None,
               queue_runner_required=False, num_classes=1001,
               total_train=0, total_val=0, train_dir=None, val_dir=None):
    self.name = name
    self.height = height
    self.width = width
    self.depth = depth or 3

    self.data_dir = data_dir
    self._queue_runner_required = queue_runner_required
    self._num_classes = num_classes
    self.total_train = total_train
    self.total_val = total_val
    self.train_dir = train_dir
    self.val_dir = val_dir

  def tf_record_pattern(self, subset):
    if subset == 'train' and self.train_dir:
      return os.path.join(self.train_dir, 'shard-*')
    if subset == 'validation' and self.val_dir:
      return os.path.join(self.val_dir, 'shard-*')
    return os.path.join(self.data_dir, '%s-*-of-*' % subset)

  def reader(self):
    return tf.TFRecordReader()

  @property
  def num_classes(self):
    return self._num_classes

  @num_classes.setter
  def num_classes(self, val):
    self._num_classes = val

  @abstractmethod
  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return self.total_train
    elif subset == 'validation':
      return self.total_val
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

  def __str__(self):
    return self.name

  def get_image_preprocessor(self, input_preprocessor='default'):
    if self.use_synthetic_gpu_images():
      return preprocessing.SyntheticImagePreprocessor
    return _SUPPORTED_INPUT_PREPROCESSORS[self.name][input_preprocessor]

  def queue_runner_required(self):
    return self._queue_runner_required

  def use_synthetic_gpu_images(self):
    return not (self.data_dir or self.train_dir)


class FlowersData(Dataset):
  """Configuration for Imagenet dataset."""

  def __init__(self, data_dir=None):
    super(FlowersData, self).__init__('flowers', 300, 300, data_dir=data_dir, num_classes=6)

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return 3170
    elif subset == 'validation':
      return 500
    else:
      raise ValueError('Invalid data subset "%s"' % subset)


class ImagenetData(Dataset):
  """Configuration for Imagenet dataset."""

  def __init__(self, data_dir=None):
    super(ImagenetData, self).__init__('imagenet', 300, 300, data_dir=data_dir)

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return IMAGENET_NUM_TRAIN_IMAGES
    elif subset == 'validation':
      return IMAGENET_NUM_VAL_IMAGES
    else:
      raise ValueError('Invalid data subset "%s"' % subset)


class Cifar10Data(Dataset):
  """Configuration for cifar 10 dataset.

  It will mount all the input images to memory.
  """

  def __init__(self, data_dir=None):
    super(Cifar10Data, self).__init__('cifar10', 32, 32, data_dir=data_dir,
                                      queue_runner_required=True,
                                      num_classes=11)

  def read_data_files(self, subset='train'):
    """Reads from data file and returns images and labels in a numpy array."""
    assert self.data_dir, ('Cannot call `read_data_files` when using synthetic '
                           'data')
    if subset == 'train':
      filenames = [os.path.join(self.data_dir, 'data_batch_%d' % i)
                   for i in xrange(1, 6)]
    elif subset == 'validation':
      filenames = [os.path.join(self.data_dir, 'test_batch')]
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

    inputs = []
    for filename in filenames:
      with gfile.Open(filename, 'r') as f:
        inputs.append(cPickle.load(f))
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    all_images = np.concatenate(
        [each_input['data'] for each_input in inputs]).astype(np.float32)
    all_labels = np.concatenate(
        [each_input['labels'] for each_input in inputs])
    return all_images, all_labels

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return 50000
    elif subset == 'validation':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)


_SUPPORTED_DATASETS = {
    'imagenet': ImagenetData,
    'cifar10': Cifar10Data,
}

_SUPPORTED_INPUT_PREPROCESSORS = {
    'imagenet': {
        'default': preprocessing.RecordInputImagePreprocessor,
        'official_models_imagenet': preprocessing.ImagenetPreprocessor,
    },
    'flowers': {
        'default': preprocessing.GeneralImagePreprocessor,
    },
    'general': {
        'default': preprocessing.GeneralImagePreprocessor,
    },
    'cifar10': {
        'default': preprocessing.Cifar10ImagePreprocessor
    }
}


def create_dataset(data_dir, data_name, train_db=None, validation_db=None,
                   labels_file=None):
  """Create a Dataset instance based on data_dir and data_name."""
  if not data_dir and not data_name:
    # When using synthetic data, use synthetic imagenet images by default.
    data_name = 'general'

  # Infere dataset name from data_dir if data_name is not provided.
  if data_name is None:
    for supported_name in _SUPPORTED_DATASETS:
      if supported_name in data_dir:
        data_name = supported_name
        break
    else:  # Failed to identify dataset name from data dir.
      raise ValueError('Could not identify name of dataset. '
                       'Please specify with --data_name option.')
  if data_name not in _SUPPORTED_DATASETS:
    if not labels_file:
      raise ValueError('Could not find labels.txt in %s' % data_dir)
    labels_list = [f for f in open(labels_file, 'r').read().splitlines()]

    train_file_pattern = os.path.join(train_db, 'shard-*')
    train_files = tf.gfile.Glob(train_file_pattern)
    total_train = 0
    for shard_file in train_files:
      record_iter = tf.python_io.tf_record_iterator(shard_file)
      for r in record_iter:
        total_train += 1
    val_file_pattern = os.path.join(validation_db, 'shard-*')
    val_files = tf.gfile.Glob(val_file_pattern)
    total_val = 0
    for shard_file in val_files:
      record_iter = tf.python_io.tf_record_iterator(shard_file)
      for r in record_iter:
        total_val += 1
    return Dataset(data_name,
                   data_dir=data_dir,
                   num_classes=len(labels_list)+1,
                   total_train=total_train,
                   total_val=total_val,
                   train_dir=train_db,
                   val_dir=validation_db)
    #raise ValueError('Unknown dataset. Must be one of %s', ', '.join(
    #    [key for key in sorted(_SUPPORTED_DATASETS.keys())]))

  return _SUPPORTED_DATASETS[data_name](data_dir)
