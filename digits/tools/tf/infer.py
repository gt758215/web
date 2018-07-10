import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.platform import gfile
from tensorflow.python.lib.io import file_io
import model_config
import data_config
import batch_allreduce
import logging
import numpy as np
import time
import json
import os

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
flags.DEFINE_string('data_dir', None,'data_dir')
flags.DEFINE_string('filename', None,'filename')
flags.DEFINE_string('network', None,'network')
flags.DEFINE_string('networkDirectory', None,'networkDirectory')
flags.DEFINE_integer('batch_size', 1, 'batch size per compute device')
flags.DEFINE_string('labels_list', None,'labels_list')
flags.DEFINE_string('device', 'gpu', 'device [gpu | cpu]')
flags.DEFINE_string('train_dir', None, 'train_dir')
flags.DEFINE_enum('data_format', 'NCHW', ('NHWC', 'NCHW'),
                  'Data layout to use: NHWC (TF native) or NCHW (cuDNN '
                  'native, requires GPU).')


def savable_variables():
    """Return the set of variables used for saving/loading the model."""
    params = []
    for v in tf.global_variables():
      split_name = v.name.split('/')
      if split_name[0] == 'tower_0' or not v.name.startswith('tower_'):
        params.append(v)
    return params

class CheckpointNotFoundException(Exception):
  pass

def create_config_proto():
  """Returns session config proto.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  return config

def load_checkpoint(saver, sess, ckpt_dir):
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if ckpt and ckpt.model_checkpoint_path:
    if os.path.isabs(ckpt.model_checkpoint_path):
      # Restores from checkpoint with absolute path.
      model_checkpoint_path = ckpt.model_checkpoint_path
    else:
      # Restores from checkpoint with relative path.
      model_checkpoint_path = os.path.join(ckpt_dir, ckpt.model_checkpoint_path)
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/imagenet_train/model.ckpt-0,
    # extract global_step from it.
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    if not global_step.isdigit():
      global_step = 0
    else:
      global_step = int(global_step)
    saver.restore(sess, model_checkpoint_path)
    print('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)
    return global_step
  else:
    raise CheckpointNotFoundException('No checkpoint file found.')


class BenchmarkCNN(object):
  def __init__(self):
    self.data_dir = FLAGS.data_dir
    self.filename = FLAGS.filename
    self.network = FLAGS.network
    self.networkDirectory = FLAGS.networkDirectory
    self.batch_size = FLAGS.batch_size
    self.labels_list = FLAGS.labels_list
    self.labels = [line for line in open(self.labels_list) if line.strip()]
    self.num_classes = len(self.labels)
    print("data_dir: %s, filename: %s" % (self.data_dir, self.filename))
    self.dataset = data_config.Dataset(self.data_dir,
                                       FLAGS.labels_list,
                                       self.filename)
    print("dataset.file_pattern: %s" % self.dataset.file_pattern)
    self.model = model_config.get_model_config(
        FLAGS.networkDirectory,
        FLAGS.network)
    self.raw_devices = ['/%s:0' % (FLAGS.device)]
    self.data_type = tf.float32
    self.data_format = FLAGS.data_format
    self.train_dir = FLAGS.train_dir

  def run(self):
    with tf.Graph().as_default():
      return self._inference_cnn()

  def _inference_cnn(self):
    fetches = self._build_graph()
    saver = tf.train.Saver(savable_variables(), save_relative_paths=True)
    local_var_init_op_group = self.get_init_op_group()
    self._inference_once(saver, local_var_init_op_group, fetches)

  def _inference_once(self, saver, local_var_init_op_group, fetches):
    with tf.Session(
        config=create_config_proto()) as sess:
      try:
        global_step = load_checkpoint(saver, sess, self.train_dir)
      except CheckpointNotFoundException:
        print('Checkpoint not found in %s' % self.train_dir)
        return
      sess.run(local_var_init_op_group)
      rtop_1, rtop_5, rlogits, rprediction, rlabels, softmax = sess.run([fetches['top_1_op'], 
                                          fetches['top_5_op'], 
                                          fetches['logits'], 
                                          fetches['prediction'], 
                                          fetches['labels'],
                                          fetches['softmax']])
      logging.info('Predictions for image ' + str(1) + ': ' + json.dumps(softmax.tolist()))
      logging.info('Predictions for top_1: ' + str(rtop_1))
      logging.info('Predictions for top_5: ' + str(rtop_5))
      logging.info('Predictions for logits: ' + json.dumps(rlogits.tolist()))
      logging.info('Predictions for rprediction: ' + json.dumps(rprediction.tolist()))
      logging.info('Predictions for rlabels: ' + json.dumps(rlabels.tolist()))

  def _build_graph(self):
    tf.set_random_seed(1234)
    np.random.seed(4321)
    # get global step
    with tf.device('/cpu:0'):
      global_step = tf.train.get_or_create_global_step()
    # get datainputs and build model
    dataloader = data_config.DataLoader(
      self.model.get_image_size(), self.model.get_image_size(),
      self.batch_size, 1,
      '/cpu:0',
      self.raw_devices, tf.float32,
      'validation', self.dataset)
    update_ops = None
    current_scope = 'tower_0'
    with tf.variable_scope(current_scope), tf.name_scope(current_scope):
      results = self._add_forward_pass_and_gradients(
        phase_train=False, device_num=0, dataloader=dataloader, batch_size=self.batch_size)
    #fetches = tf.nn.softmax(results['logits'])
    return results

  def _add_forward_pass_and_gradients(self,
                                      phase_train,
                                      device_num,
                                      dataloader,
                                      batch_size):
    """Add ops for forward-pass and gradient computations."""
    nclass = self.dataset.num_classes
    image_size = self.model.get_image_size()
    # build network per tower
    with tf.device(self.raw_devices[device_num]):
      images, labels = dataloader.get_images_and_labels(device_num, self.data_type)
      images = tf.reshape(
          images,
          shape=[
              batch_size, image_size, image_size,
              3
          ])
      logits, aux_logits = self.model.build_network(
          images, phase_train, nclass, 3, self.data_type,
          self.data_format)
      results = {}  # The return value
      top_1_op = tf.reduce_sum(
          tf.cast(tf.nn.in_top_k(logits, labels, 1), self.data_type))
      top_5_op = tf.reduce_sum(
          tf.cast(tf.nn.in_top_k(logits, labels, 5), self.data_type))
      results['top_1_op'] = top_1_op
      results['top_5_op'] = top_5_op
      results['logits'] = logits
      results['prediction'] = tf.argmax(logits, 1)
      results['labels'] = labels
      results['softmax'] = tf.nn.softmax(logits)
      return results

  def get_init_op_group(self):
    local_var_init_op = tf.local_variables_initializer()
    table_init_ops = tf.tables_initializer()
    variable_mgr_init_ops = [local_var_init_op]
    if table_init_ops:
      variable_mgr_init_ops.extend([table_init_ops])
    with tf.control_dependencies([local_var_init_op]):
      variable_mgr_init_ops.extend(self.get_post_init_ops())
    local_var_init_op_group = tf.group(*variable_mgr_init_ops)
    return local_var_init_op_group

  def get_post_init_ops(self):
    # Copy initialized values for variables on GPU 0 to other GPUs.
    global_vars = tf.global_variables()
    var_by_name = dict([(v.name, v) for v in global_vars])
    post_init_ops = []
    for v in global_vars:
      split_name = v.name.split('/')
      # TODO(b/62630508): use more specific prefix than v or v0.
      if split_name[0] == 'tower_0' or not v.name.startswith('tower'):
        continue
      split_name[0] = 'tower_0'
      copy_from = var_by_name['/'.join(split_name)]
      post_init_ops.append(v.assign(copy_from.read_value()))
    #post_init_ops += self._warmup_ops
    return post_init_ops

def tensorflow_version_tuple():
  v = tf.__version__
  major, minor, patch = v.split('.')
  return (int(major), int(minor), patch)
 
def main(_):
  bench = BenchmarkCNN()
  tfversion = tensorflow_version_tuple()
  logging.info('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))
  bench.run()

if __name__ == '__main__':
  tf.app.run()
