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
flags.DEFINE_string('result_dir', None,'result_dir')
flags.DEFINE_integer('batch_size', 1, 'batch size per compute device')
flags.DEFINE_string('labels_list', None,'labels_list')
flags.DEFINE_string('device', 'gpu', 'device [gpu | cpu]')
flags.DEFINE_string('train_dir', None, 'train_dir')
flags.DEFINE_bool('gen_metrics', False, 'generate confusion matrix and related info')
flags.DEFINE_enum('data_format', 'NCHW', ('NHWC', 'NCHW'),
                  'Data layout to use: NHWC (TF native) or NCHW (cuDNN '
                  'native, requires GPU).')


######### data cleaning #########
import sklearn as sk

from sklearn.metrics import confusion_matrix

class image_prediction:
  def __init__(self, r_id, top_1, top_1_ids, top_5, top_5_ids, logits, pred, y_batch):
    self.r_id = r_id
    self.top_1 = top_1
    self.top_1_ids = top_1_ids
    self.top_5 = top_5
    self.top_5_ids = top_5_ids
    self.logits = logits
    self.pred = pred
    self.y_batch = y_batch

  @property
  def gen_json_data(self):
    data = {'id': self.r_id,
            'top_1': self.top_1,
            'top_1_ids': self.top_1_ids,
            'top_5': self.top_5,
            'top_5_ids': self.top_5_ids,
            'logits': self.logits,
            'prediction': self.pred,
            'y_batch_label': self.y_batch
            }
    return data

class image_prediction_dict:
  def __init__(self):
    self.img_pred_dict = {}
    self.id_list = []
    self.pred_list = []
    self.y_batch_list = []

  def convert_to_list(self):
    for key, value in self.img_pred_dict.iteritems():
      self.id_list.append(key)
      self.pred_list.append(value.pred)
      self.y_batch_list.append(value.y_batch)

  def gen_json_data(self):
    data = {'id_list': self.id_list,
            'prediction_list': self.pred_list,
            'y_batch_list': self.y_batch_list
           }
    return data

class confusion_matrix:
  def __init__(self, img_pred_dict, pred_list, y_batch_list, label_list):
    if len(pred_list) != len(y_batch_list):
      raise Exception('Prediction length is different from Label list!')
    self.img_pred_dict = img_pred_dict
    self.matrix_size = len(label_list)
    # this matrix are 3 dimensions(y_batch, pred, id_lists)
    self.matrix = [[[] for x in range(self.matrix_size)] for y in range(self.matrix_size)]
    self.precision = sk.metrics.precision_score(y_batch_list, pred_list, labels=label_list, average='weighted')
    self.recall = sk.metrics.recall_score(y_batch_list, pred_list, labels=label_list, average='weighted')
    self.f1_score = sk.metrics.f1_score(y_batch_list, pred_list, labels=label_list, average='weighted')
    self.confusion_matrix = sk.metrics.confusion_matrix(y_batch_list, pred_list, labels=label_list)

  def calculate_ids_in_confusion_matrix(self):
    for key, value in self.img_pred_dict.iteritems():
	  # dimension => [y_batch, pred, id_lists]
	  self.matrix[value.y_batch][value.pred].append(value.r_id)

  def gen_json_data(self):
    data = {'precision': str(self.precision.tolist()),
            'recall': str(self.recall),
            'f1_score': str(self.f1_score),
            'confusion_matrix': self.confusion_matrix.tolist(),
            'confusion_matrix with ids': self.matrix
           }
    return data

  def dump_data_to_file(self, path):
    with open(os.path.join(path, 'confusion_matrix.json'), 'w') as outfile:
      json.dump(self.gen_json_data(), outfile, sort_keys=True, indent=4)

#################################


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
    self.result_dir = FLAGS.result_dir
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
    self.gen_metrics = FLAGS.gen_metrics

    #data cleaning
    if self.gen_metrics is True:
      self.image_prediction_dict = image_prediction_dict()


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

      #data cleaning
      if self.gen_metrics is True:
        for i in range(self.batch_size):
          # prepare individul image prediction
          tmp_img_pred = image_prediction(i, #id
          rtop_1[0][i], #top_1
          rtop_1[1][i], #top_1_ids
          rtop_5[0][i], #top_5
          rtop_5[1][i], #top_5_ids
          rlogits[i], #logits
          rprediction[i], #prediction
          rlabels[i]) #y_batch
          # add image prediction to dict for generating total prediction and y_batch list
          self.image_prediction_dict.img_pred_dict[i] = tmp_img_pred

        self.image_prediction_dict.convert_to_list()
        logging.info("id_list:" + str(self.image_prediction_dict.id_list))
        logging.info("pred_list:" + str(self.image_prediction_dict.pred_list))
        logging.info("y_batch_list:" + str(self.image_prediction_dict.y_batch_list))

        #confusion_matrix
        self.confusion_matrix = confusion_matrix(self.image_prediction_dict.img_pred_dict,
                                                 self.image_prediction_dict.pred_list, 
                                                 self.image_prediction_dict.y_batch_list,
                                                 range(len(self.labels)))
        self.confusion_matrix.calculate_ids_in_confusion_matrix()

        logging.info('Predictions for top_1: ' + str(rtop_1))
        logging.info('Predictions for top_5: ' + str(rtop_5))
        logging.info('Predictions for logits: ' + json.dumps(rlogits.tolist()))
        logging.info('Predictions for rprediction: ' + json.dumps(rprediction.tolist()))
        logging.info('Predictions for rlabels: ' + json.dumps(rlabels.tolist()))

        #logging.info('matrix_size:' + str(self.confusion_matrix.matrix_size))
        #logging.info('precision:' + str(self.confusion_matrix.precision))
        #logging.info('recall:' + str(self.confusion_matrix.recall))
        #logging.info('f1_score:' + str(self.confusion_matrix.f1_score))
        #logging.info('confusion_matrix:' + str(self.confusion_matrix.confusion_matrix))
        #logging.info('confusion_matrix with ids:' + str(self.confusion_matrix.matrix))
        logging.info('json dump to confusion_matrix:' + json.dumps(self.confusion_matrix.gen_json_data()))
        self.confusion_matrix.dump_data_to_file(self.result_dir)

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
      top_1_op = tf.nn.top_k(logits, 1)
      top_5_op = tf.nn.top_k(logits, 5)
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

