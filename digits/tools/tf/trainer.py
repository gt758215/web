from collections import namedtuple
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.platform import gfile
import model_config
import data_config
import batch_allreduce
import logging
import numpy as np
import time
import os

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

flags.DEFINE_integer('num_gpus', 1, 'the number of GPUs to run on')
flags.DEFINE_enum('device', 'gpu', ('cpu', 'gpu', 'CPU', 'GPU'),
                  'Device to use for computation: cpu or gpu')
flags.DEFINE_enum('data_format', 'NCHW', ('NHWC', 'NCHW'),
                  'Data layout to use: NHWC (TF native) or NCHW (cuDNN '
                  'native, requires GPU).')
flags.DEFINE_enum('variable_update', 'replicated',
                  ('parameter_server', 'replicated'),
                  'The method for managing variables: parameter_server, '
                  'replicated')
flags.DEFINE_integer('summary_verbosity', 3, 'Verbosity level for summary ops. '
                     'level 0: disable any summary.\n'
                     'level 1: small and fast ops, e.g.: learning_rate, '
                     'total_loss.\n'
                     'level 2: medium-cost ops, e.g. histogram of all '
                     'gradients.\n'
                     'level 3: expensive ops: images and histogram of each '
                     'gradient.\n')
flags.DEFINE_string('train_dir', None,
                    'Path to session checkpoints. Pass None to disable saving '
                    'checkpoint at the end.')
flags.DEFINE_string('networkDirectory', None,
                    'The directory where placed model file')
flags.DEFINE_string('network', 'network.py',
                    'The network file name')
flags.DEFINE_string('save', None,
                    'Path to session checkpoints. Pass None to disable saving '
                    'checkpoint at the end.')
flags.DEFINE_string('train_db', None,
                    'Path to train dataset in TFRecord format.')
flags.DEFINE_string('validation_db', None,
                    'Path to validation dataset in TFRecord format.')
flags.DEFINE_string('labels_list', None,
                    'list of labels file ')
flags.DEFINE_string('visualizeModelPath', None,
                    'Constructs the current model for visualization.')
flags.DEFINE_integer('save_summaries_steps', 100,
                     'How often to save summaries for trained models. Pass 0 '
                     'to disable summaries.')
flags.DEFINE_integer('display_every', 10,
                     'Number of local steps after which progress is printed '
                     'out')
# tf Option
flags.DEFINE_integer('num_intra_threads', 1,
                     'Number of threads to use for intra-op parallelism. If '
                     'set to 0, the system will pick an appropriate number.')
flags.DEFINE_integer('num_inter_threads', 0,
                     'Number of threads to use for inter-op parallelism. If '
                     'set to 0, the system will pick an appropriate number.')
flags.DEFINE_boolean('force_gpu_compatible', False,
                     'whether to enable force_gpu_compatible in GPU_Options')
flags.DEFINE_boolean('allow_growth', None,
                     'whether to enable allow_growth in GPU_Options')
flags.DEFINE_float('gpu_memory_frac_for_testing', 0,
                   'If non-zero, the fraction of GPU memory that will be used. '
                   'Useful for testing the benchmark script, as this allows '
                   'distributed mode to be run on a single machine. For '
                   'example, if there are two tasks, each can be allocated '
                   '~40 percent of the memory on a single machine',
                   lower_bound=0., upper_bound=1.)
flags.DEFINE_boolean('xla', False, 'whether to enable XLA')
flags.DEFINE_boolean('enable_layout_optimizer', False,
                     'whether to enable layout optimizer')
# hparams
flags.DEFINE_enum('optimizer', 'sgd', ('momentum', 'sgd', 'rmsprop'),
                  'Optimizer to use: momentum or sgd or rmsprop')
flags.DEFINE_string('piecewise_learning_rate_schedule', None,
                    'Specifies a piecewise learning rate schedule based on the '
                    'number of epochs. This is the form LR0;E1;LR1;...;En;LRn, '
                    'where each LRi is a learning rate and each Ei is an epoch '
                    'indexed from 0. The learning rate is LRi if the '
                    'E(i-1) <= current_epoch < Ei. For example, if this '
                    'paramater is 0.3;10;0.2;25;0.1, the learning rate is 0.3 '
                    'for the first 10 epochs, then is 0.2 for the next 15 '
                    'epochs, then is 0.1 until training ends.')
flags.DEFINE_float('weight_decay', 0.00004,
                   'Weight decay factor for training.')
flags.DEFINE_float('epoch', None,
                   'number of epochs to run, excluding warmup. '
                   'This and --num_batches cannot both be specified.')
flags.DEFINE_integer('batch_size', 0, 'batch size per compute device')
flags.DEFINE_float('num_learning_rate_warmup_epochs', 0,
                   'Slowly increase to the initial learning rate in the first '
                   'num_learning_rate_warmup_epochs linearly.')
flags.DEFINE_boolean('eval', False, 'whether use eval or benchmarking')


class CheckpointNotFoundException(Exception):
  pass

def loss_function(logits, labels, aux_logits):
  """Loss function."""
  with tf.name_scope('xentropy'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  if aux_logits is not None:
    with tf.name_scope('aux_xentropy'):
      aux_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
          logits=aux_logits, labels=labels)
      aux_loss = 0.4 * tf.reduce_mean(aux_cross_entropy, name='aux_loss')
      loss = tf.add_n([loss, aux_loss])
  return loss

def create_config_proto():
  """Returns session config proto.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.intra_op_parallelism_threads = FLAGS.num_intra_threads
  config.inter_op_parallelism_threads = FLAGS.num_inter_threads
  config.gpu_options.force_gpu_compatible = FLAGS.force_gpu_compatible
  if FLAGS.allow_growth is not None:
    config.gpu_options.allow_growth = FLAGS.allow_growth
  if FLAGS.gpu_memory_frac_for_testing > 0:
    config.gpu_options.per_process_gpu_memory_fraction = (
        FLAGS.gpu_memory_frac_for_testing)
  if FLAGS.xla:
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  if FLAGS.enable_layout_optimizer:
    config.graph_options.rewrite_options.layout_optimizer = (
        rewriter_config_pb2.RewriterConfig.ON)
  return config

# How many digits to show for the loss and accuracies during training.
LOSS_AND_ACCURACY_DIGITS_TO_SHOW = 3

def get_num_batches(epochs, num_examples_per_epoch, batch_size):
    num_batches = int(float(epochs) * num_examples_per_epoch / batch_size)
    return num_batches

def get_piecewise_learning_rate(piecewise_learning_rate_schedule,
                                global_step, num_batches_per_epoch):
  pieces = piecewise_learning_rate_schedule.split(';')
  if len(pieces) % 2 == 0:
    raise ValueError('--piecewise_learning_rate_schedule must have an odd '
                     'number of components')
  values = []
  boundaries = []
  for i, piece in enumerate(pieces):
    if i % 2 == 0:
      try:
        values.append(float(piece))
      except ValueError:
        raise ValueError('Invalid learning rate: ' + piece)
    else:
      try:
        boundaries.append(int(int(piece) * num_batches_per_epoch) - 1)
      except ValueError:
        raise ValueError('Invalid epoch: ' + piece)
  return tf.train.piecewise_constant(global_step, boundaries, values,
                                     name='piecewise_learning_rate')

def get_learning_rate(global_step, num_examples_per_epoch,
                      batch_size):
  num_batches_per_epoch = (float(num_examples_per_epoch) / batch_size)
  learning_rate = get_piecewise_learning_rate(
      FLAGS.piecewise_learning_rate_schedule,
      global_step, num_batches_per_epoch)
  if FLAGS.num_learning_rate_warmup_epochs > 0:
    warmup_steps = int(num_batches_per_epoch *
                       FLAGS.num_learning_rate_warmup_epochs)
    init_lr = float(FLAGS.piecewise_learning_rate_schedule.split(';')[0])
    warmup_lr = init_lr * tf.cast(global_step, tf.float32) / tf.cast(
        warmup_steps, tf.float32)
    learning_rate = tf.cond(global_step < warmup_steps,
                            lambda: warmup_lr, lambda: learning_rate)
  return learning_rate

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
    self.train_dir = FLAGS.train_dir
    if FLAGS.save:
      self.train_dir = FLAGS.save
    if not self.train_dir:
      raise ValueError('train_dir or save not defined')
    self.cpu_device = '/cpu:0'
    self.num_gpus = FLAGS.num_gpus
    self.raw_devices = [
        '/%s:%i' % (FLAGS.device, i)
        for i in xrange(self.num_gpus)
    ]
    self.data_type = tf.float32
    self.data_format = FLAGS.data_format
    if (FLAGS.device == 'cpu' and FLAGS.data_format == 'NCHW'):
      raise ValueError('device=cpu requires that data_format=NHWC')
    self.optimizer = FLAGS.optimizer
    self.variable_update = FLAGS.variable_update
    self.batch_size = FLAGS.batch_size * self.num_gpus
    if not FLAGS.networkDirectory:
      raise ValueError('networkDirectory not defined')
    self.model = model_config.get_model_config(
        FLAGS.networkDirectory,
        FLAGS.network)
    if not FLAGS.train_db:
      raise ValueError('train_db not defined')
    if not FLAGS.validation_db:
      raise ValueError('validation_db not defined')
    if not FLAGS.labels_list:
      raise ValueError('labels_list not defined')
    self.dataset = data_config.DataLoader(
        self.model.get_image_size(), self.model.get_image_size(),
        self.batch_size, FLAGS.num_gpus, self.cpu_device,
        self.raw_devices, self.data_type, FLAGS.train_db,
        FLAGS.validation_db, FLAGS.labels_list,
        FLAGS.summary_verbosity)
    if not FLAGS.epoch:
      raise ValueError('epoch not defined')
    self.num_epochs = FLAGS.epoch
    if not FLAGS.batch_size:
      raise ValueError('batch_size not defined')
    self.train_batches = get_num_batches(self.num_epochs,
        self.dataset.num_examples_per_epoch('train'),
        self.batch_size)
    self.val_batches = get_num_batches(1,
        self.dataset.num_examples_per_epoch('validation'),
        self.batch_size)
    if not FLAGS.piecewise_learning_rate_schedule:
      raise ValueError('piecewise_learning_rate_schedule not defined')
    self.display_every = FLAGS.display_every
    if self.display_every > 0:
      display = self.train_batches // self.num_epochs // 10
      if display == 0:
        display = 1
      logging.info("adjust display_every: %d" % display)
      self.display_every = display


  def print_info(self):
    """Print basic information."""
    print('Batch size:  %s global' % (self.batch_size))
    print('             %s per device' % (self.batch_size /
                                           len(self.raw_devices)))
    print('Num train batches: %d' % self.train_batches)
    print('Num val batches: %d' % self.val_batches)
    print('Num epochs:  %.2f' % self.num_epochs)
    print('Devices:     %s' % self.raw_devices)
    print('Data format: %s' % self.data_format)
    print('Layout optimizer: %s' % FLAGS.enable_layout_optimizer)
    print('Optimizer:   %s' % FLAGS.optimizer)
    print('Variables:   %s' % self.variable_update)
    print('Lr:   %s' % FLAGS.piecewise_learning_rate_schedule)
    print('==========')

  def _add_forward_pass_and_gradients(self,
                                      phase_train,
                                      device_num,
                                      images, labels):
    """Add ops for forward-pass and gradient computations."""
    nclass = self.dataset.num_classes
    image_size = self.model.get_image_size()
    # build network per tower
    with tf.device(self.raw_devices[device_num]):
      logits, aux_logits = self.model.build_network(
          images, phase_train, nclass, self.dataset.depth, self.data_type,
          self.data_format)
      results = {}  # The return value
      top_1_op = tf.reduce_sum(
          tf.cast(tf.nn.in_top_k(logits, labels, 1), self.data_type))
      top_5_op = tf.reduce_sum(
          tf.cast(tf.nn.in_top_k(logits, labels, 5), self.data_type))
      results['top_1_op'] = top_1_op
      results['top_5_op'] = top_5_op
      if not phase_train:
        results['logits'] = logits
        return results
      base_loss = loss_function(logits, labels, aux_logits=aux_logits)
      # get per tower trainable variable
      params = [
          v for v in tf.trainable_variables()
          if v.name.startswith('tower_%s/' % device_num)
      ]
      fp32_params = params
      total_loss = base_loss
      if device_num == len(self.raw_devices) - 1:
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in fp32_params])
        weight_decay = FLAGS.weight_decay
        if weight_decay is not None and weight_decay != 0.:
          total_loss += len(self.raw_devices) * weight_decay * l2_loss
      grads = tf.gradients(total_loss, params,
          aggregation_method=tf.AggregationMethod.DEFAULT)
      param_refs = [
          v for v in tf.trainable_variables()
          if v.name.startswith('tower_%s/' % device_num)
      ]
      gradvars = list(zip(grads, param_refs))
      results['loss'] = total_loss
      results['gradvars'] = gradvars
      return results

  def allreduce_algorithm(self):
    if FLAGS.device == 'cpu':
      return batch_allreduce.CopyToDeviceAlgorithm(['/cpu:0'])
    else:
      gpu_indices = [x for x in range(FLAGS.num_gpus)]
      return batch_allreduce.AllReduceSpecAlgorithm('nccl', gpu_indices, 0, 10)

  def preprocess_device_grads(self, device_grads):
    grads_to_reduce = [[g for g, _ in grad_vars] for grad_vars in device_grads]
    algorithm = self.allreduce_algorithm()
    reduced_grads, self._warmup_ops = algorithm.batch_all_reduce(
        grads_to_reduce, 0, False, False)
    reduced_device_grads = [[
        (g, v) for g, (_, v) in zip(grads, grad_vars)
    ] for grads, grad_vars in zip(reduced_grads, device_grads)]
    return self.raw_devices, reduced_device_grads

  def get_optimizer(self, learning_rate):
    """Returns the optimizer that should be used based on params."""
    if self.optimizer == 'momentum':
      opt = tf.train.MomentumOptimizer(
          learning_rate, 0.9, use_nesterov=True)
    elif self.optimizer == 'sgd':
      opt = tf.train.GradientDescentOptimizer(learning_rate)
    elif self.optimizer == 'rmsprop':
      opt = tf.train.RMSPropOptimizer(
          learning_rate, 0.9,
          momentum=0.9,
          epsilon=1.0)
    else:
      raise ValueError('Optimizer "%s" was not recognized',
                       FLAGS.optimizer)
    return opt

  def _build_fetches(self, global_step, all_logits, losses, device_grads,
                     update_ops, all_top_1_ops, all_top_5_ops,
                     phase_train):
    """Complete construction of model graph, populating the fetches map."""
    fetches = {}
    if all_top_1_ops:
      fetches['top_1_accuracy'] = tf.reduce_sum(all_top_1_ops) / self.batch_size
      if FLAGS.summary_verbosity >= 1:
        tf.summary.scalar('top_1_accuracy', fetches['top_1_accuracy'])
    if all_top_5_ops:
      fetches['top_5_accuracy'] = tf.reduce_sum(all_top_5_ops) / self.batch_size
      if FLAGS.summary_verbosity >= 1:
        tf.summary.scalar('top_5_accuracy', fetches['top_5_accuracy'])
    # validation
    if not phase_train:
      fetches['all_logits'] = tf.concat(all_logits, 0)
      return fetches
    apply_gradient_devices, gradient_state = (
        self.preprocess_device_grads(device_grads))
    training_ops = []
    for d, device in enumerate(apply_gradient_devices):
      with tf.device(device):
        average_loss = tf.reduce_mean(losses)
        avg_grads = gradient_state[d]
        learning_rate = get_learning_rate(
            global_step,
            self.dataset.num_examples_per_epoch(),
            self.batch_size)
        learning_rate = tf.identity(learning_rate, name='learning_rate')
        opt = self.get_optimizer(learning_rate)
        training_ops.extend([opt.apply_gradients(avg_grads)])
    train_op = tf.group(*(training_ops + update_ops))
    # summaries
    with tf.device(self.cpu_device):
      if FLAGS.summary_verbosity >= 1:
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('average_loss', average_loss)
        if FLAGS.summary_verbosity >= 2:
          # Histogram of log values of all non-zero gradients.
          all_grads = []
          for grad, var in avg_grads:
            all_grads.append(tf.reshape(grad, [-1]))
          grads = tf.abs(tf.concat(all_grads, 0))
          # exclude grads with zero values.
          indices_for_non_zero_grads = tf.where(tf.not_equal(grads, 0))
          log_grads = tf.reshape(
              tf.log(tf.gather(grads, indices_for_non_zero_grads)), [-1])
          tf.summary.histogram('log_gradients', log_grads)
        if FLAGS.summary_verbosity >= 3:
          for grad, var in avg_grads:
            if grad is not None:
              tf.summary.histogram(var.op.name + '/gradients', grad)
          for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
    fetches['learning_rate'] = learning_rate
    fetches['train_op'] = train_op
    fetches['average_loss'] = average_loss
    return fetches

  GraphInfo = namedtuple(  # pylint: disable=invalid-name
      'GraphInfo',
      [
          # Fetches of sess.run()
          'fetches',
          # The global step variable
          'global_step',
      ])

  def _build_graph(self, phase_train=True):
    tf.set_random_seed(1234)
    np.random.seed(4321)
    if phase_train:
      print('Generating train model')
    else:
      print('Generating validation model')
    losses = []
    device_grads = []
    all_logits = []
    all_top_1_ops = []
    all_top_5_ops = []
    # get global step
    with tf.device(self.cpu_device):
      global_step = tf.train.get_or_create_global_step()
    # get datainputs and build model
    if phase_train:
      with tf.name_scope('train_data'):
        images_split, labels_split = self.dataset.get_inputs('train')
    else:
      with tf.name_scope('validation_data'):
        images_split, labels_split = self.dataset.get_inputs('validation')
    update_ops = None
    for device_num in range(len(self.raw_devices)):
      # only use first tower for validation
      current_scope = 'tower_%i' % device_num
      with tf.variable_scope(current_scope, reuse=tf.AUTO_REUSE):
        name_scope = 'tower_%i' % device_num
        results = self._add_forward_pass_and_gradients(
            phase_train, device_num,
            images_split[device_num],
            labels_split[device_num])
        if phase_train:
          losses.append(results['loss'])
          device_grads.append(results['gradvars'])
        else:
          all_logits.append(results['logits'])
        all_top_1_ops.append(results['top_1_op'])
        all_top_5_ops.append(results['top_5_op'])
        if device_num == 0:
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
    fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
                                  update_ops, all_top_1_ops,
                                  all_top_5_ops, phase_train)
    if not phase_train:
      return fetches
    # construct init op group
    fetches_list = nest.flatten(list(fetches.values()))
    main_fetch_group = tf.group(*fetches_list)
    with tf.device(self.cpu_device):
      with tf.control_dependencies([main_fetch_group]):
        fetches['inc_global_step'] = global_step.assign_add(1)
    # return GraphInfo
    return BenchmarkCNN.GraphInfo(
        fetches=fetches,
        global_step=global_step)

  def benchmark_one_step(self, sess, fetches, step,
                         batch_size, step_train_times, summary_op):
    summary_str = None
    start_time = time.time()
    if summary_op is None:
      results = sess.run(fetches)
    else:
      (results, summary_str) = sess.run([fetches, summary_op])
    lossval = results['average_loss']
    train_time = time.time() - start_time
    step_train_times.append(train_time)
    num_batches_per_epoch = (float(self.dataset.num_examples_per_epoch('train')) / self.batch_size)
    if (step >= 0 and
        (step == 0 or (step + 1) % self.display_every == 0)):
      log_str = "Training (epoch %.*f): loss = %.*f, accuracy = %.*f, lr = %.*f" % (
          2, float(step + 1)/num_batches_per_epoch,
          LOSS_AND_ACCURACY_DIGITS_TO_SHOW, lossval,
          LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_1_accuracy'],
          5, results['learning_rate'])
      #log_str = '%i\t%.*f' % (
      #  step + 1,
      #  LOSS_AND_ACCURACY_DIGITS_TO_SHOW, lossval)
      #log_str += '\t%.*f\t%.*f' % (
      #    LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_1_accuracy'],
      #    LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_5_accuracy'])
      logging.info(log_str)
    return summary_str

  def eval_one_epoch(self, sess, step, fetches):
    local_step = 0
    top_1_accuracy_sum = 0.0
    top_5_accuracy_sum = 0.0
    self.val_batches = 1
    while local_step < self.val_batches:
      results = sess.run(fetches)
      top_1_accuracy_sum += results['top_1_accuracy']
      top_5_accuracy_sum += results['top_5_accuracy']
      local_step += 1
    accuracy_at_1 = top_1_accuracy_sum / int(self.val_batches)
    accuracy_at_5 = top_5_accuracy_sum / int(self.val_batches)
    num_batches_per_epoch = (float(self.dataset.num_examples_per_epoch('train')) / self.batch_size)
    logging.info('Validation (epoch %.*f): accuracy = %.4f, top5_accuracy = %.4f' %
                 (2, float(step + 1)/num_batches_per_epoch,
                  accuracy_at_1, accuracy_at_5))

  def _benchmark_graph(self, graph_info, val_fetches,
                       local_var_init_op_group):
    summary_op = tf.summary.merge_all()
    summary_writer = None
    if (FLAGS.summary_verbosity and self.train_dir and
        FLAGS.save_summaries_steps > 0):
      summary_writer = tf.summary.FileWriter(self.train_dir,
                                             tf.get_default_graph())
    saver = tf.train.Saver(tf.global_variables(), save_relative_paths=True)
    init_run_options = tf.RunOptions()
    sv = tf.train.Supervisor(
        is_chief=True,
        # Log dir should be unset on non-chief workers to prevent Horovod
        # workers from corrupting each other's checkpoints.
        logdir=self.train_dir,
        ready_for_local_init_op=None,
        local_init_op=local_var_init_op_group,
        saver=saver,
        global_step=graph_info.global_step,
        summary_op=None,
        save_model_secs=0,
        summary_writer=summary_writer)
    step_train_times = []
    with sv.managed_session(
        master='',
        config=create_config_proto(),
        start_standard_services=False) as sess:
      self.init_global_step, = sess.run([graph_info.global_step])
      print('Running warm up')
      print('init_global_step: {}'.format(self.init_global_step))
      local_step = -10
      end_local_step = self.train_batches - self.init_global_step
      loop_start_time = time.time()
      while local_step < end_local_step:
        if local_step == 0:
          print('Done warm up')
          header_str = ('Step\ttotal_loss')
          header_str += '\ttop_1_accuracy\ttop_5_accuracy'
          print(header_str)
          step_train_times = []
          loop_start_time = time.time()
        if (summary_writer and 
            (local_step + 1) % FLAGS.save_summaries_steps == 0):
          fetch_summary = summary_op
        else:
          fetch_summary = None
        summary_str = self.benchmark_one_step(
            sess, graph_info.fetches, local_step,
            self.batch_size, step_train_times, fetch_summary)
        if summary_str is not None:
          sv.summary_computed(sess, summary_str)
        local_step += 1
        #validation per epoch end
        num_batches_per_epoch = (float(self.dataset.num_examples_per_epoch('train')) // self.batch_size)
        if local_step % num_batches_per_epoch == 0:
          self.eval_one_epoch(sess, local_step, val_fetches)
      # loop End
      loop_end_time = time.time()
      elapsed_time = loop_end_time - loop_start_time
      average_wall_time = elapsed_time / local_step if local_step > 0 else 0
      images_per_sec = (local_step * self.batch_size /
                        elapsed_time)
      print('-' * 64)
      print('total images/sec: %.2f' % images_per_sec)
      print('-' * 64)
      # Save the model checkpoint.
      if self.train_dir is not None:
        checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
        if not gfile.Exists(self.train_dir):
          gfile.MakeDirs(self.train_dir)
        sv.saver.save(sess, checkpoint_path, graph_info.global_step)
        filename_graph = os.path.join(self.train_dir, "graph.bp")
        if not os.path.isfile(filename_graph):
          with open(filename_graph, 'wb') as f:
            logging.info('Saving graph to %s', filename_graph)
            f.write(sess.graph_def.SerializeToString())
            logging.info('Saved graph to %s', filename_graph)
    sv.stop()

  def _eval_cnn(self):
    fetches = self._build_graph(phase_train=False)
    params = []
    for v in tf.global_variables():
      split_name = v.name.split('/')
      if split_name[0] == 'tower_0' or not v.name.startswith('tower'):
        params.append(v)
    saver = tf.train.Saver(params)
    # get local init op group
    local_var_init_op_group = self.get_init_op_group()
    self._eval_once(saver, local_var_init_op_group, fetches)

  def _eval_once(self, saver, local_var_init_op_group, fetches):
    with tf.Session(
        config=create_config_proto()) as sess:
      try:
        global_step = load_checkpoint(saver, sess, self.train_dir)
      except CheckpointNotFoundException:
        print('Checkpoint not found in %s' % self.train_dir)
        return
      sess.run(local_var_init_op_group)
      local_step = 0
      top_1_accuracy_sum = 0.0
      top_5_accuracy_sum = 0.0
      total_eval_count = self.val_batches * self.batch_size
      while local_step < self.val_batches:
        results = sess.run(fetches)
        top_1_accuracy_sum += results['top_1_accuracy']
        top_5_accuracy_sum += results['top_5_accuracy']
        local_step += 1
      accuracy_at_1 = top_1_accuracy_sum / int(self.val_batches)
      accuracy_at_5 = top_5_accuracy_sum / int(self.val_batches)
      print('Accuracy @ 1 = %.4f Accuracy @ 5 = %.4f [%d examples]' %
             (accuracy_at_1, accuracy_at_5, total_eval_count))

  def run(self):
    if FLAGS.eval:
      with tf.Graph().as_default():
        return self._eval_cnn()
    # _benchmark_cnn
    graph = tf.Graph()
    with graph.as_default():
      #with tf.name_scope('train'):
      train_result = self._build_graph()
      #with tf.name_scope('val'):
      val_fetches = self._build_graph(phase_train=False)
      local_var_init_op_group = self.get_init_op_group()
    with graph.as_default():
      self._benchmark_graph(train_result, val_fetches,
                            local_var_init_op_group)

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
  bench.print_info()
  bench.run()

if __name__ == '__main__':
  tf.app.run()
