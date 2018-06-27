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
"""TensorFlow benchmark library.

See the README for more information.
"""

from __future__ import print_function

import argparse
from collections import namedtuple
import math
import multiprocessing
import os
import threading
import time

#from absl import flags as absl_flags
import numpy as np

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from google.protobuf import text_format

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest
import benchmark_storage
import cnn_util
import constants
import data_utils
import datasets
#import flags
import variable_mgr
import variable_mgr_util
#from cnn_util import log_fn
from models import model_config
from platforms import util as platforms_util

import logging
from tensorflow.python.lib.io import file_io

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

_DEFAULT_NUM_BATCHES = 100

# TODO(reedwm): add upper_bound and lower_bound to appropriate integer and
# float flags, and change certain string flags to enum flags.

flags.DEFINE_string('model', 'trivial',
                    'Name of the model to run, the list of supported models '
                    'are defined in models/model.py')
# The code will first check if it's running under benchmarking mode
# or evaluation mode, depending on 'eval':
# Under the evaluation mode, this script will read a saved model,
#   and compute the accuracy of the model against a validation dataset.
#   Additional ops for accuracy and top_k predictors are only used under
#   this mode.
# Under the benchmarking mode, user can specify whether nor not to use
#   the forward-only option, which will only compute the loss function.
#   forward-only cannot be enabled with eval at the same time.
flags.DEFINE_boolean('eval', False, 'whether use eval or benchmarking')
flags.DEFINE_integer('eval_interval_secs', 0,
                     'How often to run eval on saved checkpoints. Usually the '
                     'same as save_model_secs from the corresponding training '
                     'run. Pass 0 to eval only once.')
#flags.DEFINE_boolean('forward_only', False,
#                     'whether use forward-only or training for benchmarking')
flags.DEFINE_boolean('print_training_accuracy', True,
                     'whether to calculate and print training accuracy during '
                     'training')
flags.DEFINE_integer('batch_size', 0, 'batch size per compute device')
#flags.DEFINE_integer('batch_group_size', 1,
#                     'number of groups of batches processed in the image '
#                     'producer.')
flags.DEFINE_integer('num_batches', None, 'number of batches to run, excluding '
                     'warmup. Defaults to %d' % _DEFAULT_NUM_BATCHES)
flags.DEFINE_float('num_epochs', None,
                   'number of epochs to run, excluding warmup. '
                   'This and --num_batches cannot both be specified.')
flags.DEFINE_integer('num_gpus', 1, 'the number of GPUs to run on')
flags.DEFINE_integer('display_every', 100,
                     'Number of local steps after which progress is printed '
                     'out')
flags.DEFINE_string('data_dir', None,
                    'Path to dataset in TFRecord format (aka Example '
                    'protobufs). If not specified, synthetic data will be '
                    'used.')
flags.DEFINE_string('data_name', None,
                    'Name of dataset: imagenet or cifar10. If not specified, '
                    'it is automatically guessed based on data_dir.')
flags.DEFINE_string('resize_method', 'bilinear',
                    'Method for resizing input images: crop, nearest, '
                    'bilinear, bicubic, area, or round_robin. The `crop` mode '
                    'requires source images to be at least as large as the '
                    'network input size. The `round_robin` mode applies '
                    'different resize methods based on position in a batch in '
                    'a round-robin fashion. Other modes support any sizes and '
                    'apply random bbox distortions before resizing (even with '
                    'distortions=False).')
flags.DEFINE_boolean('distortions', True,
                     'Enable/disable distortions during image preprocessing. '
                     'These include bbox and color distortions.')
flags.DEFINE_boolean('use_datasets', True,
                     'Enable use of datasets for input pipeline')
flags.DEFINE_string('input_preprocessor', 'default',
                    'Name of input preprocessor. The list of supported input '
                    'preprocessors are defined in preprocessing.py.')
flags.DEFINE_string('gpu_thread_mode', 'gpu_private',
                    'Methods to assign GPU host work to threads. '
                    'global: all GPUs and CPUs share the same global threads; '
                    'gpu_private: a private threadpool for each GPU; '
                    'gpu_shared: all GPUs share the same threadpool.')
flags.DEFINE_integer('per_gpu_thread_count', 0,
                     'The number of threads to use for GPU. Only valid when '
                     'gpu_thread_mode is not global.')
flags.DEFINE_boolean('hierarchical_copy', False,
                     'Use hierarchical copies. Currently only optimized for '
                     'use on a DGX-1 with 8 GPUs and may perform poorly on '
                     'other hardware. Requires --num_gpus > 1, and only '
                     'recommended when --num_gpus=8')
# TODO(hinsu): Support auto-detection of the network topology while still
# retaining the ability to specify a particular topology for debugging.
flags.DEFINE_enum(
    'network_topology', constants.NetworkTopology.DGX1,
    (constants.NetworkTopology.DGX1, constants.NetworkTopology.GCP_V100),
    'Network topology specifies the topology used to connect multiple devices. '
    'Network topology is used to decide the hierarchy to use for the '
    'hierarchical_copy.')
flags.DEFINE_integer('gradient_repacking', 0, 'Use gradient repacking. It'
                     'currently only works with replicated mode. At the end of'
                     'of each step, it repacks the gradients for more efficient'
                     'cross-device transportation. A non-zero value specifies'
                     'the number of split packs that will be formed.',
                     lower_bound=0)
flags.DEFINE_boolean('compact_gradient_transfer', True, 'Compact gradient'
                     'as much as possible for cross-device transfer and '
                     'aggregation.')
flags.DEFINE_enum('variable_consistency', 'strong', ('strong', 'relaxed'),
                  'The data consistency for trainable variables. With strong '
                  'consistency, the variable always have the updates from '
                  'previous step. With relaxed consistency, all the updates '
                  'will eventually show up in the variables. Likely one step '
                  'behind.')
flags.DEFINE_boolean('cache_data', False,
                     'Enable use of a special datasets pipeline that reads a '
                     'single TFRecord into memory and repeats it infinitely '
                     'many times. The purpose of this flag is to make it '
                     'possible to write regression tests that are not '
                     'bottlenecked by CNS throughput.')
flags.DEFINE_enum('local_parameter_device', 'gpu', ('cpu', 'gpu', 'CPU', 'GPU'),
                  'Device to use as parameter server: cpu or gpu. For '
                  'distributed training, it can affect where caching of '
                  'variables happens.')
flags.DEFINE_enum('device', 'gpu', ('cpu', 'gpu', 'CPU', 'GPU'),
                  'Device to use for computation: cpu or gpu')
flags.DEFINE_enum('data_format', 'NCHW', ('NHWC', 'NCHW'),
                  'Data layout to use: NHWC (TF native) or NCHW (cuDNN '
                  'native, requires GPU).')
flags.DEFINE_integer('num_intra_threads', 1,
                     'Number of threads to use for intra-op parallelism. If '
                     'set to 0, the system will pick an appropriate number.')
flags.DEFINE_integer('num_inter_threads', 0,
                     'Number of threads to use for inter-op parallelism. If '
                     'set to 0, the system will pick an appropriate number.')
flags.DEFINE_string('trace_file', '',
                    'Enable TensorFlow tracing and write trace to this file.')
flags.DEFINE_boolean('use_chrome_trace_format', True,
                     'If True, the trace_file, if specified, will be in a '
                     'Chrome trace format. If False, then it will be a '
                     'StepStats raw proto.')
_NUM_STEPS_TO_PROFILE = 10
_NUM_OPS_TO_PRINT = 20
flags.DEFINE_string('tfprof_file', None,
                    'If specified, write a tfprof ProfileProto to this file. '
                    'The performance and other aspects of the model can then '
                    'be analyzed with tfprof. See '
                    'https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/command_line.md '  # pylint: disable=line-too-long
                    'for more info on how to do this. The first %d steps '
                    'are profiled. Additionally, the top %d most time '
                    'consuming ops will be printed.\n'
                    'Note: profiling with tfprof is very slow, but most of the '
                    'overhead is spent between steps. So, profiling results '
                    'are more accurate than the slowdown would suggest.' %
                    (_NUM_STEPS_TO_PROFILE, _NUM_OPS_TO_PRINT))
#flags.DEFINE_string('partitioned_graph_file_prefix', None,
#                    'If specified, after the graph has been partitioned and '
#                    'optimized, write out each partitioned graph to a file '
#                    'with the given prefix.')
flags.DEFINE_enum('optimizer', 'sgd', ('momentum', 'sgd', 'rmsprop'),
                  'Optimizer to use: momentum or sgd or rmsprop')
flags.DEFINE_float('init_learning_rate', None,
                   'Initial learning rate for training.')
flags.DEFINE_string('piecewise_learning_rate_schedule', None,
                    'Specifies a piecewise learning rate schedule based on the '
                    'number of epochs. This is the form LR0;E1;LR1;...;En;LRn, '
                    'where each LRi is a learning rate and each Ei is an epoch '
                    'indexed from 0. The learning rate is LRi if the '
                    'E(i-1) <= current_epoch < Ei. For example, if this '
                    'paramater is 0.3;10;0.2;25;0.1, the learning rate is 0.3 '
                    'for the first 10 epochs, then is 0.2 for the next 15 '
                    'epochs, then is 0.1 until training ends.')
flags.DEFINE_float('num_epochs_per_decay', 0,
                   'Steps after which learning rate decays. If 0, the learning '
                   'rate does not decay.')
flags.DEFINE_float('learning_rate_decay_factor', 0,
                   'Learning rate decay factor. Decay by this factor every '
                   '`num_epochs_per_decay` epochs. If 0, learning rate does '
                   'not decay.')
flags.DEFINE_float('num_learning_rate_warmup_epochs', 0,
                   'Slowly increase to the initial learning rate in the first '
                   'num_learning_rate_warmup_epochs linearly.')
flags.DEFINE_float('minimum_learning_rate', 0,
                   'The minimum learning rate. The learning rate will '
                   'never decay past this value. Requires `learning_rate`, '
                   '`num_epochs_per_decay` and `learning_rate_decay_factor` to '
                   'be set.')
flags.DEFINE_float('momentum', 0.9, 'Momentum for training.')
flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum in RMSProp.')
flags.DEFINE_float('rmsprop_epsilon', 1.0, 'Epsilon term for RMSProp.')
flags.DEFINE_float('gradient_clip', None,
                   'Gradient clipping magnitude. Disabled by default.')
flags.DEFINE_float('weight_decay', 0.00004,
                   'Weight decay factor for training.')
flags.DEFINE_float('gpu_memory_frac_for_testing', 0,
                   'If non-zero, the fraction of GPU memory that will be used. '
                   'Useful for testing the benchmark script, as this allows '
                   'distributed mode to be run on a single machine. For '
                   'example, if there are two tasks, each can be allocated '
                   '~40 percent of the memory on a single machine',
                   lower_bound=0., upper_bound=1.)
flags.DEFINE_boolean('use_tf_layers', True,
                     'If True, use tf.layers for neural network layers. This '
                     'should not affect performance or accuracy in any way.')
flags.DEFINE_integer('tf_random_seed', 1234,
                     'The TensorFlow random seed. Useful for debugging NaNs, '
                     'as this can be set to various values to see if the NaNs '
                     'depend on the seed.')
flags.DEFINE_string('debugger', None,
                    'If set, use the TensorFlow debugger. If set to "cli", use '
                    'the local CLI debugger. Otherwise, this must be in the '
                    'form hostname:port (e.g., localhost:7007) in which case '
                    'the experimental TensorBoard debugger will be used')
#flags.DEFINE_boolean('use_python32_barrier', False,
#                     'When on, use threading.Barrier at Python 3.2.')

flags.DEFINE_boolean('datasets_use_prefetch', True,
                     'Enable use of prefetched datasets for input pipeline. '
                     'This option is meaningless if use_datasets=False.')
flags.DEFINE_integer('datasets_prefetch_buffer_size', 1,
                     'Prefetching op buffer size per compute device.')
flags.DEFINE_integer('datasets_num_private_threads', None,
                     'Number of threads for a private threadpool created for '
                     'all datasets computation. By default, we pick an '
                     'appropriate number. If set to 0, we use the default '
                     'tf-Compute threads for dataset operations.')

# Performance tuning parameters.
flags.DEFINE_boolean('winograd_nonfused', True,
                     'Enable/disable using the Winograd non-fused algorithms.')
flags.DEFINE_boolean(
    'batchnorm_persistent', True,
    'Enable/disable using the CUDNN_BATCHNORM_SPATIAL_PERSISTENT '
    'mode for batchnorm.')
flags.DEFINE_boolean('sync_on_finish', False,
                     'Enable/disable whether the devices are synced after each '
                     'step.')
#flags.DEFINE_boolean('staged_vars', False,
#                     'whether the variables are staged from the main '
#                     'computation')
flags.DEFINE_boolean('force_gpu_compatible', False,
                     'whether to enable force_gpu_compatible in GPU_Options')
flags.DEFINE_boolean('allow_growth', None,
                     'whether to enable allow_growth in GPU_Options')
flags.DEFINE_boolean('xla', False, 'whether to enable XLA')
flags.DEFINE_boolean('fuse_decode_and_crop', True,
                     'Fuse decode_and_crop for image preprocessing.')
flags.DEFINE_boolean('distort_color_in_yiq', True,
                     'Distort color of input images in YIQ space.')
flags.DEFINE_boolean('enable_layout_optimizer', False,
                     'whether to enable layout optimizer')
flags.DEFINE_string('rewriter_config', None,
                    'Config for graph optimizers, described as a '
                    'RewriterConfig proto buffer.')
flags.DEFINE_enum('loss_type_to_report', 'total_loss',
                  ('base_loss', 'total_loss'),
                  'Which type of loss to output and to write summaries for. '
                  'The total loss includes L2 loss while the base loss does '
                  'not. Note that the total loss is always used while '
                  'computing gradients during training if weight_decay > 0, '
                  'but explicitly computing the total loss, instead of just '
                  'computing its gradients, can have a performance impact.')
flags.DEFINE_boolean('single_l2_loss_op', False,
                     'If True, instead of using an L2 loss op per variable, '
                     'concatenate the variables into a single tensor and do a '
                     'single L2 loss on the concatenated tensor.')
flags.DEFINE_boolean('use_resource_vars', False,
                     'Use resource variables instead of normal variables. '
                     'Resource variables are slower, but this option is useful '
                     'for debugging their performance.')
# Performance tuning specific to MKL.
flags.DEFINE_boolean('mkl', False, 'If true, set MKL environment variables.')
flags.DEFINE_integer('kmp_blocktime', 30,
                     'The time, in milliseconds, that a thread should wait, '
                     'after completing the execution of a parallel region, '
                     'before sleeping')
flags.DEFINE_string('kmp_affinity', 'granularity=fine,verbose,compact,1,0',
                    'Restricts execution of certain threads (virtual execution '
                    'units) to a subset of the physical processing units in a '
                    'multiprocessor computer.')
flags.DEFINE_integer('kmp_settings', 1,
                     'If set to 1, MKL settings will be printed.')
flags.DEFINE_enum('variable_update', 'replicated',
                  ('parameter_server', 'replicated', 'distributed_replicated',
                   'independent', 'distributed_all_reduce', 'horovod'),
                  'The method for managing variables: parameter_server, '
                  'replicated, distributed_replicated, independent, '
                  'distributed_all_reduce, horovod')
flags.DEFINE_string('all_reduce_spec', 'nccl',
                    'A specification of the all_reduce algorithm to be used '
                    'for reducing gradients.  For more details, see '
                    'parse_all_reduce_spec in variable_mgr.py.  An '
                    'all_reduce_spec has BNF form:\n'
                    'int ::= positive whole number\n'
                    'g_int ::= int[KkMGT]?\n'
                    'alg_spec ::= alg | alg#int\n'
                    'range_spec ::= alg_spec | alg_spec/alg_spec\n'
                    'spec ::= range_spec | range_spec:g_int:range_spec\n'
                    'NOTE: not all syntactically correct constructs are '
                    'supported.\n\n'
                    'Examples:\n '
                    '"xring" == use one global ring reduction for all '
                    'tensors\n'
                    '"pscpu" == use CPU at worker 0 to reduce all tensors\n'
                    '"nccl" == use NCCL to locally reduce all tensors.  '
                    'Limited to 1 worker.\n'
                    '"nccl/xring" == locally (to one worker) reduce values '
                    'using NCCL then ring reduce across workers.\n'
                    '"pscpu:32k:xring" == use pscpu algorithm for tensors of '
                    'size up to 32kB, then xring for larger tensors.')

# If variable_update==distributed_all_reduce then it may be advantageous
# to aggregate small tensors into one prior to reduction.  These parameters
# control that aggregation.
#flags.DEFINE_integer('agg_small_grads_max_bytes', 0,
#                     'If > 0, try to aggregate tensors of less than this '
#                     'number of bytes prior to all-reduce.')
#flags.DEFINE_integer('agg_small_grads_max_group', 10,
#                     'When aggregating small tensors for all-reduce do not '
#                     'aggregate more than this many into one new tensor.')

# Distributed training parameters.
#flags.DEFINE_enum('job_name', '', ('ps', 'worker', 'controller', ''),
#                  'One of "ps", "worker", "controller", "".  Empty for local '
#                  'training')
flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of target hosts')
flags.DEFINE_string('worker_hosts', '', 'Comma-separated list of target hosts')
#flags.DEFINE_string('controller_host', None, 'optional controller host')
#flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
flags.DEFINE_string('server_protocol', 'grpc', 'protocol for servers')
#flags.DEFINE_boolean('cross_replica_sync', True, '')
#flags.DEFINE_string('horovod_device', '', 'Device to do Horovod all-reduce on: '
#                    'empty (default), cpu or gpu. Default with utilize GPU if '
#                    'Horovod was compiled with the HOROVOD_GPU_ALLREDUCE '
#                    'option, and CPU otherwise.')

# Summary and Save & load checkpoints.
flags.DEFINE_integer('summary_verbosity', 3, 'Verbosity level for summary ops. '
                     'level 0: disable any summary.\n'
                     'level 1: small and fast ops, e.g.: learning_rate, '
                     'total_loss.\n'
                     'level 2: medium-cost ops, e.g. histogram of all '
                     'gradients.\n'
                     'level 3: expensive ops: images and histogram of each '
                     'gradient.\n')
flags.DEFINE_integer('save_summaries_steps', 100,
                     'How often to save summaries for trained models. Pass 0 '
                     'to disable summaries.')
flags.DEFINE_integer('save_model_secs', 0,
                     'How often to save trained models. Pass 0 to disable '
                     'checkpoints.')
flags.DEFINE_string('train_dir', None,
                    'Path to session checkpoints. Pass None to disable saving '
                    'checkpoint at the end.')
flags.DEFINE_string('eval_dir', '/tmp/tf_cnn_benchmarks/eval',
                    'Directory where to write eval event logs.')
flags.DEFINE_string('result_storage', None,
                    'Specifies storage option for benchmark results. None '
                    'means results won\'t be stored. '
                    '`cbuild_benchmark_datastore` means results will be stored '
                    'in cbuild datastore (note: this option requires special '
                    'permissions and meant to be used from cbuilds).')

# Benchmark logging for model garden metric
flags.DEFINE_string('benchmark_log_dir', None,
                    'The directory to place the log files containing the '
                    'results of benchmark. The logs are created by '
                    'BenchmarkFileLogger. Requires the root of the Tensorflow '
                    ' models repository to be in $PYTHTONPATH.')

flags.DEFINE_string('networkDirectory', None,
                    'The directory where placed model file')
flags.DEFINE_string('network', None,
                    'The network file name')
flags.DEFINE_string('save', None,
                    'Path to session checkpoints. Pass None to disable saving '
                    'checkpoint at the end.')
flags.DEFINE_string('train_db', None,
                    'Path to train dataset in TFRecord format.')
flags.DEFINE_string('validation_db', None,
                    'Path to validation dataset in TFRecord format.')
flags.DEFINE_enum('optimization', 'sgd', ('momentum', 'sgd', 'rmsprop'),
                  'Optimizer to use: momentum or sgd or rmsprop')
flags.DEFINE_float('epoch', None,
                   'number of epochs to run, excluding warmup. '
                   'This and --num_batches cannot both be specified.')
flags.DEFINE_string('labels_list', None,
                   'list of labels file ')
flags.DEFINE_string('visualizeModelPath', None,
                    'Constructs the current model for visualization.')

platforms_util.define_platform_params()


class GlobalStepWatcher(threading.Thread):
  """A helper class for global_step.

  Polls for changes in the global_step of the model, and finishes when the
  number of steps for the global run are done.
  """

  def __init__(self, sess, global_step_op, start_at_global_step,
               end_at_global_step):
    threading.Thread.__init__(self)
    self.sess = sess
    self.global_step_op = global_step_op
    self.start_at_global_step = start_at_global_step
    self.end_at_global_step = end_at_global_step

    self.start_time = 0
    self.start_step = 0
    self.finish_time = 0
    self.finish_step = 0

  def run(self):
    while self.finish_time == 0:
      time.sleep(.25)
      global_step_val, = self.sess.run([self.global_step_op])
      if self.start_time == 0 and global_step_val >= self.start_at_global_step:
        # Use tf.logging.info instead of log_fn, since print (which is log_fn)
        # is not thread safe and may interleave the outputs from two parallel
        # calls to print, which can break tests.
        tf.logging.info('Starting real work at step %s at time %s' %
                        (global_step_val, time.ctime()))
        self.start_time = time.time()
        self.start_step = global_step_val
      if self.finish_time == 0 and global_step_val >= self.end_at_global_step:
        tf.logging.info('Finishing real work at step %s at time %s' %
                        (global_step_val, time.ctime()))
        self.finish_time = time.time()
        self.finish_step = global_step_val

  def done(self):
    return self.finish_time > 0

  def num_steps(self):
    return self.finish_step - self.start_step

  def elapsed_time(self):
    return self.finish_time - self.start_time


class CheckpointNotFoundException(Exception):
  pass


def get_data_type(params):
  """Returns BenchmarkCNN's data type as determined by use_fp16.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  """
  #return tf.float16 if params.use_fp16 else tf.float32
  return tf.float32


# Note that we monkey patch this function in the unit tests. So if this is
# inlined or renamed, the unit tests must be updated.
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


def create_config_proto(params):
  """Returns session config proto.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.intra_op_parallelism_threads = params.num_intra_threads
  config.inter_op_parallelism_threads = params.num_inter_threads
  config.gpu_options.force_gpu_compatible = params.force_gpu_compatible
  if params.allow_growth is not None:
    config.gpu_options.allow_growth = params.allow_growth
  if params.gpu_memory_frac_for_testing > 0:
    config.gpu_options.per_process_gpu_memory_fraction = (
        params.gpu_memory_frac_for_testing)
  if params.xla:
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  if params.enable_layout_optimizer:
    config.graph_options.rewrite_options.layout_optimizer = (
        rewriter_config_pb2.RewriterConfig.ON)
  if params.rewriter_config:
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    text_format.Merge(params.rewriter_config, rewriter_config)
    config.graph_options.rewrite_options.CopyFrom(rewriter_config)
  #if params.variable_update == 'horovod':
  #  import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
  #  config.gpu_options.visible_device_list = str(hvd.local_rank())

  return config


def get_mode_from_params(params):
  """Returns the mode in which this script is running.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  Raises:
    ValueError: Unsupported params settings.
  """
  #if params.forward_only and params.eval:
  #  raise ValueError('Only one of forward_only and eval parameters is true')

  if params.eval:
    return 'evaluation'
  #if params.forward_only:
  #  return 'forward-only'
  return 'training'


# How many digits to show for the loss and accuracies during training.
LOSS_AND_ACCURACY_DIGITS_TO_SHOW = 3


def benchmark_one_step(sess,
                       fetches,
                       step,
                       batch_size,
                       num_batches_per_epoch,
                       step_train_times,
                       trace_filename,
                       partitioned_graph_file_prefix,
                       profiler,
                       image_producer,
                       params,
                       summary_op=None,
                       show_images_per_sec=True,
                       benchmark_logger=None,
                       phase_train=True):
  """Advance one step of benchmarking."""
  should_profile = profiler and 0 <= step < _NUM_STEPS_TO_PROFILE
  need_options_and_metadata = (
      should_profile or
      #((trace_filename or partitioned_graph_file_prefix) and step == -2)
      (trace_filename and step == -2)
  )
  if need_options_and_metadata:
    run_options = tf.RunOptions()
    if (trace_filename and step == -2) or should_profile:
      run_options.trace_level = tf.RunOptions.FULL_TRACE
    #if partitioned_graph_file_prefix and step == -2:
    #  run_options.output_partition_graphs = True
    run_metadata = tf.RunMetadata()
  else:
    run_options = None
    run_metadata = None
  summary_str = None
  start_time = time.time()
  if summary_op is None:
    results = sess.run(fetches, options=run_options, run_metadata=run_metadata)
  else:
    (results, summary_str) = sess.run(
        [fetches, summary_op], options=run_options, run_metadata=run_metadata)

  #if not params.forward_only:
  #  lossval = results['average_loss']
  #else:
  #  lossval = 0.
  lossval = results['average_loss']
  if image_producer is not None:
    image_producer.notify_image_consumption()
  train_time = time.time() - start_time
  step_train_times.append(train_time)
  if (show_images_per_sec and step >= 0 and
      (step == 0 or (step + 1) % params.display_every == 0)):
    if phase_train:
      log_str = "Training"
    else:
      log_str = "Validation"
    log_str += ' (epoch %.*f): loss = %.*f' % (
        2, float(step + 1)/num_batches_per_epoch,
        LOSS_AND_ACCURACY_DIGITS_TO_SHOW, lossval)
    #log_str = '%i\t%s\t%.*f' % (
    #    step + 1, get_perf_timing_str(batch_size, step_train_times),
    #    LOSS_AND_ACCURACY_DIGITS_TO_SHOW, lossval)
    if 'top_1_accuracy' in results:
      #log_str += '\t%.*f\t%.*f' % (
      #    LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_1_accuracy'],
      #    LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_5_accuracy'])
      log_str += ', accuracy = %.*f' % (
          LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_1_accuracy'])
    if 'learning_rate' in results:
      log_str += ', lr = %.*f' % (5, results['learning_rate'])
    logging.info(log_str)
    if benchmark_logger:
      # TODO(scottzhu): This might impact the benchmark speed since it writes
      # the benchmark log to local directory.
      benchmark_logger.log_evaluation_result(results)
  if need_options_and_metadata:
    if should_profile:
      profiler.add_step(step, run_metadata)
    if trace_filename and step == -2:
      logging.info('Dumping trace to %s' % trace_filename)
      trace_dir = os.path.dirname(trace_filename)
      if not gfile.Exists(trace_dir):
        gfile.MakeDirs(trace_dir)
      with gfile.Open(trace_filename, 'w') as trace_file:
        if params.use_chrome_trace_format:
          trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
        else:
          trace_file.write(str(run_metadata.step_stats))
    #if partitioned_graph_file_prefix and step == -2:
    #  path, filename = os.path.split(partitioned_graph_file_prefix)
    #  if '.' in filename:
    #    base_filename, ext = filename.rsplit('.', 1)
    #    ext = '.' + ext
    #  else:
    #    base_filename, ext = filename, ''
    #  as_text = filename.endswith('txt')
    #  for graph_def in run_metadata.partition_graphs:
    #    device = graph_def.node[0].device.replace('/', '_').replace(':', '_')
    #    graph_filename = '%s%s%s' % (base_filename, device, ext)
    #    log_fn('Writing partitioned GraphDef as %s to %s' % (
    #        'text' if as_text else 'binary',
    #        os.path.join(path, graph_filename)))
    #    tf.train.write_graph(graph_def, path, graph_filename, as_text)
  return summary_str


def get_perf_timing_str(batch_size, step_train_times, scale=1):
  times = np.array(step_train_times)
  speeds = batch_size / times
  speed_mean = scale * batch_size / np.mean(times)
  if scale == 1:
    speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
    speed_madstd = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
    speed_jitter = speed_madstd
    return ('images/sec: %.1f +/- %.1f (jitter = %.1f)' %
            (speed_mean, speed_uncertainty, speed_jitter))
  else:
    return 'images/sec: %.1f' % speed_mean


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
    logging.info('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)
    return global_step
  else:
    raise CheckpointNotFoundException('No checkpoint file found.')


# Params are passed to BenchmarkCNN's constructor. Params is a map from name
# to value, with one field per key in flags.param_specs.
#
# Call make_params() or make_params_from_flags() below to construct a Params
# tuple with default values from flags.param_specs, rather than constructing
# Params directly.
Params = namedtuple('Params', 'key')  # pylint: disable=invalid-name


def validate_params(params):
  """Validates that the Params tuple had valid values.

  When command-line flags are defined for each ParamSpec by calling
  flags.define_flags(), calling this function is unnecessary because absl
  already does flag validation. Otherwise, this function should be called.

  Args:
     params: A Params tuple.
  Raises:
    ValueError: An element of params had an invalid value.
  """
  for name, value in params._asdict().items():
    param_spec = flags.param_specs[name]
    if param_spec.flag_type in ('integer', 'float'):
      if (param_spec.kwargs['lower_bound'] is not None and
          value < param_spec.kwargs['lower_bound']):
        raise ValueError('Param %s value of %s is lower than the lower bound '
                         'of %s' %
                         (name, value, param_spec.kwargs['lower_bound']))
      if (param_spec.kwargs['upper_bound'] is not None and
          param_spec.kwargs['upper_bound'] < value):
        raise ValueError('Param %s value of %s is higher than the upper bound '
                         'of %s' %
                         (name, value, param_spec.kwargs['upper_bound']))
    elif (param_spec.flag_type == 'enum' and
          value not in param_spec.kwargs['enum_values']):
      raise ValueError('Param %s of value %s is not in %s'%
                       (name, value, param_spec.kwargs['enum_values']))


def make_params(**kwargs):
  """Create a Params tuple for BenchmarkCNN from kwargs.

  Default values are filled in from flags.param_specs.

  Args:
    **kwargs: kwarg values will override the default values.
  Returns:
    Params namedtuple for constructing BenchmarkCNN.
  """
  # Create a (name: default_value) map from flags.param_specs.
  default_kwargs = {
      name: flags.param_specs[name].default_value
      for name in flags.param_specs
  }
  params = Params(**default_kwargs)._replace(**kwargs)
  validate_params(params)
  return params


def make_params_from_flags():
  """Create a Params tuple for BenchmarkCNN from absl_flags.FLAGS.

  Returns:
    Params namedtuple for constructing BenchmarkCNN.
  """
  # Collect (name: value) pairs for absl_flags.FLAGS with matching names in
  # flags.param_specs.
  #flag_values = {name: getattr(flags.FLAGS, name)
  #               for name in flags.param_specs.keys()}
  FLAGS._parse_flags()
  flag_values = FLAGS.__flags
  return Params(**flag_values)


def get_num_batches_and_epochs(params, batch_size, num_examples_per_epoch):
  """Returns the number of batches and epochs to run for.

  Args:
    params: Params tuple, typically created by make_params or
      make_params_from_flags.
    batch_size: The number of images per step.
    num_examples_per_epoch: The number of images in a single epoch.

  Returns:
    num_batches: The number of batches to run for.
    num_epochs: The number of epochs to run for. This might be slightly
      smaller than params.num_epochs if specified, because the number of batches
      must be an integer.

  Raises:
    ValueError: Invalid or unsupported params.
  """
  if params.num_batches and params.num_epochs:
    raise ValueError('At most one of --num_batches and --num_epochs may be '
                     'specified.')
  if params.num_epochs:
    num_batches = int(float(params.num_epochs) * num_examples_per_epoch /
                      batch_size)
  else:
    num_batches = params.num_batches or _DEFAULT_NUM_BATCHES
  num_epochs = num_batches * batch_size / float(num_examples_per_epoch)
  return (num_batches, num_epochs)


def get_piecewise_learning_rate(piecewise_learning_rate_schedule,
                                global_step, num_batches_per_epoch):
  """Returns a piecewise learning rate tensor.

  Args:
    piecewise_learning_rate_schedule: The --piecewise_learning_rate_schedule
      parameter
    global_step: Scalar tensor representing the global step.
    num_batches_per_epoch: float indicating the number of batches per epoch.

  Returns:
    A scalar float tensor, representing the learning rate.

  Raises:
    ValueError: piecewise_learning_rate_schedule is not formatted correctly.
  """
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


def get_learning_rate(params, global_step, num_examples_per_epoch, model,
                      batch_size):
  """Returns a learning rate tensor based on global_step.

  Args:
    params: Params tuple, typically created by make_params or
      make_params_from_flags.
    global_step: Scalar tensor representing the global step.
    num_examples_per_epoch: The number of examples per epoch.
    model: The model.Model object to obtain the default learning rate from if no
      learning rate is specified.
    batch_size: Number of examples per step

  Returns:
    A scalar float tensor, representing the learning rate. When evaluated, the
    learning rate depends on the current value of global_step.

  Raises:
    ValueError: Invalid or unsupported params.
  """
  num_batches_per_epoch = (float(num_examples_per_epoch) / batch_size)

  if params.piecewise_learning_rate_schedule:
    if (params.init_learning_rate or params.learning_rate_decay_factor or
        params.minimum_learning_rate or params.num_epochs_per_decay):
      raise ValueError('No other learning rate-related flags can be specified '
                       'if --piecewise_learning_rate_schedule is specified')
    learning_rate = get_piecewise_learning_rate(
        params.piecewise_learning_rate_schedule,
        global_step, num_batches_per_epoch)
  elif params.init_learning_rate:
    learning_rate = params.init_learning_rate
    if (params.num_epochs_per_decay > 0 and
        params.learning_rate_decay_factor > 0):
      decay_steps = int(num_batches_per_epoch * params.num_epochs_per_decay)

      # Decay the learning rate exponentially based on the number of steps.
      learning_rate = tf.train.exponential_decay(
          params.init_learning_rate,
          global_step,
          decay_steps,
          params.learning_rate_decay_factor,
          staircase=True)

      if params.minimum_learning_rate != 0.:
        learning_rate = tf.maximum(learning_rate,
                                   params.minimum_learning_rate)
  else:
    learning_rate = model.get_learning_rate(global_step, batch_size)
  if params.num_learning_rate_warmup_epochs > 0 and (
      params.init_learning_rate or params.piecewise_learning_rate_schedule):
    warmup_steps = int(num_batches_per_epoch *
                       params.num_learning_rate_warmup_epochs)
    init_lr = (params.init_learning_rate or
               float(params.piecewise_learning_rate_schedule.split(';')[0]))
    warmup_lr = init_lr * tf.cast(global_step, tf.float32) / tf.cast(
        warmup_steps, tf.float32)
    learning_rate = tf.cond(global_step < warmup_steps,
                            lambda: warmup_lr, lambda: learning_rate)

  return learning_rate


def get_optimizer(params, learning_rate):
  """Returns the optimizer that should be used based on params."""
  if params.optimizer == 'momentum':
    opt = tf.train.MomentumOptimizer(
        learning_rate, params.momentum, use_nesterov=True)
  elif params.optimizer == 'sgd':
    opt = tf.train.GradientDescentOptimizer(learning_rate)
  elif params.optimizer == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(
        learning_rate,
        params.rmsprop_decay,
        momentum=params.rmsprop_momentum,
        epsilon=params.rmsprop_epsilon)
  else:
    raise ValueError('Optimizer "%s" was not recognized',
                     params.optimizer)
  return opt


def generate_tfprof_profile(profiler, tfprof_file):
  """Generates a tfprof profile, writing it to a file and printing top ops.

  Args:
    profiler: A tf.profiler.Profiler. `profiler.add_step` must have already been
      called.
    tfprof_file: The filename to write the ProfileProto to.
  """
  profile_proto = profiler.serialize_to_string()
  logging.info('Dumping ProfileProto to %s' % tfprof_file)
  with gfile.Open(tfprof_file, 'wb') as f:
    f.write(profile_proto)

  # Print out the execution times of the top operations. Note this
  # information can also be obtained with the dumped ProfileProto, but
  # printing it means tfprof doesn't have to be used if all the user wants
  # is the top ops.
  options = tf.profiler.ProfileOptionBuilder.time_and_memory()
  options['max_depth'] = _NUM_OPS_TO_PRINT
  options['order_by'] = 'accelerator_micros'
  profiler.profile_operations(options)


def strip_data_from_graph_def(graph_def):
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            if (tensor.tensor_content):
                tensor.tensor_content = ''
            if (tensor.string_val):
                del tensor.string_val[:]
    return strip_def


def visualize_graph(graph_def, path):
    graph_def = strip_data_from_graph_def(graph_def)
    logging.info('Writing Graph Definition..')
    file_io.write_string_to_file(path, str(graph_def))
    logging.info('Graph Definition Written.')


class BenchmarkCNN(object):
  """Class for benchmarking a cnn network."""

  def __init__(self, params):
    """Initialize BenchmarkCNN.

    Args:
      params: Params tuple, typically created by make_params or
              make_params_from_flags.
      dataset: If not None, the dataset to use. Otherwise, params is used to
               obtain the dataset.
      model: If not None, the model to use. Otherwise, params is used to obtain
             the model.
    Raises:
      ValueError: Unsupported params settings.
    """
    self.params = params
    # Cifar10Data or ImagenetData
    self.dataset = datasets.create_dataset(self.params.data_dir,
                                           self.params.data_name,
                                           self.params.train_db,
                                           self.params.validation_db,
                                           self.params.labels_list)
    logging.info("total_train: %s, total_val: %s" % (self.dataset.total_train,
                                                            self.dataset.total_val))

    self.model = model_config.get_model_config(self.params.model,
                                               self.dataset,
                                               self.params.networkDirectory,
                                               self.params.network)
    self.trace_filename = self.params.trace_file
    self.data_format = self.params.data_format
    self.enable_layout_optimizer = self.params.enable_layout_optimizer
    self.rewriter_config = self.params.rewriter_config
    self.num_warmup_batches = 10
    self.resize_method = self.params.resize_method
    self.sync_queue_counter = 0
    self.num_gpus = self.params.num_gpus
    self.gpu_indices = [x for x in range(self.num_gpus)]
    self.use_synthetic_gpu_images = self.dataset.use_synthetic_gpu_images()

    if (self.params.device == 'cpu' and self.params.data_format == 'NCHW' and
        not self.params.mkl):
      raise ValueError('device=cpu requires that data_format=NHWC')

    if ((self.params.num_epochs_per_decay or
         self.params.learning_rate_decay_factor) and
        not (self.params.init_learning_rate and self.params.num_epochs_per_decay
             and self.params.learning_rate_decay_factor)):
      raise ValueError('If one of num_epochs_per_decay or '
                       'learning_rate_decay_factor is set, both must be set'
                       'and learning_rate must be set')
    if (self.params.minimum_learning_rate and
        not (self.params.init_learning_rate and self.params.num_epochs_per_decay
             and self.params.learning_rate_decay_factor)):
      raise ValueError('minimum_learning_rate requires learning_rate,'
                       'num_epochs_per_decay, and '
                       'learning_rate_decay_factor to be set')

    if (self.params.debugger is not None and self.params.debugger != 'cli' and
        ':' not in self.params.debugger):
      raise ValueError('--debugger must be "cli" or in the form '
                       'host:port')

    if self.params.hierarchical_copy and self.params.num_gpus <= 1:
      raise ValueError('--hierarchical_copy requires --num_gpus to be greater '
                       'than 1')

    # Use the batch size from the command line if specified, otherwise use the
    # model's default batch size.  Scale the benchmark's batch size by the
    # number of GPUs.
    if self.params.batch_size > 0:
      self.model.set_batch_size(self.params.batch_size)
    self.batch_size = self.model.get_batch_size() * self.num_gpus

    self.local_parameter_device_flag = self.params.local_parameter_device

    self.task_index = 0
    self.param_server_device = '/%s:0' % self.params.local_parameter_device
    self.sync_queue_devices = [self.param_server_device]

    self.num_ps = 0

    # Device to use for ops that need to always run on the local worker's CPU.
    self.cpu_device = '/cpu:0'

    # Device to use for ops that need to always run on the local worker's
    # compute device, and never on a parameter server device.
    self.raw_devices = [
        '/%s:%i' % (self.params.device, i)
        for i in xrange(self.num_gpus)
    ]

    subset = 'validation' if params.eval else 'train'
    self.num_batches, self.num_epochs = get_num_batches_and_epochs(
        params, self.batch_size,
        self.dataset.num_examples_per_epoch(subset))
    self.num_batches_per_epoch = self.num_batches // self.num_epochs

    # adjust display_every number
    if self.params.display_every > 0:
      display = self.num_batches // self.num_epochs // 10
      logging.info("adjust display_every: %d" % display)
      self.params.display_every = display

    if self.params.variable_update == 'parameter_server':
      self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
    elif self.params.variable_update == 'replicated':
      self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
          self, self.params.all_reduce_spec,
          #self.params.agg_small_grads_max_bytes,
          0,
          #self.params.agg_small_grads_max_group)
          0)
    else:
      raise ValueError(
          'Invalid variable_update: %s' % self.params.variable_update)

    # Device to use for running on the local worker's compute device, but
    # with variables assigned to parameter server devices.
    self.devices = self.variable_mgr.get_devices()
    # self.raw_devices = [
    self.global_step_device = self.cpu_device

    self.image_preprocessor = self.get_image_preprocessor()
    self.datasets_use_prefetch = (
        self.params.datasets_use_prefetch and
        self.image_preprocessor.supports_datasets())
    self.init_global_step = 0

    self._config_benchmark_logger()

  def _config_benchmark_logger(self):
    """Config the model garden benchmark logger."""
    model_benchmark_logger = None
    if self.params.benchmark_log_dir is not None:
      try:
        from official.utils.logs import logger as models_logger  # pylint: disable=g-import-not-at-top
      except ImportError:
        tf.logging.fatal('Please include tensorflow/models to the PYTHONPATH '
                         'in order to use BenchmarkLogger. Configured '
                         'benchmark_log_dir: %s'
                         % self.params.benchmark_log_dir)
        raise
      model_benchmark_logger = models_logger.BenchmarkFileLogger(
          self.params.benchmark_log_dir)
    self.benchmark_logger = model_benchmark_logger

  def print_info(self):
    """Print basic information."""
    benchmark_info = self._get_params_info()
    logging.info('Model:       %s' % self.model.get_model())
    logging.info('Dataset:     %s' % benchmark_info['dataset_name'])
    logging.info('Mode:        %s' % get_mode_from_params(self.params))
    logging.info('SingleSess:  %s' % benchmark_info['single_session'])
    logging.info('Batch size:  %s global' % (self.batch_size))
    logging.info('             %s per device' % (self.batch_size /
                                           len(self.raw_devices)))
    logging.info('Num batches: %d' % self.num_batches)
    logging.info('Num epochs:  %.2f' % self.num_epochs)
    logging.info('Devices:     %s' % benchmark_info['device_list'])
    logging.info('Data format: %s' % self.data_format)
    logging.info('Layout optimizer: %s' % self.enable_layout_optimizer)
    if self.rewriter_config:
      logging.info('RewriterConfig: %s' % self.rewriter_config)
    logging.info('Optimizer:   %s' % self.params.optimizer)
    logging.info('Variables:   %s' % self.params.variable_update)
    if (self.params.variable_update == 'replicated'):
      logging.info('AllReduce:   %s' % self.params.all_reduce_spec)
    logging.info('==========')

  def _get_params_info(self):
    """Get the common parameters info for the benchmark run.

    Returns:
      A dict of processed parameters.
    """
    dataset_name = self.dataset.name
    if self.dataset.use_synthetic_gpu_images():
      dataset_name += ' (synthetic)'

    single_session = False

    device_list = self.raw_devices
    return {
        'dataset_name': dataset_name,
        'single_session': single_session,
        'device_list': device_list,}

  def _log_benchmark_run(self):
    """Log the benchmark info to the logger.

    The info logged here should be similar to print_info(), but in a structured
    JSON format.
    """
    if self.benchmark_logger:
      benchmark_info = self._get_params_info()

      run_param = {
          'model': self.model.get_model(),
          'dataset': benchmark_info['dataset_name'],
          'mode': get_mode_from_params(self.params),
          'single_sess': benchmark_info['single_session'],
          'devices': benchmark_info['device_list'],
          'batch_size': self.batch_size,
          'batch_size_per_device': self.batch_size / len(self.raw_devices),
          'num_batches': self.num_batches,
          'num_epochs': self.num_epochs,
          'data_format': self.data_format,
          'layout_optimizer': self.enable_layout_optimizer,
          'rewrite_config': self.rewriter_config,
          'optimizer': self.params.optimizer,
      }
      # TODO(scottzhu): tf_cnn_benchmark might execute several times with
      # different param setting on the same box. This will cause the run file to
      # only contain the latest info. The benchmark_log_dir should be updated
      # for every new run.
      self.benchmark_logger.log_run_info(
          self.model.get_model(), benchmark_info['dataset_name'], run_param)

  def run(self):
    """Run the benchmark task assigned to this process.

    Returns:
      Dictionary of statistics for training or eval.
    Raises:
       ValueError: unrecognized job name.
    """
    with tf.Graph().as_default():
      self._log_benchmark_run()
      if self.params.eval:
        return self._eval_cnn()
      else:
        return self._benchmark_cnn()

  def _eval_cnn(self):
    """Evaluate a model every self.params.eval_interval_secs.

    Returns:
      Dictionary containing eval statistics. Currently returns an empty
      dictionary.
    """
    if self.datasets_use_prefetch:
      (image_producer_ops, enqueue_ops, fetches) = (
          self._build_model_with_dataset_prefetching())
    else:
      #(image_producer_ops, enqueue_ops, fetches) = self._build_model()
      pass
    saver = tf.train.Saver(self.variable_mgr.savable_variables())
    summary_writer = tf.summary.FileWriter(self.params.eval_dir,
                                           tf.get_default_graph())
    target = ''
    local_var_init_op = tf.local_variables_initializer()
    table_init_ops = tf.tables_initializer()
    variable_mgr_init_ops = [local_var_init_op]
    if table_init_ops:
      variable_mgr_init_ops.extend([table_init_ops])
    with tf.control_dependencies([local_var_init_op]):
      variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
    local_var_init_op_group = tf.group(*variable_mgr_init_ops)
    summary_op = tf.summary.merge_all()
    # TODO(huangyp): Check if checkpoints haven't updated for hours and abort.
    while True:
      self._eval_once(saver, summary_writer, target, local_var_init_op_group,
                      image_producer_ops, enqueue_ops, fetches, summary_op)
      if self.params.eval_interval_secs <= 0:
        break
      time.sleep(self.params.eval_interval_secs)
    return {}

  def _eval_once(self, saver, summary_writer, target, local_var_init_op_group,
                 image_producer_ops, enqueue_ops, fetches, summary_op):
    """Evaluate the model from a checkpoint using validation dataset."""
    with tf.Session(
        target=target, config=create_config_proto(self.params)) as sess:
      if self.params.train_dir is None:
        raise ValueError('Trained model directory not specified')
      try:
        global_step = load_checkpoint(saver, sess, self.params.train_dir)
      except CheckpointNotFoundException:
        logging.info('Checkpoint not found in %s' % self.params.train_dir)
        return
      sess.run(local_var_init_op_group)
      if self.dataset.queue_runner_required():
        tf.train.start_queue_runners(sess=sess)
      image_producer = None
      if image_producer_ops is not None:
        image_producer = cnn_util.ImageProducer(
            #sess, image_producer_ops, self.batch_group_size,
            sess, image_producer_ops, 1,
            #self.params.use_python32_barrier)
            False)
        image_producer.start()
        for i in xrange(len(enqueue_ops)):
          sess.run(enqueue_ops[:(i + 1)])
          image_producer.notify_image_consumption()
      loop_start_time = start_time = time.time()
      top_1_accuracy_sum = 0.0
      top_5_accuracy_sum = 0.0
      total_eval_count = self.num_batches * self.batch_size
      for step in xrange(self.num_batches):
        if (self.params.save_summaries_steps > 0 and
            (step + 1) % self.params.save_summaries_steps == 0):
          results, summary_str = sess.run([fetches, summary_op])
          summary_writer.add_summary(summary_str)
        else:
          results = sess.run(fetches)
        top_1_accuracy_sum += results['top_1_accuracy']
        top_5_accuracy_sum += results['top_5_accuracy']
        if (step + 1) % self.params.display_every == 0:
          duration = time.time() - start_time
          examples_per_sec = (
              self.batch_size * self.params.display_every / duration)
          logging.info('%i\t%.1f examples/sec' % (step + 1, examples_per_sec))
          start_time = time.time()
        if image_producer is not None:
          image_producer.notify_image_consumption()
      loop_end_time = time.time()
      if image_producer is not None:
        image_producer.done()
      accuracy_at_1 = top_1_accuracy_sum / self.num_batches
      accuracy_at_5 = top_5_accuracy_sum / self.num_batches
      summary = tf.Summary()
      summary.value.add(tag='eval/Accuracy@1', simple_value=accuracy_at_1)
      summary.value.add(tag='eval/Accuracy@5', simple_value=accuracy_at_5)
      summary_writer.add_summary(summary, global_step)
      logging.info('Accuracy @ 1 = %.4f Accuracy @ 5 = %.4f [%d examples]' %
             (accuracy_at_1, accuracy_at_5, total_eval_count))
      elapsed_time = loop_end_time - loop_start_time
      images_per_sec = (self.num_batches * self.batch_size / elapsed_time)
      # Note that we compute the top 1 accuracy and top 5 accuracy for each
      # batch, which will have a slight performance impact.
      logging.info('-' * 64)
      logging.info('total images/sec: %.2f' % images_per_sec)
      logging.info('-' * 64)
      if self.benchmark_logger:
        eval_result = {
            'eval_top_1_accuracy', accuracy_at_1,
            'eval_top_5_accuracy', accuracy_at_5,
            'eval_average_examples_per_sec', images_per_sec,
            tf.GraphKeys.GLOBAL_STEP, global_step,
        }
        self.benchmark_logger.log_evaluation_result(eval_result)

  def _benchmark_cnn(self):
    """Run cnn in benchmark mode. Skip the backward pass if forward_only is on.

    Returns:
      Dictionary containing training statistics (num_workers, num_steps,
      average_wall_time, images_per_sec).
    """
    self.single_session = False
    if self.datasets_use_prefetch:
      (image_producer_ops, enqueue_ops, fetches) = (
          self._build_model_with_dataset_prefetching())
    else:
      pass
      # (image_producer_ops, enqueue_ops, fetches) = self._build_model()
    fetches_list = nest.flatten(list(fetches.values()))
    main_fetch_group = tf.group(*fetches_list)

    global_step = tf.train.get_global_step()
    with tf.device(self.global_step_device):
      with tf.control_dependencies([main_fetch_group]):
        fetches['inc_global_step'] = global_step.assign_add(1)

    local_var_init_op = tf.local_variables_initializer()
    table_init_ops = tf.tables_initializer()
    variable_mgr_init_ops = [local_var_init_op]
    if table_init_ops:
      variable_mgr_init_ops.extend([table_init_ops])
    with tf.control_dependencies([local_var_init_op]):
      variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())

    local_var_init_op_group = tf.group(*variable_mgr_init_ops)

    summary_op = tf.summary.merge_all()
    summary_writer = None
    if (self.params.summary_verbosity and self.params.train_dir and
        self.params.save_summaries_steps > 0):
      summary_writer = tf.summary.FileWriter(self.params.train_dir,
                                             tf.get_default_graph())

    saver = tf.train.Saver(
        self.variable_mgr.savable_variables(), save_relative_paths=True)

    sv = tf.train.Supervisor(
        # For the purpose of Supervisor, all Horovod workers are 'chiefs',
        # since we want session to be initialized symmetrically on all the
        # workers.
        is_chief=True,
        # Log dir should be unset on non-chief workers to prevent Horovod
        # workers from corrupting each other's checkpoints.
        logdir=self.params.train_dir,
        ready_for_local_init_op=None,
        local_init_op=local_var_init_op_group,
        saver=saver,
        global_step=global_step,
        summary_op=None,
        save_model_secs=self.params.save_model_secs,
        summary_writer=summary_writer)

    step_train_times = []
    start_standard_services = (
        self.params.summary_verbosity >= 1 or
        self.dataset.queue_runner_required())
    target = ''
    with sv.managed_session(
        master=target,
        config=create_config_proto(self.params),
        start_standard_services=start_standard_services) as sess:

      if self.params.visualizeModelPath:
        visualize_graph(sess.graph_def, self.params.visualizeModelPath)
        exit(0)

      image_producer = None
      if image_producer_ops is not None:
        image_producer = cnn_util.ImageProducer(
            sess, image_producer_ops, 1, False)
            #sess, image_producer_ops, self.batch_group_size,
            #self.params.use_python32_barrier)
        image_producer.start()
        for i in xrange(len(enqueue_ops)):
          sess.run(enqueue_ops[:(i + 1)])
          image_producer.notify_image_consumption()
      self.init_global_step, = sess.run([global_step])
      #if self.job_name and not self.params.cross_replica_sync:

      logging.info('Running warm up')
      local_step = -1 * self.num_warmup_batches

      #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6064")
      if self.params.debugger is not None:
        if self.params.debugger == 'cli':
          logging.info('The CLI TensorFlow debugger will be used.')
          sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        else:
          logging.info('The TensorBoard debugger plugin will be used.')
          sess = tf_debug.TensorBoardDebugWrapperSession(sess,
                                                         self.params.debugger)
      profiler = tf.profiler.Profiler() if self.params.tfprof_file else None
      loop_start_time = time.time()
      while not (local_step == self.num_batches):
        if local_step == 0:
          logging.info('Done warm up')
          #if execution_barrier:
          #  logging.info('Waiting for other replicas to finish warm up')
          #  sess.run([execution_barrier])

          #header_str = ('Step\tImg/sec\t' +
          #              self.params.loss_type_to_report.replace('/', ' '))
          #if self.params.print_training_accuracy or self.params.forward_only:
          #if self.params.print_training_accuracy:
          #  header_str += '\ttop_1_accuracy\ttop_5_accuracy'
          #logging.info(header_str)
          assert len(step_train_times) == self.num_warmup_batches
          # reset times to ignore warm up batch
          step_train_times = []
          loop_start_time = time.time()
        if (summary_writer and
            (local_step + 1) % self.params.save_summaries_steps == 0):
          fetch_summary = summary_op
        else:
          fetch_summary = None
        summary_str = benchmark_one_step(
            sess, fetches, local_step,
            self.batch_size,
            self.num_batches_per_epoch,
            step_train_times,
            #self.trace_filename, self.params.partitioned_graph_file_prefix,
            self.trace_filename, None,
            profiler, image_producer, self.params, fetch_summary,
            benchmark_logger=self.benchmark_logger,
            phase_train = not (self.params.eval))
        if summary_str is not None:
          sv.summary_computed(sess, summary_str)
        local_step += 1
      loop_end_time = time.time()

      elapsed_time = loop_end_time - loop_start_time
      average_wall_time = elapsed_time / local_step if local_step > 0 else 0
      images_per_sec = (local_step * self.batch_size /
                        elapsed_time)
      num_steps = local_step

      logging.info('-' * 64)
      logging.info('total images/sec: %.2f' % images_per_sec)
      logging.info('-' * 64)
      if image_producer is not None:
        image_producer.done()

      store_benchmarks({'total_images_per_sec': images_per_sec}, self.params)
      if self.benchmark_logger:
        self.benchmark_logger.log_metric(
            'average_examples_per_sec', images_per_sec, global_step=num_steps)

      # Save the model checkpoint.
      if self.params.train_dir is not None:
        checkpoint_path = os.path.join(self.params.train_dir, 'model.ckpt')
        if not gfile.Exists(self.params.train_dir):
          gfile.MakeDirs(self.params.train_dir)
        sv.saver.save(sess, checkpoint_path, global_step)

      #if execution_barrier:
      #  # Wait for other workers to reach the end, so this worker doesn't
      #  # go away underneath them.
      #  sess.run([execution_barrier])
    sv.stop()
    if profiler:
      generate_tfprof_profile(profiler, self.params.tfprof_file)
    return {
        'num_workers': 1,
        'num_steps': num_steps,
        'average_wall_time': average_wall_time,
        'images_per_sec': images_per_sec
    }

  def _build_image_processing(self, shift_ratio=0):
    """"Build the image (pre)processing portion of the model graph."""
    with tf.device(self.cpu_device):
      if self.params.eval:
        subset = 'validation'
      else:
        subset = 'train'
      image_producer_ops = []
      image_producer_stages = []
      images_splits, labels_splits = self.image_preprocessor.minibatch(
          self.dataset,
          subset=subset,
          use_datasets=self.params.use_datasets,
          cache_data=self.params.cache_data,
          shift_ratio=shift_ratio)
      images_shape = images_splits[0].get_shape()
      labels_shape = labels_splits[0].get_shape()
      for device_num in range(len(self.devices)):
        image_producer_stages.append(
            data_flow_ops.StagingArea(
                [images_splits[0].dtype, labels_splits[0].dtype],
                shapes=[images_shape, labels_shape]))
        #for group_index in xrange(self.batch_group_size):
        for group_index in xrange(1):
          if not self.use_synthetic_gpu_images:
            #batch_index = group_index + device_num * self.batch_group_size
            batch_index = group_index + device_num
            put_op = image_producer_stages[device_num].put(
                [images_splits[batch_index], labels_splits[batch_index]])
            image_producer_ops.append(put_op)
    return (image_producer_ops, image_producer_stages)


  # TODO(rohanj): Refactor this function and share with other code path.
  def _build_model_with_dataset_prefetching(self):
    """Build the TensorFlow graph using datasets prefetching."""
    #assert not self.params.staged_vars
    assert not self.variable_mgr.supports_staged_vars()

    tf.set_random_seed(self.params.tf_random_seed)
    np.random.seed(4321)
    phase_train = not (self.params.eval)

    logging.info('Generating model')
    losses = []
    device_grads = []
    all_logits = []
    all_top_1_ops = []
    all_top_5_ops = []

    with tf.device(self.global_step_device):
      global_step = tf.train.get_or_create_global_step()

    with tf.name_scope("Data"):
      # Build the processing and model for the worker.
      function_buffering_resources = data_utils.build_prefetch_image_processing(
          self.model.get_image_size(), self.model.get_image_size(),
          self.batch_size, len(
              self.devices), self.image_preprocessor.parse_and_preprocess,
          self.cpu_device, self.params, self.devices, self.dataset)

    update_ops = None

    for device_num in range(len(self.devices)):
      with self.variable_mgr.create_outer_variable_scope(
          device_num), tf.name_scope('tower_%i' % device_num) as name_scope:
        function_buffering_resource = function_buffering_resources[device_num]
        results = self.add_forward_pass_and_gradients(
            phase_train, device_num, device_num, None, None, None,
            function_buffering_resource)
        if phase_train:
          losses.append(results['loss'])
          device_grads.append(results['gradvars'])
        else:
          all_logits.append(results['logits'])
        if not phase_train or self.params.print_training_accuracy:
          all_top_1_ops.append(results['top_1_op'])
          all_top_5_ops.append(results['top_5_op'])

        if device_num == 0:
          # Retain the Batch Normalization updates operations only from the
          # first tower. These operations update the moving mean and moving
          # variance variables, which are updated (but not used) during
          # training, and used during evaluation. The moving mean and variance
          # approximate the true mean and variance across all images in the
          # dataset. Therefore, in replicated mode, these moving averages would
          # be almost identical for each tower, and so we only update and save
          # the moving averages for one tower. In parameter server mode, all
          # towers share a copy of the variables so we also only need to update
          # and save the moving averages once.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
          assert not self.variable_mgr.staging_delta_ops

    with tf.name_scope('Loss'):
      fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
                                    None, update_ops, all_top_1_ops,
                                    all_top_5_ops, phase_train)
    return (None, [], fetches)

  def _build_fetches(self, global_step, all_logits, losses, device_grads,
                     enqueue_ops, update_ops, all_top_1_ops, all_top_5_ops,
                     phase_train):
    """Complete construction of model graph, populating the fetches map."""
    fetches = {}
    if enqueue_ops:
      fetches['enqueue_ops'] = enqueue_ops
    if all_top_1_ops:
      fetches['top_1_accuracy'] = tf.reduce_sum(all_top_1_ops) / self.batch_size
      if self.task_index == 0 and self.params.summary_verbosity >= 1:
        tf.summary.scalar('top_1_accuracy', fetches['top_1_accuracy'])
    if all_top_5_ops:
      fetches['top_5_accuracy'] = tf.reduce_sum(all_top_5_ops) / self.batch_size
      if self.task_index == 0 and self.params.summary_verbosity >= 1:
        tf.summary.scalar('top_5_accuracy', fetches['top_5_accuracy'])

    if not phase_train:
      #if self.params.forward_only:
      fetches['all_logits'] = tf.concat(all_logits, 0)
      return fetches
    apply_gradient_devices, gradient_state = (
        self.variable_mgr.preprocess_device_grads(device_grads))

    training_ops = []
    for d, device in enumerate(apply_gradient_devices):
      with tf.device(device):
        average_loss = tf.reduce_mean(losses)
        avg_grads = self.variable_mgr.get_gradients_to_apply(d, gradient_state)

        gradient_clip = self.params.gradient_clip
        learning_rate = get_learning_rate(self.params, global_step,
                                          self.dataset.num_examples_per_epoch(),
                                          self.model, self.batch_size)

        if gradient_clip is not None:
          clipped_grads = [(tf.clip_by_value(grad, -gradient_clip,
                                             +gradient_clip), var)
                           for grad, var in avg_grads]
        else:
          clipped_grads = avg_grads

        learning_rate = tf.identity(learning_rate, name='learning_rate')
        opt = get_optimizer(self.params, learning_rate)
        loss_scale_params = variable_mgr_util.AutoLossScaleParams(
            enable_auto_loss_scale=False,
            loss_scale=None,
            loss_scale_normal_steps=None,
            inc_loss_scale_every_n=1000,
            is_chief=True)

        self.variable_mgr.append_apply_gradients_ops(
            gradient_state, opt, clipped_grads, training_ops, loss_scale_params)
    train_op = tf.group(*(training_ops + update_ops))

    with tf.device(self.cpu_device):
      if self.task_index == 0 and self.params.summary_verbosity >= 1:
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar(self.params.loss_type_to_report, average_loss)

        if self.params.summary_verbosity >= 2:
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

        if self.params.summary_verbosity >= 3:
          for grad, var in avg_grads:
            if grad is not None:
              tf.summary.histogram(var.op.name + '/gradients', grad)
          for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    fetches['learning_rate'] = learning_rate
    fetches['train_op'] = train_op
    fetches['average_loss'] = average_loss
    return fetches


  def add_forward_pass_and_gradients(self,
                                     phase_train,
                                     rel_device_num,
                                     abs_device_num,
                                     image_producer_stage,
                                     gpu_compute_stage_ops,
                                     gpu_grad_stage_ops,
                                     function_buffering_resource=None):
    """Add ops for forward-pass and gradient computations."""
    nclass = self.dataset.num_classes
    data_type = get_data_type(self.params)
    image_size = self.model.get_image_size()
    if self.datasets_use_prefetch and function_buffering_resource is not None:
      with tf.device(self.raw_devices[rel_device_num]):
        images, labels = data_utils.get_images_and_labels(
            function_buffering_resource, data_type)
        images = tf.reshape(
            images,
            shape=[
                self.batch_size // self.num_gpus, image_size, image_size,
                self.dataset.depth
            ])
    else:
      pass
      #if not self.use_synthetic_gpu_images:
      #  with tf.device(self.cpu_device):
      #    host_images, host_labels = image_producer_stage.get()
      #    images_shape = host_images.get_shape()
      #    labels_shape = host_labels.get_shape()
      #with tf.device(self.raw_devices[rel_device_num]):
      #  if not self.use_synthetic_gpu_images:
      #    gpu_compute_stage = data_flow_ops.StagingArea(
      #        [host_images.dtype, host_labels.dtype],
      #        shapes=[images_shape, labels_shape])
      #    # The CPU-to-GPU copy is triggered here.
      #    gpu_compute_stage_op = gpu_compute_stage.put(
      #        [host_images, host_labels])
      #    images, labels = gpu_compute_stage.get()
      #    images = tf.reshape(images, shape=images_shape)
      #    gpu_compute_stage_ops.append(gpu_compute_stage_op)
      #  else:
      #    # Minor hack to avoid H2D copy when using synthetic data
      #    image_shape = [
      #        self.batch_size // self.num_gpus, image_size, image_size,
      #        self.dataset.depth
      #    ]
      #    labels_shape = [self.batch_size // self.num_gpus]
      #    # Synthetic image should be within [0, 255].
      #    images = tf.truncated_normal(
      #        image_shape,
      #        dtype=data_type,
      #        mean=127,
      #        stddev=60,
      #        name='synthetic_images')
      #    images = tf.contrib.framework.local_variable(
      #        images, name='gpu_cached_images')
      #    labels = tf.random_uniform(
      #        labels_shape,
      #        minval=0,
      #        maxval=nclass - 1,
      #        dtype=tf.int32,
      #        name='synthetic_labels')

    with tf.device(self.devices[rel_device_num]):
      logits, aux_logits = self.model.build_network(
          images, phase_train, nclass, self.dataset.depth, data_type,
          #self.data_format, self.params.use_tf_layers, self.params.fp16_vars)
          self.data_format, self.params.use_tf_layers)
      results = {}  # The return value
      if not phase_train or self.params.print_training_accuracy:
        top_1_op = tf.reduce_sum(
            tf.cast(tf.nn.in_top_k(logits, labels, 1), data_type))
        top_5_op = tf.reduce_sum(
            tf.cast(tf.nn.in_top_k(logits, labels, 5), data_type))
        results['top_1_op'] = top_1_op
        results['top_5_op'] = top_5_op

      if not phase_train:
        results['logits'] = logits
        return results
      loss_func = self.model.loss_function or loss_function
      base_loss = loss_func(logits, labels, aux_logits=aux_logits)
      params = self.variable_mgr.trainable_variables_on_device(
          rel_device_num, abs_device_num)
      fp32_params = params
      #if data_type == tf.float16 and self.params.fp16_vars:
        # fp16 reductions are very slow on GPUs, so cast to fp32 before calling
        # tf.nn.l2_loss and tf.add_n.
        # TODO(b/36217816): Once the bug is fixed, investigate if we should do
        # this reduction in fp16.
      #  fp32_params = (tf.cast(p, tf.float32) for p in params)
      total_loss = base_loss
      if rel_device_num == len(self.devices) - 1:
        # We compute the L2 loss for only one device instead of all of them,
        # because the L2 loss for each device is the same. To adjust for this,
        # we multiply the L2 loss by the number of devices. We choose the last
        # device because for some reason, on a Volta DGX1, the first four
        # GPUs take slightly longer to complete a step than the last four.
        # TODO(reedwm): Shard the L2 loss computations across GPUs.
        if self.params.single_l2_loss_op:
          # TODO(reedwm): If faster, create a fused op that does the L2 loss on
          # multiple tensors, and use that instead of concatenating tensors.
          reshaped_params = [tf.reshape(p, (-1,)) for p in fp32_params]
          l2_loss = tf.nn.l2_loss(tf.concat(reshaped_params, axis=0))
        else:
          l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in fp32_params])
        weight_decay = self.params.weight_decay
        if weight_decay is not None and weight_decay != 0.:
          total_loss += len(self.devices) * weight_decay * l2_loss

      aggmeth = tf.AggregationMethod.DEFAULT
      scaled_loss = total_loss
      #scaled_loss = (total_loss if self.loss_scale is None
      #               else total_loss * self.loss_scale)
      grads = tf.gradients(scaled_loss, params, aggregation_method=aggmeth)
      #if self.loss_scale is not None:
        # TODO(reedwm): If automatic loss scaling is not used, we could avoid
        # these multiplications by directly modifying the learning rate instead.
        # If this is done, care must be taken to ensure that this scaling method
        # is correct, as some optimizers square gradients and do other
        # operations which might not be compatible with modifying both the
        # gradients and the learning rate.

      #  grads = [
      #      grad * tf.cast(1. / self.loss_scale, grad.dtype) for grad in grads
      #  ]

      param_refs = self.variable_mgr.trainable_variables_on_device(
          rel_device_num, abs_device_num, writable=True)
      gradvars = list(zip(grads, param_refs))
      if self.params.loss_type_to_report == 'total_loss':
        results['loss'] = total_loss
      else:
        results['loss'] = base_loss
      results['gradvars'] = gradvars
      return results

  def get_image_preprocessor(self):
    """Returns the image preprocessor to used, based on the model.

    Returns:
      The image preprocessor, or None if synthetic data should be used.
    """
    image_size = self.model.get_image_size()
    input_data_type = get_data_type(self.params)

    shift_ratio = 0

    processor_class = self.dataset.get_image_preprocessor(
        self.params.input_preprocessor)
    assert processor_class
    return processor_class(
        image_size,
        image_size,
        #self.batch_size * self.batch_group_size,
        self.batch_size,
        #len(self.devices) * self.batch_group_size,
        len(self.devices),
        dtype=input_data_type,
        train=(not self.params.eval),
        distortions=self.params.distortions,
        resize_method=self.resize_method,
        shift_ratio=shift_ratio,
        summary_verbosity=self.params.summary_verbosity,
        distort_color_in_yiq=self.params.distort_color_in_yiq,
        fuse_decode_and_crop=self.params.fuse_decode_and_crop)


def store_benchmarks(names_to_values, params):
  if params.result_storage:
    benchmark_storage.store_benchmark(names_to_values, params.result_storage)


def setup():
  """Sets up the environment that BenchmarkCNN should run in.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  Returns:
    A potentially modified params.
  Raises:
    ValueError: invalid parames combinations.
  """
  if FLAGS.batchnorm_persistent:
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
  else:
    os.environ.pop('TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT', None)
  if FLAGS.winograd_nonfused:
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  else:
    os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
  os.environ['TF_SYNC_ON_FINISH'] = str(int(FLAGS.sync_on_finish))
  argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Sets environment variables for MKL
  if FLAGS.mkl:
    os.environ['KMP_BLOCKTIME'] = str(FLAGS.kmp_blocktime)
    os.environ['KMP_SETTINGS'] = str(FLAGS.kmp_settings)
    os.environ['KMP_AFFINITY'] = FLAGS.kmp_affinity
    if FLAGS.num_intra_threads > 0:
      os.environ['OMP_NUM_THREADS'] = str(FLAGS.num_intra_threads)

  # Sets GPU thread settings
  FLAGS.gpu_thread_mode=FLAGS.gpu_thread_mode.lower()
  
  if FLAGS.gpu_thread_mode not in ['global', 'gpu_shared', 'gpu_private']:
    raise ValueError('Invalid gpu_thread_mode: %s' % FLAGS.gpu_thread_mode)
  os.environ['TF_GPU_THREAD_MODE'] = FLAGS.gpu_thread_mode

  if FLAGS.per_gpu_thread_count and FLAGS.gpu_thread_mode == 'global':
    raise ValueError(
        'Invalid per_gpu_thread_count with gpu_thread_mode=global: %s' %
        FLAGS.per_gpu_thread_count)
  # Default to two threads. One for the device compute and the other for
  # memory copies.
  per_gpu_thread_count = FLAGS.per_gpu_thread_count or 2
  total_gpu_thread_count = per_gpu_thread_count * FLAGS.num_gpus

  if FLAGS.gpu_thread_mode == 'gpu_private':
    os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
  elif FLAGS.gpu_thread_mode == 'gpu_shared':
    os.environ['TF_GPU_THREAD_COUNT'] = str(total_gpu_thread_count)

  cpu_count = multiprocessing.cpu_count()
  if not FLAGS.num_inter_threads and FLAGS.gpu_thread_mode in [
      'gpu_private', 'gpu_shared'
  ]:
    main_thread_count = max(cpu_count - total_gpu_thread_count, 1)
    FLAGS.num_inter_threads=main_thread_count

  if (FLAGS.datasets_use_prefetch and
      FLAGS.datasets_num_private_threads is None):
    # From the total cpu thread count, subtract the total_gpu_thread_count,
    # and then 2 threads per GPU device for event monitoring and sending /
    # receiving tensors
    num_monitoring_threads = 2 * FLAGS.num_gpus
    num_private_threads = max(
        cpu_count - total_gpu_thread_count - num_monitoring_threads, 1)
    FLAGS.datasets_num_private_threads=num_private_threads

  #if FLAGS.variable_update == 'horovod':
  #  import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
  #  hvd.init()
  if FLAGS.save:
    FLAGS.train_dir=FLAGS.save
  if FLAGS.optimization:
    FLAGS.optimizer=FLAGS.optimization
  if FLAGS.epoch:
    logging.info("epoch: {}".format(FLAGS.epoch))
    FLAGS.num_epochs=FLAGS.epoch
    logging.info("num_epochs: {}".format(FLAGS.num_epochs))

  platforms_util.initialize(FLAGS, create_config_proto(FLAGS))

  return FLAGS
