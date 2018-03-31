# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# This document should comply with PEP-8 Style Guide
# Linter: pylint

"""
Interface for setting up and creating a model in Tensorflow.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
from tensorflow.python.framework import ops

# Local imports
import tf_data
import utils as digits
from utils import model_property

import optimizer as opt

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Constants
SUMMARIZE_TOWER_STATS = False


# from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py
def average_gradients(tower_grads, target_device):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    # main updating gpu
    t_gpu = target_device

    with tf.name_scope('gradient_average'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            # Average over the 'tower' dimension.
            with tf.device(grads[t_gpu].device):
                grads_transformed = tf.concat(grads, 0)
                grads_transformed = tf.reduce_mean(grads_transformed, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[t_gpu][1]
            grad_and_var = (grads_transformed, v)
            average_grads.append(grad_and_var)
        return average_grads


# from tensorpack
def average_grads(all_grads):
    """
    Average the gradients, on the device of each variable.

    Args:
        all_grads (K x N x 2): A list of K lists. Each of the list is a list of N (grad, var) tuples.
            The variables have to be the same across the K lists.
        colocation (bool): colocate gradient averaging with the variable

    Returns:
        (N x 2): A list of N (grad, var) tuples, where grad is averaged over K.
    """

    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads[0]

    new_all_grads = []  # NVar * NGPU * 2
    with tf.name_scope('AvgGrad'):
        for grad_and_vars in zip(*all_grads):
            # Ngpu * 2
            grads = [g for (g, _) in grad_and_vars]
            summed = tf.multiply(tf.add_n(grads), 1.0 / nr_tower)

            grads_for_a_var = []
            for (_, v), g in zip(grad_and_vars, [summed]*nr_tower):
                grads_for_a_var.append((g, v))
            new_all_grads.append(grads_for_a_var)

    ret =  [list(k) for k in zip(*new_all_grads)]
    return ret


# from tensorpack
def allreduce_gradients_bak(tower_grads):
    from tensorflow.contrib import nccl
    nr_tower = len(tower_grads)
    new_all_grads = []  # NVar * NGPU * 2
    with tf.name_scope('gradient_allreduce'):
        for grad_and_vars in zip(*tower_grads):
            #v = grad_and_vars[0][1]
            grads = [g for g, _ in grad_and_vars]
            summed = nccl.all_sum(grads)

            grads_for_a_var = []
            for (_, v), g in zip(grad_and_vars, summed):
                with tf.device(g.device):
                    g = tf.multiply(g, 1.0 / nr_tower)
                    grads_for_a_var.append((g, v))
            new_all_grads.append(grads_for_a_var)

    # transpose
    ret =  [list(k) for k in zip(*new_all_grads)]

    return ret

def allreduce_gradients(tower_grads, target_device):
    from tensorflow.contrib import nccl
    nr_tower = len(tower_grads)

    # main updating gpu
    t_gpu = target_device

    with tf.name_scope('gradient_allreduce'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            #v = grad_and_vars[0][1]
            grads = [g for g, _ in grad_and_vars]
            summed = nccl.all_sum(grads)
            with tf.control_dependencies(summed):
                g = summed[t_gpu]
                with tf.device(g.device):
                    g = tf.multiply(g, 1.0 / nr_tower)
                
            v = grad_and_vars[t_gpu][1]
            grad_and_var = (g, v)

            average_grads.append(grad_and_var)
            #t_gpu +=1
            #t_gpu = t_gpu % nr_tower

        return average_grads


class Model(object):
    """
    Wrapper around the actual tensorflow workflow process.
    This is structured in a way that the user should only care about
    creating the model while using the DIGITS UI to select the
    optimizer and other options.

    This class is executed to start a tensorflow workflow.
    """
    def __init__(self, stage, croplen, nclasses, optimization=None, momentum=None, reuse_variable=False):
        self.stage = stage
        self.croplen = croplen
        self.nclasses = nclasses
        self.dataloader = None
        self.queue_coord = None
        self.queue_threads = None

        self._optimization = optimization
        self._momentum = momentum
        self.summaries = []
        self.towers = []
        self._train = None
        self._reuse = reuse_variable

        self._accum = None
        self._init = None
        self.small_chunk = 1
        self.nccl = False
        self.replica = False

        # Touch to initialize
        # if optimization:
        #     self.learning_rate
        #     self.global_step
        #     self.optimizer

    @staticmethod
    def get_post_init_ops():
        """
        Copy values of variables on GPU 0 to other GPUs.
        """
        # literally all variables, because it's better to sync optimizer-internal variables as well
        all_vars = tf.global_variables() + tf.local_variables()
        #var_by_name = dict([(v.name, v) for v in all_vars])
        var_by_name = dict()
        for v in all_vars:
            if v.name.startswith('tower_0'):
                split_name = v.name.split('/')
                realname = '/'.join(split_name[1:])
                var_by_name[realname] = v

        post_init_ops = []
        for v in all_vars:
            if not v.name.startswith('tower_'):
                continue
            if v.name.startswith('tower_0'):
                continue
            # in this trainer, the master name doesn't have the towerx/ prefix
            split_name = v.name.split('/')
            prefix = split_name[0]
            realname = '/'.join(split_name[1:])
            if 'AccumGrad' in realname:
                continue
            if 'counter' in realname:
                continue
            if prefix in realname:
                logger.error("[SyncMultiGPUReplicatedBuilder] variable "
                             "{} has its prefix {} appears multiple times in its name!".format(v.name, prefix))
            copy_from = var_by_name.get(realname)
            assert copy_from is not None, (realname, var_by_name.keys())
            #assert copy_from is not None, var_by_name.keys()
            post_init_ops.append(v.assign(copy_from.read_value()))
        return tf.group(*post_init_ops, name='sync_variables_from_main_tower')


    def create_dataloader(self, db_path):
        self.dataloader = tf_data.LoaderFactory.set_source(db_path, is_inference=(self.stage == digits.STAGE_INF))
        # @TODO(tzaman) communicate the dataloader summaries to our Model summary list
        self.dataloader.stage = self.stage
        self.dataloader.croplen = self.croplen
        self.dataloader.nclasses = self.nclasses

    def init_dataloader(self):
        with tf.device('/cpu:0'):
            with tf.name_scope(digits.GraphKeys.LOADER):
                self.dataloader.create_input_pipeline()

    def create_model(self, obj_UserModel, stage_scope, batch_x=None):

        if batch_x is None:
            self.init_dataloader()
            batch_x = self.dataloader.batch_x
            if self.stage != digits.STAGE_INF:
                batch_y = self.dataloader.batch_y
        else:
            assert self.stage == digits.STAGE_INF
            batch_x = batch_x

        available_devices = digits.get_available_gpus()
        if not available_devices:
            available_devices.append('/cpu:0')

        # available_devices = ['/gpu:0', '/gpu:1'] # DEVELOPMENT : virtual multi-gpu

        # Split the batch over the batch dimension over the number of available gpu's
        if len(available_devices) == 1:
            batch_x_split = [batch_x]
            if self.stage != digits.STAGE_INF:  # Has no labels
                batch_y_split = [batch_y]
        else:
            with tf.name_scope('parallelize'):
                # Split them up
                batch_x_split = tf.split(batch_x, len(available_devices), 0, name='split_batch')
                if self.stage != digits.STAGE_INF:  # Has no labels
                    batch_y_split = tf.split(batch_y, len(available_devices), 0, name='split_batch')

        # Get global regularizaion loss collection reference as a list named r_loss_global.
        # Now we can edit regularizaion loss collection by operation r_loss_global list
        r_loss_global = tf.get_collection_ref(ops.GraphKeys.REGULARIZATION_LOSSES)


        # Note: 
        # (In training stage)
        # r_loss_train_bak = [] (a bak to store all tower's regularizaion loss)
        # r_loss_global = (global regularizaion loss)'s reference 
        # For each Tower:
        #     empty r_loss_global
        #     Tower.inference (may add regularizaion loss globally) 
        #     r_loss_tain_bak += r_loss_global
        #     ...
        #
        # (restore all tower's reg. loss so validation stage could use it)
        # r_loss_global[:] = r_loss_train_bak[:] 
        #

        # (In validation stage)
        # r_loss_global = (global regularizaion loss)'s reference 
        # r_loss_val_bak = list(r_loss_global)   <= deep copy
        # For each Tower:
        #     empty r_loss_global
        #     parse element name start with 'tower_%d' % dev_i  in r_loss_val_bak
        #         ... and save to r_loss_global
        # 
        #     Tower.inference (will not add any regularizaion loss cause reuse=True)
        #     ( Some operations only catch regularizaion losses belong to current tower)
        #     ...
        #

        if self.replica:
            # Save regularizaion loss of all tower in training stage
            if self.stage != digits.STAGE_TRAIN:
                r_loss_val_bak = list(r_loss_global)
            # Create a list to store regularizaion loss
            if self.stage == digits.STAGE_TRAIN:
                r_loss_train_bak = list()

        # Run the user model through the build_model function that should be filled in
        grad_towers = []
        for dev_i, dev_name in enumerate(available_devices):
            with tf.device(dev_name):
                if self.replica :
                    r_loss_global[:] = []
                    if self.stage != digits.STAGE_TRAIN:
                        r_loss_global = [loss for loss in r_loss_val_bak if loss.name.startswith('train/tower_%d' % dev_i)]

                with tf.name_scope('tower_%d' % dev_i) as scope_tower:
                    if self.stage != digits.STAGE_INF:
                        tower_model = self.add_tower(obj_tower=obj_UserModel,
                                                     x=batch_x_split[dev_i],
                                                     y=batch_y_split[dev_i])
                    else:
                        tower_model = self.add_tower(obj_tower=obj_UserModel,
                                                     x=batch_x_split[dev_i],
                                                     y=None)

                    with tf.variable_scope('tower_0' if not self.replica else 'tower_%d' % dev_i, 
                                            reuse=(False if self.replica else dev_i > 0 ) or self._reuse):
                        tower_model.inference  # touch to initialize

                        # Reuse the variables in this scope for the next tower/device
                        tf.get_variable_scope().reuse_variables()

                        if self.stage == digits.STAGE_INF:
                            # For inferencing we will only use the inference part of the graph
                            continue

                        with tf.name_scope(digits.GraphKeys.LOSS):
                            for loss in self.get_tower_losses(tower_model, dev_i):
                                tf.add_to_collection(digits.GraphKeys.LOSSES, loss['loss'])

                            # Assemble all made within this scope so far. The user can add custom
                            # losses to the digits.GraphKeys.LOSSES collection
                            losses = tf.get_collection(digits.GraphKeys.LOSSES, scope=scope_tower)

                            if(self.replica) and self.stage == digits.STAGE_TRAIN:
                                r_loss_train_bak += r_loss_global

                            losses += ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES, scope=None)
                            tower_loss = tf.add_n(losses, name='loss')

                            self.summaries.append(tf.summary.scalar(tower_loss.op.name, tower_loss))

                        if self.stage == digits.STAGE_TRAIN:
                            grad_tower_losses = []
                            for loss in self.get_tower_losses(tower_model, dev_i):
                                # use loss + regularization loss instead of loss only
                                grad_tower_loss = self.optimizer.compute_gradients(tower_loss, loss['vars'])
                                grad_tower_loss = tower_model.gradientUpdate(grad_tower_loss)
                                grad_tower_losses.append(grad_tower_loss)
                            grad_towers.append(grad_tower_losses)

        # Assemble and average the gradients from all towers
        if self.stage == digits.STAGE_TRAIN:
            if self.replica:
                r_loss_global[:] = r_loss_train_bak[:]

            grad_accum = []
            grad_averages = []
            n_gpus = len(available_devices)

            if n_gpus == 1:
                n_losses = len(grad_towers[0])
                for loss in xrange(n_losses):
                    if(self.replica):
                        grad_averages.append([grad_towers[0][loss]])
                    else:
                        grad_averages.append(grad_towers[0][loss])
                    for g, _ in grad_towers[0][loss]:
                        grad_accum.append(g)
            else:
                n_losses = len(grad_towers[0])
                for loss in xrange(n_losses):
                    if not self.nccl:
                        if(self.replica):
                            grad_averages.append(average_grads([grad_towers[gpu][loss] for gpu in xrange(n_gpus)]))
                        else:
                            grad_averages.append(average_gradients([grad_towers[gpu][loss] for gpu in xrange(n_gpus)], 0))
                    else:
                        if(self.replica):
                            grad_averages.append(allreduce_gradients_bak([grad_towers[gpu][loss] for gpu in xrange(n_gpus)]))
                        else:
                            grad_averages.append(allreduce_gradients([grad_towers[gpu][loss] for gpu in xrange(n_gpus)], 0))

                    for gpu in xrange(n_gpus):
                        for g, _ in grad_towers[gpu][loss]:
                            grad_accum.append(g)

            apply_gradient_ops = []
            for grad_avg in grad_averages:
                if(self.replica):
                    tmp = []
                    for grad_and_vars in grad_avg:
                        for (g, v) in grad_and_vars:
                            tmp.append((g, v))
                else:
                    tmp = grad_avg

                apply_gradient_ops.append(self.optimizer.apply_gradients(tmp, global_step=self.global_step))

            self._train = apply_gradient_ops
            self._accum = tf.group(*grad_accum)
            if(self.replica):
                self._init = self.get_post_init_ops()
            else:
                self._init = []

    def start_queue_runners(self, sess):
        logging.info('Starting queue runners (%s)', self.stage)
        # Distinguish the queue runner collection (for easily obtaining them by collection key)
        queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS, scope=self.stage+'.*')
        for qr in queue_runners:
            if self.stage in qr.name:
                tf.add_to_collection(digits.GraphKeys.QUEUE_RUNNERS, qr)

        self.queue_coord = tf.train.Coordinator()
        self.queue_threads = tf.train.start_queue_runners(sess=sess, coord=self.queue_coord,
                                                          collection=digits.GraphKeys.QUEUE_RUNNERS)
        logging.info('Queue runners started (%s)', self.stage)

    def __del__(self):
        # Destructor
        if self.queue_coord:
            # Close and terminate the queues
            self.queue_coord.request_stop()
            self.queue_coord.join(self.queue_threads)

    def add_tower(self, obj_tower, x, y):
        is_training = self.stage == digits.STAGE_TRAIN
        is_inference = self.stage == digits.STAGE_INF
        input_shape = self.dataloader.get_shape()
        tower = obj_tower(x, y, input_shape, self.nclasses, is_training, is_inference)
        self.towers.append(tower)
        return tower

    @model_property
    def train(self):
        return self._train

    @model_property
    def accum(self):
        return self._accum

    @model_property
    def init(self):
        return self._init

    @model_property
    def summary(self):
        """
        Merge train summaries
        """
        for t in self.towers:
            self.summaries += t.summaries

        if not len(self.summaries):
            logging.error("No summaries defined. Please define at least one summary.")
            exit(-1)
        return tf.summary.merge(self.summaries)

    @model_property
    def global_step(self):
        # Force global_step on the CPU, becaues the GPU's first step will end at 0 instead of 1.
        with tf.device('/cpu:0'):
            return tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                   trainable=False)

    @model_property
    def learning_rate(self):
        # @TODO(tzaman): the learning rate is a function of the global step, so we could
        #  define it entirely in tf ops, instead of a placeholder and feeding.
        with tf.device('/cpu:0'):
            lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self.summaries.append(tf.summary.scalar('lr', lr))
            return lr

    def _optimizer(self):
        logging.info("Optimizer:%s", self._optimization)
        if self._optimization == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self._optimization == 'adadelta':
            return tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self._optimization == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        elif self._optimization == 'adagradda':
            return tf.train.AdagradDAOptimizer(learning_rate=self.learning_rate,
                                               global_step=self.global_step)
        elif self._optimization == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                              momentum=self._momentum)
        elif self._optimization == 'adam':
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self._optimization == 'ftrl':
            return tf.train.FtrlOptimizer(learning_rate=self.learning_rate)
        elif self._optimization == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                             momentum=self._momentum)
        else:
            logging.error("Invalid optimization flag %s", self._optimization)
            exit(-1)

    @model_property
    def optimizer(self):
        return opt.AccumGradOptimizerAlt(self._optimizer(), self.small_chunk)

    def get_tower_losses(self, tower, device):
        """
        Return list of losses

        If user-defined model returns only one loss then this is encapsulated into
        the expected list of dicts structure
        """
        # Note: Network editor have to maintain each loss with 'loss' and 'vars' if it's a list.
        if isinstance(tower.loss, list):
            return tower.loss
        else:
            tower_vars = []
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            if self.replica:
                tower_vars = [var for var in trainable_vars if(var.name.startswith('tower_%d' % device))]
            else:
                tower_vars = trainable_vars

            return [{'loss': tower.loss, 'vars': tower_vars}]

class Tower(object):

    def __init__(self, x, y, input_shape, nclasses, is_training, is_inference):
        self.input_shape = input_shape
        self.nclasses = nclasses
        self.is_training = is_training
        self.is_inference = is_inference
        self.summaries = []
        self.x = x
        self.y = y
        self.train = None

    def gradientUpdate(self, grad):
        return grad
