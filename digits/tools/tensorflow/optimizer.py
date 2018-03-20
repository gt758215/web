#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf


class ProxyOptimizer(tf.train.Optimizer):
    """
    A transparent proxy which delegates all methods of :class:`tf.train.Optimizer`
    """
    def __init__(self, opt, name='ProxyOptimizer'):
        assert isinstance(opt, tf.train.Optimizer), opt
        super(ProxyOptimizer, self).__init__(False, name)
        self._opt = opt

    def compute_gradients(self, *args, **kwargs):
        return self._opt.compute_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        return self._opt.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._opt.get_slot_names(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        return self._opt.apply_gradients(*args, **kwargs)

class AccumGradOptimizerAlt(ProxyOptimizer):
    """
    An optimizer which accumulates gradients across :math:`k` :meth:`minimize` calls,
    and apply them together in every :math:`k`th :meth:`minimize` call.
    This is equivalent to using a :math:`k` times larger batch size plus a
    :math:`k` times larger learning rate, but uses much less memory.

    Note that this implementation may not support all models.
    E.g., it doesn't support sparse gradient update.
    """

    def __init__(self, opt, niter=1):
        """
        Args:
            opt (tf.train.Optimizer): the underlying sub-optimizer.
            niter (int): number of iterations to accumulate gradients.
        """
        super(AccumGradOptimizerAlt, self).__init__(opt, 'AccumGrad')
        self._niter = int(niter)

    def compute_gradients(self, *args, **kwargs):
        # get trainalbe variable
        #if get_current_tower_context()!=None and get_current_tower_context().has_own_variables:
        #    trainable_var = get_current_tower_context().get_collection_in_tower(
        #        tf.GraphKeys.TRAINABLE_VARIABLES)
        #else:
        #    trainable_var = tf.trainable_variables()

        # Another method to get trainable variable

        #from tensorflow.python.framework import ops
        #from tensorflow.python.util import nest
        #from tensorflow.python.eager import context
        #from tensorflow.python.ops import variables

        #if context.in_eager_mode():
        #    raise RuntimeError("accum not support eager mode")
        #if(kwargs.get("var_list") != None):
        #    trainable_var = nest.flatten(kwargs.get("var_list"))
        #else:
        #    trainable_var = (
        #        variables.trainable_variables() +
        #        ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
            #raise RuntimeError("var_list can't be empty")
        #trainable_var += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)
        
        #if(kwargs.get("var_list") != None):
        #    trainable_col = set([kwargs.get("var_list")])
        #else:
        #    trainable_col = set([tf.GraphKeys.GLOBAL_VARIABLES])
        #trainable_col.remove(tf.GraphKeys.GLOBAL_VARIABLES)
        #trainable_col.add(tf.GraphKeys.LOCAL_VARIABLES)

        #kwargs['var_list'] = trainable_var
        #print kwargs['var_list']
        


        # Counter variable 
        accum_times = self._niter
        #with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            #counter = tf.get_variable(initializer=tf.constant_initializer(0), name="counter", shape=[], trainable=False, dtype=tf.int32)
        counter = tf.Variable(0, name="counter", trainable=False, dtype=tf.int32)

        # Get gradients and weights from original Optimizer
        grads_and_vars = self._opt.compute_gradients(*args, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g != None]
        #print grads_and_vars

        trainable_var =  [v for _, v in grads_and_vars]
        # Create slots for storing accumulated gradients
        with tf.variable_scope(self._name, reuse = tf.AUTO_REUSE):
            accum_grads = [self._zeros_slot(v, "accum_grad", self._name) for v in  trainable_var]

        # ==================================
        # Update counter lambda
        def counter_add():
          return tf.assign_add(counter, 1)
        def counter_reset():
          return tf.assign(counter, 1)
        # ==================================

        # Update op: is like as "counter = 1+(counter % accum_times-1)"
        update_counter = tf.cond(tf.equal(counter, accum_times), counter_reset, counter_add, name='update_counter')

        # ==================================
        # Clear grads lambda
        def grads_clear():
            clear_ops = [tf.assign(s, tf.zeros_like(s)) for s in accum_grads]
            return tf.group(*clear_ops, name='clear_grads')
        # ===================================


        # Clear slots if counter equal to 1, and only run after the update counter operation
        with tf.control_dependencies([update_counter]):
            cond_clear_grads = tf.cond(tf.equal(counter, 1), grads_clear, tf.no_op, name='cond_clear_grads')


        # Fetch gradients in grads_and_vars, and add them to accum_grads
        # Thus, tuple those accum_grads with trainable_var
        # It will compose of a list of all tuple(accum_grad, var)
        with tf.control_dependencies([cond_clear_grads]):
            return list(zip([tf.assign_add(s, tf.divide(g, accum_times)) for s, (g, _) in zip(accum_grads, grads_and_vars)], trainable_var))

