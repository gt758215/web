import tensorflow as tf

from tensorpack.graph_builder import ModelDesc
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.summary import add_param_summary, add_moving_summary
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap)


def GroupNorm(x, group):
    """
    https://arxiv.org/abs/1803.08494
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=tf.constant_initializer(1.0))
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')


def convnormrelu(x, name, chan):
    x = Conv2D(name, x, chan, 3)
    #if args.norm == 'bn':
    #    x = BatchNorm(name + '_bn', x)
    #elif args.norm == 'gn':
    #    with tf.variable_scope(name + '_gn'):
    #        x = GroupNorm(x, 32)
    x = tf.nn.relu(x, name=name + '_relu')
    return x


class Model(ModelDesc):
    weight_decay = 5e-4
    image_shape = 224
    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    def __init__(self, data_format='NCHW'):
        self.data_format = data_format

    def _get_inputs(self):
        return [InputDesc(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def get_logits(self, image):
        with argscope(Conv2D,
                      W_init=tf.variance_scaling_initializer(scale=2.)), \
                argscope([Conv2D, MaxPooling, BatchNorm], data_format='NCHW'):
            logits = (LinearWrap(image)
                      .apply(convnormrelu, 'conv1_1', 64)
                      .apply(convnormrelu, 'conv1_2', 64)
                      .MaxPooling('pool1', 2)
                      # 112
                      .apply(convnormrelu, 'conv2_1', 128)
                      .apply(convnormrelu, 'conv2_2', 128)
                      .MaxPooling('pool2', 2)
                      # 56
                      .apply(convnormrelu, 'conv3_1', 256)
                      .apply(convnormrelu, 'conv3_2', 256)
                      .apply(convnormrelu, 'conv3_3', 256)
                      .MaxPooling('pool3', 2)
                      # 28
                      .apply(convnormrelu, 'conv4_1', 512)
                      .apply(convnormrelu, 'conv4_2', 512)
                      .apply(convnormrelu, 'conv4_3', 512)
                      .MaxPooling('pool4', 2)
                      # 14
                      .apply(convnormrelu, 'conv5_1', 512)
                      .apply(convnormrelu, 'conv5_2', 512)
                      .apply(convnormrelu, 'conv5_3', 512)
                      .MaxPooling('pool5', 2)
                      # 7
                      .FullyConnected('fc6', 4096,
                                      W_init=tf.random_normal_initializer(stddev=0.001))
                      .tf.nn.relu(name='fc6_relu')
                      .Dropout('drop0', keep_prob=0.5)
                      .FullyConnected('fc7', 4096,
                                      W_init=tf.random_normal_initializer(stddev=0.001))
                      .tf.nn.relu(name='fc7_relu')
                      .Dropout('drop1', keep_prob=0.5)
                      .FullyConnected('fc8', 1000,
                                      W_init=tf.random_normal_initializer(stddev=0.01))())
        add_param_summary(('.*', ['histogram', 'rms']))
        return logits

    def _build_graph(self, inputs):
        image, label = inputs
        image = self.image_preprocess(image, bgr=True)
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self.get_logits(image)
        loss = self.compute_loss_and_error(logits, label)
        wd_loss = regularize_cost('.*/W', tf.contrib.layers.l2_regularizer(self.weight_decay),
                                  name='l2_regularize_loss')
        add_moving_summary(loss, wd_loss)
        self.cost = tf.add_n([loss, wd_loss], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def image_preprocess(self, image, bgr=True):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            image = image * (1.0 / 255)

            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std
            return image

    def compute_loss_and_error(self, logits, label):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='cross_entryopy_loss')

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        acc = tf.to_float(tf.nn.in_top_k(logits, label, 1))
        add_moving_summary(tf.reduce_mean(acc, name='accuracy'))

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss
