from tensorpack import *
from tensorpack.dataflow import dataset
import tensorflow as tf
import tensorflow.contrib.slim as slim

IMAGE_SIZE=225


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.uint8, (None, IMAGE_SIZE, IMAGE_SIZE, 3), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def image_preprocess(image, bgr=True):
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

    def _build_graph(self, inputs):
        image, label = inputs
        image = self.image_preprocess(image, bgr=True) 
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])
        with slim.arg_scope([slim.layers.fully_connected],
                            weights_regularizer=slim.l2_regularizer(1e-5)):
            net = slim.repeat(image, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding="VALID", scope='fc6')
            net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout7')
            net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
            logits = tf.squeeze(net, [1, 2], name='fc8/squeezed')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        acc = tf.to_float(tf.nn.in_top_k(logits, label, 1))
        acc = tf.reduce_mean(acc, name='accuracy')
        summary.add_moving_summary(acc)

        self.cost = cost
        summary.add_moving_summary(cost)
        summary.add_param_summary(('.*/weights', ['histogram', 'rms']))  # slim uses different variable names

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
