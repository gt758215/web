from model import Tower
from utils import model_property
import tensorflow as tf
import utils as digits

def BatchNorm(inputs, training=True, axis=3, momentum=0.9, epsilon=1e-5,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer()):
    return tf.layers.batch_normalization(
                inputs,
                axis=axis,
                training=training,
                momentum=momentum, epsilon=epsilon,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
            )

def Conv2D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='SAME',
        activation=None,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0)):
    return tf.layers.conv2d(
            inputs,
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer)

def bottleneck(l, ch_out, stride, is_training=True):
    shortcut = l
    l = Conv2D(l, ch_out, 1, 1)
    l = BatchNorm(l, is_training)
    l = tf.nn.relu(l)

    l = Conv2D(l, ch_out, 3, stride)
    l = BatchNorm(l, is_training)
    l = tf.nn.relu(l)

    l = Conv2D(l, ch_out*4, 1, 1)
    l = BatchNorm(l, is_training, gamma_initializer=tf.zeros_initializer())

    if shortcut.get_shape().as_list()[3] != ch_out*4:
        shortcut = Conv2D(shortcut, ch_out*4, 1, stride)
        shortcut = BatchNorm(shortcut, is_training)

    return tf.nn.relu(l + shortcut)


def resnet_v1(inputs,
              num_classes=None,
              is_training=True):
    with tf.variable_scope('resnet_v1'):
        net = inputs

        net = Conv2D(net, 64, 7, 2)
        net = BatchNorm(net, is_training)
        net = tf.layers.max_pooling2d(net, 3, 2, padding='SAME', scope='pool1')

        net = bottleneck(net, 64, 1, is_training)
        net = bottleneck(net, 64, 1, is_training)
        net = bottleneck(net, 64, 1, is_training)

        net = bottleneck(net, 128, 2, is_training)
        net = bottleneck(net, 128, 1, is_training)
        net = bottleneck(net, 128, 1, is_training)
        net = bottleneck(net, 128, 1, is_training)

        net = bottleneck(net, 256, 2, is_training)
        net = bottleneck(net, 256, 1, is_training)
        net = bottleneck(net, 256, 1, is_training)
        net = bottleneck(net, 256, 1, is_training)
        net = bottleneck(net, 256, 1, is_training)
        net = bottleneck(net, 256, 1, is_training)

        net = bottleneck(net, 512, 2, is_training)
        net = bottleneck(net, 512, 1, is_training)
        net = bottleneck(net, 512, 1, is_training)

        net = tf.reduce_mean(net, [1, 2], name='pool5')
        assert num_classes is not None
        net = tf.layers.dense(net,
                num_classes,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
                bias_initializer=tf.zeros_initializer(),
                scope='logits')
        net = tf.squeeze(net, [1,2], name='SpatialSqueeze')

    return net

class UserModel(Tower):
    @model_property
    def inference(self):
        x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        net = resnet_v1(x, num_classes=self.nclasses, is_training=self.is_training)
        return net

    @model_property
    def loss(self):
        model = self.inference
        loss = digits.classification_loss(model, self.y)
        acc_top1 = digits.classification_accuracy_top_n(model, self.y, 1)
        acc_top5 = digits.classification_accuracy_top_n(model, self.y, 5)
        self.summaries.append(tf.summary.scalar(acc_top1.op.name, acc_top1))
        self.summaries.append(tf.summary.scalar(acc_top5.op.name, acc_top5))
        return loss
