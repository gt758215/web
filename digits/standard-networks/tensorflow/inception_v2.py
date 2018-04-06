from model import Tower
from utils import model_property
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils as digits
from tensorflow.contrib.slim.python.slim.nets import inception_v2
from tensorflow.python.ops import array_ops


class UserModel(Tower):
    @model_property
    def inference(self):

        x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
            logits, end_points = inception_v2.inception_v2(x, num_classes=self.nclasses, is_training=self.is_training
                , spatial_squeeze=True
                )
        return logits

    @model_property
    def loss(self):
        model = self.inference
        #loss = slim.losses.softmax_cross_entropy(model, self.y)
        loss = digits.classification_loss(model, self.y)
        accuracy = digits.classification_accuracy(model, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss
~
