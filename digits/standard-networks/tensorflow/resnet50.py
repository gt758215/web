from model import Tower
from utils import model_property
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils as digits
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.python.ops import array_ops


class UserModel(Tower):
    @model_property
    def inference(self):

        x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, end_points = resnet_v1.resnet_v1_50(x, num_classes=self.nclasses, is_training=self.is_training
            #    , spatial_squeeze=True
                , global_pool=True
                )
        net = array_ops.squeeze(logits, [1,2], name='SpatialSqueeze')
        return net

    @model_property
    def loss(self):
        model = self.inference
        loss = digits.classification_loss(model, self.y)
        accuracy = digits.classification_accuracy(model, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss
