from model import Tower
from utils import model_property
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils as digits
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.python.ops import array_ops

def my_resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]

  return resnet_v1.resnet_v1(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)

class UserModel(Tower):
    @model_property
    def inference(self):
        x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, end_points = resnet_v1.resnet_v1_50(x, num_classes=self.nclasses, is_training=self.is_training
            #    , spatial_squeeze=True
                , global_pool=True
                )
        # remove in the future if squeeze build in resnet_v1 function
        net = array_ops.squeeze(logits, [1,2], name='SpatialSqueeze')
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
