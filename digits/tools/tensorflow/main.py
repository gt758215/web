from tensorpack import *
import tensorflow as tf
import os
from dataset_utils import get_data


FLAGS = tf.app.flags.FLAGS

# Basic model parameters. #float, integer, boolean, string
tf.app.flags.DEFINE_string('gpu', '', """comma separated list of GPU(s) to use.""")
tf.app.flags.DEFINE_string('networkDirectory', '', """Directory in which network exists""")
tf.app.flags.DEFINE_string('network', '', """File containing network (model)""")
tf.app.flags.DEFINE_integer('epoch', 1, """Number of epochs to train""")
tf.app.flags.DEFINE_integer('batch_size', 16, """Number of images to process in a batch""")
tf.app.flags.DEFINE_string('train_db', '', """Directory with training file source""")


def get_config():
    logger.auto_set_dir(action='d')

    path_network = os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.networkDirectory, FLAGS.network)
    print(path_network)
    exec(open(path_network).read(), globals())

    dataset_train, dataset_test = get_data(FLAGS.batch_size, FLAGS.train_db)
    from tensorpack.callbacks import (
        TFEventWriter, JSONWriter, ScalarPrinter,
        MovingAverageSummary, ProgressBar, MergeAllSummaries, RunUpdateOps)
    monitors = [TFEventWriter(), JSONWriter(), ScalarPrinter(enable_step=True, enable_epoch=True)]
    callbacks = [MovingAverageSummary(), MergeAllSummaries(period=50), RunUpdateOps()]

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[
            #ModelSaver(),
            InferenceRunner(
                dataset_test,
                ScalarStats(['cross_entropy_loss', 'accuracy'])),
        ],
        monitors=monitors,
        extra_callbacks=callbacks,
        max_epoch=FLAGS.epoch,
    )


def main(_):
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    config = get_config()
    launch_train_with_config(config, SimpleTrainer())

if __name__ == '__main__':
    tf.app.run()
