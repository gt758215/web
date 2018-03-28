from tensorpack import *
import tensorflow as tf
import os


FLAGS = tf.app.flags.FLAGS

# Basic model parameters. #float, integer, boolean, string
tf.app.flags.DEFINE_string('gpu', '', """comma separated list of GPU(s) to use.""")
tf.app.flags.DEFINE_string('networkDirectory', '', """Directory in which network exists""")
tf.app.flags.DEFINE_string('network', '', """File containing network (model)""")
tf.app.flags.DEFINE_integer('epoch', 1, """Number of epochs to train""")
tf.app.flags.DEFINE_integer('batch_size', 16, """Number of images to process in a batch""")


def get_config(networkDirectory="", network="", epoch=1, batch_size=16):
    logger.auto_set_dir(action='d')

    path_network = os.path.join(os.path.dirname(os.path.realpath(__file__)), networkDirectory, network)
    print(path_network)
    exec(open(path_network).read(), globals())

    dataset_train, dataset_test = get_data()
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
        max_epoch=epoch,
    )


def main(_):
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    config = get_config(FLAGS.networkDirectory, FLAGS.network, FLAGS.epoch, FLAGS.batch_size)
    launch_train_with_config(config, SimpleTrainer())

if __name__ == '__main__':
    tf.app.run()