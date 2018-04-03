from tensorpack import *
import tensorflow as tf
import os
from dataset_utils import get_data
from tensorpack.utils.gpu import get_nr_gpu


FLAGS = tf.app.flags.FLAGS

# Basic model parameters. #float, integer, boolean, string
tf.app.flags.DEFINE_string('gpu', '', """comma separated list of GPU(s) to use.""")
tf.app.flags.DEFINE_string('networkDirectory', '', """Directory in which network exists""")
tf.app.flags.DEFINE_string('network', '', """File containing network (model)""")
tf.app.flags.DEFINE_integer('epoch', 1, """Number of epochs to train""")
tf.app.flags.DEFINE_integer('batch_size', 16, """Number of images to process in a batch""")
tf.app.flags.DEFINE_string('train_db', '', """Directory with training file source""")


def get_config():
    nr_tower = max(get_nr_gpu(), 1)
    batch = FLAGS.batch_size // nr_tower
    total_batch = FLAGS.batch_size
    BASE_LR = 0.1 * (total_batch / 256.)

    logger.auto_set_dir(action='d')
    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))

    path_network = os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.networkDirectory, FLAGS.network)
    exec(open(path_network).read(), globals())

    dataset_train, dataset_val = get_data(batch, FLAGS.train_db)
    from tensorpack.callbacks import (
        TFEventWriter, JSONWriter, ScalarPrinter,
        MovingAverageSummary, ProgressBar, MergeAllSummaries, RunUpdateOps)
    monitors = [TFEventWriter(), JSONWriter(), ScalarPrinter(enable_step=True, enable_epoch=True)]
    extra_callbacks = [MovingAverageSummary(), MergeAllSummaries(period=50), RunUpdateOps()]
    callbacks = [
        ModelSaver(),
        # FIXME this is not working
        ScheduledHyperParamSetter(
            'learning_rate',
            [(0, 0.01), (3, max(BASE_LR, 0.01))], interp='linear'),
        ScheduledHyperParamSetter(
            'learning_rate',
            [(30, BASE_LR * 1e-1), (60, BASE_LR * 1e-2), (80, BASE_LR * 1e-3)]),
    ]

    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]
    if nr_tower == 1:
        # single-GPU inference with queue prefetch
        callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
    else:
        # multi-GPU inference (with mandatory queue prefetch)
        callbacks.append(DataParallelInferenceRunner(
            dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=callbacks,
        monitors=monitors,
        extra_callbacks=extra_callbacks,
        max_epoch=FLAGS.epoch,
        nr_tower=nr_tower,
    )


def main(_):
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    config = get_config()

    trainer = SyncMultiGPUTrainerReplicated(max(get_nr_gpu(), 1))
    launch_train_with_config(config, trainer)

if __name__ == '__main__':
    tf.app.run()
