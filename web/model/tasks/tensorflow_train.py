from __future__ import absolute_import

import operator
import os
import re
import shutil
import subprocess
import tempfile
import time
import sys

import numpy as np

from .train import TrainTask
import web
from web import utils
from web.utils import subclass, override, constants
import tensorflow as tf

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

# Constants
TENSORFLOW_MODEL_FILE = 'network.py'
TENSORFLOW_SNAPSHOT_PREFIX = 'snapshot'
TIMELINE_PREFIX = 'timeline'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def subprocess_visible_devices(gpus):
    """
    Calculates CUDA_VISIBLE_DEVICES for a subprocess
    """
    if not isinstance(gpus, list):
        raise ValueError('gpus should be a list')
    gpus = [int(g) for g in gpus]

    old_cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if old_cvd is None:
        real_gpus = gpus
    else:
        map_visible_to_real = {}
        for visible, real in enumerate(old_cvd.split(',')):
            map_visible_to_real[visible] = int(real)
        real_gpus = []
        for visible_gpu in gpus:
            real_gpus.append(map_visible_to_real[visible_gpu])
    return ','.join(str(g) for g in real_gpus)


class TensorflowTrainTask(TrainTask):
    """
    Trains a tensorflow model
    """

    TENSORFLOW_LOG = 'tensorflow_output.log'

    def __init__(self, **kwargs):
        """
        Arguments:
        network -- a NetParameter defining the network
        """
        super(TensorflowTrainTask, self).__init__(**kwargs)

        # save network description to file
        with open(os.path.join(self.job_dir, TENSORFLOW_MODEL_FILE), "w") as outfile:
            outfile.write(self.network)

        self.pickver_task_tensorflow_train = PICKLE_VERSION

        self.current_epoch = 0

        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None
        self.image_mean = None
        self.classifier = None
        self.solver = None

        self.model_file = TENSORFLOW_MODEL_FILE
        self.train_file = constants.TRAIN_DB
        self.val_file = constants.VAL_DB
        self.snapshot_prefix = TENSORFLOW_SNAPSHOT_PREFIX
        self.log_file = self.TENSORFLOW_LOG

    def __getstate__(self):
        state = super(TensorflowTrainTask, self).__getstate__()

        # Don't pickle these things
        if 'labels' in state:
            del state['labels']
        if 'image_mean' in state:
            del state['image_mean']
        if 'classifier' in state:
            del state['classifier']
        if 'tensorflow_log' in state:
            del state['tensorflow_log']

        return state

    def __setstate__(self, state):
        super(TensorflowTrainTask, self).__setstate__(state)

        # Make changes to self
        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None

        # These things don't get pickled
        self.image_mean = None
        self.classifier = None

    # Task overrides
    @override
    def name(self):
        return 'Train Tensorflow Model'

    @override
    def before_run(self):
        super(TensorflowTrainTask, self).before_run()
        self.tensorflow_log = open(self.path(self.TENSORFLOW_LOG), 'a')
        self.saving_snapshot = False
        self.receiving_train_output = False
        self.receiving_val_output = False
        self.last_train_update = None
        self.displaying_network = False
        self.temp_unrecognized_output = []
        return True

    @override
    def get_snapshot(self, epoch=-1, download=False, frozen_file=False):
        """
        return snapshot file for specified epoch
        """
        snapshot_pre = None

        if len(self.snapshots) == 0:
            return "no snapshots"

        if epoch == -1 or not epoch:
            epoch = self.snapshots[-1][1]
            snapshot_pre = self.snapshots[-1][0]
        else:
            for f, e in self.snapshots:
                if e == epoch:
                    snapshot_pre = f
                    break
        if not snapshot_pre:
            raise ValueError('Invalid epoch')
        if download:
            snapshot_file = snapshot_pre + ".data-00000-of-00001"
            meta_file = snapshot_pre + ".meta"
            index_file = snapshot_pre + ".index"
            snapshot_files = [snapshot_file, meta_file, index_file]
        elif frozen_file:
            snapshot_files = os.path.join(os.path.dirname(snapshot_pre), "frozen_model.pb")
        else:
            snapshot_files = snapshot_pre

        return snapshot_files

    @override
    def task_arguments(self, resources, env):

        args = [sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(web.__file__)), 'tools', 'tensorflow', 'main.py'),
                '--network=%s' % self.model_file,
                '--epoch=%d' % int(self.train_epochs),
                '--networkDirectory=%s' % self.job_dir,
                '--save=%s' % self.job_dir,
                '--snapshotPrefix=%s' % self.snapshot_prefix,
                '--snapshotInterval=%s' % self.snapshot_interval,
                '--lr_base_rate=%s' % self.learning_rate,
                '--lr_policy=%s' % str(self.lr_policy['policy'])
                ]

        if self.batch_size is not None:
            args.append('--batch_size=%d' % self.batch_size)

        if self.use_mean != 'none':
            mean_file = self.dataset.get_mean_file()
            assert mean_file is not None, 'Failed to retrieve mean file.'
            args.append('--mean=%s' % self.dataset.path(mean_file))

        if hasattr(self.dataset, 'labels_file'):
            args.append('--labels_list=%s' % self.dataset.path(self.dataset.labels_file))

        train_feature_db_path = self.dataset.get_feature_db_path(constants.TRAIN_DB)
        train_label_db_path = self.dataset.get_label_db_path(constants.TRAIN_DB)
        val_feature_db_path = self.dataset.get_feature_db_path(constants.VAL_DB)
        val_label_db_path = self.dataset.get_label_db_path(constants.VAL_DB)

        args.append('--train_db=%s' % train_feature_db_path)
        if train_label_db_path:
            args.append('--train_labels=%s' % train_label_db_path)
        if val_feature_db_path:
            args.append('--validation_db=%s' % val_feature_db_path)
        if val_label_db_path:
            args.append('--validation_labels=%s' % val_label_db_path)

        # learning rate policy input parameters
        if self.lr_policy['policy'] == 'fixed':
            pass
        elif self.lr_policy['policy'] == 'step':
            args.append('--lr_gamma=%s' % self.lr_policy['gamma'])
            args.append('--lr_stepvalues=%s' % self.lr_policy['stepsize'])
        elif self.lr_policy['policy'] == 'multistep':
            args.append('--lr_stepvalues=%s' % self.lr_policy['stepvalue'])
            args.append('--lr_gamma=%s' % self.lr_policy['gamma'])
        elif self.lr_policy['policy'] == 'exp':
            args.append('--lr_gamma=%s' % self.lr_policy['gamma'])
        elif self.lr_policy['policy'] == 'inv':
            args.append('--lr_gamma=%s' % self.lr_policy['gamma'])
            args.append('--lr_power=%s' % self.lr_policy['power'])
        elif self.lr_policy['policy'] == 'poly':
            args.append('--lr_power=%s' % self.lr_policy['power'])
        elif self.lr_policy['policy'] == 'sigmoid':
            args.append('--lr_stepvalues=%s' % self.lr_policy['stepsize'])
            args.append('--lr_gamma=%s' % self.lr_policy['gamma'])

        if self.shuffle:
            args.append('--shuffle=1')

        if self.crop_size:
            args.append('--croplen=%d' % self.crop_size)

        if self.use_mean == 'pixel':
            args.append('--subtractMean=pixel')
        elif self.use_mean == 'image':
            args.append('--subtractMean=image')
        else:
            args.append('--subtractMean=none')

        if self.random_seed is not None:
            args.append('--seed=%s' % self.random_seed)

        if self.solver_type == 'SGD':
            args.append('--optimization=sgd')
        elif self.solver_type == 'ADADELTA':
            args.append('--optimization=adadelta')
        elif self.solver_type == 'ADAGRAD':
            args.append('--optimization=adagrad')
        elif self.solver_type == 'ADAGRADDA':
            args.append('--optimization=adagradda')
        elif self.solver_type == 'MOMENTUM':
            args.append('--optimization=momentum')
        elif self.solver_type == 'ADAM':
            args.append('--optimization=adam')
        elif self.solver_type == 'FTRL':
            args.append('--optimization=ftrl')
        elif self.solver_type == 'RMSPROP':
            args.append('--optimization=rmsprop')
        else:
            raise ValueError('Unknown solver_type %s' % self.solver_type)

        if self.val_interval is not None:
            args.append('--validation_interval=%d' % self.val_interval)

        # if self.traces_interval is not None:
        args.append('--log_runtime_stats_per_step=%d' % self.traces_interval)

        if 'gpus' in resources:
            identifiers = []
            for identifier, value in resources['gpus']:
                identifiers.append(identifier)
            # make all selected GPUs visible to the process.
            # don't make other GPUs visible though since the process will load
            # CUDA libraries and allocate memory on all visible GPUs by
            # default.
            env['CUDA_VISIBLE_DEVICES'] = subprocess_visible_devices(identifiers)

        if self.pretrained_model:
            args.append('--weights=%s' % self.path(self.pretrained_model))

        # Augmentations
        assert self.data_aug['flip'] in ['none', 'fliplr', 'flipud', 'fliplrud'], 'Bad or unknown flag "flip"'
        args.append('--augFlip=%s' % self.data_aug['flip'])

        if self.data_aug['noise']:
            args.append('--augNoise=%s' % self.data_aug['noise'])

        if self.data_aug['contrast']:
            args.append('--augContrast=%s' % self.data_aug['contrast'])

        if self.data_aug['whitening']:
            args.append('--augWhitening=1')

        if self.data_aug['hsv_use']:
            args.append('--augHSVh=%s' % self.data_aug['hsv_h'])
            args.append('--augHSVs=%s' % self.data_aug['hsv_s'])
            args.append('--augHSVv=%s' % self.data_aug['hsv_v'])
        else:
            args.append('--augHSVh=0')
            args.append('--augHSVs=0')
            args.append('--augHSVv=0')

        return args

    @override
    def process_output(self, line):
        self.tensorflow_log.write('%s\n' % line)
        self.tensorflow_log.flush()

        # parse tensorflow output
        timestamp, level, message = self.preprocess_output_tensorflow(line)

        # return false when unrecognized output is encountered
        if not level:
            # network display in progress
            if self.displaying_network:
                self.temp_unrecognized_output.append(line)
                return True
            return False

        if not message:
            return True

        # network display ends
        if self.displaying_network:
            if message.startswith('Network definition ends'):
                self.temp_unrecognized_output = []
                self.displaying_network = False
            return True

        # Distinguish between a Validation and Training stage epoch
        pattern_stage_epoch = re.compile(r'(Validation|Training)\ \(\w+\ ([^\ ]+)\)\:\ (.*)')
        for (stage, epoch, kvlist) in re.findall(pattern_stage_epoch, message):
            epoch = float(epoch)
            self.send_progress_update(epoch)
            pattern_key_val = re.compile(r'([\w\-_]+)\ =\ ([^,^\ ]+)')
            # Now iterate through the keys and values on this line dynamically
            for (key, value) in re.findall(pattern_key_val, kvlist):
                assert not('Inf' in value or 'NaN' in value), 'Network reported %s for %s.' % (value, key)
                value = float(value)
                if key == 'lr':
                    key = 'learning_rate'  # Convert to special DIGITS key for learning rate
                if stage == 'Training':
                    self.save_train_output(key, key, value)
                elif stage == 'Validation':
                    self.save_val_output(key, key, value)
                    self.logger.debug('Network validation %s #%s: %s' % (key, epoch, value))
                else:
                    self.logger.error('Unknown stage found other than training or validation: %s' % (stage))
            self.logger.debug(message)
            return True

        # timeline trace saved
        if message.startswith('Timeline trace written to'):
            self.logger.info(message)
            self.detect_timeline_traces()
            return True

        # snapshot saved
        if self.saving_snapshot:
            if message.startswith('Snapshot saved'):
                self.logger.info(message)
            self.detect_snapshots()
            self.send_snapshot_update()
            self.saving_snapshot = False
            return True

        # snapshot starting
        match = re.match(r'Snapshotting to (.*)\s*$', message)
        if match:
            self.saving_snapshot = True
            return True

        # network display starting
        if message.startswith('Network definition:'):
            self.displaying_network = True
            return True

        if level in ['error', 'critical']:
            self.logger.error('%s: %s' % (self.name(), message))
            self.exception = message
            return True

        # skip remaining info and warn messages
        return True

    @staticmethod
    def preprocess_output_tensorflow(line):
        """
        Takes line of output and parses it according to tensorflow's output format
        Returns (timestamp, level, message) or (None, None, None)
        """
        # NOTE: This must change when the logging format changes
        # LMMDD HH:MM:SS.MICROS pid file:lineno] message
        match = re.match(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s\[(\w+)\s*]\s+(\S.*)$', line)
        if match:
            timestamp = time.mktime(time.strptime(match.group(1), '%Y-%m-%d %H:%M:%S'))
            level = match.group(2)
            message = match.group(3)
            if level == 'INFO':
                level = 'info'
            elif level == 'WARNING':
                level = 'warning'
            elif level == 'ERROR':
                level = 'error'
            elif level == 'FAIL':  # FAIL
                level = 'critical'
            return (timestamp, level, message)
        else:
            # self.logger.warning('Unrecognized task output "%s"' % line)
            return (None, None, None)

    def send_snapshot_update(self):
        """
        Sends socketio message about the snapshot list
        """
        # TODO: move to TrainTask
        from web.webapp import socketio

        socketio.emit('task update', {'task': self.html_id(),
                                      'update': 'snapshots',
                                      'data': self.snapshot_list()},
                      namespace='/jobs',
                      room=self.job_id)

    # TrainTask overrides
    @override
    def after_run(self):
        if self.temp_unrecognized_output:
            if self.traceback:
                self.traceback = self.traceback + ('\n'.join(self.temp_unrecognized_output))
            else:
                self.traceback = '\n'.join(self.temp_unrecognized_output)
                self.temp_unrecognized_output = []
        self.tensorflow_log.close()

    @override
    def after_runtime_error(self):
        if os.path.exists(self.path(self.TENSORFLOW_LOG)):
            output = subprocess.check_output(['tail', '-n40', self.path(self.TENSORFLOW_LOG)])
            lines = []
            for line in output.split('\n'):
                # parse tensorflow header
                timestamp, level, message = self.preprocess_output_tensorflow(line)

                if message:
                    lines.append(message)
            # return the last 20 lines
            traceback = '\n\nLast output:\n' + '\n'.join(lines[len(lines)-20:]) if len(lines) > 0 else ''
            if self.traceback:
                self.traceback = self.traceback + traceback
            else:
                self.traceback = traceback

    @override
    def detect_timeline_traces(self):
        timeline_traces = []
        for filename in os.listdir(self.job_dir):
            # find timeline jsons
            match = re.match(r'%s_(.*)\.json$' % TIMELINE_PREFIX, filename)
            if match:
                step = int(match.group(1))
                timeline_traces.append((os.path.join(self.job_dir, filename), step))
        self.timeline_traces = sorted(timeline_traces, key=lambda tup: tup[1])
        return len(self.timeline_traces) > 0

    @override
    def detect_snapshots(self):
        self.snapshots = []
        snapshots = []
        for filename in os.listdir(self.job_dir):
            # find models
            match = re.match(r'%s_(\d+)\.?(\d*)\.ckpt\.index$' % self.snapshot_prefix, filename)
            if match:
                epoch = 0
                # remove '.index' suffix from filename
                filename = filename[:-6]
                if match.group(2) == '':
                    epoch = int(match.group(1))
                else:
                    epoch = float(match.group(1) + '.' + match.group(2))
                snapshots.append((os.path.join(self.job_dir, filename), epoch))
        self.snapshots = sorted(snapshots, key=lambda tup: tup[1])
        return len(self.snapshots) > 0

    @override
    def est_next_snapshot(self):
        # TODO: Currently this function is not in use. Probably in future we may have to implement this
        return None

    @override
    def infer_one(self,
                  data,
                  snapshot_epoch=None,
                  layers=None,
                  gpu=None,
                  resize=True):
        # resize parameter is unused
        return self.infer_one_image(data,
                                    snapshot_epoch=snapshot_epoch,
                                    layers=layers,
                                    gpu=gpu)

    def get_layer_statistics(self, data):
        """
        Returns statistics for the given layer data:
            (mean, standard deviation, histogram)
                histogram -- [y, x, ticks]
        Arguments:
        data -- a np.ndarray
        """
        # These calculations can be super slow
        mean = np.mean(data)
        std = np.std(data)
        y, x = np.histogram(data, bins=20)
        y = list(y)
        ticks = x[[0, len(x)/2, -1]]
        x = [(x[i]+x[i+1])/2.0 for i in range(len(x)-1)]
        ticks = list(ticks)
        return (mean, std, [y, x, ticks])

    def after_test_run(self, temp_image_path):
        try:
            os.remove(temp_image_path)
        except OSError:
            pass

    @override
    def infer_many(self, data, snapshot_epoch=None, gpu=None, resize=True):
        # resize parameter is unused
        return self.infer_many_images(data, snapshot_epoch=snapshot_epoch, gpu=gpu)

    def has_model(self):
        """
        Returns True if there is a model that can be used
        """
        return len(self.snapshots) != 0

    @override
    def get_model_files(self):
        """
        return paths to model files
        """
        return {"Network": self.model_file}

    @override
    def get_network_desc(self):
        """
        return text description of network
        """
        with open(os.path.join(self.job_dir, TENSORFLOW_MODEL_FILE), "r") as infile:
            desc = infile.read()
        return desc

    @override
    def get_task_stats(self, epoch=-1):
        """
        return a dictionary of task statistics
        """

        loc, mean_file = os.path.split(self.dataset.get_mean_file())

        stats = {
            "image dimensions": self.dataset.get_feature_dims(),
            "mean file": mean_file,
            "snapshot file": self.get_snapshot_filename(epoch),
            "model file": self.model_file,
            "framework": "tensorflow",
            "mean subtraction": self.use_mean
        }

        if hasattr(self, "digits_version"):
            stats.update({"digits version": self.digits_version})

        if hasattr(self.dataset, "resize_mode"):
            stats.update({"image resize mode": self.dataset.resize_mode})

        if hasattr(self.dataset, "labels_file"):
            stats.update({"labels file": self.dataset.labels_file})

        return stats