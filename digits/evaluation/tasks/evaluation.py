# Copyright (c) 2018, ITRI.  All rights reserved.
from __future__ import absolute_import

import base64
from collections import OrderedDict
import h5py
import os.path
import tempfile
import re
import sys

import digits
from digits.task import Task
from digits.utils import subclass, override
from digits.utils.image import embed_image_html


PICKLE_VERSION = 1
CONFUSION_MATRIX_FILENAME = 'confusion_matrix.json'
IMAGE_PREDICTION_LIST_FILENAME = 'image_prediction_list.json'
LABELS_FILENAME = 'labels.txt'

@subclass
class EvaluationTask(Task):
    """
    A task for inference jobs
    """

    def __init__(self, model, dataset, dataset_db=None, batch_size=100, **kwargs):
        """
        Arguments:
        model  -- trained model to perform inference on
        images -- list of images to perform inference on, or path to a database
        """
        super(EvaluationTask, self).__init__(**kwargs)
        self.pickver_task_evaluation = PICKLE_VERSION

        # memorize parameters
        self.model = model
        self.dataset = dataset
        self.epoch = None

        # infer.py parameters
        if dataset_db is None:
            # default use training set
            self.data_dir = dataset.train_db_task().path('train_db')
            self.filename_pattern = "train-*"
        elif 'test' in dataset_db.to_lower():
            self.data_dir = dataset.train_db_task().path('test_db')
            self.filename_pattern = "test-*"
        elif 'train' in dataset_db.to_lower():
            self.data_dir = dataset.train_db_task().path('train_db')
            self.filename_pattern = "train-*"
        elif 'val' in dataset_db.to_lower():
            self.data_dir = dataset.train_db_task().path('val_db')
            self.filename_pattern = "validation-*"
        else:
            # default use training set
            self.data_dir = dataset.train_db_task().path('train_db')
            self.filename_pattern = "train-*"

        self.network = "network.py"
        self.networkDirectory = model.dir()
        self.batch_size = batch_size
        self.labels_list = '%s/labels.txt' % dataset.dir()

        self.device = None
        self.train_dir = model.dir()
        self.gen_metrics = True
        self.data_format = None

        # process log
        self.evaluation_log_file = "evaluation.log"
        self.evaluation_log = None

        # resources
        self.gpu = None

        self.confusion_matrix_path = None
        self.image_prediction_list_path = None

    def confusion_matrix_path(self):
        return os.path.join(self.job_dir, CONFUSION_MATRIX_FILENAME)

    def image_prediction_list_path(self):
        return os.path.join(self.job_dir, IMAGE_PREDICTION_LIST_FILENAME)

    def labels_path(self):
        return os.path.join(self.dataset.dir(), LABELS_FILENAME)

    @override
    def name(self):
        return 'Evaluate Data'

    @override
    def __getstate__(self):
        state = super(EvaluationTask, self).__getstate__()
        if 'dataset' in state:
            del state['dataset']
        if 'model' in state:
            del state['model']
        if 'evaluation_log' in state:
            del state['evaluation_log']
        return state

    @override
    def __setstate__(self, state):
        super(EvaluationTask, self).__setstate__(state)


    @override
    def before_run(self):
        super(EvaluationTask, self).before_run()

        # create log file
        self.evaluation_log = open(self.path(self.evaluation_log_file), 'a')



    @override
    def process_output(self, line):
        self.evaluation_log.write('%s\n' % line)
        self.evaluation_log.flush()

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        # progress
        match = re.match(r'Processed (\d+)\/(\d+)', message)
        if match:
            self.progress = float(match.group(1)) / int(match.group(2))
            return True

        # path to confusion_matrix
        match = re.match(r'Saved confusion_matrix to (.*)', message)
        if match:
            self.confusion_matrix_path = match.group(1).strip()
            return True

        # path to image_prediction_list
        match = re.match(r'Saved image_prediction_list to (.*)', message)
        if match:
            self.image_prediction_list_path = match.group(1).strip()
            return True

        return False

    @override
    def after_run(self):
        super(EvaluationTask, self).after_run()

        self.evaluation_log.write("Confusion Matrix generated %s" % self.confusion_matrix_path())
        self.evaluation_log.write("Image Prediction List generated %s" % self.image_prediction_list_path())
        self.evaluation_log.close()

    @override
    def offer_resources(self, resources):
        reserved_resources = {}
        # we need one CPU resource from inference_task_pool
        cpu_key = 'inference_task_pool'
        if cpu_key not in resources:
            return None
        for resource in resources[cpu_key]:
            if resource.remaining() >= 1:
                reserved_resources[cpu_key] = [(resource.identifier, 1)]
                # we reserve the first available GPU, if there are any
                gpu_key = 'gpus'
                if resources[gpu_key]:
                    for resource in resources[gpu_key]:
                        if resource.remaining() >= 1:
                            self.gpu = int(resource.identifier)
                            reserved_resources[gpu_key] = [(resource.identifier, 1)]
                            break
                return reserved_resources
        return None

    @override
    def task_arguments(self, resources, env):

        args = [sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(digits.__file__)), 'tools', 'tf', 'infer.py'),
                '--jobs_dir=%s' % digits.config.config_value('jobs_dir'),
                ]

        if self.epoch is not None:
            args.append('--epoch=%s' % repr(self.epoch))

        if self.gpu is not None:
            args.append('--gpu=%d' % self.gpu)

        if self.data_dir is not None:
            args.append('--data_dir=%s' % self.data_dir)

        if self.filename_pattern is not None:
            args.append('--filename=%s' % self.filename_pattern)

        if self.network is not None:
            args.append('--network=%s' % self.network)

        if self.networkDirectory is not None:
            args.append('--networkDirectory=%s' % self.networkDirectory)

        if self.job_dir is not None:
            args.append('--result_dir=%s' % self.job_dir)

        if self.batch_size is not None:
            args.append('--batch_size=%s' % self.batch_size)

        if self.dataset is not None:
            args.append('--labels_list=%s' % self.labels_path())

        if self.device is not None:
            args.append('--device=%s' % self.device)

        if self.train_dir is not None:
            args.append('--train_dir=%s' % self.train_dir)

        if self.gen_metrics is not None:
            args.append('--gen_metrics=%s' % self.gen_metrics)

        if self.gen_metrics is not None:
            args.append('--gen_metrics=%s' % self.gen_metrics)

        return args
