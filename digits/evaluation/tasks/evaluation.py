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


@subclass
class EvaluationTask(Task):
    """
    A task for inference jobs
    """

    def __init__(self, model, dataset, dataset_dbs, epoch=None, **kwargs):
        """
        Arguments:
        model  -- trained model to perform inference on
        images -- list of images to perform inference on, or path to a database
        """
        # memorize parameters
        self.model = model
        self.dataset = dataset
        self.dataset_dbs = dataset_dbs
        self.epoch = epoch


        self.evaluation_log_file = "evaluation.log"
        self.evaluation_log = None

        # resources
        self.gpu = None

        # generated data
        self.evaluation_result_filename = None

        self.inference_data_filename = None
        self.inference_inputs = None
        self.inference_outputs = None
        self.inference_layers = []


        super(EvaluationTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Evaluate Data'

    @override
    def __getstate__(self):
        state = super(EvaluationTask, self).__getstate__()
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

        # path to inference data
        match = re.match(r'Saved data to (.*)', message)
        if match:
            self.inference_data_filename = match.group(1).strip()
            return True

        return False

    @override
    def after_run(self):
        super(EvaluationTask, self).after_run()

        # retrieve inference data
        visualizations = []
        outputs = OrderedDict()
        if self.inference_data_filename is not None:
            # the HDF5 database contains:
            # - input images, in a dataset "/inputs"
            # - all network outputs, in a group "/outputs/"
            # - layer activations and weights, if requested, in a group "/layers/"
            db = h5py.File(self.inference_data_filename, 'r')

            # collect paths and data
            input_ids = db['input_ids'][...]
            input_data = db['input_data'][...]

            # collect outputs
            o = []
            for output_key, output_data in db['outputs'].items():
                output_name = base64.urlsafe_b64decode(str(output_key))
                o.append({'id': output_data.attrs['id'], 'name': output_name, 'data': output_data[...]})
            # sort outputs by ID
            o = sorted(o, key=lambda x: x['id'])
            # retain only data (using name as key)
            for output in o:
                outputs[output['name']] = output['data']

            # collect layer data, if applicable
            if 'layers' in db.keys():
                for layer_id, layer in db['layers'].items():
                    visualization = {
                        'id': int(layer_id),
                        'name': layer.attrs['name'],
                        'vis_type': layer.attrs['vis_type'],
                        'data_stats': {
                            'shape': layer.attrs['shape'],
                            'mean': layer.attrs['mean'],
                            'stddev': layer.attrs['stddev'],
                            'histogram': [
                                layer.attrs['histogram_y'].tolist(),
                                layer.attrs['histogram_x'].tolist(),
                                layer.attrs['histogram_ticks'].tolist(),
                            ]
                        }
                    }
                    if 'param_count' in layer.attrs:
                        visualization['param_count'] = layer.attrs['param_count']
                    if 'layer_type' in layer.attrs:
                        visualization['layer_type'] = layer.attrs['layer_type']
                    vis = layer[...]
                    if vis.shape[0] > 0:
                        visualization['image_html'] = embed_image_html(vis)
                    visualizations.append(visualization)
                # sort by layer ID (as HDF5 ASCII sorts)
                visualizations = sorted(visualizations, key=lambda x: x['id'])
            db.close()
            # save inference data for further use
            self.inference_inputs = {'ids': input_ids, 'data': input_data}
            self.inference_outputs = outputs
            self.inference_layers = visualizations
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
                self.image_list_path if self.image_list_path is not None else self.images,
                self.job_dir,
                self.model.id(),
                '--jobs_dir=%s' % digits.config.config_value('jobs_dir'),
                ]

        if self.epoch is not None:
            args.append('--epoch=%s' % repr(self.epoch))

        if self.gpu is not None:
            args.append('--gpu=%d' % self.gpu)

        if self.image_list_path is None:
            args.append('--db')

        if not self.resize:
            args.append('--no-resize')

        return args
