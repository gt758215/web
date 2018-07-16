# Copyright (c) 2018, ITRI.  All rights reserved.
from __future__ import absolute_import

from . import tasks
import digits.frameworks
from digits.job import Job
from digits.utils import subclass, override

PICKLE_VERSION=1

@subclass
class EvaluationJob(Job):
    """
    A Job that exercises the forward pass of a neural network
    """
    def __init__(self, model, dataset, dataset_db=None, **kwargs):
        """

        :param model: job object associated with model to perform evaluation on
        :param images: list of image path to perform evaluation on
        """
        super(EvaluationJob, self).__init__(**kwargs)
        self.pickver_job_evaluation = PICKLE_VERSION

        fw_id = model.train_task().framework_id
        fw = digits.frameworks.get_framework_by_id(fw_id)

        self.dataset = dataset
        self.model = model

        if fw is None:
            raise RuntimeError(
                'The "%s" framework cannot be found. Check your server configuration.'
                % fw_id)

        self.tasks.append(fw.create_evaluation_task(
            job_dir=self.dir(),
            model=model,
            dataset=dataset,
            dataset_db=dataset_db,
        ))

    @override
    def __getstate__(self):

        full_state = super(EvaluationJob, self).__getstate__()
        state_to_save = {}

        for field in full_state:
            if field in ['dataset', 'model'] and isinstance(full_state[field], Job):
                id = full_state[field].id()
                del full_state[field]
                full_state[field] = id
            state_to_save[field] = full_state[field]
        return state_to_save

    def evaluation_task(self):
        """Return the first and only evaluation task for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.EvaluationTask)][0]

    @override
    def __setstate__(self, state):
        super(EvaluationJob, self).__setstate__(state)

    def load_dataset(self):
        from digits.webapp import scheduler
        job = scheduler.get_job(self.dataset)
        assert job is not None, 'Cannot find dataset'
        self.dataset = job

        for task in self.tasks:
            task.dataset = job


    def load_model(self):
        from digits.webapp import scheduler
        job = scheduler.get_job(self.model)
        assert job is not None, 'Cannot find dataset'
        self.model = job

        for task in self.tasks:
            task.model = job


    def get_data(self):
        """Return evaluation data"""
        task = self.evaluation_task()
        return task.inference_inputs, task.inference_outputs, task.inference_layers

    def job_type(self):
        return 'evaluation'

    @override
    def json_dict(self, detailed=False):
        val = super(EvaluationJob, self).json_dict(detailed=detailed)
        if detailed:
            val.update({
                'dataset': {'name': self.dataset.name(), 'id': self.dataset.id()},
                'model': {'name': self.model.name(), 'id': self.model.id()},
            })
        return val
