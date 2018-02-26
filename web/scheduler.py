from collections import OrderedDict
import os
import gevent.event
import time
import shutil

from .config import config_value
from .datasets import DatasetJob
from .job import Job
from .log import logger
from web.models import ModelJob
from .pretrained_model import PretrainedModelJob
from .status import Status

class Resource(object):
    def __init__(self, identifier=None, max_value=1):
        if identifier is None:
            self.identifier = id(self)
        else:
            self.identifier = identifier
        self.max_value = max_value
        self.allocations = []


class Scheduler:
    def __init__(self, gpu_list=None, verbose=False):
        self.jobs = OrderedDict()
        self.verbose = verbose

        # Keeps track of resource usage
        self.resources = {
            # TODO: break this into CPU cores, memory usage, IO usage, etc.
            'parse_folder_task_pool': [Resource()],
            'create_db_task_pool': [Resource(max_value=4)],
            'analyze_db_task_pool': [Resource(max_value=4)],
            'inference_task_pool': [Resource(max_value=4)],
            'gpus': [Resource(identifier=index)
                     for index in gpu_list.split(',')] if gpu_list else [],
        }

        self.running = False
        self.shutdown = gevent.event.Event()

    def add_job(self, job):
        if not self.running:
            logger.error('Scheduler not running. Cannot add job.')
            return False
        else:
            self.jobs[job.id()] = job
            from web.webapp import app, socketio
            with app.app_context():
                socketio.emit('job update',
                              {
                                  'update': 'added',
                                  'job_id': job.id()
                              },
                              namespace = '/jobs',
                              room = 'job_management',
                              )
            # Let the scheduler do a little work before returning
            time.sleep(0.5)
            return True

    def delete_job(self, job):
        if isinstance(job, str) or isinstance(job, unicode):
            job_id = str(job)
        elif isinstance(job, Job):
            job_id = job.id()
        else:
            raise ValueError('called delete_job with a %s' % type(job))
        job = self.jobs.get(job_id, None)
        if job:
            # TODO check dependencies
            self.jobs.pop(job_id, None)
            job.abort()
            if os.path.exists(job.dir()):
                shutil.rmtree(job.dir())
            logger.info('Job deleted.', job_id=job_id)
            from web.webapp import socketio
            socketio.emit('job update',
                          {
                              'update': 'deleted',
                              'job_id': job.id()
                          },
                          namespace = '/jobs',
                          room = 'job_management',
                          )
            return True
        path = os.path.join(config_value('jobs_dir'), job_id)
        path = os.path.normpath(path)
        if os.path.dirname(path) == config_value('jobs_dir') and os.path.exists(path):
            shutil.rmtree(path)
            return True
        return False

    def load_past_jobs(self):
        """
                Look in the jobs directory and load all valid jobs
                """
        loaded_jobs = []
        failed_jobs = []
        for dir_name in sorted(os.listdir(config_value('jobs_dir'))):
            if os.path.isdir(os.path.join(config_value('jobs_dir'), dir_name)):
                # Make sure it hasn't already been loaded
                if dir_name in self.jobs:
                    continue

                try:
                    job = Job.load(dir_name)
                    # The server might have crashed
                    if job.status.is_running():
                        job.status = Status.ABORT
                    for task in job.tasks:
                        if task.status.is_running():
                            task.status = Status.ABORT

                    # We might have changed some attributes here or in __setstate__
                    job.save()
                    loaded_jobs.append(job)
                except Exception as e:
                    failed_jobs.append((dir_name, e))

        # add DatasetJobs or PretrainedModelJobs
        for job in loaded_jobs:
            if isinstance(job, DatasetJob) or isinstance(job, PretrainedModelJob):
                self.jobs[job.id()] = job

        # add ModelJobs
        for job in loaded_jobs:
            if isinstance(job, ModelJob):
                try:
                    # load the DatasetJob
                    job.load_dataset()
                    self.jobs[job.id()] = job
                except Exception as e:
                    failed_jobs.append((dir_name, e))

        logger.info('Loaded %d jobs.' % len(self.jobs))

        if len(failed_jobs):
            logger.warning('Failed to load %d jobs.' % len(failed_jobs))
            if self.verbose:
                for job_id, e in failed_jobs:
                    logger.debug('%s - %s: %s' % (job_id, type(e).__name__, str(e)))

    def get_job(self, job_id):
        """
        Look through self.jobs to try to find the Job
        Returns None if not found
        """
        if job_id is None:
            return None
        return self.jobs.get(job_id, None)