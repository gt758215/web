from __future__ import absolute_import

import os
from os import listdir

import werkzeug.exceptions
from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from os.path import isdir, join

from web.datasets import DatasetJob
from web.webapp import scheduler
from . import images as dataset_images
from web import datasets

blueprint = Blueprint(__name__, __name__)


@blueprint.route('/<job_id>', methods=['GET'])
def show(job_id):
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')
    related_jobs = scheduler.get_related_jobs(job)
    if isinstance(job, dataset_images.ImageDatasetJob):
        return dataset_images.views.show(job, related_jobs=related_jobs)
    else:
        raise werkzeug.exceptions.BadRequest('Invalid job type')


@blueprint.route('/', methods=['GET'])
def home(tab=2):
    completed_datasets = get_job_list(datasets.DatasetJob, False)
    return render_template('datasets/summary.html',
                           tab=tab,
                           completed_datasets = completed_datasets)


@blueprint.route('/summary', methods=['GET'])
def summary():
    try:
        datasets = []
        dataset_folders = get_dataset_list()
        for folder in dataset_folders:
            size = get_size(join('/data/datasets', folder))
            dataset = dict()
            dataset['name'] = folder
            dataset['bytes'] = size
            datasets.append(dataset)
        return render_template('datasets/datasets.html', datasets=datasets)
    except TemplateNotFound:
        abort(404)


def get_size(the_path):
    """Get size of a directory tree or a file in bytes."""
    path_size = 0
    for path, directories, files in os.walk(the_path):
        for filename in files:
            path_size += os.lstat(os.path.join(path, filename)).st_size
    path_size += os.path.getsize(the_path)
    return path_size


def get_dataset_list():
    return sorted([d for d in listdir('/data/datasets') if isdir(join('/data/datasets', d))])


def get_job_list(cls, running):
    return sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, cls) and j.status.is_running() == running],
        key=lambda j: j.status_history[0][1],
        reverse=True,
    )