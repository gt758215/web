# Copyright (c) 2018, ITRI.  All rights reserved.
from __future__ import absolute_import


import logging

import flask
import werkzeug.exceptions

from . import images as model_images
from . import EvaluationJob
from .tasks import EvaluationTask
from .forms import EvaluationForm
from digits.utils.routing import request_wants_json
from digits.webapp import scheduler
from digits.views import get_job_list
from digits.model import ImageClassificationModelJob
from digits.dataset import ImageClassificationDatasetJob
from digits.status import Status
from digits.utils.forms import fill_form_if_cloned, save_form_to_job

blueprint = flask.Blueprint(__name__, __name__)

logger = logging.getLogger('digits.evaluation.views')


@blueprint.route('/', methods=['GET'])
def home(tab=2):
    running_evaluations = get_job_list(EvaluationJob, True)
    completed_evaluations = get_job_list(EvaluationJob, False)

    new_dataset_options = {
        'Images': {
            'image-classification': {
                'title': 'Classification',
                'url': flask.url_for(
                    'digits.dataset.images.classification.views.new'),
            },
            'image-other': {
                'title': 'Other',
                'url': flask.url_for(
                    'digits.dataset.images.generic.views.new'),
            },
        },
    }

    return flask.render_template(
        'mlt/evaluation/summary.html',
        tab=tab,
        new_dataset_options=new_dataset_options,
        running_evaluations=running_evaluations,
        completed_evaluations=completed_evaluations,
    )


@blueprint.route('/new', methods=['GET'])
def new():

    form = EvaluationForm()
    form.selected_model.choices = get_models()
    form.selected_dataset.choices = get_datasets()

    return flask.render_template(
        'mlt/evaluation/new.html',
        form=form
    );
    pass


@blueprint.route('.json', methods=['POST'])
@blueprint.route('', methods=['POST'])
def create():
    form = EvaluationForm()
    form.selected_model.choices = get_models()
    form.selected_dataset.choices = get_datasets()

    dataset_id = form.selected_dataset.data
    model_id = form.selected_model.data

    dataset = scheduler.get_job(dataset_id)
    model = scheduler.get_job(model_id)

    # we should add db in dataset choice later
    try:
        job = EvaluationJob(model=model, dataset=dataset)
            # Save form data with the job so we can easily clone it later.

        save_form_to_job(job, form)

        scheduler.add_job(job)

        if request_wants_json():
            return flask.jsonify(job.json_dict())
        else:
            return flask.redirect(flask.url_for('digits.model.views.show', job_id=job.id()))

    except Exception:
        if job:
            scheduler.delete_job(job)
        raise


@blueprint.route('/<job_id>.json', methods=['GET'])
@blueprint.route('/<job_id>', methods=['GET'])
def show(job_id):
    """
    Show a EvaluationJob

    Returns JSON when requested:
        {id, name, directory, status, snapshots: [epoch,epoch,...]}
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    related_jobs = scheduler.get_related_jobs(job)

    if request_wants_json():
        return flask.jsonify(job.json_dict(True))
    else:
        if isinstance(job, model_images.ImageClassificationModelJob):
            return model_images.classification.views.show(job, related_jobs=related_jobs)
        elif isinstance(job, model_images.GenericImageModelJob):
            return model_images.generic.views.show(job, related_jobs=related_jobs)
        else:
            raise werkzeug.exceptions.BadRequest(
                'Invalid job type')


def get_models():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, ImageClassificationModelJob) and
         (j.status.is_running() or j.status == Status.DONE)],
        cmp=lambda x, y: cmp(y.id(), x.id())
    )
    ]


def get_datasets():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, ImageClassificationDatasetJob) and
         (j.status.is_running() or j.status == Status.DONE)],
        cmp=lambda x, y: cmp(y.id(), x.id())
    )
    ]