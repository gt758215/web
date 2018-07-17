# Copyright (c) 2018, ITRI.  All rights reserved.
from __future__ import absolute_import


import logging

import flask
import werkzeug.exceptions
import pprint
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

    name = form.name.data
    dataset_id = form.selected_dataset.data
    model_id = form.selected_model.data

    dataset = scheduler.get_job(dataset_id)
    model = scheduler.get_job(model_id)

    # TODO: we should add db in dataset choice later
    job = None
    try:
        # TODO: weshould change username backto authorized user name
        job = EvaluationJob(username='demo',
                            name=name,
                            model=model,
                            dataset=dataset)
            # Save form data with the job so we can easily clone it later.

        save_form_to_job(job, form)

        scheduler.add_job(job)

        if request_wants_json():
            return flask.jsonify(job.json_dict())
        else:
            return flask.redirect(flask.url_for('digits.evaluation.views.home'))

            #return flask.redirect(flask.url_for('digits.evaluation.views.show', job_id=job.id()))

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

    #related_jobs = scheduler.get_related_jobs(job)

    job_id = job.id()
    confusion_matrix = {}
    image_prediction_list = {}
    #with open(job.evaluation_task().confusion_matrix_path, 'r') as cm_file:
    with open('/home/weiru/PycharmProjects/web/digits/jobs/%s/confusion_matrix.json' % job.id(), 'r') as cm_file:
        try:
            confusion_matrix = flask.json.load(cm_file)
        except Exception as e:
            raise werkzeug.exceptions.NotFound('Confusion_matrix file not found', e)
    '''
    with open(job.evaluation_task().image_prediction_list_path, 'r') as ipl_file:
        try:
            image_prediction_list = flask.json.load(ipl_file)
        except Exception as e:
            raise werkzeug.exceptions.NotFound('Confusion_matrix file not foun', e)
    '''
    if request_wants_json():
        return flask.jsonify(job.json_dict(True))
    else:
        return flask.render_template('mlt/evaluation/show.html',
                                     job=job,
                                     job_id='test32',
                                     labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     confusion_matrix=confusion_matrix['confusion_matrix'],
                                     )


@blueprint.route('/<job_id>/editdataset/<label_x>/<label_y>')
def edit_dataset(job_id, label_x, label_y):
    pass


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