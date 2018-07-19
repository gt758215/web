# Copyright (c) 2018, ITRI.  All rights reserved.
from __future__ import absolute_import


import logging

import flask
import werkzeug.exceptions
import PIL
import math
from . import EvaluationJob
from .forms import EvaluationForm
from digits.utils.routing import request_wants_json
from digits.webapp import scheduler
from digits.views import get_job_list
from digits.model import ImageClassificationModelJob
from digits.dataset import ImageClassificationDatasetJob
from digits.status import Status
from digits.utils.forms import save_form_to_job
from digits.utils.image import embed_image_html


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
            return flask.redirect(flask.url_for('digits.evaluation.views.show', job_id=job.id()))

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

    confusion_matrix_result = {}
    confusion_matrix = []
    labels = []
    precisions = []
    recalls = []
    precision = None
    recall = None
    accuracy = None

    if job.status_of_tasks() == Status.DONE:

        confusion_matrix_result = load_confusion_matrix(job)
        labels = load_labels(job)

        confusion_matrix = confusion_matrix_result['confusion_matrix']
        precisions = ['%.2f%%' % (float(x) * 100) for x in confusion_matrix_result['precision_list']]
        recalls = ['%.2f%%' % (float(x) * 100) for x in confusion_matrix_result['recall_list']]
        precision = '%.2f%%' % (float(confusion_matrix_result['precision']) * 100)
        recall = '%.2f%%' % (float(confusion_matrix_result['recall']) * 100)
        accuracy = '%.2f%%' % (float(confusion_matrix_result['accuracy']) * 100)

    if request_wants_json():
        return flask.jsonify(job.json_dict(True))
    else:
        return flask.render_template('mlt/evaluation/show.html',
                                     job=job,
                                     job_id=job.id(),
                                     labels=labels,
                                     confusion_matrix=confusion_matrix,
                                     precisions=precisions,
                                     recalls=recalls,
                                     precision=precision,
                                     recall=recall,
                                     accuracy=accuracy)


@blueprint.route('/<job_id>/explore/<label_fact>/<label_pred>')
def explore(job_id, label_fact, label_pred):
    label_fact = int(label_fact)
    label_pred = int(label_pred)
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if job.status_of_tasks() == Status.DONE:
        confusion_matrix_result = load_confusion_matrix(job)
        image_prediction_list = load_image_prediction_list(job)
        labels = load_labels(job)
        image_ids = confusion_matrix_result["confusion_matrix with ids"][label_fact][label_pred]
        image_count = confusion_matrix_result["confusion_matrix"][label_fact][label_pred]

        page = int(flask.request.args.get('page', 0))
        size = int(flask.request.args.get('size', 25))

        page_count = int(math.ceil(image_count / size))
        min_page = max(0, page - 5)
        max_page = min(page_count, page + 5)

        start = size * page
        end = start + size
        end = end if (end < image_count) else image_count

        images = []
        file_list = image_prediction_list["filename_list"]

        logger.info("show image from %d to %d, pid %d" % (start, end, page_count))
        for img_id in image_ids[start:end]:
            image_file_name = file_list[img_id]

            images.append({
                "label": labels[label_fact],
                "tf_img_id": img_id,
                "b64": embed_image_html(PIL.Image.open(image_file_name)),
            })

    return flask.render_template('mlt/evaluation/explore.html',
                                 job=job,
                                 job_id=job.id(),
                                 page=page,
                                 size=size,
                                 pages=range(min_page, max_page),
                                 first_page=0,
                                 last_page=page_count - 1,
                                 size_ops=[25, 50, 100],
                                 label_fact=labels[label_fact],
                                 label_pred=labels[label_pred],
                                 total_entries=image_count,
                                 images=images)


def load_confusion_matrix(job):
    confusion_matrix_result = {}
    with open(job.evaluation_task().confusion_matrix_path(), 'r') as cm_file:
        try:
            confusion_matrix_result = flask.json.load(cm_file)
        except Exception as e:
            raise werkzeug.exceptions.NotFound('Confusion_matrix file not found', e)

    return confusion_matrix_result


def load_image_prediction_list(job):
    image_prediction_list = {}
    with open(job.evaluation_task().image_prediction_list_path(), 'r') as ipl_file:
        try:
            image_prediction_list = flask.json.load(ipl_file)
        except Exception as e:
            raise werkzeug.exceptions.NotFound('Confusion_matrix file not foun', e)
    return image_prediction_list


def load_labels(job):
    labels = []
    with open(job.evaluation_task().labels_path(), 'r') as label_file:
        try:

            labels = label_file.readlines()
            labels = [x.strip() for x in labels]
        except Exception as e:
            raise werkzeug.exceptions.NotFound('label file not found', e)
    return labels


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
