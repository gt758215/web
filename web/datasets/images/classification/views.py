import os
import shutil
import flask
from flask import Blueprint, render_template
from .forms import ImageClassificationDatasetForm
from web.datasets import DatasetJob
from web import utils
from web.datasets.images.forms import ImageDatasetForm
from web.datasets.images.job import ImageDatasetJob
blueprint = Blueprint(__name__, __name__)


@blueprint.route('/', methods=['GET'])
def summary():
    try:
        running_datasets = get_job_list(DatasetJob, True)
        return render_template('datasets.html', running_datasets=running_datasets)
    except TemplateNotFound:
        abort(404)


def get_job_list(cls, running):
    return sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, cls) and j.status.is_running() == running],
        key=lambda j: j.status_history[0][1],
        reverse=True,
    )


@blueprint.route('/new', methods=['GET'])
def new():
    form = ImageClassificationDatasetForm()

    if not form.validate_on_submit():
        return render_template('datasets/images/classification/new.html', form=form), 400