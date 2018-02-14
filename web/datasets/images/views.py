from __future__ import absolute_import

import os
from flask import Blueprint, render_template, abort, redirect, url_for
from jinja2 import TemplateNotFound
from web.datasets.images.forms import ImageDatasetForm
from web.datasets.images.job import ImageDatasetJob
from web.utils.forms import save_form_to_job
from web import utils
from web.webapp import scheduler
from web.datasets import tasks

blueprint = Blueprint(__name__, __name__)


def from_files(job, form):
    labels_file_from = form.textfile_local_labels_file.data.strip()
    labels_file_to = os.path.join(job.dir(), not utils.constants.LABELS_FILE)
    job.labels_file = utils.constants.LABELS_FILE
    # train
    train_file = form.textfile_train_images.data.strip()
    image_folder = form.textfile_train_folder.data.strip()
    if not image_folder:
        image_folder = None
    job.tasks.append(
        tasks.CopyFolder(
            job_dir=job.dir(),
            input_file=train_file,
            image_folder=image_folder,
            labels_file=job.labels_file,
        )
    )


@blueprint.route('/new', methods=['GET'])
def new():
    try:
        form = ImageDatasetForm()
        # if not form.validate_on_submit():
        return render_template('datasets/images/new.html', form=form)
    except TemplateNotFound:
        abort(404)


@blueprint.route('', methods=['POST'], strict_slashes=False)
def create():
    form = ImageDatasetForm
    job = None
    try:
        job = ImageDatasetJob(
            name=form.dataset_name.data,
        )
        from_files(job, form)
        save_form_to_job(job, form)

        scheduler.add_job(job)
        return redirect(url_for('web.dataset.views.show', job_id=job.id()))
    except:
        raise


def show(job, related_jobs=None):
    """
    Called from digits.dataset.views.datasets_show()
    """
    return render_template('datasets/images/show.html', job=job, related_jobs=related_jobs)