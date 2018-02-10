import os
import shutil
import flask
from flask import Blueprint, render_template
from web import utils
from web.datasets.images.forms import ImageDatasetForm
from web.datasets.images.job import ImageDatasetJob
blueprint = Blueprint(__name__, __name__)


def from_files(job, form):
    """
    Add tasks for creating a dataset by reading textfiles
    """
    # labels
    if form.textfile_use_local_files.data:
        labels_file_from = form.textfile_local_labels_file.data.strip()
        labels_file_to = os.path.join(job.dir(), utils.constants.LABELS_FILE)
        shutil.copyfile(labels_file_from, labels_file_to)
    else:
        flask.request.files[form.textfile_labels_file.name].save(
            os.path.join(job.dir(), utils.constants.LABELS_FILE)
        )
    job.labels_file = utils.constants.LABELS_FILE

    shuffle = bool(form.textfile_shuffle.data)
    backend = form.backend.data
    encoding = form.encoding.data
    compression = form.compression.data

    # train
    if form.textfile_use_local_files.data:
        train_file = form.textfile_local_train_images.data.strip()
    else:
        flask.request.files[form.textfile_train_images.name].save(
            os.path.join(job.dir(), utils.constants.TRAIN_FILE)
        )
        train_file = utils.constants.TRAIN_FILE

    image_folder = form.textfile_train_folder.data.strip()
    if not image_folder:
        image_folder = None

    job.tasks.append(
        tasks.CreateDbTask(
            job_dir=job.dir(),
            input_file=train_file,
            db_name=utils.constants.TRAIN_DB,
            backend=backend,
            image_dims=job.image_dims,
            image_folder=image_folder,
            resize_mode=job.resize_mode,
            encoding=encoding,
            compression=compression,
            mean_file=utils.constants.MEAN_FILE_CAFFE,
            labels_file=job.labels_file,
            shuffle=shuffle,
        )
    )

    # val

    if form.textfile_use_val.data:
        if form.textfile_use_local_files.data:
            val_file = form.textfile_local_val_images.data.strip()
        else:
            flask.request.files[form.textfile_val_images.name].save(
                os.path.join(job.dir(), utils.constants.VAL_FILE)
            )
            val_file = utils.constants.VAL_FILE

        image_folder = form.textfile_val_folder.data.strip()
        if not image_folder:
            image_folder = None

        job.tasks.append(
            tasks.CreateDbTask(
                job_dir=job.dir(),
                input_file=val_file,
                db_name=utils.constants.VAL_DB,
                backend=backend,
                image_dims=job.image_dims,
                image_folder=image_folder,
                resize_mode=job.resize_mode,
                encoding=encoding,
                compression=compression,
                labels_file=job.labels_file,
                shuffle=shuffle,
            )
        )

    # test

    if form.textfile_use_test.data:
        if form.textfile_use_local_files.data:
            test_file = form.textfile_local_test_images.data.strip()
        else:
            flask.request.files[form.textfile_test_images.name].save(
                os.path.join(job.dir(), utils.constants.TEST_FILE)
            )
            test_file = utils.constants.TEST_FILE

        image_folder = form.textfile_test_folder.data.strip()
        if not image_folder:
            image_folder = None

        job.tasks.append(
            tasks.CreateDbTask(
                job_dir=job.dir(),
                input_file=test_file,
                db_name=utils.constants.TEST_DB,
                backend=backend,
                image_dims=job.image_dims,
                image_folder=image_folder,
                resize_mode=job.resize_mode,
                encoding=encoding,
                compression=compression,
                labels_file=job.labels_file,
                shuffle=shuffle,
            )
        )


@blueprint.route('.json', methods=['POST'])
@blueprint.route('', methods=['POST'], strict_slashes=False)
@utils.auth.requires_login(redirect=False)
def create():
    """
    Creates a new ImageClassificationDatasetJob
    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = ImageDatasetForm()

    if not form.validate_on_submit():
        return render_template('datasets/images/classification/new.html', form=form), 400

    job = None
    try:
        job = ImageDatasetJob(
            username=utils.auth.get_username(),
            name=form.dataset_name.data,
            group=form.group_name.data,
            image_dims=(
                int(form.resize_height.data),
                int(form.resize_width.data),
                int(form.resize_channels.data),
            ),
            resize_mode=form.resize_mode.data
        )

        if form.method.data == 'folder':
            from_folders(job, form)

        elif form.method.data == 'textfile':
            from_files(job, form)

        elif form.method.data == 's3':
            from_s3(job, form)

        else:
            raise ValueError('method not supported')

        # Save form data with the job so we can easily clone it later.
        save_form_to_job(job, form)

        scheduler.add_job(job)
        if request_wants_json():
            return flask.jsonify(job.json_dict())
        else:
            return flask.redirect(flask.url_for('digits.dataset.views.show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise