from __future__ import absolute_import

import werkzeug.exceptions

from flask import Blueprint, render_template, abort, redirect, url_for
from jinja2 import TemplateNotFound
from web.models.images.forms import ImageClassificationModelForm
from .job import ImageClassificationModelJob
from web.webapp import scheduler
from web import frameworks
from web.status import Status

blueprint = Blueprint(__name__, __name__)


@blueprint.route('/new', methods=['GET'])
def new():
    try:
        form = ImageClassificationModelForm()
        form.dataset.choices = get_dataset()
        form.standard_networks.choices = get_standard_networks()
        form.standard_networks.default = get_default_standard_network()
        return render_template('models/images/classification_new.html',
                               form=form)
    except TemplateNotFound:
        abort(404)

@blueprint.route('/create', methods=['POST'])
def create():
    form = ImageClassificationModelForm()

    datasetJob = scheduler.get_job(form.dataset.data)
    if not datasetJob:
        raise werkzeug.exceptions.BadRequest(
            'Unknown dataset job_id "%s"' % form.dataset.data
        )

    sweeps = [{'learning_rate': v} for v in form.learning_rate.data]
    sweeps = [dict(s.items() + [('batch_size', bs)]) for bs in form.batch_size.data for s in sweeps[:]]
    train_epochs = form.train_epochs.data
    n_jobs = len(sweeps)

    jobs = []
    for sweep in sweeps:
        form.learning_rate.data = sweep['learning_reate']
        form.batch_size.data = sweep['batch_size']

        extra = ''
        job = None

        try:
            job = ImageClassificationModelJob(
                name = form.model_name.data + extra,
                group = form.group_name.data,
                dataset_id = datasetJob
            )
            fw = frameworks.get_framework_by_id(form.framework.data)
            pretrained_model = None
            if form.method.data == 'standard':
                found = False
                network_desc = fw.get_standard_network_desc(form.standard_networks.data)
                if network_desc:
                    found = True
                    network = fw.get_network_from_desc(network_desc)
                if not found:
                    raise werkzeug.exceptions.BadRequest(
                        'Unknown standard model "%s"' % form.standard_networks.data
                    )
            else:
                raise werkzeug.exceptions.BadRequest(
                    'Unrecognized method: "%s"' % form.method.data
                )
            policy = {'policy': form.lr_policy.data}
            if form.lr_policy.data == 'step':
                policy['stepsize'] = form.lr_step_size.data
                policy['gamma'] = form.lr_step_gamma
            else:
                raise werkzeug.exceptions.BadRequest(
                    'Invalid learning rate policy'
                )
            if form.select_gpus.data:
                selected_gpus = [str(gpu) for gpu in form.select_gpus.data]
                gpu_count = None
            elif form.select_gpu_count.data:
                gpu_count = form.select_gpu_count.data
                selected_gpus = None
            else:
                gpu_count = 1
                selected_gpus = None

            job.tasks.append(fw.create_train_task(
                job = job,
                dataset = datasetJob,
                train_epochs = form.train_epochs.data,
                snapshot_interval = form.snapshot_interval.data,
                learning_reate = form.learning_rate.data[0],
                lr_policy = policy,
                gpu_count = gpu_count,
                selected_gpus = selected_gpus,
                batch_size = form.batch_size.data[0],
                batch_accumulation = form.batch_accumulation.data,
                val_interval = form.val_interval.data,
                traces_interval = form.traces_interval.data,
                pretrained_model = pretrained_model,
                network = network,
                random_seed = form.random_seed.data,
                solver_type = form.solver_type.data,
                rms_decay = form.rms_decay.data,
            )
            )
            # TODO for clone jobs

            jobs.append(job)
            scheduler.add_job(job)
            if n_jobs == 1:
                return redirect(url_for('web.train.views.show', job_id=job.id()))
        except:
            if job:
                scheduler.delete_job(job)
            raise
    return redirect('/')


def get_dataset():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, ImageClassificationModelJob) and
         (j.status.is_running() or j.status == Status.DONE)],
        cmp=lambda x, y: cmp(y.id(), x.id())
    )]

def get_standard_networks():
    return [
        ('lenet', 'LeNet'),
        ('resnet-50', 'ResNet-50'),
        ('googlenet', 'GoogLeNet'),
    ]

def get_default_standard_network():
    return 'lenet'