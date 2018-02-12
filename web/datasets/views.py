from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from web.datasets import DatasetJob
from web.webapp import scheduler

blueprint = Blueprint(__name__, __name__)


@blueprint.route('/datasets', methods=['GET'])
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