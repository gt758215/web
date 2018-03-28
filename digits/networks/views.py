from __future__ import absolute_import
import flask
import werkzeug.exceptions
import os

from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from digits.frameworks import tensorflow_framework
from digits.webapp import scheduler
from digits.utils.routing import get_request_arg
from os import path

blueprint = Blueprint(__name__, __name__)


from digits.log import logger

@blueprint.route('/standard_networks', methods=['GET'])
def standard_networks():
    job_id = get_request_arg('job_id')
    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "..", "standard-networks/tensorflow/", job_id+'.py'))
    with open(filepath, 'r') as network_file:
        network_def = network_file.read()

    return network_def

@blueprint.route('/summary', methods=['GET'])
def summary():
    try:
        return render_template('networks/networks.html')
    except TemplateNotFound:
        abort(404)

@blueprint.route('/visualize-network', methods=['POST'])
def visualize_network():
    """
    Returns a visualization of the custom network as a string of PNG data
    """
    framework = 'tensorflow' #hardcode to tensorflow
    if not framework:
        raise werkzeug.exceptions.BadRequest('framework not provided')

    dataset = None
    if 'dataset_id' in flask.request.form:
        dataset = scheduler.get_job(flask.request.form['dataset_id'])

    #fw = frameworks.get_framework_by_id(framework)
    ret = tensorflow_framework('tensorflow').get_network_visualization(
        desc=flask.request.form['custom_network'],
        dataset=dataset,
        solver_type=flask.request.form['solver_type'] if 'solver_type' in flask.request.form else None,
        use_mean=flask.request.form['use_mean'] if 'use_mean' in flask.request.form else None,
        crop_size=flask.request.form['crop_size'] if 'crop_size' in flask.request.form else None,
        num_gpus=flask.request.form['num_gpus'] if 'num_gpus' in flask.request.form else None,
    )
    return ret