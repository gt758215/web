from __future__ import absolute_import

from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from web.model.images.forms import  ImageClassificationModelForm

blueprint = Blueprint(__name__, __name__)


@blueprint.route('/summary', methods=['GET'])
def summary():
    try:
        return render_template('train/trains.html')
    except TemplateNotFound:
        abort(404)

@blueprint.route('/image/classification/new', methods=['GET'])
def image_classification_new():
    try:
        form = ImageClassificationModelForm()
        return render_template('train/images/classification_new.html',
                               form=form)
    except TemplateNotFound:
        abort(404)

@blueprint.route('/image/classification/create', methods=['POST'])
def image_classification_create():
    form = ImageClassificationModelForm()
    train_epochs = form.train_epochs.data
    return "hello world "+ str(train_epochs)


@blueprint.route('/image/segmentation/new', methods=['GET'])
def image_segmentation_new():
    try:
        #form={'dataset':{'label':'label1', 'tooltip':'tips'}}
        form = ImageClassificationModelForm()
        return render_template('train/images/segmentation_new.html',
                               form=form)
    except TemplateNotFound:
        abort(404)
