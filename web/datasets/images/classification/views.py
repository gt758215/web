import os
import shutil
import flask
from flask import Blueprint, render_template
from .forms import ImageClassificationDatasetForm
from web import utils
from web.datasets.images.forms import ImageDatasetForm
from web.datasets.images.job import ImageDatasetJob
blueprint = Blueprint(__name__, __name__)


@blueprint.route('/new', methods=['GET'])
def new():
    form = ImageClassificationDatasetForm()

    if not form.validate_on_submit():
        return render_template('datasets/images/classification/new.html', form=form), 400