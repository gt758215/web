from __future__ import absolute_import

from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from web.datasets.images.forms import ImageDatasetForm

blueprint = Blueprint(__name__, __name__)


@blueprint.route('/new', methods=['GET', 'POST'])
def new():
    try:
        form = ImageDatasetForm()
        # if not form.validate_on_submit():
        return render_template('datasets/images/new.html', form=form), 400
    except TemplateNotFound:
        abort(404)
