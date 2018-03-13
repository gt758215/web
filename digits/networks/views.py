from __future__ import absolute_import

from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

blueprint = Blueprint(__name__, __name__)

@blueprint.route('/summary', methods=['GET'])
def summary():
    try:
        return render_template('networks/networks.html')
    except TemplateNotFound:
        abort(404)
