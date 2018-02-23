from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
import flask
import os
import platform
import json
import glob

blueprint = Blueprint(__name__, __name__)

@blueprint.route('/', methods=['GET'])
def home():
    try:
        return 'Hello, World'
        #return render_template('home.html')
    except TemplateNotFound:
        abort(404)



# Path Completion


@blueprint.route('/autocomplete/path', methods=['GET'])
def path_autocomplete():
    """
    Return a list of paths matching the specified preamble

    """
    path = flask.request.args.get('query', '')

    if not os.path.isabs(path):
        # Only allow absolute paths by prepending forward slash
        path = os.path.sep + path

    suggestions = [os.path.abspath(p) for p in glob.glob(path + "*")]
    if platform.system() == 'Windows':
        # on windows, convert backslashes with forward slashes
        suggestions = [p.replace('\\', '/') for p in suggestions]

    result = {
        "suggestions": sorted(suggestions)
    }

    return json.dumps(result)