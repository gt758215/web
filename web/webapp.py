from __future__ import absolute_import

from flask import Flask
from .config import config_value
import web.scheduler

app = Flask(__name__, static_url_path='/static')
scheduler = web.scheduler.Scheduler(config_value('gpu_list'), True)

app.config['WTF_CSRF_ENABLED'] = False
app.url_map.redirect_defaults = False

import web.views
app.register_blueprint(web.views.blueprint, url_prefix='')

import web.datasets.views
app.register_blueprint(web.datasets.views.blueprint, url_prefix='/datasets')

import web.datasets.images.views
app.register_blueprint(web.datasets.images.views.blueprint, url_prefix='/datasets/images')

#scheduler.load_past_jobs()