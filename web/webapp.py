from flask import Flask
from .config import config_value
import web.scheduler

app = Flask(__name__)
scheduler = web.scheduler.Scheduler(config_value('gpu_list'), True)

import web.views
app.register_blueprint(web.views.blueprint)

import web.datasets.views
app.register_blueprint(web.datasets.views.blueprint)

import web.datasets.images.classification.views
app.register_blueprint(web.datasets.images.classification.views.blueprint, url_prefix='/datasets/images/classification')

#scheduler.load_past_jobs()