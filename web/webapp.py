from __future__ import absolute_import

from flask import Flask
from flask_socketio import SocketIO
from .config import config_value
import web.scheduler

app = Flask(__name__, static_url_path='/static')
#scheduler = web.scheduler.Scheduler(config_value('gpu_list'), True)

#app.config['WTF_CSRF_ENABLED'] = False
#app.url_map.redirect_defaults = False
#socketio = SocketIO(app, async_mode='gevent', path='/socket.io')


import web.views
app.register_blueprint(web.views.blueprint, url_prefix='')

#import web.datasets.views
#app.register_blueprint(web.datasets.views.blueprint, url_prefix='/datasets')

#import web.datasets.images.views
#app.register_blueprint(web.datasets.images.views.blueprint, url_prefix='/datasets/images')

#import web.networks.views
#app.register_blueprint(web.networks.views.blueprint, url_prefix='/networks')

#import web.train.views
#app.register_blueprint(web.train.views.blueprint, url_prefix='/train')

#scheduler.load_past_jobs()