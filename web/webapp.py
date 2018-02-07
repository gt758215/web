from flask import Flask

app = Flask(__name__)

import web.views
app.register_blueprint(web.views.blueprint)