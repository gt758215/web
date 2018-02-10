from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
from web import utils


class DatasetForm(FlaskForm):
    dataset_name = utils.forms.StringField(u'Dataset Name',
                                           validators=[DataRequired()]
                                           )
    group_name = utils.forms.StringField('Group Name',
                                         tooltip="An optional group name for organization on the main page."
                                         )