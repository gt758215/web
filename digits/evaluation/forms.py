# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.


from flask_wtf import FlaskForm
from wtforms import validators
from digits import utils


class EvaluationForm(FlaskForm):

    # Fields
    selected_model = utils.forms.SelectField(
        'Select Model',
        choices=[],
        validators=[
            validators.DataRequired()
        ],
        tooltip='Choose model to evaluate your dataset.'
    )

    selected_dataset = utils.forms.SelectField(
        'Select Dataset',
        choices=[],
        validators=[
            validators.DataRequired()
        ],
        tooltip='choose dataset you want to evaluate.'
    )
