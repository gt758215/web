import wtforms
import requests
import os
from ..forms import DatasetForm
from web import utils
from web.utils.forms import validate_required_iff
from wtforms import validators


class ImageDatasetForm(DatasetForm):
    def validate_folder_path(form, field):
        if not field.data:
            pass
        elif utils.is_url(field.data):
            # make sure the URL exists
            try:
                r = requests.get(field.data,
                                 allow_redirects=False,
                                 timeout=utils.HTTP_TIMEOUT)
                if r.status_code not in [requests.codes.ok, requests.codes.moved, requests.codes.found]:
                    raise validators.ValidationError('URL not found')
            except Exception as e:
                raise validators.ValidationError('Caught %s while checking URL: %s' % (type(e).__name__, e))
            else:
                return True
        else:
            # make sure the filesystem path exists
            # and make sure the filesystem path is absolute
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError('Folder does not exist')
            elif not os.path.isabs(field.data):
                raise validators.ValidationError('Filesystem path is not absolute')
            else:
                return True

    textfile_train_images = utils.forms.StringField(
        u'Training images',
        validators=[
            validate_required_iff(method='folder'),
            validate_folder_path,
        ],
    )
    textfile_train_folder = wtforms.StringField(u'Training images fodler')
    textfile_val_images = wtforms.StringField(u'Validation images')
    textfile_val_folder= wtforms.StringField(u'Validation images fodler')

    textfile_local_labels_file = utils.forms.StringField(
        u'Labels',
        tooltip=("This 'i'th line of the file should give the string label "
                 "associated with the '(i-1)'th numeric label."),
    )