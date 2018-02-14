import wtforms
from ..forms import DatasetForm
from web import utils


class ImageDatasetForm(DatasetForm):
    # Image resize

    textfile_train_images = wtforms.StringField(u'Training images')
    textfile_train_folder = wtforms.StringField(u'Training images fodler')
    textfile_val_images = wtforms.StringField(u'Validation images')
    textfile_val_folder= wtforms.StringField(u'Validation images fodler')

    textfile_local_labels_file = utils.forms.StringField(
        u'Labels',
        tooltip=("This 'i'th line of the file should give the string label "
                 "associated with the '(i-1)'th numeric label."),
    )