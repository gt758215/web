import wtforms
from ..forms import DatasetForm


class ImageDatasetForm(DatasetForm):
    # Image resize

    textfile_train_images = wtforms.FileField(u'Training images')
    textfile_train_folder = wtforms.StringField(u'Training images fodler')
    textfile_val_images = wtforms.FileField(u'Validation images')
    textfile_val_folder= wtforms.StringField(u'Validation images fodler')