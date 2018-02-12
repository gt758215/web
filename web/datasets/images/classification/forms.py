import wtforms
from ..forms import ImageDatasetForm
from web import utils


class ImageClassificationDatasetForm(ImageDatasetForm):
    #
    # Method - textfile
    #
    textfile_local_train_images = wtforms.StringField(
        u'Training images',
        validators=[]
    )

    textfile_train_folder = wtforms.StringField(u'Training images folder')

    textfile_local_val_images = wtforms.StringField(u'Validation images',
                                                    validators=[]
                                                    )
    textfile_val_folder = wtforms.StringField(u'Validation images folder')

