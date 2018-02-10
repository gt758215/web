from web.pretrained_model.tasks import UploadPretrainedModelTask
from web.utils import subclass, override

@subclass
class TensorflowUploadTask(UploadPretrainedModelTask):

    def __init__(self, **kwargs):
        super(TensorflowUploadTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Upload Pretrained Tensorflow Model'