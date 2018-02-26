from ..job import ImageModelJob
from web.utils import override


class ImageClassificationModelJob(ImageModelJob):
    def __init__(self, **kwargs):
        super(ImageClassificationModelJob, self).__init__(**kwargs)

    @override
    def job_type(self):
        return 'Image Classification Model'