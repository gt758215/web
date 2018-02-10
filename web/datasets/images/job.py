from ..job import DatasetJob


class ImageDatasetJob(DatasetJob):
    def __init__(self, **kwargs):
        self.image_dims = kwargs.pop('image_dims', None)
        self.resize_mode = kwargs.pop('resize_mode', None)

        super(ImageDatasetJob, self).__init__(**kwargs)

    @staticmethod
    def resize_mode_choices():
        return [
            ('crop', 'Crop'),
            ('squash', 'Squash'),
            ('fill', 'Fill'),
            ('half_crop', 'Half crop, half fill'),
        ]

    def resize_mode_name(self):
        c = dict(self.resize_mode_choices())
        return c[self.resize_mode]