from web.task import Task


class UploadPretrainedModelTask(Task):
    """
    A task for uploading pretrained models
    """

    def __init__(self, **kwargs):
        """
        Arguments:
        weights_path -- path to model weights (**.caffemodel or ***.t7)
        model_def_path  -- path to model definition (**.prototxt or ***.lua)
        image_info -- a dictionary containing image_type, resize_mode, width, and height
        labels_path -- path to text file containing list of labels
        framework  -- framework of this job (ie caffe or torch)
        """
        self.weights_path = kwargs.pop('weights_path', None)
        self.model_def_path = kwargs.pop('model_def_path', None)
        self.image_info = kwargs.pop('image_info', None)
        self.labels_path = kwargs.pop('labels_path', None)
        self.framework = kwargs.pop('framework', None)

        # resources
        self.gpu = None

        super(UploadPretrainedModelTask, self).__init__(**kwargs)