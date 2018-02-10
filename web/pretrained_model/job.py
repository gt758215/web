from web.job import Job

from web.pretrained_model.tasks import TensorflowUploadTask

class PretrainedModelJob(Job):
    def __init__(self, weights_path, model_def_path, labels_path=None, framework="caffe",
                 image_type="3", resize_mode="Squash", width=224, height=224, **kwargs):
        super(PretrainedModelJob, self).__init__(persistent=False, **kwargs)

        self.framework = framework
        self.image_info = {
            "image_type": image_type,
            "resize_mode": resize_mode,
            "width": width,
            "height": height
        }

        self.tasks = []

        taskKwargs = {
            "weights_path": weights_path,
            "model_def_path": model_def_path,
            "image_info": self.image_info,
            "labels_path": labels_path,
            "job_dir": self.dir()
        }

        self.tasks.append(TensorflowUploadTask(**taskKwargs))

