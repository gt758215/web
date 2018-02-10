from web.job import Job


class ModelJob(Job):
    """
        A Job that creates a neural network model
        """

    def __init__(self, dataset_id, **kwargs):
        """
        Arguments:
        dataset_id -- the job_id of the DatasetJob that this ModelJob depends on
        """
        super(ModelJob, self).__init__(**kwargs)

        self.dataset_id = dataset_id
        self.load_dataset()

    def load_dataset(self):
        from web.webapp import scheduler
        job = scheduler.get_job(self.dataset_id)
        assert job is not None, 'Cannot find dataset'
        self.dataset = job
        for task in self.tasks:
            task.dataset = job