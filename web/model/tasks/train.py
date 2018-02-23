from web.task import Task


class TrainTask(Task):
    """
    Defines required methods for child classes
    """

    def __init__(self, job, dataset, train_epochs, snapshot_interval, learning_rate, lr_policy, **kwargs):
        """
        Arguments:
        job -- model job
        dataset -- a DatasetJob containing the dataset for this model
        train_epochs -- how many epochs of training data to train on
        snapshot_interval -- how many epochs between taking a snapshot
        learning_rate -- the base learning rate
        lr_policy -- a hash of options to be used for the learning rate policy
        Keyword arguments:
        gpu_count -- how many GPUs to use for training (integer)
        selected_gpus -- a list of GPU indexes to be used for training
        batch_size -- if set, override any network specific batch_size with this value
        batch_accumulation -- accumulate gradients over multiple batches
        val_interval -- how many epochs between validating the model with an epoch of validation data
        traces_interval -- amount of steps in between timeline traces
        pretrained_model -- filename for a model to use for fine-tuning
        crop_size -- crop each image down to a square of this size
        use_mean -- subtract the dataset's mean file or mean pixel
        random_seed -- optional random seed
        data_aug -- data augmentation options
        """
        self.gpu_count = kwargs.pop('gpu_count', None)
        self.selected_gpus = kwargs.pop('selected_gpus', None)
        self.batch_size = kwargs.pop('batch_size', None)
        self.batch_accumulation = kwargs.pop('batch_accumulation', None)
        self.val_interval = kwargs.pop('val_interval', None)
        self.traces_interval = kwargs.pop('traces_interval', None)
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.crop_size = kwargs.pop('crop_size', None)
        self.use_mean = kwargs.pop('use_mean', None)
        self.random_seed = kwargs.pop('random_seed', None)
        self.solver_type = kwargs.pop('solver_type', None)
        self.rms_decay = kwargs.pop('rms_decay', None)
        self.shuffle = kwargs.pop('shuffle', None)
        self.network = kwargs.pop('network', None)
        self.framework_id = kwargs.pop('framework_id', None)
        self.data_aug = kwargs.pop('data_aug', None)
        self.rampup_lr = kwargs.pop('rampup_lr', None)
        self.rampup_epoch = kwargs.pop('rampup_epoch', None)
        self.weight_decay = kwargs.pop('weight_decay', None)

        super(TrainTask, self).__init__(job_dir=job.dir(), **kwargs)

        self.job = job
        self.dataset = dataset
        self.train_epochs = train_epochs
        self.snapshot_interval = snapshot_interval
        self.learning_rate = learning_rate
        self.lr_policy = lr_policy

        self.current_epoch = 0
        self.snapshots = []
        self.timeline_traces = []