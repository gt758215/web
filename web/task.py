import gevent.event
from .status import StatusCls


class Task(StatusCls):
    def __init__(self, job_dir, parents=None):
        super(Task, self).__init__()

        self.job_dir = job_dir
        self.job_id = os.path.basename(job_dir)

        if parents is None:
            self.parents = None
        elif isinstance(parents, (list, tuple)):
            self.parents = parents
        elif isinstance(parents, Task):
            self.parents = [parents]
        else:
            raise TypeError('parents is %s' % type(parents))

        self.exception = None
        self.traceback = None
        self.aborted = gevent.event.Event()
        self.set_logger()
        self.p = None  # Subprocess object for training

    def name(self):
        raise NotImplementedError