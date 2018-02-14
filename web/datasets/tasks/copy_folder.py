import os.path
from web.task import Task

class CopyFolderTask(Task):
    def __init__(selfself, folder, **kwargs):
        super(CopyFolderTask, self).__init__(**kwargs)
