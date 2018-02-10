option_list = {}

from . import (
    gpu_list,
    jobs_dir,
    log_file,
)


def config_value(option):
    return option_list[option]