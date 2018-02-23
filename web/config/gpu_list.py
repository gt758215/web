from __future__ import absolute_import
from . import option_list
import web.device_query


option_list['gpu_list'] = ','.join([str(x) for x in range(len(web.device_query.get_devices()))])