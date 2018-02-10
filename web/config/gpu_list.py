from . import option_list
import web.device_query


option_list['gpu_list'] = ','.join([str(x) for x in xrange(len(web.device_query.get_devices()))])