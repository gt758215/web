import os
import web
from . import option_list

if 'WEB_JOBS_DIR' in os.environ:
    value = os.environ['WEB_JOBS_DIR']
else:
    value = os.path.join(os.path.dirname(web.__file__), 'jobs')

try:
    value = os.path.abspath(value)
    if os.path.exists(value):
        if not os.path.isdir(value):
            raise IOError('No such directory: "%s"' % value)
        if not os.access(value, os.W_OK):
            raise IOError('Permission denied: "%s"' % value)
    if not os.path.exists(value):
        os.makedirs(value)
except:
    print '"%s" is not a valid value for jobs_dir.' % value
    print 'Set the envvar WEB_JOBS_DIR to fix your configuration.'
    raise

option_list['jobs_dir'] = value