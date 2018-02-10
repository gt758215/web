import inspect
import math


def subclass(cls):
    """
    Verify all @override methods
    Use a class decorator to find the method's class
    """
    for name, method in cls.__dict__.iteritems():
        if hasattr(method, 'override'):
            found = False
            for base_class in inspect.getmro(cls)[1:]:
                if name in base_class.__dict__:
                    if not method.__doc__:
                        # copy docstring
                        method.__doc__ = base_class.__dict__[name].__doc__
                    found = True
                    break
            assert found, '"%s.%s" not found in any base class' % (cls.__name__, name)
    return cls


def override(method):
    """
    Decorator implementing method overriding in python
    Must also use the @subclass class decorator
    """
    method.override = True
    return method


def sizeof_fmt(size, suffix='B'):
    """
    Return a human-readable string representation of a filesize
    Arguments:
    size -- size in bytes
    """
    try:
        size = int(size)
    except ValueError:
        return None
    if size <= 0:
        return '0 %s' % suffix

    size_name = ('', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    i = int(math.floor(math.log(size, 1024)))
    if i >= len(size_name):
        i = len(size_name) - 1
    p = math.pow(1024, i)
    s = size / p
    # round to 3 significant digits
    s = round(s, 2 - int(math.floor(math.log10(s))))
    if s.is_integer():
        s = int(s)
    if s > 0:
        return '%s %s%s' % (s, size_name[i], suffix)
    else:
        return '0 %s' % suffix