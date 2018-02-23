from .tensorflow_framework import TensorflowFramework

__all__ = [
    'TensorflowFramework',
]

tensorflow = TensorflowFramework()


def get_frameworks():
    """
    return list of all available framework instances
    there may be more than one instance per framework class
    """
    frameworks = [tensorflow]
    return frameworks


def get_framework_by_id(framework_id):
    """
    return framework instance associated with given id
    """
    for fw in get_frameworks():
        if fw.get_id() == framework_id:
            return fw
    return None