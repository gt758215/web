import os
import inspect

def get_model_config(model_dir, file_name="network.py"):
    model_file = os.path.join(model_dir, file_name)
    if not os.path.isfile(model_file):
        raise ValueError('Could not find %s in %s for model file' % (model_file, model_dir))
    exec(open(model_file).read(), globals())
    try:
        UserModel
    except NameError:
        tf.logging.fatal("The user model class 'UserModel' is not defined.")
        exit(-1)
    if not inspect.isclass(UserModel):
        raise ValueError("The user model class 'UserModel' is not a class.")
    return UserModel() 
