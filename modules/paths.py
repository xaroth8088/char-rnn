import os


def get_checkpoint_path(base):
    return os.path.join(base, 'training_checkpoints')


def get_config_path(base):
    return os.path.join(base, 'model_config.json')
