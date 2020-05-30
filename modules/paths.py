import os
import shutil


def get_checkpoint_path(base):
    return os.path.join(base, 'training_checkpoints')


def get_config_path(base):
    return os.path.join(base, 'model_config.json')


def cleanup_files(base):
    # Remove previous checkpoint files
    try:
        shutil.rmtree(get_checkpoint_path(base))
    except FileNotFoundError:
        pass

    # Remove previous model configuration
    try:
        os.remove(get_config_path(base))
    except FileNotFoundError:
        pass

    # Remove previous hyperparams tuning data
    try:
        shutil.rmtree(os.path.join(base, 'hyperparams-optimization'))
    except FileNotFoundError:
        pass


def get_input_path(base):
    return os.path.join(base, 'input.txt')
