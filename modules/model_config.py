import json
from modules.paths import get_config_path


def save_config(save_path, rnn_units, num_layers, vocab):
    with open(get_config_path(save_path), "w") as file:
        json.dump({
            "rnn_units": rnn_units,
            "num_layers": num_layers,
            "vocab": vocab
        }, file)


def load_config(load_path):
    with open(get_config_path(load_path), "r") as file:
        config = json.load(file)

    return config['rnn_units'], config['num_layers'], config['vocab']
