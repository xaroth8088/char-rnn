import itertools
import tensorflow as tf
import numpy as np
import os
import argparse
from random import sample
from datetime import datetime

from modules.model import train
from modules.paths import get_input_path, get_checkpoint_path, cleanup_files, get_checkpoint_prefix
from modules.preprocessing import create_indexes
from modules.model_config import save_config
from modules.hardware_setup import setup_hardware

#############################
#   CLI setup
#############################
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directories
parser.add_argument('--data_dir', type=str, default=os.path.join('data', 'shakespeare'),
                    help='data directory containing input.txt with training examples')
parser.add_argument('--save_every', type=int, default=1000,
                    help='Save frequency. Number of passes between checkpoints of the model.')
# parser.add_argument('--init_from', type=str, default=None,
#                     help="""continue training from saved model at this path (usually "save").
#                         Path must contain files saved by previous training process:
#                         'config.pkl'        : configuration;
#                         'chars_vocab.pkl'   : vocabulary definitions;
#                         'checkpoint'        : paths to model file(s) (created by tf).
#                                               Note: this file contains absolute paths, be careful when moving files around;
#                         'model.ckpt-*'      : file(s) with model definition (created by tf)
#                          Model params must be the same between multiple runs (model, rnn_size, num_layers and seq_length).
#                     """)

# Model params
# parser.add_argument('--model', type=str, default='lstm',
#                     help='lstm, rnn, gru, or nas')
parser.add_argument('--rnn_units', type=int, default=1024,
                    help='size of RNN hidden state')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')

# Optimization
parser.add_argument('--seq_length', type=int, default=50,
                    help='RNN training sequence length')
parser.add_argument('--batch_size', type=int, default=16,
                    help="""minibatch size. It's not recommended to go larger than 32.""")
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs. Number of full passes through the training examples.')
parser.add_argument('--find_best_hyperparams', type=bool, default=False,
                    help='Instead of training, attempt to find ideal settings')
parser.add_argument('--num_trials', type=int, default=10,
                    help="""How many trials to run.  Each trial will run --num_epochs epochs.
                            This option is unused if --find_best_hyperparams is not set""")

args = parser.parse_args()

setup_hardware()
cleanup_files(args.data_dir)

#############################
#   Prepare input text
#############################

# Read, then decode for py2 compat.
with open(get_input_path(args.data_dir), 'rb') as file:
    text = file.read().decode(encoding='utf-8')

# The unique characters in the file + index tables
vocab = sorted(set(text))

if args.find_best_hyperparams is False:
    save_config(args.data_dir, args.rnn_units, args.num_layers, vocab)

char2idx, idx2char = create_indexes(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


# Our loss function
def loss(labels, logits):
    # TODO: make the loss function configurable?
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# Train it!
if args.find_best_hyperparams is False:
    # Directory where the checkpoints will be saved
    checkpoint_dir = get_checkpoint_path(args.data_dir)
    os.mkdir(checkpoint_dir)

    # Name of the checkpoint files
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=get_checkpoint_prefix(checkpoint_dir),
        save_weights_only=True)

    train(
        vocab=vocab,
        char_dataset=char_dataset,
        rnn_units=args.rnn_units,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        checkpoint_callbacks=[checkpoint_callback],
        num_epochs=args.num_epochs,
        loss_function=loss
    )
elif args.find_best_hyperparams is True:
    # Define the search space for the parameters
    common_params = {
        "vocab": vocab,
        "char_dataset": char_dataset,
        "checkpoint_callbacks": [],
        "num_epochs": args.num_epochs,
        "loss_function": loss
    }

    keys = ['rnn_units', 'batch_size', 'num_layers', 'seq_length']
    # TODO: CLI to give sets for these params to test within
    search_space = list(itertools.product(
        [x for x in range(32, 1024, 64)],  # rnn_units
        [2, 4, 8, 16, 32],  # batch_size
        [1, 2, 3],  # num_layers
        [x for x in range(1, 240, 5)],  # seq_length
    ))

    results = []

    for _ in range(args.num_trials):
        hyperparams = dict(zip(keys, sample(search_space, 1)[0]))
        print("BEGINNING TRIAL")
        print(hyperparams)
        try:
            start = datetime.now()
            history = train(**common_params, **hyperparams)
            end = datetime.now()
            trial_loss = history.history['loss'][-1:]
            results.append({
                "hyperparams": hyperparams,
                "loss": trial_loss,
                "duration": end - start
            })
        except Exception as e:
            print(e)

    print(results)
