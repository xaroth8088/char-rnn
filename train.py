import tensorflow as tf
import shutil
import numpy as np
import os
import argparse
from modules.model import build_model
from modules.paths import get_input_path, get_checkpoint_path, cleanup_files
from modules.preprocessing import create_indexes, split_input_target
from modules.model_config import save_config
from modules.hardware_setup import setup_hardware

# CLI setup
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
parser.add_argument('--batch_size', type=int, default=32,
                    help="""minibatch size. It's not recommended to go larger than 32.""")
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs. Number of full passes through the training examples.')
parser.add_argument('--find_best_hyperparams', type=bool, default=False,
                    help='Instead of training, attempt to find ideal settings')

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

save_config(args.data_dir, args.rnn_units, args.num_layers, vocab)

char2idx, idx2char = create_indexes(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(args.seq_length + 1, drop_remainder=True)

# For each sequence, duplicate and shift it to form the input and target text by using the map method to apply a simple
# function to each batch:
dataset = sequences.map(split_input_target)

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

# Shuffle the dataset
dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(args.batch_size, drop_remainder=True)

#############################
#   Build the model
#############################

# Construct the model
model = build_model(
    vocab_size=len(vocab),
    rnn_units=args.rnn_units,
    batch_size=args.batch_size,
    num_layers=args.num_layers)


#############################
#   Train the model
#############################

# Directory where the checkpoints will be saved
checkpoint_dir = get_checkpoint_path(args.data_dir)
os.mkdir(checkpoint_dir)

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(dataset, epochs=args.num_epochs, callbacks=[checkpoint_callback])
