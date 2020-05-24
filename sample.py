import argparse
import random
import os
import tensorflow as tf
from modules.model import build_model
from modules.paths import get_checkpoint_path
from modules.preprocessing import create_indexes
from modules.model_config import load_config
from modules.hardware_setup import setup_hardware

# CLI setup
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directories
parser.add_argument('--data_dir', type=str, default=os.path.join('data', 'shakespeare'),
                    help='data directory containing input.txt with training examples')

# Sampling
parser.add_argument('--sample_len', type=int, default=1000,
                    help='number of characters to sample')
parser.add_argument('--temperature', type=float, default=1.0,
                    help="""Low temperatures results in more predictable text.
                            Higher temperatures results in more surprising text.""")
parser.add_argument('--prime', type=str, default=u'',
                    help='Before sampling, prime the output with this input.')

args = parser.parse_args()


def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = args.sample_len

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)

        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / args.temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


setup_hardware()

# Load config
rnn_units, num_layers, vocab = load_config(args.data_dir)

# The unique characters in the file + index tables
char2idx, idx2char = create_indexes(vocab)

# Load model from the latest checkpoint, but with a batch size of 1 for sampling
model = build_model(len(vocab), rnn_units, batch_size=1, num_layers=num_layers)

model.load_weights(tf.train.latest_checkpoint(get_checkpoint_path(args.data_dir)))

model.build(tf.TensorShape([1, None]))

# Pick a random character to start with if no priming string is given
priming_string = args.prime
if priming_string == u'':
    priming_string = random.sample(vocab, 1)[0]

print(generate_text(model, start_string=priming_string))
