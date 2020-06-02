import tensorflow as tf
from modules.preprocessing import split_input_target

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000


def build_model(vocab_size, rnn_units, batch_size, num_layers):
    # TODO: what does this param do?  Should it be configurable?
    embedding_dim = 256

    layers = []

    # Input layer
    layers.append(tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                            batch_input_shape=[batch_size, None]))

    # Hidden layers
    for _ in range(num_layers):
        layers.append(
            tf.keras.layers.LSTM(rnn_units,  # TODO: make this layer's type (GRU, LSTM, etc.) configurable via CLI
                                 return_sequences=True,
                                 stateful=True)
        )

    # Output layer
    layers.append(
        tf.keras.layers.Dense(vocab_size)
    )

    return tf.keras.Sequential(layers)


def train(vocab=None, char_dataset=None, checkpoint_callbacks=None, num_epochs=None, loss_function=None,
          rnn_units=None, batch_size=None, num_layers=None, seq_length=None):
    #############################
    #   Prep the data for training
    #############################
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    # For each sequence, duplicate and shift it to form the input and target text by using the map method to apply a
    # simple function to each batch:
    dataset = sequences.map(split_input_target)

    # Shuffle the dataset
    dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)

    #############################
    #   Build the model
    #############################

    # Construct the model
    model = build_model(
        vocab_size=len(vocab),
        rnn_units=rnn_units,
        batch_size=batch_size,
        num_layers=num_layers
    )

    # TODO: Should the choice of optimizer also be configurable?
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(),
        loss=loss_function
    )

    #############################
    #   Train the model
    #############################

    history = model.fit(
        dataset,
        epochs=num_epochs,
        callbacks=checkpoint_callbacks,
        workers=4,
        use_multiprocessing=True
    )
    return history
