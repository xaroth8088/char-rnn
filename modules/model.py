import tensorflow as tf


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
