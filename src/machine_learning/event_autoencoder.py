from keras import layers, models

def build_event_autoencoder(output_size: int):
    """
    This works very poorly
    """
    encoder = models.Sequential([
        # layers.Dense(64, activation='relu', input_shape=(output_size,)),
        layers.LSTM(64, input_shape=(output_size, 1), return_sequences=True)
    ])
    encoder.summary()

    decoder = models.Sequential([
        # layers.Dense(output_size, activation='relu', input_shape=(64,)),
        layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), input_shape=(output_size, 64))
    ])
    decoder.summary()

    model = models.Sequential([encoder, decoder])
    model.summary()
    model.compile(loss='mse', metrics=['acc'])

    return model, encoder, decoder