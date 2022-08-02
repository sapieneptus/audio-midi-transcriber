import numpy as np
import note_seq
import note_seq
from tensorflow.keras.callbacks import EarlyStopping
from keras import models, layers
from data_processors import utils, audio
from typing import Tuple

"""
I attempted to build an audio autoencoder to verify that wav data can be efficiently
compressed via a single LSTM layer.

The training accuracy wasn't amazing, but the recovered files sounded quite recognizably the same
as the original (plus some extra digital noise). Enough to verify that the LSTM layer is working as
an encoder.
"""

def train_audio_autoencoder(inputs, num_epochs: int, batch_size: int, reconstruct_inputs=False):
    inputs = np.array(inputs)
    inputs = note_seq.audio_io.float_samples_to_int16(inputs)
    scaler, scaled_inputs = utils.scale_inputs(inputs)

    model, encoder, decoder = build_audio_autoencoder()

    early_stop = EarlyStopping(monitor='loss', patience=2)
    history = model.fit(
        scaled_inputs,
        scaled_inputs,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[early_stop]
    )

    utils.plot_history(history)
    prediction = model.predict(scaled_inputs)

    if reconstruct_inputs:
        reconstruct_wav(prediction, scaler)
    
    return model, encoder, decoder

def reconstruct_wav(prediction, scaler, output_file='moo.wav'):
    prediction = scaler.inverse_transform(prediction.reshape(-1, prediction.shape[-1])).reshape(prediction.shape)
    prediction = prediction.astype(np.int16)
    prediction = note_seq.audio_io.int16_samples_to_float32(prediction)
    prediction = prediction.flatten()
    wav_data = note_seq.audio_io.samples_to_wav_data(prediction, audio.DEFAULT_SAMPLE_RATE)

    with open(output_file, 'wb') as f:
        f.write(wav_data)

def build_audio_autoencoder():
    frames_per_second = int(audio.frames_per_second())
    frame_length = int(audio.DEFAULT_HOP_WIDTH)

    encoder = models.Sequential([
        # Reduce length of each frame
        layers.LSTM(32, input_shape=(frames_per_second, frame_length), return_sequences=True, name='encoder_lstm')
    ])
    encoder.summary()

    decoder = models.Sequential([
        layers.LSTM(frames_per_second, input_shape=(frames_per_second, 32), return_sequences=True),
        layers.TimeDistributed(layers.Dense(frame_length, ))
    ])
    decoder.summary()

    model = models.Sequential([encoder, decoder])
    model.summary()
    model.compile(loss='mse', metrics=['acc'])

    return model, encoder, decoder