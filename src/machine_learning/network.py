from keras import layers, Input, Model
from data_processors import audio, utils, dataset
from typing import Tuple, Sequence
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# https://stackoverflow.com/questions/54030842/character-lstm-keeps-generating-same-character-sequence
def sample(preds, temperature=0.4):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def predict_midi_events(
    audio_input,
    encoder: Model,
    decoder: Model,
    num_decoder_tokens: int,
    target_token_index: dict,
    max_decoder_seq_length: int,
) -> Sequence[int]:
    """
    Given an audio input, predict up to `max_decoder_seq_length` midi events
    """
    audio_input = audio_input[np.newaxis, ...] # hack to make dimensionality work

    # Encode the input as state vectors.
    states_value = encoder.predict(audio_input)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((len(audio_input), 1, num_decoder_tokens))

    reverse_target_event_index = dict((i, event) for event, i in target_token_index.items())
    decoded_events = []

    while True:
        output_tokens, h, c = decoder.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = sample(output_tokens[0, -1, :])
        sampled_event = reverse_target_event_index[sampled_token_index]
        decoded_events.append(sampled_event)

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_events) > max_decoder_seq_length:
            break

        # Update the target sequence (of length 1).
        target_seq = np.zeros((len(audio_input), 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]

    return decoded_events

def extract_coders(model, latent_dim: int) -> Tuple[Model, Model]:
    """
    Given an instance of the model, extracts the encoder and decoder sub-networks
    """
    encoder_inputs = model.input[0]  # input_1
    _, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder = Model(encoder_inputs, encoder_states, name='encoder')

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states,
        name='decoder'
    )

    return encoder, decoder

def predict(
    model,
    scaled_inputs,
    latent_dim: int,
    num_decoder_tokens: int,
    target_token_index: dict,
    max_decoder_seq_length: int,
    audio_duration: int = 10
) -> Sequence[Sequence[int]]:
    """
    Use the model to predict a sequence of midi event batches (one for each batch of audio).

    Due to high processing time, limit the sequence reconstruction to only `audio_duration` seconds.
    """
    encoder, decoder = extract_coders(model, latent_dim)
    num_batches = int(audio_duration / dataset.SECONDS_PER_GROUP)

    midi_batches = [
        predict_midi_events(
            input,
            encoder=encoder,
            decoder=decoder,
            num_decoder_tokens=num_decoder_tokens,
            target_token_index=target_token_index,
            max_decoder_seq_length=max_decoder_seq_length
        ) for input in scaled_inputs[:num_batches]
    ]
    
    return midi_batches

def build_model(num_decoder_tokens: int, latent_dim: int=16) -> Model:
    """
    Adapted from Keras language translation tutorial at
    https://keras.io/examples/nlp/lstm_seq2seq/
    """
    frame_length = int(audio.DEFAULT_HOP_WIDTH)

    encoder_inputs = Input(shape=(None, frame_length), name='encoder_input')
    encoder = layers.LSTM(latent_dim, return_state=True, name='encoder_lstm')
    _, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
    
    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = layers.Dense(num_decoder_tokens, activation="softmax", name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["acc"]
    )

    return model

def train(
    song_count_limit: int, 
    latent_dim: int, 
    batch_size: int, 
    num_epochs: int, 
    shuffle_data:bool=True, 
    patience: int = None
) -> Tuple[Model, int, dict]:
    """
    Train the model on the dataset.
    """

    inputs, outputs = dataset.generate_data_set(song_count_limit=song_count_limit, shuffle=shuffle_data)
    
    (scaled_inputs,
    num_decoder_tokens,
    decoder_input_data,
    decoder_target_data,
    target_token_index) = dataset.preprocess_data(inputs, outputs)

    model = build_model(num_decoder_tokens=num_decoder_tokens, latent_dim=latent_dim)
   
    model.summary()

    patience = patience if patience is not None else num_epochs

    print(f'Training on {len(inputs) * len(inputs[0])} samples')
    history = model.fit(
        [scaled_inputs, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor="loss", patience=patience)
        ]
    )
    utils.plot_history(history)

    return model, num_decoder_tokens, target_token_index