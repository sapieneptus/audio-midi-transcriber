import matplotlib.pyplot as plt
import os
import json
import numpy as np
from keras import Model, models
from mt3_modified import vocabularies
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

def create_codec():
    vocab_config = vocabularies.VocabularyConfig()
    codec = vocabularies.build_codec(vocab_config)
    return codec

CODEC = create_codec()

def scale_inputs(inputs) -> Tuple[MinMaxScaler, np.ndarray]:
    """
    Use a MinMax scaler to scale the inputs to the range [0, 1].
    Outputs will all be in the range [0, 1] so this keeps them in the same scale
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(inputs.reshape(-1, inputs.shape[-1])).reshape(inputs.shape)
    return scaler, scaled

def plot_history(history):
    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.plot(epochs, acc, color='b', label='Training acc')
    plt.plot(epochs, loss, color='g', label='Training loss')
    plt.plot(epochs, val_acc, color='purple', label='Validation acc')
    plt.plot(epochs, val_loss, color='black', label='Validation loss')
    plt.title(f'loss & accuracy')
    plt.legend()

    plt.show()

def load_model(save_dir: str='saved_model') -> Tuple[Model, int, dict]:
    filepath = os.path.join(save_dir, 'model.h5')
    model = models.load_model(filepath)

    with open(os.path.join(save_dir, 'meta.json'), 'r') as f:
        data = json.load(f)

    num_decoder_tokens = data['num_decoder_tokens']
    target_token_index = { int(k): int(v) for k, v in data['target_token_index'].items() }
    return model, num_decoder_tokens, target_token_index

def save_model(model: Model, num_decoder_tokens: int, target_token_index: dict, save_dir: str='saved_model'):
    filepath = os.path.join(save_dir, 'model.h5')
    model.save(filepath)

    data = {
        'num_decoder_tokens': num_decoder_tokens,
        'target_token_index': { int(k): int(v) for k, v in target_token_index.items() }
    }

    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(data, f)

    print(f'Model saved to {filepath}')
