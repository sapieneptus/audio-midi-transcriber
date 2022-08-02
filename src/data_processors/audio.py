import note_seq
import tensorflow as tf
import numpy as np

from typing import Tuple, Sequence

DEFAULT_SAMPLE_RATE = 16000 # Normal audio is 44.1khz
DEFAULT_HOP_WIDTH = 80
DEFAULT_NUM_MEL_BINS = 512

# The following function is taken from the mt3 library jupyter notebook
# https://github.com/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb
def samples_to_dataset(samples) -> tf.data.Dataset:
    """Create a TF Dataset of spectrograms from input audio."""
    frames, frame_times = samples_to_frames(samples)
    return tf.data.Dataset.from_tensors({
        'inputs': frames,
        'input_times': frame_times,
    })

# The following function is adapted from the mt3 library jupyter notebook
# https://github.com/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb
def samples_to_frames(samples) -> Tuple[Sequence[int], Sequence[float]]:
    """Compute spectrogram frames from audio."""
    frame_size = DEFAULT_HOP_WIDTH
    padding = [0, frame_size - len(samples) % frame_size]
    audio = np.pad(samples, padding, mode='constant')
    frames = split_audio(audio)
    num_frames = len(audio) // frame_size
    times = np.arange(num_frames) / frames_per_second()
    return frames, times

def frames_per_second():
    return DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH

def split_audio(samples, hop_width: int=DEFAULT_HOP_WIDTH):
    return tf.signal.frame(
        samples,
        frame_length=hop_width,
        frame_step=hop_width,
        pad_end=True
    )

def wav_file_to_samples(wav_path):
    with open(wav_path, 'rb') as f:
        return note_seq.audio_io.wav_data_to_samples_librosa(f.read(), sample_rate=DEFAULT_SAMPLE_RATE)
