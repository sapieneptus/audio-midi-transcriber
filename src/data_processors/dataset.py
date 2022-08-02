from data_processors import audio, midi, utils

import math
import note_seq
import os
import numpy as np
import pdb

from typing import Tuple, Sequence

SECONDS_PER_GROUP = 0.2

def group_samples(audio_frames: Sequence[float], num_groups: int, frames_per_sec: int) -> np.ndarray:
    num_usable_samples = int(num_groups * frames_per_sec * SECONDS_PER_GROUP)
    groups = np.split(audio_frames[:num_usable_samples], num_groups)
    return groups

def group_events(ns: note_seq.NoteSequence, audio_times: Sequence[float], num_groups: int, frames_per_sec: int) -> np.ndarray:
    frames_per_sec = int(audio.frames_per_second())
    num_usable_samples = int(num_groups * frames_per_sec * SECONDS_PER_GROUP)
    audio_time_groups = np.split(audio_times[:num_usable_samples], num_groups)

    subsequences = note_seq.split_note_sequence(ns, SECONDS_PER_GROUP)[:num_groups] # truncate in case the midi is longer than the audio
    groups = []
    for i, subseq in enumerate(subsequences):
        frame_times = audio_time_groups[i]
        # Shift the frame times to be 0-indexed
        frame_times -= (i + 1)
        subseq = note_seq.apply_sustain_control_changes(subseq)
        midi_times, midi_events = midi.note_sequence_to_events(subseq)
        del subseq.control_changes[:]

        events, _, _, _, _ = midi.encode_midi_events(frame_times, midi_times, midi_events)
        groups.append(events)
    
    return groups

def audio_to_frame_batch(audio_file: str) -> np.ndarray:
    """
    Load a wav file and convert to a set of input vectors.
    Applies scaling as well.
    """
    samples = audio.wav_file_to_samples(audio_file)
    audio_frames, _ = audio.samples_to_frames(samples)

    frames_per_sec = int(audio.frames_per_second())
    audio_duration = len(audio_frames) // frames_per_sec
    num_groups = int(audio_duration / SECONDS_PER_GROUP)
    
    frame_batch = group_samples(audio_frames, num_groups, frames_per_sec)
    frame_batch = np.array(frame_batch, dtype=object).astype(np.float32)

    int_inputs = note_seq.audio_io.float_samples_to_int16(frame_batch)
    _, scaled_inputs = utils.scale_inputs(int_inputs)

    return scaled_inputs

def create_data_pair(file_id: str, data_dir: str='data') -> Tuple[np.ndarray, np.ndarray]:
    audio_path = os.path.join(data_dir, 'audio', file_id + '.wav')
    midi_path = os.path.join(data_dir, 'midi', file_id + '.mid')

    samples = audio.wav_file_to_samples(audio_path)
    audio_frames, audio_times = audio.samples_to_frames(samples)

    ns = midi.midi_file_to_note_sequence(midi_path)
    ns = note_seq.apply_sustain_control_changes(ns)

    midi_times, _ = midi.note_sequence_to_events(ns)
    midi_duration = math.ceil(max(midi_times)) # they're not in order so we need to use `max`

    frames_per_sec = int(audio.frames_per_second())
    audio_duration = len(audio_frames) // frames_per_sec

    # The two tracks may not have exactly the same duration. It can go a either way, so we'll
    # just use the shorter of the two. Note: this isn't perfect, but it'll have to do for now.
    duration = min(midi_duration, audio_duration)
    duration = int(duration - (duration % SECONDS_PER_GROUP)) # cut any trailing seconds
    num_groups = int(duration / SECONDS_PER_GROUP)
    
    sample_groups = group_samples(audio_frames, num_groups, frames_per_sec)
    midi_groups = group_events(ns, audio_times, num_groups, frames_per_sec)

    assert len(sample_groups) == len(midi_groups)

    return sample_groups, midi_groups

# An implementation of runlength encoding, taken from Stack Overflow
# Not currently in use, but was used in experiments to compress "shifts" in the midi events
# https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])

def preprocess_data(inputs, outputs):
    """
    Adapted from Keras language translation tutorial at
    https://keras.io/examples/nlp/lstm_seq2seq/
    """
    int_inputs = note_seq.audio_io.float_samples_to_int16(inputs)
    _, scaled_inputs = utils.scale_inputs(int_inputs)

    output_vocab = set()
    [output_vocab.update(output) for output in outputs]
    
    num_decoder_tokens = len(output_vocab)
    max_decoder_seq_length = max([len(output) for output in outputs])

    print(f'Number of decoder tokens: {num_decoder_tokens}')
    print(f'Max decoder sequence length: {max_decoder_seq_length}')

    target_token_index = dict([(char, i) for i, char in enumerate(output_vocab)])
    decoder_input_data = np.zeros(
        (len(outputs), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(outputs), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

    for i, target_events in enumerate(outputs):
        for t, event in enumerate(target_events):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[event]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[event]] = 1.0

    return scaled_inputs, num_decoder_tokens, decoder_input_data, decoder_target_data, target_token_index

def generate_data_set(song_count_limit: int = None, shuffle: bool = True, data_dir: str='data') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a data set mapping subseries of audio frames to a set of midi events.
    The samples are grouped into batches of size audio.DEFAULT_HOP_WIDTH - the idea
    was originally to preprocess these batches using a dsp transformation (e.g. calculating 
    amplitude or something) but this is not implemented yet.
    """
    inputs = []
    targets = []
    songs_processed = 0

    # get all wavs in the data directory
    files = [f for f in os.listdir(os.path.join(data_dir, 'audio')) if f.endswith('.wav')]

    if shuffle:
        np.random.shuffle(files)

    for file in files:
        id = file.split('.')[0]
        print(f"Processing Song: {id}")
        sample_series, events = create_data_pair(id)
        inputs += sample_series

        targets += events

        songs_processed += 1
        if song_count_limit is not None and songs_processed >= song_count_limit:
            break
    
    return np.array(inputs, dtype=object).astype(np.float32), np.array(targets, dtype=object)