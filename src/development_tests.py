"""
This file contains tests that I wrote to verify my understanding
of the various library functions and ensure that my outputs were
formed in the way that I expected.

It is not code that is terribly valuable on its own, except to demonstrate
the use of the various libraries.
"""

from data_processors import utils, audio, midi
import note_seq

def test_note_sequence_coding():
    # Ensure we can recover the original midi file after conversion with NoteSequence
    ns = midi.midi_file_to_note_sequence('./data/midi/1727.mid')
    midi.note_sequence_to_midi_file(ns, 'moo.mid')

def test_event_coding():
    # Ensure we can recover the original midi file after conversion to note events
    samples = audio.wav_file_to_samples('./data/audio/1727.wav')
    _, audio_times = audio.samples_to_frames(samples)

    ns = midi.midi_file_to_note_sequence('./data/midi/1727.mid')
    ns = note_seq.apply_sustain_control_changes(ns)
    midi_times, midi_events = midi.note_sequence_to_events(ns)
    del ns.control_changes[:]

    events, _, _, _, _ = utils.encode_midi_events(audio_times, midi_times, midi_events)

    codec = utils.CODEC
    decoded_ns = midi.events_to_note_sequence(events, codec)
    midi.note_sequence_to_midi_file(decoded_ns, 'moo.mid')

def test_split_reconstruct_sequences():
    # Split and rejoin the merge sequences to see if the resulting midi is the same

    samples = audio.wav_file_to_samples('./data/audio/1727.wav')
    audio_frames, audio_times = audio.samples_to_frames(samples)

    ns = midi.midi_file_to_note_sequence('./data/midi/1727.mid')
    ns = note_seq.apply_sustain_control_changes(ns)
    midi_times, midi_events = midi.note_sequence_to_events(ns)
    del ns.control_changes[:]

    print("Splitting into subsequences")
    subsequences = note_seq.split_note_sequence(ns, 1)
    event_batches = []
    for i, subseq in enumerate(subsequences):
        subseq = note_seq.apply_sustain_control_changes(subseq)
        midi_times, midi_events = midi.note_sequence_to_events(subseq)
        del subseq.control_changes[:]

        events, _, _, _, _ = midi.encode_midi_events(audio_times, midi_times, midi_events)
        event_batches.append(events)

    reconstructed = midi.event_batches_to_note_sequence(event_batches, codec=utils.CODEC)

    midi.note_sequence_to_midi_file(reconstructed, 'moo.mid')