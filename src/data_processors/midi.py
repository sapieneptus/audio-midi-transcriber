import note_seq

from mt3_modified import note_sequences, run_length_encoding, event_codec
from typing import Sequence, Tuple
from data_processors import utils
import pdb

def midi_file_to_note_sequence(midi_path) -> note_seq.NoteSequence:
    """
    Convert a midi file to a list of onset and offset times and pitches
    """
    print(f"Converting midi file to note sequence: {midi_path}")
    ns = note_seq.midi_file_to_note_sequence(midi_path)
    # Note: MT3 calls 'apply_sustain_control_changes', but it seems like this results in a worse transcription ?
    return ns

def note_sequence_to_events(ns: note_seq.NoteSequence) -> Tuple[Sequence[float], Sequence[note_sequences.NoteEventData]]:
    return note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns)

def events_to_note_sequence(events, codec: event_codec.Codec = utils.CODEC) -> note_seq.NoteSequence:
    print("converting events to note sequence")
    decoding_state = note_sequences.NoteDecodingState()
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=events, start_time=0, max_time=None,
        codec=codec, decode_event_fn=note_sequences.decode_note_event)
    ns = note_sequences.flush_note_decoding_state(decoding_state)
    
    print(f'Dropped {dropped_events} events')
    print(f'Invalid ids: {invalid_ids}')
    return ns

def event_batches_to_note_sequence(event_batches, codec: event_codec.Codec=utils.CODEC) -> note_seq.NoteSequence:
    print("converting event batches to note sequence")
    decoding_state = note_sequences.NoteDecodingState()
    total_invalid_ids = 0
    total_dropped_events = 0

    for events in event_batches:
        invalid_ids, dropped_events = run_length_encoding.decode_events(
            state=decoding_state,
            tokens=events,
            start_time=decoding_state.current_time,
            max_time=None,
            codec=codec,
            decode_event_fn=note_sequences.decode_note_event
        )
        total_invalid_ids += invalid_ids
        total_dropped_events += dropped_events
        
    ns = note_sequences.flush_note_decoding_state(decoding_state)
    
    print(f'Dropped {total_dropped_events} events')
    print(f'Invalid ids: {total_invalid_ids}')
    return ns

def note_sequence_to_midi_file(ns: note_seq.NoteSequence, midi_path: str):
    """
    Convert a list of onset and offset times and pitches to a midi file
    """
    print(f"Converting events to midi file: {midi_path}")

    return note_seq.midi_io.note_sequence_to_midi_file(ns, midi_path)

def reconstruct_midi(event_batches, output_file: str):
    """
    Convert a set of event batches to a midi file
    """
    reconstructed = event_batches_to_note_sequence(event_batches)
    note_sequence_to_midi_file(reconstructed, output_file)

def encode_midi_events(
    audio_frame_times: Sequence[float],
    midi_event_times: Sequence[float],
    midi_event_values: Sequence[note_sequences.NoteEventData]
) -> Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    """
    """

    events, event_start_indices, event_end_indices, state_events, state_event_indices = run_length_encoding.encode_and_index_events(
        state=note_sequences.NoteEncodingState(),
        event_times=midi_event_times,
        event_values=midi_event_values,
        encode_event_fn=note_sequences.note_event_data_to_events,
        codec=utils.CODEC,
        frame_times=audio_frame_times,
        encoding_state_to_events_fn=note_sequences.note_encoding_state_to_events
    )
    return events, event_start_indices, event_end_indices, state_events, state_event_indices