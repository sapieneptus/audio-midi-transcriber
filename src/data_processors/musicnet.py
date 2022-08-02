import os
import shutil

def rename_data(audio_dir: str='./musicnet', midi_dir: str='./musicnet_midis', output_audio_dir: str='data/audio', output_midi_dir: str='data/midi'):
    """
    The musicnet dataset is in a somewhat opinionated and inconvenient file structure.
    We'll rename and move things so we end up with two directories, audio and midi,
    with the files named by just their ID (plus .mid or .wav as appropriate)
    """
    # Ensure output dirs exist
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_midi_dir, exist_ok=True)

    # Copy and rename all midi files
    for composer in os.listdir(midi_dir):
        if composer == '.DS_Store': # Ignore OSX hidden files
            continue
    
        for midi_file in os.listdir(os.path.join(midi_dir, composer)):
            if midi_file == '.DS_Store':
                continue

            id = midi_file.split('_')[0]
            shutil.copy(os.path.join(midi_dir, composer, midi_file), os.path.join(output_midi_dir, f'{id}.mid'))
    
    # Move and rename all audio files
    audio_dirs = [os.path.join(audio_dir, 'train_data'), os.path.join(audio_dir, 'test_data')]
    for dir in audio_dirs:
        if dir == '.DS_Store':
            continue

        for filename in os.listdir(dir):
            if filename == '.DS_Store':
                continue
            shutil.copy(os.path.join(dir, filename), os.path.join(output_audio_dir, filename))

