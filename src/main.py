import pdb
import sys

from data_processors import dataset, midi, utils
from machine_learning import network

def main(
    num_epochs: int=100,                    # number of epochs to train for
    batch_size: int=1,                      # batch size for training
    latent_dim: int=16,                     # latent dimension for the encoder/decoder
    max_decoder_sequence_length: int=10,    # maximum length of decoder sequences
    song_count_limit:int=1,                 # number of songs to use for training
    shuffle_data: bool=False,               # shuffle the data before training
    audio_duration: int=10,                 # max length of audio input for prediction
):
    if len(sys.argv) == 1:
        print('Please specify a commmand:')
        print('\ttrain')
        print('\tconvert [audio_file] [output_file_path]')
        return
    
    command = sys.argv[1]
    
    if command == 'train':
        """
        [Re]train the model
        """
        (model,
        num_decoder_tokens,
        target_token_index) = network.train(
            song_count_limit=song_count_limit,
            latent_dim=latent_dim,
            batch_size=batch_size,
            num_epochs=num_epochs,
            shuffle_data=shuffle_data
        )
        utils.save_model(model, num_decoder_tokens, target_token_index)

    elif command == 'convert':
        """
        Use the saved model to convert an audio file to a midi file.
        It will only attempt to reconstruct the beginning of the file, and it can take a while to complete.
        
        It will also be terrible.
        """
        if len(sys.argv) < 4:
            print('Please specify an audio file and output file path')
            return
        audio_file = sys.argv[2]
        output_file_path = sys.argv[3]
        
        scaled_inputs = dataset.audio_to_frame_batch(audio_file)
        model, num_decoder_tokens, target_token_index = utils.load_model()
        event_batches = network.predict(
            model,
            scaled_inputs,
            latent_dim=latent_dim,
            num_decoder_tokens=num_decoder_tokens,
            target_token_index=target_token_index,
            max_decoder_seq_length=max_decoder_sequence_length,
            audio_duration=audio_duration
        )
        midi.reconstruct_midi(event_batches, output_file_path)
    else:
        raise Exception('Unknown command: {}'.format(command))
    

if __name__ == '__main__':
    main()