- [An ok audio transcriber](#an-ok-audio-transcriber)
- [Installing Dependencies](#installing-dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Using](#using)
- [References](#references)
- [MT3](#mt3)

# An ok audio transcriber


# Installing Dependencies

There's no easy way to say this - you're on your own for dependency installation :( . I am running an M1 mac which means I have some hardware constraints (since native binaries need to be arm64-compatible). This required me to split my dependency installation between conda, pip, and some manually installed wheels. Thanks AAPL.

You can find the full list of dependencies between `requirements-pip.txt` and `requirements-conda.txt` as well as any .whl bundles that exist in the `src` directory. If you're running on linux or an older mac, you can probably make a simple `requirements.txt` file from these.

# Data Preprocessing

First you'll need to download the musicnet midis and audio from the [Musicnet dataset](https://musicnet-inspector.github.io/). Untar them in `src` directory. There should be a `musicnet_midis` folder containing subfolders corresponding to composer names, each with a set of midi files. There shoudl be a `musicnet` folder which contains `train_data` and `test_data` (as well as `train_labels` and `test_labels`, but we'll ignore those), each containing a set of audio files.

We want to create a simple parallel file structure so we can have a set of files in the form `./data/midi/<id>.mid` and `./data/audio/<id>.wav`.

At this point, after manually inspecting some of the midi and audio pairs, I realized a problem: the lengths were not the same. This means I couldn't make a stable mapping of a fixed number of samples to a fixed expected output window (e.g. 16000 samples per 1 second output or similar). If I use the files as-is, it means I would need to have both a variable input size and output size, since the time ratio is different for every midi-wav pair.

So much for 'nice clean dataset'.

Fortunately, I am left with one option which may work: convert the midi files to wav (using any number of digital instrument libraries) and use those as my training inputs. They won't be nearly as human as the actual performance recordings and will likely suffer testing errors because midi-generated wavs are extremely clean audio, unlike many real-world recorded audio samples. Still, it's a starting point, and if it works I could as a next step add noise / distortions to the generated wavs in order to make the model more robust to real world imperfections.

Anyway, the next step in data preparation is to convert all midi files to wavs on the command line. You will need the following packages:
- `libsndfile` 
-  `fluidsynth` 
(available from `homebrew` on mac)

You will also need a Soundfont to give voicings to the midi. I used "Essential Keys-sforzando-v9.6" available from https://sites.google.com/site/soundfonts4u/.

```bash
mv data/audio data/original_audio # we'll be replacing the 'data/audio' dir

for f in $(ls data/midi); do
  wav_filename=$(echo $f | sed s/.mid/.wav/g)
  echo "Converting ${f} to ${wav_filename}"
  soundfont=./Essential\ Keys-sforzando-v9.6.sf2
  fluidsynth -F "./data/audio/$wav_filename" ${soundfont} "./data/midi/$f"
done 
```
You may see a warning about no preset on channel 9 (the drum channel). This is because the soundfont has no drums, nor 
does it need them as none of the tracks have drums.

Note that the results may not be 100% identical in terms of duration (e.g. if the instrument voicing the last note has a tail, 
then the resulting wav file may be a fraction of a second longer than the midi counterpart which wouldn't account for 
real instrument decay). However they will be close enough that we can just add some zero padding if necessary.

Note that, for some reason, file 2570 resulted in an empty wav file and could not be used.

# Training

# Using

# References

- [Musicnet dataset](https://musicnet-inspector.github.io/)
- [Google mt3 model](https://github.com/magenta/mt3)

# MT3

TODO