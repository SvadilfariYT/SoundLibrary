import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

audio_files = []

def loadAudioFiles(path, accepted_filetypes):
    """
    This function loads all Audio files in a given folder into an array.

    params:
    path: file path

    returns:
    audio_files: all loaded files
    """
    for type in accepted_filetypes:
        loaded = glob(path+ '*.' + type)
        if len(loaded) != 0:
            for file in loaded:
                audio_files.append(file)
        print("Loaded " + str(len(loaded)) + " file/s of type " + type)

    for audio in audio_files:
        print(audio)
    
    print("Totally loaded: " + str(len(audio_files)) + " audio files.")

    return audio_files

def plotWaveform(audio_file):
    """
    This function plots the waveform of a given audio file.

    params:
    audio_file: audio file to be plotted

    returns:
    None
    """
    # Load Audio File
    y, sr = librosa.load(audio_file)

    # Plot Waveform
    pd.Series(y).plot(figsize=(10,5), lw=1, title="Tram Audio Example", color=color_pal[0])

# function to trim the audio file to a given length
def trimAudioFile(audio_file, top_db):
    """
    This function trims an audio file to a given length.

    params:
    audio_file: audio file to be trimmed
    length: length to trim to
    top_db: The threshold (in decibels) below reference to consider as silence

    returns:
    trimmed_audio_file: trimmed audio file
    """
    # Load Audio File
    y, sr = librosa.load(audio_file)

    # Trim Audio File
    y_trimmed = librosa.effects.trim(y, top_db=top_db, frame_length=512, hop_length=64)


    return y_trimmed

# function to plot the spectrogram of an audio file
def plotSpectrogram(audio_file, y, x_axis="time", y_axis="linear", cmap="gray_r"):
    """
    This function plots the spectrogram of a given audio file.

    params:
    audio_file: audio file to be plotted

    returns:
    None
    """
    # Load Audio File
    y, sr = librosa.load(audio_file)
    
    # Plot Spectrogram
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis=x_axis, y_axis=y_axis, cmap=cmap)
    plt.axis('off')
    plt.margins(0)
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge

    plt.savefig("test.png", bbox_inches='tight')

# MAIN METHOD
if __name__ == '__main__':
    # Load Audio Files
    accepted_filetypes = ['WAV','MP3']
    loadAudioFiles('', accepted_filetypes)

    for audio in audio_files:
        plotSpectrogram(audio, trimAudioFile(audio, 40))