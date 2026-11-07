'''
Time Dependent Fourier Transform (Spectrogram)
Bryant Moquist
'''

import numpy as np

def tdft(audio, srate, windowsize, windowshift,fftsize):

    """Calculate the real valued fast Fourier transform of a segment of audio multiplied by a 
    a Hamming window.  Then, convert to decibels by multiplying by 20*log10.  Repeat for all
    segments of the audio."""

    windowsamp = int(windowsize*srate)
    shift = int(windowshift*srate)
    if windowsamp <= 0 or shift <= 0:
        raise ValueError('windowsize/windowshift is too small for the given sample rate')

    audio = np.asarray(audio)
    #Downmix multi-channel (e.g. stereo) audio to mono so the window can be applied.
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if len(audio) < windowsamp:
        raise ValueError('audio is shorter than one analysis window')

    window = np.hamming(windowsamp)
    tiny = np.finfo(float).tiny
    spectrogram = np.array([20*np.log10(np.maximum(abs(np.fft.rfft(window*audio[i:i+windowsamp],fftsize)),tiny))
                     for i in range(0, len(audio)-windowsamp+1, shift)])
    return spectrogram
