import numpy as np
import librosa
from IPython.display import Audio
from librosa import display

'''

Perform Pre-emphasis

source: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

'''

def pre_emphasis(signal, pre_emphasis_coeff = 0.95):  # most commonly used values are 0.95 and 0.97
    pre_emphasis_coeff = pre_emphasis_coeff
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])
    return emphasized_signal

def MFCC(emphasized_signal):
    mel = librosa.feature.mfcc(emphasized_signal)
    mel_scaled = scale(mel, axis = 1)
    return mel_scaled

def Zero_crossing_rate(emphasized_signal, eps = 0.001):     # To prevent silence being mistaken as noise
    zero_crossing = librosa.feature.zero_crossing_rate(emphasized_signal + eps)
    zero_crossing = zero_crossing[0]
    return zero_crossing

def Spectral_centroid(emphasized_signal, eps = 0.001):
    spec_centroid = librosa.feature.spectral_centroid(emphasized_signal + eps)
    spec_centroid = spec_centroid[0]
    return spec_centroid

def Spectral_rolloff(emphasized_signal, eps = 0.001):
    spec_rolloff = librosa.feature.spectral_rolloff(emphasized_signal + eps)
    spec_rolloff = spec_rolloff[0]
    return spec_rolloff

def Chroma_feat(emphasized_signal):
    chroma = librosa.feature.chroma_stft(emphasized_signal, hop_length=1024)
    return chroma
