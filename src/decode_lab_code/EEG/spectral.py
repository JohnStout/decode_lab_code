# spectral.py
# 
# Group of spectral based functions for LFP analysis
#
# Written by John Stout

from scipy.signal import welch, coherence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# run below in interactive window for troubleshooting
"""
    All inputs will be:
        >>> data
        >>> fs
        >>> frequency_range
"""

print("Cite Pynapple and PYNWB")

def filt():
    pass

def power(data, fs: float, frequency_range: list = [1,100]):

    """
    Run welch's power analysis

    Requires pynapple based loading

    Args:
        >>> data: numpy array
        >>> fs: sampling rate
        >>> frequency_range: list for range of frequency

    Returns: 
        >>> PSpect: power spectrum in frequency range
        >>> fSpec: frequency range in the defined band
        >>> PSpecLog: log transformed power spectrum
    """

    PSpec = []; PSpecLog = [] 
    for i in range(len(data)):

        # power spectrum
        f,Ptemp = welch(data,fs,nperseg=fs)

        # restrict data to 1-50hz for plot proofing
        #f[f>1]
        idxspec = np.where((f>frequency_range[0]) & (f<frequency_range[1]))
        fSpec = f[idxspec]
        PSpec = Ptemp[idxspec]

        # log10 transform
        PSpecLog = np.log10(PSpec)

    return PSpec, fSpec, PSpecLog

# TODO
def fieldfield_coherence(x, y, fs, nperseg=100, frequency_range: list = [1,100]):

    f, cxy = coherence(x,y,fs,nperseg=100)
    idx = np.logical_and(f>= frequency_range[0],f<=frequency_range[1])
    plt.plot(f[idx],cxy[idx],linewidth=2,color='r')
    plt.ylabel('Mean Squared Coherence')
    plt.xlabel('Frequency (hz)')

    return f, cxy

def spikefield_coherence():
    pass

def spikephase_entrainment():
    pass

def spikephase_precession():
    pass




    
    





