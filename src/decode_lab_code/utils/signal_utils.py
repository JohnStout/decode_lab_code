# signal_utils
# 
# This code is meant to perform signal processing procedures for LFP/EEG
#
# The core of this code is "process_signal", an object that represents our data
# as a 1D numpy array and allows for modification with methods like 3rd degree butter
# worth filtering.
#
# Each time you add an attribute to your object, a history attribute will be updated.

from scipy.signal import butter, sosfilt, lfilter
import numpy as np

class process_signal:

    """
    process_signal: a class designed for working with LFP/EEG data
    
    """

    def __init__(self, data: float):

        """
        Args:
            data: numpy array vector. Will be reshaped as needed.
        """

        # tracking signal processing steps
        self.history = [] # list

        # check the shape of the data and fix as needed
        if len(data.shape) > 1:

            # you have to reshape if you want any filtering to actually work
            self.data = np.reshape(data,len(data))

            # store a history of what you've done
            self.history.append("Reshaped data array to 1D array")

        else:
            self.data = data

    def butterworth_filter(self, lowcut: int, highcut:int, fs: int):

        """
        Third degree butterworth filter

        Args:
            data: vector array of floating point numbers
            lowcut: lowcut for bandpass filtering
            highcut: highcut for pandpass filtering
            fs: sampling rate

        Returns:
            signal_filtered: filtered signal

        """

        # filter signal
        nyq = 0.5 * fs # nyquist theorum
        low = lowcut / nyq
        high = highcut / nyq
        order = 3 # third degree butterworth filter

        # butter filter
        sos = butter(N = order, Wn = [low,high], analog = False, btype='band', output='sos')
        self.signal_filtered = sosfilt(sos, self.data)
        resh = np.reshape(self.data, len(self.data))
        filt = sosfilt(sos, resh)

        # record
        self.history.append("signal_filtered: third degree butterworth filtering between "+str(lowcut)+" to "+str(highcut))
