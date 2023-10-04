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
import matplotlib.pyplot as plt
from copy import deepcopy

# TODO: implement base inheritence to at least keep a track record of where this dataset is coming from

class process_signal:

    """
    process_signal: a class designed for working with LFP/EEG data
    
    """

    def __init__(self, data: float, fs: int):

        """
        Args:
            data: numpy array vector. Will be reshaped as needed.
            fs: sampling rate as an integer

            TODO: implement a way to detect if input is a list of signals
        """

        # tracking signal processing steps
        self.history = [] # list

        # make sure we don't work with wrong kind of data
        if type(data) is dict:
            self.data = dict()
            csc_ids = data.keys()
            for csci in csc_ids:
                if len(data[csci].shape) > 1:
                    self.data[csci] = np.reshape(data[csci],len(data[csci]))
            self.history.append("Reshaped data arrays to 1D")

        # if just working with a single array, convert to dict
        if type(data) is np.array:
            # TODO: Make this compatible with dictionary type processing

            # check the shape of the data and fix as needed
            if len(data.shape) > 1:

                # you have to reshape if you want any filtering to actually work
                self.data = np.reshape(data,len(data))

                # store a history of what you've done
                self.history.append("Reshaped data array to 1D array")

            else:
                self.data = data

        # store sampling rate
        self.fs = fs

    def rereference(self, rereference_mode: str = 'CAR'):
        """
        This code supports common average or common median rereferencing techniques, which
        are extremely useful techniques when working with a lot of recording channels from the same
        structure

        TODO: Implement an approach that supports matrix style or list style data inputs to self
        
        Args:
            rereference_mode: Common average rereference, 'CAR' is default.
                                Common medial rereference, 'CMR' is an option
        
        Returns:
            self.data: rereferenced data

        """
        if 'CAR' in rereference_mode:
            mode = 'common average'
        elif 'CMR' in rereference_mode:
            mode = 'common median'

        # rereference each signal
        if type(self.data) is dict:

            # document history
            self.history.append("signal_reref: rereferenced using the "+mode)

            # reformat to np.array
            csc_names = list(self.data.keys())
            csc_array = np.zeros(shape=(len(self.data[csc_names[0]]),len(csc_names)))
            for csci in range(len(csc_names)):
                csc_array[:,csci] = self.data[csc_names[csci]]
            
            # collect the common '' rereference values
            if 'CAR' in rereference_mode:
                ref = np.mean(csc_array,axis=1)
            elif 'CMR' in rereference_mode:
                ref = np.median(csc_array,axis=1)

            # rereference via subtraction and save output to self
            self.signal_rereferenced = dict()
            for i in range(csc_array.shape[1]):
                self.signal_rereferenced[csc_names[i]] = csc_array[:,i]-ref

    def butterworth_filter(self, data = None, lowcut = None, highcut = None):

        """
        Third degree butterworth filter

        Args:
            data: vector array of floating point numbers
            lowcut: lowcut for bandpass filtering
            highcut: highcut for pandpass filtering

        Returns:
            signal_filtered: filtered signal

        """

        # TODO: Make it so that this works on filtered or non-filtered signal
        # [i for i in self.history if 'ref' in i]

        if data is None:
            data = self.data

        if type(data) is dict:
            self.signal_filtered = dict()
            csc_ids = self.data.keys()
            for csci in csc_ids:
                #print(csci)
                self.signal_filtered[csci] = butter_filt(data[csci],bandpass=[lowcut,highcut],fs=self.fs)
                           
        elif type(data) is np.array:
            self.signal_filtered = butter_filt(data,bandpass=[lowcut,highcut],fs=self.fs)
            
        # record
        self.history.append("signal_filtered: third degree butterworth filtering between "+str(lowcut)+" to "+str(highcut))

# functions used for methods in object
def butter_filt(signal: float, bandpass: list, fs: float):
    """
    butterworth filter
    Args:
        signal: input signal as a floating point variable
        bandpass: list of bandpass filter, bandpass = [4, 12] for theta
        fs: sampling rate

    returns:
        filt: filtered signal
    """
               
    # parameter assignment for filtering
    nyq = 0.5 * fs # nyquist theorum
    low = bandpass[0] / nyq
    high = bandpass[1] / nyq
    order = 3 # third degree butterworth filter

    # third degree butterworth filter
    sos = butter(N = 3, Wn = [low,high], analog = False, btype='band', output='sos')
    signal_filtered = sosfilt(sos, signal)

    return signal_filtered 

# TODO: a plot_signal class to quickly plot and visualize data

# lets inherit the __init__ from process_signal
def multi_plotter(data: list, fs: int, time_range: list = [0, 1], color: str = 'k'):
    """
    Generate plotting function that plots as many rows as there are signals

    Args:
        data: list of csc data to plot
        fs: sampling rate of csc data
    
    Optional:
        time_range: list telling the figure what to plot. Default is the first second.
        color: default is a single value, 'k'. This can take as many colors as there are data points.

    """
    if len(color) == 1:
        group_color = [color[0] for i in range(len(data))]

    fig, axes = plt.subplots(nrows=len(data),ncols=1)
    key_names = list(data.keys())
    idx = [int(time_range[0]*fs),int(time_range[1]*fs)]
    for i in range(len(data)):
        if i == len(data)-1:
            x_data = np.linspace(time_range[0],time_range[1],int(fs*(time_range[1]-time_range[0])))
            axes[i].plot(x_data, data[key_names[i]][idx[0]:idx[1]],color[i],linewidth=0.75)
            axes[i].set_xlabel("Time (sec)")
        else:
            axes[i].plot(data[key_names[i]][idx[0]:idx[1]],color[i],linewidth=0.75)
            #axes[i].xlabel("Time (sec)")
    fig.show()