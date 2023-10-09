# ephys_tools 
# builds attributes for ephys processing

# packages
import os
import numpy as np
import matplotlib.pyplot as plt
from decode_lab_code.core.base import base
import pynapple as pyn

import re
from typing import Dict, Union, List, Tuple

# datetime stuff
from datetime import datetime
from dateutil import tz
from pathlib import Path
from uuid import uuid4

# pynwb
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.file import Subject

class ephys_tools(base):

    def rereference(self, rereference_mode: str = 'CAR', csc_attribute: str = 'csc_data'):

        """
        This code supports common average or common median rereferencing techniques, which
        are extremely useful techniques when working with a lot of recording channels from the same
        structure

        TODO: Implement an approach that supports matrix style or list style data inputs to self
        
        Args:
            rereference_mode: Common average rereference, 'CAR' is default.
                                Common medial rereference, 'CMR' is an option
            csc_attribute: csc_data is the default, but you could change if you have added an attribute or something
        
        Returns:
            self.data: rereferenced data

        """
        if 'CAR' in rereference_mode:
            mode = 'common average'
        elif 'CMR' in rereference_mode:
            mode = 'common median'

        # document history
        self.history.append("csc_data_reref: rereferenced using the "+mode)

        # get csc_data
        csc_data = getattr(self,csc_attribute)        

        # reformat to np.array
        csc_names = list(csc_data.keys())
        csc_array = np.zeros(shape=(len(csc_data[csc_names[0]]),len(csc_names)))
        for csci in range(len(csc_names)):
            csc_array[:,csci] = csc_data[csc_names[csci]]
        
        # collect the common '' rereference values
        if 'CAR' in rereference_mode:
            ref = np.mean(csc_array,axis=1)
        elif 'CMR' in rereference_mode:
            ref = np.median(csc_array,axis=1)

        # rereference via subtraction and save output to self
        self.csc_data_reref = dict()
        for i in range(csc_array.shape[1]):
            self.csc_data_reref[csc_names[i]] = csc_array[:,i]-ref

    def butterworth_filter(self, lowcut = None, highcut = None, csc_attribute: str = 'csc_data'):

        """
        Third degree butterworth filter

        Args:
            data: vector array of floating point numbers
            lowcut: lowcut for bandpass filtering
            highcut: highcut for pandpass filtering
            csc_attribute: csc_data is the default, but you could change if you have added an attribute or something

        Returns:
            signal_filtered: filtered signal

        """

        # get csc_data
        csc_data = getattr(self,csc_attribute)

        # TODO: Make it so that this works on filtered or non-filtered signal
        # [i for i in self.history if 'ref' in i]
        self.signal_filtered = dict()
        csc_ids = csc_data.keys()
        for csci in csc_ids:
            #print(csci)
            self.signal_filtered[csci] = butter_filt(csc_data[csci],bandpass=[lowcut,highcut],fs=self.fs)

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

    # lets inherit the __init__ from process_signal
    def lfp_plotter(self, time_range: list = [0, 1], csc_attribute: str = 'csc_data', csc_names = None, color = None):

        """
        Generate plotting function that plots as many rows as there are signals

        Optional:
            time_range: list telling the figure what to plot. Default is the first second.
            csc_attribute: name of attribute you want to plot. Default is self.csc_data.
                this is useful if you've changed the data or added an attribute that you want
                to plot
            csc_names: names of csc channels to plot as denoted by self.data.csc_data_names
            color: default is a single value, 'k'. This can take as many colors as there are data points.

        """

        # get csc_data
        csc_data = getattr(self,csc_attribute)

        if color is None:
            temp_color = 'k'
            color = [temp_color[0] for i in range(len(self.csc_data))]

        if csc_names is None:
            csc_names = list(self.csc_data.keys())
        elif type(csc_names) is str:
            csc_temp = []
            csc_temp[0] = csc_names; del csc_names
            csc_names = csc_temp; del csc_temp
        
        # get sampling rate - this method and class will be inherited by ioreaders
        fs = self.csc_data_fs
        fig, axes = plt.subplots(nrows=len(csc_names),ncols=1,)
        idx = [int(time_range[0]*fs),int(time_range[1]*fs)]
        x_data = np.linspace(time_range[0],time_range[1],int(fs*(time_range[1]-time_range[0])))
        for i in range(len(csc_names)):
            if len(axes) > 1:
                if i == len(csc_names)-1:
                    axes[i].plot(x_data, csc_data[csc_names[i]][idx[0]:idx[1]],color[i],linewidth=0.75)
                    axes[i].set_xlabel("Time (sec)")
                else:
                    axes[i].plot(x_data, csc_data[csc_names[i]][idx[0]:idx[1]],color[i],linewidth=0.75)
                    #axes[i].xlabel("Time (sec)")
                axes[i].yaxis.set_tick_params(labelsize=8)
                axes[i].xaxis.set_tick_params(labelsize=8)
            else:
                axes.plot(x_data, csc_data[csc_names[i]][idx[0]:idx[1]],color[i],linewidth=0.75)
                axes.set_xlabel("Time (sec)")  
                axes.yaxis.set_tick_params(labelsize=8)
                axes.xaxis.set_tick_params(labelsize=8)                              
            #var(axes[i])
        fig.show()

    def artifact_detect(self):
        # TODO: add some artifact detection code to identify timestamps for potential exlusion
        
        pass