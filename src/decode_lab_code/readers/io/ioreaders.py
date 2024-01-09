# ioreaders
#
# Purpose of this code is to ready your data for analysis or convert file types
#
# For calcium imaging, this code converts and processes your data to .tif files for caiman
#
# written by John Stout

# get some packages
from datetime import datetime
from dateutil import tz
from pathlib import Path
from uuid import uuid4
import copy

import re
import os
import pandas as pd
import json
import glob
import psutil
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import imageio

# can I import this later??
import cv2

# numpy
import numpy as np

# loading neo package
from decode_lab_code.readers.helper.neuralynxrawio import NeuralynxRawIO
from decode_lab_code.readers.helper.neuralynxio import NeuralynxIO
from decode_lab_code.readers.helper import nlxhelper
from decode_lab_code.readers.core.base import base

# calcium imaging stuff
from decode_lab_code.calcium_imaging.preprocessing_utils import miniscope_to_tif, mp4_to_tif, modify_movie, split_4D_tif
from decode_lab_code.utils.indexing_tools import dsearchn

# pyedflib
from pyedflib import highlevel

import matplotlib.pyplot as plt

from tifffile import imsave, memmap, imread, imwrite, TiffFile

# neuralynx data
class neuralynx(base):

    def read_all(self):

        """
        TODO: read all data at once
        Argument that allows the user to read all information from a file using the methods
        ascribed below
        """

        # just run everything below
        self.read_ephys()
        self.read_events()
        self.read_header()
        self.read_vt()

    def read_ephys(self):

        """
        A method to read electrophysiology data acquired by Neuralynx Cheetah in DECODE lab

        Args:
            TODO: opts: optional argument for which data to load in
        
        Returns:
            csc_data: data acquired and stored as .ncs

        """

        # TODO: Build some code that checks for timestamps in spikes outside of timestamps in LFP
        
        
        # Use Neo package
        print("Cite Neo https://github.com/NeuralEnsemble/python-neo/blob/master/CITATION.txt")

        # read events data
        # TODO: Make events into a dictionary
        self.read_events()

        # group data according to extension, then by naming
        split_contents = [i.split('.') for i in self.dir_contents]

        # extract extension values
        ext = [split_contents[i][1] for i in range(len(split_contents)) if len(split_contents[i])>1]

        # extract pre-extension names, if . was used to split
        pre_ext = [split_contents[i][0] for i in range(len(split_contents)) if len(split_contents[i])>1]

        # group extensions
        unique_ext = np.unique(ext) # can test for unique extension names

        # here is a way to do a letter-to-digit search and return letter combo
        #naming = "".join([i for i in pre_ext[10] if i.isdigit()==False])

        # group data based on extension type
        csc_names = []; tt_names = []
        for ci in self.dir_contents:
            if '.ncs' in ci.lower():
                csc_names.append(ci)
            elif '.ntt' in ci.lower():
                tt_names.append(ci)

        # sort files
        def atoi(text):
            return int(text) if text.isdigit() else text
        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)',text) ]

        # sort
        csc_names.sort(key=natural_keys)
        tt_names.sort(key=natural_keys)

        # now lets put these into a dict for working with in NeuralynxIO
        neural_dict = {'CSC': csc_names, 
                        'TT': tt_names}
        
        # Here we create separate dictionaries containing datasets with their corresponding labels
        dict_keys = neural_dict.keys()
        self.csc_data = dict(); self.tt_data = dict(); self.csc_data_fs = dict()
        csc_added = False; tt_added = False
        for groupi in dict_keys: # grouping variable to get TT data
            print("Working with",groupi)
            for datai in neural_dict[groupi]: # now we can get data

                # read data using Neo's NeuralynxIO
                if 'blks' in locals():
                    del blks
                blks = NeuralynxIO(filename=self.folder_path+self.slash+datai, keep_original_times=True).read(lazy=False) # blocks
                #blks = NeuralynxRawIO(filename =folder_path+'/'+datai).parse_header()

                if len(blks) > 1:
                    TypeError("Blocks exceeding size 1. This indicates that multiple sessions detected. The following code will be terminated.")

                # get blocked data
                blk = blks[0]

                # TODO: Handle multisegments (CSC1 from /Users/js0403/local data/2020-06-26_16-56-10 9&10eb male ACTH ELS)
                # You can probably just combine the csc_times and csc_data into one vector

                # TODO: Get sampling rate

                # organize data accordingly 
                # it is VERY important that you only include LFP times between starting/stopping recording
                if 'CSC' in groupi: # CSC files referenced to a different channel
                    
                    # doesn't matter how many blocks there are, concatenate, then separate by events
                    
                    # do a search for starting/stopping recordings
                    counter=0; start_times = []; end_times = []
                    for i in self.event_strings:
                        if 'start' in i.lower():
                            start_times.append(self.event_times[counter])
                        elif 'stop' in i.lower():
                            end_times.append(self.event_times[counter])
                        #print(counter)
                        counter+=1

                    # restrict CSC data to these times
                    temp_csc = []; temp_times = []; csc_fs = []
                    for segi in range(len(blk.segments)):
                        temp_csc.append(blk.segments[segi].analogsignals[0].magnitude.flatten())
                        temp_times.append(blk.segments[segi].analogsignals[0].times.flatten())

                    if len(temp_times) > 1:
                        Warning("Multiple segments detected. Check code.")

                    # now restrict CSC data and times to be within event_times
                    for i in range(len(start_times)):
                        # convert to numpy
                        temp_times[i]=np.array(temp_times[i])
                        temp_csc[i]=np.array(temp_csc[i])
                        # get index of start/stop using dsearchn
                        idx_start = int(dsearchn(temp_times[i],start_times[i])[0])
                        idx_end = int(dsearchn(temp_times[i],end_times[i])[0])
                        # get data in between - have to add 1 at the end because python sees this as [0:-1] (all but the last datapoint)
                        temp_csc[i] = temp_csc[i][idx_start:idx_end+1]
                        temp_times[i] = temp_times[i][idx_start:idx_end+1]

                    # horizontally stack data
                    self.csc_data[datai] = np.hstack(temp_csc)
                    self.csc_times = np.hstack(temp_times) # only need to save one. TODO: make more efficient                        

                    # add sampling rate if available

                    #TODO: add fs for each csc channel and segment!!!
                    temp_fs = str(blk.segments[0].analogsignals[0].sampling_rate)
                    self.csc_data_fs[datai] = csc_fs
                    csc_added = True

                # Notice that all values are duplicated. This is because tetrodes record the same spike times.
                # It is the amplitude that varies, which is not captured here, despite calling magnitude.
                elif 'TT' in groupi: # .ntt TT files with spike data

                    if len(blk.segments) > 1:
                        InterruptedError("Detected multiple stop/starts in spike times. No code available to collapse recordings. Please add code")

                    spikedata = blk.segments[0].spiketrains
                    num_tts = len(spikedata)
                    if num_tts > 4:
                        print("Detected clustered data in",datai)
                        num_trains = len(spikedata) # there will be 4x the number of clusters extracted
                        num_clust = int(num_trains/4) # /4 because there are 4 tetrodes
                        temp_dict = dict()
                        for i in range(num_clust):
                            if i > 0: # skip first cluster, it's noise
                                temp_dict['cluster'+str(i)+'spiketimes'] = spikedata[i].magnitude
                        self.tt_data[datai] = temp_dict
                    else:
                        temp_dict = dict()
                        for i in range(num_tts):
                            temp_dict['channel'+str(i)+'spiketimes'] = spikedata[i].magnitude
                        self.tt_data[datai] = temp_dict
                    tt_added = True
                    self.tt_data_fs = int(32000) # hardcoded

        # history tracking
        if 'blk_logger' in locals():
            self.history.append("LOGGER: multiple start/stop recordings detected. CSC data is ")

        # get keys of dictionary
        if csc_added is True:
            self.csc_data_names = csc_names
            self.history.append("csc_data: CSC data as grouped by ext .ncs")
            self.history.append("csc_data_names: names of data in csc_data as organized by .ncs files")
            self.history.append("csc_data_fs: sampling rate for CSC data, defined by .ncs extension")
            self.history.append("csc_times: timestamps for csc data - accounts for multiple start/stop times")

            # add a grouping table for people to dynamically edit
            self.csc_grouping_table = pd.DataFrame(self.csc_data_names)
            self.csc_grouping_table.columns=['Name']
            self.csc_grouping_table['TetrodeGroup'] = [[]] * self.csc_grouping_table.shape[0]
            self.csc_grouping_table['BrainRegion'] = [[]] * self.csc_grouping_table.shape[0]
            self.csc_grouping_table['Inclusion'] = [True] * self.csc_grouping_table.shape[0]

            self.history.append("csc_grouping_table: pandas DataFrame to organize csc. This is good if you want to cluster data as the NWB file will detect your organization. try adding structure columns and tetrode grouping columns!")
            self.history.append("csc_grouping_table.TetrodeGroup: group for tetrode assignment (CSC1-4 might belong to Tetrode 1)")
            self.history.append("csc_grouping_table.BrainRegion: Enter Structure")
            self.history.append("csc_grouping_table.Inclusion: default is True, set to False if you want to exclude grouping in NWB")


        if tt_added is True:
            self.tt_data_names = tt_names
            self.history.append("tt_data: Tetrode data as grouped by ext .ntt")
            self.history.append("tt_data_names: names of data in tt_data as organized by .ntt files")
            self.history.append("tt_data_fs: hard coded to 32kHz after not detected neo extraction of sampling rate")
            
            # add a grouping table for people to dynamically edit
            self.tt_grouping_table = pd.DataFrame(self.tt_data_names)
            self.tt_grouping_table.columns=['Name']
            self.tt_grouping_table['TetrodeGroup'] = [[]] * self.tt_grouping_table.shape[0]
            self.tt_grouping_table['BrainRegion'] = [[]] * self.tt_grouping_table.shape[0]
            self.tt_grouping_table['Inclusion'] = [True] * self.tt_grouping_table.shape[0]

            self.history.append("tt_grouping_table: pandas DataFrame to organize csc. This is good if you want to cluster data as the NWB file will detect your organization. try adding structure columns and tetrode grouping columns!")
            self.history.append("tt_grouping_table.TetrodeGroup: group for tetrode assignment (CSC1-4 might belong to Tetrode 1)")
            self.history.append("tt_grouping_table.BrainRegion: Enter Structure")
            self.history.append("tt_grouping_table.Inclusion: default is True, set to False if you want to exclude grouping in NWB")

    def read_ncs_file(self, wire_names: str):

        '''
        This function reads whichever ncs files the user requires

        Args:
            >>> wire_names: list array containing wire names. Takes partial inputs (e.g. 'CSC1' rather than 'CSC1.ncs')
        
        '''

        # Use Neo package
        print("Cite Neo https://github.com/NeuralEnsemble/python-neo/blob/master/CITATION.txt")

        # read events data
        # TODO: Make events into a dictionary
        self.read_events()

        # group data according to extension, then by naming
        split_contents = [i.split('.') for i in self.dir_contents]

        # extract extension values
        ext = [split_contents[i][1] for i in range(len(split_contents)) if len(split_contents[i])>1]

        # extract pre-extension names, if . was used to split
        pre_ext = [split_contents[i][0] for i in range(len(split_contents)) if len(split_contents[i])>1]

        # group extensions
        unique_ext = np.unique(ext) # can test for unique extension names

        # check that wire_names is a list
        if type(wire_names) is str:
            wire_names = [wire_names] # wrap string into list
            print("Single named input wrapped to list")

        # do a global path search for the files input
        csc_names = []; tt_names = []
        for ci in self.dir_contents:
            for wi in wire_names:
                if wi.lower() in ci.lower():
                    csc_names.append(ci)

        if len(csc_names) == 0:
            raise TypeError("No CSC files found at:",self.folder_path)
        
        # sort files
        def atoi(text):
            return int(text) if text.isdigit() else text
        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)',text) ]
        
        # sort
        csc_names.sort(key=natural_keys)
        tt_names.sort(key=natural_keys)

        # now lets put these into a dict for working with in NeuralynxIO
        neural_dict = {'CSC': csc_names}
        
        # Here we create separate dictionaries containing datasets with their corresponding labels
        dict_keys = neural_dict.keys()
        self.csc_data = dict(); self.tt_data = dict(); self.csc_data_fs = dict()
        csc_added = False; tt_added = False
        for groupi in dict_keys: # grouping variable to get TT data
            print("Working with",groupi)
            for datai in neural_dict[groupi]: # now we can get data

                # read data using Neo's NeuralynxIO
                if 'blks' in locals():
                    del blks
                blks = NeuralynxIO(filename=self.folder_path+self.slash+datai, keep_original_times=True).read(lazy=False) # blocks
                #blks = NeuralynxRawIO(filename =folder_path+'/'+datai).parse_header()

                if len(blks) > 1:
                    TypeError("Blocks exceeding size 1. This indicates that multiple sessions detected. The following code will be terminated.")

                # get blocked data
                blk = blks[0]

                # TODO: Handle multisegments (CSC1 from /Users/js0403/local data/2020-06-26_16-56-10 9&10eb male ACTH ELS)
                # You can probably just combine the csc_times and csc_data into one vector

                # TODO: Get sampling rate

                # organize data accordingly 
                # it is VERY important that you only include LFP times between starting/stopping recording
                if 'CSC' in groupi: # CSC files referenced to a different channel
                    
                    # doesn't matter how many blocks there are, concatenate, then separate by events
                    
                    # do a search for starting/stopping recordings
                    counter=0; start_times = []; end_times = []
                    for i in self.event_strings:
                        if 'start' in i.lower():
                            start_times.append(self.event_times[counter])
                        elif 'stop' in i.lower():
                            end_times.append(self.event_times[counter])
                        #print(counter)
                        counter+=1

                    # restrict CSC data to these times
                    temp_csc = []; temp_times = []; csc_fs = []
                    for segi in range(len(blk.segments)):
                        temp_csc.append(blk.segments[segi].analogsignals[0].magnitude.flatten())
                        temp_times.append(blk.segments[segi].analogsignals[0].times.flatten())

                    if len(temp_times) > 1:
                        Warning("Multiple segments detected. Check code.")

                    # now restrict CSC data and times to be within event_times
                    for i in range(len(start_times)):
                        # convert to numpy
                        temp_times[i]=np.array(temp_times[i])
                        temp_csc[i]=np.array(temp_csc[i])
                        # get index of start/stop using dsearchn
                        idx_start = int(dsearchn(temp_times[i],start_times[i])[0])
                        idx_end = int(dsearchn(temp_times[i],end_times[i])[0])
                        # get data in between - have to add 1 at the end because python sees this as [0:-1] (all but the last datapoint)
                        temp_csc[i] = temp_csc[i][idx_start:idx_end+1]
                        temp_times[i] = temp_times[i][idx_start:idx_end+1]

                    # horizontally stack data
                    self.csc_data[datai] = np.hstack(temp_csc)
                    self.csc_times = np.hstack(temp_times) # only need to save one. TODO: make more efficient                        

                    # add sampling rate if available

                    #TODO: add fs for each csc channel and segment!!!
                    temp_fs = str(blk.segments[0].analogsignals[0].sampling_rate)
                    self.csc_data_fs[datai] = csc_fs
                    csc_added = True        

    def read_vt(self):

        # Get VT data from .NVT files
        vt_name = [i for i in self.dir_contents if '.nvt' in i.lower()][0]

        # get video tracking data if it's present
        filename = os.path.join(self.folder_path,vt_name)
        # Example usage:

        # data = read_nvt("path_to_your_file.nvt")

        vt_data = nlxhelper.read_nvt(filename = filename)
        self.vt_x = vt_data['Xloc']
        self.vt_y = vt_data['Yloc']
        self.vt_t = vt_data['TimeStamp']

        # add history
        self.history.append("vt_x: x-position data obtained from .nvt files")
        self.history.append("vt_y: y-position data obtained from .nvt files")
        self.history.append("vt_t: timestamp data obtained from .nvt files")

    def read_events(self):

        """
        TODO: Read events information and this information will be packaged into nwb.epochs
        
        """
        # Get VT data from .NVT files
        ev_name = [i for i in self.dir_contents if '.nev' in i.lower()][0]

        # get video tracking data if it's present
        filename = os.path.join(self.folder_path,ev_name)

        # read data
        blks = NeuralynxIO(filename=filename, keep_original_times=True).read(lazy=False) # blocks       
        if len(blks) > 1:
            TypeError("Blocks exceeding size 1. This indicates that multiple sessions detected. The following code will be terminated.")

        # get blocked data
        blk = blks[0]     

        # loop over blocked data if multiple blocks exist
        event_strings = []; event_times = []
        if len(blk.segments) > 1:
            TypeError("CODE DOES NOT HANDLE MULTIPLE BLOCKS - FIX")
        else:
            event_dict = blk.segments[0].events[0].__dict__
            event_strings = event_dict['_labels']
            event_times = blk.segments[0].events[0].times.magnitude

        self.event_strings = event_strings
        self.event_times = event_times
        self.history.append("event_strings: Event variables during recordings (in written format)")
        self.history.append("event_times: Event variables during recordings (in seconds)")

    def read_header(self):

        # attempt to read files until you get a header
        ncs_file = [i for i in self.dir_contents if '.ncs' in i.lower()]
        for i in ncs_file:
            try:
                reader = NeuralynxRawIO(filename = os.path.join(self.folder_path,i))
                reader.parse_header()
                file_header = reader.file_headers
            except:
                pass
            if 'file_header' in locals():
                break

        # time information for NWB
        header_dict = dict(list(file_header.values())[0])
        #datetime_str = header_list['recording_opened']
        self.header = header_dict
        self.history.append("header: example header from the filepath of a .ncs file")

# pinnacle data
class pinnacle(base):

    def read_edf(self):
        '''
        Reads .edf files
            Args:
                >>> dir: directory with the .edf extension

            Returns:
                signals: signal data
                signal_headers: header files
        '''
        # this will be a loop
        #dir_edf = os.path.join(dir)
        signals, signal_headers, header = highlevel.read_edf(self.folder_path)    

    def ncs_to_edf(self):
        pass

folder_path = r'/Volumes/decode/Emily/TSC project/Single Unit Recordings/87mTSC1 L4/2024-01-04_09-51-20'
self = neuralynx(folder_path=folder_path)
self.read_ncs_file(wire_names = 'CSC1')
self.read_events()
self.read_header()

# miniscope data
class miniscope(base):
    
    def convert_to_tif(self):

        # root back to preprocessing_utils
        miniscope_to_tif(movie_path = self.folder_path)

# lionheart calcium imaging
class lionheart(base):

    def process(self, split_4D: bool = False):

        # convert to tif file
        fname = mp4_to_tif(movie_path = self.folder_path)

        # split 4D_array
        if split_4D is True:
            split_4D_tif(movie_path = fname)
    
# procedures for olympus calcium imaging recordings
class olympus(base):

    def process(self, split_4D: bool = False):

        # convert to tif file
        fname = mp4_to_tif(movie_path = self.folder_path)

        # split 4D_array
        if split_4D is True:
            split_4D_tif(movie_path = fname)

#%%  some general helper functions for nwb stuff

# TODO: This could be its own function
def read_movie(movie_name: str):

    """
    Args:
        >>> movie_name: name of the movie to load with extension

    John Stout
    """

    # read movie file
    movie_path = os.path.join(self.folder_path, movie_name)
    print("Reading movie from: ", movie_path)
    cap = cv2.VideoCapture(movie_path) 
    movie_data = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        else:
            movie_data.append(frame[:,:,0]) # only the first array matters
    movie_mat = np.dstack(movie_data)
    movie_data = np.moveaxis(movie_mat, -1, 0)
