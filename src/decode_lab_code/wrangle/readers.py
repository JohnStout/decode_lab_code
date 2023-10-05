# readers

# get some packages
from datetime import datetime
from dateutil import tz
from pathlib import Path
from uuid import uuid4

import re
import os

# pynwb
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.file import Subject

# numpy
import numpy as np

# neo: TODO: modify neo function rawio/neuralyxrawio/nlxheader
#from neo.io.neuralynxio import NeuralynxIO
#from neo.io.neuralynxio import NeuralynxRawIO
from decode_lab_code.utils.neuralynxrawio import NeuralynxRawIO
from decode_lab_code.utils.neuralynxio import NeuralynxIO

# from utils
from decode_lab_code.utils import nlxhelper

# our labs code (folder "core", file "base", class "base")
from decode_lab_code.core.base import base # this is a core base function to organize data

print("Cite NWB")
print("Cite CatalystNeuro: NeuroConv toolbox if converting Neuralynx data")

# a specific class for unpacking neuralynx data
class read_nlx(base):

    def read_all(self):
        """
        TODO: read all data at once
        Argument that allows the user to read all information from a file using the methods
        ascribed below
        """
        pass

    def read_ephys(self, opts = None):

        """
        A method to read electrophysiology data acquired by Neuralynx Cheetah in DECODE lab

        Args:
            TODO: opts: optional argument for which data to load in
        
        Returns:
            csc_data: data acquired and stored as .ncs

        """
        
        # Use Neo package
        print("Cite Neo https://github.com/NeuralEnsemble/python-neo/blob/master/CITATION.txt")

        # TODO: group data by file type, then separate by common naming conventions so that we never
        # have to worry about changing naming conventions

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
        self.csc_data = dict(); self.tt_data = dict()
        csc_added = False; tt_added = False
        for groupi in dict_keys: # grouping variable to get TT data
            print("Working with",groupi)
            for datai in neural_dict[groupi]: # now we can get data

                # read data using Neo's NeuralynxIO
                if 'blks' in locals():
                    del blks
                blks = NeuralynxIO(filename=self.folder_path+'/'+datai, keep_original_times=True).read(lazy=False) # blocks
                #blks = NeuralynxRawIO(filename =folder_path+'/'+datai).parse_header()

                if len(blks) > 1:
                    TypeError("Blocks exceeding size 1. This indicates that multiple sessions detected. The following code will be terminated.")

                # get blocked data
                blk = blks[0]

                # TODO: Handle multisegments (CSC1 from /Users/js0403/local data/2020-06-26_16-56-10 9&10eb male ACTH ELS)
                # You can probably just combine the csc_times and csc_data into one vector

                # TODO: Get sampling rate

                # organize data accordingly
                if 'CSC' in groupi: # CSC files referenced to a different channel
                    
                    if len(blk.segments) > 1:
                        blk_logger = ("Multiple blocks detected in "+datai+". LFP and LFP times have been collapsed into a single array.")
                        print(blk_logger)
                        temp_csc = []; temp_times = []
                        for segi in range(len(blk.segments)):
                            temp_csc.append(blk.segments[segi].analogsignals[0].magnitude)
                            temp_times.append(blk.segments[segi].analogsignals[0].times)
                        self.csc_data[datai] = np.vstack(temp_csc)
                        self.csc_times = np.hstack(temp_times) # only need to save one. TODO: make more efficient
                    else:                   
                        self.csc_data[datai] = blk.segments[0].analogsignals[0].magnitude
                        self.csc_times = blk.segments[0].analogsignals[0].times
                        
                    # add sampling rate if available
                    if 'csc_fs' not in locals():
                        temp_fs = str(blk.segments[0].analogsignals[0].sampling_rate)
                        csc_fs = int(temp_fs.split('.')[0])
                        self.csc_data_fs = csc_fs
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
            self.history.append("LOGGER: csc_data had multiple blocks. This is likely due to multiple start/stops when recording. LFP and times were concatenated into a single array.")

        # get keys of dictionary
        if csc_added is True:
            self.csc_data_names = csc_names
            self.history.append("csc_data: CSC data as grouped by ext .ncs")
            self.history.append("csc_data_names: names of data in csc_data as organized by .ncs files")
            self.history.append("csc_data_fs: sampling rate for CSC data, defined by .ncs extension")

        if tt_added is True:
            self.tt_data_names = tt_names
            self.history.append("tt_data: Tetrode data as grouped by ext .ntt")
            self.history.append("tt_data_names: names of data in tt_data as organized by .ntt files")
            self.history.append("tt_data_fs: hard coded to 32kHz after not detected neo extraction of sampling rate")

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
        TODO: Read events information
        
        """

        pass
