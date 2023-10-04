# converters.py
#
# This collection of objects will support the conversion of data into something readible for NWB input/output
#
# 
# This module is meant to act as an interface to read, write, and convert data into NWB ready formats
#
# Non-native dependencies:
#   Neo
#   pynwb
#   packages for tifffile video
#
#   neuroconv
#       pynwb (wrapped in neuroconv)
#
# To work if using decode_lab_code: 
#
#   1) Open terminal
#   2) conda create -n neuroconv python=3.11.0
#   3) pip install neuroconv
#   4) cd decode_lab_code
#   5) pip install -e .
#
#
# UPDATE: Use conversions
#   1) conda create -n conversion python=3.11.0
#   2) conda activate conversions
#   3) pip install neo
#   4) pip install pynwb
#   5) cd decode_lab_code
#   6) pip install -e .
# 
# Written by John Stout

# get some packages
from datetime import datetime
from dateutil import tz
from pathlib import Path
from uuid import uuid4

import re

import os

from typing import Dict, Union, List, Tuple

# pynwb
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.file import Subject

# numpy
import numpy as np

# neo: TODO: modify neo function rawio/neuralyxrawio/nlxheader
from neo.io.neuralynxio import NeuralynxIO
from neo.io.neuralynxio import NeuralynxRawIO

# our labs code (folder "core", file "base", class "base")
from decode_lab_code.core.base import base # this is a core base function to organize data

print("Cite NWB")
print("Cite CatalystNeuro: NeuroConv toolbox if converting Neuralynx data")

# inherit the "core" __init__ from "base" object
class nwb_utils(base):

    def read_nwb(self, data_name: str):
        """
            Read NWB files

            Args:
                data_name: name of nwb file
        """
        
        io = NWBHDF5IO(self.folder_path+'/'+data_name, mode="r")
        nwb_file = io.read()

        return nwb_file

    def write_nwb(self,data_name = None, nwb_file=None):
        """
            Write NWB files

            Args:
                folder_name: location of data
                data_name (OPTIONAL): name of nwb file
                nwb_file: nwb file type
        """
        if data_name is None:
            data_name = "nwb_file"

        with NWBHDF5IO(self.folder_path+'/'+data_name, "w") as io:
            io.write(nwb_file)

        print("Wrote nwb_file to: ",self.folder_name+'/'+data_name)

    def generate_nwb_sheet(self):
        """
        This function is meant to generate and save out a .txt or .doc file for the user to fill out

        It can then be loading into the nwb creation functions

        """

    def nwb2nlx(self):
        """
        TODO: This function will read in the nwb file, then extract LFPs and spiking into dictionary arrays, like
        the output from read_ephys
        """
        pass

    def nlx2nwb(self, save_name = 'data_nwb', csc_data_dict: dict = None, tt_data_dict: dict = None, nwb_sheet_directory = None, ):
        """

        TODO: Add csc_data_dict and tt_data_dict as inputs so that you can control nwb

        Write an NWB file using ephys data collected with neuralynx acquisition system

        ---------------------------------------
        Args:
            folder_path: string input that defines the directory to extract data
            TODO: Add arguments for csc_data, tt_data, etc.. 
            You want to tell the code where to store these items in the NWB

        Optional Args:
            save_name: defaults to 'data_nwb', but you could redefine this
            nwb_sheet_directory: location to load NWB sheet. THis is used to automatically populated the NWB file information
            
        Returns:
            Saves a .nwb file with the name following "save_name" argument. Default is 'data_nwb.nwb'

        ---------------------------------------
        STORING DATA:
            You should store your raw nlx files in one folder, and your other files in a "processed" folder.
            Inside that folder, you can specificy what was done to the files and store them hierarchically (e.g. filtered, clustered)

        -----------------------------------------
        # IMPORTANT:
            If you are running into weird errors that tell you to contact the developers, it just means the header
            isn't being read correctly. This is because the xplorefinder.m creates a strange headerfile that is missing elements
            Try pasting the following code in replacing line 246:end of __init__ in the following directory:
                    neuroconv/lib/python3.11/site-packages/neo/rawio/neuralyxrawio/nlxheader.py


            # opening time - for the xplorefinder.m code, this is recognized as None, because it provides no info about date/time
            sr = re.search(hpd['datetime1_regex'], txt_header)
            if not sr:
                if av == Version('5.6.3'):
                    print("Filling in missing datetime for Cheetah version: 5.6.3")
                    #current_date = datetime.datetime.now()
                    self['recording_opened'] = datetime.datetime.now().replace(microsecond=0)
                else:
                    raise IOError("No matching header open date/time for application {} " +
                                "version {}. Please contact developers.".format(an, av))
            else:
                dt1 = sr.groupdict()
                self['recording_opened'] = datetime.datetime.strptime(
                    dt1['date'] + ' ' + dt1['time'], hpd['datetimeformat'])
            print(self['recording_opened'])

            # close time, if available
            if 'datetime2_regex' in hpd:
                sr = re.search(hpd['datetime2_regex'], txt_header)
                if not sr:
                    if av == Version('5.6.3'):
                        print("Filling in missing datetime for Cheetah version: 5.6.3")
                        self['recording_closed'] = datetime.datetime.now().replace(microsecond=0)
                    else:
                        raise IOError("No matching header close date/time for application {} " +
                                    "version {}. Please contact developers.".format(an, av))
                else:
                    dt2 = sr.groupdict()
                    self['recording_closed'] = datetime.datetime.strptime(
                        dt2['date'] + ' ' + dt2['time'], hpd['datetimeformat'])
        """

        #%%

        # Extract header information to document the NWB file

        # attempt to read files until you get a header
        next = 0; looper = 0
        while next == 0:
            ncs_file = [i for i in self.dir_contents if '.ncs' in i.lower()][looper]
            reader = NeuralynxRawIO(filename = os.path.join(self.folder_path,self.dir_contents[looper]))
            reader.parse_header()
            file_header = reader.file_headers
            if bool(file_header) is False:
                looper = looper+1
            else:
                next = 1
            if looper == len(self.dir_contents)-1:
                raise ValueError('Could not extract information from header')

        # time information for NWB
        header_list = list(file_header.values())[0]
        datetime_str = header_list['recording_opened']
        #date_obj = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')

        #%% 

        # interface with the user
        #folder_path = input("Enter directory of recording session: ")
        #recording_notes = input("Enter information about your recording wires (e.g. TT1 wasn't working great today...)")

        # create NWB file
        nwbfile = NWBFile(
            session_description=input("Enter a brief discription of the experiment: "),
            identifier=str(uuid4()),
            session_start_time = datetime_str,
            experimenter = input("Enter the name(s) of the experimenter(s): "),
            lab="Hernan Lab",
            institution="Nemours Children's Hospital",
            session_id=self.session_id
        )

        # enter subject specific information
        subject = Subject(
                subject_id=input("Enter subject ID: "),
                age=input("Enter age of subject (PD): "),
                description=input("Enter notes on this mouse as needed: "),
                species=input("Enter species type (e.g. mus musculus (C57BL, etc...), Rattus rattus, homo sapiens): "),
                sex=input("Enter sex of subject: "),
            )

        nwbfile.subject = subject

        # add recording device information
        device = nwbfile.create_device(
            name="Cheetah", 
            description="Tetrode array", 
            manufacturer="Neuralynx recording system and self fabricated arrays"
            )
        
        #%% Add electrode column and prepare for adding actual data

        # The first step to adding ephys data is to create categories for organization
        nwbfile.add_electrode_column(name='label', description="label of electrode")

        # add csc channels. Sometimes the lab uses CSC1-4 as TT1, sometimes we call it TTa-d.
        group_csc = 'into tetrode' # TODO: Make this as an input
        brain_area = 'PFC' # TODO: add brain_area as an input, but also make code flexible for multipl structures
        
        if 'tetrode' in group_csc:
            group_csc_to_tt = int(len(csc_data)/4) # grouped tts
            idx = [1,2,3,4] # index for tetrode assignment
            electrode_counter = 0 # used later
            for ei in range(group_csc_to_tt): # for loop over electrodes (ei)
                if ei > 0:
                    idx = [idxi+4 for idxi in idx] # index that changes according to tetrode assignment
                print(idx) # parsing error

                # create an electrode group for a given tetrode
                electrode_group = nwbfile.create_electrode_group(
                    name='Tetrode{}'.format(ei+1),
                    description='Raw tetrode data',
                    device=device,
                    location=brain_area)     

                # add unit column per tetrode
                    

                # add each wire to the electrode group
                csc_names_use = []
                csc_names_use = [csc_names[idx[i]-1] for i in range(len(idx))]
                print(csc_names_use) # changes with loop

                for csci in csc_names_use:
                    nwbfile.add_electrode(
                        group = electrode_group,
                        label = csci.split('.')[0],
                        location=brain_area
                    )
                    electrode_counter += 1
        nwbfile.electrodes.to_dataframe()

        #%% NOW we work on adding our data. For LFPs, we store in ElectricalSeries object

        # create dynamic table
        all_table_region = nwbfile.create_electrode_table_region(
            region=list(range(electrode_counter)),  # reference row indices 0 to N-1
            description="all electrodes",
        )

        # now lets get our raw data into a new format
        csc_all = np.zeros(shape=(len(csc_data[csc_names[0]]),electrode_counter))
        for csci in range(len(csc_data)):
            csc_all[:,csci] = np.reshape(csc_data[csc_names[csci]],len(csc_data[csc_names[csci]]))

        raw_electrical_series = ElectricalSeries(
            name="ElectricalSeries",
            data=csc_all,
            timestamps = csc_times, # need timestamps
            electrodes=all_table_region,
            #starting_time=0.0,  # timestamp of the first sample in seconds relative to the session start time
            #rate=32000.0,  # in Hz
        )
        nwbfile.add_acquisition(raw_electrical_series)

        #%% Add spiketimes
        nwbfile.add_unit_column(name="quality", description="sorting quality")

        # get unit IDs for grouping
        unit_ids = tt_clust_data.keys()

        unit_num = 0
        for i in unit_ids:
            for clusti in tt_clust_data[i]:
                print(clusti)
                #clust_id = i.split('.ntt')[0]+'_'+clusti
                nwbfile.add_unit(spike_times = tt_clust_data[i][clusti],
                                quality = "good",
                                id = unit_num)
                unit_num += 1
        nwbfile.units.to_dataframe()

        #%% Add behavioraly relevant times

        #%% Save NWB file


#%%

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

        vt_data = read_nvt(filename = filename)
        self.vt_x = vt_data['Xloc']
        self.vt_y = vt_data['Yloc']
        self.vt_t = vt_data['TimeStamp']

        # add history
        self.history.append("vt_x: x-position data obtained from .nvt files")
        self.history.append("vt_y: y-position data obtained from .nvt files")
        self.history.append("vt_t: timestamp data obtained from .nvt files")

        def read_nvt(filename: str) -> Union[Dict[str, np.ndarray], None]:
            """
            Reads a NeuroLynx NVT file and returns its data.

            Parameters
            ----------
            filename : str
                Path to the NVT file.

            Returns
            -------
            Union[Dict[str, np.ndarray], None]
                Dictionary containing the parsed data if file exists, None otherwise.

            Raises
            ------
            FileNotFoundError
                If the specified file does not exist.


            Ben Dichtor wrote code
            """
            
            # Constants for header size and record format
            HEADER_SIZE = 16 * 1024
            RECORD_FORMAT = [
                ("swstx", "uint16"),
                ("swid", "uint16"),
                ("swdata_size", "uint16"),
                ("TimeStamp", "uint64"),
                ("dwPoints", "uint32", 400),
                ("sncrc", "int16"),
                ("Xloc", "int32"),
                ("Yloc", "int32"),
                ("Angle", "int32"),
                ("dntargets", "int32", 50),
            ]
            
            # Check if file exists
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File {filename} not found.")
            
            # Reading and parsing data
            with open(filename, 'rb') as file:
                file.seek(HEADER_SIZE)
                dtype = np.dtype(RECORD_FORMAT)
                records = np.fromfile(file, dtype=dtype)
                return {name: records[name].squeeze() for name, *_ in RECORD_FORMAT}

        def handle_missing_data(filename: str, missing_data = None):

            """
            Reads neuralynx NVT files and handles missing data

            TODO: add interpolation of missing data. Might be good as a method of a class

            Args:
                filename: directory of data with .nvt extension
                missing_data: str, option to handle missing data. Default = None.
                                Accepts: 
                                    'NaN': makes 0 values nan

            Ben Dichtor wrote code. John Stout wrapped into function

            """

            # read data
            data = read_nvt(filename)

            # make 0s nan
            if missing_data == 'NaN':
                x = data["Xloc"].astype(float)
                x[data["Xloc"] <= 0] = np.nan

                y = data["Yloc"].astype(float)
                y[data["Yloc"] <= 0] = np.nan

                t = data["Tloc"].astype(float)

            return x,y,t

    def read_events(self):

        """
        TODO: Read events information
        
        """

        pass


