# ioreaders
#
# input-output readers
#
# specific purpose is to convert between raw data and dictionaries
#
# written by John Stout

# get some packages
from datetime import datetime
from dateutil import tz
from pathlib import Path
from uuid import uuid4

import re
import os

import pandas as pd

# pynwb
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.file import Subject
from pynwb import validate

# numpy
import numpy as np

# loading neo package
from decode_lab_code.utils.neuralynxrawio import NeuralynxRawIO
from decode_lab_code.utils.neuralynxio import NeuralynxIO

# from utils
from decode_lab_code.utils import nlxhelper

# multiple inheritance - ephys gets its __init__ from base and gives it to read_nlx
from decode_lab_code.core.ephys import ephys_tools # this is a core base function to organize data
from decode_lab_code.core.base import base

print("Cite NWB")
print("Cite CatalystNeuro: NeuroConv toolbox if converting Neuralynx data")

# a specific class for unpacking neuralynx data
class read_nlx(ephys_tools):

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
                            temp_csc.append(blk.segments[segi].analogsignals[0].magnitude.flatten())
                            temp_times.append(blk.segments[segi].analogsignals[0].times.flatten())
                        self.csc_data[datai] = np.hstack(temp_csc)
                        self.csc_times = np.hstack(temp_times) # only need to save one. TODO: make more efficient
                    else:                   
                        self.csc_data[datai] = blk.segments[0].analogsignals[0].magnitude.flatten()
                        self.csc_times = blk.segments[0].analogsignals[0].times.flatten()

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

    def read_header(self):

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
        header_dict = dict(list(file_header.values())[0])
        #datetime_str = header_list['recording_opened']
        self.header = header_dict
        self.history.append("header: example header from the filepath of a .ncs file")

    def write_nwb(self):

        """
        All .ncs files will be taken in

        """

        # make sure you have the header
        self.read_header()
        datetime_str = self.header['recording_opened']

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
            description=input("Type of array? (e.g. tetrode/probe)"), 
            manufacturer="Neuralynx"
            )

        #%% before moving forward, remove any rows set to be removed in the pandas array
               
        # remove any rows of the pandas array if the inclusion is set to False
        rem_data_csc = self.csc_grouping_table['Inclusion'][self.csc_grouping_table['Inclusion']==False].index.tolist()
        rem_data_tt = self.tt_grouping_table['Inclusion'][self.tt_grouping_table['Inclusion']==False].index.tolist()

        print("Removing:\n",self.tt_grouping_table.iloc[rem_data_tt].Name,self.csc_grouping_table.iloc[rem_data_csc].Name)

        self.csc_grouping_table=self.csc_grouping_table.drop(index=rem_data_csc)
        self.tt_grouping_table=self.tt_grouping_table.drop(index=rem_data_tt)

        #%% Add electrode column and prepare for adding actual data

        # The first step to adding ephys data is to create categories for organization
        nwbfile.add_electrode_column(name='label', description="label of electrode")

        # loop over pandas array, first organize by array, then index tetrode and electrode
        brain_regions = self.csc_grouping_table['BrainRegion'].unique().tolist()
        electrode_group = self.csc_grouping_table['TetrodeGroup'].unique().tolist() # grouped into tetrde
        csc_table_names = self.csc_grouping_table['Name'].tolist()
        self.csc_grouping_table.set_index('Name', inplace=True) # set Name as the index
        #self.csc_grouping_table.reset_index(inplace=True)
        
        # now create electrode groups according to brain area and electrode grouping factors
        for bi in brain_regions: # loop over brain regions

            for ei in electrode_group: # loop over tetrode or probe

                # create an electrode group for a given tetrode
                electrode_group = nwbfile.create_electrode_group(
                    name='Tetrode{}'.format(ei),
                    description='Raw tetrode data',
                    device=device,
                    location=bi)         

        # loop over the pandas array for csc data, then assign the data accordingly
        electrode_counter = 0
        for csci in csc_table_names: # loop over electrodes within tetrode

            # get index of csc belonging to brain region bi and electrode ei
            pd_series = self.csc_grouping_table.loc[csci]
            #electrode_group = nwbfile.electrode_groups['Tetrode'+pd_series.TetrodeGroup]
            nwbfile.add_electrode(
                group = nwbfile.electrode_groups['Tetrode'+str(pd_series.TetrodeGroup)], 
                label = csci.split('.')[0],
                location=pd_series.BrainRegion
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
        print("This make take a few moments if working with a lot of CSC data...")
        csc_all = np.zeros(shape=(len(self.csc_data[self.csc_data_names[0]]),electrode_counter))
        self.csc_grouping_table.reset_index(inplace=True)
        counter = 0
        for csci in self.csc_grouping_table.Name:
            csc_all[:,counter]=self.csc_data[csci]
            counter += 1

        raw_electrical_series = ElectricalSeries(
            name="ElectricalSeries",
            data=csc_all,
            timestamps = self.csc_times, # need timestamps
            electrodes=all_table_region,
            #starting_time=0.0,  # timestamp of the first sample in seconds relative to the session start time
            #rate=32000.0,  # in Hz
        )
        nwbfile.add_acquisition(raw_electrical_series)

        #%% Add spiketimes
        nwbfile.add_unit_column(name="quality", description="sorting quality")

        # get unit IDs for grouping
        unit_ids = self.tt_grouping_table.Name.tolist()
        self.tt_grouping_table.set_index('Name', inplace=True) # set Name as the index
        #self.tt_grouping_table.reset_index(inplace=True)

        unit_num = 0
        for i in unit_ids:
            #print(self.tt_data[i])
            tetrode_num = self.tt_grouping_table.loc[i].TetrodeGroup
            brain_reg = self.tt_grouping_table.loc[i].BrainRegion
            for clusti in self.tt_data[i]:
                #print(i+' '+clusti)
                #clust_id = i.split('.ntt')[0]+'_'+clusti
                nwbfile.add_unit(spike_times = self.tt_data[i][clusti],
                                 electrode_group = nwbfile.electrode_groups['Tetrode'+str(tetrode_num)], 
                                 quality = "good",
                                 id = unit_num)
                unit_num += 1
        nwbfile.units.to_dataframe()

        #%% Save NWB file
        save_nwb(folder_path=self.folder_path,nwb_file=nwbfile)
        val_out = validate(paths=[os.path.join(self.folder_path,'nwbfile.nwb')], verbose=True)
        
        print("NWB validation may be incorrect. Still need an invalid NWB file to check against....10/10/2023")
        if val_out[1]==0:
            print("No errors detected in NWB file")
        else:
            print("Error detected in NWB file")

# some general helper functions for nwb stuff
def load_nwb(folder_path: str, data_name: str = 'nwbfile.nwb'):
    """
        Read NWB files

        Args:
            folder_path: directory of data
            data_name: (OPTIONAL). Recommend to standardize this.
    """
    io = NWBHDF5IO(folder_path+'/'+data_name, mode="r")
    nwb_file = io.read()

    return nwb_file

def save_nwb(folder_path: str, data_name: str = 'nwbfile.nwb', nwb_file=None):
    """
        Write NWB files

        Args:
            folder_name: location of data
            data_name (OPTIONAL): name of nwb file
            nwb_file: nwb file type
    """

    with NWBHDF5IO(folder_path+'/'+data_name, "w") as io:
        io.write(nwb_file)

    print("Save .nwb file to: ",folder_path+'/'+data_name)
