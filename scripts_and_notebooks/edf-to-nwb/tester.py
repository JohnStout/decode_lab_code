# this code represents the first pass at a simple add-on to hernan-to-nwb
#
# this is highly experiment specific and so should be treated as such
#
# This is for Emilys project on TSC with EEG and behavioral recordings

# numpy, pyEDFlib, ipykernel, matplotlib, openpyxl
import os
from pyedflib import highlevel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from pandasgui import show

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from pynwb.ecephys import LFP, ElectricalSeries
from uuid import uuid4

from decode_lab_code.utils import nwb_utils

# INPUTS
dir = '/Users/js0403/edf recording'
experimenters = ['Emily']
session_id = ['First few days']

# automation begins...
dir_contents = sorted(os.listdir(dir))
edf_names = [i for i in dir_contents if '.edf' in i]

# this will be a loop
dir_edf = os.path.join(dir,edf_names[0])
signals, signal_headers, header = highlevel.read_edf(dir_edf)

# 4 animals, each have 3 channels
animal_ids = [i['transducer'] for i in signal_headers]
electr_ids = [i['label'].split(' ')[-1] for i in signal_headers]
fs = [i['sample_rate']for i in signal_headers]
unit = [i['dimension']for i in signal_headers]

# make pandas array
data_table = pd.DataFrame(animal_ids,columns=['Subject'])
data_table['Sex'] = [[]] * data_table.shape[0]
data_table['Genotype'] = [[]] * data_table.shape[0]
data_table['Genotype'] = [[]] * data_table.shape[0]
data_table['Age Range'] = [[]] * data_table.shape[0]
data_table['Electrodes'] = electr_ids
data_table['Skull Location'] = [[]] * data_table.shape[0]
data_table['Sampling Rate'] = fs
data_table['Unit'] = unit

# excel file gets saved out, edited, then reloaded
data_table = nwb_utils.pandas_excel_interactive(dir = dir, df = data_table)

# TODO: Add preprocessing module to the NWB file for times when the user does things like filters/rereferences
datetime_str = header['startdate']

# loop over each animal and create an NWB file
animal_ids_uniq = list(np.unique(np.array(animal_ids)))
for i in animal_ids_uniq:

    # test if an nwbfile exists and remove it
    if 'nwbfile' in locals():
        del nwbfile

    # restrict the pandas table to your subject
    temp_table = data_table.loc[data_table['Subject']==i]

    # create NWB file
    nwbfile = NWBFile(
        session_description="Continuous EEG recordings in TSC mice",
        identifier=str(uuid4()),
        session_start_time = datetime_str,
        experimenter = experimenters,
        lab="Hernan Lab",
        institution="Nemours Children's Hospital"
    )

    # enter subject specific information
    subject_id = 'mouse-'+i.split(':')[-1]
    subject = Subject(
            subject_id=subject_id,
            age=str(list(temp_table['Age Range'])[0]),
            description=str([]),
            species=str(list(temp_table['Genotype'])[0]),
            sex=str(list(temp_table['Sex'])[0]),
        )
    nwbfile.subject = subject

    # add recording device information
    device = nwbfile.create_device(
        name="Pinnacle", 
        description="EEG", 
        manufacturer="Pinnacle"
        )
    
    # loop over groups and create electrode objects
    nwbfile.add_electrode_column(name='label', description="label of electrode")
    brain_grouping = temp_table['Skull Location']
    for braini in temp_table.index: # loop over brain regions

        # create an electrode group for a given tetrode
        electrode_group = nwbfile.create_electrode_group(
            name=temp_table['Electrodes'][braini],
            description='EEG data',
            device=device,
            location=temp_table['Skull Location'][braini]) 

        nwbfile.add_electrode(
            group = electrode_group,
            label = temp_table['Electrodes'][braini],
            location = temp_table['Skull Location'][braini]) 

    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(temp_table.shape[0])),
        description="all electrodes")

    # get signal data and make into a numpy array (samples,num wires)
    sig_temp = np.array([])
    sig_temp = np.array(signals[list(temp_table.index)]).T
    fs = np.unique(np.array(temp_table['Sampling Rate']))
    if len(fs) > 1:
        ValueError("Different sampling rates within a single subject was detected. Must fix code to handle this... ")
    fs = float(fs)

    # add EEG data
    eeg_electrical_series = ElectricalSeries(
            name="ElectricalSeries",
            data=sig_temp,
            electrodes=all_table_region,
            starting_time=0.0,
            rate=fs)
    nwbfile.add_acquisition(eeg_electrical_series)

    dir_save = os.path.join(dir,subject_id+'.nwb')
    print("Writing .nwb file as ",dir_save)
    with NWBHDF5IO(dir_save, "w") as io:
        io.write(nwbfile)    
        io.close()




# ignore header, wasn't entered

# which channels belong to which mouse???
temp_data = signals[0]
plt.subplot(4,1,1).plot(signals[0][256*98:256*100])
plt.subplot(4,1,2).plot(signals[1][256*98:256*100])
plt.subplot(4,1,3).plot(signals[2][256*98:256*100])
plt.subplot(4,1,4).plot(signals[3][256*98:256*100])

plt.subplot(4,1,1).plot(signals[4][256*98:256*100])
plt.subplot(4,1,2).plot(signals[5][256*98:256*100])
plt.subplot(4,1,3).plot(signals[6][256*98:256*100])
plt.subplot(4,1,4).plot(signals[7][256*98:256*100])


# neuroconv
from datetime import datetime
from dateutil import tz
from pathlib import Path
from neuroconv.datainterfaces import EDFRecordingInterface

file_path = os.path.join(dir,edf_names[0])
#f"{ECEPHY_DATA_PATH}/edf/edf+C.edf"
# Change the file_path to the location in your system
interface = EDFRecordingInterface(file_path=file_path, verbose=False)

# Extract what metadata we can from the source files
metadata = interface.get_metadata()
# For data provenance we add the time zone information to the conversion
session_start_time = datetime(2020, 1, 1, 12, 30, 0, tzinfo=tz.gettz("US/Pacific"))
metadata["NWBFile"].update(session_start_time=session_start_time)

# Choose a path for saving the nwb file and run the conversion
nwbfile_path = os.path.join(dir,'nwbfile.nwb')
interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)

with NWBHDF5IO(nwbpath, "r+") as io:
    print("Reading nwbfile from: ",nwbpath)
    nwbfile = io.read()