# Python 3.9

"""

Download the latest release for miniscope software: 

        https://github.com/Aharoni-Lab/Miniscope-DAQ-QT-Software/releases

pip install numpy pynwb pandas opencv-python ipykernel matplotlib

This package works with miniscope v4.4.

This package will create a very simple NWB file, after which you can load and append new information to. Like for example, if you want to add motion corrected or ROI data, you would do that

John Stout

"""


#%% 
import os
import json
import pandas

# This code will generate an NWB file for ophys data
from datetime import datetime
from dateutil import tz
from dateutil.tz import tzlocal

from uuid import uuid4

import cv2

import numpy as np
import matplotlib.pyplot as plt

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

#from skimage import io

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.image import ImageSeries
from pynwb.ophys import (
    CorrectedImageStack,
    Fluorescence,
    ImageSegmentation,
    MotionCorrection,
    OnePhotonSeries,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
)

#%% This chunk is dedicated to defining directories and loading .json or .csv files

# I really should build in a search method in case these things change with updated versions
#metaData_json = [i for i in dir_contents if 'metaData' in i][0]

# could also run a search for anything json related and store that as a 'metaData' file
#   and anything .csv related

# dir will be the only input to this code
dir = '/Users/js0403/miniscope/data/134A/AAV2/3-Syn-GCaMP8f/2023_11_14/13_21_49'
dir_contents = sorted(os.listdir(dir))

# Directly accessible information
folder_metaData = json.load(open(os.path.join(dir,'metaData.json')))
folder_notes = pandas.read_csv(os.path.join(dir,'notes.csv'))

# behavior
behavior_id = [i for i in dir_contents if 'behavior' in i][0]
behavior_dir = os.path.join(dir,behavior_id)
behavior_metaData = json.load(open(os.path.join(behavior_dir,'metaData.json')))
behavior_pose = pandas.read_csv(os.path.join(behavior_dir,'pose.csv'))

# cameraDevice
camera_id = [i for i in dir_contents if 'camera' in i][0]
camera_dir = os.path.join(dir,camera_id)
camera_metaData = json.load(open(os.path.join(camera_dir,'metaData.json')))
camera_times = pandas.read_csv(os.path.join(camera_dir,'timeStamps.csv'))

# miniscopeDevice - where the miniscope data is located - use this to identify miniscope file name
miniscope_id = [i for i in dir_contents if 'miniscope' in i][0]
miniscope_dir = os.path.join(dir,miniscope_id)
miniscope_data = [i for i in sorted(os.listdir(miniscope_dir)) if '.avi' in i]
miniscope_metaData = json.load(open(os.path.join(miniscope_dir,'metaData.json')))
miniscope_times = pandas.read_csv(os.path.join(miniscope_dir,'timeStamps.csv'))
miniscope_head_orientation = pandas.read_csv(os.path.join(miniscope_dir,'headOrientation.csv'))

# experiment
print("This version does not support the experiment folder due to no testing data")

# %% Put data into NWB

# Can I write the data to an NWB file lazily? So like, load a bunch of frames, write a bunch of frames, cont....

# year, month, day, hour, minute, second
time_data = folder_metaData['recordingStartTime']
rec_time = datetime(time_data['year'],time_data['month'],time_data['day'],
                    time_data['hour'],time_data['minute'],time_data['second'],
                    time_data['msec'],tzinfo=tzlocal())

# creating the NWBFile
print("This file does not handle multiple custom entries")
nwbfile = NWBFile(
    session_description=input("Enter a description of what you did this session: "),
    identifier=str(uuid4()),
    session_start_time=rec_time,
    experimenter=[folder_metaData['researcherName']],
    lab=input("Enter lab name: "),
    institution=input("Enter institution name: "),
    experiment_description=folder_metaData['experimentName'],
    session_id=folder_metaData['baseDirectory'].split('/')[-1],
    notes = folder_metaData['customEntry0']
    #viral_construct = input("Enter the virus used for imaging: ")
)

# TODO subject

# imaging device
device = nwbfile.create_device(
    name = miniscope_metaData['deviceType'],
    description="UCLA Miniscope v4.4",
    manufacturer="Open Ephys",
)
optical_channel = OpticalChannel(
    name="OpticalChannel",
    description="an optical channel",
    emission_lambda=500.0, # NOT SURE HOW I FIND THIS
)

imaging_plane = nwbfile.create_imaging_plane(
    name="ImagingPlane",
    optical_channel=optical_channel,
    imaging_rate=float(miniscope_metaData['frameRate']),
    description=input("What kinds of cells are you targeting? "),
    device=device,
    excitation_lambda=600.0, # WHERE DO I FIND THIS??
    indicator=input("Enter the viral construct used for imaging (e.g. AAV2/3-Syn-GCaMP8f): "),
    location=input("Enter your brain structure (e.g. PFC/V1/M2/CA1 etc...)"),
)

# save the nwb file
nwbpath = os.path.join(dir,"nwbfile.nwb")
with NWBHDF5IO(nwbpath, mode="w") as io:
    io.write(nwbfile)

#%% Writing data to NWB file

# We are going to load in data from miniscope, this isn't a big deal.
# The miniscope only records 1000 samples at a time. 
# The big deal is being able to write the file as a new one-p-timeseries lazily



# STUCK HERE. HOW CAN WE ITERATIVELY ADD AN ONEPHOTONSERIES OBJECT TO DISK????


# read video
print("Reading video")
movie_data = []
cap = cv2.VideoCapture(os.path.join(miniscope_dir,miniscope_data[0])) 
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        break
    else:
        movie_data.append(frame[:,:,0]) # only the first array matters
movie_mat = np.dstack(movie_data)
plt.imshow(movie_mat[:,:,0])

with NWBHDF5IO(nwbpath, "r") as io:
    nwbfile = io.read()

    # using internal data. this data will be stored inside the NWB file
    one_p_series = OnePhotonSeries(
        name="OnePhotonSeries_internal2",
        data=np.ones((1000, 100, 100)),
        imaging_plane=imaging_plane,
        rate=1.0,
        unit="normalized amplitude",
    )    
    nwbfile.add_acquisition(one_p_series)

    with NWBHDF5IO(nwbpath, mode="w") as io:
        io.write(nwbfile)
io.close()

    del nwbfile



# there has got to be a way to save the .nwb file as a bunch of zeros, but then write each image individually
miniscope_series = []; counter = 0
for i in miniscope_data:
    temp_dir = os.path.join(miniscope_dir,i)
    print(temp_dir)

    # using internal data. this data will be stored inside the NWB file
    one_p_series = OnePhotonSeries(
        name="OnePhotonSeries_internal",
        data=np.ones((1000, 100, 100)),
        imaging_plane=imaging_plane,
        rate=1.0,
        unit="normalized amplitude",
    )

    nwbfile.add_acquisition(one_p_series)
