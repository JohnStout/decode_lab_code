# This code will generate an NWB file for ophys data
from datetime import datetime
from uuid import uuid4
import numpy as np
from dateutil import tz
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from decode_lab_code.preprocessing.ophys.caiman_wrapper import caiman_preprocess

session_start_time = datetime(2018, 4, 25, 2, 30, 3, tzinfo=tz.gettz("US/Pacific"))
# initialize the nwbfile
nwbfile = NWBFile(
    session_description=input("Enter description of your recording session: "),  # required
    identifier=str(uuid4()),  # required
    session_start_time=session_start_time,  # required
    session_id=input("Enter unique identifier for session: "),  # optional
    experimenter=[
        input("Enter experimenter name: "),
    ],  # optional
    lab=input("Enter lab name: "),  # optional
    institution=input("Enter institution name: "),  # optional
    experiment_description=input("Enter a description of your experiment"),  # optional
    related_publications=input("Enter any information about publication (if relevant)"),  # optional
)

# enter information about subject
nwbfile.subject = Subject(
    subject_id=input("Enter subject ID: "),
    age=input("Enter subject age as such (PD100):  "),
    description=input("Enter subject identifier: "),
    species=input("Enter species name: "),
    sex=input("Enter sex of subject: "),
)

# directory information
folder_name = input("Enter the folder name for your data: ")
fname_neuron = input("Enter file name with extension: ")
frame_rate = float(input("Enter the frame rate: "))

# if you get the Error: "Exception: A cluster is already runnning", restart the kernel
cp = caiman_preprocess(folder_name,fname_neuron,frame_rate,False)

# lets load in 
data = cp.get_frames()

time_series_with_rate = TimeSeries(
    name="ophys",
    data=data,
    unit="pixels",
    starting_time=0.0,
    # I'm not sure if this is numsamples/sec or sec/numsamples
    rate=frame_rate, # sampled every second (make sure this is correct***)
)
time_series_with_rate
nwbfile.add_acquisition(time_series_with_rate)

# write
with NWBHDF5IO(folder_name+"/data_nwb.nwb", "w") as io:
    io.write(nwbfile)