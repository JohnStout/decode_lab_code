# This code will generate an NWB file for ophys data
from datetime import datetime
from uuid import uuid4
import numpy as np
from dateutil import tz
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

import matplotlib.pyplot as plt
import numpy as np
from dateutil.tz import tzlocal

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

# working on getting the real ophys way working. nwbwidgets spits an error
ophys_prep = 1
if ophys_prep == 1:
    # create device
    device = nwbfile.create_device(
        name="Microscope",
        description="My two-photon microscope",
        manufacturer="The best microscope manufacturer",
    )
    optical_channel = OpticalChannel(
        name="OpticalChannel",
        description="an optical channel",
        emission_lambda=525.0,
    )

    # create imagingplane object
    imaging_plane = nwbfile.create_imaging_plane(
        name="ImagingPlane",
        optical_channel=optical_channel,
        imaging_rate=frame_rate,
        description="Activation of cells",
        device=device,
        excitation_lambda=600.0,
        indicator="GFP",
        location="Somewhere",
        grid_spacing=[0.01, 0.01],
        grid_spacing_unit="meters",
        origin_coords=[1.0, 2.0, 3.0],
        origin_coords_unit="meters",
    )

    # using internal data. this data will be stored inside the NWB file
    one_p_series1 = OnePhotonSeries(
        name="CalciumDye",
        data=data,
        imaging_plane=imaging_plane,
        rate=10.0,
        unit="pixels",
    )

    nwbfile.add_acquisition(one_p_series1)
    with NWBHDF5IO(folder_name+"/data_ophys_nwb.nwb", "w") as io:
        io.write(nwbfile)

else:
    # time series option
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