# This function is meant to return an nwb file
#
# -- INPUTS -- #
# fileDir: directory of file
# fileName: name of file
#
# -- OUTPUTS -- #
# nwbfile: nwb file
#
# - JS

# import some packages
import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO

def readNWB(fileDir,fileName):

    # Open the file in read mode "r", and specify the driver as "ros3" for S3 files
    #filePath = '/Users/js0403/Sample-data/data_nwb'
    filePath = fileDir+fileName
    io = NWBHDF5IO(filePath, mode="r")
    nwbfile = io.read()
    return nwbfile

