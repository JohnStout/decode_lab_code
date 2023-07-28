# convertNLX2NWB
#
# This code is meant to take a full sessions recording and convert it to a single nwb file
#
# -- Dependencies/how to -- #
#
# This code depends on: https://neuroconv.readthedocs.io/en/main/index.html
# If you have never run this code, make sure that:
#   1) Open terminal
#   2) Download anaconda and git (for terminal), open terminal
#   3) git clone https://github.com/catalystneuro/neuroconv
#   4) cd neuroconv
#   5) conda env create -f make_environment.yml
#   6) conda activate neuroconv_environment
#
# If you are a returning user OR are new and have done the steps above:
#   1) right-click the play button and "run in interactive"
#   2) select the neuroconv_environment interpreter in the interactive window
#
# This code was adapted by catalyst neuro
#
# - written by JS 07/24/23 by adapting code from catalystNeuro

print("Cite NWB and CatalystNeuro")

# get some packages
#from datetime import datetime
#from dateutil import tz
#from pathlib import Path
#from neuroconv.datainterfaces import NeuralynxRecordingInterface

if demo == 'y':
    try:
        writeNWB(folder_path)
        nwb_data = readNWB(folder_path)
    except:
        print("nwb_data file likely already present, attempting to load...")
        nwb_data = readNWB(folder_path)
else:
    nwb_data = readNWB(folder_path)

# interface with the user
folder_path = input("Enter directory of recording session: ")
session_description = input("Enter description of this recording session: ")
session_notes = input("Enter notes pertaining to this session: ")
session_experimenter = input("Enter the name of the experimenter: ")

# function for saving nwb data - this is not so useful right now
def writeNWB(folder_path):

    # Change the folder_path to the appropriate location in your system
    interface = NeuralynxRecordingInterface(folder_path=folder_path, verbose=False)

    # Extract what metadata we can from the source files
    metadata = interface.get_metadata() # here we should change them
    metadata['NWBFile']['session_description'] = session_description
    metadata['NWBFile']['notes'] = session_notes
    metadata['NWBFile']['Experimenter']
    
    # Choose a path for saving the nwb file and run the conversion
    nwbfile_path = folder_path+'/data_nwb'  # This should be something like: "./saved_file.nwb"
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)
    print("NWB file created and saved to:",nwbfile_path) 

# function for reading nwb data
def readNWB(fileDir,fileName):

    # Open the file in read mode "r", and specify the driver as "ros3" for S3 files
    #filePath = '/Users/js0403/Sample-data/data_nwb'
    filePath = fileDir+'/data_nwb'
    io = NWBHDF5IO(filePath, mode="r")
    nwbfile = io.read()
    return nwbfile 