# convert_neuralynx2nwb
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
# - JS 07/24/23 adapted code from catalystNeuro

print("Cite NWB and CatalystNeuro")

# get some packages
from datetime import datetime
from dateutil import tz
from pathlib import Path
from neuroconv.datainterfaces import NeuralynxRecordingInterface

# For this data interface we need to pass the folder where the data is
folder_path = input("Enter the data directory: ")

# Change the folder_path to the appropriate location in your system
interface = NeuralynxRecordingInterface(folder_path=folder_path, verbose=False)

# Extract what metadata we can from the source files
metadata = interface.get_metadata()

# Choose a path for saving the nwb file and run the conversion
nwbfile_path = folder_path+"/data_nwb"  # This should be something like: "./saved_file.nwb"
interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)
print("NWB file created and saved to:",nwbfile_path)