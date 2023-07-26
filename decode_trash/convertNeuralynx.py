# convert_neuralynx2nwb
#
# This code is meant to take a full sessions recording and convert it to a single nwb file
# - 07/24/23

# get some packages
from datetime import datetime
from dateutil import tz
from pathlib import Path
from neuroconv.datainterfaces import NeuralynxRecordingInterface

# For this data interface we need to pass the folder where the data is
folder_path = input("Enter the data directory: ")
#folder_path = f"/Users/js0403/Sample data"

# Change the folder_path to the appropriate location in your system
interface = NeuralynxRecordingInterface(folder_path=folder_path, verbose=False)

# Extract what metadata we can from the source files
metadata = interface.get_metadata()

# Choose a path for saving the nwb file and run the conversion
nwbfile_path = f"/Users/js0403/Sample data converted/nwb.nwb"  # This should be something like: "./saved_file.nwb"
interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)
