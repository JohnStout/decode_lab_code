'''
convert miniscope movies to tif
You should run this script after you have collected your miniscope data 
and are ready to convert to .tif files

The miniscope_to_tif function takes advantage of lazy loading and writes new data
to disk in a memory-efficient manner

You should set your environment to caiman or mescore and if you run into an error
related to the imageio package, you probably just need to update it or add ffmpeg.
The instructions should be raised in the error itself.

John Stout
'''

# import packages
import numpy as np
import matplotlib.pyplot as plt
from decode_lab_code.calcium_imaging import preprocessing_utils as pu

# change me
movie_path = r'/Users/js0403/miniscope/data/134A/AAV2/3-Syn-GCaMP8f/2024_01_23/14_04_52/miniscopeDeviceName'

# run
pu.miniscope_to_tif(movie_path = movie_path)