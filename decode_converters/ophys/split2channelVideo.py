# split2channelVideo
# This script is meant to split calcium image videos by channel.
# For example, 1 channel might image calcium transients, while the other
#   images a static color
#
# -- Dependencies -- 
# This code requires that you have downloaded the caiman pipeline successfully
# In VScode, you should activate your environment and CD to your downloaded caiman folder
#
# - JS - 07/21/23

import caiman as cm
import os
import tifffile as tiff

# interface with user
loadFolder = input("Please enter the directory to load data: ")
fileName   = input("Please enter the file name, include the file type: ")
saveFolder = input("Please enter the directory to save data: ")

# Split 2-dye calcium imaging videos and save separately
#loadFolder = '/Users/js0403/caiman_data/example_movies'
#fileName   = '750K2_200ms_4XSum_video_RedGreen_depolar002.tif'
path_movie = [os.path.join(loadFolder, fileName)]
print("loading movie...")
m_orig = cm.load_movie_chain(path_movie, is3D=True)
m_temp = m_orig[:,1,:,:]
m_temp.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=frate, magnification=0.5)   # play movie (press q to exit)

# enter which channel is which
channel1 = input("Enter name for channel1: ")
channel2 = input("Enter name for channel2: ")

# extract movies
m_ch1 = m_orig[:,0,:,:]
m_ch2 = m_orig[:,1,:,:]

# save videos
path_ch1 = os.path.join(saveFolder,channel1+'_'+fileName)
path_ch2 = os.path.join(saveFolder,channel2+'_'+fileName)
print("Saving as", path_ch1)
print("Saving as", path_ch2)

# play the videos
frate = float(input("Enter the framerate: "))
downsample_ratio = .2  # motion can be perceived better when downsampling in time
m_ch1.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=frate, magnification=0.5)   # play movie (press q to exit)
userConfirm = input("Does this video look okay? [y/n]: ")
if userConfirm == 'y':
    tiff.imsave(path_ch1,m_ch1)

m_ch2.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=frate, magnification=0.5)   # play movie (press q to exit)
userConfirm = input("Does this video look okay? [y/n]: ")
if userConfirm == 'y':
    tiff.imsave(path_ch2,m_ch2)

print("Split complete")