# this code requires the following packages to have been run:
#   bftools
#       cd bftools
#       ffmpeg -i /Users/js0403/ophysdata/Akanksha_data/Movie_11_static.avi -vf format=pix_fmts=rgba /Users/js0403/ophysdata/Akanksha_data/tiffdata/Movie_11_static%4d.tiff

from decode_lab_code.preprocessing.ophys.caiman_wrapper import caiman_preprocess
import numpy as np
from tifffile import imwrite

folder_name = '/Users/js0403/ophysdata/Akanksha_data'
file_name = 'Movie_11_static'
extension = '.avi'
frame_rate = 30
cp = caiman_preprocess(folder_name,file_name+extension,frame_rate,activate_cluster=False)
images = np.array(cp.get_frames())



fname = [download_demo(file_name+extension,folder_name)]
movieFrames = cm.load_movie_chain(fname,fr=frame_rate) # frame rate = 30f/s


import glob
import tifffile as tf

# pull in all files with the .tif extension and sort them
pathnames = glob.glob('/Users/js0403/ophysdata/Akanksha_data/tiffdata/*.tiff')
pathnames.sort()

# read in one image to get shape
im = tf.imread(pathnames[0])
image_shape = im.shape

images = []
counter = 0
for iname in pathnames:
    im = tf.imread(iname)
    images.append(tf.imread(iname))
    counter = counter+1
    print("Completed with",round((counter/len(pathnames)*100)),'%')



