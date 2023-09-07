# required packages
import numpy as np
import tifffile as tf
import glob

def stacktiff(dir: str, dir_save = None):
    """
    This function takes a folder with a bunch of .tif images and stacks them

    Args:
        dir: directory containing image data to stack
        dir_save: OPTIONAL but recommended. Directory to save stacked data.
    """

    if dir_save is None:
        dir_save = dir

    # change me
    # dir = directory of data (change me)
    #dir = '/Volumes/decode/Akanksha/Slice_calcium_imaging_videos_images/Pilots/Static_recordings/08-31-2023/Tiff_series_Process_7'
    extension = '.tif' # no need to change
    mid_ext = '/*' # don't change

    # do stuff
    pathnames = glob.glob(dir+mid_ext+extension)
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
        print("Completed with",(counter/len(pathnames)*100),'%')

    tf.imwrite(dir_save+'/tiff_stack.tif')



