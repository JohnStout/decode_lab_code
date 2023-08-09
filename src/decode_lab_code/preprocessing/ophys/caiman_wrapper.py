# caiman_wrapper
# This wrapper module is meant to make certain preprocessing steps of caiman into one-liners.
# For example, motion correction shouldn't be a multiline process, but instead a one-liner.
# Extracting data and watching the video playback should be easy. Viewing the results should
# be easy
#
# this code is specifically for miniscope. Future additions will include 2p
#
# written by John Stout using the caiman demos

# prep work
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Rectangle as box
import tifffile as tiff

# parent class
class caiman_preprocess:

    def __init__(self, folder_name: str, file_name: str, frate: int, activate_cluster: bool):
        self.fname = [download_demo(file_name,folder_name)]
        self.frate = frate
        print("Loading movie for",self.fname)
        self.movieFrames = cm.load_movie_chain(self.fname,fr=self.frate) # frame rate = 30f/s

        # this actually doesn't function without activating the cluster
        if activate_cluster:            
            # start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
            if 'dview' in locals():
                cm.stop_server(dview=dview)
            # n_processes reflect cluster num
            print("cluster set-up")
            self.c, self.dview, self.n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)
            
        # autoparameterize

    def get_init_vars(self):
        init_dict = {
            'fname': self.fname,
            'frame_rate': self.frate,
            'frame_data': self.movieFrames,
            'cluster_processes (n_processes)': self.n_processes,
            'ipyparallel.Client_object': self.c,
            'ipyparallel_dview_object': self.dview
        }
        return init_dict
        
    def watch_movie(self):
        # playback
        downsample_ratio = .2  # motion can be perceived better when downsampling in time
        self.movieFrames.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=self.frate, magnification=0.5)   # play movie (press q to exit)   

    def get_frames(self): 
        # get frames
        movieFrames = self.movieFrames
        return movieFrames
    
    def test_patch_size(self, patch_size: int, patch_overlap: int):
        # This function will interact with the user to test the size of patches to play with
        movieFrames = self.movieFrames
        exData = np.mean(movieFrames,axis=0)

        fig, ax = plt.plot()
        plt.imshow(exData)        
        ax.add_patch(box(xy=(0,0), width=patch_size, height=patch_size, edgecolor = 'yellow',fill=False))
        ax.add_patch(box(xy=(0+patch_overlap,0), width=patch_size, height=patch_size, edgecolor = 'yellow',fill=False))

        return fig
    
    def spatial_downsample(self, downsample_factor: int):
        """
        Spatially downsample your data (downsample pixels) by a chosen factor
        
        --- INPUTS ---
        downsample_factor: # of times to downsample. If downsample_factor = 1, then you will
            spatially downsample the dataset every other datapoint. If downsample_factor = 2, then
            you will spatially downsample the data two separate times along each axis.

        """
        for i in range(downsample_factor):

            # when slicing, start:stop:step
            frameShape = self.movieFrames.shape # frameshape

            # downsample
            self.movieFrames = self.movieFrames[:,0:frameShape[1]:2,0:frameShape[2]:2]

        return self.movieFrames
    
    def temporal_downsample(self, downsample_factor: int):
        """
        Spatially downsample your data (downsample pixels) by a chosen factor
        
        --- INPUTS ---
        downsample_factor: # of times to downsample. If downsample_factor = 1, then you will
            temporally downsample your dataset, essentially cutting your frame rate by downsample_factor

        """
        for i in range(downsample_factor):

            # when slicing, start:stop:step
            frameShape = self.movieFrames.shape # frameshape

            # downsample
            self.movieFrames = self.movieFrames[0:frameShape[0]:2,:,:]

        return self.movieFrames    
    
    def save_output(self):
        """
        Saving the output. This is useful if you downsampled your dataset and wish to reload the results
        """
        print("Saving output")
        self.fname
        self.file_root = self.fname[0].split('.')[0]
        tiff.imsave(self.file_root+'.tif',self.movieFrames)
