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

# parent class
class caiman_preprocess:

    def __init__(self, folder_name: str, file_name: str, frate: int, activate_cluster: bool):
        self.fname = [download_demo(file_name,folder_name)]
        self.frate = frate
        print("Loading movie")
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
        movieFrames = self.movieFrames
        return movieFrames
    
    def motion_correct(self, motion_correct: bool, opts):

        # Motion correction - don't need to worry about this much for Akanksha's dataset
        print("This might take a minute...")
        if motion_correct:
            # do motion correction rigid
            mc = MotionCorrect(self.fname, dview=self.dview, **opts.get_group('motion'))
            mc.motion_correct(save_movie=True)
            fname_mc = mc.fname_tot_els if opts.motion['pw_rigid'] else mc.fname_tot_rig
            if opts.motion['pw_rigid']:
                bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                            np.max(np.abs(mc.y_shifts_els)))).astype(int)
            else:
                bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
                plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
                plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
                plt.legend(['x shifts', 'y shifts'])
                plt.xlabel('frames')
                plt.ylabel('pixels')

            bord_px = 0 if opts.motion['border_nan'] == 'copy' else bord_px
            fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                    border_to_0=bord_px)
        else:  # if no motion correction just memory map the file
            bord_px = 0 if opts.motion['border_nan'] == 'copy' else bord_px
            fname_new = cm.save_memmap(self.fname, base_name='memmap_',
                                    order='C', border_to_0=0, dview=self.dview)