# caiman based analysis

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

    def __init__(self, folder_name: str, file_name: str, frate: int):
        self.fname = [download_demo(file_name,folder_name)]
        self.frate = frate
        self.movieFrames = cm.load_movie_chain(self.fname,fr=self.frate) # frame rate = 30f/s

        # start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        self.c, self.dview, self.n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)
        
    def get_cluster_vars(self):
        return self.n_processes, self.c, self.dview
        
    def watch_movie(self):
        # playback
        downsample_ratio = .2  # motion can be perceived better when downsampling in time
        self.movieFrames.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=self.frate, magnification=0.5)   # play movie (press q to exit)   

    def get_frames(self): 
        movieFrames = self.movieFrames
        return movieFrames
    
    def motion_correct(self, motion_correct: bool):

        # Set parameters

        # default parameters
        decay_time = 0.4 # length of a typical transient in seconds

        # motion correction parameters - set to false for trevor/akanksha data
        # motion_correct = False   # flag for performing motion correction 
        pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
        gSig_filt = (3, 3)       # size of high pass spatial filtering, used in 1p data
        max_shifts = (5, 5)      # maximum allowed rigid shift
        strides = (48, 48)     # 48 start a new patch for pw-rigid motion correction every x pixels
        overlaps = (24, 24)      # 24 overlap between pathes (size of patch strides+overlaps)
        max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
        border_nan = 'copy'      # replicate values along the boundaries

        mc_dict = {
            'fnames': self.fname,
            'fr': self.frate,
            'decay_time': decay_time,
            'pw_rigid': pw_rigid,
            'max_shifts': max_shifts,
            'gSig_filt': gSig_filt,
            'strides': strides,
            'overlaps': overlaps,
            'max_deviation_rigid': max_deviation_rigid,
            'border_nan': border_nan,
        }
        opts = params.CNMFParams(params_dict=mc_dict)

        # Motion correction - don't need to worry about this much for Akanksha's dataset
        print("This might take a minute...")
        if motion_correct:
            # do motion correction rigid
            mc = MotionCorrect(self.fname, dview=dview, **opts.get_group('motion'))
            mc.motion_correct(save_movie=True)
            fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
            if pw_rigid:
                bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                            np.max(np.abs(mc.y_shifts_els)))).astype(int)
            else:
                bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
                plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
                plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
                plt.legend(['x shifts', 'y shifts'])
                plt.xlabel('frames')
                plt.ylabel('pixels')

            bord_px = 0 if border_nan == 'copy' else bord_px
            fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                    border_to_0=bord_px)
        else:  # if no motion correction just memory map the file
            bord_px = 0 if border_nan == 'copy' else bord_px
            fname_new = cm.save_memmap(self.fname, base_name='memmap_',
                                    order='C', border_to_0=0, dview=self.dview)
            
        return fname_new, opts
    