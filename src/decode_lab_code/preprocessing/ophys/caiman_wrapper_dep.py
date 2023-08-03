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

        fig, ax = plt.subplots()
        plt.imshow(exData)        
        ax.add_patch(box(xy=(0,0), width=patch_size, height=patch_size, edgecolor = 'yellow',fill=False))
        ax.add_patch(box(xy=(0+patch_overlap,0), width=patch_size, height=patch_size, edgecolor = 'yellow',fill=False))

        return fig

    def get_params(self, motion_correct: bool, patch_size: int, neuron_diameter=int(13), neurons_per_patch=False, calcium_transient_decay=0.4, plot_fig=False):

        """
        This function is meant to motion correct the calcium imaging video.
        In many cases, parameters for caiman don't require changing.

        -- INPUTS -- 
        motion_correct: True or False
        patch_size: size of patch for calcium imaging analysis (the box)
        
        -- OPTIONAL -- 
        neuron_diameter: Average size of neuron, default = 13
        neurons_per_patch: Estimated number of neurons in a given patch, try .test_patch_size to figure this out
        calcium_transient_decay: typical decay of calcium transient. Default is 0.4
        plot_fig: set to True to plot results, recommended False
        """

        # Most parameters do not require changing
        frate = self.frate
        decay_time = calcium_transient_decay  # length of a typical transient in seconds

        # motion correction parameters
        motion_correct = motion_correct   # flag for performing motion correction
        pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
        gSig_filt = ((neuron_diameter-1)/4, (neuron_diameter-1)/4)       # size of high pass spatial filtering, used in 1p data
        max_shifts = (5, 5)      # maximum allowed rigid shift
        strides = (patch_size, patch_size)       # 48 start a new patch for pw-rigid motion correction every x pixels
        overlaps = (patch_size/2, patch_size/2)      # 24 overlap between pathes (size of patch strides+overlaps)
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
            'border_nan': border_nan
        }

        opts = params.CNMFParams(params_dict=mc_dict)        

        # Motion correction - don't need to worry about this much for Akanksha's dataset
        if motion_correct:
            print("Motion correcting...")
            # do motion correction rigid
            mc = MotionCorrect(self.fname, dview=self.dview, **opts.get_group('motion'))
            mc.motion_correct(save_movie=True)
            fname_mc = mc.fname_tot_els if opts.motion['pw_rigid'] else mc.fname_tot_rig
            if opts.motion['pw_rigid']:
                bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                            np.max(np.abs(mc.y_shifts_els)))).astype(int)
            else:
                bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
                if plot_fig:
                    plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
                    plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
                    plt.legend(['x shifts', 'y shifts'])
                    plt.xlabel('frames')
                    plt.ylabel('pixels')

            bord_px = 0 if opts.motion['border_nan'] == 'copy' else bord_px
            fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                    border_to_0=bord_px)
        else:  # if no motion correction just memory map the file
            print("Not motion correcting...")
            bord_px = 0 if opts.motion['border_nan'] == 'copy' else bord_px
            fname_new = cm.save_memmap(self.fname, base_name='memmap_',
                                    order='C', border_to_0=0, dview=self.dview)
        self.fname_new = fname_new
        self.motion_correction_params = opts

        # parameters for source extraction and deconvolution
        p = 1                 # order of the autoregressive system
        K = neurons_per_patch # upper bound on number of components per patch, in general None
        #gSig = (3, 3)        # gaussian width of a 2D gaussian kernel, which approximates a neuron
        gSiz = (neuron_diameter, neuron_diameter)     # average diameter of a neuron, in general 4*gSig+1 (or gSiz-1/4)
        gSig = ((gSiz[0]-1)/4, (gSiz[1]-1)/4)
        Ain = None          # possibility to seed with predetermined binary masks
        merge_thr = .7      # merging threshold, max correlation allowed
        rf = patch_size/2   # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
        stride_cnmf = patch_size/4    # amount of overlap between the patches in pixels
        tsub = 2            # downsampling factor in time for initialization, increase if you have memory problems
        ssub = 1            # downsampling factor in space for initialization,increase if you have memory problems
        low_rank_background = None  # None leaves background of each patch intact, True performs global low-rank approximation if gnb>0
        gnb = 0             # number of background components (rank) if positive,
        nb_patch = 0        # number of background components (rank) per patch if gnb>0, else it is set automatically
        min_corr = .8       # min peak value from correlation image
        min_pnr = 10        # min peak to noise ration from PNR image
        ssub_B = 2          # additional downsampling factor in space for background
        ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

        opts.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                        'K': K,
                                        'gSig': gSig,
                                        'gSiz': gSiz,
                                        'merge_thr': merge_thr,
                                        'p': p,
                                        'tsub': tsub,
                                        'ssub': ssub,
                                        'rf': rf,
                                        'stride': stride_cnmf,
                                        'only_init': True,    # set it to True to run CNMF-E
                                        'nb': gnb,
                                        'nb_patch': nb_patch,
                                        'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                        'low_rank_background': low_rank_background,
                                        'update_background_components': True,  # sometimes setting to False improve the results
                                        'min_corr': min_corr,
                                        'min_pnr': min_pnr,
                                        'normalize_init': False,               # just leave as is
                                        'center_psf': True,                    # leave as is for 1 photon
                                        'ssub_B': ssub_B,
                                        'ring_size_factor': ring_size_factor,
                                        'del_duplicates': True,                # whether to remove duplicates from initialization
                                        'border_pix': bord_px})                # number of pixels to not consider in the borders)

        # load memory mapped file
        Yr, dims, T = cm.load_memmap(fname_new)
        images = Yr.T.reshape((T,) + dims, order='F')        

        # compute some summary images (correlation and peak to noise)
        print("Estimating minimum correlation and peak-to-noise...")
        cn_filter, pnr = cm.summary_images.correlation_pnr(images, gSig=gSig[0], swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
        min_corr = round(np.min(cn_filter),1)
        min_pnr  = round(np.min(pnr),1)
        opts.change_params(params_dict={
            'min_corr': min_corr,
            'min_pnr': min_pnr})        

        self.opts = opts
        return self.opts