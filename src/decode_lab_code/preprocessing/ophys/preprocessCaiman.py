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

class caiman_process:

    def __init__(self, folder_name: str, file_name: str, frate: int):
        self.fname = [download_demo(file_name,folder_name)]
        self.frate = frate
        self.movieFrames = cm.load_movie_chain(self.fname,fr=self.frate) # frame rate = 30f/s

    def start_cluster(self):
        # start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)
        
    def watch_movie(self):
        # playback
        downsample_ratio = .2  # motion can be perceived better when downsampling in time
        self.movieFrames.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=self.frate, magnification=0.5)   # play movie (press q to exit)   

    def motion_correct(self, motion_correct: bool):

        # Set parameters

        # dataset dependent parameters 
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
            fname_new = cm.save_memmap(fname, base_name='memmap_',
                                    order='C', border_to_0=0, dview=dview)
        self.fname_new = fname_new

        # load memory mappable file
        Yr, dims, T = cm.load_memmap(fname_new)
        images = Yr.T.reshape((T,) + dims, order='F')
        self.images = images

        # parameters for source extraction and deconvolution
        p = 1               # order of the autoregressive system
        K = None            # upper bound on number of components per patch, in general None
        gSig = (3, 3)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
        gSiz = (13, 13)     # average diameter of a neuron, in general 4*gSig+1
        Ain = None          # possibility to seed with predetermined binary masks
        merge_thr = .7      # merging threshold, max correlation allowed
        rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
        stride_cnmf = 20    # amount of overlap between the patches in pixels
        #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
        tsub = 2            # downsampling factor in time for initialization,
        #                     increase if you have memory problems
        ssub = 1            # downsampling factor in space for initialization,
        #                     increase if you have memory problems
        #                     you can pass them here as boolean vectors
        low_rank_background = None  # None leaves background of each patch intact,
        #                     True performs global low-rank approximation if gnb>0
        gnb = 0             # number of background components (rank) if positive,
        #                     else exact ring model with following settings
        #                         gnb= 0: Return background as b and W
        #                         gnb=-1: Return full rank background B
        #                         gnb<-1: Don't return background
        nb_patch = 0        # number of background components (rank) per patch if gnb>0,
        #                     else it is set automatically
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
        self.opts = opts

    def show_summary(self):

        # compute some summary images (correlation and peak to noise)
        cn_filter, pnr = cm.summary_images.correlation_pnr(self.images, gSig=gSig[0], swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile

        # inspect the summary images and set the parameters
        nb_inspect_correlation_pnr(cn_filter, pnr)