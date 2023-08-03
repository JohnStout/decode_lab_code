#%% Script for messing with caiman parameters
#
# This in-vitro calcium imaging dataset is unique from 1p because the quality
# is so good, but unique from 2p and 1p because the dynamics are relatively poor.
#
# This in-vitro calcium imaging experiment is somewhere between 1p and 2p and 
# therefore, the parameters must treat the data as such
#
# -JS

#%% 

# importing packages/modules
from decode_lab_code.preprocessing.ophys.caiman_wrapper import caiman_preprocess

import matplotlib.pyplot as plt

from caiman.source_extraction import cnmf
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params

import numpy as np

#%%

# assign a folder name for analysis
folder_name = '/Users/js0403/ophysdata/Trevor_750K2_200ms_RedGreen_depolar002'
file_name = 'rec_neuron.tif'
frame_rate = 10
cp = caiman_preprocess(folder_name,file_name,frame_rate,activate_cluster=False)

#%%

# play video
cp.watch_movie()

#%%

# plot figure to determine cell size
frameData = cp.get_frames()
plt.imshow(frameData[10,:,:])
plt.ylim(220, 190)
plt.xlim(50, 100)
plt.title("Neuron size ~= 10-13 pixels")

# based on the visualization, 13 pixels is consistent with caiman
neuron_size = 13 # pixels - this can be adjusted as needed after visualizing results

#%% lets identify a good patch size
patch_size = 192; patch_overlap = patch_size/2
cp.test_patch_size(patch_size,patch_overlap)

# %%

# dataset dependent parameters
fname = cp.fname  # directory of data
fr = frame_rate   # imaging rate in frames per second
decay_time = 0.4  # length of a typical transient in seconds

# motion correction parameters - we don't worry about mc
motion_correct = False      # flag for motion correcting - we don't need to here
strides = (patch_size, patch_size)          # start a new patch for pw-rigid motion correction every x pixels
overlaps = (patch_overlap, patch_overlap)   # overlap between pathes (size of patch strides+overlaps)
max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
pw_rigid = False            # flag for performing non-rigid motion correction
border_nan = 'copy'         # replicate values along the border
if border_nan == 'copy': bord_px = 0
gSig_filt = (3,3) # change

# parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system
gnb = 2                     # number of global background components
merge_thr = 0.85            # merging threshold, max correlation allowed
rf = int(patch_size/2)         # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = int(patch_size/4)  # amount of overlap between the patches in pixels
K = 4                       # number of components per patch
gSiz = (neuron_size,neuron_size) # estimate size of neuron
gSig = [int(round(neuron_size-1)/2), int(round(neuron_size-1)/2)] # expected half size of neurons in pixels
method_init = 'corr_pnr'  # greedy_roi, initialization method (if analyzing dendritic data using 'sparse_nmf'), if 1p, use 'corr_pnr'
ssub = 2                    # spatial subsampling during initialization
tsub = 1                    # temporal subsampling during intialization
ssub_B = 2                  # additional downsampling factor in space for background
low_rank_background = None  # None leaves background of each patch intact, True performs global low-rank approximation if gnb>0
gnb = 0                     # number of background components, gnb= 0: Return background as b and W, gnb=-1: Return full rank background B, gnb<-1: Don't return background
nb_patch = 0                # number of background components (rank) per patch if gnb>0, else it is set automatically
ring_size_factor = 1.4      # radius of ring is gSiz*ring_size_factor

# These values need to change based on the correlation image
min_corr = .8               # min peak value from correlation image
min_pnr = 10                # min peak to noise ration from PNR image

# parameters for component evaluation
min_SNR = 2.0    # signal to noise ratio for accepting a component
rval_thr = 0.85  # space correlation threshold for accepting a component
cnn_thr = 0.99   # threshold for CNN based classifier
cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

# %%

# create a parameterization object
opts_dict = {
            # parameters for opts.data
            'fnames': fname,
            'fr': fr,
            'decay_time': decay_time,

            # parameters for opts.motion  
            'strides': strides,
            'pw_rigid': pw_rigid,
            'border_nan': border_nan,
            'gSig_filt': gSig_filt,
            'max_deviation_rigid': max_deviation_rigid,   
            'overlaps': overlaps,
            'max_shifts': max_shifts,    

            # parameters for preprocessing
            #'n_pixels_per_process': None, # how is this estimated?

            # parameters for opts.init
            'K': K, 
            'gSig': gSig,
            'gSiz': gSiz,  
            'nb': gnb, # also belongs to params.temporal 
            'normalize_init': False,   
            'rolling_sum': True,    
            'ssub': ssub,
            'tsub': tsub,
            'ssub_B': ssub_B,    
            'center_psf': True,
            'min_corr': min_corr,
            'min_pnr': min_pnr,            

            # parameters for opts.patch
            'border_pix': bord_px,  
            'del_duplicates': True,
            'rf': rf,  
            'stride': stride_cnmf,
            'low_rank_background': low_rank_background,                     
            'only_init': True,

            # parameters for opts.spatial
            'update_background_components': True,

            # parameters for opts.temporal
            'method_deconvolution': 'oasis',
            'p': p,

            # parameters for opts.quality
            'min_SNR': min_SNR,
            'cnn_lowest': cnn_lowest,
            'rval_thr': rval_thr,
            'use_cnn': True,
            'min_cnn_thr': cnn_thr,

            # not sure
            'method_init': method_init,
            'merge_thr': merge_thr, 
            'ring_size_factor': ring_size_factor}

opts = params.CNMFParams(params_dict=opts_dict)

#%% 

# start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

#%% MEMORY MAPPING

# memory map the file in order 'C'
fname_new = cm.save_memmap(fname, base_name='memmap_',
                            order='C', border_to_0=0, dview=dview) # exclude borders

# now load the file
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F') 
#images = Yr.T.reshape((T,) + dims, order='F')
    #load frames in python format (T x X x Y)

#%% 

# restart cluster to clean up memory
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# %%

# I really need to understand these summary images, like how to interpret

# run summary image
cn_filter, pnr = cm.summary_images.correlation_pnr(
    images, gSig=gSig[0],center_psf=True,swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile

# change params for min_corr and min_pnr
min_corr = round(np.min(cn_filter),1)
min_pnr  = round(np.min(pnr),1)
opts.change_params(params_dict={
    'min_corr': min_corr,
    'min_pnr': min_pnr})

# plot images
plt.subplot(1,3,1).imshow(np.mean(images,axis=0))
plt.title("Raw data")
plt.subplot(1,3,2).imshow(cn_filter)
plt.title("CN_filter")
plt.subplot(1,3,3).imshow(pnr)
plt.title("PNR")

# %%

# fit data with cnmf
cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=None, params=opts)
cnm.fit(images)

# %%
