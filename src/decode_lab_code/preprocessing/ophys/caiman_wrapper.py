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
from datetime import datetime

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

        fig, ax = plt.subplots()
        plt.imshow(exData)        
        ax.add_patch(box(xy=(0,0), width=patch_size, height=patch_size, edgecolor = 'yellow',fill=False))
        ax.add_patch(box(xy=(0+patch_overlap,0), width=patch_size, height=patch_size, edgecolor = 'yellow',fill=False))

        #return fig
    
    def spatial_downsample(self, downsample_factor: int):
        """
        Spatially downsample your data (downsample pixels) by a chosen factor
        
        --- INPUTS ---
        downsample_factor: factor to spatially downsample your dataset

        """

        # when slicing, start:stop:step
        frameShape = self.movieFrames.shape # frameshape

        # downsample
        self.movieFrames = self.movieFrames[:,0:frameShape[1]:downsample_factor,0:frameShape[2]:downsample_factor]

        return self.movieFrames
    
    def temporal_downsample(self, downsample_factor: int):
        """
        Spatially downsample your data (downsample pixels) by a chosen factor
        
        --- INPUTS ---
        downsample_factor: scale for downsampling
        """

        # when slicing, start:stop:step
        frameShape = self.movieFrames.shape # frameshape

        # downsample
        self.movieFrames = self.movieFrames[0:frameShape[0]:downsample_factor,:,:]

        self.frate = self.frate/downsample_factor

        return self.movieFrames  
    
    def motion_correct(self, dview, opts, pw_rigid=False):
            
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

            self.fname_new = cm.save_memmap(fname_mc, base_name=file_name+'_memmap_', order='C',
                                    border_to_0=bord_px)   
    
    def save_memap(self, file_name: str, dview):
        """
            This function saves your data as a memory-mapped file 

            --INPUTS--
            file_name: file directory
            dview: pool output from cluster_helper.start_cluster or cluster_helper.refresh_cluster

        """

        self.fname_new = cm.save_memmap(self.fname, base_name=file_name+'_memmap_',
                        order='C', border_to_0=0, dview=dview)

    def save_output(self):
        """
        Saving the output. This is useful if you downsampled your dataset and wish to reload the results
        """
        self.fname
        self.file_root = self.fname[0].split('.')[0]
        print("Saving output as",self.file_root+'.tif')
        tiff.imsave(self.file_root+'.tif',self.movieFrames)

class caiman_cnm_curation:

    def __init__(self):
        """
        """

    def component_eval(images, cnm, dview, min_SNR=2, r_values_min=0.9):

        """ 
        component_eval: function meant to evaluate components. Must run this before cleaning up dataset.

        -- INPUTS -- 
            cnm: cnm object
            dview: multiprocessing toolbox state
            min_SNR: signal-noise-ratio, a threshold for transient size
            r_values_min: threshold for spatial consistency (lower to increase component yield)
        
        -- OUTPUTS --
            cnm: cnm object with components
        """
        
        #min_SNR = 2            # adaptive way to set threshold on the transient size
        #r_values_min = 0.9     # threshold on space consistency (if you lower more components
        #                        will be accepted, potentially with worst quality)
        cnm.params.set('quality', {'min_SNR': min_SNR,
                                'rval_thr': r_values_min,
                                'use_cnn': False})

        # component evaluation
        cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

        print(' ***** ')
        print('Number of total components: ', len(cnm.estimates.C))
        print('Number of accepted components: ', len(cnm.estimates.idx_components))

        return cnm
    

    # a plotter function
    def plot_components(img, cnm, colors: list, colorMap='viridis'):

        """
            img: image to plot results over
            cnm: cnm object 
            colors: list of color indicators
            colorMap: color for imshow map
        """

        # extract components
        rois = cnm.estimates.coordinates # get rois
        idx = cnm.estimates.idx_components # get accepted components
        good_rois = [rois[i] for i in range(len(rois)) if i in idx]


        # plot components
        plt.subplot(1,2,1)
        plt.imshow(img,colorMap)
        for i in range(len(good_rois)):
            roi = good_rois[i].get('coordinates')
            CoM = good_rois[i].get('CoM')
            plt.plot(roi.T[0,:],roi.T[1,:],c=colors[i],linewidth=2)
            plt.text(CoM[1], CoM[0], str(i + 1), color='w', fontsize=10)

        # plot traces
        cnm.estimates.detrend_df_f(flag_auto=True, frames_window=100, detrend_only=True) # get normalized df/f
        fr = cnm.params.data['fr'] # get frame rate
        dfF_all = cnm.estimates.F_dff # filter
        dfF_good = dfF_all[cnm.estimates.idx_components] # filter
        totalTime = dfF_good.shape[1]/fr # estimate elapsed time
        xAxis = np.linspace(0,totalTime,dfF_good.shape[1]) # make x-axis

        #plt.subplot(100,2,2)
        #plt.plot(xAxis,dfF_good[0,:],c=colors[0],linewidth=1)
        #plt.subplot(100,2,4)
        #plt.plot(xAxis,dfF_good[1,:],c=colors[1],linewidth=1)

        for i in range(dfF_good.shape[0]):
            if i == 0:
                counter = 2
            ax = plt.subplot(dfF_good.shape[0],2,counter)
            plt.plot(xAxis,dfF_good[i,:],c=colors[i],linewidth=1)
            plt.title('ROI #'+str(i+1),fontsize=8,color=colors[i])
            ax.set_axis_off()
            #ax.set_ylabel('ROI #'+str(i),color=colors[i])
            #plt.Axes(frameon=False)
            counter = counter+2

    # some functions to help us merge and reject accepted components
    def merge_components(cnm,idxMergeGood):
        """
        merge_components: helper function to merge
        idxMergeGood: list of accepted components to merge
        """
        for i in range(len(idxMergeGood)):
            for ii in range(len(idxMergeGood[i])):
                idxMergeGood[i][ii]=idxMergeGood[i][ii]-1

        # subtract 1 because the visual plots below have +1 due to 0-indexing
        cnm.estimates.manual_merge(cnm.estimates.idx_components[idxMergeGood],cnm.params)

        return cnm

    def good_to_bad_components(cnm,idxGood2Bad):
        """
        This function will place good components to the rejected index
        """
        # remove components
        data2rem = cnm.estimates.idx_components[idxGood2Bad]
        cnm.estimates.idx_components = np.delete(cnm.estimates.idx_components,np.array(idxGood2Bad)-1)

        # add to rejected array
        cnm.estimates.idx_components_bad = np.sort(np.append(cnm.estimates.idx_components_bad,idxGood2Bad))

        return cnm
    
class cluster_helper:

    def __init__(self):
        """
        """

    def start_cluster():
        print("starting cluster")
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)
        return c, dview, n_processes
        
    def refresh_cluster(dview):
        print("refreshing cluster")
        cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)        
        return c, dview, n_processes

# function to establish a unique identifier for saving
def saving_identifier():
    """
        This function provides the output "time_id" as a unique identifier to save your data
    """

    now = str(datetime.now())
    now = now.split(' ')
    now = now[0]+'-'+now[1]
    now = now.split(':')
    now = now[0]+now[1]+now[2]
    now = now.split('.')
    time_id = now[0]
    return time_id