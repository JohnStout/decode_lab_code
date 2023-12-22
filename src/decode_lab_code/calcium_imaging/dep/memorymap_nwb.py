#%% This code memory maps your file
import cv2
import glob
from IPython import get_ipython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil

import caiman as cm
from pynwb import NWBHDF5IO

from decode_lab_code.calcium_imaging.dep.nwb_utils import nwb_to_tif

#%%

def tif_to_memmap(movie_path: str, nwb_path: str):
    
    # define movie paths
    movie_path = '/Users/js0403/miniscope/122A_session2_nwbfile'
    dir_contents = sorted(os.listdir(movie_path))
    fnames = [os.path.join(movie_path,i) for i in dir_contents if '.tif' in i or '.avi' in i]

    nwbpath = '/Users/js0403/miniscope/122A_session2_nwbfile.nwb'
    with NWBHDF5IO(nwbpath, "r") as io:
        # read file and get the frame rate
        read_nwbfile = io.read()
        fr = read_nwbfile.imaging_planes['ImagingPlane'].imaging_rate

    #%%

    # write to .tif
    nwbpath = r'/Users/js0403/miniscope/122A_session2_nwbfile.nwb'
    nwb_to_tif(nwbpath=nwbpath)

    #%% 
        
    print(f"You have {psutil.cpu_count()} CPUs available in your current environment")
    num_processors_to_use = psutil.cpu_count()-1
    print(f"Using",num_processors_to_use, "processors")

    #%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
    if 'cluster' in locals():  # 'locals' contains list of current local variables
        print('Closing previous cluster')
        cm.stop_server(dview=cluster)
    print("Setting up new cluster")
    dview, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', 
                                                    n_processes=num_processors_to_use, 
                                                    ignore_preexisting=False)
    print(f"Successfully set up cluster with {n_processes} processes")

    #%% memory map files
    print("Ignore file exception errors with jupyter notebook")
    cm.save_memmap_each(fnames, dview = dview, base_name = movie_path.split('/')[-1]+'_', order='C', border_to_0=0)
    cm.stop_server(dview=cluster)

    #%% join the memory mapped files

    dir_contents_new = sorted(os.listdir(movie_path))
    mmap_names = [i for i in dir_contents_new if '.mmap' in i]
    dir_mmap = [os.path.join(movie_path,i) for i in mmap_names]

    cm.save_memmap_join(dir_mmap, base_name='joined_',dview=dview)