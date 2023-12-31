# unpack nwb file movies into .tif files
from pynwb import NWBHDF5IO
from tifffile import imsave
import os
import caiman as cm

# filepath
nwbpath = r'/Users/js0403/miniscope/122A_session2_nwbfile.nwb'

def nwb_to_tif(nwbpath: str):
    '''
    Args:
        >>> nwbpath: directory to your nwb file
                >>> e.g. r'/Users/.../filename.nwb'
    
    This file saves out the individual videos in the NWB file as separate .tif files

    - John Stout
    '''

    # directory definitions
    newpath = nwbpath.split('.nwb')[0]
    rootpath = os.path.split(nwbpath)[0]

    # get directory contents
    dir_contents = sorted(os.listdir(rootpath))

    # make new path to save data
    try:
        print("New folder created: ",newpath)
        os.makedirs(newpath)
    except:
        print("Failed to create folder ",newpath, " - this folder may already exist")

    with NWBHDF5IO(nwbpath, "r+") as io:

        # read file
        read_nwbfile = io.read()

        # get the movie names
        movie_names = list(read_nwbfile.acquisition.keys())

        # load movies
        for i in movie_names:

            # load data
            data = read_nwbfile.acquisition[i].data[:]

            # save as .tif
            save_name = ('.').join([i,'tif'])
            print(save_name, "saved to", newpath)
            imsave(os.path.join(newpath,save_name), data)

def tif_to_memmap(movie_path: str, nwbpath: str):
    '''
    This function takes a bunch of .tif files, writes them to memory mapped files, then combines the results

    
    '''
    
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

    #%% clean up folder
    [os.remove(i) for i in dir_mmap]










