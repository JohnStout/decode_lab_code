# unpack nwb file movies into .tif files
from pynwb import NWBHDF5IO
from tifffile import imsave, memmap, imread, imwrite, TiffFile
import glob
import numpy as np
import os
import caiman as cm
import psutil
from pynwb import NWBHDF5IO
from scipy.signal import detrend
import matplotlib.pyplot as plt
import imageio

#% helper functions
def mp4_to_tif(movie_path: str):
    '''
    mp4_to_tif:
        This function is used to convert lionheart data to .tif files for analysis

    Args:
        >>> movie_path: string to the filetype of interest ending in '.mp4'
        >>> idx_movie: int representing the index that contains your movie. 
                These files are saved as multi-image stacks. One of them is important.
                To figure this out:
                    vid = imageio.get_reader(movie_path,  'ffmpeg')
                    plt.imshow(vid)

    '''

    # load ffmpeg backend
    vid = imageio.get_reader(movie_path,  'ffmpeg')
    vid.get_meta_data()

    # interface with user
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
    ax[0].imshow(vid.get_data(0)[::2,::2,0])
    ax[0].set_title("Dimension 1")
    ax[1].imshow(vid.get_data(0)[::2,::2,1])
    ax[1].set_title("Dimension 2")
    ax[2].imshow(vid.get_data(0)[::2,::2,2])
    ax[2].set_title("Dimension 3")
    plt.show()

    # require user interface (-1 bc 0 indexing in python)
    idx_movie = int(input("Enter which dimension [1/2/3] has your data:"))-1

    # interact with user to downsample data
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,5))
    ax[0].imshow(vid.get_data(0)[:,:,idx_movie])
    ax[0].set_title("Full Size")   
    ax[1].imshow(vid.get_data(0)[::2,::2,idx_movie])
    ax[1].set_title("Downsampled x2")       
    ax[2].imshow(vid.get_data(0)[::4,::4,idx_movie])
    ax[2].set_title("Downsampled x4")      
    ax[3].imshow(vid.get_data(0)[::6,::6,idx_movie])
    ax[3].set_title("Downsampled x6")   
    plt.show()   

    # choose how to downsample the data
    downsample_factor = int(input("Enter Downsample factor [None, 2, 4, 6, etc...]:"))

    # run a while loop to extract data
    images = []
    next = 0; counter = 0

    # create new name for tif file
    fname = movie_path.split('.mp4')[0]+'.tif'

    # get pixel shape
    if downsample_factor is None:
        pixel_shape = vid.get_meta_data()['size']
    else:
        pixel_shape = vid.get_data(0)[::downsample_factor,::downsample_factor,idx_movie].shape

    # get movie length
    counter = 0; next = 0
    while next == 0:
        try:
            vid.get_data(counter)
            counter += 1
        except:
            next = 1
    
    # create a memory mappable file
    im = memmap(
        fname,
        shape=(counter,pixel_shape[0],pixel_shape[1]),
        dtype=np.uint16,
        append=True
    )

    # now we will append to memory mapped file
    print("Please wait while data is mapped to:",fname)
    next = 0; counter = 0
    while next == 0:
        try:
            if downsample_factor is None:
                im[counter]=vid.get_data(counter)[:,:,idx_movie]
            else:
                im[counter]=vid.get_data(counter)[::downsample_factor,::downsample_factor,idx_movie]
            im.flush()  
            print("Finished with image",counter)                     
            counter += 1
        except:
            next = 1
    
# nwb_to_tif converts your nwb file movies to .tif files
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

    return newpath

# tif_to_memmap converts your tif files to memory mapped files
def tif_to_memmap(movie_path: str):
    '''
    This function takes a bunch of .tif files, writes them to memory mapped files, then combines the results.

    Environment required: CAIMAN
    
    Your movies should be saved as .avi or .tif files in a unique folder

    Args: 
        >>> movie_path: directory to your individual movies
    
    '''
    
    # define movie paths
    dir_contents = sorted(os.listdir(movie_path))
    fnames = [os.path.join(movie_path,i) for i in dir_contents if '.tif' in i or '.avi' in i]

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
    print("Saving out individual memory mapped files...")
    cm.save_memmap_each(fnames, dview = dview, base_name = movie_path.split('/')[-1]+'_', order='C', border_to_0=0)

    #%% join the memory mapped files
    dir_contents_new = sorted(os.listdir(movie_path))
    mmap_names = [i for i in dir_contents_new if '.mmap' in i]
    dir_mmap = [os.path.join(movie_path,i) for i in mmap_names]
    
    print("Joining memory mapped files...")
    cm.save_memmap_join(dir_mmap, base_name = movie_path.split('/')[-1]+'_',dview=dview)

    #%% clean up folder
    print("Removing individual memory mapped files...")
    [os.remove(i) for i in dir_mmap]
    
    # stop running on cluster
    cm.stop_server(dview=cluster)

# lazyTiff reads your tiff file lazily - this is very useful for large files
def lazyTiff(movie_path: str):
    '''
    This function is meant to allow a user to load a large .tif file into memory, by reading the data lazily

    Args:
        movie_path: directory containing your .tif file

    Returns:
        movie: movie from the .tif file

    '''
    movie = memmap(movie_path)

    return movie

# stacktiff allows a user to stack a ton of tiff files into one singular file
def stacktiff(dir: str, dir_save = None, downsample_factor = None):
    """
    This function takes a folder with a bunch of .tif images and stacks them

    TODO: THIS CODE MUST BE MODIFIED TO MEMORY MAP THE FILES! >>> See mp4_to_tif
    Otherwise, this will be a monster memory bogger

    Args:
        dir: directory containing image data to stack
        dir_save: OPTIONAL but recommended. Directory to save stacked data.
        downsample_factor: OPTIONAL.
            downsample_factor = 2 spatially reduces your dataset by a factor of 2
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
    im = imread(pathnames[0])
    image_shape = im.shape

    # TODO: Change this to memmap approach like in the mp4 code
    images = []
    counter = 0
    for iname in pathnames:
        # consider changing to memmap
        im = imread(iname)
        if downsample_factor is not None:
            im = im[0::downsample_factor,0::downsample_factor]
        images.append(im)
        counter = counter+1
        print("Completed with",(counter/len(pathnames)*100),'%')
        del im
    images = np.asarray(images) # convert to numpy array
    print("saving to ",dir_save)
    imwrite(dir_save+'/tiff_stack.tif',images) # save as tiff

# downsample_movie provides a class to downsample your dataset
class modify_movie():

    def __init__(self, movie_path):
        '''
        This class allows the user to downsample their data

        Args:
            movie_path: path to movie
        '''

        # load file lazily
        self.movie_path = movie_path
        self.movie = memmap(movie_path)

        # check that the array is 3D
        if len(self.movie.shape) < 3:
            TypeError("Data are less than 3D - check that you have input a movie file")
        elif len(self.movie.shape) > 3:
            print(">3D movie detected - this happens if you have two or more channels being recorded at once")

    def spatial_downsample(self, downsample_factor: int, save_movie: bool = False, save_path = None):
        """
        Spatially downsample your data (downsample pixels) by a chosen factor
        
        --- INPUTS ---
        data: 3D array for downsampling    
        downsample_factor: factor to spatially downsample your dataset

        """

        # when slicing, start:stop:step
        frameShape = self.movie.shape # frameshape

        # downsample
        self.movie = self.movie[:,::downsample_factor,::downsample_factor]
        print("New movie shape:", self.movie.shape)

        # save movie file
        if save_movie is True:
            if '.tif' in self.movie_path:
                root_path = self.movie_path.split('.tif')[0]
            elif '.avi' in self.movie_path:
                root_path = self.movie_path.split('.avi')[0]
            saved_path = os.path.join(root_path+"_spatDownSampledx"+str(downsample_factor)+".tif")
            imwrite(saved_path, self.movie)
            print("Saved data to:",saved_path)

    def photobleach_correction(self, save_movie: bool = False, parameter_search: bool = False, linear_detrend: bool = True):

        '''
        This code corrects photobleaching decay by detrending each pixel using a first order polynomial

        Args:
            save_movie: save file
            save_path: directory to save data
            parameter_search: Search for a model fit that has the lowest SSE and fit that to the data to detrend
        '''

        # detrend
        y_detrend = np.zeros(shape=self.movie.shape)
        for rowi in range(self.movie.shape[1]):
            for coli in range(self.movie.shape[2]):

                # get all data in the temporal domain, pixel wise
                y = self.movie[:,rowi,coli]

                # detrend data using first order polynomial
                if parameter_search is False:

                    #TODO: FIX
                    # simple linear detrend
                    if linear_detrend is False:
                        y_detrend[:,rowi,coli] = detrend(y, type='linear')
                    elif linear_detrend is True:
                        pass

                elif parameter_search is True:

                    # define an x-variable
                    x = np.linspace(0,len(y),len(y))

                    # sum of squared errors
                    sse = np.zeros(shape=(5,))
                    for i in range(5):
                        model = np.polyfit(x, y, i+1)
                        predicted = np.polyval(model, x)
                        sse[i] = sum((y-predicted)**2)

                    # using sse, fit a model
                    idx_min = np.argmin(sse)+1 # minimum sse
                    #print("SSE:",idx_min)
                    model = np.polyfit(x, y, idx_min)
                    predicted = np.polyval(model, x)

                    # detrend
                    y_detrend[:,rowi,coli] = y-predicted

            print(((rowi+1)/self.movie.shape[1])*100,"% complete")


        # save movie file
        if save_movie is True:
            if '.tif' in self.movie_path:
                root_path = self.movie_path.split('.tif')[0]
            elif '.avi' in self.movie_path:
                root_path = self.movie_path.split('.avi')[0]
            saved_path = os.path.join(root_path+"_detrended.tif")
            imwrite(saved_path, y_detrend) 
            print("Saved data to:",saved_path)

        # save out file path
        self.path_detrended = saved_path

    def split_4D_movie(self, structural_index = None, functional_index = None):

        """
        This function will take a 4D movie, split it into its components based on your index, 
        then save the output based on the save name index.

        This function was specifically designed when one records a structural channel with a functional channel. 
        For example, you might record astrocytes with an RFP, but neurons or all cells with a GCaMP.

        Args:
            self: 
            structural_index: index for structural imaging
            functional_index: index for functional imaging

        output:
            self.fname_struct: structural fname
            self.fname_funct: functional fname  
        
        """

        # rename
        if '.tif' in self.movie_path:
            file_root = self.movie_path.split('.tif')[0]
        elif '.avi' in self.movie_path:
            file_root = self.movie_path.split('.avi')[0]    
        else:
            TypeError("file_root not defined. Convert your video to .tif or .avi. Preferably to .tif")

        if structural_index is None or functional_index is None:
            idx = np.argmin(self.movie.shape)
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (5, 2.5))

            if idx == 1:
                ax[0].imshow(self.movie[0,0,:,:])
                ax[1].imshow(self.movie[0,1,:,:])
                functional_index = int(input("Which subplot is your functional image? [1 or 2]: "))-1
                if functional_index == 0:
                    structural_index = 1
                elif functional_index == 1:
                    structural_index = 0

        # create new name        
        fname_funct = file_root+'_functional.tif'
        fname_struct = file_root+'_structural.tif'
        print("Saving functional output as",fname_funct)
        print("Saving structural output as",fname_struct)

        # split data
        structMovie = self.movie[:,structural_index,:,:]
        functMovie = self.movie[:,functional_index,:,:]        
        imsave(fname_struct,structMovie)  
        imsave(fname_funct,functMovie) 

        return [fname_funct], [fname_struct], structMovie, functMovie 

 