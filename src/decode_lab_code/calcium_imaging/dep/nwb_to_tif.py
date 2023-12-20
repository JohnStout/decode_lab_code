# unpack nwb file movies into .tif files
from pynwb import NWBHDF5IO
from tifffile import imsave
import os

# create data
#d = np.ndarray(shape=(10,20), dtype=np.float32) # also supports 64bit but ImageJ does not
#d[()] = np.arange(200).reshape(10, 20)

# save 32bit float (== single) tiff
#imsave('test.tif', d) #, description="hohoho")

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






