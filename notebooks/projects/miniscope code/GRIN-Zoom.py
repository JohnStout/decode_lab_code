# GRIN-Zoom

#%%
import tifffile
import matplotlib.pyplot as plt

#%% 

# define movie paths
fname = '/Users/js0403/miniscope/122A_session2_nwbfile/122A_session2.tif'

# read movie
movie_tiff = tifffile.imread(fname)

#%% 
# CHANGE ME
idx_row = [150,425] # change me
idx_col = [200,500] # change me

# plot results
plt.subplot(1,2,1)
plt.imshow(movie_tiff[1000,:,:])
plt.title("OG movie pixels")

plt.subplot(1,2,2)
plt.imshow(movie_tiff[1000,idx_row[0]:idx_row[1], idx_col[0]:idx_col[1]])
plt.title("Updated movie pixels")

#%% SAVE DATA
new_movie = movie_tiff[:,idx_row[0]:idx_row[1], idx_col[0]:idx_col[1]]
filename = fname.split('.')
print("Writing file: ",filename)
tifffile.imwrite(fname.split('.')[0]+'_mod.tif', new_movie, photometric='rgb')