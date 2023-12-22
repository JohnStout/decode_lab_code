# calcium_imaging_utils
# group of functions to help calcium imaging analysis

import numpy as np
import tifffile as tiff
import glob
import tarfile
import matplotlib.pyplot as plt

import caiman as cm
from caiman.utils.utils import download_demo

import bokeh.plotting as bpl
import holoviews as hv
bpl.output_notebook()
hv.notebook_extension('bokeh')

# -- modifiers -- #
def spatial_downsample(movieFrames, downsample_factor: int):
    """
    Spatially downsample your data (downsample pixels) by a chosen factor
    
    --- INPUTS ---
    data: 3D array for downsampling    
    downsample_factor: factor to spatially downsample your dataset

    """

    # when slicing, start:stop:step
    frameShape = movieFrames.shape # frameshape

    # downsample
    data_spatial_downsample = movieFrames[:,::downsample_factor,::downsample_factor]

    return data_spatial_downsample

def temporal_downsample(movieFrames, downsample_factor: int, frate: float):

    """
    Spatially downsample your data (downsample pixels) by a chosen factor
    
    --- INPUTS ---
    data: 3D array for downsampling
    downsample_factor: scale for downsampling
    """

    # when slicing, start:stop:step
    frameShape = movieFrames.shape # frameshape

    # downsample
    data_temporal_downsampled = movieFrames[::downsample_factor,:,:]
    frate_new = frate/downsample_factor

    return data_temporal_downsampled

def split_4D_movie(movieFrames = None, fname = None, frame_rate = None, structural_index = None, functional_index = None):

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

    if movieFrames is None:
        print("Loading 4D array")
        movieFrames = cm.load_movie_chain(fname,fr=frame_rate,is3D=True)   

    file_root = fname[0].split('.')[0]
    fname_funct = file_root+'_functional.tif'
    fname_struct = file_root+'_structural.tif'
    print("Saving functional output as",fname_funct)
    print("Saving structural output as",fname_struct)

    # split data
    structMovie = movieFrames[:,structural_index,:,:]
    functMovie = movieFrames[:,functional_index,:,:]        
    tiff.imsave(fname_struct,structMovie)  
    tiff.imsave(fname_funct,functMovie) 

    return [fname_funct], [fname_struct], structMovie, functMovie 


# -- loading function -- #
def stacktiff(dir: str, dir_save = None, downsample_factor = None):
    """
    This function takes a folder with a bunch of .tif images and stacks them

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
    im = tiff.imread(pathnames[0])
    image_shape = im.shape

    images = []
    counter = 0
    for iname in pathnames:
        im = tiff.imread(iname)
        if downsample_factor is not None:
            im = im[0::downsample_factor,0::downsample_factor]
        images.append(im)
        counter = counter+1
        print("Completed with",(counter/len(pathnames)*100),'%')
        del im
    images = np.asarray(images) # convert to numpy array
    print("saving to ",dir_save)
    tiff.imwrite(dir_save+'/tiff_stack.tif',images) # save as tiff


# -- visualization functions -- #
def nb_inspect_correlation_pnr(corr, pnr, cmap='jet', num_bins=100):
    """
    inspect correlation and pnr images to infer the min_corr, min_pnr for cnmfe

    Args:
        corr: ndarray
            correlation image created with caiman.summary_images.correlation_pnr

        pnr: ndarray
            peak-to-noise image created with caiman.summary_images.correlation_pnr

        cmap: string
            colormap used for plotting corr and pnr images
            For valid colormaps see https://holoviews.org/user_guide/Colormaps.html

        num_bins: int
            number of bins to use for plotting histogram of corr/pnr values

    Returns:
        Holoviews plot layout (typically just plots in notebook)
    """
    import functools as fct
    hv_corr = hv.Image(corr,
                       vdims='corr',
                       label='correlation').opts(cmap=cmap)
    hv_pnr = hv.Image(pnr,
                      vdims='pnr',
                      label='pnr').opts(cmap=cmap)

    def hist(im, rx, ry, num_bins=num_bins):
        obj = im.select(x=rx, y=ry) if rx and ry else im
        return hv.operation.histogram(obj, num_bins=num_bins)

    str_corr = (hv.streams.RangeXY(source=hv_corr).rename(x_range='rx', y_range='ry'))
    str_pnr = (hv.streams.RangeXY(source=hv_pnr).rename(x_range='rx', y_range='ry'))

    hist_corr = hv.DynamicMap(
        fct.partial(hist, im=hv_corr), streams=[str_corr])

    hist_pnr = hv.DynamicMap(
        fct.partial(hist, im=hv_pnr), streams=[str_pnr])

    hv_layout = (hv_corr << hist_corr) + (hv_pnr << hist_pnr)

    return hv_layout

def get_rectangle_coords(im_dims,
                         stride,
                         overlap):
    """
    Extract rectangle (patch) coordinates: a helper function used by view_quilt().

    Given dimensions of summary image (rows x colums), stride between patches, and overlap
    between patches, returns row coordinates of the patches in patch_rows, and column
    coordinates patches in patch_cols. This is meant to be used by plot_patches().

    Args:
        im_dims: array-like
            dimension of image (num_rows, num_cols)
        stride: int
            stride between patches in pixels
        overlap: int
            overlap between patches in pixels

    Returns:
        patch_rows: ndarray
            num_patch_rows x 2 array, where row i contains onset and offset row pixels for patch row i
        patch_cols: ndarray
            num_patch_cols x 2 array, where row j contains onset and offset column pixels for patch column j

    Note:
        Currently assumes square patches so takes in a single number for stride/overlap.
    """
    patch_width = overlap + stride

    patch_onset_rows = np.array(list(range(0, im_dims[0] - patch_width, stride)) + [im_dims[0] - patch_width])
    patch_offset_rows = patch_onset_rows + patch_width
    patch_offset_rows[patch_offset_rows > im_dims[0]-1] = im_dims[0]-1
    patch_rows = np.column_stack((patch_onset_rows, patch_offset_rows))

    patch_onset_cols = np.array(list(range(0, im_dims[1] - patch_width, stride)) + [im_dims[1] - patch_width])
    patch_offset_cols = patch_onset_cols + patch_width
    patch_offset_cols[patch_offset_cols > im_dims[1]-1] = im_dims[1]-1
    patch_cols = np.column_stack((patch_onset_cols, patch_offset_cols))

    return patch_rows, patch_cols


def rect_draw(row_minmax,
              col_minmax,
              color='white',
              alpha=0.3,
              ax=None):
    """
    Draw a single transluscent rectangle on given axes object.

    Args:
        row_minmax: array-like
            [row_min, row_max] -- 2-elt int bounds for rect rows
        col_minmax: array-like
            [col_min, col_max] -- int bounds for rect cols
        color : string
            rectangle color, default 'yellow'
        alpha : float
            rectangle alpha (0. to 1., where 1 is opaque), default 0.3
        ax : pyplot.Axes object
            axes object upon which rectangle will be drawn, default None

    Returns:
        ax (Axes object)
        rect (Rectangle object)
    """
    from matplotlib.patches import Rectangle

    if ax is None:
        ax = plt.gca()

    box_origin = (col_minmax[0], row_minmax[0])
    box_height = row_minmax[1] - row_minmax[0]
    box_width = col_minmax[1] - col_minmax[0]

    rect = Rectangle(box_origin,
                     width=box_width,
                     height=box_height,
                     color=color,
                     alpha=alpha)
    ax.add_patch(rect)

    return ax, rect


def view_quilt(template_image,
               stride,
               overlap,
               color='white',
               alpha=0.2,
               vmin=None,
               vmax=None,
               figsize=(6.,6.),
               ax=None):
    """
    Plot patches on template image given stride and overlap parameters on template image.
    This can be useful for checking motion correction and cnmf spatial parameters.
    It ends up looking like a quilt pattern.

    Args:
        template_image: ndarray
            row x col summary image upon which to draw patches (e.g., correlation image)
        stride (int) stride between patches in pixels
        overlap (int) overlap between patches in pixels
        color: matplotlib color
            Any acceptable matplotlib color (r,g,b), string, etc., default 'white'
        alpha (float) : patch transparency (0. to 1.: higher is more opaque), default 0.2
        vmin (float) : vmin for plotting underlying template image, default None
        vmax (float) : vmax for plotting underlying template image, default None
        figsize (tuple) : fig size in inches (width, height), default (6.,6.)
        ax (pyplot.axes): axes object in case you want to add quilt to existing axes

    Returns:
        ax: pyplot.Axes object

    Example:
        # Uses cnm object (cnm) and correlation image (corr_image) as template:
        patch_width = 2*cnm.params.patch['rf'] + 1
        patch_overlap = cnm.params.patch['stride'] + 1
        patch_stride = patch_width - patch_overlap
        ax = view_quilt(corr_image, patch_stride, patch_overlap, vmin=0.0, vmax=0.6);
    """
    im_dims = template_image.shape
    patch_rows, patch_cols = get_rectangle_coords(im_dims, stride, overlap)

    if ax is None:
      f, ax = plt.subplots(figsize=figsize)

    ax.imshow(template_image, cmap='gray', vmin=vmin, vmax=vmax)
    for patch_row in patch_rows:
        for patch_col in patch_cols:
            ax, _ = rect_draw(patch_row,
                              patch_col,
                              color=color,
                              alpha=alpha,
                              ax=ax)

    return ax

# a plotter function
def plot_components(img, cnm, colors: list, colorMap='viridis', clim = []):

    """
    Args:
        img: image to plot results over
        cnm: cnm object 
        colors: list of color indicators
        colorMap: color for imshow map
        clim: colorbar range of heat map
    """

    # extract components
    rois = cnm.estimates.coordinates # get rois
    idx = cnm.estimates.idx_components # get accepted components
    good_rois = [rois[i] for i in range(len(rois)) if i in idx]

    # plot components
    plt.subplot(1,2,1)
    plt.imshow(img,colorMap)
    if len(clim)!=0:
        plt.clim(clim)
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



# -- starting cluster -- #
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



# -- Manual curation -- #

# some functions to help us merge and reject accepted components
def merge_components(cnm,idxMergeGood):
    """
    merge_components: helper function to merge
    idxMergeGood: list of accepted components to merge
    """
    #for i in range(len(idxMergeGood)):
        #for ii in range(len(idxMergeGood[i])):
            #idxMergeGood[i][ii]=idxMergeGood[i][ii]-1
    
    # need to make sure that we're indexing from the raw components, not the good ones

    # subtract 1 because the visual plots below have +1 due to 0-indexing
    #cnm.estimates.manual_merge(cnm.estimates.idx_components[idxMergeGood],cnm.params)
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

def inspect_corr_pnr(correlation_image_pnr, pnr_image, cbar_limits: list = []):
    import pylab as pl

    """
    inspect correlation and pnr images to infer the min_corr, min_pnr

    Args:
        correlation_image_pnr: ndarray
            correlation image created with caiman.summary_images.correlation_pnr
    
        pnr_image: ndarray
            peak-to-noise image created with caiman.summary_images.correlation_pnr

        cbar_limits: nested list containing colorbar scale

    Output:
        min_corr: Minimum correlation from the correlation_image_pnr
        min_pnr: minimum peak to noise ratio returned from the pnr_image

        * these outputs will return min values of the raw inputs, OR the cbar_limits you provide
    """

    fig = pl.figure(figsize=(10, 4))
    pl.axes([0.05, 0.2, 0.4, 0.7])
    im_cn = pl.imshow(correlation_image_pnr, cmap='jet')
    pl.title('correlation image')
    pl.colorbar()
    if len(cbar_limits)!=0:
        pl.clim(cbar_limits[0])
    else:
        pl.clim()
    
    pl.axes([0.5, 0.2, 0.4, 0.7])
    im_pnr = pl.imshow(pnr_image, cmap='jet')
    pl.title('PNR')
    pl.colorbar()
    if len(cbar_limits)!=0:
        pl.clim(cbar_limits[1])
    else:
        pl.clim()

    # assign min_corr and min_pnr based on the image you create
    if len(cbar_limits)==0:
        min_corr = round(np.min(correlation_image_pnr),1)
        min_pnr  = round(np.min(pnr_image),1)
    else:
        min_corr = cbar_limits[0][0]
        min_pnr = cbar_limits[1][0]

    print("minimum correlation: ",min_corr)
    print("minimum peak-to-noise ratio: ",min_pnr)

    return min_corr, min_pnr
