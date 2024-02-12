# indexing tools
import numpy as np
import pandas as pd

def dsearchn(x, v):
    '''
    Find value v in array x!

    Args:
        x: array of data points (np array or list)
        v: value to discover in x
    
    Returns:
        idx_where: the index where value v is found in array x!

    '''

    z=np.atleast_2d(x)-np.atleast_2d(v).T
    idx_where = np.where(np.abs(z).T==np.abs(z).min(axis=1))[0][0]
    return idx_where

def sort_data(y):
    '''
    Sort datapoints in y and return the index of sorted data

    Args:
        y: np.array or list
    
    Returns:
        sorted_y: sorted input
        sorted_idx: index of the sorted data in sorted_y that point to the original values in y

    Try:
        x = np.array([3, 1, 4, 6, 5])
        sorted_x, sorted_idx = sort_data(x)
        print(sorted_x) # sorted data
        print(sorted_idx) # pointer
        print(x[sorted_idx]) # validation
    '''

    sorted_y = np.array(sorted(y))
    sorted_idx=[]
    for i in range(len(sorted_y)):
        #print(sorted_r[i])
        sorted_idx.append(dsearchn(y,sorted_y[i]))
    
    return sorted_y, sorted_idx

def sortx2y(x,y):
    '''
    Sort datapoints in x by the datapoints in y, assuming the same values in x are found in y.
    This function is really useful if you need to sort y, then identify the values of y in x.
    
    I wrote this because I had correlation values that represent photobleaching and I wanted
    to identify candidate sessions with best photobleaching (closest to 0 correlation coefficients, which
    happened to be the max as most had a significant negative correlation). So I sorted the photobleaching
    values of y (numpy array; sorted least to greatest numerically), then identified the sessions (list; x)
    corresponding to the sorted correlation values in y

    ALTERNATIVELY, this function is useful because if you want to sort y, this function returns the index corresponding
    to the original sorted values! For example, if y[-1] was the lowest value numerically, idx[0] would point to len(y)

    Args:
        x: list or numpy array to sort
        y: numpy array to sort by (sort x according to y)

    Returns:
        sorted_x: x values sorted to y
        idx: index of sorted values in y that correspond to the original y coordinates
    
    '''

    # sort data
    sorted_y, idx = sort_data(y)
    
    sorted_x = list()
    if type(x) is list:
        # if list, list comprehension
        sorted_x = [x[i] for i in idx]
    else:
        sorted_x = x[idx]

    return sorted_x, idx

