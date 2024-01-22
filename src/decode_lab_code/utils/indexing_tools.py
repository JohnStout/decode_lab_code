# indexing tools
import numpy as np
import pandas as pd

def dsearchn(x, v):
    z=np.atleast_2d(x)-np.atleast_2d(v).T
    return np.where(np.abs(z).T==np.abs(z).min(axis=1))[0]
