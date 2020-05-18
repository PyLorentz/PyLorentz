"""Generate longitudinal derivatives through stack of intensity values. 

___This has bugs!___
Using a longitudinal derivative instead of 3-point will, for some experimental
datasets, give a reconstructed phase shift that is an order of magnitude 
different. This does not seem to be the case for simulated datasets. 
Additionally, using this derivative can affect ideal Thikonov filter values. 

This file contains routines for generating the longitudinal derivative through 
a stack of images by fitting a quadratic polynomial to the intensity values for 
each (y,x) pixel. 

Previously it was found that for large datasets this can be slow, and as such
a multiprocessing approach is implemented. For almost all cases however this is 
not necessary and for small images and stacks it will be slower due to 
additional setup time.

AUTHOR:
Arthur McCray, ANL, May 2020.
--------------------------------------------------------------------------------
"""

import numpy as np
import multiprocessing
from numpy.polynomial import polynomial as P
import time

def polyfit_deriv(stack, defvals):
    """
    Calculates a longitudinal derivative of intensity values taken at different
    defocus values. Expects the firt image to be most underfocused, and last to
    be most overfocused. 
    Gives a completeion percentage every 5 seconds, though for reasonable sized 
    datasets (7 x 4096 x 4096) it only takes ~ 7 seconds.  

    Args:
        stack: 3D array (L x M x N). There are L individual (M x N) iamges.
        defvals: 1D array length L. The defocus value of the images.

    Returns: 
        Numpy array (M x N) of derivative values.  
    """
    stack = np.array(stack)
    dim_y, dim_x = np.shape(stack[0])
    derivatives = np.zeros(np.shape(stack[0]))
    starttime = time.time()
    print('00.00%')
    for i in range(dim_y):
        if time.time() - starttime >= 5:
            print('{:.2f}%'.format(i/dim_y*100))
            starttime = time.time()
        
        unf_d = P.polyfit(defvals,stack[:,i],2)
        derivatives[i] = unf_d[1]
    print('100.0%')
    return derivatives


def polyfit_deriv_multiprocess(stack, defvals):
    """
    Calculates a longitudinal derivative of intensity values taken at different
    defocus values. Expects the firt image to be most underfocused, and last to
    be most overfocused. 
    Uses python multprocessing to theoretically improve on speed. 

    Args:
        stack: 3D array (L x M x N). There are L individual (M x N) iamges.
        defvals: 1D array length L. The defocus value of the images.

    Returns: 
        Numpy array (M x N) of derivative values.  
    """
    stack = np.array(stack)
    dim_y, dim_x = np.shape(stack[0])
    derivatives = np.zeros(np.shape(stack[0]))

    pool = multiprocessing.Pool()
    results = [pool.apply_async(run_polyfit, args=(i,defvals, stack[:,i],2,)) for i in range(dim_y)]
    for p in results:
        i, val = p.get()
        derivatives[i] = val
    return derivatives


def run_polyfit(i,x, y, deg):
    """ Helper function for multiprocessing. """
    print(np.shape(P.polyfit(x, y, deg)[1]))
    return(i, P.polyfit(x, y, deg)[1])
