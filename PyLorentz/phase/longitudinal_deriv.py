"""Generate longitudinal derivatives through stack of intensity values.

This file contains routines for generating the longitudinal derivative through
a stack of images by fitting a quadratic polynomial to the intensity values for
each (y,x) pixel.

Author: Arthur McCray, ANL, May 2020.
"""

import numpy as np
import time


def polyfit_deriv(stack, defvals, v=1):
    """
    Calculates a longitudinal derivative of intensity values taken at different
    defocus values. Expects the first image to be most underfocused, and last to
    be most overfocused.

    Prints progress every 5 seconds, though for reasonable sized datasets
    (7 x 4096 x 4096) it only takes ~ 7 seconds.

    Args:
        stack (3D array): NumPy array (L x M x N). There are L individual
            (M x N) images.
        defvals (1D array): Array of length L. The defocus values of the images.
        v (int): Verbosity. Set v=0 to suppress all print statements.

    Returns:
        ``ndarray``: Numpy array (M x N) of derivative values.
    """
    vprint = print if v >= 1 else lambda *a, **k: None
    stack = np.array(stack)
    dim_y, dim_x = np.shape(stack[0])
    derivatives = np.zeros((dim_y, dim_x))
    starttime = time.time()
    vprint("00.00%")
    for i in range(dim_y):
        if time.time() - starttime >= 5:
            vprint("{:.2f}%".format(i / dim_y * 100))
            starttime = time.time()

        unf_d = np.polyfit(defvals, stack[:, i], 2)
        derivatives[i] = unf_d[1]

    vprint("100.0%")
    return derivatives
