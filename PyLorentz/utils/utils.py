from typing import Optional, Tuple, Union

import numpy as np
from scipy.signal.windows import tukey



def dist4(dim, shifted=True) -> np.ndarray:
    """
    4-fold symmetric distance map (center is 0) even at small radii
    centered in the middle (i.e. fft shifted) by default
    """
    d = np.fft.fftfreq(dim, 1/dim)
    d = np.abs(d + 0.5)-0.5 if dim % 2 == 0 else d
    if shifted:
        d = np.fft.fftshift(d)
    rr = np.sqrt(d[None,]**2 + d[...,None]**2)
    return rr

def circ4(dim: int, rad: float):
    """4-fold symmetric circle even at small dimensions"""
    return (dist4(dim) < rad).astype("int")


def norm_image(image: Union[np.ndarray, list]):
    """Normalize image intensities to between 0 and 1. Returns copy"""
    image = np.array(image)
    if image.max() == image.min():
        image = image - np.max(image)
    else:
        image = image - np.min(image)
        image = image / np.max(image)
    return image


def Tukey2D(shape: Tuple[int, int], alpha: float = 0.5, sym: bool = True) -> np.ndarray:
    """
    Create a 2D Tukey window.

    Args:
        shape: Shape of the window (height, width).
        alpha: Shape parameter of the Tukey window.
        sym: If True, makes the window symmetric.

    Returns:
        2D Tukey window.
    """
    dimy, dimx = shape
    ty = tukey(dimy, alpha=alpha, sym=sym)
    filt_y = np.tile(ty.reshape(dimy, 1), (1, dimx))
    tx = tukey(dimx, alpha=alpha, sym=sym)
    filt_x = np.tile(tx, (dimy, 1))
    output = filt_x * filt_y
    return output
