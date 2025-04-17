from __future__ import annotations
import numpy as np
from scipy.signal.windows import tukey


def dist4(dim, shifted=True) -> np.ndarray:
    """
    4-fold symmetric distance map (center is 0) even at small radii
    centered in the middle (i.e. fft shifted) by default
    """
    d = np.fft.fftfreq(dim, 1 / dim)
    d = np.abs(d + 0.5) - 0.5 if dim % 2 == 0 else d
    if shifted:
        d = np.fft.fftshift(d)
    rr = np.sqrt(d[None,] ** 2 + d[..., None] ** 2)
    return rr


def circ4(dim: int, rad: float):
    """4-fold symmetric circle even at small dimensions"""
    return (dist4(dim) < rad).astype("int")


def norm_image(image: np.ndarray | list) -> np.ndarray:
    """Normalize image intensities to between 0 and 1. Returns copy"""
    im = np.array(image)
    if im.max() == im.min():
        im = im - np.max(im)
    else:
        im = im - np.min(im)
        im = im / np.max(im)
    return im


def tukey2d(shape, alpha=0.5, sym=True, shrink=0, shifted=False):
    """
    Create a 2D Tukey window.

    Args:
        shape: Shape of the window (height, width).
        alpha: Shape parameter of the Tukey window.
        sym: If True, makes the window symmetric.
        shrink: N pix extra to shrink the window on each side (default 0)
        shifted: If True, will corner-center the window with fftshift

    Returns:
        2D Tukey window.


    shrink = N pix to extra shrink the window on each side
    """
    dimy, dimx = shape
    assert shrink >= 0
    dimy -= 2*shrink
    dimx -= 2*shrink
    ty = tukey(dimy, alpha=alpha, sym=sym)
    filt_y = np.tile(ty.reshape(dimy, 1), (1, dimx))
    tx = tukey(dimx, alpha=alpha, sym=sym)
    filt_x = np.tile(tx, (dimy, 1))
    output = filt_x * filt_y
    if shrink > 0:
        output = np.pad(output, shrink, mode='constant')
    if shifted:
        output = np.fft.fftshift(output)
    return output