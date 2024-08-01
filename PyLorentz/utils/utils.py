from typing import Optional, Tuple, Union

import numpy as np
from scipy.signal.windows import tukey


def dist4(dim, norm=False) -> np.ndarray:
    """
    4-fold symmetric distance map (center is 0) even at small radii
    """
    if dim % 2 == 0:
        d2 = dim // 2
        a = np.arange(d2)
        b = np.arange(d2)
        if norm:
            a = a / (2 * d2)
            b = b / (2 * d2)
        x, y = np.meshgrid(a, b)
        quarter = np.sqrt(x**2 + y**2)
        dist = np.zeros((dim, dim))
        dist[d2:, d2:] = quarter
        dist[d2:, :d2] = np.fliplr(quarter)
        dist[:d2, d2:] = np.flipud(quarter)
        dist[:d2, :d2] = np.flipud(np.fliplr(quarter))
    else:
        d2 = dim // 2 + 1
        a = np.arange(d2)
        b = np.arange(d2)
        if norm:
            a = a / (2 * d2)
            b = b / (2 * d2)
        x, y = np.meshgrid(a, b)
        quarter = np.sqrt(x**2 + y**2)
        dist = np.zeros((dim, dim))
        dist[d2 - 1 :, d2 - 1 :] = quarter
        dist[d2 - 1 :, :d2] = np.fliplr(quarter)
        dist[:d2, d2 - 1 :] = np.flipud(quarter)
        dist[:d2, :d2] = np.flipud(np.fliplr(quarter))
    return dist


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
