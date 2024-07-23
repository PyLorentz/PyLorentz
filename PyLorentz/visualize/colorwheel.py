"""Creates RGB images from vector fields.

This module contains several routines for plotting colormaps from input
data consisting of 2D images of the 2D or 3D vector field. A variety of colormaps are
available using the Colorcet package.

https://colorcet.holoviz.org/user_guide/Continuous.html#cyclic-colormaps
Good Colour Maps: How to Design Them, Peter Kovesi (2015) https://arxiv.org/abs/1509.03700
"""

import colorsys
from typing import Optional, Union

import matplotlib as mpl
import numpy as np
from matplotlib import colors
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt

try:
    import colorcet as cc
except ModuleNotFoundError:
    pass

def roll_cmap(
    cmap: Union[Colormap, str],
    frac: float,
    invert: bool = False,
) -> Colormap:
    """
    Shifts a matplotlib colormap by rolling.

    Args:
        cmap (mpl.colors.Colormap | str): The colormap to be shifted. Can be a colormap name or a Colormap object.
        frac (float): The fraction of the colorbar by which to shift (must be between 0 and 1).
        invert (bool, optional): Whether to invert the colormap. Defaults to False.

    Returns:
        mpl.colors.Colormap: The shifted colormap.
    """
    N = 256
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    n = cmap.name
    x = np.linspace(0, 1, N)
    out = np.roll(x, int(N * frac))
    if invert:
        out = 1 - out
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(f"{n}_s", cmap(out))
    return new_cmap

def shift_cmap_center(
    cmap: Union[Colormap, str],
    vmin: float = -1,
    vmax: float = 1,
    midpointval: float = 0,
    invert: bool = False,
) -> Colormap:
    """
    Shifts a matplotlib colormap such that the center is moved and the scaling is consistent.

    Args:
        cmap (mpl.colors.Colormap | str): The colormap to be shifted. Can be a colormap name or a Colormap object.
        vmin (float, optional): Minimum value for scaling. Defaults to -1.
        vmax (float, optional): Maximum value for scaling. Defaults to 1.
        midpointval (float, optional): Value at the center of the colormap. Defaults to 0.
        invert (bool, optional): Whether to invert the colormap. Defaults to False.

    Returns:
        mpl.colors.Colormap: The shifted colormap.
    """
    assert vmax > vmin

    midpoint_loc = (midpointval - vmin) / (vmax - vmin)
    used_ratio = (vmax - vmin) / (2 * max(abs(vmax - midpointval), abs(vmin - midpointval)))
    N = round(256 / used_ratio)

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)

    x = np.linspace(0, 1, N)
    roll_frac = used_ratio if midpoint_loc < 0.5 else used_ratio * -1

    roll_ind = round(N * roll_frac)
    out = np.roll(x, roll_ind)

    if midpoint_loc < 0.5:
        out = out[:256]
    elif midpoint_loc > 0.5:
        out = out[-256:]

    if invert:
        out = 1 - out
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(f"{cmap.name}_s", cmap(out))
    return new_cmap

def get_cmap(cmap: str | None = None, **kwargs) -> Colormap:
    """
    Take a colormap or string input and return a Colormap object.

    Args:
        cmap (str | None, optional): String corresponding to a colorcet colormap name,
            a mpl.colors.LinearSegmentedColormap object, or a
            mpl.colors.ListedColormap object. Defaults to None -> CET_C7.

    Keyword Args:
        shift (float, optional): The amount to shift the colormap by in radians. Defaults to 0.
        invert (bool, optional): Whether to invert the colormap. Defaults to False.

    Raises:
        TypeError: If the input type is not recognized.

    Returns:
        mpl.colors.Colormap: Matplotlib.colors.Colormap object.
    """
    if cmap is None:
        cmap = "linear"
    elif isinstance(cmap, colors.LinearSegmentedColormap) or isinstance(cmap, colors.ListedColormap):
        return cmap
    elif isinstance(cmap, str):
        cmap = cmap.lower()
    else:
        raise TypeError(f"Unknown input type {type(cmap)}, please input a matplotlib colormap or valid string")

    shift = kwargs.get("shift", 0)
    invert = kwargs.get("invert", False)
    try:
        if cmap in ["linear", "lin", ""]:
            cmap = plt.get_cmap("gray")
        elif cmap in ["diverging", "div"]:
            cmap = plt.get_cmap("coolwarm")
        elif cmap in ["linear_cbl", "cbl", "lin_cbl"]:
            cmap = cc.cm.CET_CBL1
        elif cmap in ["diverging_cbl", "div_cbl"]:
            cmap = cc.cm.CET_CBD1
        elif cmap in ["cet_rainbow", "cet_r1", "r1"]:
            cmap = cc.cm.CET_R1
        elif cmap in ["legacy4fold", "cet_c2", "c2", "cet_2"]:
            cmap = cc.cm.CET_C2
            shift += -np.pi / 2  # matching directions of legacy 4-fold
        elif cmap in ["purehsv", "legacyhsv"]:
            cmap = plt.get_cmap("hsv")
            invert = not invert
            shift += np.pi / 2
        elif cmap in ["cet_c6", "c6", "cet_6", "6fold", "sixfold", "hsv", "cyclic"]:
            cmap = cc.cm.CET_C6
            invert = not invert
            shift += np.pi / 2
        elif cmap in ["cet_c7", "c7", "cet_7", "4fold", "fourfold", "4-fold"]:
            cmap = cc.cm.CET_C7
            invert = not invert
        elif cmap in ["cet_c8", "c8", "cet_8"]:
            cmap = cc.cm.CET_C8
        elif cmap in ["cet_c10", "c10", "cet_10", "isolum", "isoluminant", "iso"]:
            cmap = cc.cm.CET_C10
        elif cmap in ["cet_c11", "c11", "cet_11"]:
            cmap = cc.cm.CET_C11
        elif cmap in plt.colormaps():
            cmap = plt.get_cmap(cmap)
        else:
            print(f"Unknown colormap input '{cmap}'.")
            print("You can also pass a colormap object directly.")
            print("Proceeding with default cc.cm.CET_C7.")
            cmap = cc.cm.CET_C7
    except NameError:
        print("Colorcet not installed, proceeding with hsv from mpl")
        cmap = plt.get_cmap("hsv")
        invert = not invert
        shift -= np.pi / 2
    if shift != 0:  # given as radian convert to [0,1]
        shift = shift % (2 * np.pi) / (2 * np.pi)
    if shift != 0 or invert:
        cmap = roll_cmap(cmap, shift, invert)
    return cmap

def color_im(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: Optional[np.ndarray] = None,
    cmap: Optional[Union[str, Colormap]] = None,
    rad: Optional[int] = None,
    background: str = "black",
    **kwargs,
) -> np.ndarray:
    """
    Make the RGB image from vector maps. Takes 2D array inputs for x, y, (and
    optionally z) vector components.

    Unless otherwise specified, the color intensity corresponds to the magnitude of the
    in-plane vector component normalized to the vector with the largest in-plane
    magnitude. If a z-component is given, it will map from black (negative) to white
    (positive).

    Good colormaps are notoriously difficult to design [1], and cyclic colormaps
    especially so. We recommend and use colormaps provided by the Colorcet package [2],
    with the default being CET_C6, a nice 6-fold improved-hsv colorwheel. Other colormaps we suggest
    include:
    - C7: A nice 4-fold map
    - C10: Isoluminescent 4-fold map

    these can be specified with strings as detailed in get_cmap(). Additionally, any matplotlib or
    colorcet Colormap object can be passed and will be used here.

        [1] Kovesi, Peter. "Good colour maps: How to design them."
        arXiv preprint arXiv:1509.03700 (2015).
        [2] https://colorcet.holoviz.org


    Args:
        vx (np.ndarray): (M x N) array consisting of the x-component of the vector field.
        vy (np.ndarray): (M x N) array consisting of the y-component of the vector field.
        vz (np.ndarray | None, optional): (M x N) array consisting of the z-component of the vector field.
            If vz is given, black corresponds to z<0 and white to z>0. Default is None.
        cmap (str | mpl.colors.Colormap | None, optional): Specification for the colormap to be used.
            This is passed to get_cmap() which returns the colormap object if a string is given.
            Defaults to None -> colorcet.cm.CET_C7.
        rad (int | None, optional): Radius of color-wheel in pixels. Set rad = 0 to remove color-wheel.
            Default is None -> height/16.
        background (str, optional):
            - 'black' -- (default) magnetization magnitude corresponds to value.
            - 'white' -- magnetization magnitude corresponds to saturation.
            - if vz is given, this argument will not do anything. Default "black"

    Keyword Args:
        one_pi (bool): Whether to map the direction or orientation of the vector field.
            one_pi = True will modulo the vectors by pi. Default False.
        shift (float): Rotate the colorwheel and orientation map by the specified amount in radians. Default 0.
        invert (bool): Whether or not to invert the directions of the orientation map. Default False.
        uni_mag (bool): Normally the color intensity (saturation/value) corresponds to the magnitude of the vector,
            scaled relative to the largest vector in the image. If uni_mag = True (specifying uniform_magnitude),
            then all vectors larger with a magnitude larger than uni_mag_cutoff * max_magnitude will be displayed
            while others will map to background colors or z-direction if vz is given. Default False.
        uni_mag_cutoff (float): Value [0,1], specifying the magnitude, as a fraction of the maximum vector length
            in the image, above which a vector will be plotted. Default 0.5.
        HSL (bool): When give a z-component, this function normally maps vector orientation to hue and vector magnitude
            to saturation/value, with black/white corresponding to if the vector points in/out of the page.
            This can cause problems as scaling RGB values can make it appear that there is strong in-plane signal
            where there really isn't. An improvement is to use a HSL color space and map vector orientation to hue,
            z-component to lightness, and in-plane magnitude to saturation. Setting HSL=True will use this type of mapping,
            and set the colormap to a true hsv colormap, as more complex color maps get distorted when changing the
            saturation and lightness independently. Default False.

    Returns:
        np.ndarray: Numpy array (M x N x 3) containing the RGB color image.
    """
    vx, vy = np.squeeze(vx), np.squeeze(vy)
    assert vx.ndim == vy.ndim == 2
    assert np.shape(vy) == np.shape(vx)
    if vz is not None:
        HSL = kwargs.get("HSL", None)
        if HSL is None:
            HSL = kwargs.get("HLS", False)  # i can never remember if it's HSL or HLS
    else:
        HSL = False

    if HSL:
        cmap = "purehsv"
    elif cmap is None:
        cmap = "hsv"
    cmap = get_cmap(cmap, **kwargs)

    if rad is None:
        rad = vx.shape[0] // 16
        rad = max(rad, 16)

    raw_inp_mags = np.sqrt(vx**2 + vy**2)
    if np.min(raw_inp_mags) == np.max(raw_inp_mags):
        mags = np.ones_like(vx)
    else:
        mags = raw_inp_mags - np.min(raw_inp_mags)
        mags = mags / np.max(mags)  # normalize [0,1]

    if kwargs.get("uni_mag", False):
        cutoff = kwargs.get("uni_mag_cutoff", 0.5)
        mags = np.where(mags > cutoff, 1, 0)
    if vz is None:
        bkgs = np.where(mags == 0)
    else:
        bkgs = np.where(np.sqrt(vx**2 + vy**2 + vz**2) == 0)

    if rad > 0:
        pad = min(2 * (rad // 10), 40)  # padding between edge of image and color-wheel
    else:
        pad = 0
        rad = 0

    dimy = np.shape(vy)[0]
    if dimy < 2 * rad:
        rad = dimy // 2
    dimx = np.shape(vy)[1] + 2 * rad + pad
    cimage = np.zeros((dimy, dimx, 3))

    # azimuth maps to hue
    if kwargs.get("one_pi", False):
        azimuth = np.mod((np.arctan2(vx, vy) + np.pi), np.pi) / np.pi
    else:
        azimuth = (np.arctan2(vx, vy) + np.pi) / (2 * np.pi)

    # apply colormap to angle
    imrgb = cmap(azimuth)[..., :3]  # remove alpha channel

    if vz is None:
        if background.lower() == "black":
            for i in range(3):
                imrgb[:, :, i] *= mags
        else:
            for i in range(3):
                imrgb[:, :, i] = 1 - (1 - imrgb[:, :, i]) * mags
    else:  # mz > 1 -> white, mz < 1 -> black
        theta = np.arctan2(vz, raw_inp_mags)
        # from hipl.utils.show import show_im
        # show_im(mags, 'mags')
        # show_im(theta, 'theta')
        if HSL:
            H = colors.rgb_to_hsv(imrgb)[:, :, 0]
            # **3 is due to move more values closer to 0.5, max color intensity
            L = (np.sin(theta) ** 3 + 1) / 2  # theta [-pi,pi] -> [0,1]
            S = mags
            S[bkgs] = 0
            L[bkgs] = 0.5

            for j in range(np.shape(vx)[0]):
                for i in range(np.shape(vx)[1]):
                    imrgb[j, i, :] = colorsys.hls_to_rgb(H[j, i], L[j, i], S[j, i])
        else:
            if kwargs.get("uni_mag", False):
                pos = np.where((np.sin(theta) > 0) & (mags == 0), 0, 1)
                neg = np.where((np.sin(theta) < 0) & (mags == 0), 0, 1)
            else:
                neg = np.where(theta < 0, np.cos(theta), 1)
                pos = np.where(theta > 0, np.cos(theta), 1)
            for i in range(3):
                imrgb[:, :, i] = 1 - (1 - imrgb[:, :, i]) * pos
                imrgb[:, :, i] *= neg
                imrgb[:, :, i][bkgs] = 0.5

    # colorwheel
    if rad <= 0:
        return imrgb
    else:
        cimage[:, : -2 * rad - pad, :] = imrgb
        if vz is None:
            wbkg = "black" if background == "white" else "white"
            wheel = make_colorwheel(
                rad, cmap, background=wbkg, core=background, **kwargs
            )
            if background == "black":  # have white sidebar
                cimage[:, dimx - 2 * rad - pad :] = 1

        else:
            wheel = make_colorwheelz(rad, cmap, **kwargs)
            cimage[:, dimx - 2 * rad - pad :] = 0
            cimage[:, dimx - 2 * rad - pad] = 1

        cimage[
            dimy // 2 - rad : dimy // 2 + rad,
            dimx - 2 * rad - pad // 2 : -pad // 2,
            :,
        ] = wheel
        return cimage

def make_colorwheel(
    rad: int,
    cmap: mpl.colors.Colormap,
    background: str = "black",
    core: str | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Makes an RGB image of a colorwheel for a given colormap.

    Args:
        rad (int): Radius of the colormap in pixels.
        cmap (mpl.colors.Colormap): Matplotlib Colormap object.
        background (str, optional): Background color. Defaults to "black".
        core (str | None, optional): Core color. Defaults to background color.

    Keyword Args:
        one_pi (bool): Whether to map the direction or orientation of the vector field.
            one_pi = True will modulo the vectors by pi. Default False.
        uni_mag (bool): Ring colormap showing orientation only, no magnitude.

    Returns:
        np.ndarray: Numpy array (rad*2 x rad*2 x 3) containing the RGB color image.
    """
    cmap = get_cmap(cmap)
    background = background.lower()
    X, Y = np.mgrid[-rad:rad, -rad:rad]
    if kwargs.get("one_pi", False):
        azimuth = np.mod((np.arctan2(Y, X) + np.pi), np.pi) / np.pi
    else:
        azimuth = (np.arctan2(Y, X) + np.pi) / (2 * np.pi)
    imrgb = cmap(azimuth)[..., :3]
    rr = dist4(rad * 2)
    mask = np.where(rr < rad, 1, 0)
    rr *= mask
    rr /= np.max(rr)
    if kwargs.get("uni_mag", False):
        cutoff = 0.4
        rr = np.where(rr > cutoff, 1, 0)

    if core is None:
        core = background
    if core == "black":
        for i in range(3):
            imrgb[:, :, i] *= rr * mask
    else:
        for i in range(3):
            imrgb[:, :, i] = 1 - (1 - imrgb[:, :, i]) * rr

    if background == "black":
        for i in range(3):
            imrgb[:, :, i] = imrgb[:, :, i] * mask
    else:
        for i in range(3):
            imrgb[:, :, i] = 1 - (1 - imrgb[:, :, i]) * mask

    return imrgb

def make_colorwheelz(
    rad: int,
    cmap: Colormap,
    outside: str = "black",
    **kwargs,
) -> np.ndarray:
    """
    Makes an RGB image of a colorwheel where z-direction corresponds to black/white.

    Args:
        rad (int): Radius of the colormap in pixels.
        cmap (mpl.colors.Colormap): Matplotlib Colormap object.
        outside (str, optional): Whether the outside portion of the colormap will be
            black or white. Inside color will necessarily be the opposite. Defaults to "black".

    Keyword Args:
        one_pi (bool): Whether to map the direction or orientation of the vector field.
            one_pi = True will modulo the vectors by pi. Default False.
        uni_mag (bool): Ring colormap showing orientation only, no magnitude.
        HSL (bool): Use HSL color space for mapping. Defaults to False.

    Returns:
        np.ndarray: Numpy array (rad*2 x rad*2 x 3) containing the RGB color image.
    """
    cmap = get_cmap(cmap)
    HSL = kwargs.get("HSL", None)
    if HSL is None:
        HSL = kwargs.get("HLS", False)
    X, Y = np.mgrid[-rad:rad, -rad:rad]
    if kwargs.get("one_pi", False):
        azimuth = np.mod((np.arctan2(Y, X) + np.pi), np.pi) / np.pi
    else:
        azimuth = (np.arctan2(Y, X) + np.pi) / (2 * np.pi)
    imrgb = cmap(azimuth)[..., :3]
    rr = dist4(rad * 2)
    mask = np.where(rr < rad, 1, 0)
    rr *= mask
    rr /= np.max(rr)

    theta = (rr - 0.5) * np.pi
    if kwargs.get("uni_mag", False):
        cutoff = np.pi * 0.25
        inner = np.where(theta + cutoff < 0, 0, 1)
        outer = np.where(theta - cutoff > 0, 0, 1)
    else:
        if HSL:
            H = colors.rgb_to_hsv(imrgb)[:, :, 0]
            # **3 is due to move more values closer to 0.5, max color intensity
            if outside == "black":
                L = 1 - ((np.sin(theta) ** 3 + 1) / 2)  # theta [-pi,pi] -> [0,1]
            else:
                L = (np.sin(theta) ** 3 + 1) / 2  # theta [-pi,pi] -> [0,1]
            rr2 = np.abs(rr - 0.5) * -1 + 1  # rescale to maximize in middle
            S = rr2**2

            for j in range(2 * rad):
                for i in range(2 * rad):
                    imrgb[j, i, :] = colorsys.hls_to_rgb(H[j, i], L[j, i], S[j, i])
            if outside == "black":
                for i in range(3):
                    imrgb[:, :, i] *= mask
            else:
                for i in range(3):
                    imrgb[:, :, i] += 1 - mask

        else:
            inner = np.where(theta < 0, np.cos(theta), 1)
            outer = np.where(theta > 0, np.cos(theta), 1)
    if not HSL:
        for i in range(3):
            if outside == "black":
                imrgb[:, :, i] = 1 - (1 - imrgb[:, :, i]) * inner
                imrgb[:, :, i] *= mask * outer
            else:
                imrgb[:, :, i] *= inner
                imrgb[:, :, i] = 1 - (1 - imrgb[:, :, i]) * outer
                imrgb[:, :, i] = 1 - (1 - imrgb[:, :, i]) * mask
    return imrgb

def dist4(dim: int, norm: bool = False) -> np.ndarray:
    """
    Radial distance map that is 4-fold symmetric even at small sizes.

    Args:
        dim (int): Desired dimension of output.
        norm (bool, optional): Normalize maximum of output to 1. Defaults to False.

    Returns:
        np.ndarray: 2D (dim, dim) array.
    """
    d2 = dim // 2
    a = np.arange(d2)
    b = np.arange(d2)
    if norm:
        a = a / (2 * d2)
        b = b / (2 * d2)
    x, y = np.meshgrid(a, b)
    quarter = np.sqrt(x**2 + y**2)
    sym_dist = np.zeros((dim, dim))
    sym_dist[d2:, d2:] = quarter
    sym_dist[d2:, :d2] = np.fliplr(quarter)
    sym_dist[:d2, d2:] = np.flipud(quarter)
    sym_dist[:d2, :d2] = np.flipud(np.fliplr(quarter))
    return sym_dist


def _white_to_transparent(image, magz=None):
    rgba_image = np.ones((image.shape[0], image.shape[1], 4), dtype=float)
    rgba_image[:, :, :3] = image[:, :, :3]  # Convert RGB values to 0-255 range
    if magz is not None:
        alpha = np.where(magz > 0, 1 - magz**10, 1)
    else:
        print(rgba_image[:,:,:2].sum(axis=2).min(), rgba_image[:,:,:2].sum(axis=2).max())
        alpha = np.where(rgba_image[:,:,:2].sum(axis=2)==0, 0, 1).astype('float')
        # alpha = rgba_image[:,:,:2].sum(axis=2)
        # return alpha
    rgba_image[:, :, 3] = alpha
    return rgba_image
