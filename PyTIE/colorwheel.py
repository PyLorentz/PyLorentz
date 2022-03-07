"""Creates RGB images from vector fields.

This module contains several routines for plotting colormaps from input
data consisting of 2D images of the vector field. The output image
will be stored as a tiff color image. There are options to save it
using a custom RGB or standard HSV color-wheel.

Author: Arthur McCray, C. Phatak, ANL, Summer 2019.
"""

import numpy as np
from matplotlib import colors
import textwrap
import sys
from TIE_helper import dist


def color_im(Bx, By, Bz=None, rad=None, hsvwheel=True, background="black"):
    """Make the RGB image from x and y component vector maps.

    The color intensity corresponds to the in-plane vector component. If a
    z-component is given, it will map from black (negative) to white (positive).

    Args:
        Bx (2D array): (M x N) array consisting of the x-component of the vector
            field.
        By (2D array): (M x N) array consisting of the y-component of the vector
            field.
        Bz (2D array): optional (M x N) array consisting of the y-component of
            the vector field.
        rad (int): (`optional`) Radius of color-wheel in pixels. (default None -> height/16)
            Set rad = 0 to remove color-wheel.
        hsvwheel (bool):
            - True  -- (default) use a standard HSV color-wheel (3-fold)
            - False -- use a four-fold color-wheel
        background (str):
            - 'black' -- (default) magnetization magnitude corresponds to value.
            - 'white' -- magnetization magnitude corresponds to saturation.

    Returns:
        ``ndarray``: Numpy array (M x N x 3) containing the color-image.
    """

    if rad is None:
        rad = Bx.shape[0] // 16
        rad = max(rad, 16)

    bmag = np.sqrt(Bx ** 2 + By ** 2)

    if rad > 0:
        pad = 10  # padding between edge of image and color-wheel
    else:
        pad = 0
        rad = 0

    dimy = np.shape(By)[0]
    if dimy < 2 * rad:
        rad = dimy // 2
    dimx = np.shape(By)[1] + 2 * rad + pad
    cimage = np.zeros((dimy, dimx, 3))

    if hsvwheel:
        # Here we will proceed with using the standard HSV color-wheel routine.
        # Get the Hue (angle) as By/Bx and scale between [0,1]
        hue = (np.arctan2(By, Bx) + np.pi) / (2 * np.pi)

        if Bz is None:
            z_wheel = False
            # make the color image
            if background == "white":  # value is ones, magnitude -> saturation
                cb = np.dstack(
                    (hue, bmag / np.max(bmag), np.ones([dimy, dimx - 2 * rad - pad]))
                )
            elif background == "black":  # saturation is ones, magnitude -> values
                cb = np.dstack(
                    (hue, np.ones([dimy, dimx - 2 * rad - pad]), bmag / np.max(bmag))
                )
            else:
                print(
                    textwrap.dedent(
                        """
                    An improper argument was given in color_im().
                    Please choose background as 'black' or 'white.
                    'white' -> magnetization magnitude corresponds to saturation.
                    'black' -> magnetization magnitude corresponds to value."""
                    )
                )
                sys.exit(1)

        else:
            z_wheel = True
            theta = np.arctan2(Bz, np.sqrt(Bx ** 2 + By ** 2))
            value = np.where(theta < 0, np.cos(2 * theta) / 2 + 1 / 2, 1)
            # value = np.where(theta<0, 1-1/(1+np.exp(10*theta*2/np.pi+5)), 1)#sigmoid
            sat = np.where(theta > 0, np.cos(2 * theta) / 2 + 1 / 2, 1)
            # sat = np.where(theta>0, 1-1/(1+np.exp(-10*theta*2/np.pi+5)), 1)#sigmoid
            cb = np.dstack((hue, sat, value))

        if rad > 0:  # add the color-wheel
            cimage[:, : -2 * rad - pad, :] = cb
            # make the color-wheel and add to image
            wheel = colorwheel_HSV(rad, background=background, z=z_wheel)
            cimage[
                dimy // 2 - rad : dimy // 2 + rad,
                dimx - 2 * rad - pad // 2 : -pad // 2,
                :,
            ] = wheel
        else:
            cimage = cb
        # Convert to RGB image.
        cimage = colors.hsv_to_rgb(cimage)

    else:  # four-fold color wheel
        bmag = np.where(bmag != 0, bmag, 1.0001)
        cang = Bx / bmag  # cosine of the angle
        sang = np.sqrt(1 - cang ** 2)  # and sin

        # define the 4 color quadrants
        q1 = ((Bx >= 0) * (By <= 0)).astype(int)
        q2 = ((Bx < 0) * (By < 0)).astype(int)
        q3 = ((Bx <= 0) * (By >= 0)).astype(int)
        q4 = ((Bx > 0) * (By > 0)).astype(int)

        # as is By = Bx = 0 -> 1,1,1 , so to correct for that:
        no_B = np.where((Bx == 0) & (By == 0))
        q1[no_B] = 0
        q2[no_B] = 0
        q3[no_B] = 0
        q4[no_B] = 0

        # Apply to green, red, blue
        green = q1 * bmag * np.abs(sang)
        green += q2 * bmag
        green += q3 * bmag * np.abs(cang)

        red = q1 * bmag
        red += q2 * bmag * np.abs(sang)
        red += q4 * bmag * np.abs(cang)

        blue = (q3 + q4) * bmag * np.abs(sang)

        # apply to cimage channels and normalize
        cimage[:, : dimx - 2 * rad - pad, 0] = red
        cimage[:, : dimx - 2 * rad - pad, 1] = green
        cimage[:, : dimx - 2 * rad - pad, 2] = blue
        cimage = cimage / np.max(cimage)

        # add color-wheel
        if rad > 0:
            mid_y = dimy // 2
            cimage[mid_y - rad : mid_y + rad, dimx - 2 * rad :, :] = colorwheel_RGB(rad)

    return cimage


def colorwheel_HSV(rad, background, z=False, **kwargs):
    """Creates an HSV color-wheel as a np array to be inserted into the color-image.

    Args:
        rad (int): (`optional`) Radius of color-wheel in pixels. (default None -> height/16)
            Set rad = 0 to remove color-wheel.
        background (str):
            - 'black' -- (default) magnetization magnitude corresponds to value.
            - 'white' -- magnetization magnitude corresponds to saturation.

    Returns:
        ``ndarray``: Numpy array of shape (2*rad, 2*rad).
    """
    inverse = kwargs.get("inverse", False)
    line = np.arange(2 * rad) - rad
    [X, Y] = np.meshgrid(line, line, indexing="xy")
    th = np.arctan2(Y, X)
    # shift angles to [0,2pi]
    # hue maps to angle
    h_col = (th + np.pi) / 2 / np.pi
    # saturation maps to radius
    rr = np.sqrt(X ** 2 + Y ** 2)
    msk = np.zeros(rr.shape)
    msk[np.where(rr <= rad)] = 1.0
    # mask and normalize
    rr *= msk
    rr /= np.amax(rr)

    if z:
        if inverse:
            rr += -1 * (msk - 1)
            msk = np.ones_like(msk)
            theta = (rr - 0.5) * np.pi
            value = np.where(theta < 0, np.cos(2 * theta) / 2 + 1 / 2, 1)
            sat = np.where(theta > 0, np.cos(2 * theta) / 2 + 1 / 2, 1)
        else:
            theta = (rr - 0.5) * -1 * np.pi
            value = np.where(theta < 0, np.cos(2 * theta) / 2 + 1 / 2, 1)
            sat = np.where(theta > 0, np.cos(2 * theta) / 2 + 1 / 2, 1)
            value *= msk
            sat *= msk
        return np.dstack((h_col, sat, value))

    else:
        val_col = np.ones(rr.shape) * msk
        if background == "white":
            return np.dstack((h_col, rr, val_col))
        else:
            return np.dstack((h_col, val_col, rr))


def colorwheel_RGB(rad):
    """Makes a 4-quadrant RGB color-wheel as a np array to be inserted into the color-image

    Args:
        rad (int): (`optional`) Radius of color-wheel in pixels. (default None -> height/16)
            Set rad = 0 to remove color-wheel.

    Returns:
        ``ndarray``: Numpy array of shape (2*rad, 2*rad).
    """

    # make black -> white gradients
    dim = rad * 2
    grad_x = np.array([np.arange(dim) - rad for _ in range(dim)]) / rad
    grad_y = grad_x.T

    # make the binary mask
    tr = dist(dim, dim, shift=True) * dim
    circ = np.where(tr > rad, 0, 1)

    # magnitude of RGB values (equiv to value in HSV)
    bmag = np.sqrt((grad_x ** 2 + grad_y ** 2))

    # remove 0 to divide, make other grad distributions
    bmag = np.where(bmag != 0, bmag, 1)
    cang = grad_x / bmag
    sang = np.sqrt(1 - cang * cang)

    # define the quadrants
    q1 = ((grad_x >= 0) * (grad_y <= 0)).astype(int)
    q2 = ((grad_x < 0) * (grad_y < 0)).astype(int)
    q3 = ((grad_x <= 0) * (grad_y >= 0)).astype(int)
    q4 = ((grad_x > 0) * (grad_y > 0)).astype(int)

    # Apply to colors
    green = q1 * bmag * np.abs(sang)
    green = green + q2 * bmag
    green = green + q3 * bmag * np.abs(cang)

    red = q1 * bmag
    red = red + q2 * bmag * np.abs(sang)
    red = red + q4 * bmag * np.abs(cang)

    blue = (q3 + q4) * bmag * np.abs(sang)

    # apply masks
    green = green * circ
    red = red * circ
    blue = blue * circ

    # stack into one image and fix center from divide by 0 error
    cwheel = np.dstack((red, green, blue))
    cwheel[rad, rad] = [0, 0, 0]
    cwheel = np.array(cwheel / np.max(cwheel))
    return cwheel
