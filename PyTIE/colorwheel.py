"""Creates RGB images from vector fields. 

This file contains several routines for plotting colormaps from input 
data consisting of 2D images of the vector field. The output image
will be stored as a tiff color image. There are options to save it
using custom RGB colorwheel, or standard HSV colorwheel.

AUTHOR:
Arthur McCray, C. Phatak, ANL, Summer 2019.
--------------------------------------------------------------------------------
"""

import numpy as np 
from matplotlib import colors
import textwrap
import sys
from TIE_helper import dist as dist


def color_im(Bx, By, rad = None, hsvwheel = False, background = 'black'):
    """Make the RGB image from x and y component vector maps. 

    Args: 
        Bx: 2D Array (M x N) consisting of the x-component of the vector field
        By: 2D Array (M x N) consisting of the y-component of the vector field
    Optional Args: 
        rad: Int. Radius of colorwheel in pixels. (default None -> height/16)
            Set rad = 0 to remove colorhweel. 
        hsvwheel: Bool. (default False)
            True  -- use a standard HSV colorhweel (3-fold)
            False -- use a four-foldcolor image using the standard HSV scheme
        background: String. (default 'black')
            'white' -- magnetization magnitude corresponds to saturation. 
            'black' -- magnetization magnitude corresponds to value.
    
    Returns: 
        Numpy array (M x N x 3)c  containing the color-image.
    """

    if rad is None:
        rad = Bx.shape[0]//16
        rad = max(rad, 16)

    mmax = max(np.abs(Bx).max(), np.abs(By).max())
    Bx = Bx/mmax
    By = By/mmax
    bmag = np.sqrt(Bx**2 + By**2) 

    if rad > 0: 
        pad = 10 # padding between edge of image and colorhweel
    else:
        pad = 0
        rad = 0
    dimy = np.shape(By)[0] 
    dimx = np.shape(By)[1] + 2*rad + pad
    cimage = np.zeros((dimy, dimx, 3))

    if hsvwheel:
        # Here we will proceed with using the standard HSV colorwheel routine.
        # Get the Hue (angle) as By/Bx and scale between [0,1]
        hue = (np.arctan2(By,Bx) + np.pi)/(2*np.pi)

        # make the color image
        if background == 'white': # value is ones, magnitude -> saturation
            cb = np.dstack((hue, bmag/np.max(bmag), np.ones([dimy, dimx-2*rad-pad])))
        elif background == 'black': # saturation is ones, magnitude -> values
            cb = np.dstack((hue, np.ones([dimy, dimx-2*rad-pad]), bmag/np.max(bmag)))
        else:
            print(textwrap.dedent("""
                An improper argument was given in color_im(). 
                Please choose background as 'black' or 'white.
                'white' -> magnetization magnitude corresponds to saturation. 
                'black' -> magnetization magnitude corresponds to value."""))
            sys.exit(1)

        if rad > 0: # add the colorhweel
            cimage[:,:-2*rad-pad,:] = cb
            # make the colorwheel and add to image
            wheel = colorwheel_HSV(rad, background = background)
            cimage[dimy//2-rad:dimy//2+rad, dimx-2*rad - pad//2:-pad//2,:] = wheel
        else: 
            cimage = cb
        # Convert to RGB image.
        cimage = colors.hsv_to_rgb(cimage)
    
    else: # four-fold color wheel
        bmag = np.where(bmag != 0, bmag, 1.0001)
        cang = Bx/bmag # cosine of the angle
        sang = np.sqrt(1 - cang**2) # and sin

        # define the 4 color quadrants
        q1 = ((Bx >= 0) * (By <= 0)).astype(int) 
        q2 = ((Bx < 0) * (By < 0)).astype(int)
        q3 = ((Bx <= 0) * (By >= 0)).astype(int)
        q4 = ((Bx > 0) * (By > 0)).astype(int)

        # as is By = Bx = 0 -> 1,1,1 , so to correct for that:
        no_B = np.where((Bx==0) & (By==0))
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
        cimage[:, :dimx - 2*rad - pad, 0] = red
        cimage[:, :dimx - 2*rad - pad, 1] = green
        cimage[:, :dimx - 2*rad - pad, 2] = blue
        cimage = cimage/np.max(cimage)

        # add colorwheel
        if rad > 0:
            mid_y = dimy // 2 
            cimage[mid_y-rad : mid_y+rad, dimx-2*rad:, :] = colorwheel_RGB(rad)

    return(cimage)


def colorwheel_HSV(rad, background):
    """Creates the HSV colorwheel as a np array to be inserted into the cimage."""
    line = np.arange(2*rad) - float(rad)
    [X,Y] = np.meshgrid(line,line,indexing = 'xy')
    th = np.arctan2(Y,X)
    # shift angles to [0,2pi]
    # hue maps to angle
    h_col = (th + np.pi)/2/np.pi
    # saturation maps to radius
    rr = np.sqrt(X**2 + Y**2)
    msk = np.zeros(rr.shape)
    msk[np.where(rr <= rad)] = 1.0
    # mask and normalize
    rr *= msk
    rr /= np.amax(rr)
    # value is always 1
    val_col = np.ones(rr.shape) * msk
    if background == 'white':
        return np.dstack((h_col, rr, val_col))
    elif background == 'black':
        return np.dstack((h_col, val_col, rr))
    else:
        sys.exit(1)


def colorwheel_RGB(rad):
    """Makes a 4-quadrant RGB colorwheel as a np array to be inserted into the cimage"""

    # make black -> white gradients
    dim = rad * 2 
    grad_x = np.array([np.arange(dim) - rad for _ in range(dim)]) / rad
    grad_y = grad_x.T
    
    # make the binary mask
    tr = dist(dim, dim, shift=True) * dim
    circ = np.where(tr > rad, 0, 1)

    # magnitude of RGB values (equiv to value in HSV)
    bmag = np.sqrt((grad_x**2 + grad_y**2))

    # remove 0 to divide, make other grad distributions
    bmag = np.where(bmag != 0, bmag, 1)
    cang = grad_x/bmag
    sang = np.sqrt(1 - cang*cang)

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

    blue = (q3+q4) * bmag *np.abs(sang)

    # apply masks
    green = green*circ
    red = red*circ
    blue = blue*circ

    # stack into one image and fix center from divide by 0 error
    cwheel = np.dstack((red,green,blue))
    cwheel[rad,rad] = [0,0,0]
    cwheel = np.array(cwheel/np.max(cwheel))
    return cwheel


def UniformBicone(Bx, By, ldim = None, style = 'four', w_cen = False):
    """
    Makes an RGB image using the (more) perceptually uniform  colorwheels made
    by Will Lenthe: https://github.com/wlenthe/UniformBicone and based on 
    the paper by Peter Kovesi: https://peterkovesi.com/projects/colourmaps/

    Args: 
        Bx: 2D Array (M x N) consisting of the x-component of the vector field
        By: 2D Array (M x N) consisting of the y-component of the vector field
    Optional Args: 
        ldim: Int. Diameter of colorwheel in pixels. (default None -> height/8)
            Set ldim = 0 to remove colorhweel. 
        style: String. (default 'four')
            'four' -- Four-fold colorwheel
            'six' -- sixfold colorwheel
        w_cen: Bool. (default False)
            False -- zero magnetization corresponds to black 
            True -- zero magnetization corresponds to white
    
    Returns: 
        Numpy array (M x N x 3)c  containing the color-image.
    """
    import colormap as cm

    if ldim is None:
        ldim = Bx.shape[0]//8

    mag = np.sqrt(Bx**2 + By**2)
    angle = np.arctan2(-By, -Bx)
    color = cm.disk(mag, angle, scale = True, map = style,  w_cen = w_cen, float = True)
    legend  = cm.disk_legend(map = style, w_cen = w_cen, width = ldim, float = True)

    # assemble the colormap with colorhweel
    dy, dx, dz = np.shape(color)
    cmap = np.zeros((dy, dx+ldim, dz))
    cmap[:dy, :dx,:] = color
    cmap[(dy-ldim)//2 : (dy + ldim)//2, dx:,:] = legend
    cmap[np.isnan(cmap)] = 0

    if w_cen: # make sidebar where legend is white
        cmap[:,:,0] += np.where(cmap[:,:,0] > 0, 0, 1)    
        cmap[:,:,1] += np.where(cmap[:,:,1] > 0, 0, 1)    
        cmap[:,:,2] += np.where(cmap[:,:,2] > 0, 0, 1)    
    return cmap


