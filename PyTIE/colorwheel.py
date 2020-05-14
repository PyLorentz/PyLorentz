#!/usr/bin/python
#
# NAME: colorwheel.py
#
# PURPOSE:
# This routine will be used for plotting the colormap from the input
# data consisting of 2D images of the vector field. The output image
# will be stored as a tiff color image. There are options to save it
# using custom RGB colorwheel, or standard HSV colorwheel.
#
# CALLING SEQUENCE:
# result = Plot_ColorMap(Bx = Bx, By = By, hsvwheel = True, filename = filename)
#
# PARAMETERS:
#  Bx : 2D Array consisting of the x-component of the vector field
#  By : 2D Array consisting of the y-component of the vector field
#  hsvwheel : If True then colorimage using the standard HSV scheme or using the custom RGB scheme.
#  filename : The output filename to be used for saving the color image. If not provided, default
#             Vector_ColorMap.jpeg will be used.
#
# RETURNS:(M x N x 3) array containing the color-image.
#
# AUTHOR:
# Arthur McCray, C. Phatak, ANL, Summer 2019.
#----------------------------------------------------------------------------------------------------

import numpy as np 
from matplotlib import colors 


def UniformBicone(Bx, By, ldim = None, style = 'four', w_cen = False):
    '''Makes an RGB image using the (more) perceptually uniform  colorwheel
    by Will Lenthe: https://github.com/wlenthe/UniformBicone and based on 
    the paper by Peter Kovesi: https://peterkovesi.com/projects/colourmaps/''' 
    import colormap as cm

    if ldim is None:
        ldim = Bx.shape[0]//8

    mag = np.sqrt(Bx**2 + By**2)
    angle = np.arctan2(-By, -Bx)

    color = cm.disk(mag, angle, scale = True, map = style,  w_cen = w_cen, float = True)

    legend  = cm.disk_legend(map = style, w_cen = w_cen, width = ldim, float = True)

    dy, dx, dz = np.shape(color)
    cmap = np.zeros((dy, dx+ldim, dz))

    cmap[:dy, :dx,:] = color
    cmap[(dy-ldim)//2 : (dy + ldim)//2, dx:,:] = legend

    return cmap


def color_im(Bx, By, rad = None, hsvwheel = False, background = 'black'):
    ''' Make the RGB image from x and y component vector maps. 
    Default is a 4-corner colorwheel, set hsvwheel = True for a 3-fold colorwheel.
    For hsvwheel can set "background" = 'black' or 'white.
    'white' -> magnetization magnitude corresponds to saturation. 
    'black' -> magnetization magnitude corresponds to value.''' 

    if rad is None:
        rad = Bx.shape[0]//16

    mmax = max(np.abs(Bx).max(), np.abs(By).max())
    Bx = Bx/mmax
    By = By/mmax
    bmag = np.sqrt(Bx**2 + By**2) 

    pad = 10
    dimy = np.shape(By)[0] 
    dimx = np.shape(By)[1] + 2*rad + pad
    cimage = np.zeros((dimy, dimx, 3))

    if hsvwheel:
        # Here we will proceed with using the standard HSV colorwheel routine.
        # Get the Hue (angle) as By/Bx and scale between [0,1]
        hue = (np.arctan2(By,Bx) + np.pi)/2/np.pi

        # make the color image
        if background == 'white': 
            cb = np.dstack((hue, bmag/np.max(bmag), np.ones([dimy, dimx-2*rad-pad])))
        elif background == 'black':
            cb = np.dstack((hue, np.ones([dimy, dimx-2*rad-pad]), bmag/np.max(bmag)))
        else:
            print("""Please choose background as 'black' or 'white.
                'white' -> magnetization magnitude corresponds to saturation. 
                'black' -> magnetization magnitude corresponds to value.""")
            return 0 

        cimage[:,:-2*rad-pad,:] = cb

        # Draw the colorwheel
        wheel = colorwheel_HSV(rad, background = background)
        cimage[dimy//2-rad:dimy//2+rad, dimx-2*rad - pad//2:-pad//2,:] = wheel

        # Convert to RGB image.
        cimage = colors.hsv_to_rgb(cimage)
    
    else: 
        bmag = np.where(bmag != 0, bmag, 1.0001)
        # compute cosine of the angle
        cang = Bx/bmag
        from TIE_helper import show_im
        #put back
        # bmag = np.where(bmag == 0,  1.0001, bmag)
        # and sin
        sang = np.sqrt(1 - cang**2)

        # define the 4 color quadrants
        q1 = ((Bx >= 0) * (By <= 0)).astype(int)
        q2 = ((Bx < 0) * (By < 0)).astype(int)
        q3 = ((Bx <= 0) * (By >= 0)).astype(int)
        q4 = ((Bx > 0) * (By > 0)).astype(int)

        # By = Bx = 0 are turned into 1,1,1 so to correct that:
        no_B = np.where((Bx==0) & (By==0))
        q1[no_B] = 0
        q2[no_B] = 0
        q3[no_B] = 0
        q4[no_B] = 0

        # first apply to green
        green = q1 * bmag * np.abs(sang)
        green += q2 * bmag
        green += q3 * bmag * np.abs(cang)
        
        # red
        red = q1 * bmag 
        red += q2 * bmag * np.abs(sang)
        red += q4 * bmag * np.abs(cang)

        # blue
        blue = (q3+q4) * bmag *np.abs(sang)

        cimage[:,:dimx - 2*rad - pad, 0] = red
        cimage[:,:dimx - 2*rad - pad, 1] = green
        cimage[:,:dimx - 2*rad - pad, 2] = blue

        cimage = cimage/np.max(cimage)

        mid_y = dimy // 2 
        cimage[mid_y - rad : mid_y + rad, dimx - 2*rad: , :] = colorwheel_RGB(rad)

    return(cimage)

def colorwheel_HSV(rad, background):
    ''' A HSV colorwheel'''
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
        print("How did you get here? background should be 'white' or 'black'")
        return 0 


def colorwheel_RGB(rad):
    '''Makes a RGBY four-quadrant colorwheel used for B induction maps'''

    # make black -> white gradients
    dim = rad * 2 
    grad_x = np.array([np.arange(dim) - rad for _ in range(dim)]) / rad
    grad_y = grad_x.T
    
    # make the binary mask
    tr = np.roll(np.roll(dist(dim), rad,axis = 0), rad,axis=1)
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
    # green
    green = q1 * bmag * np.abs(sang)
    green = green + q2 * bmag
    green = green + q3 * bmag * np.abs(cang)
    # red
    red = q1 * bmag 
    red = red + q2 * bmag * np.abs(sang)
    red = red + q4 * bmag * np.abs(cang)
    # blue
    blue = (q3+q4) * bmag *np.abs(sang)

    # apply masks
    green = green*circ
    red = red*circ
    blue = blue*circ

    # stack into one image and fix center from divish 0 error
    cwheel = np.dstack((red,green,blue))
    cwheel[rad,rad] = [0,0,0]
    cwheel = np.array(cwheel/np.max(cwheel))
    
    return cwheel


def dist(n):
    '''
    Implementation of the IDL DIST function. 
    "Returns a rectangular array in which the value of each element is 
    proportional to its frequency." 
    My description: "Creates an array where each value is smallest distance to a 
    corner (measured from upper left corner of pixel/array box)"
    right now just for square matrices
    '''
    axis = np.linspace(-n//2+1, n//2, n)
    result = np.sqrt(axis**2 + axis[:,np.newaxis]**2)
    return np.roll(result, n//2+1, axis=(0,1))
