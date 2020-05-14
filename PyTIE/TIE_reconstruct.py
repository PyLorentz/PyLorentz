#!/usr/bin/python
#
# NAME: TIE_reconstruct.py
#
# PURPOSE:
# A routine for solving the transport of intensity equation; for use with Lorentz
# TEM throughfocal series to reconstruct B field magnetization of the sample. 
#
# CALLING SEQUENCE:
# To be called when running the Jupyter notebook TIE_template.ipynb
# results = TIE(TIE_params, flip_unflip_defocus_stack, defocus_value,
#                        microscope_object, dataname,  save = bool)
#
# PARAMETERS:
#  ptie: A parameters class from tie_params.py
#  tifs : Five image array in order: [ +- , -- , 0 , ++ , -+ ]
#         (+- = unflip/flip, +- = over/underfocus)
#  pscope : Microscope parameters class from miscroscopes.py
#  dataname : String ,the output filename to be used for saving the images (32 bit tiffs)
#             Files will be saved ptie.data_loc/images/dataname_defval_<key>.tiff
#  sym: Boolean, if you want to symmetrize the image before reconstructing (reconstructs 4x as large image). Default False
#  qc: Float, the Tikhonov frequency, or "percent" to use 15% of q, Defaulte None
#  save: Boolean, False if you don't want to save the data. Default True
#
# RETURNS:
#  result : A dictionary with keys: 
#       'byt' : y-component of integrated magnetic induction,
#       'bxt' : x-copmonent of integrated magnetic induction,
#       'bbt' : magnitude of integrated magnetic induction, 
#       'phase_m' : magnetic phase shift (radians),
#       'phase_e' : electrostatic phase shift (if using flip stack) (radians),
#       'dIdZ_m' : intensity derivative for calculating phase_m,
#       'dIdZ_e' : intensity derivative for calculating phase_m (if using flip stack), 
#       'color_b' : RGB image of magnetization,
#       'inf_im' : the in-focus image
#
# AUTHOR:
# Arthur R. C. McCray, ANL, Summer 2019.
#----------------------------------------------------------------------------------------------------


import numpy as np
from TIE_helper import dist, scale_stack, show_im, select_tifs
import skimage.external.tifffile as tifffile
from colorwheel import color_im, UniformBicone
import os
from copy import deepcopy
import scipy 
from pathlib import Path
from longitudinal_deriv import *

import time



def TIE(i, ptie, pscope, dataname = '', long_deriv = False, sym=False, qc=None, save = True, v = 1):
    '''
    Calculates the necessary arrays, derivatives, etc. and then calls 
    phase_reconstruct to solve the TIE. 
    
    For now only uses 3-point derivative method, and therefore expects a set of 
    five images in array: 
    [ - , flip - , infocus, + , flip + ]
    
    save = True  ->  saves all working images. 
    save = 'b'   ->  save just bx, by, and color_b
    save = False ->  don't save.
    Saves the images to ptie.data_loc/images/

    Returns a dictionary:
    results = {
        'byt' : y-component of integrated magnetic induction,
        'bxt' : x-copmonent of integrated magnetic induction,
        'bbt' : magnitude of integrated magnetic induction, 
        'phase_m' : magnetic phase shift (radians),
        'phase_e' : electrostatic phase shift (if using flip stack) (radians),
        'dIdZ_m' : intensity derivative for calculating phase_m,
        'dIdZ_e' : intensity derivative for calculating phase_m (if using flip stack), 
        'color_b' : RGB image of magnetization,
        'inf_im' : the in-focus image}

    '''
    results = {
        'byt' : None,
        'bxt' : None,
        'bbt' : None, 
        'phase_e' : None,
        'phase_m' : None,
        'dIdZ_m' : None,
        'dIdZ_e' : None, 
        'color_b' : None,
        'inf_im' : None}

    if long_deriv:
        unders = list(reversed([-1*i for i in ptie.defvals]))
        defval = unders + [0] + ptie.defvals
        if ptie.flip:
            print('Aligning with complete longitudinal derivates:\n', defval, '\nwith both flip/unflip tfs.')
        else:
            print('Aligning with complete longitudinal derivates:\n', defval, '\nwith only unflip tfs.')
    else:
        defval = ptie.defvals[i]
        if ptie.flip:
            print('Aligning for defocus value: ', defval, ' with both flip/unflip tfs.')
        else:
            print('Aligning for defocus value: ', defval, ' with only unflip tfs.')

    right, left = ptie.crop['right']  , ptie.crop['left']
    bottom, top = ptie.crop['bottom'] , ptie.crop['top']
    dim_y = bottom - top 
    dim_x = right - left 
    tifs = select_tifs(i, ptie, long_deriv)

    mask = ptie.mask[top:bottom, left:right]
    if np.max(mask) == np.min(mask): # doesn't like all white mask
        mask = scipy.ndimage.morphology.binary_erosion(mask) 

    # crop images and apply mask 
    for ii in range(len(tifs)):
        tifs[ii] = tifs[ii][top:bottom, left:right]
        tifs[ii] *= mask

    if sym:
        print("Reconstructing with symmetrized image.")
        dim_y *= 2
        dim_x *= 2

    # make the inverse laplacian, uses python implementation of IDL dist funct 
    q = dist(dim_y,dim_x)/np.sqrt(dim_y*dim_x)
    q[0, 0] = 1
    if qc is not None and qc is not False:
        if qc == 'percent':
            print("Reconstructing with Tikhonov percentage: 15%")
            qc = 0.15 * q * ptie.scale**2
        else:
            qc = qc 
            print("Reconstructing with Tikhonov value: {:}".format(qc))

        qi = q**2 / (q**2 + qc**2)**2
    else: # normal laplacian method
        print("Reconstructing with normal Laplacian method")
        qi = 1 / q**2
    qi[0, 0] = 0
    ptie.qi = qi # saves the freq dist
   
    # Normalizing, scaling the images 
    scaled_tifs = scale_stack(tifs)
    # very small offset from 0, affects uniform magnetizations
    # but can be compensated for (if 0) by symmetrizing the image.
    scaled_tifs += 1e-9 

    # get the infocus image
    if long_deriv and ptie.flip:
        inf_unflip = scaled_tifs[len(tifs)//4]
        inf_flip = scaled_tifs[3*len(tifs)//4]
        inf_im = (inf_unflip+inf_flip)/2
    else: 
        inf_im = scaled_tifs[len(tifs)//2]

    # Inverting masked areas on infocus image because we divide by it
    inf_im += 1 - mask
    # Make sure there are no zeros left: 
    inf_im = np.where(scaled_tifs[len(tifs)//2] == 0, 0.001, inf_im)
    results['inf_im'] = inf_im    
    
    if v >= 2: 
        print("""Scaled images (+- = unflip/flip, +- = over/underfocus)
             in order [ +- , -- , 0 , ++ , -+ ]""")
        for im in scaled_tifs:
            print("max: {:.3f}, min: {:.2f}, total intensity: {:.4f}\n".format(
                np.max(im), np.min(im), np.sum(im)))

    # Calculate derivatives
    if long_deriv: 
        unflip_stack = tifs[:ptie.num_files]
        flip_stack = tifs[ptie.num_files:]

        if long_deriv == 'multi': 
            print('Computing the longitudinal derivative with Multiprocessing.')
            unflip_deriv = polyfit_deriv_multiprocess(unflip_stack, defval)
        else:
            print('Computing the longitudinal derivative normally.')
            unflip_deriv = polyfit_deriv(unflip_stack, defval)

        if ptie.flip:
            if long_deriv == 'multi':
                print('Computing the flip stack longitudinal derivative with Multiprocessing.')
                flip_deriv = polyfit_deriv_multiprocess(flip_stack, defval)
            else:
                print('Computing the flip stack longitudinal derivative normally.')
                flip_deriv = polyfit_deriv(flip_stack, defval)

            dIdZ_m = (unflip_deriv - flip_deriv)/2 
            dIdZ_e = (unflip_deriv + flip_deriv)/2 
        else:
            dIdZ_m = unflip_deriv 

    else:
        if ptie.flip:
            dIdZ_m = 1/2 * (scaled_tifs[3] - scaled_tifs[0] - 
                          (scaled_tifs[4] - scaled_tifs[1]))
            dIdZ_e = 1/2 * (scaled_tifs[3] - scaled_tifs[0] + 
                          (scaled_tifs[4] - scaled_tifs[1]))
        else:
            dIdZ_m = scaled_tifs[2] - scaled_tifs[0]

    
    dIdZ_m *= mask 
    if ptie.flip:
        dIdZ_e += (1-mask) * np.mean(dIdZ_e)

    # Set derivatives to have 0 total "energy" 
    totm = np.sum(dIdZ_m)/np.sum(mask)
    dIdZ_m -= totm
    results['dIdZ_m'] = dIdZ_m
    if ptie.flip:
        tote = np.sum(dIdZ_e)/np.sum(mask)
        dIdZ_e -= tote
        results['dIdZ_e'] = dIdZ_e

    if sym:
        dIdZ_m = symmetrize(dIdZ_m)
        if ptie.flip:
            dIdZ_e = symmetrize(dIdZ_e)

    dIdZ_m *= mask 
    if ptie.flip:
        dIdZ_e *= mask 

    ### Now time to call phase_reconstruct, first for E if we have a flipped tfs 
    print('Calling TIE solver\n')
    if ptie.flip:
        resultsE = phase_reconstruct(ptie, inf_im, dIdZ_e, pscope, 
                                defval, sym = sym, long_deriv = long_deriv)   
        # We only care about the E phase.  
        results['phase_e'] = resultsE['phase']

    ### Now run for B, 
    resultsB = phase_reconstruct(ptie, inf_im, dIdZ_m, pscope, 
                                defval, sym = sym, long_deriv = long_deriv)
    results['byt'] = resultsB['ind_y']
    results['bxt'] = resultsB['ind_x']
    results['bbt'] = np.sqrt(results['bxt']**2 + results['byt']**2)
    results['phase_m'] = resultsB['phase']
    results['color_b'] = color_im(results['bxt'], results['byt'],
                                    hsvwheel=True, background='black') 

    if v >= 1:
        show_im(results['color_b'], "B-field color HSV colorwheel")

    # save the images
    if save: 
        save_results(i, results, ptie, dataname, sym, qc, save, v, long_deriv = long_deriv)

    print('Phase reconstruction completed.')
    return results


def SITIE(ptie, pscope, dataname = '', sym =False, qc = None, save = True, scale = None, v = 1):
    '''
    Uses a modified TIE to get the magnetic phase shift and induction from a single image. 
    Only applicable to _uniform and thin_ magnetic samples. 
    Please see: Chess, J. J. et al. Ultramicroscopy 177, 78–83 (2018).
    '''
    results = {
        'byt' : None,
        'bxt' : None,
        'bbt' : None, 
        'phase_m' : None,
        'color_b' : None}

    defval = ptie.defvals[0]
    right, left = ptie.crop['right']  , ptie.crop['left']
    bottom, top = ptie.crop['bottom'] , ptie.crop['top']
    dim_y = bottom - top 
    dim_x = right - left 

    image = ptie.dm3stack[0].data[top:bottom, left:right]

    if sym:
        print("Reconstructing with symmetrized image.")
        dim_y *= 2
        dim_x *= 2

    # q = dist(dim_y,dim_x)/np.sqrt(dim_y*dim_x)
    q = dist(dim_y,dim_x)/np.sqrt(dim_y*dim_x)

    q[0, 0] = 1
    if qc is not None and qc is not False:
        if qc == 'percent':
            print("Reconstructing with Tikhonov percentage: 15%")
            qc = 0.15 * q * ptie.scale**2
        else:
            qc = qc 
            print("Reconstructing with Tikhonov value: {:}".format(qc))

        qi = q**2 / (q**2 + qc**2)**2
    else: # normal laplacian method
        print("Reconstructing with normal Laplacian method")
        qi = 1 / q**2

    qi[0, 0] = 0
    ptie.qi = qi # saves the freq dist

    # constructing "infocus" image
    infocus = np.ones(np.shape(image))*np.mean(image)
    # calculate "derivative" and normalize
    dIdZ = 2 * (image - infocus) 
    dIdZ -= np.sum(dIdZ)/np.size(infocus)

    if sym:
        dIdZ = symmetrize(dIdZ)

    ### Now calling the phase reconstruct in the normal way
    print('Calling SITIE solver\n')
    resultsB = phase_reconstruct(ptie, infocus, 
                                dIdZ, pscope, defval, sym = sym)

    results['byt'] = resultsB['ind_y']
    results['bxt'] = resultsB['ind_x']
    results['bbt'] = np.sqrt(results['bxt']**2 + results['byt']**2)
    results['phase_m'] = resultsB['phase']
    results['color_b'] = color_im(results['bxt'], results['byt'],
        hsvwheel=True, background='black') # Change to 4-fold colorwheel if applicable
    if v >= 1:
        show_im(results['color_b'], "B field color, HSV colorhweel")
    
    # save the images
    if save: 
        save_results(0, results, ptie, dataname, sym, qc, save, v)

    print('Phase reconstruction completed.')
    return results


def phase_reconstruct(ptie, infocus, dIdZ, pscope, defval, sym = False, long_deriv = False):
    '''
    Solve the transport of intensity equation (TIE) with the inverse laplacian
    method. Return a dictionary of arrays.
    '''
    results = {}
    # actual image dimensions regardless of symmetrize 
    dim_y = infocus.shape[0]
    dim_x = infocus.shape[1]
    qi = ptie.qi

    if sym:
        infocus = symmetrize(infocus)
        y = dim_y * 2
        x = dim_x * 2
    else:
        y = dim_y
        x = dim_x

    # fourier transform of longitudnal derivatives
    fd = np.fft.fft2(dIdZ, norm='ortho')  

    # applying 2/3 qc cutoff mask (see de Graef 2003)
    gy, gx = np.ogrid[-y//2:y//2, -x//2:x//2]
    rad = y/3 
    qc_mask = gy**2 + gx**2 <= rad**2
    qc_mask = np.fft.ifftshift(qc_mask)
    fd *= qc_mask
        
    # apply first inverse laplacian operator
    tmp = -1*np.fft.ifft2(fd*qi, norm='ortho') 
    
    # apply gradient operator and divide by in focus image
    grad_y1, grad_x1 = np.real(np.gradient(tmp)/infocus)

    # apply second gradient operator
    grad_y2 = np.gradient(grad_y1)[0]
    grad_x2 = np.gradient(grad_x1)[1]
    tot = grad_y2 + grad_x2
    
    # apply second inverse Laplacian
    fd = np.fft.fft2(tot, norm='ortho')
    fd *= qc_mask
    tmp = -1*np.fft.ifft2(fd*qi, norm='ortho')
    
    # scale
    if long_deriv:
        pre_Lap = -2*ptie.pre_Lap(pscope, 1)  
    else:
        pre_Lap = -1*ptie.pre_Lap(pscope, defval)
        
    if sym:
        results['phase'] = np.real(pre_Lap*tmp[:dim_y, :dim_x]) 
    else:
        results['phase'] = np.real(pre_Lap*tmp)  

    ### getting magnetic induction
    grad_y, grad_x = np.gradient(results['phase']) 
    pre_B = scipy.constants.hbar/(scipy.constants.e * ptie.scale) * 10**18 # T*nm^2 
    results['ind_x'] = pre_B * grad_y 
    results['ind_y'] = -1*pre_B * grad_x 
    return results


def symmetrize(image):
    ''' Makes the even symmetric extension of an image 
    (4x as large as original) '''

    sz_y, sz_x = image.shape
    dimy = 2 * sz_y
    dimx = 2 * sz_x

    imi = np.zeros((dimy,dimx))
    imi[ :sz_y, :sz_x] = image 
    imi[sz_y: , :sz_x] = np.flipud(image) # *-1 for odd symm
    imi[:,sz_x:] = np.fliplr(imi[:,:sz_x]) # *-1 for odd sym

    return imi


def save_results(i, results, ptie, dataname, sym, qc, save, v, long_deriv=False):
    '''
    save options: 'b', 'color', 
    '''
    if long_deriv:
        defval = 'long'
    else:
        defval = ptie.defvals[i]
    print('Saving images')
    if save == 'b':
        b_keys = ['bbt', 'color_b'] # add bxt and byt if you'd like
    elif save == 'color': 
        b_keys = ['color_b']

    
    # for some reason imagej scale requires an int
    res = np.int(1/ptie.scale) 
    if not dataname.endswith('_'):
        dataname += '_'
    
    if ptie.SITIE:
        save_path = os.path.join(ptie.data_loc, 'SITIE')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.path.join(ptie.data_loc, 'images')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    for key,value in results.items():
        # save either all or just some of the images
        if save == 'b':
            if key not in b_keys:
                continue
        if value is None:
            continue

        if key == 'color_b':
            im = (value * 255).astype('uint8')
        else: 
            im = value.astype('float32')
        
        save_name = dataname + str(defval)+'_' + key + '.tiff'
        if v >= 2: 
            print(f'Saving {os.path.join(Path(save_path).absolute(), save_name)}.tiff')
        tifffile.imsave(os.path.join(save_path, save_name), im, 
            imagej = True,
            resolution = (res, res),
            metadata={'unit': 'um'})

    # make a txt file with parameters: 
    with open(os.path.join(save_path, dataname + "recon_params.txt"), "w") as txt:
        txt.write("Reconstruction parameters for {:}\n".format(dataname[:-1]))
        txt.write("Defocus value: {} nm\n".format(defval))
        txt.write("Full E and M reconstruction: {} \n".format(ptie.flip))
        txt.write("Symmetrized: {} \n".format(sym))
        txt.write("Thikonov filter: {} \n".format(qc))
        
    return 0
