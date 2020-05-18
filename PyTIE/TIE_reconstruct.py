"""File containing TIE and SITIE reconstruction routines. 

Known Bugs: 
- Reconstructing with non-square regions causes a scaling error in the magnetization
    in the x and y directions. 
- Longitudinal derivative gives a magnetization scaling error for some 
    experimental datasets. 

Routines for solving the transport of intensity equation; for use with Lorentz
TEM throughfocal series to reconstruct B field magnetization of the sample. 

AUTHOR:
Arthur McCray, ANL, Summer 2019.
--------------------------------------------------------------------------------
"""

import numpy as np
from TIE_helper import dist, scale_stack, show_im, select_tifs
import skimage.external.tifffile as tifffile
from colorwheel import color_im
import os
import sys
import scipy 
from pathlib import Path
from longitudinal_deriv import polyfit_deriv, polyfit_deriv_multiprocess

"""
PARAMETERS:
 ptie: A parameters class from tie_params.py
 tifs : Five image array in order: [ +- , -- , 0 , ++ , -+ ]
        (+- = unflip/flip, +- = over/underfocus)
 pscope : Microscope parameters class from miscroscopes.py
 dataname : String ,the output filename to be used for saving the images (32 bit tiffs)
            Files will be saved ptie.data_loc/images/dataname_defval_<key>.tiff
 sym: Boolean, if you want to symmetrize the image before reconstructing (reconstructs 4x as large image). Default False
 qc: Float, the Tikhonov frequency, or "percent" to use 15% of q, Defaulte None
 save: Boolean, False if you don't want to save the data. Default True

RETURNS:
 result : A dictionary with keys: 
      'byt' : y-component of integrated magnetic induction,
      'bxt' : x-copmonent of integrated magnetic induction,
      'bbt' : magnitude of integrated magnetic induction, 
      'phase_m' : magnetic phase shift (radians),
      'phase_e' : electrostatic phase shift (if using flip stack) (radians),
      'dIdZ_m' : intensity derivative for calculating phase_m,
      'dIdZ_e' : intensity derivative for calculating phase_e (if using flip stack), 
      'color_b' : RGB image of magnetization,
      'inf_im' : the in-focus image"""

def TIE(i=-1, ptie=None, pscope=None, dataname='', sym=False, qc=None, save=False, long_deriv=False, v=1):
    """Sets up the TIE reconstruction and calls phase_reconstruct. 

    This function calculates the necessary arrays, derivatives, etc. and then 
    calls passes them to phase_reconstruct which solve the TIE. 
    
    Args: 
        i: Int. index of ptie.defvals to use for reconstruction. Default value is -1
            which corresponds to the most defocused images for a central 
            difference method derivative. i is ignored if using a longitudinal
            derivative. 
        ptie: TIE_params object. Object containing the image from TIE_params.py
        pscope : microscope object. Should have correct accelerating voltage as
            the microscope that took the images.
        dataname : String. The output filename to be used for saving the images. 
                Files will be saved ptie.data_loc/images/dataname_<defval>_<key>.tiff
        sym: Boolean. Fourier edge effects are marginally improved by 
            symmetrizing the images before reconstructing (image reconstructed 
            is 4x as large). Default False.
        qc: Float. The Tikhonov frequency to use as filter, or "percent" to use 
            15% of q, Default None. 
        save: Bool or string. Whether you want to save the output. Default False. 
            save = True    ->  saves all images. 
            save = 'b'     ->  save just bx, by, and color_b
            save = 'color' ->  saves just color_b
            save = False   ->  don't save.
            Saves the images to ptie.data_loc/images/
        long_deriv: Bool. Whether to use the longitudinal derivative (True) or 
            central difference method (False). Default False. 
        v: Int. Verbosity. 
            0 : ##TODO no output
            1 : Default output
            2 : Extended output for debugging. 

    Returns: A dictionary of arrays. 
        results = {
            'byt' : y-component of integrated magnetic induction,
            'bxt' : x-copmonent of integrated magnetic induction,
            'bbt' : magnitude of integrated magnetic induction, 
            'phase_m' : magnetic phase shift (radians),
            'phase_e' : electrostatic phase shift (if using flip stack) (radians),
            'dIdZ_m' : intensity derivative for calculating phase_m,
            'dIdZ_e' : intensity derivative for calculating phase_e (if using flip stack), 
            'color_b' : RGB image of magnetization,
            'inf_im' : the in-focus image
        }
    """
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
        unders = list(reversed([-1*ii for ii in ptie.defvals]))
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
        print("""\nScaled images (+- = unflip/flip, +- = over/underfocus)
        in order [ +- , -- , 0 , ++ , -+ ]""")
        for im in scaled_tifs:
            print("max: {:.3f}, min: {:.2f}, total intensity: {:.4f}".format(
                np.max(im), np.min(im), np.sum(im)))
        print()

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
    
    # Set derivatives to have 0 total "energy" 
    dIdZ_m *= mask 
    totm = np.sum(dIdZ_m)/np.sum(mask)
    dIdZ_m -= totm
    dIdZ_m *= mask 
    results['dIdZ_m'] = dIdZ_m

    if ptie.flip:
        dIdZ_e *= mask
        tote = np.sum(dIdZ_e)/np.sum(mask)
        dIdZ_e -= tote
        dIdZ_e *= mask 
        results['dIdZ_e'] = dIdZ_e

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
    results['bbt'] = np.sqrt(resultsB['ind_x']**2 + resultsB['ind_y']**2)
    results['phase_m'] = resultsB['phase']
    results['color_b'] = color_im(resultsB['ind_x'], resultsB['ind_y'],
                                    hsvwheel=True, background='black') 

    if v >= 1:
        show_im(results['color_b'], "B-field color HSV colorwheel")

    # save the images
    if save: 
        save_results(defval, results, ptie, dataname, sym, qc, save, v, long_deriv = long_deriv)

    print('Phase reconstruction completed.')
    return results


def SITIE(ptie=None, pscope=None, dataname='', sym=False, qc=None, save=True, i=-1, flipstack=False, v=1):
    """Uses a modified derivative to get the magnetic phase shift with TIE from a single image.

    This technique is only appplicable to uniformly thin samples from which the 
    only source of contrast is magnetic Fresnel contrast. All other sources of 
    contrast including dirt on the sample, thickness variation, and diffraction 
    contrast will give false magnetic inductions. For more information please 
    refer to: Chess, J. J. et al. Ultramicroscopy 177, 78–83 (2018).
    
    Args: 
        ptie: TIE_params object. Object containing the image from TIE_params.py
        pscope : microscope object. Should have correct accelerating voltage as
            the microscope that took the images.
        dataname : String. The output filename to be used for saving the images. 
                Files will be saved ptie.data_loc/images/dataname_<defval>_<key>.tiff
        sym: Boolean. Fourier edge effects are marginally improved by 
            symmetrizing the images before reconstructing (image reconstructed 
            is 4x as large). Default False.
        qc: Float. The Tikhonov frequency to use as filter, or "percent" to use 
            15% of q, Default None. 
        save: Bool or string. Whether you want to save the output. Default False. 
            save = True    ->  saves all images. 
            save = 'b'     ->  save just bx, by, and color_b
            save = 'color' ->  saves just color_b
            save = False   ->  don't save.
            Saves the images to ptie.data_loc/images/
        long_deriv: Bool. Whether to use the longitudinal derivative (True) or 
            central difference method (False). Default False. 
        i: Int. index of __the ptie.dm3stack or ptie.flip_dm3stack__ to 
            reconstruct. This is not the defocus index like in TIE. Default 
            value is -1 which corresponds to the most overfocused image. 
        flipstack: Bool. Whether to pull the image from the ptie.dmrstack[i] or
            ptie.flip_dm3stack[i]. Default is False, calls image from dm3stack.
        v: Int. Verbosity. 
            0 : ##TODO no output
            1 : Default output
            2 : Extended output for debugging. 

    Returns: A dictionary of arrays. 
        results = {
            'byt' : y-component of integrated magnetic induction,
            'bxt' : x-copmonent of integrated magnetic induction,
            'bbt' : magnitude of integrated magnetic induction, 
            'phase_m' : magnetic phase shift (radians),
            'color_b' : RGB image of magnetization,
        }
    """
    results = {
        'byt' : None,
        'bxt' : None,
        'bbt' : None, 
        'phase_m' : None,
        'color_b' : None}

    # selecting the right defocus value for the image
    if i >= 2*len(ptie.defvals)+1:
        print("i given outside range.")
    else:
        if ptie.num_files > 1: 
            unders = list(reversed([-1*i for i in ptie.defvals]))
            defvals = unders + [0] + ptie.defvals
            defval = defvals[i]
        else:
            defval = ptie.defvals[0]
        print(f'SITIE defocus: {defval} nm')

    right, left = ptie.crop['right']  , ptie.crop['left']
    bottom, top = ptie.crop['bottom'] , ptie.crop['top']
    dim_y = bottom - top 
    dim_x = right - left 

    if i > len(ptie.dm3stack):
        print("You're selecting an image outside of the length of your stack.")
        return 0 
    else:
        if flipstack:
            print("Reconstructing with single flipped image.")
            image = ptie.flip_dm3stack[i].data[top:bottom, left:right]
        else:
            image = ptie.dm3stack[i].data[top:bottom, left:right]

    if sym:
        print("Reconstructing with symmetrized image.")
        dim_y *= 2
        dim_x *= 2

    # setup the inverse frequency distribution
    q = dist(dim_y,dim_x)/np.sqrt(dim_y*dim_x)
    q[0, 0] = 1
    if qc is not None and qc is not False: # add Tikhonov filter
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

    ### Now calling the phase reconstruct in the normal way
    print('Calling SITIE solver\n')
    resultsB = phase_reconstruct(ptie, infocus, 
                                dIdZ, pscope, defval, sym = sym)
    results['byt'] = resultsB['ind_y']
    results['bxt'] = resultsB['ind_x']
    results['bbt'] = np.sqrt(resultsB['ind_x']**2 + resultsB['ind_y']**2)
    results['phase_m'] = resultsB['phase']
    results['color_b'] = color_im(resultsB['ind_x'], resultsB['ind_y'],
        hsvwheel=True, background='black')
    if v >= 1:
        show_im(results['color_b'], "B field color, HSV colorhweel")
    
    # save the images
    if save: 
        save_results(defval, results, ptie, dataname, sym, qc, save, v, directory = "SITIE")
    print('Phase reconstruction completed.')
    return results


def phase_reconstruct(ptie, infocus, dIdZ, pscope, defval, sym=False, long_deriv=False):
    """The function that actually solves the TIE. 

    This function takes all the necessary inputs from TIE or SITIE and solves
    the TIE using the inverse Laplacian method. 

    Args: 
        ptie: TIE_params object. Object containing the image from TIE_params.py
        infocus: The infocus image. Should not have any zeros as we divide by it.
        dIdZ: The intensity derivative. 
        pscope : microscope object. Should have correct accelerating voltage as
            the microscope that took the images.
        sym: Boolean. Fourier edge effects are marginally improved by 
            symmetrizing the images before reconstructing (image reconstructed 
            is 4x as large). 
        long_deriv: Bool. Whether or not the longitudinal derivative was used. 
            Only affects the prefactor. 

    Returns: A dictionary of arrays.
        results = {
            'ind_y' : y-component of integrated induction,
            'ind_x' : x-copmonent of integrated induction,
            'phase' : phase shift (radians),
        }
    """
    results = {}
    # actual image dimensions regardless of symmetrize 
    dim_y = infocus.shape[0]
    dim_x = infocus.shape[1]
    qi = ptie.qi

    if sym:
        infocus = symmetrize(infocus)
        dIdZ = symmetrize(dIdZ)
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
        pre_Lap = -2*ptie.pre_Lap(pscope)  
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
    """Makes the even symmetric extension of an image (4x as large)."""
    sz_y, sz_x = image.shape
    dimy = 2 * sz_y
    dimx = 2 * sz_x

    imi = np.zeros((dimy,dimx))
    imi[ :sz_y, :sz_x] = image 
    imi[sz_y: , :sz_x] = np.flipud(image) # *-1 for odd symm
    imi[:,sz_x:] = np.fliplr(imi[:,:sz_x]) # *-1 for odd sym
    return imi


def save_results(defval, results, ptie, dataname, sym, qc, save, v, directory = None, long_deriv=False):
    """Save the contents of results dictionary as 32 bit tiffs.
    
    This function saves the contents of the supplied dictionary (either all or 
    a portion) to ptie.data_loc with the appropriate tags from results. It also
    creates a recon_params.txt file containing the reconstruction parameters. 

    The images are formatted as 32 bit tiffs with the resolution included so 
    they can be opened as-is by ImageJ with the scale set. 

    Args: 
        defval: Float. The defocus value for the reconstruction. Not used if 
            long_deriv == True. 
        results: Dict. Dictionary containing the 2D numpy arrays. 
        ptie: TIE_params object. 
        dataname: String. Name attached to the saved images. 
        sym: Bool. If the symmetrized method was used. Only relevant as its 
            included in the recon_params.txt.
        qc: Float. Same as sym, included in the text file. 
        save: Bool or string. How much of the results directory to save.  
            save = True    ->  saves all images. 
            save = 'b'     ->  save just bx, by, and color_b
            save = 'color' ->  saves just color_b
        v: Int. Verbosity. 
            1 : Default output (none)
            2 : Extended output, prints files as saving. 
        directory: String. The directory name to store the saved files. If 
            default (None) saves to ptie.data_loc/Images/ 
        long_deriv: Bool. Same as qc. Included in text file. 

    Returns: None
    """
    if long_deriv:
        defval = 'long'

    print('Saving images')
    if save == 'b':
        b_keys = ['bxt', 'byt', 'color_b']
    elif save == 'color': 
        b_keys = ['color_b']
    
    # for some reason imagej scale requires an int
    res = np.int(1/ptie.scale) 
    if not dataname.endswith('_'):
        dataname += '_'
    
    if directory is not None:
        save_path = os.path.join(ptie.data_loc, str(directory))
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
        txt.write("Longitudinal derivative: {} \n".format(long_deriv))

    return
