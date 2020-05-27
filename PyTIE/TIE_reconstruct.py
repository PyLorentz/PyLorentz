"""File containing TIE and SITIE reconstruction routines. 

Known Bugs: 
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

def TIE(i=-1, ptie=None, pscope=None, dataname='', sym=False, qc=None, hsv=True, save=False, long_deriv=False, v=1,
        rotate_translate=None):
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
            15% of q, Default None. If you use a Tikhonov filter the resulting 
            magnetization is no longer quantitative!
        hsv: Bool. Chooses the type of colorwheel to display.
            hsv = True     ->  An hsv colorwheel.
            hsv = False    ->  A 4-fold colorwheel.
        save: Bool or string. Whether you want to save the output. Default False.
            save = True    ->  saves all images. 
            save = 'b'     ->  save just bx, by, and color_b
            save = 'color' ->  saves just color_b
            save = False   ->  don't save.
            Saves the images to ptie.data_loc/images/
        long_deriv: Bool. Whether to use the longitudinal derivative (True) or 
            central difference method (False). Default False. 
        v: Int. Verbosity. 
            0 : no output
            1 : Default output
            2 : Extended output for debugging.
        rotate_translate: None or Tuple. This will adjust the view of
            the image by rotating and translating the data.
            None : Due not apply any transformation
            Tuple : Index 0 - Rotation
                    Index 1 - X Translation
                    Index 2 - Y Translation

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

    # turning off the print function if v=0
    vprint = print if v>=1 else lambda *a, **k: None
    if long_deriv:
        unders = list(reversed([-1*ii for ii in ptie.defvals]))
        defval = unders + [0] + ptie.defvals
        if ptie.flip:
            vprint('Aligning with complete longitudinal derivates:\n', defval, '\nwith both flip/unflip tfs.')
        else:
            vprint('Aligning with complete longitudinal derivates:\n', defval, '\nwith only unflip tfs.')
    else:
        defval = ptie.defvals[i]
        if ptie.flip:
            vprint('Aligning for defocus value: ', defval, ' with both flip/unflip tfs.')
        else:
            vprint('Aligning for defocus value: ', defval, ' with only unflip tfs.')

    right, left = ptie.crop['right']  , ptie.crop['left']
    bottom, top = ptie.crop['bottom'] , ptie.crop['top']
    dim_y = bottom - top 
    dim_x = right - left 
    tifs = select_tifs(i, ptie, long_deriv)

    if sym:
        vprint("Reconstructing with symmetrized image.")
        dim_y *= 2
        dim_x *= 2

    # make the inverse laplacian, uses python implementation of IDL dist funct 
    q = dist(dim_y,dim_x)
    q[0, 0] = 1
    if qc is not None and qc is not False:
        if qc == 'percent':
            vprint("Reconstructing with Tikhonov percentage: 15%")
            qc = 0.15 * q * ptie.scale**2
        else:
            qc = qc 
            vprint("Reconstructing with Tikhonov value: {:}".format(qc))

        qi = q**2 / (q**2 + qc**2)**2
    else: # normal laplacian method
        vprint("Reconstructing with normal Laplacian method")
        qi = 1 / q**2
    qi[0, 0] = 0
    ptie.qi = qi # saves the freq dist

    # If rotation and translation to be applied
    if rotate_translate is not None:
        rotate, x_shift, y_shift = rotate_translate
        for ii in range(len(tifs)):
            tifs[ii] = scipy.ndimage.rotate(tifs[ii], rotate, reshape=False, order=0)
            tifs[ii] = scipy.ndimage.shift(tifs[ii], (-y_shift, x_shift), order=0)
            if ii == 1:
                show_im(tifs[ii], "stop")
        mask = scipy.ndimage.rotate(ptie.mask, rotate, reshape=False, order=0)
        mask = scipy.ndimage.shift(mask, (-y_shift, x_shift), order=0)

    # crop images and apply mask
    if rotate_translate is None:
        mask = ptie.mask[top:bottom, left:right]
    else:
        mask = mask[top:bottom, left:right]
    for ii in range(len(tifs)):
        tifs[ii] = tifs[ii][top:bottom, left:right]
        tifs[ii] *= mask

    if np.min(mask) == np.max(mask):
        mask[0, 0] = 0

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
            vprint('Computing the longitudinal derivative with Multiprocessing.')
            unflip_deriv = polyfit_deriv_multiprocess(unflip_stack, defval)
        else:
            vprint('Computing the longitudinal derivative normally.')
            unflip_deriv = polyfit_deriv(unflip_stack, defval, v)

        if ptie.flip:
            if long_deriv == 'multi':
                vprint('Computing the flip stack longitudinal derivative with Multiprocessing.')
                flip_deriv = polyfit_deriv_multiprocess(flip_stack, defval)
            else:
                vprint('Computing the flip stack longitudinal derivative normally.')
                flip_deriv = polyfit_deriv(flip_stack, defval, v)

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
    vprint('Calling TIE solver\n')
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
                                    hsvwheel=hsv, background='black')

    # if v >= 1:
    #     show_im(results['color_b'], "B-field color HSV colorwheel")

    # save the images
    if save: 
        save_results(defval, results, ptie, dataname, sym, qc, save, v, long_deriv = long_deriv)

    vprint('Phase reconstruction completed.')
    return results


def SITIE(ptie=None, pscope=None, dataname='', sym=False, qc=None, save=True, i=-1, flipstack=False, v=1,
          rotate_translate=None):
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
        i: Int. index of __the ptie.imstack or ptie.flipstack__ to 
            reconstruct. This is not the defocus index like in TIE. Default 
            value is -1 which corresponds to the most overfocused image. 
        flipstack: Bool. Whether to pull the image from the ptie.dmrstack[i] or
            ptie.flipstack[i]. Default is False, calls image from imstack.
        v: Int. Verbosity. 
            0 : ##TODO no output
            1 : Default output
            2 : Extended output for debugging.
        rotate_translate: None or Tuple. This will adjust the view of
            the image by rotating and translating the data.
            None : Due not apply any transformation
            Tuple : Index 0 - Rotation
                    Index 1 - X Translation
                    Index 2 - Y Translation

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

    # turning off the print function if v=0
    vprint = print if v>=1 else lambda *a, **k: None

    # selecting the right defocus value for the image
    if i >= ptie.num_files:
        print("i given outside range.")
        sys.exit(1)
    else:
        if ptie.num_files > 1: 
            unders = list(reversed([-1*i for i in ptie.defvals]))
            defvals = unders + [0] + ptie.defvals
            defval = defvals[i]
        else:
            defval = ptie.defvals[0]
        vprint(f'SITIE defocus: {defval} nm')

    right, left = ptie.crop['right'] , ptie.crop['left']
    bottom, top = ptie.crop['bottom'], ptie.crop['top']
    # print('mask', right, left, bottom, top)
    dim_y = bottom - top
    dim_x = right - left 

    if flipstack:
        print("Reconstructing with single flipped image.")
        image = ptie.flipstack[i].data
    else:
        image = ptie.imstack[i].data[top:bottom, left:right]


    if sym:
        print("Reconstructing with symmetrized image.")
        dim_y *= 2
        dim_x *= 2

    # setup the inverse frequency distribution
    q = dist(dim_y,dim_x)
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
    # if v >= 1:
    #     show_im(results['color_b'], "B field color, HSV colorhweel")
    
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
    fft1 = np.fft.fft2(dIdZ)  

    # applying 2/3 qc cutoff mask (see de Graef 2003)
    gy, gx = np.ogrid[-y//2:y//2, -x//2:x//2]
    rad = y/3 
    qc_mask = gy**2 + gx**2 <= rad**2
    qc_mask = np.fft.ifftshift(qc_mask)
    fft1 *= qc_mask
        
    # apply first inverse laplacian operator
    tmp1 = -1*np.fft.ifft2(fft1*qi) 
    
    # apply gradient operator and divide by in focus image
    # using kernel because np.gradient doesn't allow edge wrapping
    kx = [[0,0,0], [1/2,0,-1/2], [0,0,0]]
    ky = [[0,1/2,0], [0,0,0], [0,-1/2,0]]
    grad_y1 = scipy.signal.convolve2d(tmp1, ky, mode='same', boundary='wrap')
    grad_y1 = np.real(grad_y1/infocus)
    grad_x1 = scipy.signal.convolve2d(tmp1, kx, mode='same', boundary='wrap')
    grad_x1 = np.real(grad_x1/infocus)

    # apply second gradient operator
    # Applying laplacian directly doesn't give as good results. 
    grad_y2 = scipy.signal.convolve2d(grad_y1, ky, mode='same', boundary='wrap')
    grad_x2 = scipy.signal.convolve2d(grad_x1, kx, mode='same', boundary='wrap')
    tot = grad_y2 + grad_x2
    
    # apply second inverse Laplacian
    fft2 = np.fft.fft2(tot)
    fft2 *= qc_mask
    tmp2 = -1*np.fft.ifft2(fft2*qi)
    
    # scale
    if long_deriv:
        pre_Lap = -2*ptie.pre_Lap(pscope)  
    else:
        pre_Lap = -1*ptie.pre_Lap(pscope, defval)
        
    if sym:
        results['phase'] = np.real(pre_Lap*tmp2[:dim_y, :dim_x]) 
    else:
        results['phase'] = np.real(pre_Lap*tmp2)  

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

def print2(v=1, *args):
    """A helper print function to disable outputs if verbosity = 0"""
    if v>=1:
        print(args)
    return
