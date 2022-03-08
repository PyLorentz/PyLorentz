"""Module containing TIE and SITIE reconstruction routines.

Routines for solving the transport of intensity equation; for use with Lorentz
TEM through focal series (tfs) to reconstruct B field magnetization of the sample.

Known Bugs:

- Longitudinal derivative gives a magnetization scaling error for some
  experimental datasets.

Author: Arthur McCray, ANL, Summer 2019.
"""

import os
import sys
from pathlib import Path

import numpy as np
import scipy
import tifffile
from skimage import io

from colorwheel import color_im
from longitudinal_deriv import polyfit_deriv
from microscopes import Microscope
from TIE_helper import dist, scale_stack, select_tifs, show_im
from TIE_params import TIE_params


def TIE(
    i=-1,
    ptie=None,
    pscope=None,
    dataname="",
    sym=False,
    qc=None,
    save=False,
    hsv=True,
    long_deriv=False,
    v=1,
):
    """Sets up the TIE reconstruction and calls phase_reconstruct.

    This function calculates the necessary arrays, derivatives, etc. and then
    passes them to phase_reconstruct which solve the TIE.

    Results are not quantitatively correct if a Tikhonov filter is used.

    Args:
        i (int): Index of ptie.defvals to use for reconstruction. Default value
            is -1 which corresponds to the most defocused images for a central
            difference method derivative. i is ignored if using a longitudinal
            derivative.
        ptie (``TIE_params`` object): Object containing the images and other
            data parameters. From TIE_params.py
        pscope (``Microscope`` object): Should have same accelerating voltage
            as the microscope that took the images.
        dataname (str): The output filename to be used for saving the images.
        sym (bool): (`optional`) Fourier edge effects are marginally improved by
            symmetrizing the images before reconstructing. Default False.
        qc (float/str): (`optional`) The Tikhonov frequency to use as filter [1/nm].
            Default None. If you use a Tikhonov filter the resulting
            phase shift and induction is no longer quantitative.
        save (bool/str): Whether you want to save the output.

            ===========  ============
            input value  saved images
            ===========  ============
            True         All images
            'b'          bx, by, and color image
            'color'      Color image
            False        None
            ===========  ============

            Files will be saved as
            ptie.data_loc/images/dataname_<defval>_<key>.tiff, where <key> is
            the key for the results dictionary that corresponds to the image.
        hsv (bool): Whether to use the hsv colorwheel (True)
            or the 4-fold colorwheel (False).
        long_deriv (bool): Whether to use the longitudinal derivative (True) or
            central difference method (False). Default False.
        v (int): (`optional`) Verbosity.

            ===  ========
            v    print output
            ===  ========
            0    No output
            1    Default output
            2    Extended output for debugging.
            ===  ========

    Returns:
        dict: A dictionary of image arrays

        =========  ==============
        key        value
        =========  ==============
        'byt'      y-component of integrated magnetic induction [T*nm]
        'bxt'      x-component of integrated magnetic induction [T*nm]
        'bbt'      Magnitude of integrated magnetic induction
        'phase_b'  Magnetic phase shift (radians)
        'phase_e'  Electrostatic phase shift (if using flip stack) (radians)
        'dIdZ_m'   Intensity derivative for calculating phase_b
        'dIdZ_e'   Intensity derivative for calculating phase_e (if using flip stack)
        'color_b'  RGB image of magnetization
        'inf_im'   In-focus image
        =========  ==============
    """
    results = {
        "byt": None,
        "bxt": None,
        "bbt": None,
        "phase_e": None,
        "phase_b": None,
        "dIdZ_m": None,
        "dIdZ_e": None,
        "color_b": None,
        "inf_im": None,
    }

    # turning off the print function if v=0
    vprint = print if v >= 1 else lambda *a, **k: None
    if long_deriv:
        unders = list(reversed([-1 * ii for ii in ptie.defvals]))
        defval = unders + [0] + ptie.defvals
        if ptie.flip:
            vprint(
                "Aligning with complete longitudinal derivatives:\n",
                defval,
                "\nwith both flip/unflip tfs.",
            )
        else:
            vprint(
                "Aligning with complete longitudinal derivatives:\n",
                defval,
                "\nwith only unflip tfs.",
            )
    else:
        defval = ptie.defvals[i]
        if ptie.flip:
            vprint(
                f"Aligning for defocus value: {defval:g}, with both flip/unflip tfs."
            )
        else:
            vprint(f"Aligning for defocus value: {defval:g}, with only unflip tfs.")

    right, left = ptie.crop["right"], ptie.crop["left"]
    bottom, top = ptie.crop["bottom"], ptie.crop["top"]
    dim_y = bottom - top
    dim_x = right - left
    tifs = select_tifs(i, ptie, long_deriv)

    if sym:
        vprint("Reconstructing with symmetrized image.")
        dim_y *= 2
        dim_x *= 2

    # make the inverse laplacian, uses python implementation of IDL dist funct
    q = dist(dim_y, dim_x)
    q[0, 0] = 1
    if qc is not None and qc is not False:
        vprint("Reconstructing with Tikhonov value [1/nm]: {:}".format(qc))
        qi = q ** 2 / (q ** 2 + (qc * ptie.scale) ** 2) ** 2  # qc in 1/pix
    else:  # normal Laplacian method
        vprint("Reconstructing with normal Laplacian method")
        qi = 1 / q ** 2
    qi[0, 0] = 0
    ptie.qi = qi  # saves the freq dist

    # If rotation and translation to be applied
    if ptie.rotation != 0 or ptie.x_transl != 0 or ptie.y_transl != 0:
        rotate, x_shift, y_shift = ptie.rotation, ptie.x_transl, ptie.y_transl
        for ii in range(len(tifs)):
            tifs[ii] = scipy.ndimage.rotate(tifs[ii], rotate, reshape=False, order=0)
            tifs[ii] = scipy.ndimage.shift(tifs[ii], (-y_shift, x_shift), order=0)
        mask = scipy.ndimage.rotate(ptie.mask, rotate, reshape=False, order=0)
        mask = scipy.ndimage.shift(mask, (-y_shift, x_shift), order=0)

    # crop images and apply mask
    if ptie.rotation == 0 and ptie.x_transl == 0 and ptie.y_transl == 0:
        mask = ptie.mask[top:bottom, left:right]
    else:
        mask = mask[top:bottom, left:right]

    # crop images and apply mask
    # mask = ptie.mask[top:bottom, left:right]
    for ii in range(len(tifs)):
        tifs[ii] = tifs[ii][top:bottom, left:right]
        tifs[ii] *= mask

    # Normalizing, scaling the images
    scaled_tifs = scale_stack(tifs)
    scaled_tifs += 1e-9

    # get the infocus image
    if long_deriv and ptie.flip:
        inf_unflip = scaled_tifs[len(tifs) // 4]
        inf_flip = scaled_tifs[3 * len(tifs) // 4]
        inf_im = (inf_unflip + inf_flip) / 2
    else:
        inf_im = scaled_tifs[len(tifs) // 2]

    # Inverting masked areas on infocus image because we divide by it
    inf_im += 1 - mask
    # Make sure there are no zeros left:
    inf_im = np.where(scaled_tifs[len(tifs) // 2] == 0, 0.001, inf_im)
    results["inf_im"] = inf_im

    if v >= 2:
        print(
            """\nScaled images (+- = unflip/flip, +- = over/underfocus)
        in order [ +- , -- , 0 , ++ , -+ ]"""
        )
        for im in scaled_tifs:
            print(
                "max: {:.3f}, min: {:.2f}, total intensity: {:.4f}".format(
                    np.max(im), np.min(im), np.sum(im)
                )
            )
        print()

    # Calculate derivatives
    if long_deriv:
        # have to renormalize each stack
        unflip_stack = tifs[: ptie.num_files]
        unflip_stack = scale_stack(unflip_stack) + 1e-9
        flip_stack = tifs[ptie.num_files :]
        flip_stack = scale_stack(flip_stack) + 1e-9
        vprint("Computing the longitudinal derivative.")
        unflip_deriv = polyfit_deriv(unflip_stack, defval, v)
        if ptie.flip:
            vprint("Computing the flip stack longitudinal derivative.")
            flip_deriv = polyfit_deriv(flip_stack, defval, v)
            dIdZ_m = (unflip_deriv - flip_deriv) / 2
            dIdZ_e = (unflip_deriv + flip_deriv) / 2
        else:
            dIdZ_m = unflip_deriv

    else:  # three point derivative, default.
        if ptie.flip:
            dIdZ_m = (
                1
                / 2
                * (scaled_tifs[3] - scaled_tifs[0] - (scaled_tifs[4] - scaled_tifs[1]))
            )
            dIdZ_e = (
                1
                / 2
                * (scaled_tifs[3] - scaled_tifs[0] + (scaled_tifs[4] - scaled_tifs[1]))
            )
        else:
            dIdZ_m = scaled_tifs[2] - scaled_tifs[0]

    # Set derivatives to have 0 total "energy"
    dIdZ_m *= mask
    totm = np.sum(dIdZ_m) / np.sum(mask)
    dIdZ_m -= totm
    dIdZ_m *= mask
    results["dIdZ_m"] = dIdZ_m

    if ptie.flip:
        dIdZ_e *= mask
        tote = np.sum(dIdZ_e) / np.sum(mask)
        dIdZ_e -= tote
        dIdZ_e *= mask
        results["dIdZ_e"] = dIdZ_e

    ### Now time to call phase_reconstruct, first for E if we have a flipped tfs
    vprint("Calling TIE solver\n")
    if ptie.flip:
        resultsE = phase_reconstruct(
            ptie, inf_im, dIdZ_e, pscope, defval, sym=sym, long_deriv=long_deriv
        )
        # We only care about the E phase.
        results["phase_e"] = resultsE["phase"]

    ### Now run for B,
    resultsB = phase_reconstruct(
        ptie, inf_im, dIdZ_m, pscope, defval, sym=sym, long_deriv=long_deriv
    )
    results["byt"] = resultsB["ind_y"]
    results["bxt"] = resultsB["ind_x"]
    results["bbt"] = np.sqrt(resultsB["ind_x"] ** 2 + resultsB["ind_y"] ** 2)
    results["phase_b"] = resultsB["phase"]
    results["color_b"] = color_im(
        resultsB["ind_x"], resultsB["ind_y"], hsvwheel=hsv, background="black"
    )

    # Make black background for inf image
    results["inf_im"] = results["inf_im"] * mask

    if v >= 1:
        show_im(
            results["color_b"],
            "B-field color HSV colorwheel",
            cbar=False,
            scale=ptie.scale,
        )

    # save the images
    if save:
        save_results(
            defval, results, ptie, dataname, sym, qc, save, v, long_deriv=long_deriv
        )

    vprint("Phase reconstruction completed.")
    return results


def SITIE(
    image=None,
    defval=None,
    scale=1,
    E=200e3,
    ptie=None,
    i=-1,
    flipstack=False,
    pscope=None,
    data_loc="",
    dataname="",
    sym=False,
    qc=None,
    norm=False,
    save=False,
    v=1,
):
    """Uses a modified derivative to get the magnetic phase shift with TIE from
       a single image.

    This technique is only applicable to uniformly thin samples from which the
    only source of contrast is magnetic Fresnel contrast. All other sources of
    contrast including sample contamination, thickness variation, and diffraction
    contrast will give false magnetic inductions. For more information please
    refer to: Chess, J. J. et al. Ultramicroscopy 177, 78–83 (2018).

    This function has two ways of picking which image to use. First, if an image
    is given directly along with a defocus value, it will use that. You should
    also be sure to specify the scale of the image and accelerating voltage of
    the microscope (default 200kV).

    You can also choose to pass it an image from a ``TIE_params`` object, in which
    case you need specify only whether to choose from the imstack or flipstack
    and the index of the image to use. It's possible that in the future this
    method of selecting an image will be removed or moved to a separate function.

    Results are not quantitatively correct if a a Tikhonov filter is used.

    Args:
        image (2D array): Input image to reconstruct.
        defval (float): Defocus value corresponding to ``image``.
        scale (float): Scale (nm/pixel) corresponding to ``image``.
        E (float): Accelerating voltage of microscope that produced ``image``.
        ptie (``TIE_params`` object): Object containing the image. From
            TIE_params.py
        i (int): Index of `the ptie.imstack` or `ptie.flipstack` to
            reconstruct. This is not the defocus index like in TIE. Default
            value is -1 which corresponds to the most overfocused image.
        flipstack (bool): (`optional`) Whether to pull the image from ptie.imstack[i] or
            ptie.flipstack[i]. Default is False, calls image from imstack.
        pscope (``Microscope`` object): Should have same accelerating voltage
            as the microscope that took the images.
        dataname (str): The output filename to be used for saving the images.
        sym (bool): (`optional`) Fourier edge effects are marginally improved by
            symmetrizing the images before reconstructing. Default False.
        qc (float/str): (`optional`) The Tikhonov frequency to use as filter [1/nm].
            Default None. If you use a Tikhonov filter the resulting
            phase shift and induction is no longer quantitative.
        norm (bool): (`optional`) Normalizes the input image to [0,1]. This can
            preserve consistent outputs between images that have the same
            contrast patterns but different scales and ranges, but will also
            make the reconstruction non-quantitative.
        save (bool/str): Whether you want to save the output.

            ===========  ============
            input value  saved images
            ===========  ============
            True         All images
            'b'          bx, by, and color image
            'color'      Color image
            False        None
            ===========  ============

            Files will be saved as
            ptie.data_loc/images/dataname_<defval>_<key>.tiff, where <key> is
            the key for the returned dictionary that corresponds to the image.
        v (int): (`optional`) Verbosity.

            ===  ========
            v    print output
            ===  ========
            0    No output
            1    Default output
            2    Extended output for debugging.
            ===  ========

    Returns:
        dict: A dictionary of image arrays

        =========  ==============
        key        value
        =========  ==============
        'byt'      y-component of integrated magnetic induction [T*nm]
        'bxt'      x-component of integrated magnetic induction [T*nm]
        'bbt'      Magnitude of integrated magnetic induction
        'phase_b'  Magnetic phase shift (radians)
        'color_b'  RGB image of magnetization
        'image'    Image used for reconstruction. Saved as it might be preprocessed.
        =========  ==============
    """
    results = {"byt": None, "bxt": None, "bbt": None, "phase_b": None, "color_b": None}

    # turning off the print function if v=0
    vprint = print if v >= 1 else lambda *a, **k: None

    if image is not None and defval is not None:
        vprint(
            f"Running with given image, defval = {defval:g}nm, and scale = {scale:.3g}nm/pixel"
        )
        ptie = TIE_params(imstack=[image], defvals=[defval], data_loc=data_loc, v=0)
        ptie.set_scale(scale)
        dim_y, dim_x = image.shape
        if pscope is None:
            pscope = Microscope(E=E)
    else:
        # selecting the right defocus value for the image
        if i >= ptie.num_files:
            print("i given outside range.")
            sys.exit(1)
        else:
            if ptie.num_files > 1:
                unders = list(reversed([-1 * i for i in ptie.defvals]))
                defvals = unders + [0] + ptie.defvals
                defval = defvals[i]
            else:
                defval = ptie.defvals[0]
            vprint(f"SITIE defocus: {defval:g} nm")

        right, left = ptie.crop["right"], ptie.crop["left"]
        bottom, top = ptie.crop["bottom"], ptie.crop["top"]
        dim_y = bottom - top
        dim_x = right - left
        vprint(f"Reconstructing with ptie image {i} and defval {defval}")

        if flipstack:
            print("Reconstructing with single flipped image.")
            image = ptie.flipstack[i].data[top:bottom, left:right]
        else:
            image = ptie.imstack[i].data[top:bottom, left:right]

    if norm:  # this can mess up quantitative results
        cp = np.copy(image)
        image = (cp - cp.min()) / np.max(cp - cp.min())  # normalize image [0,1]

    if sym:
        print("Reconstructing with symmetrized image.")
        dim_y *= 2
        dim_x *= 2

    # setup the inverse frequency distribution
    q = dist(dim_y, dim_x)
    q[0, 0] = 1
    if qc is not None and qc is not False:  # add Tikhonov filter
        print("Reconstructing with Tikhonov value [1/nm]: {:}".format(qc))
        qi = q ** 2 / (q ** 2 + (qc * ptie.scale) ** 2) ** 2  # put qc in 1/pix
    else:  # normal laplacian method
        print("Reconstructing with normal Laplacian method")
        qi = 1 / q ** 2
    qi[0, 0] = 0
    ptie.qi = qi  # saves the freq dist

    # constructing "infocus" image
    infocus = np.ones(np.shape(image)) * np.mean(image)
    # calculate "derivative" and normalize
    dIdZ = 2 * (image - infocus)
    dIdZ -= np.sum(dIdZ) / np.size(infocus)

    ### Now calling the phase reconstruct in the normal way
    print("Calling SITIE solver\n")
    resultsB = phase_reconstruct(ptie, infocus, dIdZ, pscope, defval, sym=sym)
    results["byt"] = resultsB["ind_y"]
    results["bxt"] = resultsB["ind_x"]
    results["bbt"] = np.sqrt(resultsB["ind_x"] ** 2 + resultsB["ind_y"] ** 2)
    results["phase_b"] = resultsB["phase"]
    results["color_b"] = color_im(
        resultsB["ind_x"], resultsB["ind_y"], hsvwheel=True, background="black"
    )
    results["image"] = image  # save because sometimes the image will be preprocessed
    if v >= 1:
        show_im(
            results["color_b"], "B field color, HSV colorhweel", cbar=False, scale=scale
        )

    # save the images
    if save:
        save_results(
            defval, results, ptie, dataname, sym, qc, save, v, directory="SITIE"
        )
    print("Phase reconstruction completed.")
    return results


def phase_reconstruct(ptie, infocus, dIdZ, pscope, defval, sym=False, long_deriv=False):
    """The function that actually solves the TIE.

    This function takes all the necessary inputs from TIE or SITIE and solves
    the TIE using the inverse Laplacian method.

    Args:
        ptie (``TIE_params`` object): Reconstruction parameters.
        infocus (2D array): The infocus image. Should not have any zeros as
            we divide by it.
        dIdZ (2D array): The intensity derivative array.
        pscope (``Microscope`` object): Should have same accelerating voltage
            as the microscope that took the images.
        defval (float): The defocus value for the reconstruction. Not used if
            long_deriv == True.
        sym (bool): Fourier edge effects are marginally improved by
            symmetrizing the images before reconstructing. Default False.
        long_deriv (bool): Whether or not the longitudinal derivative was used.
            Only affects the prefactor.

    Returns:
        dict: A dictionary of image arrays

        =========  ==============
        key        value
        =========  ==============
        'ind_y'    y-component of integrated induction [T*nm]
        'ind_x'    x-component of integrated induction [T*nm]
        'phase'    Phase shift (radians)
        =========  ==============
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

    # Fourier transform of longitudinal derivatives
    fft1 = np.fft.fft2(dIdZ)

    # applying 2/3 qc cutoff mask (see de Graef 2003)
    gy, gx = np.ogrid[-y // 2 : y // 2, -x // 2 : x // 2]
    rad = y / 3
    qc_mask = gy ** 2 + gx ** 2 <= rad ** 2
    qc_mask = np.fft.ifftshift(qc_mask)
    fft1 *= qc_mask

    # apply first inverse Laplacian operator
    tmp1 = -1 * np.fft.ifft2(fft1 * qi)

    # apply gradient operator and divide by in focus image
    # using kernel because np.gradient doesn't allow edge wrapping
    kx = [[0, 0, 0], [1 / 2, 0, -1 / 2], [0, 0, 0]]
    ky = [[0, 1 / 2, 0], [0, 0, 0], [0, -1 / 2, 0]]
    grad_y1 = scipy.signal.convolve2d(tmp1, ky, mode="same", boundary="wrap")
    grad_y1 = np.real(grad_y1 / infocus)
    grad_x1 = scipy.signal.convolve2d(tmp1, kx, mode="same", boundary="wrap")
    grad_x1 = np.real(grad_x1 / infocus)

    # apply second gradient operator
    # Applying laplacian directly doesn't give as good results.
    grad_y2 = scipy.signal.convolve2d(grad_y1, ky, mode="same", boundary="wrap")
    grad_x2 = scipy.signal.convolve2d(grad_x1, kx, mode="same", boundary="wrap")
    tot = grad_y2 + grad_x2

    # apply second inverse Laplacian
    fft2 = np.fft.fft2(tot)
    fft2 *= qc_mask
    tmp2 = -1 * np.fft.ifft2(fft2 * qi)

    # scale
    if long_deriv:
        pre_Lap = -2 * ptie.pre_Lap(pscope)
    else:
        pre_Lap = -1 * ptie.pre_Lap(pscope, defval)

    if sym:
        results["phase"] = np.real(pre_Lap * tmp2[:dim_y, :dim_x])
    else:
        results["phase"] = np.real(pre_Lap * tmp2)

    ### getting magnetic induction
    grad_y, grad_x = np.gradient(results["phase"])
    pre_B = scipy.constants.hbar / (scipy.constants.e * ptie.scale) * 10 ** 18  # T*nm^2
    results["ind_x"] = pre_B * grad_y
    results["ind_y"] = -1 * pre_B * grad_x
    return results


def symmetrize(image):
    """Makes the even symmetric extension of an image (4x as large).

    Args:
        image (2D array): input image (M,N)

    Returns:
        ``ndarray``: Numpy array of shape (2M,2N)
    """
    sz_y, sz_x = image.shape
    dimy = 2 * sz_y
    dimx = 2 * sz_x

    imi = np.zeros((dimy, dimx))
    imi[:sz_y, :sz_x] = image
    imi[sz_y:, :sz_x] = np.flipud(image)  # *-1 for odd symm
    imi[:, sz_x:] = np.fliplr(imi[:, :sz_x])  # *-1 for odd sym
    return imi


def save_results(
    defval,
    results,
    ptie,
    dataname,
    sym,
    qc,
    save,
    v,
    directory=None,
    long_deriv=False,
    filenames=None,
):
    """Save the contents of results dictionary as 32 bit tiffs.

    This function saves the contents of the supplied dictionary (either all or
    a portion) to ptie.data_loc with the appropriate tags from results. It also
    creates a recon_params.txt file containing the reconstruction parameters.

    The images are formatted as 32 bit tiffs with the resolution included so
    they can be opened as-is by ImageJ with the scale set.

    Args:
        defval (float): The defocus value for the reconstruction. Not used if
            long_deriv == True.
        results (dict): Dictionary containing the 2D numpy arrays.
        ptie (``TIE_params`` object): Reconstruction parameters.
        dataname (str): Name attached to the saved images.
        sym (bool): If the symmetrized method was used. Only relevant as its
            included in the recon_params.txt file.
        qc (float): [1/nm] Same as sym, included in the text file.
        save (bool/str): How much of the results dictionary to save.

            ===========  ============
            input value  saved images
            ===========  ============
            True         All images
            'b'          bx, by, and color image
            'color'      Color image
            ===========  ============

            Files will be saved as
            ptie.data_loc/images/dataname_<defval>_<key>.tiff, where <key> is
            the key for the results dictionary that corresponds to the image.
        v (int): (`optional`) Verbosity.

            ===  ===============
            v    print output
            ===  ===============
            0    No output
            1    Default output
            2    Extended output, prints filenames as saving.
            ===  ===============
        directory (str): An override directory name to store the saved files. If
            None (default), saves to ptie.data_loc/Images/
        long_deriv (bool): Same as qc. Included in text file.
        filenames (list[str]): The list of filenames to save. Defaults to None, this is
            for manual file saving.

    Returns:
        None
    """
    if long_deriv:
        defval = "long"

    if v >= 1:
        print("Saving images")
    if save == "b":
        b_keys = ["bxt", "byt", "color_b"]
    elif save == "color":
        b_keys = ["color_b"]
    elif save == "manual":
        b_keys = []
        for key, value in results.items():
            for name in filenames:
                if key in name and key not in b_keys:
                    b_keys.append(key)

    res = 1 / ptie.scale
    if not dataname.endswith("_"):
        dataname += "_"

    if directory is not None:
        save_path = os.path.join(ptie.data_loc, str(directory))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.path.join(ptie.data_loc, "images")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    for key, value in results.items():
        # save either all or just some of the images
        if save == "b" or save == "color" or save == "manual":
            if key not in b_keys:
                continue
        if value is None:
            continue

        if key == "color_b":
            im = (value * 255).astype("uint8")
        else:
            im = value.astype("float32")

        save_name = f"{dataname}{defval}_{key}.tiff"
        if v >= 2:
            print(f"Saving {os.path.join(Path(save_path).absolute(), save_name)}")
        tifffile.imsave(
            os.path.join(save_path, save_name),
            im,
            imagej=True,
            resolution=(res, res),
            metadata={"unit": "nm"},
        )

    # make a txt file with parameters:
    with open(os.path.join(save_path, dataname + "recon_params.txt"), "w") as txt:
        txt.write("Reconstruction parameters for {:}\n".format(dataname[:-1]))
        txt.write("Defocus value: {} nm\n".format(defval))
        txt.write("Full E and M reconstruction: {} \n".format(ptie.flip))
        txt.write("Symmetrized: {} \n".format(sym))
        txt.write("Tikhonov filter [1/nm]: {} \n".format(qc))
        txt.write("Longitudinal derivative: {} \n".format(long_deriv))

    return


### End ###
