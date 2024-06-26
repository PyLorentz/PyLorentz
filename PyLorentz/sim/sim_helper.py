"""Helper functions for simulating LTEM images.

An assortment of helper functions broadly categorized into four sections

- Simulating images from phase shifts
- Processing micromagnetic outputs
- Helper functions for displaying vector fields
- Generating variations of magnetic vortex/skyrmion states

Known Bugs:

- Simulating images with sharp edges in the electrostatic phase shift and
  thickness maps gives images with incorrect Fresnel fringes. This can be
  resolved by applying a light filter to the electrostatic phase shift and
  thickness map before simulating images.

Author: Arthur McCray, ANL, Summer 2019.
"""

import numpy as np
import sys as sys

import os
from .comp_phase import mansPhi, linsupPhi
from PyLorentz.utils.microscopes import Microscope

import textwrap
import io
from PyLorentz.TIE.TIE_reconstruct import TIE
from PyLorentz.io.io import load_ovf
from PyLorentz.TIE.TIE_params import TIE_params
from PyLorentz.utils.utils import dist
from scipy.ndimage import rotate, gaussian_filter
import tifffile



# ================================================================= #
#                 Simulating phase shift and images                 #
# ================================================================= #


def sim_images(
    mphi=None,
    ephi=None,
    pscope=None,
    isl_shape=None,
    del_px=1,
    def_val=0,
    add_random=False,
    save_path=None,
    save_name=None,
    isl_thk=20,
    isl_xip0=50,
    mem_thk=50,
    mem_xip0=1000,
    v=1,
    Filter=True,
):
    """Simulate LTEM images for a given electron phase shift through a sample.

    This function returns LTEM images simulated for in-focus and at +/- def_val
    for comparison to experimental data and reconstruction.

    It was primarily written for simulating images of magnetic island
    structures, and as such the sample is defined in two parts: a uniform
    support membrane across the region and islands of magnetic material defined
    by an array isl_shape. The magnetization is defined with 2D arrays
    corresponding to the x- and y-components of the magnetization vector.

    There are many required parameters here that must be set to account for the
    support membrane. The default values apply to 20nm permalloy islands on a
    50nm SiN membrane window.

    There is a known bug where sharp edges in the ephi creates problems with the
    image simulations. As a workaround, this function applies a light gaussian
    filter (sigma = 1 pixel) to the ephi and isl_shape arrays. This can be
    controlled with the ``filter`` argument.

    Args:
        mphi (2D array): Numpy array of size (M, N), magnetic phase shift
        ephi (2D array): Numpy array of size (M, N), Electrostatic phase shift
        pscope (``Microscope`` object): Contains all microscope parameters
            as well as methods for propagating electron wave functions.
        isl_shape (2D/3D array): Array of size (z,y,x) or (y,x). If 2D the
            thickness will be taken as the isl_shape values multiplied by
            isl_thickness. If 3D, the isl_shape array will be summed along
            the z-axis becoming and multiplied by isl_thickness.
            (Default) None -> uniform flat material with thickness = isl_thk.
        del_px (Float): Scale factor (nm/pixel). Default = 1.
        def_val (Float): The defocus values at which to calculate the images.
        add_random (Float): Whether or not to add amorphous background to
            the simulation. True or 1 will add a default background, any
            other value will be multiplied to scale the additional phase term.
        save_path: String. Will save a stack [underfocus, infocus, overfocus]
            as well as (mphi+ephi) as tiffs along with a params.text file.
            (Default) None -> Does not save.
        save_name (str): Name prepended to saved files.
        v (int): Verbosity. Set v=0 to suppress print statements.
        isl_thk (float): Island thickness (nm). Default 20 (nm).
        isl_xip0 (float): Island extinction distance (nm). Default 50 (nm).
        mem_thk (float): Support membrane thickness (nm). Default 50 (nm).
        mem_xip0 (float): Support membrane extinction distance (nm). Default
            1000 (nm).
        Filter (Bool): Apply a light gaussian filter to ephi and isl_shape.

    Returns:
        tuple: (Tphi, im_un, im_in, im_ov)

        - Tphi (`2D array`) -- Numpy array of size (M,N). Total electron phase shift (ephi+mphi).
        - im_un (`2D array`) -- Numpy array of size (M,N). Simulated image at delta z = -def_val.
        - im_in (`2D array`) -- Numpy array of size (M,N). Simulated image at zero defocus.
        - im_ov (`2D array`) -- Numpy array of size (M,N). Simulated image at delta z = +def_val.

    """
    vprint = print if v >= 1 else lambda *a, **k: None

    if Filter:
        ephi = gaussian_filter(ephi, sigma=1)

    Tphi = mphi + ephi
    vprint(
        f"Total fov is ({np.shape(Tphi)[1]*del_px:.3g},{np.shape(Tphi)[0]*del_px:.3g}) nm"
    )
    dy, dx = Tphi.shape

    if add_random:
        if type(add_random) == bool:
            add_random = 1.0
        ran_phi = np.random.uniform(
            low=-np.pi / 128 * add_random, high=np.pi / 128 * add_random, size=[dy, dx]
        )
        if np.max(ephi) > 1:  # scale by ephi only if it's given and relevant
            ran_phi *= np.max(ephi)
        Tphi += ran_phi

    # amplitude
    if isl_shape is None:
        thk_map = np.ones(Tphi.shape) * isl_thk
    else:
        if type(isl_shape) != np.ndarray:
            isl_shape = np.array(isl_shape)
        if isl_shape.ndim == 3:
            thk_map = np.sum(isl_shape, axis=0) * isl_thk
        elif isl_shape.ndim == 2:
            thk_map = isl_shape * isl_thk
        else:
            vprint(
                textwrap.dedent(
                    f"""
                Mask given must be 2D (y,x) or 3D (z,y,x) array.
                It was given as a {isl_shape.ndim} dimension array."""
                )
            )
            sys.exit(1)
        if Filter:
            thk_map = gaussian_filter(thk_map, sigma=1)

    Amp = np.exp((-np.ones([dy, dx]) * mem_thk / mem_xip0) - (thk_map / isl_xip0))
    if np.min(Amp) == np.max(Amp) and np.max(Amp) < 0.01:
        Amp = np.ones_like(Amp)
    ObjWave = Amp * (np.cos(Tphi) + 1j * np.sin(Tphi))

    # compute unflipped images
    qq = dist(dy, dx, shift=True)
    pscope.defocus = 0.0
    im_in = pscope.getImage(ObjWave, qq, del_px)
    pscope.defocus = -def_val
    im_un = pscope.getImage(ObjWave, qq, del_px)
    pscope.defocus = def_val
    im_ov = pscope.getImage(ObjWave, qq, del_px)

    if save_path is not None:
        vprint(f"saving to {save_path}")
        im_stack = np.array([im_un, im_in, im_ov])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        res = 1 / del_px
        tifffile.imsave(
            os.path.join(save_path, f"{save_name}_align.tiff"),
            im_stack.astype("float32"),
            imagej=True,
            resolution=(res, res),
            metadata={"unit": "nm"},
        )
        tifffile.imsave(
            os.path.join(save_path, f"{save_name}_phase.tiff"),
            Tphi.astype("float32"),
            imagej=True,
            resolution=(res, res),
            metadata={"unit": "nm"},
        )

        with io.open(os.path.join(save_path, f"{save_name}_params.txt"), "w") as text:
            text.write(f"def_val (nm) \t{def_val:g}\n")
            text.write(f"del_px (nm/pix) \t{del_px:g}\n")
            text.write(f"scope En. (V) \t{pscope.E:g}\n")
            text.write(f"im_size (pix) \t({dy:g},{dx:g})\n")
            text.write(f"sample_thk (nm) \t{isl_thk:g}\n")
            text.write(f"sample_xip0 (nm) \t{isl_xip0:g}\n")
            text.write(f"mem_thk (nm) \t{mem_thk:g}\n")
            text.write(f"mem_xip0 (nm) \t{mem_xip0:g}\n")
            text.write(f"add_random \t{add_random:g}\n")

    return (Tphi, im_un, im_in, im_ov)


def std_mansPhi(
    mag_x=None,
    mag_y=None,
    mag_z=None,
    zscale=1,
    del_px=1,
    isl_shape=None,
    pscope=None,
    b0=1e4,
    isl_thk=20,
    isl_V0=20,
    mem_thk=50,
    mem_V0=10,
    add_random=None,
):
    """Calculates the electron phase shift through a given 2D magnetization.

    This function was originally created for simulating LTEM images of magnetic
    island structures, and it is kept as an example of how to set up and use the
    mansPhi function. It defines the sample in two parts: a uniform membrane
    across the region and an island structure defined by isl_shape. The
    magnetization is defined with 2D arrays corresponding to the x- and y-
    components of the magnetization vector.

    The magnetic phase shift is calculated using the Mansuripur algorithm (see
    comp_phase.py), and the electrostatic phase shift is computed directly from
    the materials parameters and geometry given.

    Args:
        mag_x (2D/3D Array): X-component of the magnetization at each pixel.
        mag_y (2D/3D Array): Y-component of the magnetization at each pixel.
        mag_z (2D/3D Array): Z-component of the magnetization at each pixel.
        isl_shape (2D/3D array): Array of size (z,y,x) or (y,x). If 2D the
            thickness will be taken as the isl_shape values multiplied by
            isl_thickness. If 3D, the isl_shape array will be summed along
            the z-axis becoming and multiplied by isl_thickness.
            (Default) None -> uniform flat material with thickness = isl_thk.
        zscale (float): Scale factor (nm/pixel) along beam direction.
        del_px (float): Scale factor (nm/pixel) along x/y directions.
        pscope (``Microscope`` object): Accelerating voltage is the only
            relevant parameter here.
        b0 (float): Saturation induction (gauss). Default 1e4.
        isl_thk (float): Island thickness (nm). Default 20.
        isl_V0 (float): Island mean inner potential (V). Default 20.
        mem_thk (float): Support membrane thickness (nm). Default 50.
        mem_V0 (float): Support membrane mean inner potential (V). Default 10.
        add_random (float): Account for the random phase shift of the amorphous
            membrane. Phase shift will scale with mem_V0 and mem_thk, but the
            initial factor has to be set by hand. True -> 1/32, which is
            a starting place that works well for some images.

    Returns:
        tuple: (ephi, mphi)

        - ephi (`2D array`) -- Numpy array of size (M,N). Electrostatic phase
          shift.
        - mphi (`2D array`) -- Numpy array of size (M,N). Magnetic phase shift.

    """
    if pscope is None:
        pscope = Microscope(E=200e3)

    dim_z = 1
    if type(mag_x) != np.ndarray:
        mag_x = np.array(mag_x)
        mag_y = np.array(mag_y)
    if mag_x.ndim == 3:
        dim_z = np.shape(mag_x)[0]
        mag_x = np.sum(mag_x, axis=0)
        mag_y = np.sum(mag_y, axis=0)
        if mag_z is not None:
            mag_z = np.sum(mag_z, axis=0)

    thk2 = isl_thk / zscale  # thickness in pixels
    phi0 = 2.07e7  # Gauss*nm^2 flux quantum
    pre_B = 2 * np.pi * b0 * zscale * del_px / (thk2 * phi0)

    # calculate magnetic phase shift with mansuripur algorithm
    mphi = mansPhi(mx=mag_x, my=mag_y, thick=thk2) * pre_B

    # and now electric phase shift
    if isl_shape is None:
        thk_map = np.ones(mag_x.shape) * isl_thk
    else:
        if type(isl_shape) != np.ndarray:
            isl_shape = np.array(isl_shape)
        if isl_shape.ndim == 3:
            thk_map = np.sum(isl_shape, axis=0) * zscale
        elif isl_shape.ndim == 2:
            thk_map = isl_shape * isl_thk
        else:
            print(
                textwrap.dedent(
                    f"""
                Mask given must be 2D (y,x) or 3D (y,x,z) array.
                It was given as a {isl_shape.ndim} dimension array."""
                )
            )
            sys.exit(1)

    if add_random is None:
        ephi = pscope.sigma * (
            thk_map * isl_V0 + np.ones(mag_x.shape) * mem_thk * mem_V0
        )
    else:
        if type(add_random) == bool:
            add_random = 1 / 64
        ran_phi = (
            np.random.uniform(low=-np.pi, high=np.pi, size=mag_x.shape) * add_random
        )
        ephi = pscope.sigma * (
            thk_map * isl_V0 + ran_phi * np.ones(mag_x.shape) * mem_thk * mem_V0
        )

    ephi -= np.sum(ephi) / np.size(ephi)
    return (ephi, mphi)


# ================================================================= #
#            Simulating images from micromagnetic output            #
# ================================================================= #


def make_thickness_map(mag_x=None, mag_y=None, mag_z=None, file=None, D3=True):
    """Define a 2D thickness map where magnetization is 0.

    Island structures or empty regions are often defined in micromagnetic
    simulations as regions with (0,0,0) magnetization. This function creates an
    array where those values are 0, and 1 otherwise. It then sums along the z
    direction to make a 2D map.

    It can take inputs either as magnetization components or a filename which
    it will read with the load_ovf() function.

    Args:
        mag_x (3D array): Numpy array of x component of the magnetization.
        mag_y (3D array): Numpy array of y component of the magnetization.
        mag_z (3D array): Numpy array of z component of the magnetization.
        file (str): Path to .ovf or .omf file.
        D3 (bool): Whether or not to return the 3D map.

    Returns:
        ``ndarray``: 2D array of thickness values scaled to total thickness.
        i.e. 0 corresponds to 0 thickness and 1 to zscale*zdim.
    """
    if file is not None:
        mag_x, mag_y, mag_z, del_px, zscale = load_ovf(file, sim="norm", v=0)
    elif mag_z is None:
        mag_z = np.zeros(mag_x.shape)
    if D3 and len(mag_x.shape) == 2:
        mag_x = np.expand_dims(mag_x, axis=0)
        mag_y = np.expand_dims(mag_y, axis=0)
        mag_z = np.expand_dims(mag_z, axis=0)

    nonzero = (
        mag_x.astype("bool") | mag_y.astype("bool") | mag_z.astype("bool")
    ).astype(float)
    if len(mag_x.shape) == 3:
        if D3:
            return nonzero
        zdim = mag_x.shape[0]
        thk_map = np.sum(nonzero, axis=0)
    else:
        assert len(mag_x.shape) == 2
        thk_map = nonzero
    return thk_map


def reconstruct_ovf(
    file=None,
    savename=None,
    save=1,
    v=1,
    flip=True,
    thk_map=None,
    pscope=None,
    defval=0,
    theta_x=0,
    theta_y=0,
    B0=1e4,
    sample_V0=10,
    sample_xip0=50,
    mem_thk=50,
    mem_xip0=1000,
    add_random=0,
    sym=False,
    qc=None,
    method="mans",
    precompile=False,
):
    """Load a micromagnetic output file and reconstruct simulated LTEM images.

    This is an "all-in-one" function that takes a magnetization datafile,
    material parameters, and imaging conditions to simulate LTEM images and
    reconstruct them.

    The image simulation step uses the Mansuripur algorithm [1] for calculating
    the phase shift if theta_x and theta_y == 0, as it is computationally very
    efficient. For nonzero tilts it employs the linear superposition method for
    determining phase shift, which allows for 3d magnetization inputs and robust
    tilting the sample. A substrate can be accounted for as well, though it is
    assumed to be uniform and non-magnetic, i.e. applying a uniform phase shift.

    Imaging parameters are defined by the defocus value, tilt angles, and
    microscope object which contains accelerating voltage, aberrations, etc.

    Args:
        file (str): Path to file.
        savename (str): Name prepended to saved files. If None -> filename
        save (int): Integer value that sets which files are saved.

            - 0 -- Saves nothing, still returns results.
            - 1 -- (default) Saves simulated images, simulated phase shift, and
              reconstructed magnetizations, both the color image and x/y
              components.
            - 2 -- Saves simulated images, simulated phase shift, and all
              reconstruction TIE images.

        v (int): Integer value which sets verbosity

            - 0: All output suppressed.
            - 1: Default prints and final reconstructed image displayed.
            - 2: Extended output. Prints full datafile header, displays simulated tfs.

        flip (bool): Whether to use a single through focal series (tfs) (False)
            or two (True), one for sample in normal orientation and one with it
            flipped in the microscope. Samples that are not uniformly thin
            require flip=True.
        thk_map (2D/3D array): Numpy array of same (y,x) shape as magnetization
            arrays. Binary shape function of the sample, 1 where the sample is
            and 0 where there is no sample. If a 2D array is given it will be
            tiled along the z-direction to make it the same size as the
            magnetization arrays.

            The make_thickness_map() function can be useful for island
            structures or as a guide of how to define a thickness map.

            Default None -> Uniform thickness, equivalent to array of 1's.

        pscope (``Microscope`` object): Contains all microscope parameters such
            as accelerating voltage, aberrations, etc., along with the methods to
            propagate the electron wave function.
        def_val (float): The defocus values at which to calculate the images.
        theta_x (float): Rotation around x-axis (degrees). Default 0. Rotates
            around x axis then y axis if both are nonzero.
        theta_y (float): Rotation around y-axis (degrees). Default 0.
        B0 (float): Saturation induction (gauss).
        sample_V0 (float): Mean inner potential of sample (V).
        sample_xip0 (float): Extinction distance (nm).
        mem_thk (float): Support membrane thickness (nm). Default 50.
        mem_xip0 (float): Support membrane extinction distance (nm). Default 1000.
        add_random: (float): Whether or not to add amorphous background to
            the simulation. True or 1 will add a default background, any
            other value will be multiplied to scale the additional phase term.
        sym (bool): (`optional`) Fourier edge effects are marginally improved by
            symmetrizing the images before reconstructing, but the process is
            more computationally intensive as images are 4x as large.
            (default) False.
        qc (float): (`optional`) The Tikhonov frequency to use as a filter [1/nm],
            (default) None. If you use a Tikhonov filter the resulting
            magnetization is no longer quantitative
        method (str): (`optional`) Method of phase calculation to use if theta_x == 0 and
            theta_y == 0. If either are nonzero then the linear superposition
            method will be used.

            - "Mans" : Use Mansuripur algorithm (default)
            - "Linsup" : Use linear superposition method
        precompile (bool): (`optional`) Only relevant if using the linear superposition
            method of calculating phase and for the first time the phase is calculated
            for a given kernel. The first run requires compiling the function, which can
            be slow. This performs an initial run on a small image which can be faster
            for some situations.

    Returns:
        tuple: (dict, TIE_params)
        dict: A dictionary of image arrays

        =============  =========================================================
        key            value
        =============  =========================================================
        'byt'          y-component of integrated magnetic induction
        'bxt'          x-component of integrated magnetic induction
        'bbt'          Magnitude of integrated magnetic induction
        'phase_b_sim'  Simulated magnetic phase shift
        'phase_e_sim'  Simulated electrostatic phase shift
        'phase_b'      Magnetic phase shift (radians)
        'phase_e'      Electrostatic phase shift (if using flip stack) (radians),
        'dIdZ_m'       Intensity derivative for calculating phase_b
        'dIdZ_e'       Intensity derivative for calculating phase_e (if using
                       flip stack)
        'color_b'      RGB image of magnetization
        'inf_im'       In-focus image
        'mag_x'        x-component of input magnetization
        'mag_y'        y-component of input magnetization
        'mag_z'        z-component of input magnetization
        =============  =========================================================
    """
    vprint = print if v >= 1 else lambda *a, **k: None
    directory, filename = os.path.split(file)
    directory = os.path.abspath(directory)
    if savename is None:
        savename = os.path.splitext(filename)[0]
    elif savename[0] == "/":
        savename = savename[1:]
    if save < 1:
        save_path_tfs = None
        TIE_save = False
    else:
        save_path_tfs = os.path.join(directory, "sim_tfs")
        if save < 2:
            TIE_save = "b"
        else:
            TIE_save = True

    cache_found = False
    # TODO implement for caching of the phase shift
    # if not flip:
    #     phase_path = os.path.join(save_path_tfs, f'{savename}_phase.tiff')
    #     if os.path.isfile(phase_path):
    #         vprint("Found phase file, loading and using it.")
    #         mphi = skimage_io.imread(phase_path)
    #         ephi = np.ones_like(mphi)
    #         cache_found = True

    mag_x, mag_y, mag_z, del_px, zscale = load_ovf(file, sim="norm", v=v, B0=B0)
    (zsize, ysize, xsize) = mag_x.shape

    if thk_map is not None:
        if type(thk_map) != np.ndarray:
            thk_map = np.array(thk_map)
        if thk_map.ndim == 3:
            thk_map_3D = thk_map
            thk_map_2D = np.sum(thk_map, axis=0)
            thickness_nm = np.max(thk_map_2D) * zscale
        elif thk_map.ndim == 2:
            thk_map_3D = np.tile(thk_map, (zsize, 1, 1))
            thickness_nm = zscale * zsize
    else:
        thk_map_3D = None
        thickness_nm = zscale * zsize

    if not cache_found:
        if theta_x == 0 and theta_y == 0 and method.lower() == "mans":
            vprint("Calculating phase shift with Mansuripur algorithm. ")
            ephi, mphi = std_mansPhi(
                mag_x,
                mag_y,
                mag_z,
                zscale=zscale,
                isl_thk=thickness_nm,
                del_px=del_px,
                isl_shape=thk_map_3D,
                pscope=pscope,
                b0=B0,
                isl_V0=sample_V0,
            )
        else:
            vprint("Calculating phase shift with the linear superposition method.")
            # define numerical prefactors for phase shift calculation
            phi0 = 2.07e7  # Gauss*nm^2
            pre_B = 2 * np.pi * B0 / phi0 * zscale * del_px  # 1/px^2
            pre_E = pscope.sigma * sample_V0 * zscale  # 1/px

            if precompile:
                print("Precompiling linsupPhi")
                _ephi, _mphi = linsupPhi(
                    mx=np.ones((1, 16, 16)),
                    my=np.ones((1, 16, 16)),
                    mz=np.ones((1, 16, 16)),
                    v=0,
                )

            # calculate phase shifts with linear superposition method
            ephi, mphi = linsupPhi(
                mx=mag_x,
                my=mag_y,
                mz=mag_z,
                Dshp=thk_map_3D,
                v=v,
                theta_x=theta_x,
                theta_y=theta_y,
                pre_B=pre_B,
                pre_E=pre_E,
            )

    sim_name = savename
    # adjust thickness to account for rotation for sample, not taken care of
    # natively when simming the images like it is for phase computation.
    if thk_map_3D is not None:
        if np.max(thk_map_3D) != np.min(thk_map_3D):
            thk_map_tilt, isl_thk_tilt = rot_thickness_map(
                thk_map_3D, -1 * theta_x, theta_y, zscale
            )
        else:  # it's a uniform thickness map, assuming infinitely large to avoid
            # the edge effects, setting to none.
            thk_map_tilt = None
            isl_thk_tilt = thickness_nm
    else:
        thk_map_tilt = None
        isl_thk_tilt = thickness_nm

    if flip:
        sim_name = savename + "_flip"
        Tphi_flip, im_un_flip, im_in_flip, im_ov_flip = sim_images(
            mphi=-1 * mphi,
            ephi=ephi,
            isl_shape=thk_map_tilt,
            pscope=pscope,
            del_px=del_px,
            def_val=defval,
            add_random=add_random,
            isl_thk=isl_thk_tilt,
            isl_xip0=sample_xip0,
            mem_thk=mem_thk,
            mem_xip0=mem_xip0,
            v=v,
            save_path=save_path_tfs,
            save_name=sim_name,
        )
        sim_name = savename + "_unflip"

    Tphi, im_un, im_in, im_ov = sim_images(
        mphi=mphi,
        ephi=ephi,
        isl_shape=thk_map_tilt,
        pscope=pscope,
        del_px=del_px,
        def_val=defval,
        add_random=add_random,
        isl_thk=isl_thk_tilt,
        isl_xip0=sample_xip0,
        mem_thk=mem_thk,
        mem_xip0=mem_xip0,
        v=0,
        save_path=save_path_tfs,
        save_name=sim_name,
    )

    if v >= 2:
        show_sims(Tphi, im_un, im_in, im_ov, title="Simulated Unflipped Images")
        if flip:
            show_sims(
                Tphi_flip,
                im_un_flip,
                im_in_flip,
                im_ov_flip,
                title="Simulated Flipped Images",
            )

    if flip:
        ptie = TIE_params(
            imstack=[im_un, im_in, im_ov],
            flipstack=[im_un_flip, im_in_flip, im_ov_flip],
            defvals=[defval],
            scale=del_px,
            flip=True,
            data_loc=directory,
            no_mask=True,
            v=0,
        )
    else:
        ptie = TIE_params(
            imstack=[im_un, im_in, im_ov],
            flipstack=[],
            defvals=[defval],
            scale=del_px,
            flip=False,
            data_loc=directory,
            no_mask=True,
            v=0,
        )

    results = TIE(
        i=0,
        ptie=ptie,
        pscope=pscope,
        dataname=savename,
        sym=sym,
        qc=qc,
        save=TIE_save,
        v=v,
    )

    results["phase_b_sim"] = mphi
    results["phase_e_sim"] = ephi
    results["mag_x"] = mag_x
    results["mag_y"] = mag_y
    results["mag_z"] = mag_z

    return results, ptie


def rot_thickness_map(thk_map_3D=None, theta_x=0, theta_y=0, zscale=None):
    """Tilt a thickness map around the x and y axis.

    While the phase calculation takes care of tilting the sample in the algorithm,
    image simualtions require a thickness map to calculate the wave function
    amplitude, and this must be rotated according to theta_x and theta_y.

    This function returns a rotated thickness map of the same shape inputted as
    well as the thickness scaling factor.

    This function only works for high tilt angles when zscale = del_px.

    Args:
        thk_map_3D (3D array): Numpy array of shape (z,y,x) shape to be rotated.
        theta_x (float): Rotation around x-axis (degrees). Rotates around x axis
            then y axis if both are nonzero.
        theta_y (float): Rotation around y-axis (degrees).
        zscale (float): Scaling (nm/pixel) in z direction.

    Returns:
        tuple: (thk_map_tilt, isl_thickness_tilt)

        - thk_map_tilt (``ndarray``) -- 2D Numpy array, tilted thickness map
          normalized to values [0,1]
        - isl_thickness_tilt (float) -- scale factor (nm) for thk_map_tilt.
          Multiplying thk_map_tilt * isl_thickness_tilt gives the thickness map
          in nanometers.

    """
    dim_z, dim_y, dim_x = thk_map_3D.shape
    # rotate the thickness map around x then y
    # pad first with 0's so edges scale correctly in the rotate function.
    thk_map_pad = np.pad(thk_map_3D, pad_width=20, mode="constant")
    tilt1 = rotate(thk_map_pad, theta_x, axes=(0, 1))
    tilt2 = rotate(tilt1, theta_y, axes=(0, 2)).sum(axis=0)

    # tilted region might be larger (in one dimension) than original region. crop.
    t2y, t2x = tilt2.shape
    if t2y > dim_y:
        tilt2 = tilt2[(t2y - dim_y) // 2 : t2y // 2 + dim_y // 2]
    if t2x > dim_x:
        tilt2 = tilt2[:, t2x // 2 - dim_x // 2 : t2x // 2 + dim_x // 2]

    # and it might be smaller, in that case pad.
    pad_y = (dim_y - tilt2.shape[0]) / 2
    p1 = int(np.floor(pad_y))
    p2 = int(np.ceil(pad_y))
    pad_x = (dim_x - tilt2.shape[1]) / 2
    p3 = int(np.floor(pad_x))
    p4 = int(np.ceil(pad_x))
    thk_map_tilt = np.pad(tilt2, ((p1, p2), (p3, p4)))

    isl_thickness_tilt = np.max(thk_map_tilt * zscale)
    thk_map_tilt = thk_map_tilt / np.max(thk_map_tilt)

    return thk_map_tilt, isl_thickness_tilt



### End ###
