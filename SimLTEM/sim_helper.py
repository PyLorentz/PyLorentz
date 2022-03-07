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
from matplotlib import pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import sys as sys

sys.path.append("../PyTIE/")
import os
from comp_phase import mansPhi, linsupPhi
from microscopes import Microscope
from skimage import io as skimage_io

# from TIE_helper import *
import textwrap
from itertools import takewhile
import io
from TIE_reconstruct import TIE
from skimage import io as skio
from TIE_params import TIE_params
from TIE_helper import dist, show_im
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
        skio.imsave(
            os.path.join(save_path, f"{save_name}_align.tiff"),
            im_stack.astype("float32"),
            imagej=True,
            resolution=(res, res),
            metadata={"unit": "nm"},
        )
        skio.imsave(
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


def load_ovf(file=None, sim="norm", B0=1e4, v=1):
    """Load a .ovf or .omf file of magnetization values.

    This function takes magnetization output files from OOMMF or Mumax, pulls
    some data from the header and returns 3D arrays for each magnetization
    component as well as the pixel resolutions.

    Args:
        file (string): Path to file
        sim (string): Define how the magnetization is scaled as it's read from
            the file. OOMMF writes .omf files with vectors in units of A/m,
            while Mumax writes .omf files with vectors normalized. This allows
            the reading to scale the vectors appropriately to gauss or simply
            make sure everything is normalized (as is needed for the phase
            calculation).

            - "OOMMF": Vectors scaled by mu0 and output in Tesla
            - "mumax": Vectors scaled by B0 and given those units (gauss or T)
            - "norm": (default) Normalize all vectors (does not change (0,0,0) vectors)
            - "raw": Don't do anything with the values.

        B0 (float): Saturation induction (gauss). Only relevant if sim=="mumax"
        v (int): Verbosity.

            - 0 : No output
            - 1 : Default output
            - 2 : Extended output, print full header.

    Returns:
        tuple: (mag_x, mag_y, mag_z, del_px)

        - mag_x (`2D array`) -- x-component of magnetization (units depend on `sim`).
        - mag_y (`2D array`) -- y-component of magnetization (units depend on `sim`).
        - mag_z (`2D array`) -- z-component of magnetization (units depend on `sim`).
        - del_px (float) -- Scale of datafile in y/x direction (nm/pixel)
        - zscale (float) -- Scale of datafile in z-direction (nm/pixel)
    """
    vprint = print if v >= 1 else lambda *a, **k: None

    with io.open(file, mode="r") as f:
        try:
            header = list(takewhile(lambda s: s[0] == "#", f))
        except UnicodeDecodeError:  # happens with binary files
            header = []
            with io.open(file, mode="rb") as f2:
                for line in f2:
                    if line.startswith("#".encode()):
                        header.append(line.decode())
                    else:
                        break
    if v >= 2:
        ext = os.path.splitext(file)[1]
        print(f"-----Start {ext} Header:-----")
        print("".join(header).strip())
        print(f"------End {ext} Header:------")

    dtype = None
    header_length = 0
    for line in header:
        header_length += len(line)
        if line.startswith("# xnodes"):
            xsize = int(line.split(":", 1)[1])
        if line.startswith("# ynodes"):
            ysize = int(line.split(":", 1)[1])
        if line.startswith("# znodes"):
            zsize = int(line.split(":", 1)[1])
        if line.startswith("# xstepsize"):
            xscale = float(line.split(":", 1)[1])
        if line.startswith("# ystepsize"):
            yscale = float(line.split(":", 1)[1])
        if line.startswith("# zstepsize"):
            zscale = float(line.split(":", 1)[1])
        if line.startswith("# Begin: Data Text"):
            vprint("Text file found")
            dtype = "text"
        if line.startswith("# Begin: Data Binary 4"):
            vprint("Binary 4 file found")
            dtype = "bin4"
        if line.startswith("# Begin: Data Binary 8"):
            vprint("Binary 8 file found")
            dtype = "bin8"

    if xsize is None or ysize is None or zsize is None:
        print(
            textwrap.dedent(
                f"""\
    Simulation dimensions are not given. \
    Expects keywords "xnodes", "ynodes, "znodes" for number of cells.
    Currently found size (x y z): ({xsize}, {ysize}, {zsize})"""
            )
        )
        sys.exit(1)
    else:
        vprint(f"Simulation size (z, y, x) : ({zsize}, {ysize}, {xsize})")

    if xscale is None or yscale is None or zscale is None:
        vprint(
            textwrap.dedent(
                f"""\
    Simulation scale not given. \
    Expects keywords "xstepsize", "ystepsize, "zstepsize" for scale (nm/pixel).
    Found scales (z, y, x): ({zscale}, {yscale}, {xscale})"""
            )
        )
        del_px = np.max([i for i in [xscale, yscale, 0] if i is not None]) * 1e9
        if zscale is None:
            zscale = del_px
        else:
            zscale *= 1e9
        vprint(
            f"Proceeding with {del_px:.3g} nm/pixel for in-plane and \
            {zscale:.3g} nm/pixel for out-of-plane."
        )
    else:
        assert xscale == yscale
        del_px = xscale * 1e9  # originally given in meters
        zscale *= 1e9
        if zscale != del_px:
            vprint(f"Image (x-y) scale : {del_px:.3g} nm/pixel.")
            vprint(f"Out-of-plane (z) scale : {zscale:.3g} nm/pixel.")
        else:
            vprint(f"Image scale : {del_px:.3g} nm/pixel.")

    if dtype == "text":
        data = np.genfromtxt(file)  # takes care of comments automatically
    elif dtype == "bin4":
        # for binaries it has to give count or else will take comments at end as well
        data = np.fromfile(
            file, dtype="f", count=xsize * ysize * zsize * 3, offset=header_length + 4
        )
    elif dtype == "bin8":
        data = np.fromfile(
            file, dtype="f", count=xsize * ysize * zsize * 3, offset=header_length + 8
        )
    else:
        print("Unkown datatype given. Exiting.")
        sys.exit(1)

    data = data.reshape(
        (zsize, ysize, xsize, 3)
    )  # binary data not always shaped nicely

    if sim.lower() == "oommf":
        vprint("Scaling for OOMMF datafile.")
        mu0 = 4 * np.pi * 1e-7  # output in Tesla
        data *= mu0
    elif sim.lower() == "mumax":
        vprint(f"Scaling for mumax datafile with B0={B0:.3g}.")
        data *= B0  # output is same units as B0
    elif sim.lower() == "raw":
        vprint("Not scaling datafile.")
    elif sim.lower() == "norm":
        data = data.reshape((zsize * ysize * xsize, 3))  # to iterate through vectors
        row_sums = np.sqrt(np.sum(data ** 2, axis=1))
        rs2 = np.where(row_sums == 0, 1, row_sums)
        data = data / rs2[:, np.newaxis]
        data = data.reshape((zsize, ysize, xsize, 3))
    else:
        print(
            textwrap.dedent(
                """\
        Improper argument given for sim. Please set to one of the following options:
            'oommf' : vector values given in A/m, will be scaled by mu0
            'mumax' : vectors all of magnitude 1, will be scaled by B0
            'raw'   : vectors will not be scaled."""
            )
        )
        sys.exit(1)

    mag_x = data[:, :, :, 0]
    mag_y = data[:, :, :, 1]
    mag_z = data[:, :, :, 2]

    return (mag_x, mag_y, mag_z, del_px, zscale)


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
            flip=True,
            no_mask=True,
            data_loc=directory,
            v=0,
        )
        ptie.set_scale(del_px)
    else:
        ptie = TIE_params(
            imstack=[im_un, im_in, im_ov],
            flipstack=[],
            defvals=[defval],
            flip=False,
            no_mask=True,
            data_loc=directory,
            v=0,
        )
        ptie.set_scale(del_px)

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

    return results


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


# ================================================================= #
#           Various functions for displaying vector fields          #
# ================================================================= #

# These display functions were largely hacked together, any improvements that
# work within jupyter rnotebooks would be appreciated. email: amccray@anl.gov


def show_3D(mag_x, mag_y, mag_z, a=15, ay=None, az=15, l=None, show_all=True):
    """Display a 3D vector field with arrows.

    Arrow color is determined by direction, with in-plane mapping to a HSV
    color-wheel and out of plane to white (+z) and black (-z).

    Plot can be manipulated by clicking and dragging with the mouse. a, ay, and
    az control the  number of arrows that will be plotted along each axis, i.e.
    there will be a*ay*az total arrows. In the default case a controls both ax
    and ay.

    Args:
        mag_x (3D array): (z,y,x). x-component of magnetization.
        mag_y (3D array): (z,y,x). y-component of magnetization.
        mag_z (3D array): (z,y,x). z-component of magnetization.
        a (int): Number of arrows to plot along the x-axis, if ay=None then this
            sets the y-axis too.
        ay (int): (`optional`) Number of arrows to plot along y-axis. Defaults to a.
        az (int): Number of arrows to plot along z-axis. if az > depth of array,
            az is set to 1.
        l (float): Scale of arrows. Larger -> longer arrows.
        show_all (bool):
            - True: (default) All arrows are displayed with a grey background.
            - False: Alpha value of arrows is controlled by in-plane component.
              As arrows point out-of-plane they become transparent, leaving
              only in-plane components visible. The background is black.

    Returns:
        None: None. Displays a matplotlib axes3D object.
    """
    if ay is None:
        ay = a

    bmax = max(mag_x.max(), mag_y.max(), mag_z.max())

    if l is None:
        l = mag_x.shape[1] / (2 * bmax * a)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if mag_x.ndim == 3:
        dimz, dimy, dimx = mag_x.shape
        if az > dimz:
            az = 1
        else:
            az = ((dimz - 1) // az) + 1
    else:
        dimy, dimx = mag_x.shape
        dimz = 1

    Z, Y, X = np.meshgrid(
        np.arange(0, dimz, 1),
        np.arange(0, dimy, 1),
        np.arange(0, dimx, 1),
        indexing="ij",
    )
    ay = ((dimy - 1) // ay) + 1
    axx = ((dimx - 1) // a) + 1

    # doesnt handle (0,0,0) arrows very well, so this puts in very small ones.
    zeros = ~(mag_x.astype("bool") | mag_y.astype("bool") | mag_z.astype("bool"))
    zinds = np.where(zeros)
    mag_z[zinds] = bmax / 1e5
    mag_x[zinds] = bmax / 1e5
    mag_y[zinds] = bmax / 1e5

    U = mag_x.reshape((dimz, dimy, dimx))
    V = mag_y.reshape((dimz, dimy, dimx))
    W = mag_z.reshape((dimz, dimy, dimx))

    # maps in plane direction to hsv wheel, out of plane to white (+z) and black (-z)
    phi = np.ravel(np.arctan2(V[::az, ::ay, ::axx], U[::az, ::ay, ::axx]))

    # map phi from [pi,-pi] -> [1,0]
    hue = phi / (2 * np.pi) + 0.5

    # setting the out of plane values now
    theta = np.arctan2(
        W[::az, ::ay, ::axx],
        np.sqrt(U[::az, ::ay, ::axx] ** 2 + V[::az, ::ay, ::axx] ** 2),
    )
    value = np.ravel(np.where(theta < 0, 1 + 2 * theta / np.pi, 1))
    sat = np.ravel(np.where(theta > 0, 1 - 2 * theta / np.pi, 1))

    arrow_colors = np.squeeze(np.dstack((hue, sat, value)))
    arrow_colors = colors.hsv_to_rgb(arrow_colors)

    if show_all:  # all alpha values one
        alphas = np.ones((np.shape(arrow_colors)[0], 1))
    else:  # alpha values map to inplane component
        alphas = np.minimum(value, sat).reshape(len(value), 1)
        value = np.ones(value.shape)
        sat = np.ravel(1 - abs(2 * theta / np.pi))
        arrow_colors = np.squeeze(np.dstack((hue, sat, value)))
        arrow_colors = colors.hsv_to_rgb(arrow_colors)

        ax.set_facecolor("black")
        ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
        # ax.xaxis.pane.set_edgecolor('w')
        # ax.yaxis.pane.set_edgecolor('w')
        ax.grid(False)

    # add alpha value to rgb list
    arrow_colors = np.array(
        [np.concatenate((arrow_colors[i], alphas[i])) for i in range(len(alphas))]
    )
    # quiver colors shaft then points: for n arrows c=[c1, c2, ... cn, c1, c1, c2, c2, ...]
    arrow_colors = np.concatenate((arrow_colors, np.repeat(arrow_colors, 2, axis=0)))

    # want box to be square so all arrow directions scaled the same
    dim = max(dimx, dimy, dimz)
    ax.set_xlim(0, dim)
    ax.set_ylim(0, dimy)
    if az >= dimz:
        ax.set_zlim(-dim // 2, dim // 2)
    else:
        ax.set_zlim(0, dim)
        Z += (dim - dimz) // 2

    q = ax.quiver(
        X[::az, ::ay, ::axx],
        Y[::az, ::ay, ::axx],
        Z[::az, ::ay, ::axx],
        U[::az, ::ay, ::axx],
        V[::az, ::ay, ::axx],
        W[::az, ::ay, ::axx],
        color=arrow_colors,
        length=float(l),
        pivot="middle",
        normalize=False,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def show_sims(phi, im_un, im_in, im_ov, title=None, save=None):
    """Plot phase, underfocus, infocus, and overfocus images in one plot.

    Uses same scale of intensity values for all simulated images but not phase.

    Args:
        phi (2D array): Image of phase shift of object.
        im_un (2D array): Underfocus image.
        im_in (2D array): Infocus image.
        im_ov (2D array): Overfocus image.
        save (str): (`optional`) Filepath for which to save this image.

    Returns:
        None: Displays matplotlib plot.
    """
    vmax = np.max(phi) + 0.05
    vmin = np.min(phi) - 0.05
    fig = plt.figure(figsize=(12, 3))
    ax11 = fig.add_subplot(141)
    ax11.imshow(phi, cmap="gray", origin="upper", vmax=vmax, vmin=vmin)
    plt.axis("off")
    plt.title("Phase")
    vmax = np.max(im_un) + 0.05
    vmin = np.min(im_un) - 0.05
    ax = fig.add_subplot(142)
    ax.imshow(im_un, cmap="gray", origin="upper", vmax=vmax, vmin=vmin)
    plt.axis("off")
    plt.title("Underfocus")
    ax2 = fig.add_subplot(143)
    ax2.imshow(im_in, cmap="gray", origin="upper", vmax=vmax, vmin=vmin)
    plt.axis("off")
    plt.title("In-focus")
    ax3 = fig.add_subplot(144)
    ax3.imshow(im_ov, cmap="gray", origin="upper", vmax=vmax, vmin=vmin)
    plt.axis("off")
    plt.title("Overfocus")
    if title is not None:
        fig.suptitle(str(title))

    if save is not None:
        if not (
            save.endswith(".png") or save.endswith(".tiff") or save.endswith(".tif")
        ):
            save = save + ".png"
        plt.savefig(save, dpi=300, bbox_inches="tight")

    plt.show()
    return


# ================================================================= #
#                                                                   #
#                Making vortex magnetization states                 #
#                                                                   #
# ================================================================= #


def Lillihook(dim, rad=None, Q=1, gamma=1.5708, P=1, show=False):
    """Define a skyrmion magnetization.

    This function makes a skyrmion magnetization as calculated and defined in
    [1]. It returns three 2D arrays of size (dim, dim) containing the x, y, and
    z magnetization components at each pixel.

    Args:
        dim (int): Dimension of lattice.
        rad (float): Radius parameter (see [1]). Default dim//16.
        Q (int): Topological charge.
            - 1: skyrmion
            - 2: biskyrmion
            - -1: antiskyrmion
        gamma (float): Helicity.
            - 0 or Pi: Neel
            - Pi/2 or 3Pi/2: Bloch
        P (int): Polarity (z direction in center), +/- 1.
        show (bool): (`optional`) If True, will plot the x, y, z components.

    Returns:
        tuple: (mag_x, mag_y, mag_z)

        - mag_x (``ndarray``) -- 2D Numpy array (dim, dim). x-component of
          magnetization vector.
        - mag_y (``ndarray``) -- 2D Numpy array (dim, dim). y-component of
          magnetization vector.
        - mag_z (``ndarray``) -- 2D Numpy array (dim, dim). z-component of
          magnetization vector.

    References:
        1) Lilliehöök, D., Lejnell, K., Karlhede, A. & Sondhi, S.
           Quantum Hall Skyrmions with higher topological charge.
           Phys. Rev. B 56, 6805–6809 (1997).

    """

    cx, cy = [dim // 2, dim // 2]
    cy = dim // 2
    cx = dim // 2
    if rad is None:
        rad = dim // 16
        print(f"Radius parameter set to {rad}.")
    a = np.arange(dim)
    b = np.arange(dim)
    x, y = np.meshgrid(a, b)
    x -= cx
    y -= cy
    dist = np.sqrt(x ** 2 + y ** 2)
    zeros = np.where(dist == 0)
    dist[zeros] = 1

    f = ((dist / rad) ** (2 * Q) - 4) / ((dist / rad) ** (2 * Q) + 4)
    re = np.real(np.exp(1j * gamma))
    im = np.imag(np.exp(1j * gamma))

    mag_x = -np.sqrt(1 - f ** 2) * (
        re * np.cos(Q * np.arctan2(y, x)) + im * np.sin(Q * np.arctan2(y, x))
    )
    mag_y = -np.sqrt(1 - f ** 2) * (
        -1 * im * np.cos(Q * np.arctan2(y, x)) + re * np.sin(Q * np.arctan2(y, x))
    )

    mag_z = -P * f
    mag_x[zeros] = 0
    mag_y[zeros] = 0

    if show:
        show_im(mag_x, "mag x")
        show_im(mag_y, "mag y")
        show_im(mag_z, "mag z")
        x = np.arange(0, dim, 1)
        fig, ax = plt.subplots()
        ax.plot(x, mag_z[dim // 2], label="mag_z profile along x-axis.")
        ax.set_xlabel("x-axis, y=0")
        ax.set_ylabel("mag_z")
        plt.legend()
        plt.show()
    return (mag_x, mag_y, mag_z)


def Bloch(dim, chirality="cw", pad=True, ir=0, show=False):
    """Create a Bloch vortex magnetization structure.

    Unlike Lillihook, this function does not produce a rigorously calculated
    magnetization structure. It is just a vortex that looks like some
    experimental observations.

    Args:
        dim (int): Dimension of lattice.
        chirality (str):

            - 'cw': clockwise rotation
            - 'ccw': counter-clockwise rotation.
        pad (bool): Whether or not to leave some space between the edge of the
            magnetization and the edge of the image.
        ir (float): Inner radius of the vortex in pixels.
        show (bool): If True, will show the x, y, z components in plot form.

    Returns:
        tuple: (mag_x, mag_y, mag_z)

        - mag_x (``ndarray``) -- 2D Numpy array of shape (dim, dim). x-component
          of magnetization vector.
        - mag_y (``ndarray``) -- 2D Numpy array of shape (dim, dim). y-component
          of magnetization vector.
        - mag_z (``ndarray``) -- 2D Numpy array of shape (dim, dim). z-component
          of magnetization vector.

    """
    cx, cy = [dim // 2, dim // 2]
    if pad:
        rad = 3 * dim // 8
    else:
        rad = dim // 2

    # mask
    x, y = np.ogrid[:dim, :dim]
    cy = dim // 2
    cx = dim // 2
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    circmask = r2 <= rad * rad
    circmask *= r2 >= ir * ir

    # making the magnetizations
    a = np.arange(dim)
    b = np.arange(dim)
    x, y = np.meshgrid(a, b)
    x -= cx
    y -= cy
    dist = np.sqrt(x ** 2 + y ** 2)

    mag_x = (
        -np.sin(np.arctan2(y, x))
        * np.sin(np.pi * dist / (rad - ir) - np.pi * (2 * ir - rad) / (rad - ir))
        * circmask
    )
    mag_y = (
        np.cos(np.arctan2(y, x))
        * np.sin(np.pi * dist / (rad - ir) - np.pi * (2 * ir - rad) / (rad - ir))
        * circmask
    )
    mag_x /= np.max(mag_x)
    mag_y /= np.max(mag_y)

    mag_z = (-ir - rad + 2 * dist) / (ir - rad) * circmask
    mag_z[np.where(dist < ir)] = 1
    mag_z[np.where(dist > rad)] = -1

    mag = np.sqrt(mag_x ** 2 + mag_y ** 2 + mag_z ** 2)
    mag_x /= mag
    mag_y /= mag
    mag_z /= mag

    if chirality == "ccw":
        mag_x *= -1
        mag_y *= -1

    if show:
        show_im(mag_x, "mag x")
        show_im(mag_y, "mag y")
        show_im(mag_z, "mag z")
        x = np.arange(0, dim, 1)
        fig, ax = plt.subplots()
        ax.plot(x, mag_z[dim // 2], label="mag_z profile along x-axis.")
        plt.legend()
        plt.show()
    return (mag_x, mag_y, mag_z)


def Neel(dim, chirality="io", pad=True, ir=0, show=False):
    """Create a Neel magnetization structure.

    Unlike Lillihook, this function does not produce a rigorously calculated
    magnetization structure.

    Args:
        dim (int): Dimension of lattice.
        chirality (str):

            - 'cw': clockwise rotation
            - 'ccw': counter-clockwise rotation.
        pad (bool): Whether or not to leave some space between the edge of the
            magnetization and the edge of the image.
        ir (float): Inner radius of the vortex in pixels.
        show (bool): If True, will show the x, y, z components in plot form.

    Returns:
        tuple: (mag_x, mag_y, mag_z)

        - mag_x (``ndarray``) -- 2D Numpy array of shape (dim, dim). x-component
          of magnetization vector.
        - mag_y (``ndarray``) -- 2D Numpy array of shape (dim, dim). y-component
          of magnetization vector.
        - mag_z (``ndarray``) -- 2D Numpy array of shape (dim, dim). z-component
          of magnetization vector.

    """
    cx, cy = [dim // 2, dim // 2]
    if pad:
        rad = 3 * dim // 8
    else:
        rad = dim // 2

    x, y = np.ogrid[:dim, :dim]
    x = np.array(x) - dim // 2
    y = np.array(y) - dim // 2

    circmask = circ4(dim, rad)
    circ_ir = circ4(dim, ir)
    zmask = -1 * np.ones_like(circmask) + circmask + circ_ir
    circmask -= circ_ir

    dist = dist4(dim)
    mag_y = (
        -x
        * np.sin(np.pi * dist / (rad - ir) - np.pi * (2 * ir - rad) / (rad - ir))
        * circmask
    )
    mag_x = (
        -y
        * np.sin(np.pi * dist / (rad - ir) - np.pi * (2 * ir - rad) / (rad - ir))
        * circmask
    )
    mag_x /= np.max(mag_x)
    mag_y /= np.max(mag_y)

    # b = 1
    # mag_z = (b - 2*b*dist/rad) * circmask
    mag_z = (-ir - rad + 2 * dist) / (ir - rad) * circmask

    mag_z[np.where(zmask == 1)] = 1
    mag_z[np.where(zmask == -1)] = -1

    mag = np.sqrt(mag_x ** 2 + mag_y ** 2 + mag_z ** 2)
    mag_x /= mag
    mag_y /= mag
    mag_z /= mag

    if chirality == "oi":
        mag_x *= -1
        mag_y *= -1

    if show:
        show_im(np.sqrt(mag_x ** 2 + mag_y ** 2 + mag_z ** 2), "mag")
        show_im(mag_x, "mag x")
        show_im(mag_y, "mag y")
        show_im(mag_z, "mag z")

        x = np.arange(0, dim, 1)
        fig, ax = plt.subplots()
        ax.plot(x, mag_x[dim // 2], label="x")
        ax.plot(x, -mag_y[:, dim // 2], label="y")
        ax.plot(x, mag_z[dim // 2], label="z")

        plt.legend()
        plt.show()

    return (mag_x, mag_y, mag_z)


def dist4(dim, norm=False):
    # 4-fold symmetric distance map even at small radiuses
    d2 = dim // 2
    a = np.arange(d2)
    b = np.arange(d2)
    if norm:
        a = a / (2 * d2)
        b = b / (2 * d2)
    x, y = np.meshgrid(a, b)
    quarter = np.sqrt(x ** 2 + y ** 2)
    dist = np.zeros((dim, dim))
    dist[d2:, d2:] = quarter
    dist[d2:, :d2] = np.fliplr(quarter)
    dist[:d2, d2:] = np.flipud(quarter)
    dist[:d2, :d2] = np.flipud(np.fliplr(quarter))
    return dist


def circ4(dim, rad):
    # 4-fold symmetric circle even at small dimensions
    return (dist4(dim) < rad).astype("int")


### End ###
