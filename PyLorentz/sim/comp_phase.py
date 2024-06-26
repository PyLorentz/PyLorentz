"""This module consists of functions for simulating the phase shift of a given
object.

It contained two functions:

1) linsupPhi - using the linear superposition principle for application in model
   based iterative reconstruction (MBIR) type 3D reconstruction of magnetization
   (both magnetic and electrostatic). This also includes a helper function that
   makes use of numba and just-in-time (jit) compilation.
2) mansPhi - using the Mansuripur Algorithm to compute the phase shift (only
   magnetic)

Authors: CD Phatak, Arthur McCray June 2020
"""

import numpy as np
import time
import numba
from numba import jit


@jit(nopython=True, parallel=True)
def exp_sum(mphi_k, ephi_k, inds, KY, KX, j_n, i_n, my_n, mx_n, Sy, Sx):
    """Called by linsupPhi when running with multiprocessing and numba.

    Numba incorporates just-in-time (jit) compiling and multiprocessing to numpy
    array calculations, greatly speeding up the phase-shift computation beyond
    that of pure vectorization and without the memory cost. Running this
    for the first time each session will take an additional 5-10 seconds as it is
    compiled.

    This function could be further improved by sending it to the GPU, or likely
    by other methods we haven't considered. If you have suggestions (or better
    yet, written and tested code) please email amccray@anl.gov.
    """
    for i in numba.prange(np.shape(inds)[0]):
        z = int(inds[i, 0])
        y = int(inds[i, 1])
        x = int(inds[i, 2])
        sum_term = np.exp(-1j * (KY * j_n[z, y, x] + KX * i_n[z, y, x]))
        ephi_k += sum_term
        mphi_k += sum_term * (my_n[z, y, x] * Sx - mx_n[z, y, x] * Sy)
    return ephi_k, mphi_k


def linsupPhi(
    mx=1.0,
    my=1.0,
    mz=1.0,
    Dshp=None,
    theta_x=0.0,
    theta_y=0.0,
    pre_B=1.0,
    pre_E=1,
    v=1,
    multiproc=True,
):
    """Applies linear superposition principle for 3D reconstruction of magnetic and electrostatic phase shifts.

    This function will take 3D arrays with Mx, My and Mz components of the
    magnetization, the Dshp array consisting of the shape function for the
    object (1 inside, 0 outside), and the tilt angles about x and y axes to
    compute the magnetic and the electrostatic phase shift. Initial computation
    is done in Fourier space and then real space values are returned.

    Args:
        mx (3D array): x component of magnetization at each voxel (z,y,x)
        my (3D array): y component of magnetization at each voxel (z,y,x)
        mz (3D array): z component of magnetization at each voxel (z,y,x)
        Dshp (3D array): Binary shape function of the object. Where value is 0,
            phase is not computed.
        theta_x (float): Rotation around x-axis (degrees). Rotates around x axis
            then y axis if both are nonzero.
        theta_y (float): Rotation around y-axis (degrees)
        pre_B (float): Numerical prefactor for unit conversion in calculating
            the magnetic phase shift. Units 1/pixels^2. Generally
            (2*pi*b0*(nm/pix)^2)/phi0 , where b0 is the Saturation induction and
            phi0 the magnetic flux quantum.
        pre_E (float): Numerical prefactor for unit conversion in calculating the
            electrostatic phase shift. Equal to sigma*V0, where sigma is the
            interaction constant of the given TEM accelerating voltage (an
            attribute of the microscope class), and V0 the mean inner potential.
        v (int): Verbosity. v >= 1 will print status and progress when running
            without numba. v=0 will suppress all prints.
        mp (bool): Whether or not to implement multiprocessing.

    Returns:
        tuple: Tuple of length 2: (ephi, mphi). Where ephi and mphi are 2D numpy
        arrays of the electrostatic and magnetic phase shifts respectively.
    """
    vprint = print if v >= 1 else lambda *a, **k: None

    assert mx.ndim == my.ndim == mz.ndim
    if mx.ndim == 2:
        mx = mx[None, ...]
        my = my[None, ...]
        mz = mz[None, ...]

    [dimz, dimy, dimx] = mx.shape
    dx2 = dimx // 2
    dy2 = dimy // 2
    dz2 = dimz // 2

    ly = (np.arange(dimy) - dy2) / dimy
    lx = (np.arange(dimx) - dx2) / dimx
    [Y, X] = np.meshgrid(ly, lx, indexing="ij")
    dk = 2.0 * np.pi  # Kspace vector spacing
    KX = X * dk
    KY = Y * dk
    KK = np.sqrt(KX ** 2 + KY ** 2)  # same as dist(ny, nx, shift=True)*2*np.pi
    zeros = np.where(KK == 0)  # but we need KX and KY later.
    KK[zeros] = 1.0  # remove points where KK is zero as will divide by it

    # compute S arrays (will apply constants at very end)
    inv_KK = 1 / KK ** 2
    Sx = 1j * KX * inv_KK
    Sy = 1j * KY * inv_KK
    Sx[zeros] = 0.0
    Sy[zeros] = 0.0

    # Get indices for which to calculate phase shift. Skip all pixels where
    # thickness == 0
    if Dshp is None:
        Dshp = np.ones(mx.shape)
    # exclude indices where thickness is 0, compile into list of ((z1,y1,x1), (z2,y2...
    zz, yy, xx = np.where(Dshp != 0)
    inds = np.dstack((zz, yy, xx)).squeeze()

    # Compute the rotation angles
    st = np.sin(np.deg2rad(theta_x))
    ct = np.cos(np.deg2rad(theta_x))
    sg = np.sin(np.deg2rad(theta_y))
    cg = np.cos(np.deg2rad(theta_y))

    x = np.arange(dimx) - dx2
    y = np.arange(dimy) - dy2
    z = np.arange(dimz) - dz2
    Z, Y, X = np.meshgrid(
        z, y, x, indexing="ij"
    )  # grid of actual positions (centered on 0)

    # compute the rotated values;
    # here we apply rotation about X first, then about Y
    i_n = Z * sg * ct + Y * sg * st + X * cg
    j_n = Y * ct - Z * st

    mx_n = mx * cg + my * sg * st + mz * sg * ct
    my_n = my * ct - mz * st

    # setup
    mphi_k = np.zeros(KK.shape, dtype=complex)
    ephi_k = np.zeros(KK.shape, dtype=complex)

    nelems = np.shape(inds)[0]
    stime = time.time()
    vprint(f"Beginning phase calculation for {nelems:g} voxels.")
    if multiproc:
        vprint("Running in parallel with numba.")
        ephi_k, mphi_k = exp_sum(
            mphi_k, ephi_k, inds, KY, KX, j_n, i_n, my_n, mx_n, Sy, Sx
        )

    else:
        vprint("Running on 1 cpu.")
        otime = time.time()
        vprint("0.00%", end=" .. ")
        cc = -1
        for ind in inds:
            ind = tuple(ind)
            cc += 1
            if time.time() - otime >= 15:
                vprint(f"{cc/nelems*100:.2f}%", end=" .. ")
                otime = time.time()
            # compute the expontential summation
            sum_term = np.exp(-1j * (KY * j_n[ind] + KX * i_n[ind]))
            ephi_k += sum_term
            mphi_k += sum_term * (my_n[ind] * Sx - mx_n[ind] * Sy)
        vprint("100.0%")

    vprint(
        f"total time: {time.time()-stime:.5g} sec, {(time.time()-stime)/nelems:.5g} sec/voxel."
    )
    # Now we have the phases in K-space. We convert to real space and return
    ephi_k[zeros] = 0.0
    mphi_k[zeros] = 0.0
    ephi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(ephi_k)))).real * pre_E
    mphi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(mphi_k)))).real * pre_B

    return (ephi, mphi)


def mansPhi(mx=1.0, my=1.0, mz=None, beam=[0.0, 0.0, 1.0], thick=1.0, embed=0.0):
    """Calculate magnetic phase shift using Mansuripur algorithm [1].

    Unlike the linear superposition method, this algorithm only accepts 2D
    samples. The input given is expected to be 2D arrays for mx, my, mz. It can
    compute beam angles close to (0,0,1), but for tilts greater than a few
    degrees (depending on sample conditions) it loses accuracy.

    The `embed` argument places the magnetization into a larger array to increase
    Fourier resolution, but this also seems to introduce a background phase shift
    into some images. Use at your own risk.

    Args:
        mx (2D array): x component of magnetization at each pixel.
        my (2D array): y component of magnetization at each pixel.
        mz (2D array): z component of magnetization at each pixel.
        beam (list): Vector direction of beam [x,y,z]. Default [001].
        thick (float): Thickness of the sample in pixels. i.e. thickness in nm
            divided by del_px which is nm/pixel.
        embed (int): Whether or not to embed the mx, my, mz into a larger array
            for Fourier-space computation. In theory this improves edge effects
            at the cost of reduced speed, however it also seems to add a
            background phase gradient in some simulations.

            ===========  ===========================
            embed value  effect
            ===========  ===========================
            0            Do not embed (default)
            1            Embed in (1024, 1024) array
            x (int)      Embed in (x, x) array.
            ===========  ===========================

    Returns:
        ``ndarray``: 2D array of magnetic phase shift

    References:

        1) Mansuripur, M. Computation of electron diffraction patterns in Lorentz
           electron microscopy of thin magnetic films. J. Appl. Phys. 69, 5890 (1991).

    """
    # Normalize the beam direction
    beam = np.array(beam)
    beam = beam / np.sqrt(np.sum(beam ** 2))

    # Get dimensions
    [ysz, xsz] = mx.shape

    # Embed
    if embed == 1.0:
        bdim = 1024
        bdimx, bdimy = bdim, bdim
    elif embed == 0.0:
        bdimx, bdimy = xsz, ysz
    else:
        bdim = int(embed)
        bdimx, bdimy = bdim, bdim

    bigmx = np.zeros([bdimy, bdimx])
    bigmy = np.zeros([bdimy, bdimx])
    bigmx[
        (bdimy - ysz) // 2 : (bdimy + ysz) // 2, (bdimx - xsz) // 2 : (bdimx + xsz) // 2
    ] = mx
    bigmy[
        (bdimy - ysz) // 2 : (bdimy + ysz) // 2, (bdimx - xsz) // 2 : (bdimx + xsz) // 2
    ] = my
    if mz is not None:
        bigmz = np.zeros([bdimy, bdimx])
        bigmz[
            (bdimy - ysz) // 2 : (bdimy + ysz) // 2,
            (bdimx - xsz) // 2 : (bdimx + xsz) // 2,
        ] = mz

    # Compute the auxiliary arrays requried for computation
    dsx = 2.0 * np.pi / bdimx
    linex = (np.arange(bdimx) - bdimx / 2) * dsx
    dsy = 2.0 * np.pi / bdimy
    liney = (np.arange(bdimy) - bdimy / 2) * dsy
    [Sx, Sy] = np.meshgrid(linex, liney)
    S = np.sqrt(Sx ** 2 + Sy ** 2)
    zinds = np.where(S == 0)
    S[zinds] = 1.0
    sigx = Sx / S
    sigy = Sy / S
    sigx[zinds] = 0.0
    sigy[zinds] = 0.0

    # compute FFTs of the B arrays.
    fmx = np.fft.fftshift(np.fft.fft2(bigmx))
    fmy = np.fft.fftshift(np.fft.fft2(bigmy))

    if mz is not None:
        fmz = np.fft.fftshift(np.fft.fft2(bigmz))

    # Compute vector products and Gpts
    if mz is None:  # eq 13a in Mansuripur
        if not np.array_equal(beam, [0, 0, 1]):
            print("Using a tilted beam requires a nonzero mz input")
            print("Proceeding with beam [0,0,1].")
        prod = sigx * fmy - sigy * fmx
        Gpts = 1 + 1j * 0

    else:
        e_x, e_y, e_z = beam
        prod = sigx * (
            fmy * e_x ** 2 - fmx * e_x * e_y - fmz * e_y * e_z + fmy * e_z ** 2
        ) + sigy * (fmy * e_x * e_y - fmx * e_y ** 2 + fmz * e_x * e_z - fmx * e_z ** 2)
        arg = np.pi * thick * (sigx * e_x + sigy * e_y) / e_z
        denom = 1.0 / ((sigx * e_x + sigy * e_y) ** 2 + e_z ** 2)
        qq = np.where(arg == 0)
        arg[qq] = 1
        Gpts = (denom * np.sin(arg) / arg).astype(complex)
        Gpts[qq] = denom[qq]

    # prefactor
    prefac = 1j * thick / S
    # F-space phase
    fphi = prefac * Gpts * prod
    fphi[zinds] = 0.0
    phi = np.fft.ifft2(np.fft.ifftshift(fphi)).real

    # return only the actual phase part from the embed file
    if embed != 0:
        ret_phi = phi[
            (bdimx - xsz) // 2 : (bdimx + xsz) // 2,
            (bdimy - ysz) // 2 : (bdimy + ysz) // 2,
        ]
    else:
        ret_phi = phi

    return ret_phi


### End ###
