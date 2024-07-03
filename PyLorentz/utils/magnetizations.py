
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from ..visualize.show import show_im
from .utils import circ4, dist4


def hopfion(dim=128,dimz=None, R=None, H=None, wr=None, wh=None, type="bloch", Q=1):
    """Magnetization pattern for a hopfion with Hopf index +/-

    Args:
        dim (int): xy dimension. Defaults to 128.
        dimz (int, optional): z dimension. Defaults to dim/4.
        R (float, optional): Hopfion radius. Defaults to dim*0.15.
        H (float, optional): Hopfion height. Defaults to dim*0.08.
        wr (float, optional): Hopfion radial domain wall parameter. Defaults to dim*0.08.
        wh (float, optional): Hopfion z-direction domain wall parameter. Defaults to dim*0.04.
        type (str, optional): Hopfion type, "bloch" or "neel". Defaults to "bloch".
        Q (int, optional): Hopf charge, +/- 1. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # Wang, X. S., Qaiumzadeh, A. & Brataas, A. Current-Driven Dynamics of Magnetic Hopfions. Phys. Rev. Lett. 123, 147203 (2019).

    if R is None:
        R = dim/128*20
    if wr is None:
        wr = dim/128*10
    if dimz is None:
        dimz = dim//4
    if H is None:
        H = dim/128*10
    if wh is None:
        wh = dim/128*5

    x_ = np.linspace(0, dim - 1, dim) - (dim - 1) / 2
    y_ = np.linspace(0, dim - 1, dim) - (dim - 1) / 2
    z_ = np.linspace(0, dimz - 1, dimz) - (dimz - 1) / 2
    z, y, x = np.meshgrid(z_, y_, x_, indexing="ij")
    r = np.sqrt(x**2 + y**2)

    rp = (np.exp(r/wr)-1) / (np.exp(R/wr)-1)
    zp = np.abs(z)/z *  (np.exp(np.abs(z)/wh)-1) / (np.exp(np.abs(H)/wh)-1)

    if type.lower() == "bloch":
        if Q == -1:
            rp = (np.exp(R/wr)-1) / (np.exp(r/wr)-1)
            mx = (4*rp*(-2*zp*(x/r) - (y/r)*(rp**2+zp**2-1))) / (1+rp**2+zp**2)**2
            my = (4*rp*(-2*zp*(y/r) + (x/r)*(rp**2+zp**2-1))) / (1+rp**2+zp**2)**2
            mz = 1 - (8*(rp**2)) / (1+rp**2+zp**2)**2
        elif Q == +1:
            rp = (np.exp(r/wr)-1) / (np.exp(R/wr)-1)
            mx = -1*(4*rp*(-2*zp*(x/r) - (y/r)*(rp**2+zp**2-1))) / (1+rp**2+zp**2)**2
            my = -1*(4*rp*(-2*zp*(y/r) + (x/r)*(rp**2+zp**2-1))) / (1+rp**2+zp**2)**2
            mz = 1 - (8*(rp**2)) / (1+rp**2+zp**2)**2
    elif type.lower() == "neel":
        rp = (np.exp(R/wr)-1) / (np.exp(r/wr)-1)
        if Q == 1:
            mx = (4*rp*(-2*zp*(y/r) - (x/r)*(rp**2+zp**2-1))) / (1+rp**2+zp**2)**2
            my = -1*(4*rp*(-2*zp*(x/r) + (y/r)*(rp**2+zp**2-1))) / (1+rp**2+zp**2)**2
            mz = 1 - (8*(rp**2)) / (1+rp**2+zp**2)**2
        else:
            mx = -1*(4*rp*(-2*zp*(y/r) + (x/r)*(rp**2+zp**2-1))) / (1+rp**2+zp**2)**2
            my = (4*rp*(-2*zp*(x/r) - (y/r)*(rp**2+zp**2-1))) / (1+rp**2+zp**2)**2
            mz = 1 - (8*(rp**2)) / (1+rp**2+zp**2)**2

    mags = np.sqrt(mx**2 + my**2 + mz**2)
    assert np.allclose(mags, np.ones_like(mags))

    return np.array([mz, my, mx])


def hopfion_cylinder(L=32, dim=None, pad=None, background="none"):
    """From Suctcliffe: Hopfions in chiral magnets
    L is height, 3L is radius, padded with 2L, so total dims (L, 8L, 8L)

    this is an in-plane hopfion of sorts, with an extra winding around the outside as it
    is confined to a patterned cylinder.
    """

    def Omega(z, L):
        return np.tan(np.pi * z / L)

    def Xi(z, rho, L):
        a = 1 + (2 * z / L) ** 2
        b = 1 / (np.cos(np.pi * rho / (2 * L)) * L)
        return a * b

    def Lambda(z, rho, L):
        return Xi(z, rho, L) ** 2 * rho**2 + Omega(z, L) ** 2 / 4

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi * -1)

    # setup coordinates, z, rho, theta
    if dim is None:
        dimz = L
        dimy = dimx = 3 * L

    x_ = np.linspace(0, dimx - 1, dimx) - (dimx - 1) / 2
    y_ = np.linspace(0, dimy - 1, dimy) - (dimy - 1) / 2
    z_ = np.linspace(0, dimz - 1, dimz) - (dimz - 1) / 2

    z, y, x = np.meshgrid(z_, y_, x_, indexing="ij")
    rho, theta = cart2pol(x, y)

    mz = 1 - (8 * Xi(z, rho, L) ** 2 * rho**2) / ((1 + Lambda(z, rho, L)) ** 2)

    prefac = (4 * Xi(z, rho, L) * rho) / ((1 + Lambda(z, rho, L)) ** 2)

    my = prefac * (
        Omega(z, L) * np.sin(theta) + (Lambda(z, rho, L) - 1) * np.cos(theta)
    )

    mx = prefac * (
        Omega(z, L) * np.cos(theta) + (Lambda(z, rho, L) - 1) * np.sin(theta)
    )

    window = np.where(rho > (dimx - 1) / 2)
    mx[window] = 0
    my[window] = 0
    if background == "pos":
        mz[window] = 1
    elif background == 'neg':
        mz[window] = -1
    else:
        mz[window] = 0

    return np.array([mz, my, mx])




def lillihook(dim, rad=None, Q=1, gamma=1.5708, P=1, show=False):
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
        ndarray: mags, array of shape [3, dim, dim], along first axis is stacked
                 (magz, magy, magx)

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
    dist = np.sqrt(x**2 + y**2)
    zeros = np.where(dist == 0)
    dist[zeros] = 1

    f = ((dist / rad) ** (2 * Q) - 4) / ((dist / rad) ** (2 * Q) + 4)
    re = np.real(np.exp(1j * gamma))
    im = np.imag(np.exp(1j * gamma))

    mag_x = -np.sqrt(1 - f**2) * (
        re * np.cos(Q * np.arctan2(y, x)) + im * np.sin(Q * np.arctan2(y, x))
    )
    mag_y = -np.sqrt(1 - f**2) * (
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
    return np.array([mag_z, mag_y, mag_x])


def bloch(dim, chirality="cw", pad=True, ir=0, show=False, bkg="pos", sigma=None,
          empty_bkg=False):
    """Create a Bloch vortex magnetization structure.

    Unlike Lillihook, this function does not produce a rigorously calculated
    magnetization structure.

    For the chirality, "cw" vs "ccw" is defined for oring='upper' (y goes dowwn)

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
        ndarray: mags, array of shape [3, dim, dim], along first axis is stacked
                 (magz, magy, magx)
    """
    cx, cy = [dim // 2, dim // 2]
    sigma = dim // 40 if sigma is None else sigma
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
    dist = np.sqrt(x**2 + y**2)

    phi = np.arctan2(y, x)
    radcurve = (
        np.sin(np.pi * dist / (rad - ir) - np.pi * (2 * ir - rad) / (rad - ir))
        * circmask
    )
    radcurve = gaussian_filter(radcurve, sigma=sigma)

    mag_x = -np.sin(phi) * radcurve
    mag_y = np.cos(phi) * radcurve

    # smooth with gaussian
    mag_x = gaussian_filter(mag_x, sigma=sigma)
    mag_y = gaussian_filter(mag_y, sigma=sigma)

    mag_x /= np.max(mag_x)
    mag_y /= np.max(mag_y)

    mag_z = (-ir - rad + 2 * dist) / (ir - rad) * circmask
    mag_z[np.where(dist < ir)] = 1
    mag_z[np.where(dist > rad)] = -1

    mag = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
    mag_x /= mag
    mag_y /= mag
    mag_z /= mag

    if chirality == "cw":
        mag_x *= -1
        mag_y *= -1
    if bkg != "neg":
        mag_z *= -1

    if empty_bkg:
        if bkg != "neg":
            empties = np.where(mag_z>0.99)
        else:
            empties = np.where(mag_z<-0.99)
        mag_x[empties] = 0
        mag_y[empties] = 0
        mag_z[empties] = 0

    if show:
        show_im(mag_x, "mag x")
        show_im(mag_y, "mag y")
        show_im(mag_z, "mag z")
        x = np.arange(0, dim, 1)
        fig, ax = plt.subplots()
        ax.plot(x, mag_z[dim // 2], label="mag_z profile along x-axis.")
        plt.legend()
        plt.show()
    return np.array([mag_z, mag_y, mag_x])


def neel(dim, chirality="io", pad=True, ir=0, show=False):
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
        ndarray: mags, array of shape [3, dim, dim], along first axis is stacked
                 (magz, magy, magx)

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

    mag = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
    mag_x /= mag
    mag_y /= mag
    mag_z /= mag

    if chirality == "oi":
        mag_x *= -1
        mag_y *= -1

    if show:
        # show_im(np.sqrt(mag_x**2 + mag_y**2 + mag_z**2), "mag")
        # show_im(mag_x, "mag x")
        # show_im(mag_y, "mag y")
        # show_im(mag_z, "mag z")

        x = np.arange(0, dim, 1)
        fig, ax = plt.subplots()
        ax.plot(x, mag_x[dim // 2], label="x")
        ax.plot(x, -mag_y[:, dim // 2], label="y")
        ax.plot(x, mag_z[dim // 2], label="z")
        ax.set_xlabel("pixels")
        ax.set_ylabel("M")

        plt.legend()
        plt.show()

    return np.array([mag_z, mag_y, mag_x])


def blochII(
    dim, direction="right", pad=True, ir=0, show=False, sigma=None, cp1=None, cp2=None
):
    """Create a type II Bloch bubble.

    Args:
        dim (int): Dimension of lattice.
        pad (bool): Whether or not to leave some space between the edge of the
            magnetization and the edge of the image.
        ir (float): Inner radius of the vortex in pixels.
        show (bool): If True, will show the x, y, z components in plot form.

    Returns:
        ndarray: mags, array of shape [3, dim, dim], along first axis is stacked
                 (magz, magy, magx)

    """
    cy, cx = [dim // 2, dim // 2]
    sigma = dim // 40 if sigma is None else sigma
    cp1 = dim // 200 if cp1 is None else cp1
    cp2 = dim // 100 if cp2 is None else cp2
    cp2 = max(cp2, cp1+1)

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
    dist = np.sqrt(x**2 + y**2)

    phi = np.arctan2(y, x)
    # show_im(phi, "phi original")

    # defining middle section
    phi[cy : cy + cp1] = np.pi / 2
    phi[cy - cp1 : cy] = -np.pi / 2
    phi[cy - cp2 : cy - cp1] = np.linspace(phi[cy - cp2], -np.pi / 2, cp2 - cp1)
    phi[cy + cp1 : cy + cp2] = np.linspace(np.pi / 2, phi[cy + cp2], cp2 - cp1)

    radcurve = (
        np.sin(np.pi * dist / (rad - ir) - np.pi * (2 * ir - rad) / (rad - ir))
        * circmask
    )
    radcurve = gaussian_filter(radcurve, sigma=sigma)

    mag_x = -np.sin(phi) * radcurve
    mag_y = np.cos(phi) * radcurve

    # # making type II
    mag_x[:cy] *= -1
    mag_y[:cy] *= -1

    # smooth with gaussian
    mag_x = gaussian_filter(mag_x, sigma=sigma)
    mag_y = gaussian_filter(mag_y, sigma=sigma)

    mag_x /= np.max(mag_x)
    mag_y /= np.max(mag_y)
    mag_z = (-ir - rad + 2 * dist) / (ir - rad) * circmask
    mag_z[np.where(dist < ir)] = 1
    mag_z[np.where(dist > rad)] = -1

    mag = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
    mag_x /= mag
    mag_y /= mag
    mag_z /= mag

    if direction in ["top", "bottom"]:
        mag_x, mag_y = -1*np.rot90(mag_y), np.rot90(mag_x)

    if direction in ["left", "top"]:
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
    return np.array([mag_z, mag_y, mag_x])
