"""
Add any other stuff from hipl that i've accumulated

"""

import numpy as np


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
    dist = np.sqrt(x**2 + y**2)

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

    mag = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
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

    mag = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
    mag_x /= mag
    mag_y /= mag
    mag_z /= mag

    if chirality == "oi":
        mag_x *= -1
        mag_y *= -1

    if show:
        show_im(np.sqrt(mag_x**2 + mag_y**2 + mag_z**2), "mag")
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
