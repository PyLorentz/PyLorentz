from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from .colorwheel import color_im, get_cmap
from .show import show_im


def show_2D(
    Vx: np.ndarray,
    Vy: np.ndarray,
    Vz: Optional[np.ndarray] = None,
    num_arrows: int = 0,
    arrow_size: Optional[float] = None,
    arrow_width: Optional[float] = None,
    title: Optional[str] = None,
    color: bool = True,
    cmap: str = "hsv",
    origin: str = "upper",
    save: Optional[str] = None,
    figax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
    rad: Optional[int] = None,
    scale: Optional[float] = None,
    **kwargs,
) -> plt.Figure:
    """
    Display a 2D vector field with arrows and optional color mapping.

    Args:
        Vx (np.ndarray): X-component of the vector field.
        Vy (np.ndarray): Y-component of the vector field.
        Vz (np.ndarray, optional): Z-component of the vector field.
        num_arrows (int): Number of arrows to plot along x and y axes.
        arrow_size (float, optional): Scale factor for arrow length.
        arrow_width (float, optional): Width of arrows.
        title (str, optional): Title of the plot.
        color (bool): Whether to display a colormap underneath the arrows.
        cmap (str): Colormap to use for color mapping.
        origin (str): Origin of the image coordinate system.
        save (str, optional): Path to save the figure.
        figax (tuple, optional): Figure and axes to plot on.
        rad (int, optional): Radius for the color wheel.
        scale (float, optional): Scale factor for the plot.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    assert Vx.ndim == Vy.ndim
    if Vx.ndim == 3:
        print("Summing along first axis")
        Vx = np.sum(Vx, axis=0)
        Vy = np.sum(Vy, axis=0)
        if Vz is not None:
            Vz = np.sum(Vz, axis=0)

    if num_arrows > 0:
        a = int(((Vx.shape[0] - 1) / num_arrows) + 1)
    else:
        a = -1

    dimy, dimx = Vx.shape
    X = np.arange(0, dimx, 1)
    Y = np.arange(0, dimy, 1)
    U = Vx
    V = Vy

    sz_inches = kwargs.pop("figsize", 5)
    if isinstance(sz_inches, (list, tuple, np.ndarray)):
        sz_inches = sz_inches[0]  # Aspect ratio depends on rad, use one value
    if color:
        if rad is None:
            rad = Vx.shape[0] // 16
            rad = max(rad, 16)
            pad = 10  # pixels
            width = np.shape(Vy)[1] + 2 * rad + pad
            aspect = dimy / width
        elif rad == 0:
            width = np.shape(Vy)[1]
            aspect = dimy / width
        else:
            pad = 10  # pixels
            width = np.shape(Vy)[1] + 2 * rad + pad
            aspect = dimy / width
    else:
        aspect = dimy / dimx

    if figax is None:
        fig = plt.figure()
        size = (sz_inches, sz_inches * aspect)
        fig.set_size_inches(size)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
        ax.set_aspect(aspect)
    else:
        fig, ax = figax

    if color:
        cmap = get_cmap(cmap, **kwargs)
        cim = color_im(
            Vx,
            Vy,
            Vz,
            cmap=cmap,
            rad=rad,
            **kwargs,
        )

        show_bbox = kwargs.pop("show_bbox", None)
        if show_bbox is None:
            show_bbox = title is not None
        show_im(
            cim,
            cmap=cmap,
            origin=origin,
            ticks_off=scale is None or kwargs.pop("ticks_off", False),
            scale=scale,
            figax=(fig, ax),
            title=title,
            show_bbox=show_bbox,
            **kwargs,
        )
        arrow_color = "white"
    else:
        arrow_color = "black"
    arrow_color = kwargs.get("arrow_color", arrow_color)

    if a > 0:
        ashift = (dimx - 1) % a // 2
        arrow_scale = 1 / abs(arrow_size) if arrow_size is not None else None
        ax.quiver(
            X[ashift::a],
            Y[ashift::a],
            U[ashift::a, ashift::a],
            V[ashift::a, ashift::a],
            units="xy",
            scale=arrow_scale,
            scale_units="xy",
            width=arrow_width,
            angles="xy",
            pivot="mid",
            color=arrow_color,
        )

    if not color and a > 0 and origin == "upper":
        ax.invert_yaxis()

    if save is not None:
        print(f"Saving: {save}")
        plt.axis("off")
        dpi = kwargs.get("dpi", max(dimy, dimx) * 5 / sz_inches)
        if title is None:
            plt.savefig(save, dpi=dpi, bbox_inches=0, transparent=True)
        else:
            plt.savefig(save, dpi=dpi, bbox_inches="tight", transparent=True)

    return fig


def show_3D(
    Vx: np.ndarray,
    Vy: np.ndarray,
    Vz: np.ndarray,
    num_arrows: int = 15,
    ay: Optional[int] = None,
    num_arrows_z: int = 15,
    arrow_size: Optional[float] = None,
    show_all: bool = True,
) -> None:
    """
    Display a 3D vector field with arrows, using color to represent vector direction.

    Arrow color is determined by direction, with in-plane mapping to an HSV
    color-wheel and out of plane to white (+z) and black (-z).

    Args:
        Vx (np.ndarray): (z, y, x) X-component of the vector field.
        Vy (np.ndarray): (z, y, x) Y-component of the vector field.
        Vz (np.ndarray): (z, y, x) Z-component of the vector field.
        num_arrows (int): Number of arrows to plot along the x-axis.
        ay (int, optional): Number of arrows to plot along the y-axis.
        num_arrows_z (int): Number of arrows to plot along the z-axis.
        arrow_size (float, optional): Scale factor for arrow length.
        show_all (bool): Whether to show all arrows with equal opacity.

    Returns:
        None
    """
    bmax = max(Vx.max(), Vy.max(), Vz.max())

    if arrow_size is None:
        arrow_size = Vx.shape[1] / (2 * bmax * num_arrows)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if Vx.ndim == 3:
        dimz, dimy, dimx = Vx.shape
        if num_arrows_z > dimz:
            az = 1
        else:
            az = ((dimz - 1) // num_arrows_z) + 1
    else:
        dimy, dimx = Vx.shape
        dimz = 1

    Z, Y, X = np.meshgrid(
        np.arange(0, dimz, 1),
        np.arange(0, dimy, 1),
        np.arange(0, dimx, 1),
        indexing="ij",
    )
    ay = ((dimy - 1) // num_arrows) + 1 if ay is None else ay
    axx = ((dimx - 1) // num_arrows) + 1

    zeros = ~(Vx.astype("bool") | Vy.astype("bool") | Vz.astype("bool"))
    zinds = np.where(zeros)
    Vz[zinds] = bmax / 1e5
    Vx[zinds] = bmax / 1e5
    Vy[zinds] = bmax / 1e5

    U = Vx.reshape((dimz, dimy, dimx))
    V = Vy.reshape((dimz, dimy, dimx))
    W = Vz.reshape((dimz, dimy, dimx))

    phi = np.ravel(np.arctan2(V[::az, ::ay, ::axx], U[::az, ::ay, ::axx]))

    hue = phi / (2 * np.pi) + 0.5
    theta = np.arctan2(
        W[::az, ::ay, ::axx],
        np.sqrt(U[::az, ::ay, ::axx] ** 2 + V[::az, ::ay, ::axx] ** 2),
    )
    value = np.ravel(np.where(theta < 0, 1 + 2 * theta / np.pi, 1))
    sat = np.ravel(np.where(theta > 0, 1 - 2 * theta / np.pi, 1))

    arrow_colors = np.squeeze(np.dstack((hue, sat, value)))
    arrow_colors = colors.hsv_to_rgb(arrow_colors)

    if show_all:
        alphas = np.ones((np.shape(arrow_colors)[0], 1))
        tcolor = "k"
    else:
        tcolor = "w"
        alphas = np.minimum(value, sat).reshape(len(value), 1)
        value = np.ones(value.shape)
        sat = np.ravel(1 - abs(2 * theta / np.pi))
        arrow_colors = np.squeeze(np.dstack((hue, sat, value)))
        arrow_colors = colors.hsv_to_rgb(arrow_colors)

        ax.set_facecolor("black")
        for axs in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axs.set_pane_color((0, 0, 0, 1.0))
            axs.pane.set_edgecolor(tcolor)
            [t.set_color(tcolor) for t in axs.get_ticklines()]
            [t.set_color(tcolor) for t in axs.get_ticklabels()]
        ax.grid(False)

    arrow_colors = np.array(
        [np.concatenate((arrow_colors[i], alphas[i])) for i in range(len(alphas))]
    )
    arrow_colors = np.concatenate((arrow_colors, np.repeat(arrow_colors, 2, axis=0)))

    dim = max(dimx, dimy, dimz)
    ax.set_xlim(0, dim)
    ax.set_ylim(0, dimy)
    if az >= dimz:
        ax.set_zlim(-dim // 2, dim // 2)
    else:
        ax.set_zlim(0, dim)
        Z += (dim - dimz) // 2

    ax.quiver(
        X[::az, ::ay, ::axx],
        Y[::az, ::ay, ::axx],
        Z[::az, ::ay, ::axx],
        U[::az, ::ay, ::axx],
        V[::az, ::ay, ::axx],
        W[::az, ::ay, ::axx],
        color=arrow_colors,
        length=float(arrow_size),
        pivot="middle",
        normalize=False,
    )

    ax.set_xlabel("x", c=tcolor)
    ax.set_ylabel("y", c=tcolor)
    ax.set_zlabel("z", c=tcolor)
    plt.show()
