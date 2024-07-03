import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from scipy import ndimage
from PyLorentz.visualize.colorwheel import color_im, get_cmap
from matplotlib import colors


def show_2D(
    mag_y,
    mag_x,
    mag_z=None,
    num_arrows=0,
    arrow_size=None,
    arrow_width=None,
    title=None,
    color=True,
    cmap=None,
    cbar=False,
    origin="upper",
    save=None,
    ax=None,
    rad=None,
    whitetotransparent=False,
    **kwargs,
):
    """Display a 2D vector arrow plot.

    Displays an an arrow plot of a vector field, with arrow length scaling with
    vector magnitude. If color=True, a colormap will be displayed under the
    arrow plot.

    If mag_z is included and color=True, a spherical colormap will be used with
    color corresponding to in-plane and white/black to out-of-plane vector
    orientation.

    Args:
        mag_x (2D array): x-component of magnetization.
        mag_y (2D array): y-component of magnetization.
        mag_z (2D array): optional z-component of magnetization.
        a (int): Number of arrows to plot along the x and y axes. Default 15.
        l (float): Scale factor of arrows. Larger l -> shorter arrows. Default None
            guesses at a good value. None uses matplotlib default.
        w (float): Width scaling of arrows. None uses matplotlib default.
        title (str): (`optional`) Title for plot. Default None.
        color (bool): (`optional`) Whether or not to show a colormap underneath
            the arrow plot. Color image is made from colorwheel.color_im().
        hsv (bool): (`optional`) Only relevant if color == True. Whether to use
            an hsv or 4-fold color-wheel in the color image.
        origin (str): (`optional`) Control image orientation.
        save (str): (`optional`) Path to save the figure.

    Returns:
        fig: Returns the figure handle.
    """
    assert mag_x.ndim == mag_y.ndim
    if mag_x.ndim == 3:
        print("Summing along first axis")
        mag_x = np.sum(mag_x, axis=0)
        mag_y = np.sum(mag_y, axis=0)
        if mag_z is not None:
            mag_z = np.sum(mag_z, axis=0)

    if num_arrows > 0:
        a = int(((mag_x.shape[0] - 1) / num_arrows) + 1)

    dimy, dimx = mag_x.shape
    X = np.arange(0, dimx, 1)
    Y = np.arange(0, dimy, 1)
    U = mag_x
    V = mag_y

    sz_inches = kwargs.pop("figsize", 5)
    if color:
        if rad is None:
            rad = mag_x.shape[0] // 16
            rad = max(rad, 16)
            pad = 10  # pixels
            width = np.shape(mag_y)[1] + 2 * rad + pad
            aspect = dimy / width
        elif rad == 0:
            width = np.shape(mag_y)[1]
            aspect = dimy / width
        else:
            pad = 10  # pixels
            width = np.shape(mag_y)[1] + 2 * rad + pad
            aspect = dimy / width
    else:
        aspect = dimy / dimx

    if ax is None:
        if save is not None:  # and title is None: # to avoid white border when saving
            fig = plt.figure()
            size = (sz_inches, sz_inches * aspect)
            fig.set_size_inches(size)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            fig.add_axes(ax)
        else:
            fig, ax = plt.subplots()
        ax.set_aspect(aspect)
    if color:
        cmap = get_cmap(cmap, **kwargs)
        cim = color_im(
            mag_x,
            mag_y,
            mag_z,
            cmap=cmap,
            rad=rad,
            **kwargs,
        )
        if whitetotransparent and mag_z is not None:
            print("making transparent")
            cim = _white_to_transparent(cim, mag_z)
        im = ax.matshow(cim, cmap=cmap, origin=origin)
        if cbar:
            # TODO set cbar height, and labels to be 0 - 2Pi
            plt.colorbar(
                im,
                ax=ax,
                pad=0.02,
                format="%g",
                label=str(kwargs.get("cbar_title", "")),
            )
        arrow_color = "white"
        ax.set_xticks([])
        ax.set_yticks([])

    else:
        arrow_color = "black"
    arrow_color = kwargs.get("arrow_color", arrow_color)

    if a > 0:
        ashift = (dimx - 1) % a // 2
        arrow_scale = 1/abs(arrow_size) if arrow_size is not None else None
        q = ax.quiver(
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

    if not color and a > 0:
        if origin == "upper":
            ax.invert_yaxis()

    if title is not None:
        tr = False
        ax.set_title(title)
    else:
        tr = True

    plt.tick_params(axis="x", labelbottom=False, bottom=False, top=False)
    plt.tick_params(axis="y", labelleft=False, left=False, right=False)
    # ax.set_aspect(aspect)

    if save is not None:
        if not color:
            tr = False
        tr = True
        print(f"Saving: {save}")
        plt.axis("off")
        dpi = kwargs.get("dpi", max(dimy, dimx) * 5 / sz_inches)
        # sets dpi to 5 times original image dpi so arrows are reasonably sharp
        if title is None:  # for no padding
            plt.savefig(save, dpi=dpi, bbox_inches=0, transparent=tr)
        else:
            plt.savefig(save, dpi=dpi, bbox_inches="tight", transparent=tr)

    return


def show_3D(mag_z, mag_y, mag_x, num_arrows=15, ay=None, num_arrows_z=15, l=None, show_all=True):
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
        tcolor = "k"
    else:  # alpha values map to inplane component
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

    ax.set_xlabel("x", c=tcolor)
    ax.set_ylabel("y", c=tcolor)
    ax.set_zlabel("z", c=tcolor)
    plt.show()


def _white_to_transparent(image, magz):
    rgba_image = np.ones((image.shape[0], image.shape[1], 4), dtype=float)
    rgba_image[:, :, :3] = image[:, :, :3]  # Convert RGB values to 0-255 range
    alpha = np.where(magz > 0, 1 - magz**10, 1)
    rgba_image[:, :, 3] = alpha
    return rgba_image
