import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from scipy import ndimage
from PyLorentz.visualize.colorwheel import color_im, get_cmap
from matplotlib import colors



# =============================================== #
#            Various display functions            #
# =============================================== #
""" Not all of these are used in TIE_reconstruct, but I often find them useful
to have handy when working in Jupyter notebooks."""


def show_im(
    image,
    title=None,
    simple=False,
    origin="upper",
    cbar=True,
    cbar_title="",
    scale=None,
    **kwargs,
):
    """Display an image on a new axis.

    Takes a 2D array and displays the image in grayscale with optional title on
    a new axis. In general it's nice to have things on their own axes, but if
    too many are open it's a good idea to close with plt.close('all').

    Args:
        image (2D array): Image to be displayed.
        title (str): (`optional`) Title of plot.
        simple (bool): (`optional`) Default output or additional labels.

            - True, will just show image.
            - False, (default) will show a colorbar with axes labels, and will adjust the
              contrast range for images with a very small range of values (<1e-12).

        origin (str): (`optional`) Control image orientation.

            - 'upper': (default) (0,0) in upper left corner, y-axis goes down.
            - 'lower': (0,0) in lower left corner, y-axis goes up.

        cbar (bool): (`optional`) Choose to display the colorbar or not. Only matters when
            simple = False.
        cbar_title (str): (`optional`) Title attached to the colorbar (indicating the
            units or significance of the values).
        scale (float): Scale of image in nm/pixel. Axis markers will be given in
            units of nanometers.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    if not simple and np.max(image) - np.min(image) < 1e-12:
        # adjust coontrast range
        vmin = np.min(image) - 1e-12
        vmax = np.max(image) + 1e-12
        im = ax.matshow(image, cmap="gray", origin=origin, vmin=vmin, vmax=vmax)
    else:
        im = ax.matshow(image, cmap="gray", origin=origin, **kwargs)

    if title is not None:
        ax.set_title(str(title), pad=0)

    if simple:
        plt.axis("off")
    else:
        plt.tick_params(axis="x", top=False)
        ax.xaxis.tick_bottom()
        ax.tick_params(direction="in")
        if scale is None:
            ticks_label = "pixels"
        else:

            def mjrFormatter(x, pos):
                return f"{scale*x:.3g}"

            fov = scale * max(image.shape[0], image.shape[1])

            if fov < 4e3:  # if fov < 4um use nm scale
                ticks_label = " nm "
            elif fov > 4e6:  # if fov > 4mm use m scale
                ticks_label = "  m  "
                scale /= 1e9
            else:  # if fov between the two, use um
                ticks_label = " $\mu$m "
                scale /= 1e3

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))

        if origin == "lower":
            ax.text(y=0, x=0, s=ticks_label, rotation=-45, va="top", ha="right")
        elif origin == "upper":  # keep label in lower left corner
            ax.text(
                y=image.shape[0], x=0, s=ticks_label, rotation=-45, va="top", ha="right"
            )

        if cbar:
            plt.colorbar(im, ax=ax, pad=0.02, format="%.2g", label=str(cbar_title))

    plt.show()
    return


def show_stack(
    images,
    ptie=None,
    origin="upper",
    titles=None,
    titletext="",
):
    """Shows a stack of dm3s or np images with a slider to navigate slice axis.

    Uses ipywidgets.interact to allow user to view multiple images on the same
    axis using a slider. There is likely a better way to do this, but this was
    the first one I found that works...

    If a TIE_params object is given, only the regions corresponding to ptie.crop
    will be shown.

    Args:
        images (list): List of 2D arrays. Stack of images to be shown.
        ptie (``TIE_params`` object): Will use ptie.crop to show only the region
            that will remain after being cropped.
        origin (str): (`optional`) Control image orientation.
        title (bool): (`optional`) Try and pull a title from the signal objects.
    Returns:
        None
    """
    images = np.array(images)

    if ptie is None:
        t, b = 0, images[0].shape[0]
        l, r = 0, images[0].shape[1]
    else:
        if ptie.rotation != 0 or ptie.x_transl != 0 or ptie.y_transl != 0:
            rotate, x_shift, y_shift = ptie.rotation, ptie.x_transl, ptie.y_transl
            for i in range(len(images)):
                images[i] = ndimage.rotate(images[i], rotate, reshape=False)
                images[i] = ndimage.shift(images[i], (-y_shift, x_shift))
        t = ptie.crop["top"]
        b = ptie.crop["bottom"]
        l = ptie.crop["left"]
        r = ptie.crop["right"]

    images = images[:, t:b, l:r]

    _fig, _ax = plt.subplots()
    plt.axis("off")
    N = images.shape[0]

    def view_image(i=0):
        _im = plt.imshow(images[i], cmap="gray", interpolation="nearest", origin=origin)
        if titles is not None:
            plt.title(f"{titletext} {titles[i]}")

    interact(view_image, i=(0, N - 1))
    return


def show_2D(
    mag_x,
    mag_y,
    mag_z=None,
    a=0,
    l=None,
    w=None,
    title=None,
    color=True,
    cmap=None,
    origin="upper",
    save=None,
    GUI_handle=False,
    GUI_color_array=None,
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
        GUI_handle (bool): ('optional') Handle for indicating if using GUI.
            Default is False.
        GUI_color_array (2D array): ('optional') The colored image array passed from the GUI,
            it is for creating the overlaying the arrows without using color_im().

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

    if a > 0:
        a = ((mag_x.shape[0] - 1) // a) + 1

    dimy, dimx = mag_x.shape
    X = np.arange(0, dimx, 1)
    Y = np.arange(0, dimy, 1)
    U = mag_x
    V = mag_y

    sz_inches = 8
    if not GUI_handle or save is not None:
        if color:
            rad = mag_x.shape[0] // 16
            rad = max(rad, 16)
            pad = 10  # pixels
            width = np.shape(mag_y)[1] + 2 * rad + pad
            aspect = dimy / width
        else:
            aspect = dimy / dimx

    if GUI_handle and save is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.ioff()
        ax.set_aspect("equal", adjustable="box")
    else:
        fig, ax = plt.subplots()
        ax.set_aspect(aspect)

    if color:
        if not GUI_handle or save is not None:
            cmap = get_cmap(cmap)
            cim = color_im(
                    mag_x,
                    mag_y,
                    mag_z,
                    cmap=cmap,
                    rad=rad,
                )
            im = ax.matshow(cim, cmap=cmap, origin=origin)

        else:
            im = ax.matshow(GUI_color_array, cmap="gray", origin=origin, aspect="equal")

        arrow_color = "white"
        plt.axis("off")
    else:
        arrow_color = "black"
        if GUI_handle and save is None:
            white_array = np.zeros([dimy, dimx, 3], dtype=np.uint8)
            white_array.fill(255)
            im = ax.matshow(white_array, cmap="gray", origin=origin, aspect="equal")
            plt.axis("off")
        elif GUI_handle and save:
            white_array = np.zeros([dimy, dimx, 3], dtype=np.uint8)
            white_array.fill(255)
            im = ax.matshow(white_array, cmap="gray", origin=origin)
            fig.tight_layout(pad=0)
            ax.xaxis.set_major_locator(mpl.ticker.NullLocator())
            ax.yaxis.set_major_locator(mpl.ticker.NullLocator())
            plt.axis("off")

    if a > 0:
        ashift = (dimx - 1) % a // 2
        q = ax.quiver(
            X[ashift::a],
            Y[ashift::a],
            U[ashift::a, ashift::a],
            V[ashift::a, ashift::a],
            units="xy",
            scale=l,
            scale_units="xy",
            width=w,
            angles="xy",
            pivot="mid",
            color=arrow_color,
        )

    if not color:
        if not GUI_handle:
            # qk = ax.quiverkey(
            #     q,
            #     X=0.95,
            #     Y=0.98,
            #     U=1,
            #     label=r"$Msat$",
            #     labelpos="S",
            #     coordinates="axes",
            # )
            # qk.text.set_backgroundcolor("w")
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
    if not GUI_handle:
        plt.show()

    if save is not None:
        if not color:
            tr = False
        fig.set_size_inches(8, 8 / aspect)
        print(f"Saving: {save}")
        plt.axis("off")
        # sets dpi to 5 times original image dpi so arrows are reasonably sharp
        dpi2 = max(dimy, dimx) * 5 / sz_inches
        plt.savefig(save, dpi=dpi2, bbox_inches="tight", transparent=tr)

    if GUI_handle:
        return fig, ax
    else:
        return




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
        tcolor = 'k'
    else:  # alpha values map to inplane component
        tcolor = 'w'
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
    vmax = np.max([im_un, im_in, im_ov])
    vmin = np.min([im_un, im_in, im_ov])
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
