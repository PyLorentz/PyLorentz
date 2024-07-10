import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from scipy import ndimage
from PyLorentz.visualize.colorwheel import color_im, get_cmap
from matplotlib import colors


def show_im(
    image,
    title=None,
    simple=False,
    origin="upper",
    cbar=None,
    cbar_title="",
    scale=None,
    save=None,
    roi=None,
    figax=None,
    intensity_range="minmax",
    **kwargs,
):
    """Display an image on a new axis.
    # TODO add colorcet colormaps here. basically just allow using my colorwheel
    version of get_cmap, expanded to look for other strings with matplotlib

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

        intensity_range (str): (`optional`) Control how vmin and vmax are set. Based on
            py4DSTEM.show function. Default 'minmax' will set intensity range (matplotlib
            vmin and vmax) to vmin and vmax if given (kwargs) or to the full range of the
            image if not given. Other supported option is 'ordered' which will set vmin
            and vmax to fractions of the pixel value distributions. e.g. vmin=0.01 will
            saurate the lower 1% of pixels. If intensity_range == 'ordered', vmin and vmax
            will default to 0.01 and 0.99 unless otherwise specified.

    Returns:
        None
    """
    image = np.array(image)
    if image.dtype == "bool":
        image = image.astype("int")
    if cbar is None:
        cbar = not simple
    ndim = np.ndim(image)
    if ndim == 2:
        pass
    elif ndim == 3:
        if image.shape[2] not in ([3, 4]):
            if image.shape[0] != 1:
                s = (
                    "Input image is 3D and does not seem to be a color image.\n"
                    "Summing along first axis"
                )
                print(s)
            image = np.sum(image, axis=0)
        else:
            cbar = False
    else:
        s = (
            f"Input image is of dimension {ndim}. "
            "Please input a 2D image or 3D color image."
        )
        print(s)
        return

    if figax is not None:
        fig, ax = figax
    else:
        aspect = image.shape[0] / image.shape[1]
        size = kwargs.pop("figsize", (5, 5 * aspect))
        if isinstance(size, (int, float)):
            size = (size, size)
        if simple and title is None:
            # all this to avoid a white border when saving the image without a title or cbar
            fig = plt.figure()
            fig.set_size_inches(size)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            fig.add_axes(ax)
        else:
            _fig, ax = plt.subplots(figsize=size)

    cmap = kwargs.get("cmap", "gray")

    if intensity_range.lower() == "minmax":
        vmin = kwargs.get("vmin", None)
        vmax = kwargs.get("vmax", None)
        if (not cbar and np.ptp(image) < 1e-12) or (np.ptp(image) < 1e-15):
            # adjust coontrast range if minimal range detected, avoids people thinking 0
            # phase shift images (E-15) are real when no cbar to make this obvious
            # can circumvent this by setting vmin and vmax manually if desired
            if vmin is None:
                vmin = np.min(image) - 1e-12
            if vmax is None:
                vmax = np.max(image) + 1e-12

    elif intensity_range.lower() == "ordered":
        # scaled display, saturating highest/lowest 1% (default) pixels
        vmin = kwargs.get("vmin", 0.01)
        vmax = kwargs.get("vmax", 0.99)
        vals = np.sort(image.ravel())
        ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
        ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
        ind_vmin = np.max([0, ind_vmin])
        ind_vmax = np.min([len(vals) - 1, ind_vmax])
        vmin = vals[ind_vmin]
        vmax = vals[ind_vmax]
        if vmax == vmin:
            print("vmax = vmin, setting intensity range to full")
            vmin = vals[0]
            vmax = vals[-1]

    im = ax.matshow(image, origin=origin, vmin=vmin, vmax=vmax, cmap=cmap)

    if title is not None:
        ax.set_title(str(title))

    if simple or kwargs.pop("ticks_off", False):
        if title is not None or cbar:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_axis_off()  # this will not draw bounding box as well
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
            if kwargs.get("scale_units", None) is None:
                if fov < 4:  # if fov < 4 nm use A scale
                    ticks_label = r"  Å  "  # extra spaces to pad away from ticks
                    scale *= 10
                elif fov < 4e3:  # if fov < 4um use nm scale
                    ticks_label = " nm "
                elif fov < 4e6:  # fov < 4 mm use um scale
                    ticks_label = r" $\mu$m "
                    scale /= 1e3
                else:  # if fov > 4mm use m scale
                    ticks_label = "  m  "
                    scale /= 1e9
            else:
                ticks_label = kwargs.get("scale_units")

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))

        if kwargs.pop("ticks_label_off", False):
            pass
        elif origin == "lower":
            ax.text(y=0, x=0, s=ticks_label, rotation=-45, va="top", ha="right")
        elif origin == "upper":  # keep label in lower left corner
            ax.text(
                y=image.shape[0], x=0, s=ticks_label, rotation=-45, va="top", ha="right"
            )

    if roi is not None:
        lw = kwargs.get("roi_lw", 2)
        pad = kwargs.get("roi_pad", 2)
        color = kwargs.get("roi_color", "white")
        dy, dx = image.shape
        left = (roi["left"] - lw * pad) / dx
        bottom = (dy - roi["bottom"] - lw * pad) / dy
        width = (roi["right"] - roi["left"] + 2 * lw * pad) / dx
        height = (roi["bottom"] - roi["top"] + 2 * lw * pad) / dy

        p = plt.Rectangle(
            (left, bottom), width, height, fill=False, edgecolor=color, linewidth=lw
        )

        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)

    if cbar:
        # for matching cbar height to image height even with title
        aspect = image.shape[-2] / image.shape[-1]
        plt.colorbar(
            im,
            ax=ax,
            pad=0.02,
            format="%g",
            label=str(cbar_title),
            fraction=0.047 * aspect,
        )

    if save:
        print("saving: ", save)
        dpi = kwargs.get("dpi", 400)
        if simple and title is None:
            plt.savefig(save, dpi=dpi, bbox_inches=0)
        else:
            plt.savefig(save, dpi=dpi, bbox_inches="tight")
    else:
        plt.tight_layout()

    if figax is None:
        plt.show()
    return


def show_stack(images, titles=None, scale_each=True, **kwargs):
    """
    Uses ipywidgets.interact to allow user to view multiple images on the same
    axis using a slider. There is likely a better way to do this, but this was
    the first one I found that works...

    Args:
        images (list): List of 2D arrays. Stack of images to be shown.
        origin (str): (`optional`) Control image orientation.
        title (bool): (`optional`) Try and pull a title from the signal objects.
    Returns:
        None
    """
    _fig, _ax = plt.subplots()
    images = np.array(images)
    if not scale_each:
        vmin = np.min(images)
        vmax = np.max(images)
    else:
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)

    N = images.shape[0]
    if titles is not None:
        assert len(titles) == len(images)
    else:
        title = kwargs.pop("title", None)
        titles = [title] * len(images)

    show_im(
        images[0],
        figax=(_fig, _ax),
        title=titles[0],
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    def view_image(i=0):
        t = titles[i]
        show_im(
            images[i],
            figax=(_fig, _ax),
            title=t,
            ticks_label_off=True,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

    interact(view_image, i=(0, N - 1))
    return


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


def show_im_peaks(im=None, peaks=None, peaks2=None, title=None, cbar=False, **kwargs):
    """
    peaks an array of shape (2, N): [[x1, x2, ..., xn], [y1, y2, ..., yn]]
    """
    fig, ax = plt.subplots()
    if im is not None:
        show_im(im, title=title, cbar=cbar, figax=(fig, ax), **kwargs)
    if peaks is not None:
        peaks = np.array(peaks)
        assert peaks.ndim == 2, f"Peaks dimension {peaks.ndim} != 2"
        if peaks.shape[1] == 2 and peaks.shape[0] != 2:
            peaks = peaks.T
        ax.plot(
            peaks[1],
            peaks[0],
            c=kwargs.get("color1", "r"),
            alpha=kwargs.get("alpha", 0.9),
            ms=kwargs.get("ms", None),
            marker="o",
            fillstyle="none",
            linestyle="none",
        )
    if peaks2 is not None and np.size(peaks2) != 0:
        peaks2 = np.array(peaks2)
        if peaks2.shape[1] == 2 and peaks2.shape[0] != 2:
            peaks2 = peaks2.T
        ax.plot(
            peaks2[1],
            peaks2[0],
            c=kwargs.get("color2", "b"),
            alpha=kwargs.get("alpha", 0.9),
            ms=kwargs.get("ms", None),
            marker="o",
            fillstyle="none",
            linestyle="none",
        )
    plt.show()
