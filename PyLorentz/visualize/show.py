import warnings
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage
from ipywidgets import interact

from PyLorentz.visualize.colorwheel import get_cmap

# warnings.filterwarnings("error")  # plt.tight_layout() sometimes throws a UserWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


def show_im(
    image: np.ndarray,
    title: Optional[str] = None,
    scale: Optional[float] = None,
    simple: bool = False,
    save: Optional[str] = None,
    cmap: str = "gray",
    figax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
    roi: Optional[dict] = None,
    cbar_title: Optional[str] = None,
    cbar: Optional[bool] = None,
    intensity_range: str = "minmax",
    origin: str = "upper",
    **kwargs,
) -> None:
    """
    Display an image with optional features like colorbar, title, and scale.

    Args:
        image (np.ndarray): 2D image to be displayed.
        title (str, optional): Title of the plot.
        scale (float, optional): Scale in nm/pixel for axis markers.
        simple (bool): Simplified display without colorbar and labels.
        save (str, optional): Path to save the figure.
        cmap (str): Colormap for displaying the image.
        figax (tuple, optional): Figure and axis to plot on.
        roi (dict, optional): Region of interest to highlight.
        cbar_title (str, optional): Title for the colorbar.
        cbar (bool, optional): Whether to display the colorbar.
        intensity_range (str): Method to set intensity range. 'minmax' or 'ordered'.
        origin (str): Image origin ('upper' or 'lower').
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    image = np.array(image, dtype=np.float64)
    if image.dtype == "bool":
        image = image.astype("int")

    if cbar is None:
        cbar = not simple

    ndim = np.ndim(image)
    if ndim == 2:
        pass
    elif ndim == 3:
        if image.shape[2] not in [3, 4]:
            if image.shape[0] != 1:
                print("Summing along first axis")
            image = np.sum(image, axis=0)
        else:
            cbar = False
    else:
        print(f"Input image is of dimension {ndim}. Please input a 2D or 3D image.")
        return

    if figax is not None:
        fig, ax = figax
    else:
        aspect = image.shape[0] / image.shape[1]
        size = kwargs.pop("figsize", (5, 5 * aspect))
        if isinstance(size, (int, float)):
            size = (size, size)
        if simple and title is None:
            fig = plt.figure()
            fig.set_size_inches(size)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            fig.add_axes(ax)
        else:
            fig, ax = plt.subplots(figsize=size)

    cmap = get_cmap(cmap, **kwargs)

    if intensity_range.lower() in ["minmax", "abs", "absolute"]:
        vmin = kwargs.get("vmin", None)
        vmax = kwargs.get("vmax", None)
        vm_check = not cbar and np.ptp(image) < 1e-12
        ptp_check = np.ptp(image) < 1e-15
        mode_check = intensity_range.lower() == "minmax"
        if (vm_check or ptp_check) and mode_check:
            if vmin is None:
                vmin = np.min(image) - 1e-12
            if vmax is None:
                vmax = np.max(image) + 1e-12
    elif intensity_range.lower() == "ordered":
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
        ax.set_title(str(title), fontsize=kwargs.pop("title_fontsize", 12))

    if simple or kwargs.pop("ticks_off", False):
        if (title is not None or cbar) or (kwargs.get("show_bbox", False) or not save):
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_axis_off()
        if not kwargs.pop("show_bbox", True):
            ax.set_axis_off()
    else:
        plt.tick_params(axis="x", top=False)
        if not kwargs.get("show_bbox", True):
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax.xaxis.tick_bottom()
        ax.tick_params(direction=kwargs.get("tick_direction", "out"))
        # if scale is None:
        if not isinstance(scale, (float, int)):
            ticks_label = kwargs.get("scale_units", "pixels")
        else:
            ax_ysize_inch = ax.get_position().height * fig.get_size_inches()[1]
            ax_xsize_inch = ax.get_position().width * fig.get_size_inches()[0]
            num_ticks_y = max(round(ax_ysize_inch + 1), 3)
            num_ticks_x = max(round(ax_xsize_inch + 1), 3)
            fov_y, fov_x = np.array(image.shape)[:2] * scale

            ylim = ax.get_ylim()
            ymax = ylim[0] if origin == "upper" else ylim[1]
            nround_y = len(str(round(ymax * scale))) - 2
            floor_fov_y = np.floor(fov_y / 10**nround_y) * 10**nround_y
            yticks = np.linspace(0, floor_fov_y / scale, int(num_ticks_y))
            if origin == "lower":
                yticks = yticks[1:]
            ax.set_yticks(yticks - 0.5)
            ylabs, unit = tick_label_formatter(yticks, fov_y, scale, kwargs.get("scale_units"))
            ax.set_yticklabels(ylabs)

            ticks_label = unit

            _, xmax = ax.get_xlim()
            nround_x = len(str(round(xmax * scale))) - 2
            floor_fov_x = np.floor(fov_x / 10**nround_x) * 10**nround_x
            xticks = np.linspace(0, floor_fov_x / scale, int(num_ticks_x))[1:]
            ax.set_xticks(xticks - 0.5)
            xlabs, unit = tick_label_formatter(xticks, fov_y, scale, kwargs.get("scale_units"))
            ax.set_xticklabels(xlabs)

        if kwargs.pop("ticks_label_off", False):
            pass
        elif origin == "lower":
            ax.text(y=-0.5, x=-0.5, s=ticks_label, rotation=-45, va="top", ha="right")
        elif origin == "upper":
            ax.text(
                y=image.shape[0] - 0.5,
                x=-0.5,
                s=ticks_label,
                rotation=-45,
                va="top",
                ha="right",
            )

    if roi is not None:
        lw = kwargs.get("roi_lw", 2)
        pad = kwargs.get("roi_pad", 0)
        color = kwargs.get("roi_color", "white")
        dy, dx = image.shape
        left = (roi["left"] - lw * pad) / dx
        bottom = (dy - roi["bottom"] - lw * pad) / dy
        width = (roi["right"] - roi["left"] + 2 * lw * pad) / dx
        height = (roi["bottom"] - roi["top"] + 2 * lw * pad) / dy

        p = plt.Rectangle((left, bottom), width, height, fill=False, edgecolor=color, linewidth=lw)
        if "rotation" in roi.keys():
            transform = (
                mpl.transforms.Affine2D().rotate_deg_around(0.5, 0.5, -1 * roi["rotation"])
                + ax.transAxes
            )
        else:
            transform = ax.transAxes
        p.set_transform(transform)
        p.set_clip_on(False)
        ax.add_patch(p)

    if cbar:
        aspect = image.shape[-2] / image.shape[-1]
        cb = plt.colorbar(
            im,
            ax=ax,
            pad=0.02,
            format="%g",
            fraction=0.047 * aspect,
        )
        if cbar_title is not None:
            cb.set_label(str(cbar_title), labelpad=5)

    if save:
        print("saving: ", save)
        dpi = kwargs.get("dpi", 400)
        if simple and title is None:
            plt.savefig(save, dpi=dpi, bbox_inches=0)
        else:
            plt.savefig(save, dpi=dpi, bbox_inches="tight")

    if figax is None:
        # try:
        #     plt.tight_layout()
        # except (UserWarning, RuntimeWarning):
        #     pass
        plt.show()


def show_stack(
    images: List[np.ndarray], titles: Optional[List[str]] = None, scale_each: bool = True, **kwargs
) -> None:
    """
    Display a stack of images interactively using a slider.

    Args:
        images (list): List of 2D numpy arrays representing the images.
        titles (list, optional): List of titles for each image.
        scale_each (bool): Scale each image individually or use the same scale.

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


def show_sims(
    phi: np.ndarray,
    im_un: np.ndarray,
    im_in: np.ndarray,
    im_ov: np.ndarray,
    title: Optional[str] = None,
    save: Optional[str] = None,
) -> None:
    """
    Plot phase, underfocus, infocus, and overfocus images in one plot.

    Args:
        phi (np.ndarray): Image of phase shift.
        im_un (np.ndarray): Underfocus image.
        im_in (np.ndarray): Infocus image.
        im_ov (np.ndarray): Overfocus image.
        title (str, optional): Title for the plot.
        save (str, optional): Path to save the figure.

    Returns:
        None
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
        if not (save.endswith(".png") or save.endswith(".tiff") or save.endswith(".tif")):
            save = save + ".png"
        plt.savefig(save, dpi=300, bbox_inches="tight")

    plt.show()


def show_im_peaks(
    im: Optional[np.ndarray] = None,
    peaks: Optional[np.ndarray] = None,
    peaks2: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    cbar: bool = False,
    **kwargs,
) -> None:
    """
    Show image with overlaid peaks.

    Args:
        im (np.ndarray, optional): Image to display.
        peaks (np.ndarray, optional): Peaks to overlay on the image.
        peaks2 (np.ndarray, optional): Second set of peaks to overlay.
        title (str, optional): Title for the plot.
        cbar (bool): Whether to display a colorbar.

    Returns:
        None
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
            marker=kwargs.get("marker", "o"),
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
            marker=kwargs.get("marker", "o"),
            fillstyle="none",
            linestyle="none",
        )
    plt.show()


def tick_label_formatter(
    ticks: np.ndarray, fov: float, scale: float, scale_units: Optional[str] = None
) -> Tuple[List[str], str]:
    """
    Format tick labels for display.

    Args:
        ticks (np.ndarray): Tick positions.
        fov (float): Field of view.
        scale (float): Scale in nm/pixel.
        scale_units (str, optional): Units for the scale.

    Returns:
        Tuple[List[str], str]: Formatted labels and unit.
    """
    labels = None
    unit = None
    if scale_units is None:
        if fov < 4:  # if fov < 4 nm use A scale
            unit = r"  Å  "  # extra spaces to pad away from ticks
            ticks *= 10
        elif fov < 2e3:  # if fov < 4um use nm scale
            unit = " nm "
        elif fov < 2e6:  # fov < 4 mm use um scale
            unit = r" $\mu$m "
            ticks /= 1e3
        else:  # if fov > 4mm use m scale
            unit = "  m  "
            ticks /= 1e9
    else:
        unit = scale_units
        return ticks, unit

    labels = [
        f"{v:.0f}" if v > 10 else f"{v:.0f}" if v == 0 else f"{v:.2f}" for v in ticks * scale
    ]
    return labels, unit


def show_fft(fft: np.ndarray, **kwargs) -> None:
    """
    Display the FFT of an image with logarithmic scaling.

    Args:
        fft (np.ndarray): FFT of the image.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    mag = np.abs(np.fft.fftshift(fft))
    bads = np.where(mag == 0)
    mag[bads] = 1
    lg = np.log(mag)
    lg[bads] = np.min(lg)

    show_im(lg, **kwargs)


def lineplot_im(
    image: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    phi: float = 0,
    linewidth: int = 1,
    line_len: int = -1,
    show: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Generate a line plot through an image.

    Args:
        image (np.ndarray): Image to analyze.
        center (tuple, optional): Center point for the line plot (cy, cx).
        phi (float): Angle for the line plot in degrees.
        linewidth (int): Line width for the plot.
        line_len (int): Length of the line plot.
        show (bool): Whether to display the plot.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: Line plot data.
    """
    im = np.array(image)
    if np.ndim(im) > 2:
        print("More than 2 dimensions given, collapsing along first axis")
        im = np.sum(im, axis=0)
    dy, dx = im.shape
    if center is None:
        center = (dy // 2, dx // 2)
    cy, cx = round(center[0]), round(center[1])

    sp, ep = _box_intercepts(im.shape, center, phi, line_len)

    profile = skimage.measure.profile_line(
        im, sp, ep, linewidth=linewidth, mode="constant", reduce_func=np.mean
    )
    if line_len > 0 and len(profile) > line_len:
        lp = int(len(profile))
        profile = profile[(lp - line_len) // 2 : -(lp - line_len) // 2]

    if show:
        show_scan = kwargs.get("show_scan", True)
        if show_scan:
            _fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
            ax0.plot(profile)
            ax0.set_aspect(1 / ax0.get_data_ratio(), adjustable="box")
            ax0.set_ylabel("intensity")
            ax0.set_xlabel("pixels")
        else:
            _fig, ax1 = plt.subplots()

        cmap = kwargs.get("cmap", "gray")
        img = ax1.matshow(
            im, cmap=cmap, vmin=kwargs.get("vmin", None), vmax=kwargs.get("vmax", None)
        )
        if kwargs.get("cbar", False):
            plt.colorbar(img, ax=ax1, pad=0.02)

        if linewidth > 1:
            th = np.arctan2((ep[0] - sp[0]), (ep[1] - sp[1]))
            spp, epp = _box_intercepts(
                im.shape,
                (cy + np.cos(th) * linewidth / 2, cx - np.sin(th) * linewidth / 2),
                phi,
                line_len,
            )
            spm, epm = _box_intercepts(
                im.shape,
                (cy - np.cos(th) * linewidth / 2, cx + np.sin(th) * linewidth / 2),
                phi,
                line_len,
            )
            color = kwargs.get("color", "r")
            ax1.fill(
                [spp[1], epp[1], epm[1], spm[1]],
                [spp[0], epp[0], epm[0], spm[0]],
                alpha=0.3,
                facecolor=color,
                edgecolor=None,
            )
            ax1.plot([sp[1], ep[1]], [sp[0], ep[0]], color=color, linewidth=0.5)
        else:
            ax1.plot(
                [sp[1], ep[1]],
                [sp[0], ep[0]],
                color=kwargs.get("color", "r"),
                linewidth=1,
            )

        ax1.set_xlim([0, im.shape[1] - 1])
        ax1.set_ylim([im.shape[0] - 1, 0])

        plt.show()

    return profile


def _box_intercepts(
    dims: Tuple[int, int], center: Tuple[int, int], phi: float, line_len: int = -1
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Calculate box intercept points for a line in a box.

    Args:
        dims (tuple): Dimensions of the box (dy, dx).
        center (tuple): Center of the line (cy, cx).
        phi (float): Angle of the line in degrees.
        line_len (int): Length of the line.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: Start and end points of the line.
    """
    dy, dx = dims
    cy, cx = center
    phir = np.deg2rad(phi)
    tphi = np.tan(phir)
    tphi2 = np.tan(phir - np.pi / 2)

    epy = round((dx - cx) * tphi + cy)
    if 0 <= epy < dy:
        epx = dx - 1
    elif epy < 0:
        epy = 0
        epx = round(cx + cy * tphi2)
    else:
        if phir == 0:
            raise ValueError(f"Center y = {cy} and dimy = {dy} with phi == 0")
        epy = dy - 1
        epx = round(cx + (dy - cy) / tphi)

    spy = round(cy - cx * tphi)
    if 0 <= spy < dy:
        spx = 0
    elif spy >= dy:
        spy = dy - 1
        spx = round(cx - (dy - cy) * tphi2)
    else:
        spy = 0
        spx = round(cx - cy / tphi)

    if line_len > 0:
        sp2y = cy - np.sin(np.deg2rad(phi)) * line_len / 2
        sp2x = cx - np.cos(np.deg2rad(phi)) * line_len / 2
        ep2y = cy + np.sin(np.deg2rad(phi)) * line_len / 2
        ep2x = cx + np.cos(np.deg2rad(phi)) * line_len / 2
        spy = spy if sp2y < 0 else sp2y
        spx = spx if sp2x < 0 else sp2x
        epy = epy if ep2y > dy - 1 else ep2y
        epx = epx if ep2x > dx - 1 else ep2x

    sp = (spy, spx)  # start point
    ep = (epy, epx)  # end point
    return sp, ep
