"""Helper functions for TIE.

An assortment of helper functions that load images, pass data, and generally
are used in the reconstruction. Additionally, a couple of functions used for
displaying images and stacks.

Author: Arthur McCray, ANL, Summer 2019.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import hyperspy.api as hs
import sys
from skimage import io
from scipy.ndimage.filters import median_filter
from scipy import ndimage
from ipywidgets import interact
import hyperspy  # just for checking type in show_stack.
from copy import deepcopy
from TIE_params import TIE_params
import textwrap
import os


# ============================================================= #
#      Functions used for loading and passing the TIE data      #
# ============================================================= #


def load_data(
    path=None, fls_file="", al_file="", flip=None, flip_fls_file=None, filtersize=3
):
    """Load files in a directory (from a .fls file) using hyperspy.

    For more information on how to organize the directory and load the data, as
    well as how to setup the .fls file please refer to the README or the
    TIE_template.ipynb notebook.

    Args:
        path (str): Location of data directory.
        fls_file (str): Name of the .fls file which contains the image names and
            defocus values.
        al_file (str): Name of the aligned stack image file.
        flip (Bool): True if using a flip stack, False otherwise. Uniformly
            thick films can be reconstructed without a flip stack. The
            electrostatic phase shift will not be reconstructed.
        flip_fls_file (str): Name of the .fls file for the flip images if they
            are not named the same as the unflip files. Will only be applied to
            the /flip/ directory.
        filtersize (int): (`optional`) The images are processed with a median
            filter to remove hot pixels which occur in experimental data. This
            should be set to 0 for simulated data, though generally one would
            only use this function for experimental data.

    Returns:
        list: List of length 3, containing the following items:

        - imstack: array of hyperspy signal2D objects (one per image)
        - flipstack: array of hyperspy signal2D objects, empty array if
          flip == False
        - ptie: TIE_params object holding a reference to the imstack and many
          other parameters.

    """

    unflip_files = []
    flip_files = []

    # Finding the unflip fls file
    path = os.path.abspath(path)
    if not fls_file.endswith(".fls"):
        fls_file += ".fls"
    if os.path.isfile(os.path.join(path, fls_file)):
        fls_full = os.path.join(path, fls_file)
    elif os.path.isfile(os.path.join(path, "unflip", fls_file)):
        fls_full = os.path.join(path, "unflip", fls_file)
    elif os.path.isfile(os.path.join(path, "tfs", fls_file)) and not flip:
        fls_full = os.path.join(path, "tfs", fls_file)
    else:
        print("fls file could not be found.")
        sys.exit(1)

    if flip_fls_file is None:  # one fls file given
        fls = []
        with open(fls_full) as file:
            for line in file:
                fls.append(line.strip())

        num_files = int(fls[0])
        if flip:
            for line in fls[1 : num_files + 1]:
                unflip_files.append(os.path.join(path, "unflip", line))
            for line in fls[1 : num_files + 1]:
                flip_files.append(os.path.join(path, "flip", line))
        else:
            if os.path.isfile(os.path.join(path, "tfs", fls[2])):
                tfs_dir = "tfs"
            else:
                tfs_dir = "unflip"
            for line in fls[1 : num_files + 1]:
                unflip_files.append(os.path.join(path, tfs_dir, line))

    else:  # there are 2 fls files given
        if not flip:
            print(
                textwrap.dedent(
                    """
                You probably made a mistake.
                You're defining both unflip and flip fls files but have flip=False.
                Proceeding anyways, will only load unflip stack (if it doesnt break).\n"""
                )
            )
        # find the flip fls file
        if not flip_fls_file.endswith(".fls"):
            flip_fls_file += ".fls"
        if os.path.isfile(os.path.join(path, flip_fls_file)):
            flip_fls_full = os.path.join(path, flip_fls_file)
        elif os.path.isfile(os.path.join(path, "flip", flip_fls_file)):
            flip_fls_full = os.path.join(path, "flip", flip_fls_file)

        fls = []
        flip_fls = []
        with open(fls_full) as file:
            for line in file:
                fls.append(line.strip())

        with open(flip_fls_full) as file:
            for line in file:
                flip_fls.append(line.strip())

        assert int(fls[0]) == int(flip_fls[0])
        num_files = int(fls[0])
        for line in fls[1 : num_files + 1]:
            unflip_files.append(os.path.join(path, "unflip", line))
        for line in flip_fls[1 : num_files + 1]:
            flip_files.append(os.path.join(path, "flip", line))

    # Actually load the data using hyperspy
    imstack = hs.load(unflip_files)
    if flip:
        flipstack = hs.load(flip_files)
    else:
        flipstack = []

    # convert scale dimensions to nm
    for sig in imstack + flipstack:
        sig.axes_manager.convert_units(units=["nm", "nm"])

    if unflip_files[0][-4:] != ".dm3" and unflip_files[0][-4:] != ".dm4":
        # if not dm3's then they generally don't have the title metadata.
        for sig in imstack + flipstack:
            sig.metadata.General.title = sig.metadata.General.original_filename

    # load the aligned tifs and update the dm3 data to match
    # The data from the dm3's will be replaced with the aligned image data.
    try:
        al_tifs = io.imread(os.path.join(path, al_file))
    except FileNotFoundError as e:
        print("Incorrect aligned stack filename given.")
        raise e

    if flip:
        tot_files = 2 * num_files
    else:
        tot_files = num_files

    for i in range(tot_files):
        # pull slices from correct axis, assumes fewer slices than images are tall
        if al_tifs.shape[0] < al_tifs.shape[2]:
            im = al_tifs[i]
        elif al_tifs.shape[0] > al_tifs.shape[2]:
            im = al_tifs[:, :, i]
        else:
            print("Bad stack\n Or maybe the second axis is slice axis?")
            print("Loading failed.\n")
            sys.exit(1)

        # then median filter to remove "hot pixels"
        im = median_filter(im, size=filtersize)

        # and assign to appropriate stack
        if i < num_files:
            print("loading unflip:", unflip_files[i])
            imstack[i].data = im
        else:
            j = i - num_files
            print("loading flip:", flip_files[j])
            flipstack[j].data = im

    # read the defocus values
    defvals = fls[-(num_files // 2) :]
    assert num_files == 2 * len(defvals) + 1
    defvals = [float(i) for i in defvals]  # defocus values +/-

    # Create a TIE_params object
    ptie = TIE_params(imstack, flipstack, defvals, flip, path)
    print("Data loaded successfully.")
    return (imstack, flipstack, ptie)


def load_data_GUI(path, fls_file1, fls_file2, al_file="", single=False, filtersize=3):
    """Load files in a directory (from a .fls file) using hyperspy.

    For more information on how to organize the directory and load the data, as
    well as how to setup the .fls file please refer to the README or the
    TIE_template.ipynb notebook.

    Args:
        path (str): Location of data directory.
        fls_file1 (str): Name of the .fls file which contains the image names and
            defocus values.
        fls_file2 (str): Name of the .fls file for the flip images if they
            are not named the same as the unflip files. Will only be applied to
            the /flip/ directory.
        al_file (str): Name of the aligned stack image file.
        single (Bool): True if using a single stack, False otherwise. Uniformly
            thick films can be reconstructed with a single stack. The
            electrostatic phase shift will not be reconstructed.
        filtersize (int): (`optional`) The images are processed with a median
            filter to remove hot pixels which occur in experimental data. This
            should be set to 0 for simulated data, though generally one would
            only use this function for experimental data.

    Returns:
        list: List of length 3, containing the following items:

        - imstack: array of hyperspy signal2D objects (one per image)
        - flipstack: array of hyperspy signal2D objects, empty array if
          flip == False
        - ptie: TIE_params object holding a reference to the imstack and many
          other parameters.

    """

    unflip_files = []
    flip_files = []

    if fls_file2 is None:  # one fls file given
        u_files = []
        with open(fls_file1) as file:
            for line in file:
                u_files.append(line.strip())

        num_files = int(u_files[0])
        if not single:
            for line in u_files[1 : num_files + 1]:
                unflip_files.append(os.path.join(path, "unflip", line))
            for line in u_files[1 : num_files + 1]:
                flip_files.append(os.path.join(path, "flip", line))
        else:
            if os.path.exists(os.path.join(path, "tfs")):
                sub_dir = "tfs"
            else:
                sub_dir = "unflip"
            for line in u_files[1 : num_files + 1]:
                unflip_files.append(os.path.join(path, sub_dir, line))

    else:  # there are 2 fls files given
        if single:
            print(
                textwrap.dedent(
                    """
                You probably made a mistake.
                You're defining both unflip and flip fls files but have flip=False.
                Proceeding anyways, will only load unflip stack (if it doesnt break).\n"""
                )
            )

        u_files = []
        f_files = []
        with open(fls_file1) as file:
            for line in file:
                u_files.append(line.strip())

        with open(fls_file2) as file:
            for line in file:
                f_files.append(line.strip())

        assert int(u_files[0]) == int(f_files[0])
        num_files = int(u_files[0])
        for line in u_files[1 : num_files + 1]:
            unflip_files.append(os.path.join(path, "unflip", line))
        for line in f_files[1 : num_files + 1]:
            flip_files.append(os.path.join(path, "flip", line))

    # Actually load the data using hyperspy
    imstack = hs.load(unflip_files)
    if not single:
        flipstack = hs.load(flip_files)
    else:
        flipstack = []

    # convert scale dimensions to nm
    for sig in imstack + flipstack:
        sig.axes_manager.convert_units(units=["nm", "nm"])

    if unflip_files[0][-4:] != ".dm3" and unflip_files[0][-4:] != ".dm4":
        # if not dm3's then they generally don't have the title metadata.
        for sig in imstack + flipstack:
            sig.metadata.General.title = sig.metadata.General.original_filename

    # load the aligned tifs and update the dm3 data to match
    # The data from the dm3's will be replaced with the aligned image data.
    try:
        al_tifs = io.imread(al_file)
    except FileNotFoundError as e:
        print("Incorrect aligned stack filename given.")
        raise e

    if not single:
        tot_files = 2 * num_files
    else:
        tot_files = num_files

    for i in range(tot_files):
        # pull slices from correct axis, assumes fewer slices than images are tall
        if al_tifs.shape[0] < al_tifs.shape[2]:
            im = al_tifs[i]
        elif al_tifs.shape[0] > al_tifs.shape[2]:
            im = al_tifs[:, :, i]
        else:
            print("Bad stack\n Or maybe the second axis is slice axis?")
            print("Loading failed.\n")
            sys.exit(1)

        # then median filter to remove "hot pixels"
        im = median_filter(im, size=filtersize)

        # and assign to appropriate stack
        if i < num_files:
            print("loading unflip:", unflip_files[i])
            imstack[i].data = im
        else:
            j = i - num_files
            print("loading flip:", flip_files[j])
            flipstack[j].data = im

    # read the defocus values
    defvals = u_files[-(num_files // 2) :]
    assert num_files == 2 * len(defvals) + 1
    defvals = [float(i) for i in defvals]  # defocus values +/-

    # Create a TIE_params object
    if single:
        single = None
    else:
        single = True
    ptie = TIE_params(imstack, flipstack, defvals, single, path)
    print("Data loaded successfully.")
    return (imstack, flipstack, ptie)


def select_tifs(i, ptie, long_deriv=False):
    """Returns a list of the images which will be used in TIE() or SITIE().

    Uses copy.deepcopy() as the data will be modified in the reconstruction
    process, and we don't want to change the original data. This method is
    likely not best practice. In the future this might get moved to the
    TIE_params class.

    Args:
        i (int): Index of defvals for which to select the tifs.
        ptie (``TIE_params`` object): Parameters for reconstruction, holds the
            images.

    Returns:
        list: List of np arrays, return depends on parameters:

        - if long_deriv == False:

            - if ptie.flip == True: returns [ +- , -- , 0 , ++ , -+ ]
            - elif ptie.flip == False:  returns [+-, 0, ++]
            - where first +/- is unflip/flip, second +/- is over/underfocus.
              E.g. -+ is the flipped overfocused image. 0 is the averaged
              infocus image.

        - elif long_deriv == True: returns all images in imstack followed by
          all images in flipstack.

    """
    if long_deriv:
        recon_tifs = []
        for sig in ptie.imstack:
            recon_tifs.append(sig.data)
        if ptie.flip:
            for sig in ptie.flipstack:
                recon_tifs.append(sig.data)

    else:
        if i < 0:
            i = len(ptie.defvals) + i
        num_files = ptie.num_files
        under = num_files // 2 - (i + 1)
        over = num_files // 2 + (i + 1)
        imstack = ptie.imstack
        flipstack = ptie.flipstack
        if ptie.flip:
            recon_tifs = [
                imstack[under].data,  # +-
                flipstack[under].data,  # --
                (imstack[num_files // 2].data + flipstack[num_files // 2].data)
                / 2,  # infocus
                imstack[over].data,  # ++
                flipstack[over].data,  # -+
            ]
        else:
            recon_tifs = [
                imstack[under].data,  # +-
                imstack[num_files // 2].data,  # 0
                imstack[over].data,  # ++
            ]
    try:
        recon_tifs = deepcopy(recon_tifs)
    except TypeError:
        print("TypeError in select_tifs deepcopy. Proceeding with originals.")
    return recon_tifs


def dist(ny, nx, shift=False):
    """Creates a frequency array for Fourier processing.

    Args:
        ny (int): Height of array
        nx (int): Width of array
        shift (bool): Whether to center the frequency spectrum.

            - False: (default) smallest values are at the corners.
            - True: smallest values at center of array.

    Returns:
        ``ndarray``: Numpy array of shape (ny, nx).
    """
    ly = (np.arange(ny) - ny / 2) / ny
    lx = (np.arange(nx) - nx / 2) / nx
    [X, Y] = np.meshgrid(lx, ly)
    q = np.sqrt(X ** 2 + Y ** 2)
    if not shift:
        q = np.fft.ifftshift(q)
    return q


def scale_stack(imstack):
    """Scale a stack of images so all have the same total intensity.

    Args:
        imstack (list): List of 2D arrays.

    Returns:
        list: List of same shape as imstack
    """
    imstack = deepcopy(imstack)
    tots = np.sum(imstack, axis=(1, 2))
    t = max(tots) / tots
    for i in range(len(tots)):
        imstack[i] *= t[i]
    return imstack / np.max(imstack)


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


def show_stack(images, ptie=None, origin="upper", title=False):
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
    sig = False
    if type(images[0]) == hyperspy._signals.signal2d.Signal2D:
        sig = True
        imstack = []
        titles = []
        for signal2D in images:
            imstack.append(signal2D.data)
            titles.append(signal2D.metadata.General.title)
        images = np.array(imstack)
    else:
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
        if title:
            if sig:
                plt.title("Image title: {:}".format(titles[i]))
            else:
                plt.title("Stack[{:}]".format(i))

    interact(view_image, i=(0, N - 1))
    return


def show_2D(
    mag_x,
    mag_y,
    mag_z=None,
    a=15,
    l=None,
    w=None,
    title=None,
    color=False,
    hsv=True,
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
            from colorwheel import color_im

            im = ax.matshow(
                color_im(mag_x, mag_y, mag_z, hsvwheel=hsv, rad=rad),
                cmap="gray",
                origin=origin,
            )
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
            qk = ax.quiverkey(
                q,
                X=0.95,
                Y=0.98,
                U=1,
                label=r"$Msat$",
                labelpos="S",
                coordinates="axes",
            )
            qk.text.set_backgroundcolor("w")
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
