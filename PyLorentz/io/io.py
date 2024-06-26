import os
from pathlib import Path
import textwrap
from scipy.ndimage import median_filter
from tifffile import TiffFile
import tifffile
from PyLorentz.tie.TIE_params import TIE_params
# from PyLorentz.io.read import read_image
from itertools import takewhile
from skimage import io as skio
import numpy as np
import io
import sys
from ncempy.io import dm


def load_data(
    path=None, fls_file="", al_file="", flip=None, flip_fls_file=None, filtersize=3
):
    """Load files in a directory (from a .fls file) using ncempy.

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

        - imstack: list of numpy arrays
        - flipstack: list of numpy arrays, empty list if flip == False
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
        raise FileNotFoundError("fls file could not be found.")

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

    f_inf = unflip_files[num_files // 2]
    try:
        _, scale = read_image(f_inf)
    except FileNotFoundError as e:
        print("Unflip infocus file not found.")
        print(f"Please set scale manually with")
        print(">>ptie.scale = XX #nm/pixel")

    try:
        al_stack, _ = read_image(os.path.join(path, al_file))
    except FileNotFoundError as e:
        print("Incorrect aligned stack filename given.")
        raise e

    # quick median filter to remove hotpixels, kinda slow
    print("filtering takes a few seconds")
    al_stack = median_filter(al_stack, size=(1, filtersize, filtersize))

    if flip:
        f_inf_flip = flip_files[num_files // 2]
        try:
            _, scale_flip = read_image(f_inf_flip)
            if scale is not None:
                if round(scale, 3) != round(scale_flip, 3):
                    print("Scale of the two infocus images are different.")
                    print(f"Scale of unflip image: {scale:.4f} nm/pix")
                    print(f"Scale of flip image: {scale_flip:.4f} nm/pix")
                    print("Proceeding with scale from unflip image.")
                    print("If this is incorrect, change value with >>ptie.scale = XX #nm/pixel")
        except FileNotFoundError as e:
            print("Flip infocus file not found.")
            print("If unflip file also not found, scale will not be set automatically.")
        imstack = al_stack[:num_files]
        flipstack = al_stack[num_files:]
    else:
        imstack = al_stack
        flipstack = []

    # necessary for filtered images where negative values occur???
    for i in range(num_files):
        imstack[i] -= imstack[i].min()
        if np.any(flipstack):
            flipstack[i] -= flipstack[i].min()

    # read the defocus values
    defvals = fls[-(num_files // 2) :]
    assert num_files == 2 * len(defvals) + 1
    defvals = [float(i) for i in defvals]  # defocus values +/-

    # Create a TIE_params object
    ptie = TIE_params(imstack, flipstack, defvals, scale, flip, path)
    print("Data loaded successfully.")
    return (imstack, flipstack, ptie)






def read_image(f):
    """Uses Tifffile or ncempy.io load an image and read the scale if there is one.

    Args:
        f (str): file to read

    Raises:
        NotImplementedError: If unknown scale type is given, or Tif series is given.
        RuntimeError: If uknown file type is given, or number of pages in tif is wrong

    Returns:
        tuple:  (image, scale), image given as 2D or 3D numpy array, and scale given in
                nm/pixel if scale is found, or None.
    """
    f = Path(f)
    if f.suffix in [".tif", ".tiff"]:
        with TiffFile(f, mode="r") as tif:
            if tif.imagej_metadata is not None and "unit" in tif.imagej_metadata:
                res = tif.pages[0].tags["XResolution"].value
                scale = res[1] / res[0]  # to nm/pixel
                if tif.imagej_metadata["unit"] == "nm":
                    pass
                elif tif.imagej_metadata["unit"] in ["um", "µm", "micron"]:
                    scale *= 1e3
                elif tif.imagej_metadata["unit"] == "mm":
                    scale *= 1e6
                else:
                    raise NotImplementedError(
                        f'Found unknown scale unit of: {tif.imagej_metadata["unit"]}'
                    )
            else:
                scale = None

            if len(tif.series) != 1:
                raise NotImplementedError(
                    "Not sure how to deal with multi-series stack"
                )
            if len(tif.pages) > 1:  # load as stack
                out_im = []
                for page in tif.pages:
                    out_im.append(page.asarray())
                out_im = np.array(out_im)
            elif len(tif.pages) == 1:  # single image
                out_im = tif.pages[0].asarray()
            else:
                raise RuntimeError(
                    f"Found an unexpected number of pages: {len(tif.pages)}"
                )

    elif f.suffix in [".dm3", ".dm4", ".dm5"]:
        with dm.fileDM(f) as im:
            im = im.getDataset(0)
            assert im["pixelUnit"][0] == im["pixelUnit"][1]
            assert im["pixelSize"][0] == im["pixelSize"][1]

            if im["pixelUnit"][0] == "nm":
                scale = im["pixelSize"][0]
            elif im["pixelUnit"][0] in ["um", "µm"]:
                scale = im["pixelSize"][0] * 1000
            elif im["pixelUnit"][0] == "mm":
                scale = im["pixelSize"][0] * 1e6
            else:
                print(f'Found unknown scale unit of: {tif.imagej_metadata["unit"]}')
                print("Setting scale = -1, please adjust this with ptie.scale = XX #nm/pixel")
                scale = -1
                # raise NotImplementedError(
                #     f'Found unknown scale unit of: {tif.imagej_metadata["unit"]}'
                # )
            out_im = im["data"]

    else:
        print(
            "If a proper image file is given, then\n"
            "likely just need to implement with ncempy.read or something."
        )
        raise RuntimeError(f"Unknown filetype given: {f.suffix}")

    return out_im, scale

