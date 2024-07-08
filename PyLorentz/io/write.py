import os
from pathlib import Path
import textwrap
from scipy.ndimage import median_filter
from tifffile import TiffFile
import tifffile
from PyLorentz.tie.TIE_params import TIE_params
from ncempy.io import dm as ncempy_dm
from ncempy.io.emdVelox import fileEMDVelox
from itertools import takewhile
from skimage import io as skio
import numpy as np
import io
import sys
import json
from PyLorentz.utils.utils import norm_image
import warnings


def write_tif(data, path, scale, v=1, unit="nm", overwrite=True, color=False):
    """
    scale in nm/pixel default,
    saves as float32 if greyscale, or uint8 if color image
    """
    if scale is not None:
        res = 1 / scale
    else:
        res = 0

    if not overwrite:
        path = overwrite_rename(path)

    if v >= 1:
        print("Saving: ", path)

    if color:
        im = (255 * norm_image(data)).astype(np.uint8)
    else:
        if np.ndim(data) == 3:
            if np.shape(data)[0] in [3, 4]:
                warnings.warn("If this is a color image, save with color=True")
        im = data.astype(np.float32)

    tifffile.imwrite(
        path,
        im,
        imagej=True,
        resolution=(res, res),
        metadata={"unit": unit},
    )
    return


save_tif = write_tif  # alias
write_tiff = write_tif  # alias


def overwrite_rename(filepath, spacer="_", incr_number=True):
    """Given a filepath, check if file exists already. If so, add numeral 1 to end,
    if already ends with a numeral increment by 1.

    Args:
        filepath (str): filepath to be checked

    Returns:
        Path: [description]
    """

    filepath = str(filepath)
    file, ext = os.path.splitext(filepath)
    if os.path.isfile(filepath):
        if file[-1].isnumeric() and incr_number:
            file, num = splitnum(file)
            nname = file + str(int(num) + 1) + ext
            return overwrite_rename(nname)
        else:
            return overwrite_rename(file + spacer + "1" + ext, incr_number=True)
    else:
        return Path(filepath)


def overwrite_rename_dir(dirpath, spacer="_"):
    """Given a filepath, check if file exists already. If so, add numeral 1 to end,
    if already ends with a numeral increment by 1.

    Args:
        filepath (str): filepath to be checked

    Returns:
        str: [description]
    """

    dirpath = Path(dirpath)
    if dirpath.is_dir():
        if not any(dirpath.iterdir()):  # directory is empty
            return dirpath
        dirname = dirpath.stem
        if dirname[-1].isnumeric():  # TODO check if this is date format
            dirname, num = splitnum(dirname)
            nname = dirname + str(int(num) + 1) + "/"
            return overwrite_rename_dir(dirpath.parents[0] / nname)
        else:
            return overwrite_rename_dir(dirpath.parents[0] / (dirname + spacer + "1/"))
    else:
        return dirpath


def splitnum(s):
    """split the trailing number off a string. Returns (stripped_string, number)"""
    head = s.rstrip("-.0123456789")
    tail = s[len(head) :]
    return head, tail


def prep_dict_for_json(d):
    """
    still plenty of things it doesn't handle
    """
    def _json_serializable(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, Path):
            return str(val)
        elif isinstance(val, list):
            for i in range(len(val)):
                val[i] = _json_serializable(val[i])
        else:
            return val

    for key, val in d.items():
        d[key] = _json_serializable(val)

    return d


def write_json(dict, path, overwrite=True, v=1):
    path = Path(path)
    d2 = prep_dict_for_json(dict.copy())
    if not path.suffix.lower() in [".json", ".txt"]:
        path = path.parent / (path.name + '.json')

    if path.exists() and not overwrite:
        path = overwrite_rename(path)

    if v>= 1:
        print(f"Saving json {path}")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d2, f, ensure_ascii=False, indent=4, sort_keys=True)

    return