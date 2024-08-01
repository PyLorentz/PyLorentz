import json
import os
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tifffile

from PyLorentz.utils.utils import norm_image


def write_tif(
    data: np.ndarray,
    path: os.PathLike,
    scale: float,
    v: Optional[float] = 1,
    unit: Optional[str] = "nm",
    overwrite: Optional[bool] = True,
    color: Optional[bool] = False,
):
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
        print("Saving tif: ", path)

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


def overwrite_rename(
    filepath: os.PathLike, spacer: Optional[bool] = "_", incr_number: Optional[str] = True
):
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


def overwrite_rename_dir(dirpath: os.PathLike, spacer: Optional[str] = "_"):
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
        if dirname[-1].isnumeric():  # TODO add check for if is date format
            dirname, num = splitnum(dirname)
            nname = dirname + str(int(num) + 1) + "/"
            return overwrite_rename_dir(dirpath.parents[0] / nname)
        else:
            return overwrite_rename_dir(dirpath.parents[0] / (dirname + spacer + "1/"))
    else:
        return dirpath


def splitnum(s: str):
    """split the trailing number off a string. Returns (stripped_string, number)"""
    head = s.rstrip("-.0123456789")
    tail = s[len(head) :]
    return head, tail


def prep_dict_for_json(d: any):
    """
    still plenty of things it doesn't handle
    """

    def _json_serializable(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, np.generic):
            return val.item()
        elif isinstance(val, Path):
            return str(val)
        elif isinstance(val, list):
            for i in range(len(val)):
                val[i] = _json_serializable(val[i])
            return val
        # elif isinstance(val, dict): # causes dict -> null?
        #     for key, v2 in val.items():
        #         val[key] = _json_serializable(v2)
        # elif: ## as things come up will need to add
        else:
            return val

    for key, val in d.items():
        d[key] = _json_serializable(val)

    return d


def write_json(d: dict, path: os.PathLike, overwrite: Optional[bool] = True, v: Optional[int] = 1):
    path = Path(path)
    d2 = prep_dict_for_json(d.copy())
    if not path.suffix.lower() in [".json", ".txt"]:
        path = path.parent / (path.name + ".json")

    if path.exists() and not overwrite:
        path = overwrite_rename(path)

    if v >= 1:
        print(f"Saving json: {path}")

    # for key, val in d2.items():
    #     print(f"key: {key} | type {type(val)} ")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(d2, f, ensure_ascii=False, indent=4, sort_keys=True)

    return


def format_defocus(defval: Union[float, int], digits: int = 3, spacer: str = ""):
    "returns a string of defocus value converted to nm, um, or mm as appropriate"

    rnd_digits = len(str(round(defval))) - digits
    rnd_abs = round(defval, -1 * rnd_digits)

    if abs(rnd_abs) < 1e3:  # nm
        return f"{rnd_abs:.0f}{spacer}nm"
    elif abs(rnd_abs) < 1e6:  # um
        return f"{rnd_abs/1e3:.0f}{spacer}um"
    elif abs(rnd_abs) < 1e9:  # mm
        return f"{rnd_abs/1e6:.0f}{spacer}mm"
    else:
        return f"{rnd_abs/1e9:.0f}{spacer}m"
