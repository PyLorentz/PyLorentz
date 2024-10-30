import os
from pathlib import Path
import textwrap
from scipy.ndimage import median_filter
from tifffile import TiffFile
import tifffile
from ncempy.io import dm as ncempy_dm
from ncempy.io.emdVelox import fileEMDVelox
from itertools import takewhile
from skimage import io as skio
import numpy as np
import io
import sys
import json


def read_image(f: os.PathLike) -> tuple[np.ndarray, dict]:
    """Uses Tifffile or ncempy.io load an image and read the scale if there is one.

    Args:
        f (str): file to read

    Raises:
        NotImplementedError: If unknown scale type is given, or Tif series is given.
        RuntimeError: If uknown file type is given, or number of pages in tif is wrong

    Returns:
        tuple:  (image, mdata), image given as 2D or 3D numpy array,
            mdata has keys:
                filepath: str
                filename: str
                scale: nm/pixel
                defocus_values: nm
                scale_unit: str
                defocus_unit: str
                beam_energy: float
    """
    f = Path(f)
    if not f.exists():
        raise FileNotFoundError(str(f.absolute()))
    metadata = {
        "filepath": str(f.absolute()),
        "filename": f.stem + "".join(f.suffixes),
    }
    defocus = None
    defocus_unit = None
    beam_energy = None
    if f.suffix in [".tif", ".tiff"]:
        with TiffFile(f, mode="r") as tif:
            if tif.imagej_metadata is not None and "unit" in tif.imagej_metadata:
                res = tif.pages[0].tags["XResolution"].value
                if res[0] == 0:
                    scale = None
                else:
                    scale = res[1] / res[0]  # to nm/pixel
                if tif.imagej_metadata["unit"] == "nm":
                    pass
                elif tif.imagej_metadata["unit"] in ["um", "µm", "micron"]:
                    scale *= 1e3
                elif tif.imagej_metadata["unit"] in ["mm", "millimeter"]:
                    scale *= 1e6
                else:
                    print(f'unknown scale type: {tif.imagej_metadata["unit"]}')
                    raise NotImplementedError
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
        with ncempy_dm.fileDM(f) as im:
            dset = im.getDataset(0)
            mdata = im.getMetadata(0)

            if any(["def" in i for i in mdata.keys()]):
                print("possibly found defocus metadata in dm file. update load!")

            out_im = dset["data"]

            if len(out_im.shape) == 3:
                assert dset["pixelUnit"][2] == dset["pixelUnit"][1]
                assert dset["pixelSize"][2] == dset["pixelSize"][1]
                pixel_unit = dset["pixelUnit"][1]
                pixel_size = dset['pixelSize'][1]
            elif len(out_im.shape) == 2:
                assert dset["pixelUnit"][0] == dset["pixelUnit"][1]
                assert dset["pixelSize"][0] == dset["pixelSize"][1]
                pixel_unit = dset["pixelUnit"][0]
                pixel_size = dset['pixelSize'][0]
            else:
                raise ValueError(f"don't know how to handle shape {out_im.shape}")

            if pixel_unit == "nm":
                scale = pixel_size
            elif pixel_unit == "µm":
                scale = pixel_size * 1000
            else:
                print(f"unknown scale type {pixel_unit}")
                raise NotImplementedError

            if 'Microscope Info Voltage' in mdata:
                beam_energy = mdata['Microscope Info Voltage']

    elif f.suffix in [".emd"]:  # TODO test but make this for dmx as well?
        with fileEMDVelox(f) as emd:
            out_im, mdata = emd.get_dataset(0)
            defocus = float(emd.metaDataJSON["Optics"]["Defocus"]) * 1e9  # nm
            defocus_unit = "nm"
            metadata["AcquisitionTime"] = str(mdata["AcquisitionTime"].time())
            metadata["AcquisitionDate"] = str(mdata["AcquisitionTime"].date())
            assert mdata["pixelUnit"][0] == mdata["pixelUnit"][1]
            assert mdata["pixelSize"][0] == mdata["pixelSize"][1]

            if mdata["pixelUnit"][0] == "nm":
                scale = mdata["pixelSize"][0]
            elif mdata["pixelUnit"][0] == "µm":
                scale = mdata["pixelSize"][0] * 1000
            else:
                print(f"unknown scale type {mdata['pixelUnit'][0]}")
                raise NotImplementedError
        raise NotImplementedError("look for beam energy")

    elif f.suffix in [".png", ".jpg", ".jpeg"]:
        out_im = skio.imread(f)
        scale = None

    else:
        raise RuntimeError(f"Unknown filetype given: {f.suffix}")

    metadata["scale"] = scale
    metadata["scale_unit"] = "nm"
    metadata["defocus_values"] = defocus
    metadata["defocus_unit"] = defocus_unit
    metadata["beam_energy"] = beam_energy

    return out_im, metadata


def read_json(file):
    """
    read json
    """
    with open(file, "r") as f:
        d = json.load(f)
    return d
