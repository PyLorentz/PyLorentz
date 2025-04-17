from __future__ import annotations
import json
import os
from pathlib import Path
from warnings import warn 

import numpy as np
from ncempy.io import dm as ncempy_dm
from ncempy.io import read as ncemread 
from ncempy.io.emdVelox import fileEMDVelox
from skimage import io as skio
from tifffile import TiffFile


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
    metadata: dict[str, str| float| None] = {
        "filepath": str(f.absolute()),
        "filename": f.stem + "".join(f.suffixes),
    }
    defocus = None
    defocus_unit = None
    beam_energy = None
    scale_unit = None 
    if f.suffix in [".tif", ".tiff"]:
        with TiffFile(f, mode="r") as tif:
            if tif.imagej_metadata is not None and "unit" in tif.imagej_metadata:
                res = tif.pages[0].tags["XResolution"].value
                if res[0] == 0:
                    scale = None
                else:
                    scale = res[1] / res[0]  # to nm/pixel
                scale, scale_unit = _convert_scale(scale, tif.imagej_metadata["unit"])
            else:
                scale = None

            if len(tif.series) != 1:
                raise NotImplementedError("Not sure how to deal with multi-series stack")
            if len(tif.pages) > 1:  # load as stack
                out_im = []
                for page in tif.pages:
                    out_im.append(page.asarray())
                out_im = np.array(out_im)
            elif len(tif.pages) == 1:  # single image
                out_im = tif.pages[0].asarray()
            else:
                raise RuntimeError(f"Found an unexpected number of pages: {len(tif.pages)}")

    elif f.suffix in [".dm3", ".dm4", ".dm5"]: 
        # Don't remember the reason for not using ncempy.read, but I know there was one
        # likely that it didn't handle stacks very well or defocus or something
        with ncempy_dm.fileDM(f) as dm_file:
            dset = dm_file.getDataset(0)
            mdata = dm_file.getMetadata(0)

            if any(["def" in i for i in mdata.keys()]):
                warn("possibly found defocus metadata in dm file? update PyLorentz.io.read.read_image")

            out_im = dset["data"]

            if len(out_im.shape) not in [2,3,4]: 
                raise ValueError(f"don't know how to handle shape {out_im.shape}")
            assert dset["pixelUnit"][-1] == dset["pixelUnit"][-2]
            assert dset["pixelSize"][-1] == dset["pixelSize"][-2]
            pixel_unit = dset["pixelUnit"][-2]
            pixel_size = float(dset["pixelSize"][-2])
            
            scale, scale_unit = _convert_scale(pixel_size, pixel_unit)

            if "Microscope Info Voltage" in mdata:
                beam_energy = float(mdata["Microscope Info Voltage"])
                
            _4dstem_shape = _process_NCEM_TitanX_Tags(dm_file)
            if _4dstem_shape is not None: 
                out_im = np.reshape(out_im, _4dstem_shape + out_im.shape[-2:])

    elif f.suffix in [".emd"]:  # TODO test but make this for dmx as well?
        with fileEMDVelox(f) as emd:
            out_im, mdata = emd.get_dataset(0)
            defocus = float(emd.metaDataJSON["Optics"]["Defocus"]) * 1e9  # nm #type:ignore
            defocus_unit = "nm"
            metadata["AcquisitionTime"] = str(mdata["AcquisitionTime"].time())
            metadata["AcquisitionDate"] = str(mdata["AcquisitionTime"].date())
            assert mdata["pixelUnit"][0] == mdata["pixelUnit"][1]
            assert mdata["pixelSize"][0] == mdata["pixelSize"][1]            
            scale, scale_unit = _convert_scale(mdata["pixelSize"][0], mdata["pixelUnit"][0])
        raise NotImplementedError("look for beam energy in .emd files")

    elif f.suffix in [".ser"]: # Not sure how standard these are, for now using ncempy read
        data = ncemread(f) 
        out_im = data['data']
        assert data['pixelSize'][0] == data['pixelSize'][1]
        assert data['pixelUnit'][0] == data['pixelUnit'][1]
        scale, scale_unit = _convert_scale(data['pixelSize'][0], data['pixelUnit'][0])
        defocus = None
        defocus_unit = "" 
        beam_energy = None 

    elif f.suffix in [".png", ".jpg", ".jpeg"]:
        out_im = skio.imread(f)
        scale = None

    else:
        raise RuntimeError(f"Unknown filetype given: {f.suffix}")

    metadata["scale"] = scale
    metadata["scale_unit"] = scale_unit
    metadata["defocus_values"] = defocus
    metadata["defocus_unit"] = defocus_unit
    metadata["beam_energy"] = beam_energy

    return out_im, metadata

def _convert_scale(scale: float|None, unit:str):
    """
    Converst scale to desired output units. Currently just does nm and 1/nm, but future
    will be modified to convert to desired units. Also will have to check then if/not realspace.
    
    doing diffraction units as nm^-1 or A^-1 to match py4DSTEM. 
    """ 
    if scale is None: 
        return 1, "pixels"
    assert scale > 0, f"Pixel scale should be > 0, got: {scale}"
    
    unit = unit.lower() 
    
    if unit in ["nm", "nanometer"]:
        return scale , "nm"
    elif unit in ["a", "å", "angstrom"]: 
        return scale / 10, "nm" 
    elif unit in ["um", "µm", "micron"]:
        return scale * 1e3, "nm"
    elif unit in ["mm", "millimeter"]:
        return scale * 1e6, "nm"
    elif unit in ["m", "meter"]: 
        return scale * 1e9, "nm"
    elif unit in ["1/nm", "nm^-1"]: 
        return scale, "nm^-1"
    elif unit in ["1/a", "1/å", "a^-1", "å^-1"]: 
        return scale*10, "nm^-1"
    else:
        raise NotImplementedError(f'unknown scale type: {unit}')
    
def read_json(file):
    """
    read json
    """
    with open(file, "r") as f:
        d = json.load(f)
    return d


def _process_NCEM_TitanX_Tags(dmFile) -> tuple[int, int] | None:
    """
    Based on the py4DSTEM function. 
    Check the metadata in the DM File for certain tags which are added by the NCEM TitanX.
    """
    scanx = [v for k, v in dmFile.allTags.items() if "4D STEM Tags.Scan shape X" in k]
    scany = [v for k, v in dmFile.allTags.items() if "4D STEM Tags.Scan shape Y" in k]
    if len(scanx) >= 1 and len(scany) >= 1:
        # TitanX tags found!
        dimy = int(scany[0]) 
        dimx = int(scanx[0])
        return (dimy, dimx) 
    else: 
        return None 