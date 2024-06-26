import numpy as np
from pathlib import Path
from PyLorentz.io.read import read_image, read_json
from PyLorentz.io.write import write_json
import os


"""
Recommended file structure

- tfs1
    - dm_files
        - f1.dm3
        - f2.dm3
        - ...
    - aligned_stack.tif
    - aligned_stack_flip.tif
    - tfs_metadata.json
        * contains defocus values and scale, defocus values -2, -1, ..., 2

Options
    * including a aligned_stack path will set flip=True, but manually
      setting flip=True will then assume that the loaded aligned stack
      includes a unflip / flip concatenated stacks
    * aligned_files_list


Optional legacy filepath same as previous


if metadata file given, that will override
if not, then will try to get scale from tif, and will ask for scale and defocus

len(defvals) = len(tifstack) = len(flipstack) if there is one

"""


class DefocusedDataset(object):

    def __init__(
        self,
        imstack: np.ndarray = None,
        flipstack: np.ndarray = None,
        flip: bool = False,
        scale: float = None,
        defvals: np.ndarray = None,
        use_mask: bool = True,
        verbose: bool = True,
    ):
        assert np.ndim(imstack) == 3

        self.imstack = np.array(imstack)
        self.flipstack = np.array(flipstack) if flipstack is not None else np.array([])
        self.flip = flip if len(self.flipstack) > 0 else False  # either if flipstack
        self._scale = scale
        self._defvals = defvals
        self._verbose = verbose

        self.shape = imstack.shape[1:]
        self.num_files = len(self.imstack) + len(self.flipstack)

        self._crop = {
            "top": 0,
            "bottom": self.shape[0],
            "left": 0,
            "right": self.shape[1],
        }
        self._transformations = {
            "rotation": 0,
            "y_transl": 0,
            "x_transl": 0,
        }

        self.preprocess(filter_hotpix=True, use_mask=use_mask)

        return

    @classmethod
    def load(
        cls,
        aligned_file: str | os.PathLike,
        aligned_flip_file: str | os.PathLike | None = None,
        metadata_file: str | os.PathLike | None = None,
        flip: bool = False,
        scale: float | None = None,
        defocus_values: list | None = None,
        dump_metadata: bool = True,
        use_mask: bool = True,
        legacy_data_loc: str | os.PathLike | None = None,
        legacy_fls_file: str | os.PathLike | None = None,
        legacy_flip_fls_file: str | os.PathLike | None = None,
        verbose: bool = True,
    ):
        """
        relevant metadata:
            - defocus for all images
            - scale (assumes uniform)

        Loading options:
            - (recommended): aligned stack filepath (and flip_aligned path), list of defocus values, scale
                and have a recommended saving the metadata
            - (legacy): fls file(s) + orig_ims w/ metadata + aligned stack
            - aligned_stack + manual metadata
            - numpy aligned stack + manual metadata


        Attributes:
            - scale
            - defocus_vals
            - tfs_aligned
            - filepath_tfs_aligned
            - flip_tfs_aligned
            - filepath_tfs_flip_aligned
            - _tfs_orig
            - _tfs_filepath_orig
            - flip

        Later defined attributes/methods
            - ROI

        """
        _verbose = verbose
        vprint = print if _verbose >= 1 else lambda *a, **k: None

        ### load aligned images
        assert Path(aligned_file).exists()
        imstack, mdata = read_image(aligned_file)
        if aligned_flip_file is not None:
            assert Path(aligned_flip_file).exists()
            flipstack, _ = read_image(aligned_flip_file)
        else:
            if flip:  # flipstack combines flip and unflip
                assert (
                    len(imstack) % 2 == 0
                ), f"Flip is True but imstack has odd length: {len(imstack)}"
                flipstack = imstack[len(imstack) // 2 :].copy()
                imstack = imstack[: len(imstack) // 2]
            else:
                assert (
                    len(imstack) % 2 == 1
                ), f"Flip is False but imstack has even length: {len(imstack)}"
                flipstack = None

        ### load metadata
        if metadata_file is not None:
            mdata = read_json(metadata_file)
            loaded_defvals = mdata["defocus_values"]
            if mdata["defocus_unit"] != "nm":
                raise NotImplementedError(
                    f"Unknown defocus unit {mdata['defocus_unit']}"
                )
            loaded_scale = mdata["scale"]
            if mdata["scale_unit"] != "nm":
                raise NotImplementedError(f"Unknown scale unit {mdata['scale_unit']}")
        elif legacy_data_loc is not None:
            raise NotImplementedError
        else:
            assert scale is not None and defocus_values is not None
            scale = float(scale)
            defvals = np.array(defocus_values)

        if scale is not None and (
            metadata_file is not None or legacy_data_loc is not None
        ):
            vprint(
                f"Overwriting loaded scale, {loaded_scale:.2f} nm/pix, with set value {scale:.2f} nm/pix"
            )
            scale = float(scale)
        else:
            scale = float(loaded_scale)

        if defocus_values is not None and (
            metadata_file is not None or legacy_data_loc is not None
        ):
            vprint(
                f"Overwriting scale value:\n\t{loaded_scale}\nwith set value:\n\t{scale}"
            )
            defvals = np.array(defocus_values)
        else:
            defvals = np.array(loaded_defvals)

        if metadata_file is None and dump_metadata:
            new_mdata_file = aligned_file.absolute().parents[0] / (
                aligned_file.stem + "_mdata.json"
            )
            new_mdata_dict = {
                "scale": scale,
                "scale_unit": "nm",
                "defocus_values": defvals,
                "defocus_unit": "nm",
            }
            vprint("Writing new metadata file:")
            write_json(new_mdata_dict, new_mdata_file)

        DD = cls(
            imstack=imstack,
            flipstack=flipstack,
            flip=flip,
            scale=scale,
            defvals=defvals,
            use_mask=use_mask,
        )

        return DD

    @classmethod
    def load_DD(cls, filepath: str | os.PathLike):
        """
        Load from saved json file
        """
        return cls(filepath)

    def save_DD(self, filepath="", copy_data: bool = False):
        """
        Save self dict as json and either list of filepaths, or full datasets as specified
        make sure includes crop and such
        """
        return

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val: float):
        if val > 0:
            self._scale = val
        else:
            raise ValueError(f"scale must be > 0, received {val}")

    @property
    def defvals(self):
        return self._defvals

    @defvals.setter
    def defvals(self, vals: float):
        if len(vals) != len(self.imstack):
            raise ValueError(
                f"defvals must have same length as imstack, should be {len(self.imstack)} but was {len(vals)}"
            )
        self._defvals = vals

    def dump_metadata(self):
        # save a json file to the location of the aligned stack
        return

    def preprocess(self, filter_hotpix=True, use_mask=True, median_filter=False):
        # filter hotpixels, option for median filter

        return

    def make_mask(
        self,
    ):

        return

    def crop(self):

        return

    # def __len__
