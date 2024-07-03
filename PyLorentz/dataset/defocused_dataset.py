import numpy as np
from pathlib import Path
from PyLorentz.io.read import read_image, read_json
from PyLorentz.io.write import write_json
from .base_dataset import BaseDataset
from .data_legacy import legacy_load
import os
import scipy.ndimage as ndi
from PyLorentz.utils.filter import filter_hotpix
from tqdm import tqdm
import time


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

notes
len(defvals) = len(tifstack) = len(flipstack) if there is one

"""


class DefocusedDataset(BaseDataset):

    def __init__(
        self,
        imstack: np.ndarray = None,
        flipstack: np.ndarray = None,
        flip: bool = False,
        scale: float | None = None,
        defvals: np.ndarray | None = None,
        mask: bool = True,
        data_dir: os.PathLike|None=None,
        verbose: int | bool = 1,
        **kwargs,
    ):
        assert np.ndim(imstack) == 3, f"Bad input shape {imstack.shape}"
        BaseDataset.__init__(self, imshape=imstack.shape[1:], data_dir=data_dir)

        vprint = print if verbose >= 1 else lambda *a, **k: None

        self.imstack = np.array(imstack)
        self.flipstack = np.array(flipstack) if flipstack is not None else np.array([])
        self._flip = flip if len(self.flipstack) > 0 else False  # either if flipstack
        self._scale = scale
        self.defvals = defvals
        self._verbose = verbose
        self._apply_mask = mask
        self._preprocessed = False

        self._orig_imstack = self.imstack.copy()
        self._orig_flipstack = self.flipstack.copy()
        self._orig_shape = self._orig_imstack.shape[1:]
        self.mask = None

        if scale is None:
            vprint("No scale found. Set scale with DD.scale = <x> [nm/pix]")
        if defvals is None:
            vprint("No scale found. Set scale with DD.scale = <x> [nm/pix]")

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
        mask: bool = True,
        legacy_data_loc: str | os.PathLike | None = None,
        legacy_fls_filename: str | os.PathLike | None = None,
        verbose: int | bool = True,
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
        vprint = print if verbose >= 1 else lambda *a, **k: None

        ### load metadata
        if metadata_file is not None:
            mdata = cls._read_mdata(metadata_file)
            loaded_defvals = mdata["defocus_values"]
            loaded_scale = mdata["scale"]
        elif legacy_data_loc is not None:
            loaded_scale, loaded_defvals = legacy_load(
                legacy_data_loc, legacy_fls_filename
            )
        else:
            assert scale is not None and defocus_values is not None
            scale = float(scale)
            defvals = np.array(defocus_values)
            loaded_scale = None
            loaded_defvals = None

        if loaded_scale is not None:
            if scale is None:
                scale = float(loaded_scale)
            else:
                vprint(
                    f"Overwriting loaded scale, {loaded_scale:.2f} nm/pix,"
                    + f"with set value {scale:.2f} nm/pix"
                )
                scale = float(scale)

        if loaded_defvals is not None:
            if defocus_values is not None:
                vprint(
                    f"Overwriting loaded defocus values:\n\t{loaded_defvals}"
                    + f"with set value:\n\t{defvals}"
                )
                defvals = np.array(defocus_values)
            else:
                defvals = np.array(loaded_defvals)

        ### load aligned images
        aligned_file = Path(aligned_file)
        assert aligned_file.exists()
        imstack, _mdata = read_image(aligned_file)
        if "scale" in _mdata and scale is None:
            scale = _mdata["scale"]
        if aligned_flip_file is not None:
            aligned_flip_file = Path(aligned_flip_file)
            assert aligned_flip_file.exists()
            flipstack, _ = read_image(aligned_flip_file)
            flip = True
        else:
            if len(imstack) % 2 == 0:
                assert len(imstack) == 2 * len(
                    defvals
                ), f"Imstack has even length ({len(imstack)}), but does not match \
                    2 * #defocus_values ({len(defvals)}) for a flip/unflip reconstruction."
                flip = True
                flipstack = imstack[len(imstack) // 2 :].copy()
                imstack = imstack[: len(imstack) // 2]
            else:
                assert len(imstack) == len(
                    defvals
                ), f"Imstack has odd length ({len(imstack)})\
                    that does not match the # defocus values ({len(defvals)})"
                if flip:
                    vprint(
                        f"Flip was True but only a single tfs was given. Setting Flip=False"
                    )
                    flip = False
                flipstack = None

        if metadata_file is None and dump_metadata:
            new_mdata_file = aligned_file.absolute().parents[0] / (
                aligned_file.stem + "_mdata.json"
            )
            if not new_mdata_file.exists():
                new_mdata_dict = {
                    "scale": scale,
                    "scale_unit": "nm",
                    "defocus_values": defvals,
                    "defocus_unit": "nm",
                }
                vprint("Writing new metadata file:")
                write_json(new_mdata_dict, new_mdata_file)

        data_dir = aligned_file.absolute().parents[0]

        dd = cls(
            imstack=imstack,
            flipstack=flipstack,
            flip=flip,
            scale=scale,
            defvals=defvals,
            mask=mask,
            data_dir=data_dir,
            verbose=verbose,
        )

        return dd

    @property
    def flip(self):
        return self._flip

    @property
    def defvals(self):
        return self._defvals

    @property
    def defvals_index(self):
        return self._defvals[self.len_tfs//2+1:].copy()

    @defvals.setter
    def defvals(self, vals: float):
        if not isinstance(vals, (np.ndarray, list, tuple)):
            raise TypeError(
                f"defvals type should be list or ndarray, found {type(vals)}"
            )

        if len(vals) != len(self.imstack):
            raise ValueError(
                f"defvals must have same length as imstack, should be {len(self.imstack)} but was {len(vals)}"
            )
        if len(vals) %2 != 1:
            raise ValueError(
                f"Expects a tfs with odd number of images, received even number of defocus values: {len(vals)}"
            )
        for a0 in range(len(vals)//2):
            if np.sign(vals[a0]) == np.sign(vals[-1*(a0+1)]):
                raise ValueError(
                    f"Underfocus and overfocus values have same sign: {vals[a0]} and {vals[-1*(a0+1)]}"
                )
        self._defvals = np.array(vals)

    @property
    def full_stack(self):
        return np.concatenate([self.imstack, self.flipstack])

    @property
    def full_defvals(self):
        if self._flip:
            return np.concatenate([self.defvals, self.defvals])
        else:
            return self.defvals

    @property
    def infocus(self):
        inf_index = self.len_tfs//2
        if self._flip:
            ave_infocus = (self.imstack[inf_index] + self.flipstack[inf_index]) / 2
            return ave_infocus
        else:
            return self.imstack[inf_index]

    @property
    def shape(self):
        return self.imstack.shape[1:]

    @property
    def len_tfs(self):
        if self._flip:
            assert len(self.imstack) == len(self.flipstack)
        return len(self.imstack)

    @property
    def _len_full_tfs(self):
        return len(self.imstack) + len(self.flipstack)

    def preprocess(
        self, hotpix: bool = True, median_filter_size: int | None = None, fast: bool=True, **kwargs
    ):
        vprint = print if self._verbose >= 1 else lambda *a, **k: None

        self._make_mask(self._apply_mask)

        # filter hotpixels, option for median filter
        self.imstack = self._orig_imstack.copy()
        self.flipstack = self._orig_flipstack.copy()

        self.imstack *= self.mask[None]
        self.flipstack *= self.mask[None]

        if hotpix:
            vprint("Filtering hot/dead pixels")
            for i in tqdm(range(self.len_tfs)):
                self.imstack[i] = filter_hotpix(self.imstack[i], fast=fast, **kwargs)

                if self._flip:
                    self.flipstack[i] = filter_hotpix(self.flipstack[i], fast=fast, **kwargs)

        if median_filter_size is not None:
            self.imstack = ndi.median_filter(
                self.imstack, size=(1, median_filter_size, median_filter_size)
            )
            if self._flip:
                self.flipstack = ndi.median_filter(
                    self.flipstack, size=(1, median_filter_size, median_filter_size)
                )

        # subtract minimum
        self.imstack -= self.imstack.min(axis=(1,2))[..., None, None]
        if self._flip:
            self.flipstack -= self.flipstack.min(axis=(1,2))[..., None, None]

        self._preprocessed = True
        self._orig_imstack_preprocessed = self.imstack.copy()
        self._orig_flipstack_preprocessed = self.flipstack.copy()

        return

    def _make_mask(self, no_mask: bool = False, threshold: float = 0):
        """Sets self.mask to be a binary bounding mask from imstack and flipstack.

        Makes all images binary using a threshold value, and multiplies these arrays.

        # The inverse Laplacian reconstruction does not deal well with a mask that
        # is all ones, that is accounted for in TIE() function rather than here.

        Args:
            no_mask (bool): (`optional`) If True, will set mask to uniform ones.
                Default is False.
            threshold (float): (`optional`) Pixel value with which to threshold the
                images. Default is 0.

        Returns:
            None. Assigns result to self.mask()
        """
        if no_mask:
            self.mask = np.ones(self.shape)
        elif self.len_tfs == 1:
            # SITIE should have no mask
            self.mask = np.ones(self.shape)

        mask = np.where(self.full_stack > threshold, 1, 0)
        mask = np.prod(mask, axis=0)

        # shrink mask slightly
        iters = int(min(15, self.shape[0] // 250, self.shape[1] // 250))
        if iters >= 1:  # binary_erosion fails if iterations=0
            mask = ndi.morphology.binary_erosion(mask, iterations=iters)
        mask = mask.astype(np.float32, copy=False)
        mask = ndi.gaussian_filter(mask, 2)
        self.mask = mask
        return



    def apply_transforms(self, v=1):
        # apply rotation -> crop, and set imstack and flipstack
        # for either stack or single image
        # might have to copy this over to show functions as well
        if self._verbose or v>=1:
            print("Applying transformations...")
        if self._preprocessed:
            imstack = self._orig_imstack_preprocessed.copy()
            flipstack = self._orig_flipstack_preprocessed.copy()
        else:
            imstack = self._orig_imstack.copy()
            flipstack = self._orig_flipstack.copy()
        if self._transformations["rotation"] != 0:
            imstack = ndi.rotate(imstack,
                                 self._transformations["rotation"],
                                 reshape=False,
                                 axes=(1,2))
            if self._flip:
                flipstack = ndi.rotate(flipstack,
                                    self._transformations["rotation"],
                                    reshape=False,
                                    axes=(1,2))

        top = self._transformations["top"]
        bottom = self._transformations["bottom"]
        left = self._transformations["left"]
        right = self._transformations["right"]
        imstack = imstack[:, top:bottom, left:right]
        if self._flip:
            flipstack = flipstack[:, top:bottom, left:right]

        self.imstack = imstack
        self.flipstack = flipstack
        return

    def select_ROI(self, image: np.ndarray|None=None):
        # select image as infocus orig image if none given
        if image is None:
            inf_ind = self.len_tfs//2
            if self._flip:
                if self._preprocessed:
                    image = self._orig_imstack_preprocessed[inf_ind] + self._orig_flipstack_preprocessed[inf_ind]
                else:
                    image = self._orig_imstack[inf_ind] + self._orig_flipstack[inf_ind]
            else:
                if self._preprocessed:
                    image = self._orig_imstack_preprocessed[inf_ind]
                else:
                    image = self._orig_imstack[inf_ind]
            if self.mask is not None:
                image *= self.mask
        else:
            assert image.shape == self._orig_shape

        self._select_ROI(image)

        return


    @classmethod
    def load_DD(cls, filepath: str | os.PathLike):
        """
        Load from saved json file
        """
        return cls(filepath)

    def save_DD(self, filepath="", copy_data: bool = False):
        """
        Save self dict as json and either list of filepaths, or full datasets as specified
        make sure includes crop and such.
        could/should probably build this around EMD file, see how datacubes are saved
        """
        return