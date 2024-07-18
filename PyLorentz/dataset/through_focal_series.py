import os
import time
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm

from PyLorentz.io import format_defocus, read_image, write_json
from PyLorentz.utils.filter import filter_hotpix
from PyLorentz.visualize import show_im

from .base_dataset import BaseDataset
from .data_legacy import legacy_load

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


class ThroughFocalSeries(BaseDataset):

    def __init__(
        self,
        imstack: np.ndarray,
        flipstack: Optional[np.ndarray] = None,
        flip: Optional[bool] = False,
        scale: Optional[float] = None,
        defvals: Optional[np.ndarray] = None,
        beam_energy: Optional[float] = None,
        use_mask: Optional[bool] = True,
        simulated: Optional[bool] = False,
        data_dir: os.PathLike | None = None,
        data_files: list[os.PathLike] = [],
        verbose: Optional[int] = 1,
    ):
        imstack = np.array(imstack)
        assert np.ndim(imstack) == 3, f"Bad input shape {imstack.shape}"
        BaseDataset.__init__(
            self,
            imshape=imstack.shape[1:],
            data_dir=data_dir,
            scale=scale,
            verbose=verbose,
        )

        self.imstack = imstack
        if flipstack is not None:
            if np.size(flipstack) != 0:
                if len(flipstack) != len(imstack):
                    raise ValueError(
                        f"len imstack, {len(imstack)} != len flipstack, {len(flipstack)}"
                    )
                self.flipstack = np.array(flipstack)
            else:  # is empty array
                self.flipstack = np.array([])
        else:
            self.flipstack = np.array([])

        self.flip = flip if len(self.flipstack) > 0 else False  # either if flipstack
        self.defvals = defvals
        self.beam_energy = beam_energy
        self._simulated = simulated
        self.data_files = data_files
        if simulated:
            use_mask = False
        self._use_mask = use_mask
        self._preprocessed = False
        self._cropped = False
        self._filtered = False

        self._orig_imstack = self.imstack.copy()
        self._orig_flipstack = self.flipstack.copy()
        self._orig_imstack_preprocessed = None
        self._orig_flipstack_preprocessed = None
        self._imstack_crop = None
        self._flipstack_crop = None
        self._imstack_filtered = None
        self._flipstack_filtered = None
        self._orig_shape = self._orig_imstack.shape[1:]
        self.mask = None

        if scale is None:
            self.vprint("No scale found. Set with: DD.scale = <x> [nm/pix]")
        if defvals is None:
            self.vprint("No scale found. Set with: DD.defvals = <x> [nm]")
        if beam_energy is None:
            self.vprint("Beam energy not found. Set with: DD.energy = <x> [V]")
        return

    @classmethod
    def from_files(
        cls,
        aligned_file: str | os.PathLike,
        aligned_flip_file: str | os.PathLike | None = None,
        metadata_file: str | os.PathLike | None = None,
        flip: Optional[bool] = False,
        scale: Optional[float] = None,
        defocus_values: list | None = None,
        beam_energy: Optional[float] = None,
        dump_metadata: Optional[bool] = True,
        use_mask: Optional[bool] = True,
        legacy_data_loc: str | os.PathLike | None = None,
        legacy_fls_filename: str | os.PathLike | None = None,
        verbose: int | Optional[bool] = True,
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

        data_files = []
        ### load metadata
        if metadata_file is not None:
            metadata_file = Path(metadata_file).absolute()
            mdata = cls._parse_mdata(metadata_file)
            loaded_defvals = mdata["defocus_values"]
            loaded_scale = mdata["scale"]
            loaded_energy = mdata["beam_energy"]
            data_files.append(metadata_file)
        elif legacy_data_loc is not None:
            loaded_scale, loaded_defvals = legacy_load(
                legacy_data_loc, legacy_fls_filename
            )
            loaded_energy = None
        else:
            assert scale is not None and defocus_values is not None
            scale = float(scale)
            defvals = np.array(defocus_values)
            loaded_scale = None
            loaded_defvals = None
            loaded_energy = None

        if loaded_scale is not None:
            if scale is None:
                scale = float(loaded_scale)
            else:
                vprint(
                    f"Overwriting loaded scale, {loaded_scale:.2f} nm/pix,"
                    + f"with user-set value {scale:.2f} nm/pix"
                )
                scale = float(scale)

        if loaded_defvals is not None:
            if defocus_values is None:
                defvals = np.array(loaded_defvals)
            else:
                vprint(
                    f"Overwriting loaded defocus values:\n\t{loaded_defvals}"
                    + f"with user-set value:\n\t{defvals}"
                )
                defvals = np.array(defocus_values)

        if loaded_energy is not None:
            if beam_energy is None:
                beam_energy = float(loaded_energy)
            else:
                vprint(
                    f"Overwriting loaded beam energy:\n\t{loaded_energy}"
                    + f"with user-set value:\n\t{beam_energy}"
                )
                beam_energy = float(beam_energy)

        ### load aligned images
        aligned_file = Path(aligned_file)
        assert aligned_file.exists()
        data_files.append(aligned_file)
        imstack, _mdata = read_image(aligned_file)
        if "scale" in _mdata and scale is None:
            scale = _mdata["scale"]
        if aligned_flip_file is not None:
            aligned_flip_file = Path(aligned_flip_file)
            assert aligned_flip_file.exists()
            data_files.append(aligned_flip_file)
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

        tfs = cls(
            imstack=imstack,
            flipstack=flipstack,
            flip=flip,
            scale=scale,
            defvals=defvals,
            beam_energy=beam_energy,
            use_mask=use_mask,
            data_dir=data_dir,
            data_files=data_files,
            verbose=verbose,
        )

        return tfs

    @property
    def imstack(self):
        return self._imstack

    @imstack.setter
    def imstack(self, stack):
        if not hasattr(self, "_imstack"):
            if len(stack) % 2 == 0:
                raise ValueError(
                    f"Imstack must be of odd length, got length: {len(stack)}\nThere "
                    + "should be an equal number of over and under focus images, and "
                    + "one infocus image."
                )
            self._imstack = stack
        elif len(stack) != self.len_tfs:
            raise ValueError(
                f"Length of imstack, {len(stack)} must equal length of TFS, {self.len_tfs}"
            )
        self._imstack = stack

    @property
    def flipstack(self):
        return self._flipstack

    @flipstack.setter
    def flipstack(self, stack: np.ndarray):
        stack = np.array(stack)
        if np.size(stack) == 0:
            self._flipstack = stack
            self.flip = False
        elif len(stack) != len(self.imstack):
            raise ValueError(
                f"Length of flipstack, {len(stack)} must equal length of imstack, {len(self.imstack)}"
            )
        else:
            self._flipstack = stack

    @property
    def flip(self):
        return self._flip

    @flip.setter
    def flip(self, val: bool):
        if val:
            if len(self.flipstack) == 0:
                raise ValueError(
                    f"TFS does not have a flipstack, flip cannot be set to True"
                )
        self._flip = val

    @property
    def defvals(self):
        return self._defvals

    @property
    def defvals_index(self):
        # get average over/under defocus values. In ideal case would be same over focus
        # as under focus, but in case it's a little off this helps.
        dfs = [
            (self.defvals[-1 * (i + 1)] - self.defvals[i]) / 2
            for i in range(self.len_tfs // 2)
        ]
        return np.array(dfs)[::-1]

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
        if len(vals) % 2 != 1:
            raise ValueError(
                f"Expects a tfs with odd number of images, received even number of defocus values: {len(vals)}"
            )
        for a0 in range(len(vals) // 2):
            if np.sign(vals[a0]) == np.sign(vals[-1 * (a0 + 1)]):
                raise ValueError(
                    f"Underfocus and overfocus values have same sign: {vals[a0]} and {vals[-1*(a0+1)]}"
                )
        self._defvals = np.array(vals)

    @property
    def beam_energy(self):
        return self._beam_energy

    @beam_energy.setter
    def beam_energy(self, val):
        if val is None:
            self._beam_energy = None
        else:
            if not isinstance(val, (float, int)):
                raise TypeError(f"energy must be numeric, found {type(val)}")
            if val <= 0:
                raise ValueError(f"energy must be > 0, not {val}")
            self._beam_energy = float(val)

    @property
    def full_stack(self):
        if self.flip:
            return np.concatenate([self.imstack, self.flipstack])
        else:
            return self.imstack

    @property
    def full_defvals(self):
        if self.flip:
            return np.concatenate([self.defvals, self.defvals])
        else:
            return self.defvals

    @property
    def infocus(self):
        inf_index = self.len_tfs // 2
        if self.flip:
            ave_infocus = (self.imstack[inf_index] + self.flipstack[inf_index]) / 2
            return ave_infocus
        else:
            return self.imstack[inf_index]

    @property
    def orig_infocus(self):
        inf_index = self.len_tfs // 2
        if self.flip:
            if self._preprocessed:
                ave_infocus = (
                    self._orig_imstack_preprocessed[inf_index]
                    + self._orig_flipstack_preprocessed[inf_index]
                ) / 2
            else:
                ave_infocus = (
                    self._orig_imstack[inf_index] + self._orig_flipstack[inf_index]
                ) / 2

            return ave_infocus
        else:
            if self._preprocessed:
                return self._orig_imstack_preprocessed[inf_index]
            else:
                return self._orig_imstack[inf_index]

    @property
    def shape(self):
        return self.imstack.shape[1:]

    @property
    def len_tfs(self):
        if self.flip:
            assert len(self.imstack) == len(self.flipstack)
        return len(self.imstack)

    def __len__(self):
        return self.len_tfs

    @property
    def _len_full_tfs(self):
        return len(self.imstack) + len(self.flipstack)

    def preprocess(
        self,
        hotpix: Optional[bool] = True,
        median_filter_size: int | None = None,
        fast: Optional[bool] = True,
        **kwargs,
    ):
        self._make_mask(self._use_mask)
        # filter hotpixels, option for median filter
        self.imstack = self._orig_imstack.copy()
        self.flipstack = self._orig_flipstack.copy()

        self.imstack *= self.mask[None]
        if self.flip:
            self.flipstack *= self.mask[None]

        if hotpix:
            self.vprint("Filtering hot/dead pixels")
            for i in tqdm(range(self.len_tfs)):
                self.imstack[i] = filter_hotpix(self.imstack[i], fast=fast, **kwargs)

                if self.flip:
                    self.flipstack[i] = filter_hotpix(
                        self.flipstack[i], fast=fast, **kwargs
                    )

        if median_filter_size is not None:
            self.imstack = ndi.median_filter(
                self.imstack, size=(1, median_filter_size, median_filter_size)
            )
            if self.flip:
                self.flipstack = ndi.median_filter(
                    self.flipstack, size=(1, median_filter_size, median_filter_size)
                )

        self._preprocessed = True
        self._orig_imstack_preprocessed = self.imstack.copy()
        self._orig_flipstack_preprocessed = self.flipstack.copy()
        self._filters["hotpix"] = hotpix
        self._filters["median"] = median_filter_size

        return

    def filter(
        self,
        q_lowpass: Optional[float] = None,
        q_highpass: Optional[float] = None,
        filter_type: str = "butterworth",  # butterworth or gaussian
        butterworth_order: int = 2,
        show: Optional[bool] = False,
        v: int | None = None,
    ):
        """
        in terms of workflow, this should probably be used after selecting an ROI...
        because don't want effects of mask/alignment on the filtered images
        but that means it's slow because has to apply transformations fresh each time.
        hm.
        could
        """
        v = self._verbose if v is None else v

        if self._cropped:
            input_imstack = self._imstack_crop.copy()
            if self.flip:
                input_flipstack = self._flipstack_crop.copy()
        elif self._preprocessed:
            input_imstack = self._orig_imstack_preprocessed.copy()
            if self.flip:
                input_flipstack = self._orig_flipstack_preprocessed.copy()
        else:
            input_imstack = self._orig_imstack.copy()
            if self.flip:
                input_flipstack = self._orig_flipstack.copy()

        filtered_imstack = np.zeros_like(input_imstack)
        filtered_flipstack = np.zeros_like(input_imstack)
        for i in tqdm(range(len(input_imstack))):
            filtered_imstack[i] = self._bandpass_filter(
                input_imstack[i], q_lowpass, q_highpass, filter_type, butterworth_order
            )
            if self.flip:
                filtered_flipstack[i] = self._bandpass_filter(
                    input_flipstack[i],
                    q_lowpass,
                    q_highpass,
                    filter_type,
                    butterworth_order,
                )

        if show:
            fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(12, 10))
            inds = [0, len(self) // 2, -1]
            for a0 in range(3):
                show_im(
                    input_imstack[inds[a0]],
                    figax=(fig, axs[a0, 0]),
                    title=f"original df {self._fmt_defocus(self.defvals[inds[a0]])}",
                    scale=self.scale,
                    ticks_off=a0 != 0,
                )
                show_im(
                    filtered_imstack[inds[a0]],
                    figax=(fig, axs[a0, 1]),
                    title="filtered",
                    ticks_off=True,
                )
                show_im(
                    input_imstack[inds[a0]] - filtered_imstack[inds[a0]],
                    figax=(fig, axs[a0, 2]),
                    title="orig - filtered",
                    ticks_off=True,
                )

        self._filtered = True
        self._filters["q_lowpass"] = q_lowpass
        self._filters["q_highpass"] = q_highpass
        self._filters["filter_type"] = filter_type
        self._filters["butterworth_order"] = butterworth_order

        self.imstack = filtered_imstack
        self._orig_imstack_filtered = filtered_imstack
        if self.flip:
            self.flipstack = filtered_flipstack
            self._orig_flipstack_filtered = filtered_flipstack

        return

    def _make_mask(self, use_mask: Optional[bool] = True, threshold: float = 0):
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
        if not use_mask or self._simulated:
            self.mask = np.ones(self.shape)
        elif self.len_tfs == 1:
            # SITIE should have no mask -- not relevant since making other dd
            self.mask = np.ones(self.shape)
        else:
            mask = np.where(self.full_stack > threshold, 1, 0)
            mask = np.prod(mask, axis=0)

            # shrink mask slightly
            iters = int(min(15, self.shape[0] // 250, self.shape[1] // 250))
            if iters >= 1:  # binary_erosion fails if iterations=0
                mask = ndi.morphology.binary_erosion(mask, iterations=iters)
            mask = mask.astype(np.float32, copy=False)
            mask = ndi.gaussian_filter(mask, 2)
            self.mask = mask
            self._orig_mask = mask.copy()
            return

    def apply_transforms(self, v=1):
        # apply rotation -> crop, and set imstack and flipstack
        # for either stack or single image
        # might have to copy this over to show functions as well
        if self._verbose or v >= 1:
            print("Applying transforms", end="\r")
        if self._preprocessed:
            imstack = self._orig_imstack_preprocessed.copy()
            flipstack = self._orig_flipstack_preprocessed.copy()
        else:
            imstack = self._orig_imstack.copy()
            flipstack = self._orig_flipstack.copy()

        # will fail if mask not made, change to force preprocess?
        mask = self._orig_mask.copy()
        if self._transforms["rotation"] != 0:
            # same speed as doing all together, this way have progress bar
            mask = ndi.rotate(mask, self._transforms["rotation"], reshape=False)
            for a0 in tqdm(range(len(imstack)), disable=v < 1):
                imstack[a0] = ndi.rotate(
                    imstack[a0], self._transforms["rotation"], reshape=False
                )
                if self.flip:
                    flipstack[a0] = ndi.rotate(
                        flipstack[a0], self._transforms["rotation"], reshape=False
                    )
        top = self._transforms["top"]
        bottom = self._transforms["bottom"]
        left = self._transforms["left"]
        right = self._transforms["right"]
        imstack = imstack[:, top:bottom, left:right]
        self._imstack_crop = imstack.copy()
        mask = mask[top:bottom, left:right]
        if self.flip:
            flipstack = flipstack[:, top:bottom, left:right]
            self._flipstack_crop = flipstack.copy()

        if self._filtered:  # reapply filters
            for a0 in range(len(imstack)):
                imstack[a0] = self._bandpass_filter(
                    imstack[a0],
                    self._filters["q_lowpass"],
                    self._filters["q_highpass"],
                    self._filters["filter_type"],
                    self._filters["butterworth_order"],
                )
            self._imstack_filtered = imstack.copy()

            if self.flip:
                for a0 in range(len(flipstack)):
                    flipstack[a0] = self._bandpass_filter(
                        flipstack[a0],
                        self._filters["q_lowpass"],
                        self._filters["q_highpass"],
                        self._filters["filter_type"],
                        self._filters["butterworth_order"],
                    )
                self._flipstack_filtered = flipstack.copy()

        self.imstack = imstack
        if self.flip:
            self.flipstack = flipstack
        self.mask = mask
        self._transforms_modified = False
        self._cropped = True
        return

    def select_ROI(self, image: Optional[np.ndarray] = None):
        # select image as infocus orig image if none given
        if image is None:
            inf_ind = self.len_tfs // 2
            if self.flip:
                if self._preprocessed:
                    image = (
                        self._orig_imstack_preprocessed[inf_ind]
                        + self._orig_flipstack_preprocessed[inf_ind]
                    )
                else:
                    image = self._orig_imstack[inf_ind] + self._orig_flipstack[inf_ind]
            else:
                if self._preprocessed:
                    image = self._orig_imstack_preprocessed[inf_ind]
                else:
                    image = self._orig_imstack[inf_ind]

            if self._filtered:
                image = self._bandpass_filter(
                    image,
                    self._filters["q_lowpass"],
                    self._filters["q_highpass"],
                    self._filters["filter_type"],
                    self._filters["butterworth_order"],
                )
        else:
            assert image.shape == self._orig_shape
            if image.shape != self._orig_shape:
                raise ValueError(
                    f"Shape of image for choosing ROI, {image.shape}, must match "
                    + f"orig_images shape, {self._orig_shape}"
                )
            image = np.array(image)

        self._select_ROI(image)

        return

    @classmethod
    def load_DD(cls, filepath: str | os.PathLike):
        """
        Load from saved json file
        """
        return cls(filepath)

    def save_DD(self, filepath="", copy_data: Optional[bool] = False):
        """
        Save self dict as json and either list of filepaths, or full datasets as specified
        make sure includes crop and such.
        could/should probably build this around EMD file, see how datacubes are saved
        """
        return

    def show_tfs(self, **kwargs):

        ncols = len(self)
        nrows = 2 if self.flip else 1

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows)
        )
        if not self.flip:
            axs = axs[None,]

        inf_idx = len(self.defvals) // 2
        ref_image = self.imstack[[inf_idx - 1, inf_idx + 1]]
        vmin_infocus = np.min(ref_image)
        vmax_infocus = np.max(ref_image)

        for a0, df in enumerate(self.defvals):
            vmin = vmin_infocus if a0 == inf_idx else None
            vmax = vmax_infocus if a0 == inf_idx else None

            im_un = self.imstack[a0]
            show_im(
                im_un,
                figax=(fig, axs[0, a0]),
                title=f"{format_defocus(df)}",
                simple=True,
                cmap=kwargs.get("cmap", "gray"),
                vmin=vmin,
                vmax=vmax,
                cbar=kwargs.get("cbar", False),
            )
            if self.flip:
                im_flip = self.flipstack[a0]
                show_im(
                    im_flip,
                    figax=(fig, axs[1, a0]),
                    simple=True,
                    scale=self.scale,
                    cmap=kwargs.get("cmap", "gray"),
                    vmin=vmin,
                    vmax=vmax,
                    cbar=kwargs.get("cbar", False),
                )

        if self.flip:
            axs[0, 0].set_ylabel("Unflip")
            axs[1, 0].set_ylabel("Flip")

        return

    def __str__(self):
        if self.flip:
            return (
                f"ThroughFocalSeries containing two tfs (unflip/flip), each with "
                + f"{len(self)} images of shape {self.shape}"
            )
        else:
            return (
                f"ThroughFocalSeries containing one tfs with {len(self)} images of "
                + f"shape {self.shape}"
            )

    def reset_transforms(self):
        self._reset_transforms()
        self.apply_transforms()
        return
