import os
from pathlib import Path
from typing import Optional, Union, List

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

- data_directory
    - aligned_stack.tif
    - aligned_flip_stack.tif
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
"""

class ThroughFocalSeries(BaseDataset):
    """
    A class for handling through-focal series (TFS) datasets, including
    processing and visualization.

    Parameters:
        imstack (np.ndarray): Stack of images in the TFS.
        flipstack (Optional[np.ndarray]): Stack of flipped images for comparison.
        flip (Optional[bool]): Indicates if the dataset includes flipped images.
        scale (Optional[float]): The scale of the images.
        defvals (Optional[np.ndarray]): The defocus values for the images.
        beam_energy (Optional[float]): The beam energy used during imaging.
        use_mask (Optional[bool]): Whether to use a mask in processing.
        simulated (Optional[bool]): Indicates if the data is simulated.
        data_dir (Optional[os.PathLike]): Directory where data is stored.
        data_files (List[os.PathLike]): List of file paths for the data.
        verbose (Optional[int]): Verbosity level for logging.
    """

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
        data_dir: Optional[os.PathLike] = None,
        data_files: List[os.PathLike] = [],
        verbose: Optional[int] = 1,
    ):
        imstack = np.array(imstack)
        assert np.ndim(imstack) == 3, f"Bad input shape {imstack.shape}"
        super().__init__(imshape=imstack.shape[1:], data_dir=data_dir, scale=scale, verbose=verbose)

        self.imstack = imstack
        self.flipstack = np.array(flipstack) if flipstack is not None else np.array([])
        if flip and len(self.flipstack) == 0:
            raise ValueError("Flipstack provided, but no flipstack data available.")
        self.flip = flip
        self.defvals = defvals
        self.beam_energy = beam_energy
        self._simulated = simulated
        self.data_files = data_files
        self._use_mask = use_mask if not simulated else False
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
            self.vprint("No scale found. Set with: TFS.scale = <x> [nm/pix]")
        if defvals is None:
            self.vprint("No defocus values found. Set with: TFS.defvals = <x> [nm]")
        if beam_energy is None:
            self.vprint("Beam energy not found. Set with: TFS.energy = <x> [V]")

    @classmethod
    def from_files(
        cls,
        aligned_file: Union[str, os.PathLike],
        aligned_flip_file: Optional[Union[str, os.PathLike]] = None,
        metadata_file: Optional[Union[str, os.PathLike]] = None,
        flip: Optional[bool] = False,
        scale: Optional[float] = None,
        defocus_values: Optional[List[float]] = None,
        beam_energy: Optional[float] = None,
        dump_metadata: Optional[bool] = True,
        use_mask: Optional[bool] = True,
        legacy_data_loc: Optional[Union[str, os.PathLike]] = None,
        legacy_fls_filename: Optional[Union[str, os.PathLike]] = None,
        verbose: Optional[Union[int, bool]] = True,
    ) -> "ThroughFocalSeries":
        """
        Create a ThroughFocalSeries instance from files.

        Args:
            aligned_file (Union[str, os.PathLike]): Path to the aligned image stack.
            aligned_flip_file (Optional[Union[str, os.PathLike]]): Path to the flip image stack.
            metadata_file (Optional[Union[str, os.PathLike]]): Path to metadata file.
            flip (Optional[bool]): Indicates if the dataset includes flipped images.
            scale (Optional[float]): The scale of the images.
            defocus_values (Optional[List[float]]): Defocus values.
            beam_energy (Optional[float]): Beam energy used during imaging.
            dump_metadata (Optional[bool]): Whether to save metadata to a file.
            use_mask (Optional[bool]): Whether to use a mask in processing.
            legacy_data_loc (Optional[Union[str, os.PathLike]]): Path for legacy data.
            legacy_fls_filename (Optional[Union[str, os.PathLike]]): FLS filename for legacy data.
            verbose (Optional[Union[int, bool]]): Verbosity level for logging.

        Returns:
            ThroughFocalSeries: The created TFS instance.
        """
        vprint = print if verbose >= 1 else lambda *a, **k: None

        data_files = []
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
                assert len(imstack) == 2 * len(defvals), (
                    f"Imstack has even length ({len(imstack)}), but does not match "
                    f"2 * #defocus_values ({len(defvals)}) for a flip/unflip reconstruction."
                )
                flip = True
                flipstack = imstack[len(imstack) // 2 :].copy()
                imstack = imstack[: len(imstack) // 2]
            else:
                assert len(imstack) == len(
                    defvals
                ), f"Imstack has odd length ({len(imstack)}) that does not match the # defocus values ({len(defvals)})"
                if flip:
                    vprint(
                        f"Flip was True but only a single TFS was given. Setting Flip=False"
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
    def imstack(self) -> np.ndarray:
        return self._imstack

    @imstack.setter
    def imstack(self, stack: np.ndarray):
        if not hasattr(self, "_imstack"):
            if len(stack) % 2 == 0:
                raise ValueError(
                    f"Imstack must be of odd length, got length: {len(stack)}"
                )
            self._imstack = stack
        elif len(stack) != self.len_tfs:
            raise ValueError(
                f"Length of imstack, {len(stack)} must equal length of TFS, {self.len_tfs}"
            )
        self._imstack = stack

    @property
    def flipstack(self) -> np.ndarray:
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
    def flip(self) -> bool:
        return self._flip

    @flip.setter
    def flip(self, val: bool):
        if val and len(self.flipstack) == 0:
            raise ValueError("TFS does not have a flipstack, flip cannot be set to True")
        self._flip = val

    @property
    def defvals(self) -> np.ndarray:
        return self._defvals

    @property
    def defvals_index(self) -> np.ndarray:
        dfs = [
            (self.defvals[-1 * (i + 1)] - self.defvals[i]) / 2
            for i in range(self.len_tfs // 2)
        ]
        return np.array(dfs)[::-1]

    @defvals.setter
    def defvals(self, vals: Union[float, List[float], np.ndarray]):
        if not isinstance(vals, (np.ndarray, list, tuple)):
            raise TypeError(f"defvals type should be list or ndarray, found {type(vals)}")
        if len(vals) != len(self.imstack):
            raise ValueError(
                f"defvals must have same length as imstack, should be {len(self.imstack)} but was {len(vals)}"
            )
        if len(vals) % 2 != 1:
            raise ValueError(
                f"Expects a TFS with odd number of images, received even number of defocus values: {len(vals)}"
            )
        for a0 in range(len(vals) // 2):
            if np.sign(vals[a0]) == np.sign(vals[-1 * (a0 + 1)]):
                raise ValueError(
                    f"Underfocus and overfocus values have same sign: {vals[a0]} and {vals[-1*(a0+1)]}"
                )
        self._defvals = np.array(vals)

    @property
    def beam_energy(self) -> Optional[float]:
        return self._beam_energy

    @beam_energy.setter
    def beam_energy(self, val: Optional[float]):
        if val is not None:
            if not isinstance(val, (float, int)):
                raise TypeError(f"Energy must be numeric, found {type(val)}")
            if val <= 0:
                raise ValueError(f"Energy must be > 0, not {val}")
        self._beam_energy = float(val) if val is not None else None

    @property
    def full_stack(self) -> np.ndarray:
        if self.flip:
            return np.concatenate([self.imstack, self.flipstack])
        else:
            return self.imstack

    @property
    def full_defvals(self) -> np.ndarray:
        if self.flip:
            return np.concatenate([self.defvals, self.defvals])
        else:
            return self.defvals

    @property
    def infocus(self) -> np.ndarray:
        inf_index = self.len_tfs // 2
        if self.flip:
            ave_infocus = (self.imstack[inf_index] + self.flipstack[inf_index]) / 2
            return ave_infocus
        else:
            return self.imstack[inf_index]

    @property
    def orig_infocus(self) -> np.ndarray:
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
    def shape(self) -> tuple:
        return self.imstack.shape[1:]

    @property
    def len_tfs(self) -> int:
        if self.flip:
            assert len(self.imstack) == len(self.flipstack)
        return len(self.imstack)

    def __len__(self) -> int:
        return self.len_tfs

    def preprocess(
        self,
        hotpix: Optional[bool] = True,
        median_filter_size: Optional[int] = None,
        fast: Optional[bool] = True,
        **kwargs,
    ) -> None:
        """
        Preprocess the images by filtering hot pixels and applying a median filter.

        Args:
            hotpix (Optional[bool]): Whether to filter hot pixels.
            median_filter_size (Optional[int]): Size of the median filter.
            fast (Optional[bool]): Whether to use a fast filtering method.
        """
        self._make_mask(self._use_mask)
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

    def filter(
        self,
        q_lowpass: Optional[float] = None,
        q_highpass: Optional[float] = None,
        filter_type: str = "butterworth",  # butterworth or gaussian
        butterworth_order: int = 2,
        show: Optional[bool] = False,
        v: Optional[int] = None,
    ) -> None:
        """
        Apply filtering to the image stack.

        Args:
            q_lowpass (Optional[float]): Lowpass filter cutoff.
            q_highpass (Optional[float]): Highpass filter cutoff.
            filter_type (str): Type of filter ('butterworth' or 'gaussian').
            butterworth_order (int): Order of the Butterworth filter.
            show (Optional[bool]): Whether to show the filtered images.
            v (Optional[int]): Verbosity level.
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
                    input_flipstack[i], q_lowpass, q_highpass, filter_type, butterworth_order
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

    def _make_mask(self, use_mask: Optional[bool] = True, threshold: float = 0) -> None:
        """
        Create a binary mask for the image stack.

        Args:
            use_mask (Optional[bool]): Whether to create a mask.
            threshold (float): Threshold for mask creation.

        Returns:
            None
        """
        if not use_mask or self._simulated:
            self.mask = np.ones(self.shape)
        elif self.len_tfs == 1:
            self.mask = np.ones(self.shape)
        else:
            mask = np.where(self.full_stack > threshold, 1, 0)
            mask = np.prod(mask, axis=0)

            iters = int(min(15, self.shape[0] // 250, self.shape[1] // 250))
            if iters >= 1:
                mask = ndi.morphology.binary_erosion(mask, iterations=iters)
            mask = mask.astype(np.float32, copy=False)
            mask = ndi.gaussian_filter(mask, 2)
            self.mask = mask
            self._orig_mask = mask.copy()

    def apply_transforms(self, v: int = 1) -> None:
        """
        Apply image transformations, such as rotation and cropping.

        Args:
            v (int): Verbosity level for logging.

        Returns:
            None
        """
        if self._verbose or v >= 1:
            print("Applying transforms", end="\r")
        if self._preprocessed:
            imstack = self._orig_imstack_preprocessed.copy()
            flipstack = self._orig_flipstack_preprocessed.copy()
        else:
            imstack = self._orig_imstack.copy()
            flipstack = self._orig_flipstack.copy()

        mask = self._orig_mask.copy()
        if self._transforms["rotation"] != 0:
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

        if self._filtered:
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

    def select_ROI(self, image: Optional[np.ndarray] = None) -> None:
        """
        Select a region of interest (ROI) from the image.

        Args:
            image (Optional[np.ndarray]): Image to select ROI from.

        Returns:
            None
        """
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
            if image.shape != self._orig_shape:
                raise ValueError(
                    f"Shape of image for choosing ROI, {image.shape}, must match "
                    + f"orig_images shape, {self._orig_shape}"
                )
            image = np.array(image)

        self._select_ROI(image)

    def show_tfs(self, **kwargs) -> None:
        """
        Display the through-focal series images.

        Args:
            **kwargs: Additional keyword arguments for display.

        Returns:
            None
        """
        ncols = len(self)
        nrows = 2 if self.flip else 1

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows))
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

    def __str__(self) -> str:
        if self.flip:
            return (
                f"ThroughFocalSeries containing two TFS (unflip/flip), each with "
                + f"{len(self)} images of shape {self.shape}"
            )
        else:
            return (
                f"ThroughFocalSeries containing one TFS with {len(self)} images of "
                + f"shape {self.shape}"
            )

    def reset_transforms(self) -> None:
        """
        Reset transformations applied to the images.

        Returns:
            None
        """
        self._reset_transforms()
        self.apply_transforms()
