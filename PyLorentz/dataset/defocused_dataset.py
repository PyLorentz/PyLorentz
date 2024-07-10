import numpy as np
from pathlib import Path
from PyLorentz.io import read_image, read_json, write_json, format_defocus
from .base_dataset import BaseDataset
from .data_legacy import legacy_load
import os
import scipy.ndimage as ndi
from PyLorentz.utils.filter import filter_hotpix
from tqdm import tqdm
import time
from warnings import warn
import matplotlib.pyplot as plt
from PyLorentz.visualize import show_im


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
        flipstack: np.ndarray | None = None,
        flip: bool = False,
        scale: float | None = None,
        defvals: np.ndarray | None = None,
        beam_energy: float | None = None,
        use_mask: bool = True,
        simulated: bool = False,
        data_dir: os.PathLike | None = None,
        verbose: int | bool = 1,
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
        if simulated:
            use_mask = False
        self._use_mask = use_mask
        self._preprocessed = False

        self._orig_imstack = self.imstack.copy()
        self._orig_flipstack = self.flipstack.copy()
        self._orig_shape = self._orig_imstack.shape[1:]
        self.mask = None

        if scale is None:
            self.vprint("No scale found. Set with: DD.scale = <x> [nm/pix]")
        if defvals is None:
            self.vprint("No scale found. Set with: DD.defvals = <x> [nm]")
        if beam_energy is None:
            self.vprint("No beam energy found. Set with: DD.energy = <x> [V]")
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
        beam_energy: float | None = None,
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
            loaded_energy = mdata["beam_energy"]
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

        tfs = cls(
            imstack=imstack,
            flipstack=flipstack,
            flip=flip,
            scale=scale,
            defvals=defvals,
            beam_energy=beam_energy,
            mask=mask,
            data_dir=data_dir,
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
        hotpix: bool = True,
        median_filter_size: int | None = None,
        fast: bool = True,
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
                self.imstack[i] = filter_hotpix(
                    self.imstack[i], fast=fast, **kwargs
                )

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

        return

    def _make_mask(self, use_mask: bool = True, threshold: float = 0):
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
            print("Applying transformations...", end="\r")
        if self._preprocessed:
            imstack = self._orig_imstack_preprocessed.copy()
            flipstack = self._orig_flipstack_preprocessed.copy()
        else:
            imstack = self._orig_imstack.copy()
            flipstack = self._orig_flipstack.copy()

        # will fail if mask not made, change to force preprocess?
        mask = self._orig_mask.copy()
        if self._transformations["rotation"] != 0:
            # same speed as doing all together, this way have progress bar
            mask = ndi.rotate(mask, self._transformations["rotation"], reshape=False)
            for a0 in tqdm(range(len(imstack)), disable=v < 1):
                imstack[a0] = ndi.rotate(
                    imstack[a0], self._transformations["rotation"], reshape=False
                )
                if self.flip:
                    flipstack[a0] = ndi.rotate(
                        flipstack[a0], self._transformations["rotation"], reshape=False
                    )
        top = self._transformations["top"]
        bottom = self._transformations["bottom"]
        left = self._transformations["left"]
        right = self._transformations["right"]
        imstack = imstack[:, top:bottom, left:right]
        mask = mask[top:bottom, left:right]
        if self.flip:
            flipstack = flipstack[:, top:bottom, left:right]

        self.imstack = imstack
        self.flipstack = flipstack
        self.mask = mask
        print(f"Finished")
        return

    def select_ROI(self, image: np.ndarray | None = None):
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

    def show_tfs(self, **kwargs):

        ncols = len(self)
        nrows = 2 if self.flip else 1

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows)
        )

        inf_idx = len(self.defvals) // 2
        ref_image = self.imstack[inf_idx - 1].copy()
        vmin_infocus = np.min(ref_image)
        vmax_infocus = np.max(ref_image)

        for a0, df in enumerate(self.defvals):
            if self.flip:
                ax = axs[0, a0]
            else:
                ax = axs[a0]

            vmin = vmin_infocus if a0 == inf_idx else None
            vmax = vmax_infocus if a0 == inf_idx else None

            im_un = self.imstack[a0]
            show_im(
                im_un,
                figax=(fig, ax),
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


# for single images / SITIE / SIPRAD
# SITIE itself should be able to take either a DefocusedDataset or DefocusImage?
# have a from_TFS
class DefocusedDataset(BaseDataset):

    # this is from array
    def __init__(
        self,
        images: np.ndarray,
        scale: float | None = None,
        defvals: np.ndarray | None = None,
        beam_energy: float | None = None,
        data_dir: os.PathLike | None = None,
        simulated: bool = False,
        verbose: int | bool = 1,
    ):
        images = np.array(images)
        if np.ndim(images) == 2:
            images = images[None,]
        if isinstance(defvals, (float, int)):
            defvals = np.array([defvals])
        BaseDataset.__init__(
            self,
            imshape=images.shape[1:],
            data_dir=data_dir,
            scale=scale,
            verbose=verbose,
        )

        self.images = images
        self._orig_images = images.copy()
        self._orig_shape = images.shape[1:]
        self._orig_images_preprocessed = None
        self.defvals = defvals
        self.beam_energy = beam_energy
        self._simulated = simulated
        self._verbose = verbose
        # leaving mask for possible future implementation, e.g. masking center region
        self.mask = None # np.ones_like(self.shape)

        self._preprocessed = False

        return

    @classmethod
    def from_TFS(cls):
        """
        convert TFS to DD, using its data
        """
        return cls

    @classmethod
    def load(cls):
        """
        load from file path, possibly with metadata
        """
        return cls

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, ims):
        ims = np.array(ims)
        if np.ndim(ims) == 2:
            ims = ims[None]
        if hasattr(self, "_defvals"):
            if len(ims) != len(self.defvals):
                raise ValueError(
                    f"Len images, {len(ims)} must equal len defvals, {len(self.defvals)}"
                )
        self._images = ims

    @property
    def image(self, idx=0):
        return self.images[idx]

    @property
    def defvals(self):
        return self._defvals

    @defvals.setter
    def defvals(self, dfs):
        if isinstance(dfs, (float, int)):
            dfs = np.array([dfs])
        if hasattr(self, "_images"):
            if len(dfs) != len(self.images):
                raise ValueError(
                    f"Number defocus vals, {len(dfs)} must equal number images, {len(self.images)}"
                )
        self._defvals = dfs

    def __len__(self):
        return len(self.images)

    @property
    def shape(self):
        return self.images.shape[1:]

    @property
    def energy(self):
        return self._beam_energy

    @energy.setter
    def energy(self, val):
        if not isinstance(val, (float, int)):
            raise TypeError(f"energy must be numeric, found {type(val)}")
        if val <= 0:
            raise ValueError(f"energy must be > 0, not {val}")
        self._beam_energy = float(val)

    def select_ROI(self, idx: int = 0, image: np.ndarray | None = None):
        # select image as infocus orig image if none given
        if image is not None:
            image = np.shape(image)
            if image.shape != self._orig_shape:
                raise ValueError(
                    f"Shape of image for choosing ROI, {image.shape}, must match "
                    + f"orig_images shape, {self._orig_images.shape[1:]}"
                )
            image = np.array(image)
        else:
            if self._preprocessed:
                image = self._orig_images_preprocessed[idx]
            else:
                image = self._orig_images[idx]

        self._select_ROI(image)

        return

    def apply_transforms(self, v=1):
        # apply rotation -> crop, and set images
        if self._verbose or v >= 1:
            print("Applying transformations", end="\r")
        if self._preprocessed:
            images = self._orig_images_preprocessed.copy()
        else:
            images = self._orig_images.copy()

        if self._transformations["rotation"] != 0:
            for a0 in tqdm(range(len(images)), disable=v < 1):
                images[a0] = ndi.rotate(
                    images[a0], self._transformations["rotation"], reshape=False
                )
        top = self._transformations["top"]
        bottom = self._transformations["bottom"]
        left = self._transformations["left"]
        right = self._transformations["right"]
        images = images[:, top:bottom, left:right]

        self.images = images
        print(f"Finished")
        return

    def preprocess(
        self,
        hotpix: bool = True,
        median_filter_size: int | None = None,
        fast: bool = True,
        **kwargs,
    ):
        # filter hotpixels, option for median filter
        self.images = self._orig_images.copy()

        if hotpix:
            self.vprint("Filtering hot/dead pixels")
            for i in tqdm(range(len(self))):
                self.images[i] = filter_hotpix(self.images[i], fast=fast, **kwargs)

        if median_filter_size is not None:
            self.images = ndi.median_filter(
                self.images, size=(1, median_filter_size, median_filter_size)
            )

        self._preprocessed = True
        self._orig_images_preprocessed = self.images.copy()
        return

    def __str__(self):
        return f"DefocusedDataset containing {len(self)} image(s) of shape {self.shape}"
