import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PyLorentz.dataset.through_focal_series import ThroughFocalSeries
from PyLorentz.io.write import write_json
from PyLorentz.phase.base_tie import BaseTIE
from PyLorentz.visualize import show_2D, show_im
from typing import Optional, Union, List


class TIE(BaseTIE):
    """
    Class for performing Transport of Intensity Equation (TIE) phase reconstruction.
    """

    def __init__(
        self,
        tfs: ThroughFocalSeries,
        save_dir: Optional[os.PathLike] = None,
        name: Optional[str] = None,
        sym: bool = False,
        qc: Optional[float] = None,
        verbose: int = 1,
    ):
        """
        Initialize the TIE object.

        Args:
            tfs (ThroughFocalSeries): Through Focal Series dataset.
            save_dir (Optional[os.PathLike], optional): Directory to save results. Default is None.
            name (Optional[str], optional): Name for the reconstruction. Default is None.
            sym (bool, optional): Whether to symmetrize the images. Default is False.
            qc (Optional[float], optional): Tikhonov regularization parameter. Default is None.
            verbose (int, optional): Verbosity level. Default is 1.
        """
        self.tfs = tfs
        if save_dir is None:
            if tfs.data_dir is not None:
                topdir = Path(tfs.data_dir)
                if topdir.exists():
                    save_dir = topdir / "TIE_outputs"
            else:
                self.save_dir = None

        super().__init__(
            save_dir=save_dir,
            scale=tfs.scale,
            name=name,
            verbose=verbose,
            beam_energy=tfs.beam_energy,
        )
        self.qc = qc  # for type checking
        self.sym = sym
        self.scale = tfs.scale
        self._flip = tfs.flip

        self._results["phase_E"] = None
        self._results["infocus"] = None
        self._results["dIdZ_B"] = None
        self._results["dIdZ_E"] = None
        self._recon_defval = None
        self._recon_defval_index = None

        if not self.tfs._preprocessed:
            if not self.tfs._simulated:
                warnings.warn(
                    "Experimental dataset has not been preprocessed. Creating uniform mask."
                )
            self.tfs.mask = np.ones(self.tfs.shape, dtype=np.float32)

    def reconstruct(
        self,
        index: Optional[int] = None,
        name: Optional[str] = None,
        sym: bool = False,
        qc: Optional[float] = None,
        flip: Optional[bool] = None,
        save_mode: Union[bool, str, List[str]] = False,
        save_dir: Optional[os.PathLike] = None,
        verbose: Union[int, bool] = 1,
        pbcs: Optional[bool] = None,
        overwrite: bool = False,
    ) -> 'TIE':
        """
        Perform TIE reconstruction.

        Args:
            index (Optional[int], optional): Index of the image to reconstruct. Default is None.
            name (Optional[str], optional): Name for the reconstruction. Default is None.
            sym (bool, optional): Whether to symmetrize the images. Default is False.
            qc (Optional[float], optional): Tikhonov regularization parameter. Default is None.
            flip (Optional[bool], optional): Whether to use flip images. Default is None.
            save_mode (Union[bool, str, List[str]], optional): Whether and what to save. Default is False.
            save_dir (Optional[os.PathLike], optional): Directory to save results. Default is None.
            verbose (Union[int, bool], optional): Verbosity level. Default is 1.
            pbcs (Optional[bool], optional): Whether to apply periodic boundary conditions. Default is None.
            overwrite (bool, optional): Whether to overwrite existing files. Default is False.

        Returns:
            TIE: The TIE instance after reconstruction.
        """
        index = self._check_index(index)
        self._recon_defval_index = index
        self._recon_defval = self.tfs.defvals_index[index]
        self.sym = sym
        if qc is not None:
            self.qc = qc
        if flip is not None:
            self.flip = flip
        if pbcs is not None:
            self._pbcs = pbcs
        self._verbose = verbose
        if save_mode:
            self._check_save_name(save_dir, name, mode="TIE")
            self._overwrite = overwrite if overwrite is not None else self._overwrite
        if self.tfs._transforms_modified:
            self.vprint("TFS has unapplied transforms, applying now.")
            self.tfs.apply_transforms()
        self.vprint(
            f"Performing TIE reconstruction with defocus ± "
            + f"{self._fmt_defocus(self._recon_defval, spacer=' ')}, index = {index}"
        )
        if self.flip:
            self.vprint(
                "Reconstructing with two TFS flip/unflip to separate phase_B and phase_E"
            )
        else:
            self.vprint("Reconstructing with a single TFS")

        # setup data
        dimy, dimx = self.tfs.shape

        # select images
        recon_stack, infocus_im = self._select_images()

        self._results["infocus"] = infocus_im.copy() * self.tfs.mask
        recon_mask = self.tfs.mask.copy()

        if self.sym:
            dimy *= 2
            dimx *= 2
            recon_stack = self._symmetrize(recon_stack)
            recon_mask = np.abs(self._symmetrize([recon_mask]).squeeze())
            infocus_im = self._symmetrize([infocus_im]).squeeze()

        self._make_qi((dimy, dimx))

        # get derivatives
        dIdZ_B, dIdZ_E = self._get_derivatives(recon_stack, recon_mask, self.flip)
        self._results["dIdZ_B"] = dIdZ_B.copy()

        # temp checks # TODO remove
        assert dimy, dimx == recon_stack.shape[1:]
        if np.min(recon_stack) < 0:
            pass

        phase_B = self._reconstruct_phase(infocus_im, dIdZ_B, self._recon_defval)
        self._results["phase_B"] = phase_B - phase_B.min()
        By, Bx = self.induction_from_phase(phase_B)
        self._results["By"] = By
        self._results["Bx"] = Bx

        if self.flip:
            self._results["dIdZ_E"] = dIdZ_E.copy()
            phase_E = self._reconstruct_phase(infocus_im, dIdZ_E, self._recon_defval)
            self._results["phase_E"] = phase_E - phase_E.min()

        if save_mode:
            self.save_results(save_mode=save_mode, overwrite=overwrite)

        return self

    def save_results(
        self,
        save_mode: Union[bool, str, List[str]] = True,
        save_dir: Optional[os.PathLike] = None,
        name: Optional[str] = None,
        overwrite: bool = False,
    ) -> 'TIE':
        """
        Save the reconstruction results.

        Args:
            save_mode (Union[bool, str, List[str]], optional): Keys to save. Default is True.
            save_dir (Optional[os.PathLike], optional): Directory to save results. Default is None.
            name (Optional[str], optional): Name for the reconstruction. Default is None.
            overwrite (bool, optional): Whether to overwrite existing files. Default is False.

        Returns:
            TIE: The TIE instance.
        """
        self._check_save_name(save_dir, name=name, mode="TIE")

        if isinstance(save_mode, bool):
            if not save_mode:
                return self
            save_keys = ["phase_B", "Bx", "By", "color"]
            if self.flip:
                save_keys.append("phase_E")
        elif isinstance(save_mode, str):
            if save_mode.lower() in ["b", "induction"]:
                save_keys = ["Bx", "By", "color"]
            elif save_mode.lower() in ["phase"]:
                save_keys = ["phase_B"]
                if self.flip:
                    save_keys.append("phase_E")
            elif save_mode.lower() == "all":
                save_keys = list(self.results.keys())
        elif hasattr(save_mode, "__iter__"):
            save_keys = [str(k) for k in save_mode]

        self.save_dir.mkdir(exist_ok=True)
        self._save_keys(save_keys, self.recon_defval, overwrite)
        self._save_log(overwrite)
        return self

    def _save_log(self, overwrite: Optional[bool] = None):
        """
        Save the reconstruction log.

        Args:
            overwrite (Optional[bool], optional): Whether to overwrite existing files. Default is None.
        """
        log_dict = {
            "name": self.name,
            "_save_name": self._save_name,
            "defval": self.recon_defval,
            "flip": self.flip,
            "sym": self.sym,
            "qc": self.qc,
            "scale": self.scale,
            "transforms": self.tfs.transforms,
            "filters": self.tfs._filters,
            "beam_energy": self.tfs.beam_energy,
            "simulated": self.tfs._simulated,
            "use_mask": self.tfs._use_mask,
            "data_dir": self.tfs.data_dir,
            "data_files": self.tfs.data_files,
            "save_dir": self._save_dir,
        }
        ovr = overwrite if overwrite is not None else self._overwrite
        name = f"{self._save_name}_{self._fmt_defocus(self.recon_defval)}_log.json"
        self._log = log_dict
        write_json(log_dict, self.save_dir / name, overwrite=ovr, v=self._verbose)

    def _get_derivatives(self, stack: np.ndarray, mask: np.ndarray, flip: bool) -> tuple:
        """
        Compute the derivatives of the intensity along the z-axis.

        Args:
            stack (np.ndarray): Stack of images.
            mask (np.ndarray): Mask for the images.
            flip (bool): Whether to use flip images.

        Returns:
            tuple: (dIdZ_B, dIdZ_E) where dIdZ_B is the magnetic component and dIdZ_E is the electrostatic component.
        """
        if flip:
            assert len(stack) == 5, f"Expect stack len 5 with flip, got {len(stack)}"
            dIdZ_B = 0.5 * ((stack[3] - stack[0]) - (stack[4] - stack[1]))
            dIdZ_E = 0.5 * ((stack[3] - stack[0]) + (stack[4] - stack[1]))
        else:
            assert len(stack) == 3, f"Expect stack len 3 for no flip, got {len(stack)}"
            dIdZ_B = stack[2] - stack[0]
            dIdZ_E = None

        dIdZ_B *= mask
        dIdZ_B -= dIdZ_B.sum() / mask.sum()
        dIdZ_B *= mask

        if flip:
            dIdZ_E *= mask
            dIdZ_E -= dIdZ_E.sum() / mask.sum()
            dIdZ_E *= mask

        return dIdZ_B, dIdZ_E

    def _select_images(self) -> tuple:
        """
        Select the images for the reconstruction.

        Returns:
            tuple: (stack, infocus) where stack is the stack of images and infocus is the infocus image.
        """
        under_ind = self.tfs.len_tfs // 2 - (self._recon_defval_index + 1)
        over_ind = self.tfs.len_tfs // 2 + (self._recon_defval_index + 1)

        if self.flip:
            stack = np.stack(
                [
                    self.tfs.imstack[under_ind],
                    self.tfs.flipstack[under_ind],
                    self.tfs.infocus,
                    self.tfs.imstack[over_ind],
                    self.tfs.flipstack[over_ind],
                ]
            )
        else:
            stack = np.stack(
                [
                    self.tfs.imstack[under_ind],
                    self.tfs.imstack[self.tfs.len_tfs // 2],
                    self.tfs.imstack[over_ind],
                ]
            )
        stack = self._scale_stack(stack) + 1e-9

        # inverting background of infocus because dividing by it
        infocus = stack[len(stack) // 2]
        infocus += 1 - self.tfs.mask

        return stack, infocus

    def _scale_stack(self, stack: np.ndarray) -> np.ndarray:
        """
        Scale a stack of images so all have the same total intensity.

        Args:
            stack (np.ndarray): Stack of images.

        Returns:
            np.ndarray: Scaled stack of images.
        """
        imstack = stack.copy()
        tots = np.sum(imstack, axis=(1, 2))
        t = np.max(tots) / tots
        imstack *= t[..., None, None]
        return imstack / np.max(imstack)

    @property
    def _valid_def_inds(self) -> int:
        """
        Get the number of valid defocus indices.

        Returns:
            int: Number of valid defocus indices.
        """
        return len(self.tfs.defvals_index)

    @property
    def recon_defval(self) -> Optional[float]:
        """
        Get the defocus value used for reconstruction.

        Returns:
            Optional[float]: Defocus value.
        """
        if self._recon_defval is None:
            print("defval is None or has not yet been specified with an index")
        return self._recon_defval

    @property
    def flip(self) -> bool:
        """
        Get the flip status.

        Returns:
            bool: Flip status.
        """
        return self._flip

    @flip.setter
    def flip(self, val: Optional[bool]):
        if val is None:
            self._flip = self.tfs.flip
        elif not isinstance(val, bool):
            raise TypeError(f"flip must be bool, not {type(val)}")
        elif self.tfs.flip:
            if not val:
                warnings.warn("Setting flip=False even though dataset has flip/unflip TFS")
            self._flip = val
        else:
            if val:
                raise ValueError("Cannot set flip=True because dataset has only one TFS")
            self._flip = val

    @property
    def phase_E(self) -> np.ndarray:
        """
        Get the electrostatic phase shift.

        Returns:
            np.ndarray: Electrostatic phase shift.
        """
        if self.flip:
            return self.results["phase_E"]
        elif self.results["phase_E"] is not None:
            self.vprint("Returning old phase_E as currently flip=False")
        else:
            raise ValueError("phase_E does not exist because flip=False")

    def visualize(self, cbar: bool = False, plot_scale: Union[bool, str] = True):
        """
        Visualize the phase and induction maps.

        Args:
            cbar (bool, optional): Whether to display a colorbar. Default is False.
            plot_scale (Union[bool, str], optional): Whether and what scale to plot. Default is True.
        """
        if self.flip:
            ncols = 3
        else:
            ncols = 2

        fig, axs = plt.subplots(ncols=ncols, figsize=(3 * ncols, 3.0))

        if isinstance(plot_scale, str):
            if plot_scale == "all":
                ticks1 = ticks2 = ticks3 = False
            elif plot_scale.lower() == "phase":
                ticks1 = ticks2 = False
                ticks3 = True
            elif plot_scale.lower() in ["color", "induction", "b", "ind"]:
                ticks1 = ticks2 = True
                ticks3 = False
            else:
                ticks1 = False
                ticks2 = ticks3 = True
        else:
            ticks1 = False
            ticks2 = ticks3 = True

        show_im(
            self.phase_B,
            title="Magnetic phase shift",
            scale=self.scale,
            figax=(fig, axs[0]),
            ticks_off=ticks1,
            cbar=cbar,
            cbar_title="rad",
        )

        if self.flip:
            if not cbar and np.ptp(self.phase_E) < 0.01:
                vmin = self.phase_E.min() - 0.005
                vmax = self.phase_E.max() + 0.005
            else:
                vmin = vmax = None
            show_im(
                self.phase_E,
                title="Electrostatic phase shift",
                scale=self.scale,
                figax=(fig, axs[1]),
                ticks_off=ticks2,
                cbar=cbar,
                cbar_title="rad",
                vmin=vmin,
                vmax=vmax,
            )

        show_2D(
            self.Bx,
            self.By,
            figax=(fig, axs[-1]),
            scale=self.scale,
            ticks_off=ticks3,
            title="Integrated induction map",
        )
        axs[-1].axis("off")
        # cax = fig.add_subplot(5,5,15, aspect='equal', anchor="SE")
        # box = cax.get_position()
        # box.x0 = box.x0 + 0.01
        # box.x1 = box.x1 + 0.01
        # cax.set_position(box)
        # cax.axis('off')
        # cwheel = make_colorwheel(rad=100, cmap=None, background='white', core='black')
        # # show_im(cwheel, figax=(fig, cax), simple=True)
        # cax.matshow(cwheel)

        plt.tight_layout()
        plt.show()

    def _check_index(self, index: Optional[int]) -> int:
        """
        Check and adjust the index.

        Args:
            index (Optional[int]): Index to check.

        Returns:
            int: Validated index.
        """
        if index is None:
            index = self._valid_def_inds - 1
        elif abs(index + 1) > self._valid_def_inds or index < -self._valid_def_inds:
            raise ValueError(
                f"index {index} is out of bounds for defvals_index with size {self._valid_def_inds}"
            )
        elif not isinstance(index, int):
            raise IndexError(f"index must be of type int, not {type(index)}")
        index = index % self._valid_def_inds
        return index

    def show_phase_E(self, show_scale: bool = True, **kwargs):
        """
        Show the electrostatic phase shift.

        Args:
            show_scale (bool, optional): Whether to show the scale. Default is True.
        """
        dname = (
            self.name
            if self.name is not None
            else self._save_name if self._save_name is not None else ""
        )

        ticks_off = not show_scale
        show_im(
            self.phase_E,
            scale=kwargs.pop("scale", self.scale),
            cbar_title=kwargs.pop("cbar_title", "rad"),
            title=kwargs.pop("title", f"{dname} phase_E"),
            ticks_off=ticks_off,
            title_fontsize=kwargs.pop("title_fontsize", 10),
            **kwargs,
        )
