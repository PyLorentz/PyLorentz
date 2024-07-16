import numpy as np
from PyLorentz.dataset.through_focal_series import ThroughFocalSeries
import os
from PyLorentz.phase.base_tie import BaseTIE
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from PyLorentz.visualize import show_im, show_2D
from PyLorentz.visualize.colorwheel import make_colorwheel
from PyLorentz.io.write import write_json
from datetime import datetime


class TIE(BaseTIE):

    def __init__(
        self,
        tfs: ThroughFocalSeries,
        save_dir: os.PathLike | None = None,
        name: str | None = None,
        sym: bool = False,
        qc: float | None = None,
        beam_energy: float | None = None,
        verbose: int = 1,
    ):
        self.tfs = tfs
        if save_dir is None:
            if tfs.data_dir is not None:
                topdir = Path(tfs.data_dir)
                if topdir.exists():
                    save_dir = topdir / "TIE_outputs"
            else:
                self.save_dir = None

        BaseTIE.__init__(
            self,
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
                    "experimental dataset has not been preprocessed "
                    + "creating uniform mask."
                )
            self.tfs.mask = np.ones(self.tfs.shape, dtype=np.float32)

        return

    def reconstruct(
        self,
        index: int | None = None,
        name: str | None = None,
        sym: bool = False,
        qc: float | None = None,
        flip: bool | None = None,
        save_mode: bool | str | list[str] = False,
        save_dir: os.PathLike | None = None,
        verbose: int | bool = 1,
        pbcs: bool | None = None,
        overwrite: bool = False,
    ):
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
                f"Reconstructing with two TFS flip/unflip to seperate phase_B and phase_E"
            )
        else:
            self.vprint(f"Reconstructing with a single TFS")

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
            # print("\n*** bad neg value figure out *** \n")
            # print('min: ', recon_stack.min())
            # recon_stack -= recon_stack.min()
            # infocus_im -= infocus_im.min()
            # recon_stack += 1e-9
            # infocus_im += 1e-9
            # print('min: ', recon_stack.min())
        # assert np.min(recon_stack) >= 0

        # self.vprint("Calling TIE solver")

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

        return self  # self or None?

    def save_results(
        self,
        save_mode: bool | str | list[str] = True,
        save_dir: os.PathLike | None = None,
        name: str | None = None,
        overwrite: bool = False,
    ) -> ThroughFocalSeries:
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
            # is list or tuple of keys
            save_keys = [str(k) for k in save_mode]

        self.save_dir.mkdir(exist_ok=True)
        self._save_keys(save_keys, self.recon_defval, overwrite)
        self._save_log(overwrite)
        return self

    def _save_log(self, overwrite: bool | None = None):
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
        return

    def _get_derivatives(self, stack, mask, flip):
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

    def _select_images(self):
        # if ptie.flip == True: returns [ +- , -- , 0 , ++ , -+ ]
        # elif ptie.flip == False: returns [+-, 0, ++]
        # where first +/- is unflip/flip, second +/- is over/underfocus.

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
        # stack[len(stack) // 2] += 1 - self.dd.mask
        infocus = stack[len(stack) // 2]
        infocus += 1 - self.tfs.mask

        return stack, infocus

    def _scale_stack(self, stack):
        """Scale a stack of images so all have the same total intensity.

        Args:
            imstack (list): List of 2D arrays.

        Returns:
            list: List of same shape as imstack
        """
        imstack = stack.copy()
        tots = np.sum(imstack, axis=(1, 2))
        t = np.max(tots) / tots
        imstack *= t[..., None, None]
        return imstack / np.max(imstack)

    @property
    def _valid_def_inds(self):
        return len(self.tfs.defvals_index)

    @property
    def recon_defval(self):
        if self._recon_defval is None:
            print(f"defval is None or has not yet been specified with an index")
        return self._recon_defval

    @property
    def flip(self):
        return self._flip

    @flip.setter
    def flip(self, val: bool | None):
        if val is None:
            self._flip = self.tfs.flip
            return
        elif not isinstance(val, bool):
            raise TypeError(f"flip must be bool, not {type(val)}")
        if self.tfs.flip:
            if not val:
                warnings.warn(
                    f"Setting flip=False even though dataset has flip/unflip tfs"
                )
            self._flip = val
        else:
            if val:
                raise ValueError(
                    f"Cannot set flip=True because dataset has only only one TFS"
                )
            else:
                self._flip = val

    @property
    def phase_E(self):
        if self.flip:
            return self.results["phase_E"]
        else:
            if self.results["phase_E"] is not None:
                self.vprint("Returning old phase_E as currently flip=False")
            else:
                raise ValueError(f"phase_E does not exist because flip=False")

    def visualize(self, cbar=False, plot_scale: bool | str = True):
        """
        show phase + induction, if flip then show phase_e too
        options to save
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
            title="Integrated induction map      ",
            # rad=0,
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
        return

    def _check_index(self, index: int):
        if index is None:
            index = self._valid_def_inds - 1
        elif abs(index + 1) > self._valid_def_inds or index < -1 * self._valid_def_inds:
            raise ValueError(
                f"index {index} is out of bounds for defvals_index with size {self._valid_def_inds}"
            )
        elif not isinstance(index, int):
            raise IndexError(f"index must be of type int, not {type(index)}")
        index = index % self._valid_def_inds
        return index

    def show_phase_E(self, show_scale=True, **kwargs):
        """
        show induction
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
        return
