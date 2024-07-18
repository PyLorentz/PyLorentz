import numpy as np
from PyLorentz.dataset.defocused_dataset import DefocusedDataset
import os
from PyLorentz.phase.base_tie import BaseTIE
from pathlib import Path
import matplotlib.pyplot as plt
from PyLorentz.visualize import show_im, show_2D
from PyLorentz.io.write import write_json


class SITIE(BaseTIE):

    def __init__(
        self,
        dd: DefocusedDataset,
        save_dir: os.PathLike | None = None,
        name: str | None = None,
        sym: bool = False,
        qc: float | None = None,
        verbose: int = 1,
    ):
        self.dd = dd
        if save_dir is None:
            if dd.data_dir is not None:
                topdir = Path(dd.data_dir)
                if topdir.exists():
                    save_dir = topdir / "SITIE_outputs"

        BaseTIE.__init__(
            self,
            save_dir=save_dir,
            scale=dd.scale,
            beam_energy=dd.beam_energy,
            name=name,
            verbose=verbose,
        )
        self.qc = qc  # for type checking
        self.sym = sym
        self.scale = dd.scale
        self._results["input_image"] = None
        self._recon_defval = None
        self._recon_defval_index = None

        if not self.dd._preprocessed and not self.dd._simulated:
            raise ValueError(f"dataset has not been preprocessed")

        return

    @classmethod
    def from_array(
        cls,
        image: np.ndarray,
        scale: float | int | None = None,
        defvals: list[float] | None = None,
        beam_energy: float | None = None,
        name: str | None = None,
        sym: bool = False,
        qc: float | None = None,
        save_dir: os.PathLike | None = None,
        simulated: bool = False,
        verbose: int | bool = 1,
    ):
        images = np.array(image)

        dd = DefocusedDataset(
            images=images,
            defvals=defvals,
            scale=scale,
            beam_energy=beam_energy,
            simulated=simulated,
            verbose=verbose,
        )

        sitie = cls(
            dd=dd,
            save_dir=save_dir,
            name=name,
            sym=sym,
            qc=qc,
            verbose=verbose,
        )

        return sitie

    def reconstruct(
        self,
        index: int | None = None,
        name: str | None = None,
        sym: bool = False,
        qc: float | None = None,
        save: bool | str | list[str] = False,
        save_dir: os.PathLike | None = None,
        verbose: int | bool = 1,
        pbcs: bool | None = None,
        overwrite: bool = False,
    ):
        if index is None:
            index = 0
        elif index > len(self) - 1:
            raise IndexError(
                f"Index {index} not allowed for images of length {len(self)}"
            )
        else:
            assert isinstance(index, int)

        if self.dd._transforms_modified:
            self.vprint("DD has unapplied transforms, applying now.")
            self.dd.apply_transforms()

        self._recon_defval_index = index
        self._recon_defval = self.dd.defvals[index]
        self.sym = sym
        if qc is not None:
            self.qc = qc
        if pbcs is not None:
            self._pbcs = pbcs
        self._verbose = verbose
        if save:
            self._check_save_name(save_dir, name, mode="SITIE")
            self._overwrite = overwrite if overwrite is not None else self._overwrite

        self.vprint(
            f"Performing SITIE reconstruction with defocus "
            + f"{self._fmt_defocus(self._recon_defval, spacer=' ')}, index = {index}"
        )

        # setup data
        dimy, dimx = self.dd.shape

        # select image
        recon_image = self.dd.images[index].copy()
        self._results["input_image"] = recon_image.copy()

        if self.sym:
            dimy *= 2
            dimx *= 2
            recon_image = self._symmetrize(recon_image)

        self._make_qi((dimy, dimx))

        # construct the "infocus" image and get derivatives
        infocus_im = np.ones(np.shape(recon_image)) * np.mean(recon_image)
        dIdZ_B = 2 * (recon_image - infocus_im)
        dIdZ_B -= np.sum(dIdZ_B) / np.size(infocus_im)

        phase_B = self._reconstruct_phase(infocus_im, dIdZ_B, self._recon_defval)
        self._results["phase_B"] = phase_B - phase_B.min()
        By, Bx = self.induction_from_phase(phase_B)
        self._results["By"] = By
        self._results["Bx"] = Bx

        if save:
            self._save_results(save, overwrite)

        return self  # self or None?

    def _save_results(self, save, overwrite=None):
        if isinstance(save, bool):
            save_keys = ["phase_B", "Bx", "By", "color", "input_image"]
        elif isinstance(save, str):
            if save.lower() in ["b", "induction", "ind"]:
                save_keys = ["Bx", "By", "color"]
            elif save.lower() in ["phase"]:
                save_keys = ["phase_B"]
            elif save.lower() == "all":
                save_keys = list(self._results.keys())

        elif hasattr(save, "__iter__"):
            # is list or tuple of keys
            save_keys = [str(k) for k in save]

        self.save_dir.mkdir(exist_ok=True)
        self._save_keys(save_keys, self.recon_defval, overwrite)
        self._save_log(overwrite)
        return

    def _save_log(self, overwrite: bool | None = None):
        log_dict = {
            "name": self.name,
            "_save_name": self._save_name,
            "defval": self.recon_defval,
            "sym": self.sym,
            "qc": self.qc,
            "scale": self.scale,
            "transforms": self.dd.transforms,
            "filters": self.dd._filters,
            "beam_energy": self.dd.beam_energy,
            "simulated": self.dd._simulated,
            "data_dir": self.dd.data_dir,
            "data_files": self.dd.data_files,
            "save_dir": self._save_dir,
        }
        ovr = overwrite if overwrite is not None else self._overwrite
        name = f"{self._save_name}_{self._fmt_defocus(self.recon_defval)}_log.json"
        write_json(log_dict, self.save_dir / name, overwrite=ovr, v=self._verbose)
        return

    def __len__(self):
        return len(self.dd.images)


    @property
    def recon_defval(self):
        if self._recon_defval is None:
            print(f"defval is None or has not yet been specified with an index")
        return self._recon_defval


    def visualize(self, cbar=False, plot_scale: bool | str = True):
        """
        show phase + induction, if flip then show phase_e too
        options to save
        """
        fig, axs = plt.subplots(ncols=2, figsize=(6, 3))

        if isinstance(plot_scale, str):
            if plot_scale == "all":
                ticks1 = ticks2 = False
            elif plot_scale.lower() == "phase":
                ticks1 = False
                ticks2 = True
            elif plot_scale.lower() in ["color", "induction", "b", "ind"]:
                ticks1 = True
                ticks2 = False
            else:
                ticks1 = False
                ticks2 = True
        else:
            ticks1 = False
            ticks2 = True

        show_im(
            self.phase_B,
            title="Magnetic phase shift",
            scale=self.scale,
            figax=(fig, axs[0]),
            ticks_off=ticks1,
            cbar=cbar,
            cbar_title="rad",
        )

        show_2D(
            self.Bx,
            self.By,
            figax=(fig, axs[1]),
            scale=self.scale,
            ticks_off=ticks2,
            title="Integrated induction map      ",
        )
        axs[-1].axis('off')

        plt.tight_layout()
        plt.show()
        return self
