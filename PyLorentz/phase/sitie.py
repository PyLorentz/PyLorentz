import os
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from PyLorentz.dataset.defocused_dataset import DefocusedDataset
from PyLorentz.io.write import write_json
from PyLorentz.phase.base_tie import BaseTIE
from PyLorentz.visualize import show_2D, show_im


class SITIE(BaseTIE):
    """
    Class for phase reconstruction using the SITIE method.
    """

    def __init__(
        self,
        dd: DefocusedDataset,
        save_dir: Optional[os.PathLike] = None,
        name: Optional[str] = None,
        sym: bool = False,
        qc: Optional[float] = None,
        verbose: int = 1,
    ):
        """
        Initialize the SITIE object.

        Args:
            dd (DefocusedDataset): Defocused dataset.
            save_dir (Optional[os.PathLike], optional): Directory to save results. Default is None.
            name (Optional[str], optional): Name for the reconstruction. Default is None.
            sym (bool, optional): Whether to symmetrize the images. Default is False.
            qc (Optional[float], optional): Tikhonov regularization parameter. Default is None.
            verbose (int, optional): Verbosity level. Default is 1.
        """
        self.dd = dd
        if save_dir is None and dd.data_dir is not None:
            topdir = Path(dd.data_dir)
            if topdir.exists():
                save_dir = topdir / "SITIE_outputs"

        super().__init__(
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
            raise ValueError("dataset has not been preprocessed")

    @classmethod
    def from_array(
        cls,
        image: np.ndarray,
        scale: Union[float, int, None] = None,
        defval: Optional[List[float]] = None,
        beam_energy: Optional[float] = None,
        name: Optional[str] = None,
        sym: bool = False,
        qc: Optional[float] = None,
        save_dir: Optional[os.PathLike] = None,
        simulated: bool = False,
        verbose: Union[int, bool] = 1,
    ) -> "SITIE":
        """
        Create SITIE object from a numpy array.

        Args:
            image (np.ndarray): Input image array.
            scale (Union[float, int, None], optional): Scale factor for the dataset. Default is None.
            defvals (Optional[List[float]], optional): List of defocus values. Default is None.
            beam_energy (Optional[float], optional): Beam energy for the reconstruction. Default is None.
            name (Optional[str], optional): Name for the reconstruction. Default is None.
            sym (bool, optional): Whether to symmetrize the images. Default is False.
            qc (Optional[float], optional): Tikhonov regularization parameter. Default is None.
            save_dir (Optional[os.PathLike], optional): Directory to save results. Default is None.
            simulated (bool, optional): Whether the data is simulated. Default is False.
            verbose (Union[int, bool], optional): Verbosity level. Default is 1.

        Returns:
            SITIE: An instance of the SITIE class.
        """
        dd = DefocusedDataset(
            images=image,
            defvals=defval,
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
        index: Optional[int] = None,
        name: Optional[str] = None,
        sym: bool = False,
        qc: Optional[float] = None,
        save: Union[bool, str, List[str]] = False,
        save_dir: Optional[os.PathLike] = None,
        verbose: Optional[int] = None,
        pbcs: Optional[bool] = None,
        overwrite: bool = False,
    ) -> "SITIE":
        """
        Perform SITIE reconstruction.

        Args:
            index (Optional[int], optional): Index of the image to reconstruct. Default is None.
            name (Optional[str], optional): Name for the reconstruction. Default is None.
            sym (bool, optional): Whether to symmetrize the images. Default is False.
            qc (Optional[float], optional): Tikhonov regularization parameter. Default is None.
            save (Union[bool, str, List[str]], optional): Whether and what to save. Default is False.
            save_dir (Optional[os.PathLike], optional): Directory to save results. Default is None.
            verbose (Union[int, bool], optional): Verbosity level. Default is 1.
            pbcs (Optional[bool], optional): Whether to apply periodic boundary conditions. Default is None.
            overwrite (bool, optional): Whether to overwrite existing files. Default is False.

        Returns:
            SITIE: The SITIE instance after reconstruction.
        """
        if index is None:
            index = 0
        elif index > len(self) - 1:
            raise IndexError(f"Index {index} not allowed for images of length {len(self)}")
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
        self._verbose = verbose if verbose is None else self._verbose
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

    def _save_results(self, save: Union[bool, str, List[str]], overwrite: Optional[bool] = None):
        """
        Save the reconstruction results.

        Args:
            save (Union[bool, str, List[str]]): Keys to save.
            overwrite (Optional[bool], optional): Whether to overwrite existing files. Default is None.
        """
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
            save_keys = [str(k) for k in save]

        self.save_dir.mkdir(exist_ok=True)
        self._save_keys(save_keys, self.recon_defval, overwrite)
        self._save_log(overwrite)

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

    def __len__(self) -> int:
        """
        Get the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.dd.images)

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

    def visualize(self, cbar: bool = False, plot_scale: Union[bool, str] = True) -> "SITIE":
        """
        Visualize the phase and induction maps.

        Args:
            cbar (bool, optional): Whether to display a colorbar. Default is False.
            plot_scale (Union[bool, str], optional): Whether and what scale to plot. Default is True.

        Returns:
            SITIE: The SITIE instance.
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
            title="Integrated induction map",
        )
        axs[-1].axis("off")

        plt.tight_layout()
        plt.show()
        return self
