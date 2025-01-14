from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import scipy.constants as physcon

from PyLorentz.io.write import format_defocus, write_tif
from PyLorentz.utils import metrics
from PyLorentz.visualize import show_2D, show_im
from PyLorentz.visualize.colorwheel import color_im

if TYPE_CHECKING:
    import cupy as cp
    from torch import Tensor
else:
    try:
        import cupy as cp
        from torch import Tensor
    except:
        Tensor = np.ndarray


class BasePhaseReconstruction:
    """
    A base class for phase reconstruction, providing common attributes and methods.
    """

    def __init__(
        self,
        save_dir: Optional[os.PathLike | str] = None,
        name: Optional[str] = None,
        scale: Optional[float] = None,
        verbose: Union[int, bool] = 1,
    ):
        """
        Initialize the BasePhaseReconstruction object.

        Args:
            save_dir (Optional[os.PathLike], optional): Directory to save results. Default is None.
            name (Optional[str], optional): Name for the reconstruction. Default is None.
            scale (Optional[float], optional): Scale factor for the dataset. Default is None.
            verbose (Union[int, bool], optional): Verbosity level. Default is 1.
        """
        self.save_dir = save_dir
        self.name = name
        self._save_name = name
        self._verbose = verbose
        if scale is not None:
            self.scale = scale
        self._overwrite = False
        self._results = {}

        self.results = {
            "By": None,
            "Bx": None,
            "phase_B": None,
        }

    def vprint(self, *args, **kwargs):
        """Print messages if verbose is enabled."""
        if self._verbose:
            print(*args, **kwargs)

    @property
    def scale(self) -> float:
        """Get the scale factor."""
        return self._scale

    @scale.setter
    def scale(self, val: float):
        """Set the scale factor."""
        if not isinstance(val, (float, int, np.number)):
            raise TypeError(f"scale must be float/int, not {type(val)}")
        if val <= 0:
            raise ValueError(f"scale must be >0, not {val}")
        self._scale = float(val)

    @property
    def name(self) -> str:
        """Get the name."""
        return self._name

    @name.setter
    def name(self, name: str | None):
        """Set the name."""
        if name is not None:
            self._name = str(name)
        else:
            self._name = ""

    @property
    def results(self):
        """Get the results."""
        return self._results

    @results.setter
    def results(self, res: dict):
        """Set results dictionary."""
        allowed_keys = ["By", "Bx", "phase_B", "phase_E", "infocus", "dIdZ_B", "dIdZ_E"]
        for k, v in res.items():
            if k in allowed_keys:
                self._results[k] = v
            else:
                raise KeyError(f"Key {k} is not an allowed results key: {allowed_keys}")

    @property
    def phase_B(self):
        """Get the magnetic component of the phase shift."""
        return self.results["phase_B"]

    @phase_B.setter
    def phase_B(self, val):
        """Set the magnetic component of the phase shift and induction components."""
        self.results["phase_B"] = val - val.min()
        By, Bx = self.induction_from_phase(val)
        self.results["By"] = By
        self.results["Bx"] = Bx

    @property
    def By(self):
        """Get the y-component of the magnetic induction."""
        return self.results["By"]

    @property
    def Bx(self):
        """Get the x-component of the magnetic induction."""
        return self.results["Bx"]

    @property
    def Bmag(self):
        """Get the magnitude of the magnetic induction."""
        return np.sqrt(self.results["Bx"] ** 2 + self.results["By"] ** 2)

    @property
    def B(self):
        """Get the magnetic induction."""
        return np.array([self.results["By"], self.results["Bx"]])

    @property
    def save_dir(self) -> Path:
        """Get the save directory."""
        return self._save_dir

    @save_dir.setter
    def save_dir(self, p: Optional[os.PathLike | str]):
        """Set the save directory."""
        if p is None:
            self._save_dir = Path("./")
        else:
            p = Path(p).absolute()
            if not p.parents[0].exists():
                raise ValueError(f"save dir parent does not exist: {p.parents[0]}")
            else:
                self._save_dir = p

    def _save_keys(
        self,
        keys,
        defval: Optional[float] = None,
        overwrite: Optional[bool] = None,
        res_dict: Optional[dict] = None,
        **kwargs,
    ):
        """
        Save the specified keys as TIFF images.

        Args:
            keys (list): List of keys to save.
            defval (Optional[float], optional): Defocus value. Default is None.
            overwrite (Optional[bool], optional): Whether to overwrite existing files. Default is None.
        """
        ovr = overwrite if overwrite is not None else self._overwrite
        results = res_dict if res_dict is not None else self.results
        for key in keys:
            if defval is not None:
                name = f"{self._save_name}_{self._fmt_defocus(defval)}_{key}.tiff"
            else:
                name = f"{self._save_name}_{key}.tiff"
            fname = self.save_dir / name

            if "color" in key:
                image = color_im(
                    results["By"],
                    results["Bx"],
                    **kwargs,
                )
            else:
                image = results[key]

            write_tif(
                image,
                fname,
                self.scale,
                v=self._verbose,
                overwrite=ovr,
                color="color" in key,
            )

    def _check_save_name(
        self,
        save_dir: Optional[os.PathLike],
        name: Optional[str],
        mode: str = "",
        default_name: bool = True,
    ):
        """
        Check and set the save name.

        Args:
            save_dir (Optional[os.PathLike]): Directory to save results.
            name (Optional[str]): Name for the reconstruction.
            mode (str, optional): Mode for the save name. Default is "".
        """
        if save_dir is None:
            if self.save_dir is None:
                raise ValueError(f"save_dir not specified, is None")
        else:
            self.save_dir = save_dir  # checks while setting that parents exist
        if name is None and default_name:
            if self.name is None:
                now = datetime.now().strftime("%y%m%d-%H%M%S")
                if len(mode) > 0:
                    mode = "_" + mode
                self._save_name = f"{now}{mode}"
            else:
                self._save_name = self.name
        else:
            self._save_name = name

    def induction_from_phase(self, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the integrated induction from a magnetic phase shift.

        Args:
            phase (np.ndarray): 2D array of magnetic component of the phase shift in radians.

        Returns:
            tuple[np.ndarray, np.ndarray]: (By, Bx), y and x components of the magnetic induction integrated along the z-direction.
        """
        grad_y, grad_x = np.gradient(np.squeeze(phase), edge_order=2)
        pre_B = physcon.hbar / (physcon.e * self.scale) * 1e18  # T*nm^2
        Bx = pre_B * grad_y
        By = -1 * pre_B * grad_x
        return By, Bx

    def show_B(self, show_scale=False, crop=0, **kwargs):
        """
        Show the magnetic induction.

        Args:
            show_scale (bool, optional): Whether to show the scale. Default is False.
        """
        dname = self.name if self.name else self._save_name if self._save_name else ""
        sc = self.scale if show_scale else None
        if self.Bx is None or self.By is None:
            raise ValueError(f"B is None")
        if crop > 0:  # crop to eliminate edge artifacts
            crop = int(crop)
            Bx = self.Bx[crop:-crop, crop:-crop]
            By = self.By[crop:-crop, crop:-crop]
        else:
            Bx, By = self.Bx, self.By
        show_2D(
            Bx,
            By,
            scale=sc,
            title=kwargs.pop("title", f"{dname} B         "),
            title_fontsize=kwargs.pop("title_fontsize", 10),
            **kwargs,
        )

    def show_phase_B(self, show_scale=True, crop=0, **kwargs):
        """
        Show the magnetic component of the phase shift.

        Args:
            show_scale (bool, optional): Whether to show the scale. Default is True.
        """
        if self.phase_B is None:
            raise ValueError(f"self.phase_B is None")
        dname = self.name if self.name else self._save_name if self._save_name else ""
        ticks_off = kwargs.pop("ticks_off", not show_scale)
        if crop > 0:
            crop = int(crop)
            phase_B = self.phase_B[crop:-crop, crop:-crop]
        else:
            phase_B = self.phase_B
        show_im(
            phase_B,
            scale=kwargs.pop("scale", self.scale),
            cbar_title=kwargs.pop("cbar_title", "rad"),
            title=kwargs.pop("title", f"{dname} phase_B"),
            ticks_off=ticks_off,
            title_fontsize=kwargs.pop("title_fontsize", 10),
            **kwargs,
        )

    def calc_phase_metrics(
        self, t_phase: np.ndarray | Tensor, r_phase: np.ndarray | Tensor
    ) -> dict:
        """
        Calculate the correlational accuracy, SSRI, PSNR for a reconstructed and ground truth phase

        Args:
            t_phase (np.ndarray): True
        # phase and B
        """
        r_phase = self._to_numpy(r_phase)
        r_By, r_Bx = self.induction_from_phase(r_phase)
        r_Bmag = np.sqrt(r_By**2 + r_Bx**2)

        t_phase = self._to_numpy(t_phase)
        t_By, t_Bx = self.induction_from_phase(t_phase)
        t_Bmag = np.sqrt(t_By**2 + t_Bx**2)

        keys = [
            "phase",
            "By",
            "Bx",
            "Bmag",
        ]
        ssim_dict = {}
        acc_dict = {}
        psnr_dict = {}
        trues = [t_phase, t_By, t_Bx, t_Bmag]
        tests = [r_phase, r_By, r_Bx, r_Bmag]
        for key, tru, test in zip(keys, trues, tests):
            ssim_dict[f"SSIM_{key}"] = metrics.ssim(tru, test)
            acc_dict[f"acc_{key}"] = metrics.accuracy(tru, test)
            psnr_dict[f"PSNR_{key}"] = metrics.psnr(tru, test)

        ssim_dict["SSIM_Bave"] = 0.5 * (ssim_dict["SSIM_By"] + ssim_dict["SSIM_Bx"])
        acc_dict["acc_Bave"] = 0.5 * (acc_dict["acc_By"] + acc_dict["acc_Bx"])
        psnr_dict["PSNR_Bave"] = 0.5 * (psnr_dict["PSNR_By"] + psnr_dict["PSNR_Bx"])

        results = ssim_dict | acc_dict | psnr_dict  # combine dicts, should be no overlap in keys
        return results

    @staticmethod
    def _to_numpy(arr: Any) -> np.ndarray:
        module = arr.__class__.__module__
        if module == "torch":
            assert isinstance(arr, Tensor)
            return arr.cpu().detach().numpy()
        elif module == "cupy":
            return cp.asnumpy(arr)
        else:
            return np.array(arr)

    @staticmethod
    def _fmt_defocus(defval: Union[float, int], digits: int = 3, spacer="") -> str:
        """
        Format defocus value for display.

        Args:
            defval (Union[float, int]): Defocus value.
            digits (int, optional): Number of digits. Default is 3.
            spacer (str, optional): Spacer string. Default is "".

        Returns:
            str: Formatted defocus value.
        """
        return format_defocus(defval, digits, spacer=spacer)
