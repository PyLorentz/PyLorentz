import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.constants as physcon

from PyLorentz.io.write import format_defocus, write_tif
from PyLorentz.visualize import show_2D, show_im
from PyLorentz.visualize.colorwheel import color_im


class BasePhaseReconstruction:
    """
    A base class for phase reconstruction, providing common attributes and methods.
    """

    def __init__(
        self,
        save_dir: Optional[os.PathLike] = None,
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
        self.scale = scale
        self._overwrite = False

        self._results = {
            "By": None,
            "Bx": None,
            "phase_B": None,
        }

    @property
    def scale(self):
        """Get the scale factor."""
        return self._scale

    @scale.setter
    def scale(self, val: float):
        """Set the scale factor."""
        if not isinstance(val, (float, int)):
            raise TypeError(f"scale must be float/int, not {type(val)}")
        if val <= 0:
            raise ValueError(f"scale must be >0, not {val}")
        self._scale = float(val)

    @property
    def name(self):
        """Get the name."""
        return self._name

    @name.setter
    def name(self, name: str):
        """Set the name."""
        self._name = str(name)

    @property
    def results(self):
        """Get the results."""
        return self._results

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
    def phase_B(self):
        """Get the magnetic component of the phase shift."""
        return self.results["phase_B"]

    def vprint(self, *args, **kwargs):
        """Print messages if verbose is enabled."""
        if self._verbose:
            print(*args, **kwargs)

    @property
    def save_dir(self):
        """Get the save directory."""
        return self._save_dir

    @save_dir.setter
    def save_dir(self, p: Optional[os.PathLike]):
        """Set the save directory."""
        if p is None:
            self._save_dir = None
        else:
            p = Path(p).absolute()
            if not p.parents[0].exists():
                raise ValueError(f"save dir parent does not exist: {p.parents[0]}")
            else:
                self._save_dir = p

    def _save_keys(
        self, keys, defval: Optional[float] = None, overwrite: Optional[bool] = None, **kwargs
    ):
        """
        Save the specified keys as TIFF images.

        Args:
            keys (list): List of keys to save.
            defval (Optional[float], optional): Defocus value. Default is None.
            overwrite (Optional[bool], optional): Whether to overwrite existing files. Default is None.
        """
        ovr = overwrite if overwrite is not None else self._overwrite
        for key in keys:
            if defval is not None:
                name = f"{self._save_name}_{self._fmt_defocus(defval)}_{key}.tiff"
            else:
                name = f"{self._save_name}_{key}.tiff"
            fname = self.save_dir / name

            if "color" in key:
                image = color_im(
                    self.results["By"],
                    self.results["Bx"],
                    **kwargs,
                )
            else:
                image = self.results[key]

            write_tif(
                image,
                fname,
                self.scale,
                v=self._verbose,
                overwrite=ovr,
                color="color" in key,
            )

    @staticmethod
    def _fmt_defocus(defval: Union[float, int], digits: int = 3, spacer=""):
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

    def induction_from_phase(self, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the integrated induction from a magnetic phase shift.

        Args:
            phase (np.ndarray): 2D array of magnetic component of the phase shift in radians.

        Returns:
            tuple[np.ndarray, np.ndarray]: (By, Bx), y and x components of the magnetic induction integrated along the z-direction.
        """
        grad_y, grad_x = np.gradient(np.squeeze(phase), edge_order=2)
        pre_B = physcon.hbar / (physcon.e * self.scale) * 10**18  # T*nm^2
        Bx = pre_B * grad_y
        By = -1 * pre_B * grad_x
        return By, Bx

    def show_B(self, show_scale=False, **kwargs):
        """
        Show the magnetic induction.

        Args:
            show_scale (bool, optional): Whether to show the scale. Default is False.
        """
        dname = self.name if self.name else self._save_name if self._save_name else ""
        sc = self.scale if show_scale else None
        show_2D(
            self.Bx,
            self.By,
            scale=sc,
            title=kwargs.pop("title", f"{dname} B"),
            title_fontsize=kwargs.pop("title_fontsize", 10),
            **kwargs,
        )

    def show_phase_B(self, show_scale=True, **kwargs):
        """
        Show the magnetic component of the phase shift.

        Args:
            show_scale (bool, optional): Whether to show the scale. Default is True.
        """
        dname = self.name if self.name else self._save_name if self._save_name else ""
        ticks_off = not show_scale
        show_im(
            self.phase_B,
            scale=kwargs.pop("scale", self.scale),
            cbar_title=kwargs.pop("cbar_title", "rad"),
            title=kwargs.pop("title", f"{dname} phase_B"),
            ticks_off=ticks_off,
            title_fontsize=kwargs.pop("title_fontsize", 10),
            **kwargs,
        )

    def _check_save_name(self, save_dir: Optional[os.PathLike], name: Optional[str], mode: str = ""):
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
        if name is None:
            if self.name is None:
                now = datetime.now().strftime("%y%m%d-%H%M%S")
                if len(mode) > 0:
                    mode = "_" + mode
                self._save_name = f"{now}{mode}"
            else:
                self._save_name = self.name
        else:
            self._save_name = name

