import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.constants as physcon
from scipy.signal import convolve2d

from PyLorentz.io.write import format_defocus, write_tif
from PyLorentz.visualize import show_2D, show_im
from PyLorentz.visualize.colorwheel import color_im


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s:%s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


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


class BaseTIE(BasePhaseReconstruction):
    """
    A base class for Transport of Intensity Equation (TIE) reconstruction.
    """

    def __init__(
        self,
        save_dir: Optional[os.PathLike] = None,
        scale: Optional[float] = None,
        beam_energy: Optional[float] = None,
        name: Optional[str] = None,
        sym: bool = False,
        qc: Optional[float] = None,
        verbose: Union[int, bool] = 1,
    ):
        """
        Initialize the BaseTIE object.

        Args:
            save_dir (Optional[os.PathLike], optional): Directory to save results. Default is None.
            scale (Optional[float], optional): Scale factor for the dataset. Default is None.
            beam_energy (Optional[float], optional): Beam energy for the reconstruction. Default is None.
            name (Optional[str], optional): Name for the reconstruction. Default is None.
            sym (bool, optional): Whether to symmetrize the images. Default is False.
            qc (Optional[float], optional): Tikhonov regularization parameter. Default is None.
            verbose (Union[int, bool], optional): Verbosity level. Default is 1.
        """
        super().__init__(save_dir, name, scale, verbose)
        self._sym = sym
        self._qc = qc
        self._qi = None
        self._pbcs = True
        self.beam_energy = beam_energy

    @property
    def sym(self):
        """Get the symmetrization flag."""
        return self._sym

    @sym.setter
    def sym(self, val: bool):
        """Set the symmetrization flag."""
        if isinstance(val, bool):
            self._sym = val
        else:
            raise ValueError(f"sym must be bool, not {type(val)}")

    @property
    def qc(self):
        """Get the Tikhonov regularization parameter."""
        return self._qc

    @qc.setter
    def qc(self, val: Optional[float]):
        """Set the Tikhonov regularization parameter."""
        if val is None:
            self._qc = 0
        elif isinstance(val, (float, int)):
            if val < 0:
                raise ValueError(f"qc must be >= 0, not {val}")
            self._qc = float(val)
        else:
            raise ValueError(f"qc must be float, not {type(val)}")

    def _make_qi(self, shape: tuple, qc: Optional[float] = None):
        """
        Create the frequency response function for Tikhonov regularization.

        Args:
            shape (tuple): Shape of the arrays.
            qc (Optional[float], optional): Tikhonov regularization parameter. Default is None.
        """
        if qc is None:
            qc = self.qc
        ny, nx = shape
        ly = np.fft.fftfreq(ny)
        lx = np.fft.fftfreq(nx)
        X, Y = np.meshgrid(lx, ly)
        q = np.sqrt(X**2 + Y**2)
        q[0, 0] = 1
        if qc is not None and qc > 0:
            self.vprint(f"Using a Tikhonov frequency [1/nm]: {qc:.1e}")
            qi = q**2 / (q**2 + (qc * self.scale) ** 2) ** 2  # qc in 1/pix
        else:
            qi = 1 / q**2
        qi[0, 0] = 0
        self._qi = qi  # saves the freq dist

    def _reconstruct_phase(self, infocus: np.ndarray, dIdZ: np.ndarray, defval: float) -> np.ndarray:
        """
        Reconstruct the phase using the Transport of Intensity Equation (TIE).

        Args:
            infocus (np.ndarray): In-focus image.
            dIdZ (np.ndarray): Longitudinal derivative of the intensity.
            defval (float): Defocus value.

        Returns:
            np.ndarray: Reconstructed phase.
        """
        dimy, dimx = dIdZ.shape

        fft1 = np.fft.fft2(dIdZ)
        # applying 2/3 qc cutoff mask (see de Graef 2003)
        gy, gx = np.ogrid[-dimy // 2 : dimy // 2, -dimx // 2 : dimx // 2]
        rad = dimy / 3
        qc_mask = gy**2 + gx**2 <= rad**2
        qc_mask = np.fft.ifftshift(qc_mask)

        # apply first inverse Laplacian operator
        tmp1 = -1 * np.fft.ifft2(fft1 * qc_mask * self._qi)

        # apply gradient operator and divide by in focus image
        if self._pbcs:
            # using kernel because np.gradient doesn't allow edge wrapping
            kx = [[0, 0, 0], [1 / 2, 0, -1 / 2], [0, 0, 0]]
            ky = [[0, 1 / 2, 0], [0, 0, 0], [0, -1 / 2, 0]]
            grad_y1 = convolve2d(tmp1, ky, mode="same", boundary="wrap")
            grad_y1 = np.real(grad_y1 / infocus)
            grad_x1 = convolve2d(tmp1, kx, mode="same", boundary="wrap")
            grad_x1 = np.real(grad_x1 / infocus)

            # apply second gradient operator
            # Applying laplacian directly doesn't give as good results??
            grad_y2 = convolve2d(grad_y1, ky, mode="same", boundary="wrap")
            grad_x2 = convolve2d(grad_x1, kx, mode="same", boundary="wrap")
            tot = grad_y2 + grad_x2
        else:
            raise NotImplementedError

        # apply second inverse Laplacian
        fft2 = np.fft.fft2(tot)
        prefactor = self._pre_Lap(defval)
        phase = np.real(prefactor * np.fft.ifft2(fft2 * qc_mask * self._qi))

        if self.sym:
            d2y, d2x = phase.shape
            phase = phase[: d2y // 2, : d2x // 2]

        return phase

    def _pre_Lap(self, def_step=1) -> float:
        """
        Calculate the scaling prefactor used in the TIE reconstruction.

        Args:
            def_step (float, optional): Defocus value for the reconstruction. Default is 1.

        Returns:
            float: Numerical prefactor.
        """
        if self._beam_energy is None:
            raise ValueError("beam_energy must be set, is currently None.")
        epsilon = 0.5 * physcon.e / physcon.m_e / physcon.c**2
        lam = (
            physcon.h
            * 1.0e9
            / np.sqrt(2.0 * physcon.m_e * physcon.e)
            / np.sqrt(self._beam_energy + epsilon * self._beam_energy**2)
        )
        return -1 * self.scale**2 / (16 * np.pi**3 * lam * def_step)

    @property
    def beam_energy(self):
        """Get the beam energy."""
        return self._beam_energy

    @beam_energy.setter
    def beam_energy(self, val: Optional[float]):
        """Set the beam energy."""
        if val is None:
            warnings.warn(
                "BaseTIE has beam_energy=None, this must be set before reconstructing"
            )
            self._beam_energy = None
        else:
            if not isinstance(val, (float, int)):
                raise TypeError(f"energy must be numeric, found {type(val)}")
            if val <= 0:
                raise ValueError(f"energy must be > 0, not {val}")
            self._beam_energy = float(val)

    def _symmetrize(self, imstack: np.ndarray, mode="even") -> np.ndarray:
        """
        Make the even symmetric extension of an image (4x as large).

        Args:
            imstack (np.ndarray): Input image or stack of images.
            mode (str, optional): Symmetrization mode, "even" or "odd". Default is "even".

        Returns:
            np.ndarray: Symmetrized image or stack of images.
        """
        imstack = np.array(imstack)
        if imstack.ndim == 2:
            imstack = imstack[None,]
            d2 = True
        else:
            assert imstack.ndim == 3, (
                "symmetrize only supports 2D images or 3D stacks, "
                + f"not {imstack.ndim} arrays"
            )
            d2 = False
        dimz, dimy, dimx = imstack.shape
        imi = np.zeros((dimz, dimy * 2, dimx * 2))
        imi[..., :dimy, :dimx] = imstack
        if mode == "even":
            imi[..., dimy:, :dimx] = np.flip(imstack, axis=1)
            imi[..., :, dimx:] = np.flip(imi[..., :, :dimx], axis=2)
        elif mode == "odd":
            imi[..., dimy:, :dimx] = -1 * np.flip(imstack, axis=1)
            imi[..., :, dimx:] = -1 * np.flip(imi[..., :, :dimx], axis=2)
        else:
            raise ValueError(f"`mode` should be `even` or `odd`, not `{mode}`")
        return imi[0] if d2 else imi
