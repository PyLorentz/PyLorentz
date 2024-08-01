import os
import warnings
from typing import Optional, Union

import numpy as np
import scipy.constants as physcon
from scipy.signal import convolve2d

from PyLorentz.phase.base_phase import BasePhaseReconstruction


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s:%s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


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
