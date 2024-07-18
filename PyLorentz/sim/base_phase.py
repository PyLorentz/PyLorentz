from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as physcon
import scipy.ndimage as ndi

from PyLorentz.visualize import show_2D, show_3D, show_im


class BaseSim(object):
    """
    A base class for simulations, providing common attributes and methods.
    """

    _phi0: float = 2.07e7  # Gauss*nm^2 flux quantum

    _default_params = {
        "phase_method": "mansuripur",
        "B0": 1e4,
        "sample_V0": 0.0,
        "sample_xip0": 50.0,
        "mem_V0": 0.0,
        "mem_xip0": 1e3,
        "mem_thickness": 0.0,
        "theta_x": 0.0,
        "tilt_y": 0.0,
        "beam_energy": 200e3,
    }

    def __init__(
        self,
        mags: np.ndarray,
        scale: float,
        zscale: float,
        verbose: Union[float, bool] = 1,
    ):
        """
        Initialize the BaseSim object.

        Args:
            mags (np.ndarray): Magnetization array.
            scale (float): Scale factor for the simulation.
            zscale (float): Z-axis scale factor.
            verbose (float | bool, optional): Verbosity level. Default is 1.
        """
        self._mags = mags
        self._shape_func = None
        self._flat_shape_func = None
        self._scale = scale
        self._zscale = zscale
        self._verbose = verbose

        self.phase_method = None
        self.tilt_x = None
        self.tilt_y = None
        self.beam_energy = None
        self._sample_params = {}

        self._phase_B = None
        self._phase_E = None

    def vprint(self, *args, **kwargs) -> None:
        """Print messages if verbose is enabled."""
        if self._verbose:
            print(*args, **kwargs)

    @property
    def phase_B(self) -> np.ndarray:
        """Get the B-phase."""
        return self._phase_B

    @property
    def phase_E(self) -> np.ndarray:
        """Get the E-phase."""
        return self._phase_E

    @property
    def phase_t(self) -> np.ndarray:
        """Get the total phase (B-phase + E-phase)."""
        return self.phase_B + self.phase_E

    @property
    def phase_shape(self) -> Tuple[int, ...]:
        """Get the shape of the B-phase array."""
        return np.shape(self.phase_B)

    def set_sample_params(self, params: dict) -> None:
        """Set sample parameters."""
        self.B0 = params.get("B0")
        self.sample_V0 = params.get("sample_V0")
        self.sample_xip0 = params.get("sample_xip0")
        self.mem_V0 = params.get("mem_V0")
        self.mem_xip0 = params.get("mem_xip0")
        self.mem_thickness = params.get("mem_thickness")

    @property
    def sample_params(self) -> dict:
        """Get the sample parameters."""
        return self._sample_params

    @property
    def phase_method(self) -> str:
        """Get the phase method."""
        return self._phase_method

    @phase_method.setter
    def phase_method(self, mode: Optional[str]) -> None:
        """Set the phase method."""
        if mode is None:
            self._phase_method = self._default_params["phase_method"]
        else:
            mode = mode.lower()
            if mode in ["mansuripur", "mans"]:
                self._phase_method = "mansuripur"
            elif mode in ["linsup", "lin", "linear", "linear_superposition"]:
                self._phase_method = "linsup"
            else:
                raise ValueError(
                    f"Unknown phase method, {mode}, valid options are 'linsup' or 'mansuripur'"
                )

    @property
    def B0(self) -> float:
        """Get the B0 parameter."""
        return self.sample_params["B0"]

    @B0.setter
    def B0(self, val: Union[float, int, None]) -> None:
        """Set the B0 parameter."""
        if val is None:
            self._sample_params["B0"] = float(self._default_params["B0"])
        elif val < 0:
            raise ValueError(
                f"B0 must be > 0, and has units of Gauss. Bad value given {val}"
            )
        else:
            self._sample_params["B0"] = float(val)

    @property
    def sample_V0(self) -> float:
        """Get the sample mean inner potential."""
        return self.sample_params["sample_V0"]

    @sample_V0.setter
    def sample_V0(self, val: Union[float, int, None]) -> None:
        """Set the sample mean inner potential."""
        if val is None:
            self._sample_params["sample_V0"] = float(self._default_params["sample_V0"])
        elif val < 0:
            raise ValueError(
                f"sample_V0 must be > 0, and has units of Volts. Bad value given {val}"
            )
        else:
            self._sample_params["sample_V0"] = float(val)

    @property
    def sample_xip0(self) -> float:
        """Get the sample extinction distance."""
        return self.sample_params["sample_xip0"]

    @sample_xip0.setter
    def sample_xip0(self, val: Union[float, int, None]) -> None:
        """Set the sample extinction distance."""
        if val is None:
            self._sample_params["sample_xip0"] = float(
                self._default_params["sample_xip0"]
            )
        elif val < 0:
            raise ValueError(
                f"sample_xip0 must be > 0, and has units of nm. Bad value given {val}"
            )
        else:
            self._sample_params["sample_xip0"] = float(val)

    @property
    def mem_V0(self) -> float:
        """Get the membrane mean inner potential."""
        return self.sample_params["mem_V0"]

    @mem_V0.setter
    def mem_V0(self, val: Union[float, int, None]) -> None:
        """Set the membrane mean inner potential."""
        if val is None:
            self._sample_params["mem_V0"] = float(self._default_params["mem_V0"])
        elif val < 0:
            raise ValueError(
                f"mem_V0 must be > 0, and has units of Volts. Bad value given {val}"
            )
        else:
            self._sample_params["mem_V0"] = float(val)

    @property
    def mem_xip0(self) -> float:
        """Get the membrane extinction distance."""
        return self.sample_params["mem_xip0"]

    @mem_xip0.setter
    def mem_xip0(self, val: Union[float, int, None]) -> None:
        """Set the membrane extinction distance."""
        if val is None:
            self._sample_params["mem_xip0"] = float(self._default_params["mem_xip0"])
        elif val < 0:
            raise ValueError(
                f"mem_xip0 must be > 0, and has units of nm. Bad value given {val}"
            )
        else:
            self._sample_params["mem_xip0"] = float(val)

    @property
    def mem_thickness(self) -> float:
        """Get the membrane thickness."""
        return self.sample_params["mem_thickness"]

    @mem_thickness.setter
    def mem_thickness(self, val: Union[float, int, None]) -> None:
        """Set the membrane thickness."""
        if val is None:
            self._sample_params["mem_thickness"] = float(
                self._default_params["mem_thickness"]
            )
        elif val < 0:
            raise ValueError(
                f"mem_thickness must be > 0, and has units of nm. Bad value given {val}"
            )
        else:
            self._sample_params["mem_thickness"] = float(val)

    @property
    def tilt_x(self) -> float:
        """Get the tilt angle in the x direction."""
        return self._tilt_x

    @tilt_x.setter
    def tilt_x(self, val: Union[float, int, None]) -> None:
        """Set the tilt angle in the x direction."""
        if val is None:
            self._tilt_x = self._default_params["theta_x"]
        else:
            self._tilt_x = float(val)

    @property
    def tilt_y(self) -> float:
        """Get the tilt angle in the y direction."""
        return self._tilt_y

    @tilt_y.setter
    def tilt_y(self, val: Union[float, int, None]) -> None:
        """Set the tilt angle in the y direction."""
        if val is None:
            self._tilt_y = self._default_params["tilt_y"]
        else:
            self._tilt_y = float(val)

    @property
    def beam_energy(self) -> float:
        """Get the beam energy."""
        return self._beam_energy

    @beam_energy.setter
    def beam_energy(self, val: Union[float, int, None]) -> None:
        """Set the beam energy."""
        if val is None:
            self._beam_energy = self._default_params["beam_energy"]
        elif val < 0:
            raise ValueError(
                f"beam_energy must be > 0, and has units of eV. Bad value given {val}"
            )
        else:
            self._beam_energy = float(val)

    @property
    def zscale(self) -> float:
        """Get the z-axis scale factor."""
        return self._zscale

    @property
    def scale(self) -> float:
        """Get the scale factor."""
        return self._scale

    @property
    def Mz(self) -> np.ndarray:
        """Get the z-component of the magnetization."""
        return self._mags[0]

    @property
    def My(self) -> np.ndarray:
        """Get the y-component of the magnetization."""
        return self._mags[1]

    @property
    def Mx(self) -> np.ndarray:
        """Get the x-component of the magnetization."""
        return self._mags[2]

    @property
    def mags(self) -> np.ndarray:
        """Get the magnetization array."""
        return self._mags

    @property
    def _mags_shape(self) -> Tuple[int, ...]:
        """Get the shape of the magnetization array."""
        return self.mags.shape[1:]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the magnetization array."""
        return self._mags_shape

    @mags.setter
    def mags(self, vals: np.ndarray) -> None:
        """Set the magnetization array."""
        vals = np.array(vals, dtype=np.float64)
        assert np.ndim(vals) == 4
        assert vals.shape[0] == 3
        self._mags = vals

    @property
    def shape_func(self) -> np.ndarray:
        """Get the shape function."""
        return self._shape_func

    @shape_func.setter
    def shape_func(self, val: np.ndarray) -> None:
        """Set the shape function."""
        val = np.array(val).astype(np.float32)
        if val.shape != self._mags_shape:
            raise ValueError(
                f"Shape function shape, {val.shape} should equal mags shape, {self._mags_shape}"
            )
        self._shape_func = val

    @property
    def flat_shape_func(self) -> np.ndarray:
        """Get the flattened shape function."""
        return self._flat_shape_func

    def get_flat_shape_func(self, sigma: float = 0) -> None:
        """
        Get the flattened shape function.

        Args:
            sigma (float, optional): Sigma value for Gaussian filter. Default is 0.
        """
        if (
            abs(self.tilt_x) < 0.1
            and abs(self.tilt_y) < 0.1
            or np.ptp(self.shape_func) == 0
        ):
            flat = self.shape_func.sum(axis=0)
        else:
            padwidth = np.max(self._mags_shape[1:]) // 2
            rot = np.pad(
                self._shape_func, ((padwidth, padwidth), (0, 0), (0, 0))
            ).astype(np.float32)
            if abs(self.tilt_x) >= 0.1:
                rot = ndi.rotate(rot, self.tilt_x, axes=(0, 1), reshape=False)
            if abs(self.tilt_y) >= 0.1:
                rot = ndi.rotate(rot, self.tilt_y, axes=(0, 2), reshape=False)
            flat = rot.sum(axis=0)
        if sigma > 0:
            flat = ndi.gaussian_filter(flat, sigma)
        self._flat_shape_func = flat

    def get_shape_func(self, mags: Optional[np.ndarray] = None) -> None:
        """
        Return 3D shape function of the magnetization.

        Args:
            mags (np.ndarray | None, optional): Magnetization array. Default is None.
        """
        if mags is None:
            mags = self.mags

        assert mags.ndim == 4
        assert mags.shape[0] == 3

        shape_func = np.any((mags != 0), axis=0)
        self.shape_func = shape_func

    def _interaction_constant(self) -> float:
        """Compute the interaction constant for the microscope."""
        epsilon = 0.5 * physcon.e / physcon.m_e / physcon.c**2
        lam = (
            physcon.h
            * 1.0e9
            / np.sqrt(2.0 * physcon.m_e * physcon.e)
            / np.sqrt(self.beam_energy + epsilon * self.beam_energy**2)
        )  # electron wavelength
        gamma = 1.0 + physcon.e * self.beam_energy / physcon.m_e / physcon.c**2
        sigma = (
            2.0 * np.pi * physcon.m_e * gamma * physcon.e * lam * 1.0e-18 / physcon.h**2
        )

        return sigma

    def _pre_B(self) -> float:
        """Compute the pre-factor for the B-phase."""
        return 2 * np.pi * self.B0 * self.zscale * self.scale / self._phi0

    def _pre_E(self) -> float:
        """Compute the pre-factor for the E-phase."""
        return self._interaction_constant() * self.sample_V0 * self.zscale

    def show_mags(
        self,
        xy_only: bool = False,
        s3D: bool = False,
        show_scale: bool = False,
        **kwargs,
    ) -> None:
        """
        Visualize the magnetization.

        Args:
            xy_only (bool, optional): Whether to show only the XY components. Default is False.
            s3D (bool, optional): Whether to show in 3D. Default is False.
            show_scale (bool, optional): Whether to show the scale. Default is False.
            **kwargs: Additional arguments for visualization.
        """
        scale = self.scale if show_scale else None
        if s3D:
            show_3D(
                self.Mx, self.My, self.Mz, title="magnetization", scale=scale, **kwargs
            )
        else:
            if xy_only:
                show_2D(
                    Vx=self.Mx.mean(axis=0),
                    Vy=self.My.mean(axis=0),
                    title="magnetization",
                    scale=scale,
                    **kwargs,
                )
            else:
                show_2D(
                    Vx=self.Mx.mean(axis=0),
                    Vy=self.My.mean(axis=0),
                    Vz=self.Mz.mean(axis=0),
                    title="magnetization",
                    scale=scale,
                    **kwargs,
                )

    def show_thickness_map(self, **kwargs) -> None:
        """
        Visualize the thickness map.

        Args:
            **kwargs: Additional arguments for visualization.
        """
        show_im(
            self.shape_func.sum(axis=0) * self.zscale,
            scale=self.scale,
            cbar_title="thickness (nm)",
            title="thickness map",
            **kwargs,
        )

    def visualize(self, show_thickness_map: bool = True, xy_only: bool = False) -> None:
        """
        Visualize the magnetization and thickness map.

        Args:
            show_thickness_map (bool, optional): Whether to show the thickness map. Default is True.
            xy_only (bool, optional): Whether to show only the XY components. Default is False.
        """
        if not show_thickness_map:
            self.show_mags(xy_only=xy_only)
            return

        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        self.show_mags(xy_only=xy_only, figax=(fig, axs[0]))
        self.show_thickness_map(figax=(fig, axs[1]))
        plt.show()

    def show_phase(self) -> None:
        """
        Visualize the B-phase and E-phase.
        """
        fig, axs = plt.subplots(ncols=2, figsize=(9, 4))
        self.show_phase_B(figax=(fig, axs[0]), cbar_title=None)
        self.show_phase_E(
            figax=(fig, axs[1]),
            ticks_off=True,
        )
        plt.tight_layout()
        plt.show()

    def show_phase_B(self, **kwargs) -> None:
        """
        Visualize the B-phase.

        Args:
            **kwargs: Additional arguments for visualization.
        """
        show_im(
            self.phase_B,
            scale=kwargs.pop("scale", self.scale),
            cbar_title=kwargs.pop("cbar_title", "rad"),
            title="phase_B",
            **kwargs,
        )

    def show_phase_E(self, **kwargs) -> None:
        """
        Visualize the E-phase.

        Args:
            **kwargs: Additional arguments for visualization.
        """
        show_im(
            self.phase_E,
            scale=kwargs.pop("scale", self.scale),
            cbar_title=kwargs.pop("cbar_title", "rad"),
            title="phase_E",
            **kwargs,
        )
