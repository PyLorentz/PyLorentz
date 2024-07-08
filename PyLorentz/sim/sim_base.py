import numpy as np
from PyLorentz.io import read_ovf
import os
from PyLorentz.visualize import show_2D, show_3D
import scipy.constants as physcon


class SimBase(object):
    _phi0 = 2.07e7  # Gauss*nm^2 flux quantum

    def __init__(
        self,
        mags: np.ndarray,
        scale: float,
        zscale: float,
        verbose: float | bool = 1,
    ):
        self._mags = mags
        self._mags_shape_func = None
        self._scale = scale
        self._zscale = zscale
        self._verbose = verbose

        self._phase_method = None
        self._tilt_x = None
        self._tilt_y = None
        self._beam_energy = None
        self._sample_params = {}
        self._image_params = {}

        self._phase_B = None
        self._phase_B_orig = None  # copy cuz modifications like noise
        self._phase_E = None
        self._phase_E_orig = None  # copy cuz modifications like noise

        self._default_params = {
            "phase_method": "mansuripur",  # "mansuripur" or "linsup"
            "B0": 1e4,  # gauss
            "sample_V0": 20.0,  # V
            "sample_xip0": 50.0,  # nm
            "mem_V0": 10.0,  # V
            "mem_xip0": 1e3,  # nm
            "mem_thk": 50.0,  # nm
            "theta_x": 0.0,  # degrees
            "tilt_y": 0.0,  # degrees
            "beam_energy": 200e3,  # eV
        }

        return

    def vprint(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    @property
    def phase_B(self):
        return self._phase_B

    @property
    def phase_E(self):
        return self._phase_E

    @property
    def phase_t(self):
        return self.phase_B + self.phase_E

    @property
    def phase_shape(self):
        return np.shape(self.phase_B)

    def set_sample_params(self, params: dict):
        self.B0 = params.get("B0")
        self.sample_V0 = params.get("sample_V0")
        self.sample_xip0 = params.get("sample_xip0")
        self.mem_V0 = params.get("mem_V0")
        self.mem_xip0 = params.get("mem_xip0")
        self.mem_thickness = params.get("mem_thickness")
        return

    @property
    def sample_params(self):
        return self._sample_params

    @property
    def phase_method(self):
        return self._phase_method

    @phase_method.setter
    def phase_method(self, mode: str | None):
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
                    f"Unknown phase method, {mode}, valid options are 'linsup' or"
                    + "'mansuripur'"
                )

    @property
    def B0(self):
        return self.sample_params["B0"]

    @B0.setter
    def B0(self, val: float | int | None):
        if val is None:
            self._sample_params["B0"] = float(self._default_params["B0"])
        elif val < 0:
            raise ValueError(
                f"B0 must be > 0, and has units of Gauss. Bad value given {val}"
            )
        else:
            self._sample_params["B0"] = float(val)

    @property
    def sample_V0(self):
        "mean inner potential sample"
        return self.sample_params["sample_V0"]

    @sample_V0.setter
    def sample_V0(self, val: float | int | None):
        if val is None:
            self._sample_params["sample_V0"] = float(self._default_params["sample_V0"])
        elif val < 0:
            raise ValueError(
                f"sample_V0 must be > 0, and has units of Volts. Bad value given {val}"
            )
        else:
            self._sample_params["sample_V0"] = float(val)

    @property
    def sample_xip0(self):
        "sample extinction distance nm"
        return self.sample_params["sample_xip0"]

    @sample_xip0.setter
    def sample_xip0(self, val: float | int | None):
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
    def mem_V0(self):
        "mean inner potential mem"
        return self.sample_params["mem_V0"]

    @mem_V0.setter
    def mem_V0(self, val: float | int | None):
        if val is None:
            self._sample_params["mem_V0"] = float(self._default_params["mem_V0"])
        elif val < 0:
            raise ValueError(
                f"mem_V0 must be > 0, and has units of Volts. Bad value given {val}"
            )
        else:
            self._sample_params["mem_V0"] = float(val)

    @property
    def mem_xip0(self):
        "mem extinction distance nm"
        return self.sample_params["mem_xip0"]

    @mem_xip0.setter
    def mem_xip0(self, val: float | int | None):
        if val is None:
            self._sample_params["mem_xip0"] = float(self._default_params["mem_xip0"])
        elif val < 0:
            raise ValueError(
                f"mem_xip0 must be > 0, and has units of nm. Bad value given {val}"
            )
        else:
            self._sample_params["mem_xip0"] = float(val)

    @property
    def mem_thickness(self):
        "membrane thickness nm"
        return self.sample_params["mem_thickness"]

    @mem_thickness.setter
    def mem_thickness(self, val: float | int | None):
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
    def tilt_x(self):
        return self._tilt_x

    @tilt_x.setter
    def tilt_x(self, val: float | int | None):
        if val is None:
            self._tilt_x = self._default_params["theta_x"]
        else:
            self._tilt_x = float(val)

    @property
    def tilt_y(self):
        return self._tilt_y

    @tilt_y.setter
    def tilt_y(self, val: float | int | None):
        if val is None:
            self._tilt_y = self._default_params["tilt_y"]
        else:
            self._tilt_y = float(val)

    @property
    def beam_energy(self):
        return self._beam_energy

    @beam_energy.setter
    def beam_energy(self, val: float | int | None):
        if val is None:
            self._beam_energy = self._default_params["beam_energy"]
        elif val < 0:
            raise ValueError(
                f"beam_energy must be > 0, and has units of eV. Bad value given {val}"
            )
        else:
            self._beam_energy = float(val)

    @property
    def zscale(self):
        return self._zscale

    @property
    def scale(self):
        return self._scale

    @property
    def Mz(self):
        return self._mags[0]

    @property
    def My(self):
        return self._mags[1]

    @property
    def Mx(self):
        return self._mags[2]

    @property
    def mags(self):
        return self._mags

    @property
    def _mags_shape(self):
        return self.mags.shape[1:]

    @mags.setter
    def mags(self, vals):
        vals = np.array(vals, dtype=np.float64)
        assert np.ndim(vals) == 4
        assert vals.shape[0] == 3
        self._mags = vals

    @property
    def mags_shape_func(self):
        return self._mags_shape_func

    @mags_shape_func.setter
    def mags_shape_func(self, val):
        val = np.array(val)
        if val.shape != self._mags_shape[1:]:
            raise ValueError(
                f"shape function shape, {val.shape} should equal mags shape, {self._mags_shape}"
            )
        self._mags_shape_func = val

    def show_mags(self, s3D=False, xy_only=False, **kwargs):

        if s3D:
            show_3D(self.Mz, self.My, self.Mx, **kwargs)
        else:
            if xy_only:
                show_2D(mag_y=self.My, mag_x=self.Mx, **kwargs)
            else:
                show_2D(mag_y=self.My, mag_x=self.Mx, mag_z=self.Mz, **kwargs)
        return

    def get_shape_func(self, mags: np.ndarray | None = None):
        """
        Return 3D shape function of the magnetization, 1 where magnetiztion mag = 1, 0
        where = 0
        """

        if mags is None:
            mags = self.mags

        assert mags.ndim == 4
        assert mags.shape[0] == 3

        shape_func = np.any((mags != 0), axis=0)
        self.mags_shape_func = shape_func

        return

    def _interaction_constant(self):
        """microscope sigma"""

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

    def _pre_B(self):
        return 2*np.pi*self.B0*self.zscale*self.scale/self._phi0

    def _pre_E(self):
        # TODO figure out how I want to include the phase shift from the membrane
        return self._interaction_constant * self.sample_V0*self.zscale
