import numpy as np
from PyLorentz.io import read_ovf
import os
from PyLorentz.visualize import show_2D, show_3D
from .sim_base import SimBase
import time
import numba
from numba import jit
from tqdm import tqdm
from .comp_phase import LinsupPhase

class MansuripurPhase(SimBase):
    def _calc_phase_mansuripur(self):
        # calculate B phase
        # look at MagPy code and implement that version
        # including beam tilt

        # calculate E phase
        # if don't have that in magpy, use the pylorentz stuff

        return


class SimLTEM(MansuripurPhase, LinsupPhase, SimBase):
    def __init__(
        self,
        mags: np.ndarray,
        scale: float,
        zscale: float,
        verbose: float | bool = 1,
        ovf_file: os.PathLike | None = None,
    ):

        SimBase.__init__(self, mags, scale, zscale, verbose)
        self._ovf_file = ovf_file


        return

    @classmethod
    def load_ovf(cls, file, verbose=1):
        mags, scale, zscale = read_ovf(file, v=verbose)

        sim = cls(
            mags=mags,
            scale=scale,
            zscale=zscale,
            ovf_file=file,
            verbose=verbose,
        )

        sim.get_shape_func()

        return sim

    def compute_phase(
        self,
        method: str | None = "mansuripur",
        tilt_x: float | None = None,
        tilt_y: float | None = None,
        beam_energy=None,
        device="cpu",
        multiproc=True,
    ):
        if method is not None or self.phase_method is None:
            self.phase_method = method
        if tilt_x is not None or self.tilt_x is None:
            self.tilt_x = tilt_x
        if tilt_y is not None or self.tilt_y is None:
            self.tilt_y = tilt_y
        if beam_energy is not None or self.beam_energy is None:
            self.beam_energy = beam_energy

        if self.phase_method == "mansuripur":
            self._calc_phase_mansuripur()
        elif self.phase_method == "linsup":
            mphi, ephi = self._calc_phase_linsup(multiproc=False, device="cpu")
        else:
            raise ValueError(f"phase_method has bad value somehow: {self.phase_method}")

        self._phase_B = mphi
        self._phase_E = ephi
        return
