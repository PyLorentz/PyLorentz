import copy
import os
from pathlib import Path
from typing import List, Optional, Union
from warnings import warn

import numpy as np
import scipy.ndimage as ndi

from PyLorentz.dataset import DefocusedDataset, ThroughFocalSeries
from PyLorentz.io import format_defocus, read_ovf
from PyLorentz.utils import Microscope

from .base_sim import BaseSim
from .comp_phase import LinsupPhase, MansuripurPhase


class SimLTEM(MansuripurPhase, LinsupPhase, BaseSim):
    """
    A class used to simulate Lorentz Transmission Electron Microscopy (LTEM) images.
    """

    def __init__(
        self,
        mags: np.ndarray,
        scale: float,
        zscale: float,
        verbose: Union[float, bool] = 1,
        ovf_file: Optional[os.PathLike] = None,
    ):
        """
        Initialize the SimLTEM object.

        Args:
            mags (np.ndarray): Magnetization array.
            scale (float): Scale factor for the simulation.
            zscale (float): Z-axis scale factor.
            verbose (float | bool, optional): Verbosity level. Default is 1.
            ovf_file (os.PathLike | None, optional): Path to OVF file. Default is None.
        """
        BaseSim.__init__(self, mags, scale, zscale, verbose)
        self._ovf_file = ovf_file

        self.get_shape_func()

    @classmethod
    def load_ovf(cls, file: Union[str, os.PathLike], verbose: int = 1) -> "SimLTEM":
        """
        Load an OVF file and initialize a SimLTEM object.

        Args:
            file (str | os.PathLike): Path to the OVF file.
            verbose (int, optional): Verbosity level. Default is 1.

        Returns:
            SimLTEM: An instance of the SimLTEM class.
        """
        file = Path(file).absolute()
        mags, scale, zscale = read_ovf(file, v=verbose)

        sim = cls(
            mags=mags,
            scale=scale,
            zscale=zscale,
            ovf_file=file,
            verbose=verbose,
        )

        return sim

    def compute_phase(
        self,
        method: Optional[str] = "mansuripur",
        tilt_x: Optional[float] = None,
        tilt_y: Optional[float] = None,
        beam_energy: Optional[float] = None,
        device: str = "cpu",
        multiproc: bool = True,
        **kwargs,
    ) -> None:
        """
        Compute the phase shift for the simulation.

        Args:
            method (str | None, optional): Phase computation method. Options are 'mansuripur' or 'linsup'. Default is 'mansuripur'.
            tilt_x (float | None, optional): Tilt angle in the x direction. Default is None.
            tilt_y (float | None, optional): Tilt angle in the y direction. Default is None.
            beam_energy (float | None, optional): Beam energy. Default is None.
            device (str, optional): Device to use for computation. Default is 'cpu'.
            multiproc (bool, optional): Whether to use multiprocessing. Default is True.
            **kwargs: Additional arguments for phase computation.
        """
        if method is not None or self.phase_method is None:
            self.phase_method = method
        if tilt_x is not None or self.tilt_x is None:
            self.tilt_x = tilt_x
        if tilt_y is not None or self.tilt_y is None:
            self.tilt_y = tilt_y
        if beam_energy is not None or self.beam_energy is None:
            self.beam_energy = beam_energy

        if self.phase_method == "mansuripur":
            phase_B, phase_E = self._calc_phase_mansuripur(**kwargs)
        elif self.phase_method == "linsup":
            phase_B, phase_E = self._calc_phase_linsup(
                multiproc=multiproc, device=device, **kwargs
            )
            self.get_flat_shape_func()
        else:
            raise ValueError(f"phase_method has bad value somehow: {self.phase_method}")

        self._phase_B = phase_B - phase_B.min()
        self._phase_E = phase_E - phase_E.min()

    def sim_images(
        self,
        defocus_values: Union[float, List[float]],  # single defocus value or list of them
        scope: Microscope,
        flip: bool = False,
        filter_sigma: float = 1,
        amorphous_bkg: Optional[Union[bool, float]] = None,
        padded_shape: Optional[tuple] = None,
    ) -> DefocusedDataset:
        """
        Simulate images at different defocus values.

        Args:
            defocus_values (float | list): Single defocus value or list of defocus values.
            scope (Microscope): Microscope object.
            flip (bool, optional): Whether to flip the phase. Default is False.
            filter_sigma (float, optional): Sigma value for Gaussian filter. Default is 1.
            amorphous_bkg (bool | float | None, optional): Amorphous background level. Default is None.
            padded_shape (tuple | None, optional): Shape for padding. Default is None.

        Returns:
            DefocusedDataset: A dataset containing simulated defocused images.
        """
        object_wave = self._generate_object_wave(filter_sigma, amorphous_bkg, flip=flip)
        self._object_wave = object_wave

        if isinstance(defocus_values, (float, int)):
            defocus_values = [defocus_values]

        self.vprint(
            f"Simulating images for defocus values: "
            + f"{', '.join([format_defocus(i, spacer=' ') for i in defocus_values])}"
        )

        images = []
        scope.scale = self.scale
        for defval in defocus_values:
            scope.defocus = defval
            images.append(scope.compute_image(object_wave, padded_shape=padded_shape))
        images = np.array(images)

        dd = DefocusedDataset(
            images=images,
            defvals=defocus_values,
            scale=self.scale,
            beam_energy=scope.E,
            simulated=True,
            verbose=self._verbose,
            data_files=self._ovf_file,
        )

        return dd

    def sim_TFS(
        self,
        defocus_values: Union[float, List[float]],  # single defocus value or list of them
        scope: Microscope,
        flip: bool = False,
        filter_sigma: float = 1,
        amorphous_bkg: Optional[Union[bool, float]] = None,
        padded_shape: Optional[tuple] = None,
    ) -> ThroughFocalSeries:
        """
        Simulate a Through Focal Series (TFS).

        for single or set of defocus values, record a through focal series with/without flip
        one defocus val, flip=False -> [+-, +0, ++]
        one defocus val, flip=True, -> [[+-, +0, ++], [--, -0, -+]]
        multiple defocus vals,
        everything goes into a DefocusedDataset object


        Args:
            defocus_values (float | list): Single defocus value or list of defocus values.
            scope (Microscope): Microscope object.
            flip (bool, optional): Whether to flip the phase. Default is False.
            filter_sigma (float, optional): Sigma value for Gaussian filter. Default is 1.
            amorphous_bkg (bool | float | None, optional): Amorphous background level. Default is None.
            padded_shape (tuple | None, optional): Shape for padding. Default is None.

        Returns:
            ThroughFocalSeries: A series of simulated images at different focal depths.
        """
        if isinstance(defocus_values, (float, int)):
            full_defvals = [-1 * abs(defocus_values), 0, abs(defocus_values)]
        else:
            defocus_values = np.sort(np.unique(np.abs(defocus_values)))
            if defocus_values[0] == 0:
                defocus_values = defocus_values[1:]
            full_defvals = np.concatenate([-1 * defocus_values[::-1], [0], defocus_values])

        self.vprint(
            f"Simulating images for defocus values: "
            + f"{', '.join([format_defocus(i, spacer=' ') for i in full_defvals])}"
        )
        if flip:
            self.vprint("Will simulate a TFS for both unflip and flip orientations.")

        seed = np.random.randint(1e9)
        object_wave = self._generate_object_wave(
            filter_sigma, amorphous_bkg, flip=False, seed=seed
        )
        if flip:
            object_wave_flip = self._generate_object_wave(
                filter_sigma, amorphous_bkg, flip=True, seed=seed
            )
        imstack = []
        flipstack = []
        scope.scale = self.scale
        for defval in full_defvals:
            scope.defocus = defval
            imstack.append(scope.compute_image(object_wave, padded_shape=padded_shape))
            if flip:
                flipstack.append(scope.compute_image(object_wave_flip, padded_shape=padded_shape))

        tfs = ThroughFocalSeries(
            imstack=imstack,
            flipstack=flipstack,
            flip=flip,
            scale=self.scale,
            defvals=full_defvals,
            beam_energy=scope.E,
            simulated=True,
            verbose=self._verbose,
        )

        return tfs

    def _generate_object_wave(
        self,
        filter_sigma: float = 1,
        amorphous_bkg: Optional[Union[bool, float]] = None,
        flip: bool = False,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate the object wave used to simulate images.

        Args:
            filter_sigma (float, optional): Sigma value for Gaussian filter. Default is 1.
            amorphous_bkg (bool | float | None, optional): Amorphous background level. Default is None.
            flip (bool, optional): Whether to flip the phase. Default is False.
            seed (int | None, optional): Random seed for generating noise. Default is None.

        Returns:
            np.ndarray: The generated object wave.
        """
        phase_E = self.phase_E.copy()
        if filter_sigma:
            phase_E = ndi.gaussian_filter(phase_E, sigma=filter_sigma)

        if flip:
            phase_t = (phase_E - self.phase_B).astype(np.float64)
        else:
            phase_t = (phase_E + self.phase_B).astype(np.float64)

        if amorphous_bkg:
            if isinstance(amorphous_bkg, bool):
                bkg_amount = 0.01
            else:
                bkg_amount = amorphous_bkg / 1000
            seed = np.random.randint(1e9) if seed is None else seed
            rng = np.random.default_rng(seed=seed)
            random_phase = rng.uniform(low=-1 * bkg_amount, high=bkg_amount, size=phase_t.shape)
            random_phase = ndi.gaussian_filter(random_phase, 5 / self.scale, mode="wrap")
            phase_t += random_phase

        if self.flat_shape_func is None:
            print("how the heck did you get here?")
            self.get_flat_shape_func()

        thk_map = self.flat_shape_func.copy()
        if filter_sigma:
            thk_map = ndi.gaussian_filter(thk_map, sigma=filter_sigma)

        amplitude = np.exp(
            -1
            * (
                np.ones_like(phase_t) * self.mem_thickness / self.mem_xip0
                + thk_map * self.zscale / self.sample_xip0
            )
        )
        if np.ptp(amplitude) == 0 and np.max(amplitude) < 0.05:
            warn(
                f"Amplitude is uniform and has a low value: {amplitude.max()}. "
                + "Consider increasing sample xip0 or decreasing sample thickness. "
            )
        object_wave = amplitude * np.exp(1.0j * phase_t)

        return object_wave

    def _add_noise(self):
        """
        Add noise to the simulated data.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError


    def copy(self):
        """Returns a deep copy of itself."""
        return copy.deepcopy(self)
