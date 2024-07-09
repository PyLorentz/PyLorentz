import numpy as np
from PyLorentz.io import read_ovf, format_defocus
import os
from PyLorentz.visualize import show_2D, show_3D, show_im
from .sim_base import SimBase
from .comp_phase import LinsupPhase, MansuripurPhase
from PyLorentz.dataset import ThroughFocalSeries, DefocusedDataset
import scipy.ndimage as ndi
from warnings import warn
from PyLorentz.utils import Microscope


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

        self.get_shape_func()

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

        return sim

    def compute_phase(
        self,
        method: str | None = "mansuripur",
        tilt_x: float | None = None,
        tilt_y: float | None = None,
        beam_energy=None,
        device="cpu",
        multiproc=True,
        **kwargs,
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
            phase_B, phase_E = self._calc_phase_mansuripur(**kwargs)
        elif self.phase_method == "linsup":
            phase_B, phase_E = self._calc_phase_linsup(multiproc=True, device="cpu")
            self.get_flat_shape_func()
            # needed for image simulation, done in mansuripur, and here to prevent mistakes
        else:
            raise ValueError(f"phase_method has bad value somehow: {self.phase_method}")

        self._phase_B = phase_B
        self._phase_E = phase_E
        return

    def sim_images(
        self,
        defocus_values: float | list,  # single defocus value or list of them
        scope: Microscope,
        flip=False,
        filter_sigma: float = 1,
        amorphous_bkg=None,
        padded_shape=None,
    ) -> DefocusedDataset:
        # set defvals
        # defocus_values can be single item or list, will return images for each
        # if flip, will do image for -phase_B + phase_E
        object_wave = self._generate_object_wave(filter_sigma, amorphous_bkg, flip=flip)

        if isinstance(defocus_values, (float, int)):
            defocus_values = [defocus_values]

        self.vprint(
            f"Simulating images for defocus values: "
            + f"{', '.join([format_defocus(i, spacer=" ") for i in defocus_values])}"
        )

        images = []
        scope.scale = self.scale
        for defval in defocus_values:
            scope.defocus = defval
            images.append(scope.compute_image(object_wave, padded_shape=padded_shape))
        images = np.array(images)

        return images.squeeze()


    def _generate_object_wave(
        self, filter_sigma: float = 1, amorphous_bkg=None, flip: bool = False
    ):
        """
        generate the object wave used to simulate images
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

            random_phase = np.random.uniform(
                low=-1 * bkg_amount, high=bkg_amount, size=phase_t.shape
            )
            random_phase = ndi.gaussian_filter(
                random_phase, 5 / self.scale, mode="wrap"
            )
            phase_t += random_phase

        if self.flat_shape_func is None:
            # make sure theres no way to have the wrong tilt value used...
            # if mansuripur done for one tilt value, then linsup for another tilt value, could happen
            print("how the heck did you get here?")
            self.get_flat_shape_func()

        # units of pixels, multiplying by zscale is approximately correct in most cases
        # for large tilt values, requires zscale + scale
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
        if np.ptp(amplitude) == 0 and np.max(amplitude) < 0.01:
            warn(
                f"Amplitude is uniform and has a low value: {amplitude.max()}. "
                + "Consider increasing sample xip0 or decreasing sample thickness. "
            )
        object_wave = amplitude * np.exp(1.0j * phase_t)

        return object_wave

    def sim_TFS(
        self, defocus_values, scope, flip_series=False, **kwargs
    ) -> ThroughFocalSeries:
        # for single or set of defocus values, record a through focal series with/without flip
        # one defocus val, flip=False -> [+-, +0, ++]
        # one defocus val, flip=True, -> [[+-, +0, ++], [--, -0, -+]]
        # multiple defocus vals,
        # everything goes into a DefocusedDataset object

        return
