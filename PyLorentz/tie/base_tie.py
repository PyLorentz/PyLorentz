import numpy as np
from PyLorentz.dataset.defocused_dataset import DefocusedDataset as DD
import os
from pathlib import Path
from scipy.signal import convolve2d
import scipy.constants as physcon
from PyLorentz.visualize import show_im, show_2D
from PyLorentz.visualize.colorwheel import color_im, get_cmap
from PyLorentz.io.write import overwrite_rename, write_json, write_tif


class BasePhaseReconstruction(object):

    def __init__(
        self,
        save_dir: os.PathLike | None = None,
        name: str|None = None,
        scale: float | None = None,
        verbose: int | bool = 1,
    ):
        self._save_dir = Path(save_dir).absolute()
        self._name = name
        self._save_name = name
        self._verbose = verbose
        self._scale = scale
        self._overwrite = False

        self._results = {
            "By": None,
            "Bx": None,
            "phase_B": None,
        }

        return

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        if not isinstance(val, (float, int)):
            raise TypeError(f"scale must be float/int, not {type(val)}")
        if val <= 0:
            raise ValueError(f"scale must be >0, not {val}")
        self._scale = float(val)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def By(self):
        return self._results["By"]

    @property
    def Bx(self):
        return self._results["Bx"]

    @property
    def B(self):
        return np.array([self._results["By"], self._results["Bx"]])

    @property
    def phase_B(self):
        return self._results["phase_B"]

    def vprint(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, p):
        p = Path(p)
        if not p.parents[0].exists():
            raise ValueError(f"save dir parent does not exist: {p.parents[0]}")
        else:
            self._save_dir = p

    def _save_keys(self, keys, defval:float|None=None, overwrite:bool|None=None, **kwargs):
        ovr = overwrite if overwrite is not None else self._overwrite
        for key in keys:
            if defval is not None:
                name = f"{self._save_name}_{self._fmt_defocus(defval)}_{key}.tiff"
            else:
                name = f"{self._save_name}_{key}.tiff"
            fname = self.save_dir / name

            if "color" in key:
                image = color_im(
                    self._results["By"],
                    self._results["Bx"],
                    **kwargs,
                )
            else:
                image = self._results[key]

            write_tif(image,
                      fname,
                      self.scale,
                      v=self._verbose,
                      overwrite=ovr,
                      color="color" in key)
        return

    @staticmethod
    def _fmt_defocus(defval: float|int, digits:int=3):
        "returns a string of defocus value converted to nm, um, or mm as appropriate"

        rnd_digits = len(str(round(defval))) - digits
        rnd_abs = round(defval, -1*rnd_digits)

        if abs(rnd_abs) < 1e3: # nm
            return f"{rnd_abs:.0f}nm"
        elif abs(rnd_abs) < 1e6: # um
            return f"{rnd_abs/1e3:.0f}um"
        elif abs(rnd_abs) < 1e9: # mm
            return f"{rnd_abs/1e3:.0f}mm"
        return

    def induction_from_phase(self, phase):
        """Gives integrated induction in T*nm from a magnetic phase shift

        Args:
            phi (ndarray): 2D numpy array of size (dimy, dimx), magnetic component of the
                phase shift in radians
            del_px (float): in-plane scale of the image in nm/pixel

        Returns:
            tuple: (By, Bx) where each By, Bx is a 2D numpy array of size (dimy, dimx)
                corresponding to the y/x component of the magnetic induction integrated
                along the z-direction. Has units of T*nm (assuming del_px given in units
                of nm/pixel)
        """
        grad_y, grad_x = np.gradient(np.squeeze(phase), edge_order=2)
        pre_B = physcon.hbar / (physcon.e * self.scale) * 10**18  # T*nm^2
        Bx = pre_B * grad_y
        By = -1 * pre_B * grad_x
        return (By, Bx)


    def show_B(self, **kwargs):
        """
        show induction
        """
        show_2D(self.By, self.Bx, **kwargs)
        return

    # def show_phase(self):
    #     """
    #     show_phase
    #     """
    #     return


class BaseTIE(BasePhaseReconstruction):

    def __init__(
        self,
        save_dir: os.PathLike | None = None,
        scale: float | None = None,
        beam_energy: float|None=None,
        name: str|None = None,
        sym: bool = False,
        qc: float | None = None,
        verbose: int | bool = 1,
    ):
        BasePhaseReconstruction.__init__(self, save_dir, name, scale, verbose)
        self._sym = sym
        self._qc = qc
        self._qi = None
        self._pbcs = True
        self._beam_energy = beam_energy

        return

    @property
    def sym(self):
        return self._sym

    @sym.setter
    def sym(self, val):
        if isinstance(val, bool):
            self._sym = val
        else:
            raise ValueError(f"sym must be bool, not {type(val)}")

    @property
    def qc(self):
        return self._qc

    @qc.setter
    def qc(self, val):
        if val is None:
            self._qc = 0
        elif isinstance(val, (float, int)):
            if val < 0:
                raise ValueError(f"qc must be >= 0, not {val}")
            self._qc = float(val)
        else:
            raise ValueError(f"qc must be float, not {type(val)}")

    def _make_qi(self, shape: tuple, qc: float | None = None):
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
        else:  # normal Laplacian method
            # self.vprint("Reconstructing with normal Laplacian method")
            qi = 1 / q**2
        qi[0, 0] = 0
        self._qi = qi  # saves the freq dist
        return

    def _reconstruct_phase(self, infocus:np.ndarray, dIdZ:np.ndarray, defval:float):
        dimy, dimx = dIdZ.shape

        # Fourier transform of longitudinal derivatives
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
        phase = np.real(prefactor * -1 * np.fft.ifft2(fft2 * qc_mask * self._qi))

        if self.sym:
            d2y, d2x = phase.shape
            phase = phase[:d2y//2, :d2x//2]

        return phase

    def _pre_Lap(self, def_step=1):
        """Scaling prefactor used in the TIE reconstruction.

        Args:
            pscope (``Microscope`` object): Microscope object from
                microscopes.py
            def_step (float): The defocus value for which is being
                reconstructed. If using a longitudinal derivative, def_step
                should be 1.

        Returns:
            float: Numerical prefactor
        """
        epsilon = 0.5 * physcon.e / physcon.m_e / physcon.c**2
        lam = (
            physcon.h
            * 1.0e9
            / np.sqrt(2.0 * physcon.m_e * physcon.e)
            / np.sqrt(self._beam_energy + epsilon * self._beam_energy**2)
        ) # electron wavelength
        return -1 * self.scale**2 / (16 * np.pi**3 * lam * def_step)
