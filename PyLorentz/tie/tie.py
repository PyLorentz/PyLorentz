import numpy as np
from PyLorentz.dataset.defocused_dataset import DefocusedDataset as DD
import os
from PyLorentz.tie.base_tie import BaseTIE
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from PyLorentz.visualize import show_im, show_2D


class TIE(BaseTIE):

    def __init__(
        self,
        dd: DD,
        save_dir: os.PathLike | None = None,
        name: str = "",
        sym: bool = False,
        qc: float | None = None,
        verbose: int = 1,
    ):
        self.dd = dd
        if save_dir is None and dd.data_dir is not None:
            topdir = Path(dd.data_dir)
            if topdir.exists():
                save_dir = topdir / "TIE_outputs"

        BaseTIE.__init__(self, save_dir=save_dir, name=name, verbose=verbose)
        self.qc = qc  # for type checking
        self.sym = sym
        self.scale = dd.scale
        self._flip = dd.flip
        self._beam_energy = dd.energy

        self._results["phase_E"] = None
        self._results["infocus"] = None
        self._results["dIdZ_B"] = None
        self._results["dIdZ_E"] = None
        self._defval = None
        self._defval_index = None

        if not self.dd._preprocessed:
            raise ValueError(f"dataset has not be preprocessed")

        return

    def reconstruct(
        self,
        index: int | None = None,
        name: str = "",
        sym: bool = False,
        qc: float | None = None,
        flip: bool | None = None,
        save: bool = False,
        save_dir: os.PathLike | None = None,
        verbose: int | bool = 1,
        pbcs: bool|None=None,
    ):
        if index is None:
            index = self.dd.len_tfs - 1
        assert isinstance(index, int)
        assert index < len(self.dd.defvals_index)
        self._defval_index = index
        self._defval = self.dd.defvals_index[index]
        self.name = name
        self.sym = sym
        if qc is not None:
            self.qc = qc
        if flip is not None:
            self.flip = flip
        if pbcs is not None:
            self._pbcs = pbcs
        self._verbose = verbose
        if save:
            # checking here that dir exists before running
            self.save_dir = save_dir

        self.vprint(
            f"Performing TIE reconstruction with defocus ± {self._defval/1e3:.2f} um, index = {index}"
        )
        if self.flip:
            self.vprint(
                f"Reconstructing with two TFS flip/unflip to seperate phase_B and phase_E"
            )
        else:
            self.vprint(f"Reconstructing with a single TFS")

        # setup data
        dimy, dimx = self.dd.shape

        # select images
        recon_stack, infocus_im = self._select_images()
        self._results["infocus"] = infocus_im.copy() * self.dd.mask
        recon_mask = self.dd.mask.copy()

        if self.sym:
            dimy *= 2
            dimx *= 2
            recon_stack = self._sym_stack(recon_stack)
            recon_mask = self._sym_stack([recon_mask])[0]

        self._make_qi((dimy, dimx))

        # get derivatives
        dIdZ_B, dIdZ_E = self._get_derivatives(recon_stack, recon_mask, self.flip)
        self._results["dIdZ_B"] = dIdZ_B.copy()
        self._results["dIdZ_E"] = dIdZ_E.copy()

        # temp checks
        assert dimy, dimx == recon_stack.shape[1:]
        assert np.min(recon_stack) > 0

        self.vprint("Calling TIE solver")

        phase_B = self._reconstruct_phase(infocus_im, dIdZ_B, self._defval)
        self._results["phase_B"] = phase_B - phase_B.min()
        By, Bx = self.induction_from_phase(phase_B)
        self._results["By"] = By
        self._results["Bx"] = Bx

        if self.flip:
            phase_E = self._reconstruct_phase(infocus_im, dIdZ_E, self._defval)
            self._results["phase_E"] = phase_E - phase_E.min()

        if save:
            # TODO
            pass

        return

    def _get_derivatives(self, stack, mask, flip):
        if flip:
            assert len(stack) == 5, f"Expect stack len 5 with flip, got {len(stack)}"
            dIdZ_B = 0.5 * ((stack[3] - stack[0]) - (stack[4] - stack[1]))
            dIdZ_E = 0.5 * ((stack[3] - stack[0]) + (stack[4] - stack[1]))
        else:
            assert len(stack) == 3, f"Expect stack len 3 for no flip, got {len(stack)}"
            dIdZ_B = stack[2] - stack[0]
            dIdZ_E = None

        dIdZ_B *= mask
        dIdZ_B -= dIdZ_B.sum() / mask.sum()
        dIdZ_B *= mask

        if flip:
            dIdZ_E *= mask
            dIdZ_E -= dIdZ_E.sum() / mask.sum()
            dIdZ_E *= mask

        return dIdZ_B, dIdZ_E

    def _select_images(self):
        # if ptie.flip == True: returns [ +- , -- , 0 , ++ , -+ ]
        # elif ptie.flip == False: returns [+-, 0, ++]
        # where first +/- is unflip/flip, second +/- is over/underfocus.

        under_ind = self.dd.len_tfs // 2 - (self._defval_index + 1)
        over_ind = self.dd.len_tfs // 2 + (self._defval_index + 1)

        if self.flip:
            stack = np.stack(
                [
                    self.dd.imstack[under_ind],
                    self.dd.flipstack[under_ind],
                    self.dd.infocus,
                    self.dd.imstack[over_ind],
                    self.dd.flipstack[over_ind],
                ]
            )
        else:
            stack = np.stack(
                [
                    self.dd.imstack[under_ind],
                    self.dd.imstack[self.dd.len_tfs // 2],
                    self.dd.imstack[over_ind],
                ]
            )

        stack = self._scale_stack(stack) + 1e-9
        # inverting background of infocus because dividing by it
        # stack[len(stack) // 2] += 1 - self.dd.mask
        infocus = stack[len(stack) // 2]
        infocus += 1 - self.dd.mask

        return stack, infocus

    def _sym_stack(self, imstack, mode="even"):
        """Makes the even symmetric extension of an image (4x as large).

        Args:
            image (2D array): input image (M,N)

        Returns:
            ``ndarray``: Numpy array of shape (2M,2N)
        """
        if np.ndim(imstack) == 2:
            imstack = imstack[None,]
        dimz, dimy, dimx = imstack.shape
        imi = np.zeros((dimz, dimy * 2, dimx * 2))
        imi[:, :dimy, :dimx] = imstack
        if mode == "even":
            imi[:, dimy:, :dimx] = np.flip(imstack, axis=1)
            imi[:, :, dimx:] = np.flip(imi[:, :, :dimx], axis=2)
        elif mode == "odd":
            imi[:, dimy:, :dimx] = -1 * np.flip(imstack, axis=1)
            imi[:, :, dimx:] = -1 * np.flip(imi[:, :, :dimx], axis=2)
        else:
            raise ValueError(f"`mode` should be `even` or `odd`, not `{mode}`")
        return imi

    def _scale_stack(self, stack):
        """Scale a stack of images so all have the same total intensity.

        Args:
            imstack (list): List of 2D arrays.

        Returns:
            list: List of same shape as imstack
        """
        imstack = stack.copy()
        tots = np.sum(imstack, axis=(1, 2))
        t = np.max(tots) / tots
        imstack *= t[..., None, None]
        return imstack / np.max(imstack)

    @property
    def defval(self):
        if self._defval is None:
            print(f"defval is None or has not yet been specified with an index")
        return self._defval

    @property
    def flip(self):
        return self._flip

    @flip.setter
    def flip(self, val: bool | None):
        if val is None:
            self._flip = self.dd.flip
            return
        elif not isinstance(val, bool):
            raise TypeError(f"flip must be bool, not {type(val)}")
        if self.dd.flip:
            if not val:
                warnings.warn(
                    f"Setting flip=False even though dataset has flip/unflip tfs"
                )
            self._flip = val
        else:
            if val:
                raise ValueError(
                    f"Cannot set flip=True because dataset has only only one TFS"
                )
            else:
                self._flip = val

    @property
    def phase_E(self):
        if self.flip:
            return self._results["phase_E"]
        else:
            if self._results["phase_E"] is not None:
                self.vprint("Returning old phase_E as currently flip=False")
            else:
                raise ValueError(f"phase_E does not exist because flip=False")

    def visualize(self):
        """
        show phase + induction, if flip then show phase_e too
        options to save
        """
        if self.flip:
            ncols = 3
        else:
            ncols = 2

        fig, axs = plt.subplots(ncols=ncols, figsize=(4*ncols, 3))

        show_im(self.phase_B, title="Magnetic phase shift", scale=self.scale, figax=(fig, axs[0]), cbar=False)

        if self.flip:
            show_im(self.phase_E, title="Electrostatic phase shift", scale=self.scale, figax=(fig, axs[1]), ticks_off=True, cbar=False)

        show_2D(self.By, self.Bx, figax=(fig, axs[-1]), title="Integrated induction map")

        plt.show()
        return



class SITE(BaseTIE):

    def __init__(self):
        return
