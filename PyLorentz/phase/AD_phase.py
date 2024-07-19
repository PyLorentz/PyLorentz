import os
from datetime import datetime, timedelta
import warnings
from typing import Optional, Union
from PyLorentz.dataset.defocused_dataset import DefocusedDataset
from typing import List, Optional, Union
from matplotlib.ticker import FormatStrFormatter

import torch
from torch import nn
import numpy as np
import scipy.constants as physcon
from scipy.signal import convolve2d
from pathlib import Path

from PyLorentz.phase.base_phase import BasePhaseReconstruction
from PyLorentz.utils import Microscope
from .sitie import SITIE
from .DIP_NN import weight_reset
from PyLorentz.visualize import show_im
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


class ADPhase(BasePhaseReconstruction):

    _default_sample_params = {
        "dirt_V0": 20,
        "dirt_xip0": 10,
    }

    _default_LRs = {
        "phase": 2e-4,  # 2e-4 for DIP, 0.2 otherwise
        "amp": 2e-4,  # 2e-4 for DIP, 0.02 otherwise
        "amp_scale": 0.1,  # used when not solve_amp
        "TV_phase_weight": 5e-3,  # only used if reconstructing without DIPs
        "TV_amp_weight": 5e-3,  # only used if reconstructing without DIPs
    }

    def __init__(
        self,
        dd: DefocusedDataset,
        device: torch.device | str | int,
        save_dir: Optional[os.PathLike] = None,
        name: Optional[str] = None,
        verbose: bool = 1,
        scope=None,
        sample_params: dict = {},
        rng_seed: int | None = None,
        LRs={},
        scheduler_type=None,
        noise_frac=1 / 30,
        guess_phase="SITIE",
        gaussian_sigma=1,
    ):
        self.dd = dd
        if save_dir is None and dd.data_dir is not None:
            topdir = Path(dd.data_dir)
            if topdir.exists():
                save_dir = topdir / "ADPhase_outputs"

        super().__init__(save_dir, name, dd.scale, verbose)

        self.sample_params = self._default_sample_params | sample_params
        self.LRs = self._default_LRs | LRs

        self.device = device
        self.inp_ims = torch.tensor(self.dd.images, device=self.device, dtype=torch.float32)
        self.defvals = dd.defvals
        self.scope = scope
        self.shape = dd.shape
        self._rng = np.random.default_rng(rng_seed)
        self._gaussian_sigma = gaussian_sigma
        self._noise_frac = noise_frac
        self._scheduler_type = scheduler_type

        # to be set later:
        self._guess_phase: torch.Tensor = None
        self._runtype: str = None
        self._recon_amp: torch.Tensor = None
        self._recon_phase: torch.Tensor = None
        self._use_DIP: bool = None
        self._solve_amp: bool = None
        self._solve_amp_scale: bool = None
        self._best_phase: torch.Tensor = None
        self.best_amp: torch.Tensor | None = None
        self.phase_iterations: list[torch.Tensor] = None
        self.amp_iterations: list[torch.Tensor] = None

        self._TFs = self.get_TFs()

        return

    @classmethod
    def from_array(
        cls,
        image: np.ndarray,
        scale: Union[float, int, None] = None,
        defval: Optional[List[float]] = None,
        beam_energy: Optional[float] = None,
        name: Optional[str] = None,
        save_dir: Optional[os.PathLike] = None,
    ):
        return

    @classmethod
    def from_TFS(cls):
        # conver tfs to dd and load
        return

    @property
    def recon_phase(self):
        if self._recon_phase is not None:
            ph = ndi.gaussian_filter(
                self._recon_phase.cpu().detach().numpy(), self._gaussian_sigma
            )
            ph -= ph.min()
            return ph
        else:
            return None

    @property
    def best_phase(self):
        if self._best_phase is not None:
            ph = ndi.gaussian_filter(self._best_phase.cpu().detach().numpy(), self._gaussian_sigma)
            ph -= ph.min()
            return ph
        else:
            return None

    @property
    def recon_amp(self):
        if self._recon_amp is not None:
            amp = ndi.gaussian_filter(self._recon_amp.cpu().detach().numpy(), self._gaussian_sigma)
            return amp
        else:
            return None

    @property
    def _TFs(self):
        return self.transfer_functions

    @_TFs.setter
    def _TFs(self, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.tensor(arr, device=self.device, dtype=torch.complex64)
        assert len(arr.shape) == 3, f"Bad TF shape: {arr.shape}, should be {self.dd.images.shape}"
        assert len(arr) == len(self.dd), f"len TFs {len(arr)} != # defvals {len(self.dd)}"
        self.transfer_functions = arr

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, microscope):
        if not isinstance(microscope, Microscope):
            raise TypeError(
                f"microscope must be a PyLorentz.utils.microscopes.Microscope object, received {type(microscope)}"
            )
        else:
            self._scope = microscope

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, dev: Union[int, str, torch.device]):
        # accept multiple input types, make sure is allowed for num available gpus
        if isinstance(dev, torch.device):
            self._device = dev
            ind = dev.index
        elif isinstance(dev, (str, int, float)):
            if dev in ["cpu", "CPU"]:
                warnings.warn("Setting device to cpu, this will be slow and might fail.")
                ind = "cpu"
            else:
                ind = int(dev)
                if ind >= torch.cuda.device_count():
                    raise ValueError(
                        f"device must be < num_devices, which is {torch.cuda.device_count()}. Received device {ind}"
                    )
        else:
            raise TypeError(f"Device should be int, str, or torch.device. Received {type(dev)}")

        self._device = torch.device(ind)
        if isinstance(ind, int):
            self.vprint(f"Proceeding with GPU {ind}: {torch.cuda.get_device_name(ind)}")

    @property
    def model_input(self):
        return self._model_input

    @model_input.setter
    def model_input(self, arr):
        # set to tensor on device
        # make sure is same size as image
        return

    @property
    def guess_phase(self):
        """phase used to pre-train the DIP"""
        return self._guess_phase

    @guess_phase.setter
    def guess_phase(self, im: Union[np.ndarray, torch.Tensor]):
        if not isinstance(im, torch.Tensor):
            im = torch.tensor(im, dtype=torch.float32)
        im = im.to(self.device)
        if im.shape != self.shape:
            raise ValueError(
                f"Guess phase shape, {im.shape} should match input image shape, {self.shape}"
            )
        self._guess_phase = im

    @property
    def guess_amp(self):
        """amplitude used to pre-train the DIP or to use as the amplitude if solve_amp == False"""
        return self._guess_amp

    @guess_amp.setter
    def guess_amp(self, im: np.ndarray | torch.Tensor):
        if not isinstance(im, torch.Tensor):
            im = torch.tensor(im, dtype=torch.float32)
        im = im.to(self.device)
        if im.shape != self.shape:
            raise ValueError(
                f"Guess amp shape, {im.shape} should match input image shape, {self.shape}"
            )
        self._guess_amp = im

    @property
    def input_DIP(self):
        """noise used as input for one or both DIPs"""
        return self._input_DIP

    @input_DIP.setter
    def input_DIP(self, im: np.ndarray | torch.Tensor):
        if not isinstance(im, torch.Tensor):
            im = torch.tensor(im, device=self.device, dtype=torch.float32)
        if (
            im.shape != self.dd.images.shape
        ):  # TODO not sure if this is correct for multiple images
            raise ValueError(
                f"Input noise shape, {im.shape} should match input image shape, {self.shape}"
            )
        self._input_DIP = im

    def reconstruct(
        self,
        num_iter,
        model: nn.Module | list[nn.Module] | None = None,  # tuple of two models if solve_amp
        num_pretrain_iter=0,
        solve_amp=False,
        solve_amp_scale=True,
        guess_amp: float | np.ndarray = None,  # thresh value or array
        LRs={},
        scheduler_type=None,
        save=False,
        name=None,
        save_dir=None,
        noise_frac=None,
        guess_phase: str | np.ndarray = "SITIE",
        reset=True,
        print_every=-1,
        verbose=1,
        save_iters_every=-1,
        **kwargs,  # scheduler params
    ):
        ### SETUP
        self._start_time = datetime.now()
        self._num_pretrain_iter = num_pretrain_iter
        if noise_frac is not None:
            self._noise_frac = noise_frac
        if verbose is not None:
            self._verbose = verbose
        self.LRs = self.LRs | LRs
        self._solve_amp_scale = solve_amp_scale if not solve_amp else False
        if isinstance(model, nn.Module):
            DIP_phase = model
            DIP_amp = None
        elif model is not None:
            DIP_phase, DIP_amp = model
        else:
            DIP_phase = DIP_amp = None
        if scheduler_type is not None:
            self._scheduler_type = scheduler_type

        if save:
            if name is None and self.name is None:
                now = self._start_time.strftime("%y%m%d-%H%M%S")
                if len(self.dd) == 1:
                    mode = "SIPRAD"
                else:
                    mode = f"N{len(self.dd)}AD"
            self._check_save_name(save_dir, name=f"{now}_{mode}")

        self._noise_frac = noise_frac if noise_frac is not None else self._noise_frac

        ### Initialization/reset
        if not reset:
            # check if have errors and stuff
            # if not, set reset = True
            raise NotImplementedError

        if reset:
            self._set_guess_phase(guess_phase)
            self._set_guess_amp(guess_amp)
            self.loss_iterations = []
            self.LR_iterations = []
            self.phase_iterations = []
            self.amp_iterations = []
            self._amp_scale = torch.tensor([1.0], dtype=torch.float32, device=self.device)

            if model:
                self._runtype = "DIP"
                self._use_DIP = True
                self._set_input_DIP()

                DIP_phase = DIP_phase.to(self.device)
                DIP_phase.apply(weight_reset)  # TODO test that resetting works
                self.optimizer = torch.optim.Adam(
                    [{"params": DIP_phase.parameters(), "lr": self.LRs["phase"]}],
                )

                if solve_amp:
                    self._solve_amp = True
                    self._runtype += "-amp"
                    DIP_amp = DIP_amp.to(self.device)
                    DIP_amp.apply(weight_reset)
                    self.optimizer.add_param_group(
                        {"params": DIP_amp.parameters(), "lr": self.LRs["amp"]},
                    )
                else:
                    self._solve_amp = False
                    DIP_amp = None
                    self._recon_amp = self.guess_amp.clone()
                    self._recon_amp.requires_grad = False

                ### pretrain DIP
                self._pretrain_DIP(DIP_phase, DIP_amp)

            else:
                self._runtype = "AD"
                self._use_DIP = False
                DIP_phase = DIP_amp = None
                self._recon_phase = self.guess_phase.clone()
                self._recon_phase.requires_grad = True
                self.optimizer = torch.optim.Adam(
                    [{"params": self._recon_phase, "lr": self.LRs["phase"]}]
                )
                if solve_amp:
                    self._solve_amp = True
                    self._runtype += "-amp"
                    self._recon_amp = self.guess_amp.clone()
                    self._recon_amp.requires_grad = True
                    self.optimizer = self.optimizer.add_param_group(
                        {"params": self._recon_amp, "lr": self.LRs["amp"]},
                    )
                else:
                    self._solve_amp = False
                    self._recon_amp = self.guess_amp.clone()
                    self._recon_amp.requires_grad = False

            if self._solve_amp_scale:
                self.optimizer.add_param_group({"params": self._amp_scale, "lr": self.LRs["amp_scale"]})
                self._amp_scale.requires_grad = True
            else:
                self._amp_scale.requires_grad = False

            self.scheduler = self._get_scheduler(**kwargs)
        else:
            # check make sure everything exists maybe
            pass

        ### Recon
        self.vprint("Reconstructing")
        self._recon_loop(
            num_iter, print_every, DIP_phase, save, save_iters_every
        )  # maybe have loop_amp

        ttime = timedelta(seconds=(datetime.now() - self._start_time).seconds)
        print(f"total time (h:m:s) = {ttime}")
        self._recon_phase -= self._recon_phase.min()
        self._best_phase -= self._best_phase.min()

        self.phase_B = self.best_phase

        return self

    def _recon_loop(
        self, num_iter, print_every, DIP_phase: nn.Module | None, save, save_iters_every
    ):
        # not including amp stuff for now
        self._amp_phase = self._amp_to_phi(self._recon_amp)
        self._amp_phase -= self._amp_phase.min()

        stime = self._start_time
        for a0 in tqdm(range(num_iter)):
            if self._noise_frac >= 0:
                # have way of rng this? reproducible
                self.input_DIP = self.input_DIP + self._noise_frac * torch.randn(
                    self.input_DIP.shape, device=self.device
                )
            if DIP_phase is not None:
                self._recon_phase = DIP_phase.forward(self.input_DIP)[0]

            loss = self._compute_loss()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss_iterations.append(loss.item())
            self.LR_iterations.append([pg["lr"] for pg in self.optimizer.param_groups])

            if (a0 == 0 or (a0 + 1) % print_every == 0) and print_every > 0:
                lrs = [f"{pg['lr']:.2e}" for pg in self.optimizer.param_groups]
                lrsp = ", ".join(lrs)
                ctime = timedelta(seconds=(datetime.now() - stime).seconds)
                self.vprint(f"{a0+1}/{num_iter} | {ctime} | loss {loss.item():.3e} | LR {lrsp}")
                stime = datetime.now()

            if (a0 == 0 or (a0 + 1) % save_iters_every == 0) and save_iters_every > 0:
                self.phase_iterations.append(self._recon_phase.detach().clone())
                if self._solve_amp:
                    self.amp_iterations.append(self._recon_amp.detach().clone())
                if self._verbose >= 2 and a0 != 0:
                    self.show_best()

            if a0 > 100 and loss.item() < min(self.loss_iterations[:-1]):
                self._best_phase = self._recon_phase.detach().clone()
                if self._solve_amp:
                    self.best_amp = self._recon_amp.detach().clone()
                if save:
                    # TODO add a checkpoint save function (once made save function)
                    raise NotImplementedError

            if self.scheduler is not None:
                if hasattr(self.scheduler, "cooldown"):  # is plateau
                    self.scheduler.step(loss.item())
                else:
                    self.scheduler.step()

        return

    def _sim_images(self):
        obj_waves = self._recon_amp * self._amp_scale * torch.exp(1.0j * self._recon_phase)
        img_waves = torch.fft.ifft2(torch.fft.fft2(obj_waves) * self._TFs)
        images = torch.abs(img_waves) ** 2
        return images

    def _compute_loss(self, return_seperate=False):
        """
        return_seperate used for tracking both losses/calculating components, not for training
        """
        guess_ims = self._sim_images()
        MSE_loss = torch.mean((guess_ims - self.inp_ims) ** 2)
        if self._use_DIP or (self.LRs["TV_phase_weight"] == 0 and self.LRs["TV_amp_weight"] == 0):
            if return_seperate:
                return MSE_loss, None
            else:
                return MSE_loss
        else:
            TV_loss = self._calc_TV_loss_PBC()
            if return_seperate:
                return MSE_loss, TV_loss
            else:
                return MSE_loss + TV_loss

    def _calc_TV_loss_PBC(self):
        # TODO write this func
        assert self._recon_phase.ndim == 2 and self._recon_amp.ndim == 2
        dy, dx = self.shape
        if self.LRs["TV_phase_weight"] > 0:
            phase_pad_h = F.pad(self._recon_phase[None, None], (0, 0, 0, 1), mode="circular")[0, 0]
            phase_pad_w = F.pad(self._recon_phase[None, None], (0, 1, 0, 0), mode="circular")[0, 0]
            TV_phase_h = torch.pow(phase_pad_h[1:, :] - phase_pad_h[:-1, :], 2).sum()
            TV_phase_w = torch.pow(phase_pad_w[:, 1:] - phase_pad_w[:, :-1], 2).sum()
            TV_phase = self.LRs["TV_phase_weight"] * (TV_phase_h + TV_phase_w) / (dy * dx)
        else:
            TV_phase = None

        if self._solve_amp and self.LRs["TV_amp_weight"] > 0:  # amp TV
            amp_pad_h = F.pad(self._recon_amp[None, None], (0, 0, 0, 1), mode="circular")[0, 0]
            amp_pad_w = F.pad(self._recon_amp[None, None], (0, 1, 0, 0), mode="circular")[0, 0]
            TV_amp_h = torch.pow(amp_pad_h[1:, :] - amp_pad_h[:-1, :], 2).sum()
            TV_amp_w = torch.pow(amp_pad_w[:, 1:] - amp_pad_w[:, :-1], 2).sum()
            TV_amp = self.LRs["TV_amp_weight"] > 0 * (TV_amp_h + TV_amp_w) / (dy * dx)

            if TV_phase is None:
                return TV_amp
            else:
                return (TV_amp + TV_phase) / 2

        else:
            return TV_phase

    @property
    def amp2phi(self):
        return self.sample_params["dirt_V0"] * self.scope.sigma * self.sample_params["dirt_xip0"]

    def _amp_to_phi(self, amp):
        a = -1 * torch.log(amp)
        b = a - torch.min(a)
        return b * self.amp2phi

    def _pretrain_DIP(self, DIP_phase: nn.Module, DIP_amp: nn.Module | None):
        if self._num_pretrain_iter >= 0:
            self.vprint(f"Pre-training")
            for _ in tqdm(range(self._num_pretrain_iter)):
                loss = self._get_loss_pretrain(DIP_phase, DIP_amp)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self._verbose >= 2:
                ph = DIP_phase.forward(self.input_DIP).squeeze().cpu().detach().numpy()
                ph -= ph.min()
                show_im(
                    ph,
                    title=f"Recon phase after pre-training DIP for {self._num_pretrain_iter} iters",
                )
                if self._solve_amp:
                    show_im(
                        DIP_amp.forward(self.input_DIP).squeeze().cpu().detach().numpy(),
                        title=f"Recon amp after pre-training DIP for {self._num_pretrain_iter} iters",
                    )

    def _get_loss_pretrain(self, DIP_phase: nn.Module, DIP_amp: nn.Module | None):
        pred_phase = DIP_phase.forward(self.input_DIP).squeeze()
        loss = torch.mean((pred_phase - self.guess_phase) ** 2)
        if self._solve_amp:
            pred_amp = DIP_amp.forward(self.input_DIP).squeeze()
            loss += torch.mean((pred_amp - self.guess_amp) ** 2)
        return loss

    def _get_scheduler(self, **kwargs):
        mode = str(self._scheduler_type).lower()
        LR = self.LRs["phase"]
        if mode == "none":
            scheduler = None
        elif mode == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=LR / 4,
                max_lr=LR * 4,
                step_size_up=100,
                mode="triangular2",
                cycle_momentum=False,
            )
        elif mode == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.75,
                patience=kwargs.get("plateau_patience", 100),
                threshold=1e-4,
                min_lr=LR / 20,
                verbose=True,
            )
        elif mode == "exp":
            gamma = 0.9997
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        else:
            raise ValueError(f"Unknown scheduler type: {mode}")
        return scheduler

    def _set_guess_amp(self, guess_amp=None):
        if isinstance(guess_amp, (np.ndarray, torch.Tensor)):
            self.guess_amp = guess_amp
        else:
            im = self.dd.images[len(self.dd) // 2]
            if isinstance(guess_amp, (float, int)):
                thresh = guess_amp # TODO update this to be percent saturated?
                guess_amp = np.where(im >= thresh, 1, 0).astype(np.float32)
            elif self._solve_amp:
                thresh = im.min() + np.ptp(im) / 10
                guess_amp = np.where(im >= thresh, 1, 0).astype(np.float32)
            else:
                guess_amp = np.ones_like(im).astype(np.float32)
            guess_amp *= np.sqrt(im.mean())
            guess_amp = torch.tensor(guess_amp, device=self.device, dtype=torch.float32)
            self.guess_amp = guess_amp

    def _set_guess_phase(self, guess_phase: str):
        guess_phase = guess_phase.lower()
        if guess_phase == "none":
            guess_phase = None
            self._num_pretrain_iter = 0
        elif guess_phase == "uniform":
            guess_phase = np.zeros(self.shape)
        elif guess_phase == "sitie":
            sitie = SITIE(self.dd, verbose=0)
            sitie.reconstruct()
            if self._verbose > 2:
                print("SITIE guess phase:")
                sitie.visualize(cbar=True)
            guess_phase = sitie.phase_B
        self.guess_phase = torch.tensor(guess_phase, dtype=torch.float32)
        return

    def _set_input_DIP(self, input_mode: str | None = None, guess_phase=None):
        # TODO test/have various options here, but SITIE is definitely best
        # inp_noise = self._rng.random((1, *self.shape)) * 2 - 1
        # inp_noise = self._rng.random((1, *self.shape)) * 2 - 1
        # self.input_DIP = torch.tensor(
        #     inp_noise, device=self.device, dtype=torch.float32, requires_grad=False
        # )
        sitie = SITIE(self.dd, verbose=0)
        sitie.reconstruct()
        guess_phase = sitie.phase_B
        self._input_DIP = torch.tensor(
            guess_phase[None, ...], device=self.device, dtype=torch.float32, requires_grad=False
        )

    def set_LRs():

        return

    def set_scheduler():

        return

    def visualize():
        return


    def get_TFs(self):
        tfs = np.array([self._get_TF(df) for df in self.defvals])
        return torch.tensor(tfs, device=self.device, dtype=torch.complex64)

    def _get_TF(self, defocus):
        self.scope.defocus = defocus
        return self.scope.get_transfer_function(self.scale, self.shape)

    def show_best(self, **kwargs):
        minloss_iter = np.argmin(self.loss_iterations)
        if self._solve_amp:
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            show_im(
                self.best_phase,
                title=f"best phase: iter {minloss_iter} / {len(self.loss_iterations)}",
                figax=(fig, axs[0]),
                cbar_title="rad",
            )
            show_im(
                self.best_amp,
                title=f"best amp: iter {minloss_iter} / {len(self.loss_iterations)}",
                figax=(fig, axs[1]),
            )
            plt.tight_layout()
            plt.show()
        else:
            show_im(
                self.best_phase,
                title=f"best phase: iter {minloss_iter} / {len(self.loss_iterations)}",
                scale=self.scale,
                cbar_title="rad",
            )

    def show_final(self, **kwargs):
        if self._solve_amp:
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            show_im(
                self.recon_phase,
                title=f"Recon phase iter {len(self.loss_iterations)}",
                figax=(fig, axs[0]),
            )
            show_im(
                self.recon_amp,
                title=f"Recon amp iter {len(self.loss_iterations)}",
                figax=(fig, axs[1]),
            )
            plt.tight_layout()
            plt.show()
        else:
            show_im(
                self.recon_phase,
                title=f"Recon phase iter {len(self.loss_iterations)}",
                scale=self.scale,
                cbar_title="rad",
            )

    def visualize(self):
        """
        Plot phase, B, loss plot with LRs
        """

        if self._solve_amp:
            raise NotImplementedError
        else:
            fig = plt.figure(figsize=(8,8))
            ax1 = fig.add_subplot(221)
            self.show_phase_B(figax=(fig, ax1), cbar_title=None)
            # self.show_recon(figax=(fig, ax1))
            ax2 = fig.add_subplot(222)
            self.show_B(figax=(fig, ax2))
            ax3 = fig.add_subplot(212)
            l1 = ax3.semilogy(self.loss_iterations, color="tab:blue", label="loss")
            ax3.set_xlabel("iterations")
            ax3.set_ylabel("loss")
            ax4 = ax3.twinx()
            LRs = np.array(self.LR_iterations)
            l2 = ax4.semilogy(LRs[:,0], color="tab:orange", label="LR")

            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            ax4.legend(lns, labs, loc=0)
            ax4.set_ylabel("LR")
            ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

            plt.show()



        return

    def __len__(self):
        return len(self.loss_iterations)
