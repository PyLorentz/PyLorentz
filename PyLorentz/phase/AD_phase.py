import os
from datetime import datetime, timedelta
import warnings
from typing import Optional, Union
from PyLorentz.dataset.defocused_dataset import DefocusedDataset
from typing import List, Optional, Union

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


class ADPhase(BasePhaseReconstruction):

    sample_params = {
        "dirt_V0": 20,
        "dirt_xip0": 10,
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
    ):
        self.dd = dd
        if save_dir is None and dd.data_dir is not None:
            topdir = Path(dd.data_dir)
            if topdir.exists():
                save_dir = topdir / "ADPhase_outputs"

        super().__init__(save_dir, name, dd.scale, verbose)

        self.sample_params = self.sample_params | sample_params

        self.device = device
        self.inp_ims = torch.tensor(self.dd.images, device=self.device, dtype=torch.float32)
        self.defvals = dd.defvals
        self.scope = scope
        self.shape = dd.shape
        self._rng = np.random.default_rng(rng_seed)

        # to be set later:
        self._guess_phase = None
        self._runtype = None
        self.recon_amp = None
        self.recon_phase = None

        self.TFs = self.get_TFs()

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
    def TFs(self):
        return self._TFs

    @TFs.setter
    def TFs(self, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.tensor(arr, device=self.device, dtype=torch.complex64)
        assert len(arr.shape) == 3, f"Bad TF shape: {arr.shape}, should be {self.dd.images.shape}"
        assert len(arr) == len(self.dd), f"len TFs {len(arr)} != # defvals {len(self.dd)}"
        self._TFs = arr

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
        im = torch.tensor(im, device=self.device, dtype=torch.float32)
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
        im = torch.tensor(im, device=self.device, dtype=torch.float32)
        if im.shape != self.shape:
            raise ValueError(
                f"Guess amp shape, {im.shape} should match input image shape, {self.shape}"
            )
        self._guess_amp = im

    @property
    def input_noise(self):
        """noise used as input for one or both DIPs"""
        return self._input_noise

    @input_noise.setter
    def input_noise(self, im: np.ndarray | torch.Tensor):
        if not isinstance(im, torch.Tensor):
            im = torch.tensor(im, device=self.device, dtype=torch.float32)
        if im.shape != self.shape:
            raise ValueError(
                f"Input noise shape, {im.shape} should match input image shape, {self.shape}"
            )
        self._input_noise = im

    def reconstruct(
        self,
        num_iter,
        model: nn.Module | list[nn.Module] | None = None,  # tuple of two models if solve_amp
        num_pretrain_iter=0,
        guess_amp=None,
        solve_amp=False,
        LRs=None,
        scheduler_type=None,
        save=False,
        noise_frac=None,
        guess_phase="SITIE",
        reset=True,
        print_every=-1,
        verbose=None,
        scale_amplitude=False,
        **kwargs,  # scheduler params
    ):
        self._num_pretrain_iter = num_pretrain_iter
        if self._noise_frac is not None:
            self._noise_frac = noise_frac
        if verbose is not None:
            self._verbose = verbose

        self._noise_frac = noise_frac if noise_frac is not None else self._noise_frac

        if not reset:
            # check if have errors and stuff
            # if not, set reset = True
            pass

        if reset:
            self._set_guess_phase(guess_phase)
            self._set_guess_amp(guess_amp)
            self._error_iterations = []
            self._LR_iterations = []

            if model:
                self._use_DIP = True
                inp_noise = self._rng.random((1, *self.shape)) * 2 - 1
                self.input_noise = torch.tensor(
                    inp_noise, device=self.device, dtype=torch.float32, requires_grad=False
                )
                if solve_amp:
                    self._runtype = "DIP-amp"
                    DIP_phase, DIP_amp = model
                    DIP_phase = DIP_phase.to(self.device)
                    DIP_amp = DIP_amp.to(self.device)
                    DIP_phase.apply(weight_reset)  # TODO test that resetting works
                    DIP_amp.apply(weight_reset)
                    self.optimizer = torch.optim.Adam(
                        [
                            {"params": DIP_phase.parameters(), "lr": LRs["phase"]},
                            {"params": DIP_amp.parameters(), "lr": LRs["amp"]},
                        ]
                    )
                else:
                    self._runtype = "DIP"
                    DIP_phase = model.to(self.device)
                    DIP_phase.apply(weight_reset)
                    DIP_amp = None
                    self.optimizer = torch.optim.Adam(
                        [{"params": DIP_phase.parameters(), "lr": LRs["phase"]}]
                    )
                    self.recon_amp = self.guess_amp.copy()
                    self.recon_amp.requires_grad = False

                ## pretrain DIP
                self._pretrain_DIP(DIP_phase, DIP_amp)
            else:
                self._use_DIP = False
                self.recon_phase = self.guess_phase.copy()
                self.recon_phase.requires_grad = True
                if solve_amp:
                    self.recon_amp = self.guess_amp.copy()
                    self.recon_amp.requires_grad = True
                    self._runtype = "AD-amp"
                    self.optimizer = torch.optim.Adam(
                        [
                            {"params": self.recon_phase, "lr": LRs["phase"]},
                            {"params": self.recon_amp, "lr": LRs["amp"]},
                        ]
                    )
                else:
                    self._runtype = "AD"
                    self.optimizer = torch.optim.Adam(
                        [{"params": self.recon_phase, "lr": LRs["phase"]}]
                    )

            self.scheduler = self._get_scheduler(scheduler_type, LRs, self.optimizer, **kwargs)
        else:
            # check make sure everything exists maybe
            pass

        self._recon_loop(num_iter, print_every, DIP_phase)  # maybe have loop_amp

        ## add a num_iters or len function that looks at errors to get total iters run

        return

    def _recon_loop(self, num_iter, print_every, DIP_phase:nn.Module|None):
        # not including amp stuff for now
        stime = datetime.now()

        self._amp_phase = self._amp_to_phi(self.recon_amp)
        self._amp_phase -= self._amp_phase.min()

        for a0 in range(num_iter):
            if self._noise_frac >= 0:
                # have way of rng this? reproducible
                self.input_noise = self.input_noise + self._noise_frac * torch.randn(
                    self.input_noise.shape, device=self.device
                )
            if DIP_phase is not None:
                self.recon_phase = DIP_phase.forward(self.input_noise)[0]

            loss = self._compute_loss()

            pass

        return

    def _sim_images(self):
        obj_waves = self.recon_amp * torch.exp(1.j * self.recon_phase)
        img_waves = torch.fft.ifft2(torch.fft.fft2(obj_waves) * self.TFs)
        images = torch.abs(img_waves)**2
        return images

    def _compute_loss(self):
        guess_ims = self._sim_images()
        MSE_loss = torch.mean((guess_ims - self.inp_ims)**2)
        if self._use_DIP or (self.LRs["TV_phase_weight"] == 0 and self.LRs["TV_amp_weight"] == 0):
            return MSE_loss
        else:
            TV_loss = self._calc_TV_loss_PBC()
        return

    def _calc_TV_loss_PBC():
        #TODO write this func
        pass

    @property
    def amp2phi(self):
        return self.sample_params["dirt_V0"] * self.scope.sigma * self.sample_params["dirt_xip0"]

    def _amp_to_phi(self, amp):
        a = -1 * torch.log(amp)
        b = a - torch.min(a)
        return b * self.amp2phi

    def _pretrain_DIP(self, DIP_phase: nn.Module, DIP_amp: nn.Module | None):
        if self._num_pretrain_iter >= 0:
            for _ in range(self._num_pretrain_iter):
                loss = self._get_loss_pretrain(DIP_phase, DIP_amp)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self._verbose >= 2:
                show_im(
                    DIP_phase.forward(self.input_noise).squeeze().cpu().detach().numpy(),
                    title=f"Recon phase after pre-training DIP for {self._num_pretrain_iter} iters",
                )
                if DIP_amp is not None:
                    show_im(
                        DIP_amp.forward(self.input_noise).squeeze().cpu().detach().numpy(),
                        title=f"Recon amp after pre-training DIP for {self._num_pretrain_iter} iters",
                    )

    def _get_loss_pretrain(self, DIP_phase: nn.Module, DIP_amp: nn.Module | None):
        pred_phase = DIP_phase.forward(self.input_noise).squeeze()
        loss = torch.mean((pred_phase - self.guess_phase) ** 2)
        if DIP_amp is not None:
            pred_amp = DIP_amp.forward(self.input_noise).squeeze()
            loss += torch.mean((pred_amp - self.guess_amp) ** 2)
        return loss

    def _get_scheduler(self, mode: str | None, LRs, optimizer, **kwargs):
        mode = mode.lower()
        LR = LRs["phase"]
        if mode == "none":
            scheduler = None
        elif mode == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=LR / 4,
                max_lr=LR * 4,
                step_size_up=100,
                mode="triangular2",
                cycle_momentum=False,
            )
        elif mode == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.75,
                patience=kwargs.get("plateau_patience", 100),
                threshold=1e-4,
                min_lr=LR / 20,
                verbose=True,
            )
        elif mode == "exp":
            gamma = 0.9997
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        return scheduler

    def _set_guess_amp(self, guess_amp=None):
        if guess_amp is not None:
            self.guess_amp = guess_amp
        else:
            im = self.dd.images[len(self.dd) // 2]
            thresh = im.min() + np.ptp(im) / 10
            guess_amp = np.where(im >= thresh, 1, 0).astype(np.float32)
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
            sitie = SITIE(self.dd)
            sitie.reconstruct()
            if self._verbose:
                print("SITIE guess phase:")
                sitie.visualize()
            guess_phase = sitie.phase_B
        self.guess_phase = torch.tensor(guess_phase, dtype=torch.float32)
        return

    def set_LRs():

        return

    def set_scheduler():

        return

    def visualize():
        return

    def calc_accuracy(self, true_phase, recon_phase):
        # get accuracy, SS, PSNR etc.
        return

    def get_TFs(self):
        tfs = np.array([self._get_TF(df) for df in self.defvals])
        return torch.tensor(tfs, device=self.device, dtype=torch.complex64)

    def _get_TF(self, defocus):
        self.scope.defocus = defocus
        return self.scope.get_transfer_function(self.scale, self.shape)
