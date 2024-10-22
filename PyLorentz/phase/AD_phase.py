import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as TvT
except (ModuleNotFoundError, ImportError) as e:
    torch = None

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = None


from matplotlib.ticker import FormatStrFormatter
from torch import nn
from tqdm import tqdm

from PyLorentz.dataset.defocused_dataset import DefocusedDataset
from PyLorentz.io.write import write_json
from PyLorentz.phase.base_phase import BasePhaseReconstruction
from PyLorentz.utils import Microscope
from PyLorentz.visualize import show_2D, show_im

from .DIP_NN import weight_reset
from .sitie import SITIE


class ADPhase(BasePhaseReconstruction):
    """
    ADPhase class for phase reconstruction using defocused datasets and DIPs.
    """

    _default_sample_params = {
        "dirt_V0": 20,
        "dirt_xip0": 10,
    }

    _default_LRs = {
        "phase": 2e-4,  # 2e-4 for DIP, 0.2 otherwise
        "amp": 2e-4,  # 2e-4 for DIP, 0.02 otherwise
        "amp_scale": 0.1,  # used when not solve_amp
        "amp2phi_scale": 1,
        "TV_phase_weight": 5e-3,  # only used if reconstructing without DIPs
        "TV_amp_weight": 5e-3,  # only used if reconstructing without DIPs
    }

    def __init__(
        self,
        dd: DefocusedDataset,
        device: Union[str, int],
        save_dir: Optional[os.PathLike] = None,
        name: Optional[str] = None,
        verbose: bool = 1,
        scope: Optional[Microscope] = None,
        sample_params: dict = {},
        rng_seed: Optional[int] = None,
        LRs: dict = {},
        scheduler_type: Optional[str] = None,
        noise_frac: float = 1 / 100,
        gaussian_sigma: float = 1,
    ):
        """
        Initialize the ADPhase object.

        Args:
            dd (DefocusedDataset): The defocused dataset.
            device (Union[str, int]): The device to use (CPU or GPU).
            save_dir (Optional[os.PathLike], optional): Directory to save results.
            name (Optional[str], optional): Name for the results.
            verbose (bool, optional): Verbosity level.
            scope (Optional[Microscope], optional): Microscope object.
            sample_params (dict, optional): Sample parameters.
            rng_seed (Optional[int], optional): Random seed.
            LRs (dict, optional): Learning rates for optimization.
            scheduler_type (Optional[str], optional): Type of learning rate scheduler.
            noise_frac (float, optional): Fraction of noise to add.
            gaussian_sigma (float, optional): Sigma value for Gaussian filter.
        """
        self.dd = dd
        if len(dd) == 1:
            self._mode = "SIPRAD"
        else:
            self._mode = "ADPhase"
        if save_dir is None and dd.data_dir is not None:
            topdir = Path(dd.data_dir)
            if topdir.exists():
                save_dir = topdir / f"{self._mode}_outputs"

        super().__init__(save_dir, name, dd.scale, verbose)

        self.sample_params = self._default_sample_params | sample_params
        self.LRs = self._default_LRs | LRs

        self.device = device
        self.inp_ims = torch.tensor(self.dd.images, device=self.device, dtype=torch.float32)
        self.defvals = dd.defvals
        self.scope = scope
        self.shape = dd.shape
        self._rng = np.random.default_rng(rng_seed)
        self._noise_frac = noise_frac
        self._scheduler_type = scheduler_type

        # to be set later:
        self._guess_phase: Optional[Tensor] = None
        self._runtype: Optional[str] = None
        self._recon_amp: Optional[Tensor] = None
        self._recon_phase: Optional[Tensor] = None
        self._use_DIP: Optional[bool] = None
        self._solve_amp: Optional[bool] = None
        self._solve_amp_scale: Optional[bool] = None
        self._best_phase: Optional[Tensor] = None
        self._best_amp: Optional[Tensor] = None
        self._best_iter: Optional[int] = None
        self._phase_iterations: List[Tensor] = []
        self._amp_iterations: List[Tensor] = []
        self.phase_iterations: Optional[np.ndarray] = None
        self.amp_iterations: Optional[np.ndarray] = None
        self.loss_iterations = []
        self.LR_iterations = []

        self.gaussian_sigma = gaussian_sigma
        self._TFs = self.get_TFs()

    @property
    def recon_phase(self) -> Optional[np.ndarray]:
        """
        Returns the reconstructed phase after applying Gaussian filter.

        Returns:
            Optional[np.ndarray]: Reconstructed phase image.
        """
        if self._recon_phase is not None:
            ph = ndi.gaussian_filter(
                self._recon_phase.cpu().detach().numpy(), self._gaussian_sigma
            )
            ph -= ph.min()
            return ph
        else:
            return None

    @property
    def best_phase(self) -> Optional[np.ndarray]:
        """
        Returns the best phase after applying Gaussian filter.

        Returns:
            Optional[np.ndarray]: Best phase image.
        """
        if self._best_phase is not None:
            ph = ndi.gaussian_filter(self._best_phase.cpu().detach().numpy(), self._gaussian_sigma)
            ph -= ph.min()
            return ph
        else:
            return None

    def set_best_phase(self, iter_ind: int = -1) -> None:
        """
        Sets the best phase from the specified iteration index.

        Args:
            iter_ind (int, optional): Index of the iteration to use for the best phase.
        """
        self._best_phase, iter = self._phase_iterations[iter_ind]
        self._best_iter = iter
        self.phase_B = self.best_phase

    @property
    def best_amp(self) -> Optional[np.ndarray]:
        """
        Returns the best amplitude after applying Gaussian filter.

        Returns:
            Optional[np.ndarray]: Best amplitude image.
        """
        if self._best_amp is not None:
            amp = ndi.gaussian_filter(self._best_amp.cpu().detach().numpy(), self._gaussian_sigma)
            return amp
        else:
            return None

    @property
    def gaussian_sigma(self) -> float:
        """
        Returns the Gaussian sigma value.

        Returns:
            float: Gaussian sigma value.
        """
        return self._gaussian_sigma

    @gaussian_sigma.setter
    def gaussian_sigma(self, val: Optional[float]) -> None:
        """
        Sets the Gaussian sigma value and updates the blurring transformation.

        Args:
            val (Optional[float]): Gaussian sigma value.

        Raises:
            TypeError: If `val` is not a numeric type.
            ValueError: If `val` is less than 0.
        """
        if val is None:
            self._gaussian_sigma = 0
            self._blurrer = None
        elif not isinstance(val, (float, int)):
            raise TypeError(f"gaussian_sigma must be numeric, received {type(val)}")
        elif val < 0:
            raise ValueError(f"gaussian_sigma must be >= 0 or None, received {val}")
        else:
            self._blurrer = TvT.GaussianBlur(kernel_size=(9, 9), sigma=(val, val))
            self._gaussian_sigma = val
        if self.best_phase is not None:
            self.phase_B = self.best_phase
        self._set_recon_iterations()

    @property
    def recon_amp(self) -> Optional[np.ndarray]:
        """
        Returns the reconstructed amplitude after applying Gaussian filter.

        Returns:
            Optional[np.ndarray]: Reconstructed amplitude image.
        """
        if self._recon_amp is not None:
            amp = ndi.gaussian_filter(self._recon_amp.cpu().detach().numpy(), self._gaussian_sigma)
            return amp
        else:
            return None

    @property
    def _TFs(self) -> Tensor:
        """
        Returns the transfer functions.

        Returns:
            Tensor: Transfer functions.
        """
        return self.transfer_functions

    @_TFs.setter
    def _TFs(self, arr: Union[Tensor, np.ndarray]) -> None:
        """
        Sets the transfer functions.

        Args:
            arr (Union[Tensor, np.ndarray]): Transfer functions array.

        Raises:
            AssertionError: If `arr` does not match expected shape.
        """
        if not isinstance(arr, torch.Tensor):
            arr = torch.tensor(arr, device=self.device, dtype=torch.complex64)
        assert len(arr.shape) == 3, f"Bad TF shape: {arr.shape}, should be {self.dd.images.shape}"
        assert len(arr) == len(self.dd), f"len TFs {len(arr)} != # defvals {len(self.dd)}"
        self.transfer_functions = arr

    @property
    def scope(self) -> Optional[Microscope]:
        """
        Returns the microscope object.

        Returns:
            Optional[Microscope]: Microscope object.
        """
        return self._scope

    @scope.setter
    def scope(self, microscope: Microscope) -> None:
        """
        Sets the microscope object.

        Args:
            microscope (Microscope): Microscope object to set.

        Raises:
            TypeError: If `microscope` is not of type `Microscope`.
        """
        if not isinstance(microscope, Microscope):
            raise TypeError(
                f"microscope must be a PyLorentz.utils.microscopes.Microscope object, received {type(microscope)}"
            )
        else:
            self._scope = microscope

    @property
    def device(self) -> str:
        """
        Returns the device used for computation.

        Returns:
            str: Device for computation.
        """
        return self._device

    @device.setter
    def device(self, dev: Union[int, str]) -> None:
        """
        Sets the device for computation.

        Args:
            dev (Union[int, str]): Device identifier.

        Raises:
            TypeError: If `dev` is not of type int or str.
            ValueError: If `dev` exceeds available GPUs.
        """
        if isinstance(dev, torch.device):
            if dev.type == "gpu":
                self._device = dev
                ind = dev.index
            elif dev.type == "cuda":
                self._device = dev
            elif dev.type == "cpu":
                self._device = dev
            else:
                raise TypeError(
                    f"Unknown device type: {dev.type} This can likely be fixed easily."
                )
        elif isinstance(dev, (str, int, float)):
            if dev in ["cpu", "CPU"]:
                warnings.warn("Setting device to cpu, this will be slow and might fail.")
                ind = "cpu"
            elif dev in ["gpu", "GPU"]:
                assert torch.cuda.is_available(), f"No GPUs available"
                ind = 0
            else:
                assert torch.cuda.is_available(), f"No GPUs available"
                ind = int(dev)
                if ind >= torch.cuda.device_count():
                    raise ValueError(
                        f"device must be < num_devices, which is {torch.cuda.device_count()}. Received device {ind}"
                    )
            self._device = torch.device(ind)
            if isinstance(ind, int):
                self.vprint(f"Proceeding with GPU {ind}: {torch.cuda.get_device_name(ind)}")
        else:
            raise TypeError(f"Device should be int, str, or torch.device. Received {type(dev)}")

    ## TODO FIND
    @property
    def model_input(self) -> Optional[Tensor]:
        """
        Returns the model input tensor.

        Returns:
            Optional[Tensor]: Model input tensor.
        """
        return self._model_input

    @model_input.setter
    def model_input(self, arr: Tensor) -> None:
        """
        Sets the model input tensor.

        Args:
            arr (Tensor): Input tensor to set.
        """
        # set to tensor on device
        # make sure is same size as image
        return

    @property
    def guess_phase(self) -> Optional[Tensor]:
        """
        Returns the guess phase used to pre-train the DIP.

        Returns:
            Optional[Tensor]: Guess phase tensor.
        """
        return self._guess_phase

    @guess_phase.setter
    def guess_phase(self, im: Union[np.ndarray, Tensor]) -> None:
        """
        Sets the guess phase.

        Args:
            im (Union[np.ndarray, Tensor]): Guess phase image.

        Raises:
            ValueError: If `im` shape does not match input image shape.
        """
        if not isinstance(im, torch.Tensor):
            im = torch.tensor(im, dtype=torch.float32)
        im = im.to(self.device)
        if im.shape != self.shape:
            raise ValueError(
                f"Guess phase shape, {im.shape} should match input image shape, {self.shape}"
            )
        self._guess_phase = im

    @property
    def guess_amp(self) -> Optional[Tensor]:
        """
        Returns the guess amplitude used to pre-train the DIP or if `solve_amp` is False.

        Returns:
            Optional[Tensor]: Guess amplitude tensor.
        """
        return self._guess_amp

    @guess_amp.setter
    def guess_amp(self, im: Union[np.ndarray, Tensor]) -> None:
        """
        Sets the guess amplitude.

        Args:
            im (Union[np.ndarray, Tensor]): Guess amplitude image.

        Raises:
            ValueError: If `im` shape does not match input image shape.
        """
        if not isinstance(im, torch.Tensor):
            im = torch.tensor(im, dtype=torch.float32)
        im = im.to(self.device)
        if im.shape != self.shape:
            raise ValueError(
                f"Guess amp shape, {im.shape} should match input image shape, {self.shape}"
            )
        self._guess_amp = im

    @property
    def input_DIP(self) -> Optional[Tensor]:
        """
        Returns the noise used as input for one or both DIPs.

        Returns:
            Optional[Tensor]: Input noise tensor.
        """
        return self._input_DIP

    @input_DIP.setter
    def input_DIP(self, im: Union[np.ndarray, Tensor]) -> None:
        """
        Sets the input noise for DIPs.

        Args:
            im (Union[np.ndarray, Tensor]): Input noise image.

        Raises:
            ValueError: If `im` shape does not match input image shape.
        """
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
        num_iter: int,
        model: Optional[Union[nn.Module, List[nn.Module]]] = None,
        num_pretrain_iter: int = 0,
        solve_amp: bool = False,
        solve_amp_scale: bool = True,
        guess_amp: Optional[Union[float, np.ndarray]] = None,
        LRs: dict = {},
        scheduler_type: Optional[str] = None,
        save: bool = False,
        name: Optional[str] = None,
        save_dir: Optional[os.PathLike] = None,
        noise_frac: Optional[float] = None,
        guess_phase: Union[str, np.ndarray, None] = "SITIE",
        reset: bool = True,
        print_every: int = -1,
        verbose: int = 1,
        store_iters_every: int = -1,
        qc: Optional[any] = None,
        **kwargs,  # scheduler params
    ) -> None:
        """
        Performs the reconstruction process.

        Args:
            num_iter (int): Number of iterations for reconstruction.
            model (Optional[Union[nn.Module, List[nn.Module]]], optional): Model or list of models for DIP.
            num_pretrain_iter (int, optional): Number of pretraining iterations.
            solve_amp (bool, optional): Whether to solve for amplitude.
            solve_amp_scale (bool, optional): Whether to solve amplitude scale.
            guess_amp (Optional[Union[float, np.ndarray]], optional): Guess amplitude.
            LRs (dict, optional): Learning rates for optimization.
            scheduler_type (Optional[str], optional): Type of learning rate scheduler.
            save (bool, optional): Whether to save results.
            name (Optional[str], optional): Name for the saved results.
            save_dir (Optional[os.PathLike], optional): Directory to save results.
            noise_frac (Optional[float], optional): Fraction of noise to add.
            guess_phase (Union[str, np.ndarray, None], optional): Guess phase or method to obtain it.
            reset (bool, optional): Whether to reset the model.
            print_every (int, optional): Frequency of printing progress.
            verbose (int, optional): Verbosity level.
            store_iters_every (int, optional): Frequency of storing iterations.
            qc (Optional[any], optional): Quality control object.
            **kwargs: Additional keyword arguments for scheduler parameters.
        """
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
            self._use_DIP = True
            DIP_phase = model
            DIP_amp = None
        elif model is not None:
            self._use_DIP = True
            DIP_phase, DIP_amp = model
        else:
            self._use_DIP = False
            DIP_phase = DIP_amp = None
        if scheduler_type is not None:
            self._scheduler_type = scheduler_type
        self._qc = qc

        if save:
            if name is None and self.name is None:
                now = self._start_time.strftime("%y%m%d-%H%M%S")
                if len(self.dd) == 1:
                    mode = "SIPRAD"
                    self._results["input_image"] = self.dd.images[0]
                else:
                    mode = f"N{len(self.dd)}AD"
            self._check_save_name(save_dir, name=f"{now}_{mode}")

        self._noise_frac = noise_frac if noise_frac is not None else self._noise_frac

        ### Initialization/reset
        reset = True if len(self.loss_iterations) == 0 else reset

        if reset:
            # pretty sure set_guess_phase is only relevant if not using DIP, but need to confirm
            self._set_guess_phase(guess_phase)
            self._set_guess_amp(guess_amp)
            self.loss_iterations = []
            self.LR_iterations = []
            self._phase_iterations = []
            self._amp_iterations = []
            self._amp_scale = torch.tensor([1.0], dtype=torch.float32, device=self.device)
            self._amp2phi_scale = torch.tensor(
                [self._get_amp2phi_scale()], dtype=torch.float32, device=self.device
            )

            if self._use_DIP:
                self._runtype = "DIP"
                self._set_input_DIP()
                DIP_phase = DIP_phase.to(self.device)
                self.optimizer = torch.optim.Adam(
                    [{"params": DIP_phase.parameters(), "lr": self.LRs["phase"]}],
                )
                DIP_phase.apply(weight_reset)
                if solve_amp:
                    self._runtype += "-amp"
                    DIP_amp.apply(weight_reset)
                    DIP_amp = DIP_amp.to(self.device)
                    self.optimizer.add_param_group(
                        {"params": DIP_amp.parameters(), "lr": self.LRs["amp"]},
                    )
                ### pretrain DIP
                self._pretrain_DIP(DIP_phase, DIP_amp)

            else:
                self._runtype = "AD"
                self._recon_phase = self.guess_phase.clone()
                if solve_amp:
                    self._runtype += "-amp"
                    self._recon_amp = self.guess_amp.clone()

        else:
            # maybe should check that more things exist, but shouldn't have to
            self.LR_iterations = list(self.LR_iterations)
            self.loss_iterations = list(self.loss_iterations)

        # reinitializing optimizer here so have chance to change scheduler, LRs, etc.
        if reset or scheduler_type != "continue":
            if self._use_DIP:
                DIP_phase = DIP_phase.to(self.device)
                self.optimizer = torch.optim.Adam(
                    [{"params": DIP_phase.parameters(), "lr": self.LRs["phase"]}],
                )

                if solve_amp:
                    self._solve_amp = True
                    DIP_amp = DIP_amp.to(self.device)
                    self.optimizer.add_param_group(
                        {"params": DIP_amp.parameters(), "lr": self.LRs["amp"]},
                    )
                else:
                    self._solve_amp = False
                    DIP_amp = None
                    self._recon_amp = self.guess_amp.clone()
                    self._recon_amp.requires_grad = False

            else:
                DIP_phase = DIP_amp = None
                self._recon_phase.requires_grad = True
                self.optimizer = torch.optim.Adam(
                    [{"params": self._recon_phase, "lr": self.LRs["phase"]}]
                )
                if solve_amp:
                    self._solve_amp = True
                    self._recon_amp.requires_grad = True
                    self.optimizer = self.optimizer.add_param_group(
                        {"params": self._recon_amp, "lr": self.LRs["amp"]},
                    )
                else:
                    self._solve_amp = False
                    self._recon_amp = self.guess_amp.clone()
                    self._recon_amp.requires_grad = False

            if self._solve_amp_scale:
                self.optimizer.add_param_group(
                    {"params": self._amp_scale, "lr": self.LRs["amp_scale"]}
                )
                self._amp_scale.requires_grad = True
                if self._recon_amp.min() != self._recon_amp.max():
                    amp2phi_LR = self.LRs.get("amp2phi_scale", self.LRs["amp_scale"])
                    self.optimizer.add_param_group(
                        {"params": self._amp2phi_scale, "lr": amp2phi_LR}
                    )
                    self._amp2phi_scale.requires_grad = True

            else:
                self._amp_scale.requires_grad = False

            self.scheduler = self._get_scheduler(**kwargs)

            if save_dir is not None or self.save_dir is not None:
                self._check_save_name(save_dir, name, mode=f"{self._mode}_{self._runtype}")

        ### Recon
        self.vprint("Reconstructing")
        self._recon_loop(num_iter, print_every, DIP_phase, DIP_amp, save, store_iters_every)

        ttime = timedelta(seconds=(datetime.now() - self._start_time).seconds)
        print(f"total time (h:m:s) = {ttime}")
        self._recon_phase -= self._recon_phase.min()
        if self._best_phase is not None:
            self._best_phase -= self._best_phase.min()
            self.phase_B = self.best_phase
        else:
            warnings.warn("Was unable to find a best_phase, recon likely failed.")
        self.LR_iterations = np.array(self.LR_iterations)
        self.loss_iterations = np.array(self.loss_iterations)
        self._set_recon_iterations()

        return self

    def _recon_loop(
        self,
        num_iter: int,
        print_every: int,
        DIP_phase: Optional[nn.Module],
        DIP_amp: Optional[nn.Module],
        save: bool,
        store_iters_every: int,
    ) -> None:
        """
        Runs the reconstruction loop for a given number of iterations.

        Args:
            num_iter (int): Number of iterations to run.
            print_every (int): Frequency of printing progress information.
            DIP_phase (Optional[nn.Module]): DIP module for phase reconstruction.
            DIP_amp (Optional[nn.Module]): DIP module for amplitude reconstruction.
            save (bool): Whether to save the best reconstruction.
            store_iters_every (int): Frequency of storing intermediate iterations.
        """
        stime = self._start_time
        for a0 in tqdm(range(num_iter)):
            if self._noise_frac >= 0:
                self.input_DIP += self._noise_frac * torch.randn(
                    self.input_DIP.shape, device=self.device
                )
            if DIP_phase is not None:
                self._recon_phase = DIP_phase(self.input_DIP)[0]
                if self._solve_amp and DIP_amp is not None:
                    self._recon_amp = torch.abs(DIP_amp(self.input_DIP)[0])
                    if (a0 + 1) % 100 == 0:
                        print(f"a0 {a0} applying amp constraints")
                        self._apply_amp_constraints()

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

            if (a0 == 0 or (a0 + 1) % store_iters_every == 0) and store_iters_every > 0:
                self._phase_iterations.append([self._recon_phase.detach().clone(), a0 + 1])
                if self._solve_amp:
                    self._amp_iterations.append([self._recon_amp.detach().clone(), a0 + 1])
                if self._verbose >= 2 and a0 != 0:
                    self.show_final()

            if len(self.loss_iterations) > 100 and loss.item() < min(self.loss_iterations[:-1]):
                self._best_phase = self._recon_phase.detach().clone()
                self._best_iter = len(self.loss_iterations)
                self.phase_B = self.best_phase
                if self._solve_amp:
                    self._best_amp = self._recon_amp.detach().clone()
                if save:
                    # TODO maybe add a checkpoint save function (once made save function)
                    raise NotImplementedError

            if self.scheduler is not None:
                if hasattr(self.scheduler, "cooldown"):  # is plateau
                    self.scheduler.step(loss.item())
                else:
                    self.scheduler.step()

    def _apply_amp_constraints(self, mode: str = "binary") -> None:
        """
        Apply amplitude constraints based on the specified mode.

        Args:
            mode (str): The mode of constraints. Currently only "binary" is implemented.
        """
        if mode == "binary":
            ampr = self._recon_amp.ravel()
            highval = torch.mode(torch.round(ampr, decimals=2))[0]
            threshval = highval * 3 / 4
            lowval = torch.sort(ampr)[len(ampr) // 10]
            amp = torch.where(self._recon_amp > threshval, highval, lowval)
            if self._blurrer is not None:
                amp = self._blurrer(amp[None])[0]
            self._recon_amp = amp
        else:
            raise NotImplementedError

    def _sim_images(self) -> Tensor:
        """
        Simulate images from the current amplitude and phase reconstructions.

        Returns:
            Tensor: The simulated images.
        """
        obj_waves = (
            self._recon_amp
            * self._amp_scale
            * torch.exp(1.0j * (self._recon_phase - self._amp2phi_scale * self._recon_amp))
        )
        img_waves = torch.fft.ifft2(torch.fft.fft2(obj_waves) * self._TFs)
        images = torch.abs(img_waves) ** 2
        return images

    def _compute_loss(
        self, return_seperate: bool = False
    ) -> Union[Tensor, tuple[Tensor, Optional[Tensor]]]:
        """
        Compute the loss for the current reconstruction.

        Args:
            return_seperate (bool): Whether to return separate MSE and TV losses.

        Returns:
            Union[Tensor, tuple[Tensor, Optional[Tensor]]]: The total loss, and optionally the MSE and TV losses.
        """
        guess_ims = self._sim_images()
        MSE_loss = torch.mean((guess_ims - self.inp_ims) ** 2)
        if self.LRs["TV_phase_weight"] == 0 and (
            self.LRs["TV_amp_weight"] == 0 or not self._solve_amp
        ):
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

    def _calc_TV_loss_PBC(self) -> Optional[Tensor]:
        """
        Calculate the total variation loss with periodic boundary conditions.

        Returns:
            Optional[Tensor]: The total variation loss.
        """
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

        if self._solve_amp and self.LRs["TV_amp_weight"] > 0:
            amp_pad_h = F.pad(self._recon_amp[None, None], (0, 0, 0, 1), mode="circular")[0, 0]
            amp_pad_w = F.pad(self._recon_amp[None, None], (0, 1, 0, 0), mode="circular")[0, 0]
            TV_amp_h = torch.pow(amp_pad_h[1:, :] - amp_pad_h[:-1, :], 2).sum()
            TV_amp_w = torch.pow(amp_pad_w[:, 1:] - amp_pad_w[:, :-1], 2).sum()
            TV_amp = self.LRs["TV_amp_weight"] * (TV_amp_h + TV_amp_w) / (dy * dx)

            if TV_phase is None:
                return TV_amp
            else:
                return (TV_amp + TV_phase) / 2
        else:
            return TV_phase

    def _set_recon_iterations(self) -> None:
        """
        Update phase_iterations and amp_iterations with filtered tensors converted to np arrays.
        """
        if len(self._phase_iterations) > 0:
            self.phase_iterations = []
            for iter in self._phase_iterations:
                ph = ndi.gaussian_filter(iter[0].cpu().detach().numpy(), self._gaussian_sigma)
                ph -= ph.min()
                self.phase_iterations.append((ph, iter[1]))

        if len(self._amp_iterations) > 0:
            self.amp_iterations = []
            for iter in self._amp_iterations:
                amp = ndi.gaussian_filter(iter[0].cpu().detach().numpy(), self._gaussian_sigma)
                self.amp_iterations.append((amp, iter[1]))

    def _get_amp2phi_scale(self) -> float:
        """
        Calculate the scale factor for converting amplitude to phase.

        Returns:
            float: The scale factor.
        """
        return self.sample_params["dirt_V0"] * self.scope.sigma * self.sample_params["dirt_xip0"]

    def _pretrain_DIP(self, DIP_phase: nn.Module, DIP_amp: nn.Module | None):
        """Perform the pre-training of the DIP model.

        Args:
            DIP_phase (nn.Module): Guess phase (also used as input) to train towards
            DIP_amp (nn.Module | None): Guess amp (also used as input) to train towards
        """
        if self._num_pretrain_iter > 0:
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
        """Helper function for `self._pretrain_DIP`"""
        pred_phase = DIP_phase.forward(self.input_DIP).squeeze()
        loss = torch.mean((pred_phase - self.guess_phase) ** 2)
        if self._solve_amp:
            pred_amp = DIP_amp.forward(self.input_DIP).squeeze()
            loss += torch.mean((pred_amp - self.guess_amp) ** 2)
        return loss

    def _get_scheduler(self, **kwargs):
        """Return a torch scheduler according to `self._scheduler_type`"""
        mode = str(self._scheduler_type).lower()
        LR = self.LRs["phase"]
        if mode == "none":
            scheduler = None
        elif mode == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=kwargs.get("scheduler_base_lr", LR / 4),
                max_lr=kwargs.get("scheduler_max_lr", LR * 4),
                step_size_up=kwargs.get("scheduler_step_size_up", 100),
                mode=kwargs.get("scheduler_mode", "triangular2"),
                cycle_momentum=kwargs.get("scheduler_momentum", False),
            )
        elif mode.startswith("plat"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=kwargs.get("scheduler_factor", 0.75),
                patience=kwargs.get("scheduler_patience", 100),
                threshold=kwargs.get("scheduler_threshold", 1e-4),
                min_lr=kwargs.get("scheduler_min_lr", LR / 20),
            )
        elif mode in ["exp", "gamma"]:
            gamma = kwargs.get("scheduler_gamma", 0.9997)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        else:
            raise ValueError(f"Unknown scheduler type: {mode}")
        return scheduler

    def _set_guess_amp(self, guess_amp=None):
        """Setting the guess amplitude used in AD reconstruction (no DIP?)"""
        if isinstance(guess_amp, (np.ndarray, torch.Tensor)):
            self.guess_amp = guess_amp
        else:
            im = self.dd.images[len(self.dd) // 2]
            if isinstance(guess_amp, (float, int)):
                thresh = guess_amp  # TODO update this to be percent saturated?
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
        """Setting the guess phase used in AD reconstruction (no DIP?)"""
        guess_phase = guess_phase.lower()
        if guess_phase == "none":
            guess_phase = None
            self._num_pretrain_iter = 0
            return
        elif guess_phase == "uniform":
            guess_phase = np.zeros(self.shape)
        elif guess_phase == "sitie":
            sitie = SITIE(self.dd, verbose=0)
            sitie.reconstruct(qc=self._qc)
            if self._verbose >= 2:
                print("SITIE guess phase:")
                sitie.visualize(cbar=True)
            guess_phase = sitie.phase_B
        self.guess_phase = torch.tensor(guess_phase, dtype=torch.float32)
        return

    def _set_input_DIP(self, input_mode: str = "SITIE", guess_phase=None):
        """
        Generate the input to the DIP. Currently only a SITIE is available as that is frankly the
        best option, but this could be easily expanded to using input noise (like for a traditional
        DIP) or really anything
        """
        # TODO test/have various options here, but SITIE is definitely best
        # inp_noise = self._rng.random((1, *self.shape)) * 2 - 1
        # inp_noise = self._rng.random((1, *self.shape)) * 2 - 1
        # self.input_DIP = torch.tensor(
        #     inp_noise, device=self.device, dtype=torch.float32, requires_grad=False
        # )
        if guess_phase is None:
            if input_mode.lower() == "sitie":
                sitie = SITIE(self.dd, verbose=0)
                sitie.reconstruct(qc=self._qc)
                guess_phase = sitie.phase_B
            elif input_mode.lower() in ["random", "rand"]:
                guess_phase = self._rng.random(self.shape) * 2 - 1
            else:
                raise ValueError(
                    f"Input mode should be 'SITIE' or 'random', unknown mode {input_mode}"
                )
        else:
            guess_phase = np.squeeze(guess_phase)

        self._input_DIP = torch.tensor(
            guess_phase[None, ...], device=self.device, dtype=torch.float32, requires_grad=False
        )

    def get_TFs(self):
        """
        Returns a tensor containing the transfer functions according to self.scope and self.defavls
        """
        tfs = np.array([self._get_TF(df) for df in self.defvals])
        return torch.tensor(tfs, device=self.device, dtype=torch.complex64)

    def _get_TF(self, defocus):
        """Returns a single transfer function for a given defocus value in nm"""
        self.scope.defocus = defocus
        return self.scope.get_transfer_function(self.scale, self.shape)

    def show_best(self, crop=5, **kwargs):
        minloss_iter = self._best_iter  # np.argmin(self.loss_iterations)
        ph = self.best_phase
        By, Bx = self.induction_from_phase(ph)
        if crop > 0:
            crop = int(crop)
            ph = ph[crop:-crop, crop:-crop]
            Bx = Bx[crop:-crop, crop:-crop]
            By = By[crop:-crop, crop:-crop]

        if self._solve_amp:
            fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
            show_im(
                ph,
                title=f"best phase: iter {minloss_iter} / {len(self.loss_iterations)}",
                figax=(fig, axs[0]),
                cbar_title="rad",
            )
            show_2D(
                Bx,
                By,
                title=f"Best B: iter {minloss_iter} / {len(self.loss_iterations)}",
                figax=(fig, axs[1]),
            )
            show_im(
                self.best_amp,
                title=f"best amp: iter {minloss_iter} / {len(self.loss_iterations)}",
                figax=(fig, axs[2]),
            )
            plt.tight_layout()
            plt.show()
        else:
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            show_im(
                ph,
                title=f"best phase: iter {minloss_iter} / {len(self.loss_iterations)}",
                scale=self.scale,
                cbar_title="rad",
                figax=(fig, axs[0]),
            )
            show_2D(
                Bx,
                By,
                title=f"Best B: iter {minloss_iter} / {len(self.loss_iterations)}",
                figax=(fig, axs[1]),
            )
            plt.tight_layout()
            plt.show()

    def show_final(self, crop: int = 5, **kwargs) -> None:
        """Show the phase and induction of the final iteration.

        Args:
            crop (int, optional): Amount to crop off of induction maps before displaying; often
            necessary in order to avoid edge artifacts. Defaults to 5.
        """
        ph = self.recon_phase
        By, Bx = self.induction_from_phase(ph)
        if crop > 0:
            crop = int(crop)
            ph = ph[crop:-crop, crop:-crop]
            Bx = Bx[crop:-crop, crop:-crop]
            By = By[crop:-crop, crop:-crop]

        if self._solve_amp:
            fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
            show_im(
                ph,
                title=f"Recon phase: iter {len(self.loss_iterations)}",
                figax=(fig, axs[0]),
                cbar_title="rad",
            )
            show_2D(
                Bx,
                By,
                title=f"Recon B: iter {len(self.loss_iterations)}",
                figax=(fig, axs[1]),
            )
            show_im(
                self.recon_amp,
                title=f"Recon amp: iter {len(self.loss_iterations)}",
                figax=(fig, axs[2]),
            )
            plt.tight_layout()
            plt.show()
        else:
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            show_im(
                ph,
                title=f"Recon phase: iter {len(self.loss_iterations)}",
                scale=self.scale,
                cbar_title="rad",
                figax=(fig, axs[0]),
            )
            show_2D(
                Bx,
                By,
                title=f"Recon B: iter {len(self.loss_iterations)}",
                figax=(fig, axs[1]),
            )
            plt.tight_layout()
            plt.show()

    def visualize(self, crop=5):
        """Plot the best reconstructed phase and induction maps.

        Args:
            crop (int, optional): Amount to crop off of induction maps before displaying; often
            necessary in order to avoid edge artifacts. Defaults to 5.
        """

        if self._solve_amp:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(231)
            self.show_phase_B(figax=(fig, ax1), cbar_title=None, crop=crop)
            ax2 = fig.add_subplot(232)
            self.show_B(figax=(fig, ax2), crop=crop)
            ax2p5 = fig.add_subplot(233)
            show_im(self.best_amp, figax=(fig, ax2p5), ticks_off=True, title="amp")
            ax3 = fig.add_subplot(212)
            l1 = ax3.semilogy(self.loss_iterations, color="tab:blue", label="loss")
            ax3.set_xlabel("iterations")
            ax3.set_ylabel("loss")
            ax4 = ax3.twinx()
            LRs = np.array(self.LR_iterations)
            l2 = ax4.semilogy(LRs[:, 0], color="tab:orange", label="phase LR")

            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            ax4.legend(lns, labs, loc=0)
            ax4.set_ylabel("LR")
            ax4.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
        else:
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(221)
            self.show_phase_B(figax=(fig, ax1), cbar_title=None, crop=crop)
            ax2 = fig.add_subplot(222)
            self.show_B(figax=(fig, ax2), crop=crop)
            ax3 = fig.add_subplot(212)
            l1 = ax3.semilogy(self.loss_iterations, color="tab:blue", label="loss")
            ax3.set_xlabel("iterations")
            ax3.set_ylabel("loss")
            ax4 = ax3.twinx()
            LRs = np.array(self.LR_iterations)
            l2 = ax4.semilogy(LRs[:, 0], color="tab:orange", label="phase LR")

            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            ax4.legend(lns, labs, loc=0)
            ax4.set_ylabel("LR")
            ax4.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))

            plt.show()

        return

    def __len__(self):
        return len(self.loss_iterations)

    def save_results(
        self,
        iter_ind: int = None,
        save_dir: Optional[os.PathLike] = None,
        name: Optional[str] = None,
        overwrite: bool = False,
    ):
        """Save the recontructed phase, Bx, By, and color images.
        # TODO add saving amplitude and phase_E


        Args:
            iter_ind (int, optional): Index to save. Defaults to None which saves best phase.
            save_dir (os.PathLik], optional): Directory to save in. Defaults to None
            which saves in self.save_dir.
            name (str, optional): Name to prepend saved files. Defaults to None.
            overwrite (bool, optional): Whether or not to overwrite files. Defaults to False.

        """
        if save_dir is not None or name is not None:
            # don't want to overwrite original name with timestamp
            self._check_save_name(save_dir, name=name, default_name=False)

        if iter_ind is not None:
            phase, iter = self.phase_iterations[iter_ind]
            By, Bx = self.induction_from_phase(phase)
        else:
            iter = self._best_iter
            phase = self.best_phase
            Bx = self.Bx
            By = self.By

        results = {
            "phase_B": phase,
            "By": By,
            "Bx": Bx,
            "color": None,
        }
        if self._mode == "SIPRAD":
            results["input_image"] = self.dd.image
        else:
            raise NotImplementedError("need to figure out how saving defocus vals")

        save_name_no_iter = "_".join(self._save_name.split("_")[:3])
        self._save_name = save_name_no_iter + f"_i{iter}"

        self.save_dir.mkdir(exist_ok=True)
        self._save_keys(list(results.keys()), self.defvals[0], overwrite, res_dict=results)
        self._save_log(overwrite)

    def _save_log(self, overwrite: Optional[bool] = None):
        """
        Save the reconstruction log.

        Args:
            overwrite (bool, optional): Whether to overwrite existing files. Default is None.
        """
        log_dict = {
            "name": self.name,
            "_save_name": self._save_name,
            "defval": self.defvals.squeeze(),
            "scale": self.scale,
            "transforms": self.dd.transforms,
            "filters": self.dd._filters,
            "beam_energy": self.dd.beam_energy,
            "simulated": self.dd._simulated,
            "data_dir": self.dd.data_dir,
            "data_files": self.dd.data_files,
            "save_dir": self._save_dir,
        }
        ovr = overwrite if overwrite is not None else self._overwrite
        name = f"{self._save_name}_{self._fmt_defocus(self.defvals[0])}_log.json"
        write_json(log_dict, self.save_dir / name, overwrite=ovr, v=self._verbose)
