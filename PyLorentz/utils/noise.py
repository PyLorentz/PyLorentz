import numpy as np
import skimage
from skimage import exposure
from skimage.restoration import estimate_sigma
from scipy import ndimage as ndi
from scipy import stats
from skimage.util import random_noise as skrandom_noise
import torch
from pathlib import Path
from typing import Optional, Union


class ImageNoiser:
    """
    Applies a sequence of image noising steps to an image(s).

    Attributes:
        gauss (float): Percent noise value, ~20.
        poisson (float): Noise level, ~0.3-0.5.
        salt_and_pepper (float): Salt and pepper noise amount, ~50.
        blur (float): Sigma value (pixels), ~5.
        jitter (float): Jitter level, ~0-3.
        contrast (float): Gamma level, [0,1] for brighter, >1 for darker, ~2.
        bkg (float): Background noise level.
        seed (Optional[int]): Seed for random number generation.
    """

    def __init__(
        self,
        poisson: float = 0,
        gauss: float = 0,
        salt_and_pepper: float = 0,
        blur: float = 0,
        jitter: float = 0,
        contrast: float = 1,
        bkg: float = 0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize image noising parameters.
        """
        self.poisson = poisson
        self.gauss = gauss
        self.salt_and_pepper = salt_and_pepper
        self.blur = blur
        self.jitter = jitter
        self.contrast = contrast
        self.bkg = bkg
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    def run(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the specified noising steps to the image.

        Args:
            image (Union[np.ndarray, torch.Tensor]): Input image.

        Returns:
            Union[np.ndarray, torch.Tensor]: Noised image.
        """
        if isinstance(image, torch.Tensor):
            inp_torch = True
            inp_shape = image.shape
            device = image.device
            image = image.cpu().detach().numpy()
        else:
            inp_torch = False
            inp_shape = image.shape

        image = image.squeeze()
        if np.ndim(image) != 2:
            raise ValueError(f"Expected array of dimension 2, got image of shape {image.shape}")
        self.h, self.w = image.shape

        if self.poisson != 0:
            image = self.apply_poisson(image)
        if self.gauss != 0:
            image = self.apply_gauss(image)
        if self.salt_and_pepper != 0:
            image = self.apply_salt_and_pepper(image)
        if self.blur != 0:
            image = self.apply_blur(image)
        if self.jitter != 0:
            image = self.apply_jitter(image)
        if self.contrast != 1 and self.contrast is not None:
            image = self.apply_contrast(image)
        if self.bkg != 0:
            image = self.apply_bkg(image)

        if inp_torch:
            image = torch.tensor(image, device=device)

        image = image.reshape(inp_shape)
        return image

    def apply_gauss(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian noise to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Noised image.
        """
        sigma = self.gauss / 200 * np.mean(image)
        noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
        return image + noise

    def apply_poisson(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Poisson noise to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Noised image.
        """
        offset = np.min(image)
        ptp = np.ptp(image)
        im = (image - offset) / ptp  # norm image
        noisy = np.random.poisson(im * self.poisson / im.size)
        return (noisy /self.poisson * im.size * ptp) + offset

    def apply_salt_and_pepper(self, image: np.ndarray) -> np.ndarray:
        """
        Apply salt and pepper noise to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Noised image.
        """
        minval = image.min()
        imfac = (image - minval).max()
        im_sp = skrandom_noise(
            (image - minval) / imfac,
            mode="s&p",
            amount=self.salt_and_pepper * 1e-3,
            rng=self.seed,
        )
        im_sp = (im_sp * imfac) + minval
        return im_sp

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Blurred image.
        """
        return ndi.gaussian_filter(image, self.blur, mode="wrap")

    def apply_jitter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply jitter to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Jittered image.
        """
        shift_arr = stats.poisson.rvs(self.jitter, loc=0, size=self.h)
        im_jitter = np.array([np.roll(row, z) for row, z in zip(image, shift_arr)])
        return im_jitter

    def apply_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust the contrast of the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with adjusted contrast.
        """
        if self.contrast == 0:
            self.contrast == None
        return exposure.adjust_gamma(image, self.contrast)

    def apply_bkg(self, image: np.ndarray) -> np.ndarray:
        """
        Apply background noise to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with background noise.
        """

        def gauss2d(xy, x0, y0, a, b, fwhm):
            return np.exp(-np.log(2) * (a * (xy[0] - x0) ** 2 + b * (xy[1] - y0) ** 2) / fwhm**2)

        h, w = image.shape
        x, y = np.meshgrid(np.linspace(0, h, h), np.linspace(0, w, w), indexing="ij")
        x0 = np.random.randint(0, h - h // 4)
        y0 = np.random.randint(0, w - w // 4)
        a, b = np.random.randint(10, 20, 2) / 10
        fwhm = np.random.randint(min([h, w]) // 4, min([h, w]) - min([h, w]) // 2)
        Z = gauss2d([x, y], x0, y0, a, b, fwhm)
        fac = 0.05 * (np.random.randint(0, 2) * 2 - 1)
        bkg = self.bkg * fac * np.random.randint(-10, 10) * Z
        return image + bkg

    def log(self, fpath: Union[str, Path]) -> Path:
        """
        Log the current noise parameters to a file.

        Args:
            fpath (Union[str, Path]): File path to save the log.

        Returns:
            Path: Path to the saved log file.
        """
        fpath = Path(fpath).resolve()
        if fpath.suffix not in [".txt", ".log"]:
            fpath = fpath.parent / (fpath.name + ".txt")
        with open(fpath, "w") as f:
            f.write(f"gauss: {self.gauss}\n")
            f.write(f"jitter: {self.jitter}\n")
            f.write(f"poisson: {self.poisson}\n")
            f.write(f"salt_and_pepper: {self.salt_and_pepper}\n")
            f.write(f"blur: {self.blur}\n")
            f.write(f"contrast: {self.contrast}\n")
            f.write(f"bkg: {self.bkg}\n")
            f.write(f"seed: {self.seed}\n")
        return fpath


def get_percent_noise(
    noisy: Union[np.ndarray, torch.Tensor], truth: Optional[np.ndarray] = None
) -> float:
    """
    Calculate the percent noise in the image.

    Args:
        noisy (Union[np.ndarray, torch.Tensor]): Noisy image.
        truth (Optional[np.ndarray]): Ground truth image.

    Returns:
        float: Percent noise in the image.
    """
    if isinstance(noisy, torch.Tensor):
        noisy = noisy.cpu().detach().numpy()
        if isinstance(truth, torch.Tensor):
            truth = truth.cpu().detach().numpy()

    if truth is None:
        if noisy.ndim == 3:
            return np.mean(
                estimate_sigma(noisy, channel_axis=0) / (np.mean(noisy, axis=(-2, -1))) * 200
            )
        elif noisy.ndim == 2:
            return estimate_sigma(noisy) / (np.mean(noisy)) * 200
        else:
            raise NotImplementedError(f"Unsupported dimension for noisy image: {noisy.shape}")
    else:
        assert noisy.shape == truth.shape
        return np.mean(np.std(noisy - truth, axis=(-2, -1)) / np.mean(truth, axis=(-2, -1)) * 200)
