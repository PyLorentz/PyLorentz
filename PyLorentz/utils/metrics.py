from typing import Optional, Union

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def ssim(truth:np.ndarray, test:np.ndarray, **kwargs):
    """Get structural similarity index measure (SSIM) of two images.

    Args:
        truth (ndarray): ground truth
        test (ndarray): test image

    Returns:
        float: Accuracy [-1,1]. 1 corresponds to perfect correlation, 0 to no correlation
            and -1 to anticorrelation.
    """
    test = np.copy(test).astype(np.float64)
    truth = np.copy(truth).astype(np.float64)
    test -= np.min(test) # min gives highest val
    truth -= np.min(truth)
    data_range = kwargs.pop("data_range", np.ptp(truth))
    return structural_similarity(truth, test, data_range=data_range, **kwargs)


def accuracy(truth:np.ndarray, test:np.ndarray, **kwargs):
    """
    Gets correlational accuracy of two images; does not account for scaling
    differences as long as values are centered around 0

    Args:
        truth (ndarray): ground truth
        test (ndarray): test image

    Returns:
        float: Accuracy [-1,1]. 1 corresponds to perfect correlation, 0 to no correlation
            and -1 to anticorrelation.
    """
    test = test.astype(np.float64)
    truth = truth.astype(np.float64)
    test -= test.mean() # min gives highest val
    truth -= truth.mean()
    acc = (test * truth).sum() / np.sqrt(
        (test * test).sum() * (truth * truth).sum()
    )
    return acc


def psnr(truth:np.ndarray, test:np.ndarray, data_range:float|None=None):
    """
    Gets peak signal to noise of two images.

    Args:
        truth (ndarray): ground truth
        test (ndarray): test image

    Returns:
        float: PSNR
    """
    test = np.copy(test).astype(np.float64)
    truth = np.copy(truth).astype(np.float64)
    test -= np.mean(test) # mean gives best val
    truth -= np.mean(truth)
    if data_range is None:
        data_range = np.ptp(truth)
    return peak_signal_noise_ratio(truth, test, data_range=data_range)

