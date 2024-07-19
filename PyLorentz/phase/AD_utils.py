from pathlib import Path

import os

import numpy as np
import tifffile
import torch
from skimage.restoration import estimate_sigma
from scipy.signal.windows import tukey
from skimage.metrics import structural_similarity
import scipy
from scipy.spatial.transform import Rotation as R



def sim_image(tphi, pscope, defocus, scale, Amp=None):
    """
    simulate image numpy
    """

    if Amp is None:
        Amp = np.ones_like(tphi)

    ObjWave = Amp * np.exp(1j * tphi)
    (dim, dim) = tphi.shape
    d2 = dim / 2
    line = np.arange(dim) - float(d2)
    [X, Y] = np.meshgrid(line, line)
    qq = np.sqrt(X**2 + Y**2) / float(dim)
    pscope.defocus = defocus
    im = pscope.getImage(ObjWave, qq, scale)
    return im


def get_TF(pscope, shape, defocus, scale):
    pscope.defocus = defocus
    return pscope.get_transfer_function(scale, shape)


def induction_from_phase(phi, scale):
    """Gives integrated induction in T*nm from a magnetic phase shift

    Args:
        phi (ndarray): 2D numpy array of size (dimy, dimx), magnetic component of the
            phase shift in radians
        scale (float): in-plane scale of the image in nm/pixel

    Returns:
        tuple: (By, Bx) where each By, Bx is a 2D numpy array of size (dimy, dimx)
            corresponding to the y/x component of the magnetic induction integrated
            along the z-direction. Has units of T*nm (assuming scale given in units
            of nm/pixel)
    """
    grad_y, grad_x = np.gradient(phi.squeeze(), edge_order=2)
    pre_B = scipy.constants.hbar / (scipy.constants.e * scale) * 10**18  # T*nm^2
    Bx = pre_B * grad_y
    By = -1 * pre_B * grad_x
    return (By, Bx)

def get_SS(guess_phi, true_phi, scale_invar=False):
    """Get accuracy of phase reconstruction using structural similarity

    Args:
        guess_phi (ndarray): Guess phase shift
        true_phi (ndarray): Ground truth phase shift
        scale_invar (bool): Whether or not to ignore global scaling differences.
            Default False.

    Returns:
        float: Accuracy [-1,1]. 1 corresponds to perfect correlation, 0 to no correlation
            and -1 to anticorrelation.
    """
    if isinstance(guess_phi, torch.Tensor):
        guess_phi = guess_phi.cpu().detach().numpy()
    if isinstance(true_phi, torch.Tensor):
        true_phi = true_phi.cpu().detach().numpy()
    guess_phi = np.copy(guess_phi).astype("double")
    true_phi = np.copy(true_phi).astype("double")
    guess_phi -= np.min(guess_phi)
    true_phi -= np.min(true_phi)
    if scale_invar:
        if np.max(guess_phi) != 0:
            guess_phi /= np.max(guess_phi)
        if np.max(true_phi) != 0:
            true_phi /= np.max(true_phi)
    else:
        # scale = np.max(true_phi)
        # if scale != 0:
        #     guess_phi /= scale
        #     true_phi /= scale
        pass
    data_range = np.ptp(true_phi)
    return structural_similarity(guess_phi, true_phi, data_range=data_range)


def get_acc(guess_phi, true_phi):
    """Get accuracy of phase reconstruction, does not account for scaling differences
    as long as values are centered around 0

    Args:
        guess_phi (ndarray): Guess phase shift
        true_phi (ndarray): Ground truth phase shift

    Returns:
        float: Accuracy [-1,1]. 1 corresponds to perfect correlation, 0 to no correlation
            and -1 to anticorrelation.
    """
    if isinstance(guess_phi, torch.Tensor):
        guess_phi = guess_phi.cpu().detach().numpy()
    if isinstance(true_phi, torch.Tensor):
        true_phi = true_phi.cpu().detach().numpy()
    guess_phi = guess_phi.astype("double")
    true_phi = true_phi.astype("double")
    acc = (guess_phi * true_phi).sum() / np.sqrt(
        (guess_phi * guess_phi).sum() * (true_phi * true_phi).sum()
    )
    return acc

def get_all_accs(guess_phi, best_phi=None, true_phi=None, pdict=None, guess_amp=None, best_amp=None, true_amp=None):
    """
    returns: ((guess SS, guess SS norm), (best SS, best SS norm)), (guess acc, best acc)
    """
    if isinstance(guess_phi, torch.Tensor):
        guess_phi = guess_phi.cpu().detach().numpy()
    else:
        guess_phi = np.copy(guess_phi)
    if isinstance(best_phi, torch.Tensor):
        best_phi = best_phi.cpu().detach().numpy()
    elif best_phi is not None:
        best_phi = np.copy(best_phi)
    if isinstance(true_phi, torch.Tensor):
        true_phi = true_phi.cpu().detach().numpy()
    elif true_phi is not None:
        true_phi = np.copy(true_phi)
    if isinstance(guess_amp, torch.Tensor):
        guess_amp = guess_amp.cpu().detach().numpy()
    elif guess_amp is not None:
        guess_amp = np.copy(guess_amp)
    if isinstance(best_amp, torch.Tensor):
        best_amp = best_amp.cpu().detach().numpy()
    elif best_amp is not None:
        best_amp = np.copy(best_amp)
    if isinstance(true_amp, torch.Tensor):
        true_amp = true_amp.cpu().detach().numpy()
    elif true_amp is not None:
        true_amp = np.copy(true_amp)

    guess_phi -= guess_phi.mean()
    if best_phi is not None:
        best_phi -= best_phi.mean()
    if true_phi is not None:
        true_phi -= true_phi.mean()

    # if guess_amp is not None:
    #     guess_amp -= guess_amp.min()
    # if best_amp is not None:
    #     best_amp -= best_amp.min()
    # if true_amp is not None:
    #     true_amp -= true_amp.min()

    guess_By, guess_Bx = induction_from_phase(guess_phi, pdict['scale'])
    true_By, true_Bx = induction_from_phase(true_phi, pdict['scale'])

    guess_Bmag = np.sqrt(guess_Bx**2 + guess_By**2)
    true_Bmag = np.sqrt(true_Bx**2 + true_By**2)

    pdict['guess_phi_SS'] = get_SS(guess_phi, true_phi)
    pdict['guess_phi_SS_norm'] = get_SS(guess_phi, true_phi, scale_invar=True)
    pdict['guess_phi_acc'] = get_acc(guess_phi, true_phi)
    pdict['guess_Bx_SS'] = get_SS(guess_Bx, true_Bx)
    pdict['guess_Bx_SS_norm'] = get_SS(guess_Bx, true_Bx, scale_invar=True)
    pdict['guess_Bx_acc'] = get_acc(guess_Bx, true_Bx)
    pdict['guess_By_SS'] = get_SS(guess_By, true_By)
    pdict['guess_By_SS_norm'] = get_SS(guess_By, true_By, scale_invar=True)
    pdict['guess_By_acc'] = get_acc(guess_By, true_By)

    pdict['guess_Bave_SS'] = (pdict["guess_Bx_SS"] + pdict["guess_By_SS"]) / 2
    pdict['guess_Bave_SS_norm'] = (pdict["guess_Bx_SS_norm"] + pdict["guess_By_SS_norm"]) / 2
    pdict['guess_Bave_acc'] = (pdict["guess_Bx_acc"] + pdict["guess_By_acc"]) / 2

    pdict['guess_Bmag_SS'] = get_SS(guess_Bmag, true_Bmag)
    pdict['guess_Bmag_SS_norm'] = get_SS(guess_Bmag, true_Bmag, scale_invar=True)
    pdict['guess_Bmag_acc'] = get_acc(guess_Bmag, true_Bmag)


    if best_phi is not None:
        best_By, best_Bx = induction_from_phase(best_phi, pdict['scale'])
        best_Bmag = np.sqrt(best_Bx**2 + best_By**2)
        pdict['best_phi_SS'] = get_SS(best_phi, true_phi)
        pdict['best_phi_SS_norm'] = get_SS(best_phi, true_phi, scale_invar=True)
        pdict['best_phi_acc'] = get_acc(best_phi, true_phi)
        pdict['best_Bx_SS'] = get_SS(best_Bx, true_Bx)
        pdict['best_Bx_SS_norm'] = get_SS(best_Bx, true_Bx, scale_invar=True)
        pdict['best_Bx_acc'] = get_acc(best_Bx, true_Bx)
        pdict['best_By_SS'] = get_SS(best_By, true_By)
        pdict['best_By_SS_norm'] = get_SS(best_By, true_By, scale_invar=True)
        pdict['best_By_acc'] = get_acc(best_By, true_By)
        pdict['best_Bmag_SS'] = get_SS(best_Bmag, true_Bmag)
        pdict['best_Bmag_SS_norm'] = get_SS(best_Bmag, true_Bmag, scale_invar=True)
        pdict['best_Bmag_acc'] = get_acc(best_Bmag, true_Bmag)

        pdict['best_Bave_SS'] = (pdict["best_Bx_SS"] + pdict["best_By_SS"]) / 2
        pdict['best_Bave_SS_norm'] = (pdict["best_Bx_SS_norm"] + pdict["best_By_SS_norm"]) / 2
        pdict['best_Bave_acc'] = (pdict["best_Bx_acc"] + pdict["best_By_acc"]) / 2

    if pdict['solve_amp'] and guess_amp is not None:
        pdict['guess_amp_SS'] = get_SS(guess_amp, true_amp)
        pdict['guess_amp_SS_norm'] = get_SS(guess_amp, true_amp, scale_invar=True)
        pdict['guess_amp_acc'] = get_acc(guess_amp, true_amp)
        if best_amp is not None:
            pdict['best_amp_SS'] = get_SS(best_amp, true_amp)
            pdict['best_amp_SS_norm'] = get_SS(best_amp, true_amp, scale_invar=True)
            pdict['best_amp_acc'] = get_acc(best_amp, true_amp)
    return



def Tukey2D(shape, alpha=0.5, sym=True):
    """
    makes a 2D (rectangular not round) window based on a Tukey signal
    Useful for windowing images before taking FFTs
    """
    dimy, dimx = shape
    ty = tukey(dimy, alpha=alpha, sym=sym)
    filt_y = np.tile(ty.reshape(dimy, 1), (1, dimx))
    tx = tukey(dimx, alpha=alpha, sym=sym)
    filt_x = np.tile(tx, (dimy, 1))
    output = filt_x * filt_y
    return output