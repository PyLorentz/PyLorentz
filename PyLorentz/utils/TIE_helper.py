"""Helper functions for TIE.

An assortment of helper functions that load images, pass data, and generally
are used in the reconstruction. Additionally, a couple of functions used for
displaying images and stacks.

Author: Arthur McCray, ANL, Summer 2019.
"""


from copy import deepcopy
import numpy as np


# ============================================================= #
#      Functions used for loading and passing the TIE data      #
# ============================================================= #



def select_tifs(i, ptie, long_deriv=False):
    """Returns a list of the images which will be used in TIE() or SITIE().

    Uses copy.deepcopy() as the data will be modified in the reconstruction
    process, and we don't want to change the original data. This method is
    likely not best practice. In the future this might get moved to the
    TIE_params class.

    Args:
        i (int): Index of defvals for which to select the tifs.
        ptie (``TIE_params`` object): Parameters for reconstruction, holds the
            images.

    Returns:
        list: List of np arrays, return depends on parameters:

        - if long_deriv == False:

            - if ptie.flip == True: returns [ +- , -- , 0 , ++ , -+ ]
            - elif ptie.flip == False:  returns [+-, 0, ++]
            - where first +/- is unflip/flip, second +/- is over/underfocus.
              E.g. -+ is the flipped overfocused image. 0 is the averaged
              infocus image.

        - elif long_deriv == True: returns all images in imstack followed by
          all images in flipstack.

    """
    if long_deriv:
        if ptie.flip: # list of numpy arrays is expected.
            # TIE_reconstruct expects list of numpy arrays, TODO update all of PyLorentz
            # to expect only numpy arrays and no lists. Primarily lists used because
            # allows changing size/cropping each numpy array in place in list
            recon_tifs = [i for i in ptie.imstack] + [j for j in ptie.flipstack]
        else:
            recon_tifs = [i for i in ptie.imstack]

    else:
        if i < 0:
            i = len(ptie.defvals) + i
        num_files = ptie.num_files
        under = num_files // 2 - (i + 1)
        over = num_files // 2 + (i + 1)
        imstack = ptie.imstack
        flipstack = ptie.flipstack
        if ptie.flip:
            recon_tifs = [
                imstack[under],  # +-
                flipstack[under],  # --
                (imstack[num_files // 2] + flipstack[num_files // 2]) / 2,  # infocus
                imstack[over],  # ++
                flipstack[over],  # -+
            ]
        else:
            recon_tifs = [
                imstack[under],  # +-
                imstack[num_files // 2],  # 0
                imstack[over],  # ++
            ]
    try:
        recon_tifs = deepcopy(recon_tifs)
    except TypeError:
        print("TypeError in select_tifs deepcopy. Proceeding with originals.")
    return recon_tifs


def scale_stack(imstack):
    """Scale a stack of images so all have the same total intensity.

    Args:
        imstack (list): List of 2D arrays.

    Returns:
        list: List of same shape as imstack
    """
    imstack = deepcopy(imstack)
    tots = np.sum(imstack, axis=(1, 2))
    t = max(tots) / tots
    for i in range(len(tots)):
        imstack[i] *= t[i]
    return imstack / np.max(imstack)

