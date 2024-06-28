import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from PyLorentz.visualize.show import show_im, show_im_peaks


def filter_hotpix(
    image: np.ndarray,
    thresh: float = 30,
    show: bool = False,
    maxiters: int = 3,
    kernel_size: int = 3,
    fast: bool=False,
    _current_iter: int = 0,
    verbose=False,
) -> np.ndarray:
    """
    look for pixel values with an intensity >thresh std outside of mean of surrounding
    pixels. If found, replace with median value of those pixels
    """
    if _current_iter > maxiters:
        if verbose:
            print(f"Ended at {maxiters} iterations of filter_hotpix.")
        return image

    if int(kernel_size) % 2 != 1:
        kernel_size = int(kernel_size) + 1

    kernel = np.ones((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 0
    kernel = kernel / np.sum(kernel)

    image = image.astype(np.float64)

    if fast:
        mean = ndi.convolve(image, kernel, mode="reflect")
        dif = np.abs(image - mean)
        std = np.std(dif) # global
    else:
        dimy, dimx = image.shape
        inds = np.mgrid[0:dimy, 0:dimx].reshape(2, dimy*dimx)
        patches1 = extract_patches(image, inds, patch_size=kernel_size)
        patches1 = patches1 * kernel[None]
        mean = np.mean(patches1, axis=(1,2)).reshape((dimy, dimx))
        dif = np.abs(image - mean)
        std = np.std(patches1, axis=(1,2)).reshape((dimy, dimx))

    bads = np.where(dif > thresh * std)
    bads2 = np.where(dif > thresh * std, 1, 0)
    numbads = len(bads[0])

    filtered = np.copy(image)

    if numbads > 0:
        ratio = numbads / np.size(image)
        if not fast and ratio < 5e-4:
            ks2 = kernel_size
            # ks2 = kernel_size+2
            # ks2 = kernel_size * 2+1
            ks2_kernel = np.ones((ks2, ks2))
            ks2_kernel[ks2 // 2, ks2 // 2] = 0

            # get mean of area around each bad pixel, not including other bad pixels in the mean
            masked = np.ma.array(image, mask=bads2)
            patches2 = extract_patches(masked, bads, patch_size=ks2)
            means = patches2.mean(axis=(1,2)).data

            bad_means = np.all(patches2.mask, axis=(1,2)) # all masked -> true
            if np.any(bad_means):
                # for those values, use the median of surounding pixels
                means[bad_means] = np.median(patches2.data[bad_means]*ks2_kernel[None,...], axis=(1,2))
            filtered[bads] = means

        elif ratio < 5e-4:
            filtered[bads] = mean[bads]
        else:
            print(f"Bad thresh chosen in filter_hotpix, increaseing to {thresh*2}")
            thresh *= 2

        filtered = filter_hotpix(
            filtered,
            thresh=thresh,
            show=False,
            maxiters=maxiters,
            kernel_size=kernel_size,
            fast=fast,
            _current_iter=_current_iter + 1,
        )

    if show:
        show_im_peaks(
            image,
            np.transpose([bads[0], bads[1]]),
            title=f"{numbads} hotpix identified on pass {_current_iter}",
            cbar=True,
        )
        show_im_peaks(
            filtered,
            np.transpose([bads[0], bads[1]]),
            title="hotpix filtered image",
            cbar=True,
            color1="b",
        )

    return filtered

def extract_patches(array, indices, patch_size=3):
    if patch_size %2 == 0:
        patch_size += 1
    ys, xs = np.array(indices)
    patch2 = patch_size // 2

    y_offsets = np.arange(-patch2, patch2 + 1)
    x_offsets = np.arange(-patch2, patch2 + 1)
    y_grid, x_grid = np.meshgrid(y_offsets, x_offsets, indexing='ij')

    y_indices = ys[:, None, None] + y_grid
    x_indices = xs[:, None, None] + x_grid

    y_indices = np.clip(y_indices, 0, array.shape[0]-1)
    x_indices = np.clip(x_indices, 0, array.shape[1]-1)

    patches = array[y_indices, x_indices]

    return patches
