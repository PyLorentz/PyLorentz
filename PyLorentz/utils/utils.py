import numpy as np


def dist(ny, nx, shift=False):
    """Creates a frequency array for Fourier processing.

    Args:
        ny (int): Height of array
        nx (int): Width of array
        shift (bool): Whether to center the frequency spectrum.

            - False: (default) smallest values are at the corners.
            - True: smallest values at center of array.

    Returns:
        ``ndarray``: Numpy array of shape (ny, nx).
    """
    ly = (np.arange(ny) - ny / 2) / ny
    lx = (np.arange(nx) - nx / 2) / nx
    [X, Y] = np.meshgrid(lx, ly)
    q = np.sqrt(X**2 + Y**2)
    if not shift:
        q = np.fft.ifftshift(q)
    return q

def get_dist(pos1, pos2):
    """Distance between two 2D points

    Args:
        pos1 (list): [y1, x1] point 1
        pos2 (list): [y2, x2] point 2

    Returns:
        float: Euclidean distance between the two points
    """
    squared = (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
    return np.sqrt(squared)

def dist4(dim, norm=False):
    # 4-fold symmetric distance map even at small radiuses
    d2 = dim // 2
    a = np.arange(d2)
    b = np.arange(d2)
    if norm:
        a = a / (2 * d2)
        b = b / (2 * d2)
    x, y = np.meshgrid(a, b)
    quarter = np.sqrt(x**2 + y**2)
    dist = np.zeros((dim, dim))
    dist[d2:, d2:] = quarter
    dist[d2:, :d2] = np.fliplr(quarter)
    dist[:d2, d2:] = np.flipud(quarter)
    dist[:d2, :d2] = np.flipud(np.fliplr(quarter))
    return dist


def circ4(dim, rad):
    # 4-fold symmetric circle even at small dimensions
    return (dist4(dim) < rad).astype("int")


def norm_image(image):
    """Normalize image intensities to between 0 and 1"""
    if image.max() == image.min():
        image = image - np.max(image)
    else:
        image = image - np.min(image)
        image = image / np.max(image)
    return image

