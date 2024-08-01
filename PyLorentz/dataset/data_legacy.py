import os
from pathlib import Path
import numpy as np
from typing import Tuple

from PyLorentz.io.read import read_image


def legacy_load(
    data_loc: os.PathLike,
    fls_filename: os.PathLike,
) -> Tuple[float, np.ndarray]:
    """
    Load legacy data from specified file locations.

    Args:
        data_loc (os.PathLike): The directory containing the data files.
        fls_filename (os.PathLike): The filename of the .fls file.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing the scale factor and defocus values.

    Raises:
        FileNotFoundError: If the specified .fls file or infocus image file cannot be found.
    """
    data_loc = Path(data_loc)
    fls_filename = str(fls_filename)
    if not fls_filename.endswith(".fls"):
        fls_filename += ".fls"
    if (data_loc / fls_filename).exists():
        fls_full = data_loc / fls_filename
    elif (data_loc / ("unflip/" + fls_filename)).exists():
        fls_full = data_loc / ("unflip/" + fls_filename)
    elif (data_loc / ("tfs/" + fls_filename)).exists():
        fls_full = data_loc / ("tfs/" + fls_filename)
    else:
        raise FileNotFoundError(f"fls file could not be found: {fls_filename}")

    # Read scale from infocus unflip file, previously compared with flip/unflip
    fls = []
    with open(fls_full) as file:
        for line in file:
            fls.append(line.strip())

    num_files = int(fls[0])
    infocus_file = str(fls[1 + num_files // 2])
    if (data_loc / ("tfs/" + infocus_file)).exists():
        infocus_file = data_loc / ("tfs/" + infocus_file)
    else:
        infocus_file = data_loc / ("unflip/" + infocus_file)

    _, mdata = read_image(infocus_file)
    scale = mdata["scale"]

    # Read defocus values
    defvals = fls[-(num_files // 2):]
    assert num_files == 2 * len(defvals) + 1
    defvals = np.array([float(i) for i in defvals])  # defocus values +/-
    defvals = np.concatenate([-1 * defvals[::-1], [0], defvals])

    return scale, defvals
