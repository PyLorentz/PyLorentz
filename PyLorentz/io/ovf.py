import io
import os
import sys
import textwrap
from itertools import takewhile

import numpy as np


def read_ovf(file=None, mode="norm", B0=1e4, v=1):
    """Load a .ovf or .omf file of magnetization values.

    This function takes magnetization output files from OOMMF or Mumax, pulls
    some data from the header and returns 3D arrays for each magnetization
    component as well as the pixel resolutions.

    Args:
        file (string): Path to file
        mode (string): Define how the magnetization is scaled as it's read from
            the file. OOMMF writes .omf files with vectors in units of A/m,
            while Mumax writes .omf files with vectors normalized. This allows
            the reading to scale the vectors appropriately to gauss or simply
            make sure everything is normalized (as is needed for the phase
            calculation).

            - "norm": (default) Normalize all vectors (does not change (0,0,0) vectors)
            - "raw": Don't do anything with the values.
        v (int): Verbosity.

            - 0 : No output
            - 1 : Default output
            - 2 : Extended output, print full header.

    Returns:
        tuple: (mags, scale, zscale)

        - mags (`4D array`) -- shape [3, dimz, dimy, dimx], along first axis is stacked
                               (magz, magy, magx)
        - scale (float) -- Scale of datafile in y/x direction (nm/pixel)
        - zscale (float) -- Scale of datafile in z-direction (nm/pixel)
    """
    vprint = print if v >= 1 else lambda *a, **k: None

    with io.open(file, mode="r") as f:
        try:
            header = list(takewhile(lambda s: s[0] == "#", f))
        except UnicodeDecodeError:  # happens with binary files
            header = []
            with io.open(file, mode="rb") as f2:
                for line in f2:
                    if line.startswith("#".encode()):
                        header.append(line.decode())
                    else:
                        break
    if v >= 2:
        ext = os.path.splitext(file)[1]
        print(f"-----Start {ext} Header:-----")
        print("".join(header).strip())
        print(f"------End {ext} Header:------")

    dtype = None
    header_length = 0
    for line in header:
        header_length += len(line)
        if line.startswith("# xnodes"):
            xsize = int(line.split(":", 1)[1])
        if line.startswith("# ynodes"):
            ysize = int(line.split(":", 1)[1])
        if line.startswith("# znodes"):
            zsize = int(line.split(":", 1)[1])
        if line.startswith("# xstepsize"):
            xscale = float(line.split(":", 1)[1])
        if line.startswith("# ystepsize"):
            yscale = float(line.split(":", 1)[1])
        if line.startswith("# zstepsize"):
            zscale = float(line.split(":", 1)[1])
        if line.startswith("# Begin: Data Text"):
            vprint(f"Text file found: {file}")
            dtype = "text"
        if line.startswith("# Begin: Data Binary 4"):
            vprint(f"Binary 4 file found: {file}")
            dtype = "bin4"
        if line.startswith("# Begin: Data Binary 8"):
            vprint(f"Binary 8 file found: {file}")
            dtype = "bin8"

    if xsize is None or ysize is None or zsize is None:
        print(
            textwrap.dedent(
                f"""\
    Simulation dimensions are not given. \
    Expects keywords "xnodes", "ynodes, "znodes" for number of cells.
    Currently found size (x y z): ({xsize}, {ysize}, {zsize})"""
            )
        )
        sys.exit(1)
    else:
        vprint(f"Simulation size (z, y, x) : ({zsize}, {ysize}, {xsize})")

    if xscale is None or yscale is None or zscale is None:
        vprint(
            textwrap.dedent(
                f"""\
    Simulation scale not given. \
    Expects keywords "xstepsize", "ystepsize, "zstepsize" for scale (nm/pixel).
    Found scales (z, y, x): ({zscale}, {yscale}, {xscale})"""
            )
        )
        scale = np.max([i for i in [xscale, yscale, 0] if i is not None]) * 1e9
        if zscale is None:
            zscale = scale
        else:
            zscale *= 1e9
        vprint(
            f"Proceeding with {scale:.3g} nm/pixel for in-plane and \
            {zscale:.3g} nm/pixel for out-of-plane."
        )
    else:
        assert xscale == yscale
        scale = xscale * 1e9  # originally given in meters
        zscale *= 1e9
        if zscale != scale:
            vprint(f"Image (x-y) scale : {scale:.3g} nm/pixel.")
            vprint(f"Out-of-plane (z) scale : {zscale:.3g} nm/pixel.")
        else:
            vprint(f"Image scale : {scale:.3g} nm/pixel.")

    if dtype == "text":
        data = np.genfromtxt(file)  # takes care of comments automatically
    elif dtype == "bin4":
        # for binaries it has to give count or else will take comments at end as well
        data = np.fromfile(
            file, dtype="d", count=xsize * ysize * zsize * 3, offset=header_length + 4
        )
    elif dtype == "bin8":
        data = np.fromfile(
            file, dtype="d", count=xsize * ysize * zsize * 3, offset=header_length + 8
        )
    else:
        print("Unkown datatype given. Exiting.")
        sys.exit(1)

    data = data.reshape(
        (zsize, ysize, xsize, 3)
    )  # binary data not always shaped nicely

    if mode.lower() == "raw":
        vprint("Not scaling datafile.")
    elif mode.lower() == "norm":
        data = data.reshape((zsize * ysize * xsize, 3))  # to iterate through vectors
        row_sums = np.sqrt(np.sum(data**2, axis=1))
        rs2 = np.where(row_sums == 0, 1, row_sums)
        data = data / rs2[:, np.newaxis]
        data = data.reshape((zsize, ysize, xsize, 3))
    else:
        print(
            textwrap.dedent(
                """\
        Improper argument given for sim. Please set to one of the following options:
            'oommf' : vector values given in A/m, will be scaled by mu0
            'mumax' : vectors all of magnitude 1, will be scaled by B0
            'raw'   : vectors will not be scaled."""
            )
        )
        sys.exit(1)

    mag_x = data[:, :, :, 0]
    mag_y = data[:, :, :, 1]
    mag_z = data[:, :, :, 2]
    mags = np.array([mag_z, mag_y, mag_x])

    return (mags, scale, zscale)



def write_ovf(
    f: str,
    mags: np.ndarray,
    del_px: float,
    zscale: float,
    title: str = None,
    units="norm",
    overwrite = False,
):
    """Write an ovf file from a numpy array

    Args:
        f (str): File to write.
        mags (np.ndarray): Numpy vector arrays to write. Should have dimensions
            (3, dimz, dimy, dimx) with the vector components being (z, y, x) along the
            first axis.
        del_px (float): x-y direction scale in nm/pixel
        zscale (float): z-direction scale in nm/pixel
        title (str, optional): Title of file in .ovf header. Defaults to filename.
        units (str, optional): Units of vectors. If units=="norm", will normalize all
            vectors prior to saving. Otherwise will leave unscaled and fill "units"
            portion of the .ovf header with arg. Defaults to 'norm'.

    Returns:
        f (str): filepath that was written
    """
    f = str(f)
    if os.path.exists(f) and not overwrite:
        raise FileExistsError(f"File already exists at: {f}")
    if not f.endswith(".ovf"):
        f = f + ".ovf"
    print("saving: ", f)

    if units.lower() == "norm":
        # norm magnetizations
        tmags = np.sqrt(mags[0] ** 2 + mags[1] ** 2 + mags[2] ** 2)
        for i in range(3):
            mags[i] = mags[i] / tmags

    _3, dimz, dimy, dimx = mags.shape
    assert _3 == 3

    if title is None:
        title = os.path.splitext(os.path.split(f)[1])[0]

    del_px_nm = del_px / 1e9
    zscale_nm = zscale / 1e9

    header = textwrap.dedent(
        f"""
        # OOMMF OVF 2.0
        # Segment count: 1
        # Begin: Segment
        # Begin: Header
        # Title: {title}
        # meshtype: rectangular
        # meshunit: m
        # xmin: 0
        # ymin: 0
        # zmin: 0
        # xmax: {dimx*del_px_nm:.5e}
        # ymax: {dimy*del_px_nm:.5e}
        # zmax: {dimz*zscale_nm:.5e}
        # valuedim: 3
        # valuelabels: m_x_{units} m_y_{units} m_z_{units}
        # valueunits: {units} {units} {units}
        # Desc: Total simulation time:  -1  s
        # xbase: {del_px_nm/2:.5e}
        # ybase: {del_px_nm/2:.5e}
        # zbase: {zscale/2:.5e}
        # xnodes: {dimx:d}
        # ynodes: {dimy:d}
        # znodes: {dimz:d}
        # xstepsize: {del_px_nm:.5e}
        # ystepsize: {del_px_nm:.5e}
        # zstepsize: {zscale_nm:.5e}
        # End: Header
        # Begin: Data Text
        """
    ).strip()

    mags = mags[
        ::-1,
    ]  # zyx to xyz
    mags = np.rollaxis(mags, 0, 4)  # (3, dimz, dimy, dimx) -> (dimz, dimy, dimx, 3)
    data = mags.reshape(dimz * dimy * dimx, 3)

    footer = textwrap.dedent(
        """
        # End: Data Text
        # End: Segment

        """
    ).strip()

    if units.lower() == "norm":
        fmt = "%.11f"
    elif units.lower() in ["g", "gauss"]:
        fmt = "%.3f"
    np.savetxt(
        f, data.astype("float32"), header=header, footer=footer, comments="", fmt=fmt
    )
    return f

