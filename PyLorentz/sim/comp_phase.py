import numpy as np
import os
from .sim_base import SimBase
import time
import numba
from numba import jit
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np


class LinsupPhase(SimBase):
    def _calc_phase_linsup(self, multiproc=True, device="cpu", **kwargs):
        if device == "cpu":
            xp = np
        else:
            xp = cp
        # look at MagPy code and implement that version
        # for now just do cpu (with numba), then later adapt for gpu and benchmark

        KY, KX, SY, SX, zeros = self._linsup_compute_arrays(xp)
        inds = self._get_inds(xp)

        dim_z, dim_y, dim_x = self._mags_shape
        x = xp.arange(dim_x) - dim_x//2
        y = xp.arange(dim_y) - dim_y//2
        z = xp.arange(dim_z) - dim_z//2
        Z, Y, X = xp.meshgrid(z, y, x, indexing="ij")  # position grid (centered on 0)

        # Compute the rotation angles
        st = xp.sin(xp.deg2rad(self.tilt_x))
        ct = xp.cos(xp.deg2rad(self.tilt_x))
        sg = xp.sin(xp.deg2rad(self.tilt_y))
        cg = xp.cos(xp.deg2rad(self.tilt_y))

        # compute the rotated values;
        # here we apply rotation about X first, then about Y
        ## could improve indexing with ravel_multi_index, also removing need for tuple
        i_n = Z * sg * ct + Y * sg * st + X * cg
        j_n = Y * ct - Z * st

        Mx = xp.array(self.Mx)
        My = xp.array(self.My)
        Mz = xp.array(self.Mz)

        mx_n = Mx * cg + My * sg * st + Mz * sg * ct
        my_n = My * ct - Mz * st


        # print(f"Shapes: Ky: {KY.shape}, j_n: {j_n.shape}, {j_n[inds[0]].shape}, SX: {SX.shape}, my_n: {my_n.shape}, {my_n[inds[0]].shape}")
        # print("inds: ", inds.shape, inds[0], inds)
        # setup
        phase_B_k = xp.zeros(KY.shape, dtype=complex)
        phase_E_k = xp.zeros(KY.shape, dtype=complex)

        nelems = xp.shape(inds)[0]
        stime = time.time()
        self.vprint(f"Beginning linsup phase calculation for {nelems:g} voxels.")
        if device == "gpu":
            # cp.cuda.runtime.getDeviceCount()
            device = cp.cuda.Device()
            _free_mem, total_mem = device.mem_info
            _dimz, dimy, dimx = self.shape
            # batch_size = kwargs.get("batch_size", (total_mem * 2**7) // (dimy**2 * dimx**2))
            batch_size = kwargs.get("batch_size", (total_mem) // (dimy*dimx*dimy))
            self.vprint(f"Running on GPU with batch_size = {batch_size}")
            Ninds = np.prod(self.shape)
            nbatches = Ninds // batch_size

            KY = KY[None]
            KX = KX[None]
            SY = SY[None]
            SX = SX[None]
            j_n = j_n.reshape((Ninds, 1,1))
            i_n = i_n.reshape((Ninds, 1,1))
            my_n = my_n.reshape((Ninds, 1,1))
            mx_n = mx_n.reshape((Ninds, 1,1))
            batches = np.array_split(np.arange(Ninds), nbatches)
            for b in batches:
                sum_term = xp.exp(-1j * (KY * j_n[b] + KX * i_n[b]))
                phase_E_k += sum_term.sum(axis=0)
                phase_B_k += (sum_term * (my_n[b] * SX - mx_n[b] * SY)).sum(axis=0)

            cp.cuda.Stream.null.synchronize()

        elif multiproc:
            self.vprint("Running in parallel with numba.")
            phase_E_k, phase_B_k = _exp_sum(
                phase_B_k, phase_E_k, inds, KY, KX, j_n, i_n, my_n, mx_n, SY, SX
            )

        else:
            self.vprint("Running on cpu without multiprocessing.")
            # doing larger batches on cpu is actually slower
            for ind in tqdm(inds, disable=self._verbose<=0):
                ind = tuple(ind)
                sum_term = np.exp(-1j * (KY * j_n[ind] + KX * i_n[ind]))
                phase_E_k += sum_term
                phase_B_k += sum_term * (my_n[ind] * SX - mx_n[ind] * SY)


        self.vprint(
            f"total time: {time.time()-stime:.5g} sec, {(time.time()-stime)/nelems:.5g} sec/voxel."
        )
        # Now we have the phases in K-space. We convert to real space and return
        phase_E_k[zeros] = 0.0
        phase_B_k[zeros] = 0.0
        phase_E = (np.fft.ifftshift(np.fft.ifft2(phase_E_k))).real * self._pre_E()
        phase_B = (np.fft.ifftshift(np.fft.ifft2(phase_B_k))).real * self._pre_B()

        if device != "cpu":
            phase_B = phase_B.get()
            phase_E = phase_E.get()
        return phase_B, phase_E


    def _linsup_compute_arrays(self, xp=np):
        dim_z, dim_y, dim_x = self._mags_shape
        ky = xp.fft.fftfreq(dim_y) * 2 * xp.pi
        kx = xp.fft.fftfreq(dim_x) * 2 * xp.pi
        KX, KY = xp.meshgrid(kx, ky)
        KK = xp.sqrt(KX**2 + KY**2)
        zeros = xp.where(KK==0)
        KK[zeros] = 1.

        SX = 1j * KX / KK**2
        SY = 1j * KY / KK**2
        SX[zeros] = 0.0
        SY[zeros] = 0.0

        return KY, KX, SY, SX, zeros

    def _get_inds(self, xp=np):
        inds = xp.where(xp.array(self._shape_func) != 0)
        inds = xp.array(inds).T
        return inds

@jit(nopython=True, parallel=True)
def _exp_sum(phase_B_k, ephi_k, inds, KY, KX, j_n, i_n, my_n, mx_n, Sy, Sx):
    """Called by linsupPhi when running with multiprocessing and numba.

    Numba incorporates just-in-time (jit) compiling and multiprocessing to numpy
    array calculations, greatly speeding up the phase-shift computation beyond
    that of pure vectorization and without the memory cost. Running this
    for the first time each session will take an additional 5-10 seconds as it is
    compiled.

    This function could be further improved by sending it to the GPU, or likely
    by other methods we haven't considered. If you have suggestions (or better
    yet, written and tested code) please email amccray@anl.gov.
    """
    for i in numba.prange(np.shape(inds)[0]):
        z = int(inds[i, 0])
        y = int(inds[i, 1])
        x = int(inds[i, 2])
        sum_term = np.exp(-1j * (KY * j_n[z, y, x] + KX * i_n[z, y, x]))
        ephi_k += sum_term
        phase_B_k += sum_term * (my_n[z, y, x] * Sx - mx_n[z, y, x] * Sy)
    return ephi_k, phase_B_k




class MansuripurPhase(SimBase):
    def _calc_phase_mansuripur(self, padded_shape: tuple | list | None = None, pad_mode="edge"):
        """

        pad shape for fourier space
        """

        # TODO figure out -1 stuff
        beam = self._rotate_vector(Tx=-1*self._tilt_x, Ty=-1*self.tilt_y, v=[0, 0, 1])
        beam = beam / np.sqrt(np.sum(beam**2))

        _, dimy, dimx = self._mags_shape
        if padded_shape is not None:
            pdimy, pdimx = padded_shape
            if pdimy < dimy:
                raise ValueError(
                    f"Padded dimy, {pdimy}, must be > magnetization dimy, {dimy}"
                )
            elif pdimx < dimx:
                raise ValueError(
                    f"Padded dimx, {pdimx}, must be > magnetization dimx, {dimx}"
                )
            py = (pdimy - dimy) // 2
            px = (pdimx - dimx) // 2
            MY = np.pad(self.My.copy().sum(axis=0), ((py, py), (px, px)), mode=pad_mode)
            MX = np.pad(self.Mx.copy().sum(axis=0), ((py, py), (px, px)), mode=pad_mode)

        else:
            pdimy, pdimx = dimy, dimx
            MZ, MY, MX = self.mags.copy().sum(axis=1)  # make into (3, dimy, dimx)

        sY, sX, KK, zeros = self._mans_compute_arrays((pdimy, pdimx))

        # compute FFTs of the B arrays.
        fMY = np.fft.fft2(MY)
        fMX = np.fft.fft2(MX)

        # if False:
        if np.allclose(beam, [0, 0, 1]):  # simpler/faster 2D calculation, MZ not needed
            prod = sX * fMY - sY * fMX
            Gpts = 1 + 1j * 0
        else:
            if padded_shape is not None:
                MZ = np.pad(self.Mz.copy().sum(axis=0), ((py, py), (px, px)), mode=pad_mode)
            fMZ = np.fft.fft2(MZ)
            e_x, e_y, e_z = beam
            prod = sX * (
                fMY * e_x**2 - fMX * e_x * e_y - fMZ * e_y * e_z + fMY * e_z**2
            ) + sY * (
                fMY * e_x * e_y - fMX * e_y**2 + fMZ * e_x * e_z - fMX * e_z**2
            )
            arg = np.pi * (sX * e_x + sY * e_y) / e_z
            # TODO check thickness scaling is correct
            # arg = np.pi * thick * (sX * e_x + sY * e_y) / e_z # orig has thick
            denom = 1.0 / ((sX * e_x + sY * e_y) ** 2 + e_z**2)
            zeros2 = np.where(arg == 0)
            arg[zeros2] = 1
            Gpts = (denom * np.sin(arg) / arg).astype(complex)
            Gpts[zeros2] = denom[zeros2]

        # prefactor
        prefac = 1j / KK  # TODO thickness scaling
        # prefac = 1j * thickness_pixels / KK
        # F-space phase
        fphi = prefac * Gpts * prod
        fphi[zeros] = 0.0
        phase_B = np.fft.ifft2(fphi).real * self._pre_B()

        if padded_shape is not None:
            phase_B = phase_B[py:-py, px:-px]

        self.get_flat_shape_func()
        phase_E =  self.flat_shape_func * self._pre_E()
        phase_E -= phase_E.mean()

        return phase_B, phase_E

    def _mans_compute_arrays(self, shape):
        dimy, dimx = shape
        ky = np.fft.fftfreq(dimy) * 2 * np.pi
        kx = np.fft.fftfreq(dimx) * 2 * np.pi

        KX, KY = np.meshgrid(kx, ky)
        KK = np.sqrt(KX**2 + KY**2)

        zeros = np.where(KK == 0)
        KK[zeros] = 1.0

        sX = KX / KK
        sY = KY / KK
        sX[zeros] = 0.0
        sY[zeros] = 0.0

        return sY, sX, KK, zeros

    @staticmethod
    def _rotate_vector(Tx=0, Ty=0, Tz=0, v=[0, 0, 1]):
        """rotates the input vector around x, y, then z.
        assumes input vector along z axis [0,0,1]
        input in degrees, rotations around x then y then z axes.
        vector is [x,y,z]"""
        v = np.array(v)
        vx = np.array([1, 0, 0])
        vy = np.array([0, 1, 0])
        vz = np.array([0, 0, 1])

        rx = R.from_rotvec(Tx * vx, degrees=True)
        ry = R.from_rotvec(Ty * vy, degrees=True)
        rz = R.from_rotvec(Tz * vz, degrees=True)
        rtot = rz * ry * rx  # apply x then y then z
        vout = rtot.apply(v)
        return vout

