import numpy as np
import os
from .sim_base import SimBase
import time
import numba
from numba import jit
from tqdm import tqdm


class LinsupPhase(SimBase):
    def _calc_phase_linsup(self, multiproc=True, device="cpu"):
        # look at MagPy code and implement that version
        # for now just do cpu (with numba), then later adapt for gpu and benchmark

        KY, KX, SY, SX, zeros = self._compute_arrays()
        inds = self._get_inds()

        dim_z, dim_y, dim_x = self._mags_shape
        x = np.arange(dim_x) - dim_x//2
        y = np.arange(dim_y) - dim_y//2
        z = np.arange(dim_z) - dim_z//2
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")  # position grid (centered on 0)

        # Compute the rotation angles
        st = np.sin(np.deg2rad(self.tilt_x))
        ct = np.cos(np.deg2rad(self.tilt_x))
        sg = np.sin(np.deg2rad(self.tilt_y))
        cg = np.cos(np.deg2rad(self.tilt_y))

        # compute the rotated values;
        # here we apply rotation about X first, then about Y
        i_n = Z * sg * ct + Y * sg * st + X * cg
        j_n = Y * ct - Z * st

        mx_n = self.Mx * cg + self.My * sg * st + self.Mz * sg * ct
        my_n = self.My * ct - self.Mz * st

        # setup
        mphi_k = np.zeros(KY.shape, dtype=complex)
        ephi_k = np.zeros(KY.shape, dtype=complex)

        nelems = np.shape(inds)[0]
        stime = time.time()
        self.vprint(f"Beginning phase calculation for {nelems:g} voxels.")
        if multiproc:
            self.vprint("Running in parallel with numba.")
            ephi_k, mphi_k = self._exp_sum(
                mphi_k, ephi_k, inds, KY, KX, j_n, i_n, my_n, mx_n, SY, SX
            )

        else:
            self.vprint("Running on 1 cpu.")
            otime = time.time()
            cc = -1
            for ind in tqdm(inds, disable=self._verbose<=0):
                ind = tuple(ind)
                cc += 1
                if time.time() - otime >= 15:
                    self.vprint(f"{cc/nelems*100:.2f}%", end=" .. ")
                    otime = time.time()
                # compute the expontential summation
                sum_term = np.exp(-1j * (KY * j_n[ind] + KX * i_n[ind]))
                ephi_k += sum_term
                mphi_k += sum_term * (my_n[ind] * SX - mx_n[ind] * SY)

        self.vprint(
            f"total time: {time.time()-stime:.5g} sec, {(time.time()-stime)/nelems:.5g} sec/voxel."
        )
        # Now we have the phases in K-space. We convert to real space and return
        ephi_k[zeros] = 0.0
        mphi_k[zeros] = 0.0
        ephi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(ephi_k)))).real * self._pre_E
        mphi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(mphi_k)))).real * self._pre_B

        return mphi, ephi


    @jit(nopython=True, parallel=True)
    def _exp_sum(self, mphi_k, ephi_k, inds, KY, KX, j_n, i_n, my_n, mx_n, Sy, Sx):
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
            mphi_k += sum_term * (my_n[z, y, x] * Sx - mx_n[z, y, x] * Sy)
        return ephi_k, mphi_k


    def _compute_arrays(self):
        # way to inherit only calc phase linsup and not this or attributes?
        dim_z, dim_y, dim_x = self._mags_shape
        ky = np.fft.fftfreq(dim_y) * 2 * np.pi
        kx = np.fft.fftfreq(dim_x) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        KK = np.sqrt(KX**2 + KY**2)
        KK = np.fft.fftshift(KK) # TODO can simplify removing ifftshift at end of comp
        zeros = np.where(KK==0)
        KK[zeros] = 1.

        SX = 1j * KX / KK**2
        SY = 1j * KY / KK**2
        SX[zeros] = 0.0
        SY[zeros] = 0.0

        return KY, KX, SY, SX, zeros

    def _get_inds(self):
        inds = np.where(self._mags_shape_func != 0)
        inds = np.array(inds).T
        return inds
