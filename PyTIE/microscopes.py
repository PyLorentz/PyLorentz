"""A class for microscope objects.

Author: CD Phatak, ANL, 20.Feb.2015.
"""

import numpy as np
import scipy.constants as physcon
import scipy.ndimage as ndimage


class Microscope(object):
    """Class for Microscope objects.

    A class that describes a microscope for image simulation and reconstruction
    purposes. Along with accelerating voltage, aberrations, and other parameters,
    this also contains methods for propagating the electron wave and simulating
    images.

    Notes:

    - When initializing a Microscope you can set verbose=True to get a printout
      of the parameters.
    - Unlike in TIE_reconstruct, here the qq frequency spectrum is expected to be
      shifted, i.e. rather than four dark corners it's a dark circle.

    Attributes:
        E (float): Accelerating voltage (V). Default 200kV.
        Cs (float): Spherical aberration (nm). Default 1mm.
        Cc (float): Chromatic aberration (nm). Default 5mm.
        theta_c (float): Beam coherence (rad). Default 0.6mrad.
        Ca (float): 2-fold astigmatism (nm). Default 0.
        phi_a (float): 2-fold astigmatism angle (rad). Default 0.
        def_spr (float): Defocus spread (nm). Default 120nm.
        defocus (float): Defocus of the microscope (nm). Default 0nm.
        lam (float): Electron wavelength (nm) calculated from E. Default 2.51pm.
        gamma (float): Relativistic factor (unitless) from E. Default 1.39.
        sigma (float): Interaction constant (1/(V*nm)) from E. Default 0.00729.
    """

    def __init__(
        self,
        E=200.0e3,
        Cs=1.0e6,
        Cc=5.0e6,
        theta_c=6.0e-4,
        Ca=0.0e6,
        phi_a=0,
        def_spr=120.0,
        verbose=False,
    ):
        """Constructs the Microscope object.

        All arguments are optional. Set verbose=True to print microscope values.
        """
        # properties that can be changed
        self.E = E
        self.Cs = Cs
        self.Cc = Cc
        self.theta_c = theta_c
        self.Ca = Ca
        self.phi_a = phi_a
        self.def_spr = def_spr
        self.defocus = 0.0  # nm
        self.aperture = 1.0

        # properties that are derived and cannot be changed directly.
        epsilon = 0.5 * physcon.e / physcon.m_e / physcon.c ** 2
        self.lam = (
            physcon.h
            * 1.0e9
            / np.sqrt(2.0 * physcon.m_e * physcon.e)
            / np.sqrt(self.E + epsilon * self.E ** 2)
        )
        self.gamma = 1.0 + physcon.e * self.E / physcon.m_e / physcon.c ** 2
        self.sigma = (
            2.0
            * np.pi
            * physcon.m_e
            * self.gamma
            * physcon.e
            * self.lam
            * 1.0e-18
            / physcon.h ** 2
        )

        if verbose:
            print(
                """
            Creating a new microscope object with the following properties:
            Quantities preceded by a star (*) can be changed using optional arguments at call.
            ------------------------------------------------------------------
            *Accelerating voltage               E: [V]    {self.E: 4.4g}
            *Spherical Aberration             Cs: [nm]    {self.Cs: 4.4g}
            *Chromatic Aberration             Cc: [nm]    {self.Cc: 4.4g}
            *Beam Coherence             theta_c: [rad]    {self.theta_c: 4.4g}
            *2-fold astigmatism               Ca: [nm]    {self.Ca: 4.4g}
            *2-fold astigmatism angle     phi_a: [rad]    {self.phi_a: 4.4g}
            *defocus spread              def_spr: [nm]    {self.def_spr: 4.4g}
            Electron wavelength           lambda: [nm]    {self.lam: 4.4g}
            Relativistic factor             gamma: [-]    {self.gamma: 4.4g}
            Interaction constant       sigma: [1/V/nm]    {self.sigma: 4.4g}
            ------------------------------------------------------------------
            """
            )

    def getSchDef(self):
        """Calculate the Scherzer defocus"""
        return np.sqrt(1.5 * self.Cs * self.lam)

    def getOptDef(self, qq, del_px):
        """Calculate the Optimum or Lichte defocus (for holography).

        Args:
            qq (2D array): Frequency array
            del_px (float): Scale (nm/pixel)

        Returns:
            float: Optimum defocus (nm)
        """
        qmax = np.amax(qq)
        lam = self.lam / del_px
        optdef = 3.0 / 4.0 * self.Cs * lam ** 2 * qmax ** 2
        return optdef

    def setAperture(self, qq, del_px, sz):
        """Set the objective aperture

        Args:
            qq (2D array): Frequency array
            del_px (float): Scale (nm/pixel)
            sz (float): Aperture size (nm).

        Returns:
            None: Sets self.aperture.
        """
        ap = np.zeros(qq.shape)
        sz_q = qq.shape
        # Convert the size of aperture from nm to nm^-1 and then to px^-1
        ap_sz = sz / del_px
        ap_sz /= float(sz_q[0])
        ap[qq <= ap_sz] = 1.0
        # Smooth the edge of the aperture
        ap = ndimage.gaussian_filter(ap, sigma=2)
        self.aperture = ap
        return 1

    def getChiQ(self, qq, del_px):
        """Calculate the phase transfer function.

        Args:
            qq (2D array): Frequency array
            del_px (float): Scale (nm/pixel)

        Returns:
            ``ndarray``: 2D array same size as qq.
        """

        # convert all the properties to pixel values
        lam = self.lam / del_px
        def_val = self.defocus / del_px
        cs = self.Cs / del_px
        ca = self.Ca / del_px

        dy, dx = np.shape(qq)
        ly = np.arange(dy) - dy // 2
        lx = np.arange(dx) - dx // 2
        lY, lX = np.meshgrid(ly, lx, indexing="ij")
        phi = np.arctan2(lY, lX)

        # compute the required prefactor terms
        p1 = np.pi * lam * (def_val + ca * np.cos(2.0 * (phi - self.phi_a)))
        p2 = np.pi * cs * lam ** 3 * 0.5

        # compute the phase transfer function
        chiq = -p1 * qq ** 2 + p2 * qq ** 4
        return chiq

    def getDampEnv(self, qq, del_px):
        """Calculate the complete damping envelope: spatial + temporal

        Args:
            qq (2D array): Frequency array
            del_px (float): Scale (nm/pixel)

        Returns:
            ``ndarray``: Damping envelope. 2D array same size as qq.
        """

        # convert all the properties to pixel values
        lam = self.lam / del_px
        def_val = self.defocus / del_px
        spread = self.def_spr / del_px
        cs = self.Cs / del_px

        # compute prefactors
        p3 = 2.0 * (np.pi * self.theta_c * spread) ** 2
        p4 = (np.pi * lam * spread) ** 2
        p5 = np.pi ** 2 * self.theta_c ** 2 / lam ** 2
        p6 = cs * lam ** 3
        p7 = def_val * lam

        # compute the damping envelope
        u = 1.0 + p3 * qq ** 2
        arg = 1.0 / u * ((p4 * qq ** 4) / 2.0 + p5 * (p6 * qq ** 3 - p7 * qq) ** 2)
        dampenv = np.exp(-arg)
        return dampenv

    def getTransferFunction(self, qq, del_px):
        """Generate the full transfer function in reciprocal space

        Args:
            qq (2D array): Frequency array
            del_px (float): Scale (nm/pixel)

        Returns:
            ``ndarray``: Transfer function. 2D array same size as qq.
        """
        chiq = self.getChiQ(qq, del_px)
        dampenv = self.getDampEnv(qq, del_px)
        tf = (np.cos(chiq) - 1j * np.sin(chiq)) * dampenv * self.aperture
        return tf

    def PropagateWave(self, ObjWave, qq, del_px):
        """Propagate object wave function to image plane.

        This function will propagate the object wave function to the image plane
        by convolving with the transfer function of microscope, and returns the
        complex real-space ImgWave

        Args:
            ObjWave (2D array): Object wave function.
            qq (2D array): Frequency array
            del_px (float): Scale (nm/pixel)

        Returns:
            ``ndarray``: Realspace image wave function. Complex 2D array same
            size as ObjWave.
        """
        # get the transfer function
        tf = self.getTransferFunction(qq, del_px)

        # Compute Fourier transform of ObjWave and convolve with tf
        f_ObjWave = np.fft.fftshift(np.fft.fftn(ObjWave))
        f_ImgWave = f_ObjWave * tf
        ImgWave = np.fft.ifftn(np.fft.ifftshift(f_ImgWave))

        return ImgWave

    def BackPropagateWave(self, ImgWave, qq, del_px):
        """Back-propagate an image wave to get the object wave.

        This function will back-propagate the image wave function to the
        object wave plane by convolving with exp(+i*Chiq). The damping
        envelope is not used for back propagation. Returns ObjWave in real space.

        Args:
            ObjWave (2D array): Object wave function.
            qq (2D array): Frequency array
            del_px (float): Scale (nm/pixel)

        Returns:
            ``ndarray``: Realspace object wave function. Complex 2D array same
            size as ObjWave.
        """

        # Get Chiq and then compute BackPropTF
        chiq = self.getChiQ(qq, del_px)
        backprop_tf = np.cos(chiq) + 1j * np.sin(chiq)

        # Convolve with ImgWave
        f_ImgWave = np.fft.fftshift(np.fft.fftn(ImgWave))
        f_ObjWave = f_ImgWave * backprop_tf
        ObjWave = np.fft.ifftn(np.fft.ifftshift(f_ObjWave))

        return ObjWave

    def getImage(self, ObjWave, qq, del_px):
        """Produce the image at the set defocus using the methods in this class.

        Args:
            ObjWave (2D array): Object wave function.
            qq (2D array): Frequency array
            del_px (float): Scale (nm/pixel)

        Returns:
            ``ndarray``: Realspace image wave function. Real-valued 2D array
            same size as ObjWave.
        """

        # Get the Propagated wave function
        ImgWave = self.PropagateWave(ObjWave, qq, del_px)
        Image = np.abs(ImgWave) ** 2

        return Image

    def getBFPImage(self, ObjWave, qq, del_px):
        """Produce the image in the backfocal plane (diffraction)

        Args:
            ObjWave (2D array): Object wave function.
            qq (2D array): Frequency array
            del_px (float): Scale (nm/pixel)

        Returns:
            ``ndarray``: Realspace image wave function. Real-valued 2D array
            same size as ObjWave.
        """
        # get the transfer function
        tf = self.getTransferFunction(qq, del_px)

        # Compute Fourier transform of ObjWave and convolve with tf
        f_ObjWave = np.fft.fftshift(np.fft.fftn(ObjWave))
        f_ImgWave = f_ObjWave * tf
        f_Img = np.abs(f_ImgWave) ** 2

        return f_Img
