"""A class for microscope objects.

Author: CD Phatak, ANL, 20.Feb.2015.
"""

import numpy as np
import scipy.constants as physcon
import scipy.ndimage as ndimage
import textwrap


class Microscope(object):
    """Class for Microscope objects.

    A class that describes a microscope for image simulation and reconstruction
    purposes. Along with accelerating voltage, aberrations, and other parameters,
    this also contains methods for propagating the electron wave and simulating
    images.

    Notes:

    - When initializing a Microscope you can set verbose=True to get a printout
      of the parameters.

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
        scale=None,
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
        self._qq = None
        self.scale = scale

        # properties that are derived and cannot be changed directly.
        epsilon = 0.5 * physcon.e / physcon.m_e / physcon.c**2
        self.lam = (
            physcon.h
            * 1.0e9
            / np.sqrt(2.0 * physcon.m_e * physcon.e)
            / np.sqrt(self.E + epsilon * self.E**2)
        )
        self.gamma = 1.0 + physcon.e * self.E / physcon.m_e / physcon.c**2
        self.sigma = (
            2.0
            * np.pi
            * physcon.m_e
            * self.gamma
            * physcon.e
            * self.lam
            * 1.0e-18
            / physcon.h**2
        )

        if verbose:
            print(
                textwrap.dedent(
                    f"""
            Creating a new microscope object with the following properties:
            Quantities preceded by a star (*) can be changed using optional arguments at call.
            --------------------------------------------------------------
            *Accelerating voltage        [V]         E:        {self.E: 4.4g}
            *Spherical Aberration        [nm]        Cs:       {self.Cs: 4.4g}
            *Chromatic Aberration        [nm]        Cc:       {self.Cc: 4.4g}
            *Beam Coherence              [rad]       theta_c:  {self.theta_c: 4.4g}
            *2-fold astigmatism          [nm]        Ca:       {self.Ca: 4.4g}
            *2-fold astigmatism angle    [rad]       phi_a:    {self.phi_a: 4.4g}
            *defocus spread              [nm]        def_spr:  {self.def_spr: 4.4g}
            Electron wavelength          [nm]        lambda:   {self.lam: 4.4g}
            Relativistic factor          [-]         gamma:    {self.gamma: 4.4g}
            Interaction constant         [1/V/nm]    sigma:    {self.sigma: 4.4g}
            --------------------------------------------------------------
            """
                )
            )

    def get_scherzer_defocus(self):
        """Calculate the Scherzer defocus"""
        return np.sqrt(1.5 * self.Cs * self.lam)

    def get_optimum_defocus(self):
        """Calculate the Optimum or Lichte defocus (for holography).

        Returns:
            float: Optimum defocus (nm)
        """
        qmax = np.sqrt(2)/2
        lam = self.lam / self.scale
        optdef = 3.0 / 4.0 * self.Cs * lam**2 * qmax**2
        return optdef

    def _set_aperture(self, sz):
        """Set the objective aperture

        Args:
            sz (float): Aperture size (nm).

        Returns:
            None: Sets self.aperture.
        """
        sz_q = self._qq.shape
        ap = np.zeros_like(self._qq)
        # Convert the size of aperture from nm to nm^-1 and then to px^-1
        ap_sz = sz / self.scale
        ap_sz /= float(sz_q[0])
        ap[self._qq <= ap_sz] = 1.0
        # Smooth the edge of the aperture
        ap = ndimage.gaussian_filter(ap, sigma=2)
        self.aperture = ap
        return 1

    def _get_chiQ(self):
        """Calculate the phase transfer function.

        Args:

        Returns:
            ``ndarray``: 2D array.
        """

        # convert all the properties to pixel values
        lam = self.lam / self.scale
        def_val = self.defocus / self.scale
        cs = self.Cs / self.scale
        ca = self.Ca / self.scale

        dy, dx = np.shape(self._qq)
        ly = np.fft.fftfreq(dy)
        lx = np.fft.fftfreq(dx)
        [X, Y] = np.meshgrid(lx, ly)
        phi = np.arctan2(Y, X)

        # compute the required prefactor terms
        p1 = np.pi * lam * (def_val + ca * np.cos(2.0 * (phi - self.phi_a)))
        p2 = np.pi * cs * lam**3 * 0.5

        # compute the phase transfer function
        chiq = -p1 * self._qq**2 + p2 * self._qq**4
        return chiq

    def _get_damping_envelope(self):
        """Calculate the complete damping envelope: spatial + temporal

        Args:

        Returns:
            ``ndarray``: Damping envelope. 2D array.
        """

        # convert all the properties to pixel values
        lam = self.lam / self.scale
        def_val = self.defocus / self.scale
        spread = self.def_spr / self.scale
        cs = self.Cs / self.scale

        # compute prefactors
        p3 = 2.0 * (np.pi * self.theta_c * spread) ** 2
        p4 = (np.pi * lam * spread) ** 2
        p5 = np.pi**2 * self.theta_c**2 / lam**2
        p6 = cs * lam**3
        p7 = def_val * lam

        # compute the damping envelope
        u = 1.0 + p3 * self._qq**2
        arg = 1.0 / u * ((p4 * self._qq**4) / 2.0 + p5 * (p6 * self._qq**3 - p7 * self._qq) ** 2)
        dampenv = np.exp(-arg)
        return dampenv

    def get_transfer_function(self):
        """Generate the full transfer function in reciprocal space

        Args:

        Returns:
            ``ndarray``: Transfer function. 2D array.
        """
        chiq = self._get_chiQ()
        dampenv = self._get_damping_envelope()
        tf = (np.cos(chiq) - 1j * np.sin(chiq)) * dampenv * self.aperture
        return tf

    def _propagate_wave(self, ObjWave):
        """Propagate object wave function to image plane.

        This function will propagate the object wave function to the image plane
        by convolving with the transfer function of microscope, and returns the
        complex real-space ImgWave

        Args:
            ObjWave (2D array): Object wave function.

        Returns:
            ``ndarray``: Realspace image wave function. Complex 2D array same
            size as ObjWave.
        """
        # get the transfer function
        tf = self.get_transfer_function()

        # Compute Fourier transform of ObjWave and convolve with tf
        f_ObjWave = np.fft.fft2(ObjWave)
        f_ImgWave = f_ObjWave * tf
        ImgWave = np.fft.ifft2(f_ImgWave)

        return ImgWave

    def backpropagate_wave(self, ImgWave:np.ndarray):
        """Back-propagate an image wave to get the object wave.

        This function will back-propagate the image wave function to the
        object wave plane by convolving with exp(+i*Chiq). The damping
        envelope is not used for back propagation. Returns ObjWave in real space.

        Args:
            ObjWave (2D array): Object wave function.

        Returns:
            ``ndarray``: Realspace object wave function. Complex 2D array same
            size as ObjWave.
        """

        # Get Chiq and then compute BackPropTF
        self._get_qq(ImgWave.shape)
        chiq = self._get_chiQ()
        backprop_tf = np.cos(chiq) + 1j * np.sin(chiq)

        # Convolve with ImgWave
        f_ImgWave = np.fft.fft2(ImgWave)
        f_ObjWave = f_ImgWave * backprop_tf
        ObjWave = np.fft.ifft2(f_ObjWave)

        return ObjWave

    def compute_image(self, ObjWave:np.ndarray, padded_shape=None):
        """Produce the image at the set defocus using the methods in this class.

        Args:
            ObjWave (2D array): Object wave function.

        Returns:
            ``ndarray``: Realspace image wave function. Real-valued 2D array
            same size as ObjWave.
        """

        # Get the Propagated wave function
        if padded_shape is not None:
            self._get_qq(padded_shape)
            dimy, dimx = ObjWave.shape
            pdimy, pdimx = padded_shape
            py = (pdimy - dimy) // 2
            px = (pdimx - dimx) // 2
            objwave = np.pad(ObjWave, ((py, py), (px, px)), mode="edge")
            ImgWave = self._propagate_wave(objwave)
            Image = np.abs(ImgWave) ** 2
            Image = Image[py:-py, px:-px]

        else:
            self._get_qq(ObjWave.shape)
            ImgWave = self._propagate_wave(ObjWave)
            Image = np.abs(ImgWave) ** 2

        return Image

    def _get_qq(self, shape:tuple):
        """
            qq (2D array): Frequency array
        """
        ly = np.fft.fftfreq(shape[0])
        lx = np.fft.fftfreq(shape[1])
        [X, Y] = np.meshgrid(lx, ly)
        self._qq = np.sqrt(X**2 + Y**2)
        return

    def compute_diffraction_pattern(self, ObjWave:np.ndarray):
        """Produce the image in the backfocal plane (diffraction)

        Args:
            ObjWave (2D array): Object wave function.

        Returns:
            ``ndarray``: Realspace image wave function. Real-valued 2D array
            same size as ObjWave.
        """
        self._get_qq(ObjWave.shape)
        # get the transfer function
        tf = self.get_transfer_function()

        # Compute Fourier transform of ObjWave and convolve with tf
        f_ObjWave = np.fft.fft2(ObjWave)
        f_ImgWave = f_ObjWave * tf
        f_Img = np.abs(f_ImgWave) ** 2

        return f_Img
