#!/usr/bin/python
#
#Python Class file for Microscope.
#
#Written by CD Phatak, ANL, 20.Feb.2015.
#
import numpy as np
import scipy.constants as physcon
import scipy.ndimage as ndimage


class Microscope(object):

    def __init__(self, E=200.0e3, Cs=1.0e6, Cc=5.0e6, theta_c=6.0e-4, Ca=0.0e6, phi_a=0, def_spr=120.0,verbose=False):
        
        #initialize with either default values or user supplied values - properties that can be changed
        self.E = E#200.0e3
        self.Cs = Cs#1.0e6
        self.Cc = Cc#5.0e6
        self.theta_c = theta_c#6.0e-4
        self.Ca = Ca#0.0e6
        self.phi_a = phi_a#0
        self.def_spr = def_spr#120.0
        self.defocus = 0.0 #nm
        self.aperture = 1.0
        
        #properties that are derived and cannot be changed directly.
        epsilon = 0.5 * physcon.e / physcon.m_e / physcon.c**2
        self.lam = physcon.h * 1.0e9 / np.sqrt(2.0 * physcon.m_e * physcon.e) / np.sqrt(self.E + epsilon * self.E**2)
        self.gamma = 1.0 + physcon.e * self.E / physcon.m_e / physcon.c**2
        self.sigma = 2.0 * np.pi * physcon.m_e * self.gamma * physcon.e * self.lam * 1.0e-18 / physcon.h**2
        
        if verbose:
            print( "Creating a new microscope object with the following properties:")
            print( "Quantities preceded by a star (*) can be changed using optional arguments at call.")
            print( "-------------------------------------------------------------------------")
            print( "*Accelerating voltage         E: [V]      ",self.E)
            print( "*Spherical Aberration        Cs: [nm]     ",self.Cs)
            print( "*Chromatic Aberration        Cc: [nm]     ",self.Cc)
            print( "*Beam Coherence         theta_c: [rad]    ",self.theta_c)
            print( "*2-fold astigmatism          Ca: [nm]     ",self.Ca)
            print( "*2-fold astigmatism angle phi_a: [rad]    ",self.phi_a)
            print( "*defocus spread         def_spr: [nm]     ",self.def_spr)
            print( "Electron wavelength      lambda: [nm]     ",self.lam)
            print( "Relativistic factor       gamma: [-]      ",self.gamma)
            print( "Interaction constant      sigma: [1/V/nm] ",self.sigma)
            print( "-------------------------------------------------------------------------")

    def getSchDef(self):
        #Calculate the Scherzer defocus
        return np.sqrt(1.5 * self.Cs * self.lam)

    def getOptDef(self,qq,del_px):
        #Calculate the Optimum or Lichte defocus (in case for holography)
        qmax = np.amax(qq)
        lam = self.lam / del_px
        optdef = 3.0/4.0 * self.Cs * lam**2 * qmax**2
        return optdef

    def setAperture(self,qq,del_px, sz):
        #This function will set the objective aperture
        #the input size of aperture sz is given in nm.
        ap = np.zeros(qq.shape)
        sz_q = qq.shape
        #Convert the size of aperture from nm to nm^-1 and then to px^-1
        ap_sz = sz/del_px
        ap_sz /= float(sz_q[0])
        ap[qq <= ap_sz] = 1.0
        #Smooth the edge of the aperture
        ap = ndimage.gaussian_filter(ap,sigma=2)
        self.aperture = ap
        return 1

    def getChiQ(self,qq,del_px):
        #this function will calculate the phase transfer function.
        
        #convert all the properties to pixel values
        lam = self.lam / del_px
        def_val = self.defocus / del_px
        spread = self.def_spr / del_px
        cs = self.Cs / del_px
        ca = self.Ca / del_px
        phi = 0

        #compute the required prefactor terms
        p1 = np.pi * lam * (def_val + ca * np.cos(2.0 * (phi - self.phi_a)))
        p2 = np.pi * cs * lam**3 * 0.5
        p3 = 2.0 * (np.pi * self.theta_c * spread)**2

        #compute the phase transfer function
        u = 1.0 + p3 * qq**2
        chiq = -p1 * qq**2 + p2 * qq**4
        return chiq

    def getDampEnv(self,qq,del_px):
        #this function will calculate the complete damping envelope: spatial + temporal
        
        #convert all the properties to pixel values
        lam = self.lam / del_px
        def_val = self.defocus / del_px
        spread = self.def_spr / del_px
        cs = self.Cs / del_px

        #compute prefactors
        p3 = 2.0 * (np.pi * self.theta_c * spread)**2
        p4 = (np.pi * lam * spread)**2
        p5 = np.pi**2 * self.theta_c**2 / lam**2
        p6 = cs * lam**3
        p7 = def_val * lam

        #compute the damping envelope
        u = 1.0 + p3 * qq**2
        arg = 1.0/u*((p4 * qq**4) / 2.0 + p5 * (p6 * qq**3 - p7 * qq)**2)
        dampenv = np.exp(-arg)
        return dampenv

    def getTransferFunction(self,qq,del_px):
        #This function will generate the full transfer function in reciprocal space- 
        # tf = exp(-i*Chiq)*DampEnv
        chiq = self.getChiQ(qq,del_px)
        dampenv = self.getDampEnv(qq,del_px)
        tf = (np.cos(chiq) - 1j * np.sin(chiq)) * dampenv * self.aperture
        return tf

    def PropagateWave(self, ObjWave, qq, del_px):
        #This function will propagate the object wave function to the image plane
        #by convolving with the transfer function of microscope and returns the 
        #complex real-space ImgWave

        #get the transfer function
        tf = self.getTransferFunction(qq, del_px)
        
        #Compute Fourier transform of ObjWave and convolve with tf
        f_ObjWave = np.fft.fftshift(np.fft.fftn(ObjWave))
        f_ImgWave = f_ObjWave * tf
        ImgWave = np.fft.ifftn(np.fft.ifftshift(f_ImgWave))

        return ImgWave

    def BackPropagateWave(self, ImgWave, qq, del_px):
        #This function will back-propagate the image wave function to the 
        #object wave plane by convolving with exp(+i*Chiq). The damping 
        #envelope is not used for back propagation. Returns ObjWave in real space.

        #Get Chiq and then compute BackPropTF
        chiq = getChiQ(qq,del_px)
        backprop_tf = (np.cos(chiq) + 1j * np.sin(chiq))

        #Convolve with ImgWave
        f_ImgWave = np.fft.fftshift(np.fft.fftn(ImgWave))
        f_ObjWave = f_ImgWave * backprop_tf
        ObjWave = np.fft.ifftn(np.fft.ifftshift(f_ObjWave))

        return ObjWave

    def getImage(self, ObjWave, qq, del_px):
        #This function will produce the image at the set defocus using the 
        #methods in this class.

        #Get the Propagated wave function
        ImgWave = self.PropagateWave(ObjWave, qq, del_px)
        Image = np.abs(ImgWave)**2

        return Image

    def getBFPImage(self, ObjWave, qq, del_px):
        #This function will produce the image in the backfocal plane (diffraction)
        #using the transfer function and the object wave
        
        #get the transfer function
        tf = self.getTransferFunction(qq, del_px)
        
        #Compute Fourier transform of ObjWave and convolve with tf
        f_ObjWave = np.fft.fftshift(np.fft.fftn(ObjWave))
        f_ImgWave = f_ObjWave * tf
        f_Img = np.abs(f_ImgWave)**2

        return f_Img


## MAIN ##
if __name__ == '__main__':
    print( "Class definition for Microscope Class.")



