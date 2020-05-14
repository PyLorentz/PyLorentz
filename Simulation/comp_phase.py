#!/usr/bin/python
#
# This module consists of functions for simulating the phase shift of a given
# object. 
# It contained two functions:
# 1) linsupPhi - using the linear supeposition principle for application in MBIR type
#                3D reconstruction of magnetization (both magnetic and electrostatic)
# 2) mansPhi - using the Mansuripur Algorithm to compute the phase shift (only magnetic)
#
# Written, CD Phatak, ANL, 08.May.2015.
# Modified, CD Phatak, ANL, 22.May.2016.

#import necessary modules
import numpy as np
from TIE_helper import *
#import matplotlib.pyplot as p
#from scipy.interpolate import RectBivariateSpline as spline_2d
#from microscopes import Microscope
#from skimage import io as skimage_io
#import time as time

def linsupPhi(mx = 1.0,
        my = 1.0,
        mz = 1.0,
        Dshp = 1.0,
        theta_x = 0.0,
        theta_y = 0.0,
        pre_B = 1.0,
        pre_E = 1.0):
    
    # This function will take the 3D arrays with Mx, My and Mz components of the magnetization
    # and the Dshp array consisting of the shape function for the object (1 inside, 0 outside)
    # and then the tilt angles about x and y axes to compute the magnetic phase shift and
    # the electrostatic phase shift. Initial computation is done in Fourier space and then
    # real space values are returned.

    [ysz,xsz,zsz] = mx.shape
    dim = xsz #Assuming same dimensions along X and Y
    d2 = dim//2
    if zsz > 1:
        dz2 = zsz//2
    else:
        dz2 = 0

    line = np.arange(dim)-np.float(d2)
    [Y,X] = np.meshgrid(line,line)
    dk = 2.0*np.pi/np.float(dim) # Kspace vector spacing
    KX = X*dk
    KY = Y*dk
    KK = np.sqrt(KX**2 + KY**2)
    zinds = np.where(KK == 0)
    KK[zinds] = 1.0 # Need to take care of points where KK is zero since we will be dividing later

    #now compute constant factors - S
    Sx = 1j * pre_B * KX / KK**2
    Sy = 1j * pre_B * KY / KK**2
    Sx[zinds] = 0.0
    Sy[zinds] = 0.0

    #Now we loop through all coordinates and compute the summation terms
    mphi_k = np.zeros(KK.shape,dtype=complex)
    ephi_k = np.zeros(KK.shape,dtype=complex)
    
    #Trying to use nonzero elements in Dshape to limit the iterations.
    (Jn, In,Kn) = np.where(Dshp != 0)
    
    nelems = In.size
    print('nelems = ', nelems)
    for nn in range(nelems):
        if nn % 1000 == 0:
            print('{:.3}%'.format(nn/nelems*100))
        # Compute the rotation angles
        st = np.sin(np.deg2rad(theta_x))
        ct = np.cos(np.deg2rad(theta_x))
        sg = np.sin(np.deg2rad(theta_y))
        cg = np.cos(np.deg2rad(theta_y))
        # compute the rotated values; 
        # here we apply rotation about X first, then about Y
        i = In[nn] - d2
        j = Jn[nn] - d2
        k = Kn[nn] - dz2
        i_n = np.float(i) * cg + np.float(j) * sg * st + np.float(k) * sg * ct
        j_n = np.float(j) * ct - np.float(k) * st

        mx_n = mx[Jn[nn],In[nn],Kn[nn]] * cg + my[Jn[nn],In[nn],Kn[nn]] * sg * st + mz[Jn[nn],In[nn],Kn[nn]] * sg * ct
        my_n = my[Jn[nn],In[nn],Kn[nn]] * ct - mz[Jn[nn],In[nn],Kn[nn]] * st

        # compute the expontential summation
        sum_term = np.exp(-1j * (KX * j_n + KY * i_n))
        # add to ephi
        ephi_k += sum_term * Dshp[Jn[nn],In[nn],Kn[nn]]
        # add to mphi
        mphi_k += sum_term * (my_n * Sy - mx_n * Sx)

    #Now we have the phases in K-space. We convert to real space and return
    ephi_k[zinds] = 0.0
    mphi_k[zinds] = 0.0

    ephi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(ephi_k)))).real*pre_E
    mphi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(mphi_k)))).real # already have this in Sx and Sy: * pre_B

    return [ephi,mphi]

# Function for using Mansuripur Algorithm. The input given is assumed to be 2D array for Bx,By,Bz.
def mansPhi(bx = 1.0,
        by = 1.0,
        bz = 1.0,
        beam = [0.0,0.0,1.0], #beam direction
        thick = 1.0, #thickness of sample
        embed = 0.0): #embedding the array into bigger array for F-space comp.

    #Normalize the beam direction
    beam = np.array(beam)
    beam /= np.sqrt(np.sum(beam**2))

    #Get dimensions
    [xsz,ysz] = bx.shape

    #Embed
    if (embed == 1.0):
        bdim = 1024.0
        bdimx,bdimy = bdim,bdim
    elif (embed == 0.0):
        bdimx,bdimy = xsz,ysz
    else:
        bdim = np.float(embed)
        bdimx,bdimy = bdim,bdim

    bigbx = np.zeros([bdimx,bdimy])
    bigby = np.zeros([bdimx,bdimy])
    bigbx[int(bdimx/2-xsz/2):int(bdimx/2+xsz/2),int(bdimy/2-ysz/2):int(bdimy/2+ysz/2)] = bx
    bigby[int(bdimx/2-xsz/2):int(bdimx/2+xsz/2),int(bdimy/2-ysz/2):int(bdimy/2+ysz/2)] = by
    if (bz != 1.0):
        bigbz = np.zeros([bdimx,bdimy])
        bigbz[bdimx/2-xsz/2:bdimx/2+xsz/2,bdimy/2-ysz/2:bdimy/2+ysz/2] = bz

    #Compute the auxiliary arrays requried for computation
    dsx = 2.0*np.pi/bdimx 
    linex = (np.arange(bdimx)-np.float(bdimx/2))*dsx
    dsy = 2.0*np.pi/bdimy
    liney = (np.arange(bdimy)-np.float(bdimy/2))*dsy
    [Sx,Sy] = np.meshgrid(linex,liney)
    S = np.sqrt(Sx**2 + Sy**2)
    zinds = np.where(S == 0)
    S[zinds] = 1.0
    sigx = Sx/S
    sigy = Sy/S
    sigx[zinds] = 0.0
    sigy[zinds] = 0.0

    #compute FFTs of the B arrays.
    fbx = np.fft.fftshift(np.fft.fftn(bigbx))
    fby = np.fft.fftshift(np.fft.fftn(bigby))
    if (bz != 1.0):
        fbz = np.fft.fftshift(np.fft.fftn(bigbz))

    #Compute vector products and Gpts
    if (bz == 1.0): # eq 13a in Mansuripur 
        prod = sigx*fby - sigy*fbx
        Gpts = 1+1j*0
    else:
        prod = sigx*(fby*beam[0]**2 - fbx*beam[0]*beam[1] - fbz*beam[1]*beam[2]+ fby*beam[2]**2
                ) + sigy*(fby*beam[0]*beam[1] - fbx*beam[1]**2 + fbz*beam[0]*beam[2] - fbx*beam[2]**2)
        arg = np.float(np.pi*thick*(sigx*beam[0]+sigy*beam[1])/beam[2])
        qq = np.where(arg == 0)
        denom = 1.0/((sigx*beam[0]+sigy*beam[1])**2 + beam[2]**2)
        Gpts = complex(denom*np.sin(arg)/arg)
        Gpts[qq] = denom[qq]

    #prefactor
    prefac = 1j*thick/S
    
    #F-space phase
    fphi = prefac * Gpts * prod
    fphi[zinds] = 0.0
    phi = np.fft.ifftn(np.fft.ifftshift(fphi)).real

    #return only the actual phase part from the embed file
    ret_phi = phi[bdimx//2-xsz//2:bdimx//2+xsz//2,bdimy//2-ysz//2:bdimy//2+ysz//2]

    return ret_phi
#done.





