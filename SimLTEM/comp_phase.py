"""This module consists of functions for simulating the phase shift of a given
object. 

It contained two functions:
1) linsupPhi - using the linear supeposition principle for application in MBIR type
               3D reconstruction of magnetization (both magnetic and electrostatic)
2) mansPhi - using the Mansuripur Algorithm to compute the phase shift (only magnetic)

Written, CD Phatak, ANL, 08.May.2015.
Modified, CD Phatak, ANL, 22.May.2016.
"""

import numpy as np
from TIE_helper import *

def linsupPhi(mx=1.0, my=1.0, mz=1.0, Dshp=1.0, theta_x=0.0, theta_y=0.0, pre_B=1.0, pre_E=1.0):
    """Applies linear supeposition principle for 3D reconstruction of magnetic and electrostatic phase shifts.

    This function will take the 3D arrays with Mx, My and Mz components of the magnetization
    and the Dshp array consisting of the shape function for the object (1 inside, 0 outside)
    and then the tilt angles about x and y axes to compute the magnetic phase shift and
    the electrostatic phase shift. Initial computation is done in Fourier space and then
    real space values are returned.

    Args: 
        mx: 3D array. x component of magnetization at each voxel
        my: 3D array. y component of magnetization at each voxel
        mz: 3D array. z component of magnetization at each voxel
        Dshp: 3D array. Binary shape function of the object. 1 inside, 0 outside
        theta_x: Float. Rotation around x-axis (degrees) 
        theta_y: Float. Rotation around y-axis (degrees) 
        pre_B: Float. Prefactor for unit conversion in calculating the magnetic 
            phase shift. Units 1/pixels^2. Generally (2*pi*b0*(nm/pix)^2)/phi0 
            where b0 is the saturation magnetization and phi0 the magnetic flux
            quantum. 
        pre_E: Float. Prefactor for unit conversion in calculating the 
            electrostatic phase shift. Equal to sigma*V0, where sigma is the 
            interaction constant of the given TEM accelerating voltage (is an 
            attribute of the microscope class), and V0 the mean inner potential.

    Returns: [ephi, mphi]
        ephi: Electrostatic phase shift, 2D array
        mphi: magnetic phase shift, 2D array
    """

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

def mansPhi(bx = 1.0,by = 1.0,bz = 1.0,beam = [0.0,0.0,1.0],thick = 1.0,embed = 0.0): 
    """Calculate magnetic phase shift using Mansuripur algorithm [1]. 

    Unlike the linear superposition method, this algorithm only accepts 2D 
    samples. The input given is expected to be 2D arrays for Bx, By, Bz. 

    Args: 
        bx: 2D array. x component of magnetization at each pixel.
        by: 2D array. y component of magnetization at each pixel.
        bz: 2D array. z component of magnetization at each pixel.  
        beam: List [x,y,z]. Vector direction of beam. Default [001]. 
        thick: Float. Thickness of the sample in pixels, need not be an int. 
        embed:  Int. Whether or not to embed the bx, by, bz into a larger array
            for fourier-space computation. This improves edge effects at the 
            cost of reduced speed. 
            embed = 0: Do not embed (default)
            embed = 1: Embed in (1024, 1024) array
            embed = x: Embed in (x, x) array. 

    Returns: 
        Magnetic phase shift, 2D array
    
    [1] Mansuripur, M. Computation of electron diffraction patterns in Lorentz 
        electron microscopy of thin magnetic films. J. Appl. Phys. 69, 5890 (1991).
    """

# Function for using Mansuripur Algorithm. The input given is assumed to be 2D array for Bx,By,Bz.
#embedding the array into bigger array for F-space comp.


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