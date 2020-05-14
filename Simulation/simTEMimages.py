#!/usr/bin/python
#
# This module is for simulating TEM images for various requirements such as
# through-focus series, ptychographic diff. patterns etc.
# The module consists of 3 functions currently - 
# (1) Obj_Setup - This function defines the object wave function i.e. amplitude, phase for the object
#                 This will need to be modified manually to vary the configurations. Currently setup
#                 for a disc of magnetic material forming a vortex state.
# (2) simTFS -    This function will use the Object defined using Obj_Setup and simulate the through-focus
#                 series of images for either linear or quadratic defocus series.
# (3) simPtycho - This fuction will also use the Object defined using Obj_Setup and simulate the ptychographic
#                 image series for scanning the probe across the sample and storing DPs as well as ProbeGuess.
#
# Written, CD Phatak, ANL, 05.Mar.2015.

#import necessary modules
import numpy as np
import matplotlib.pyplot as p
from scipy.interpolate import RectBivariateSpline as spline_2d
from microscopes import Microscope
from skimage import io as skimage_io
import time as time
import h5py as h5py


def Obj_Setup(dim = 256,
        del_px = 1.0,
        ran_phase = False):
    # This function is currently setup for generating the amplitude and phase shift
    # for a magnetic vortex disc on a supporting membrane. Options for random phase
    # of the membrane (carbon film) can be turned ON/OFF. It will return the amplitude
    # and phase required for simulating TEM images.
    #
    # [Amp, Mphi, Ephi] = Obj_Setup(dim=256,del_px=1.0)
    
    #Dimensions and Co-ordinates.
    d2 = dim/2
    line = np.arange(dim)-float(d2)
    [X,Y] = np.meshgrid(line,line)
    th = np.arctan2(Y,X)

    #Disc parameters
    disc_rad = 64.0 #nm
    disc_rad /= del_px #px
    disc_thk = 10.0 #nm
    disc_thk /= del_px #px
    disc_V0 = 20.0 #V
    disc_xip0 = 200.0 #nm
    disc_xip0 /= del_px
    r_vec = np.sqrt(X**2 + Y**2)
    disc = np.zeros(r_vec.shape)
    disc[r_vec <= disc_rad] = 1.0

    #Support membrane
    mem_thk = 50.0 #nm
    mem_thk /= del_px
    mem_V0 = 10.0 #V
    mem_xip0 = 800.0 #nm
    mem_xip0 /= del_px
    
    #Magnetization parameters
    b0 = 1.6e4 #Gauss
    phi0 = 20.7e6 #Gauss.nm^2
    cb = b0/phi0*del_px**2 #1/px^2

    #Magnetic Phase shift - Vortex State
    mphi = np.pi * cb * disc_thk * (disc_rad - r_vec) * disc

    #Lattice potential as a phase grating - mostly will run into Nyquist freq. problems...
    mult = 1.5
    var_pot = np.sin(X*mult)

    #Electrostatic Phase shift
    ephi = disc_thk * disc_V0 * disc * del_px * var_pot
    ephi2 = mem_thk * mem_V0 * del_px * np.ones(disc.shape)

    if ran_phase:
        ephi2 = mem_thk * mem_V0 * del_px * np.random.uniform(low = -np.pi, high = np.pi, size=disc.shape)

    #Total Ephase
    ephi += ephi2

    #Amplitude
    amp = np.exp((-np.ones(disc.shape) * mem_thk / mem_xip0) - (disc_thk / disc_xip0 * disc))

    return amp, mphi, ephi

def simTFS(microscope,
        jobID = 'simTFS',
        path = '/Users/cphatak/',
        dim = 256,
        del_px = 1.0,
        num = 11,
        defstep = 500.0,
        stype = 'Linear',
        display = False,
        flip = False):

    # This function will take first argument as the microscope object and additional 
    # parameters for number of images, defocus step, type of series. The jobID is used
    # as a prefix for all the data that is saved (TFS images in float32 format).

    #Dimensions and coordinates
    d2=dim/2
    line = np.arange(dim)-float(d2)
    [X,Y] = np.meshgrid(line,line)
    th = np.arctan2(Y,X)
    qq = np.sqrt(X**2 + Y**2) / float(dim)

    #Get the Object values
    [Amp, Mphi, Ephi] = Obj_Setup(dim=dim, del_px=del_px)

    #Create Object WaveFunction
    Ephi *= microscope.sigma
    if flip:
        Tphi = Ephi - Mphi
    else:
        Tphi = Mphi + Ephi
    ObjWave = Amp * (np.cos(Tphi) + 1j * np.sin(Tphi))

    #define the required defocus values
    num_def = np.arange(num)-float(num-1)/2
    if stype == 'Linear':
        defvals = num_def * defstep #for linear defocus series
    else:
        defvals = num_def**2 * np.sign(num_def) * defstep #for quadratic defocus series

    #check display
    if display:
        p.ion()
        fig, (im1) = p.subplots(nrows=1,ncols=1,figsize=(3,3))
        time.sleep(0.05)

    #Compute the TFS images and save them.
    imstack = []
    for d in range(num):
        #set defocus value
        microscope.defocus = defvals[d]

        #get the image
        im = microscope.getImage(ObjWave, qq, del_px)

        #save the image
        if flip: 
            fname_pref = path+jobID+stype+'_'+str(microscope.defocus)+'_flip_'
        else:
            fname_pref = path+jobID+stype+'_'+str(microscope.defocus)+'_unflip_'

        if flip:
            skimage_io.imsave(fname_pref+'{:04d}.tiff'.format(d),im.astype('float32'),plugin='tifffile')
        else:
            skimage_io.imsave(fname_pref+'{:04d}.tiff'.format(d),im.astype('float32'),plugin='tifffile')
        imstack.append(im)


        #check display
        if display:
            im1.imshow(im,cmap=p.cm.gray)
            im1.axis('off')
            im1.set_title('Image',fontsize=20)
            p.draw()

    # save stack:
    if flip: 
        skimage_io.imsave(path+jobID+stype+'_flip_imstack.tiff',np.array(imstack).astype('float32'),plugin='tifffile')
    else:
        skimage_io.imsave(path+jobID+stype+'_unflip_imstack.tiff',np.array(imstack).astype('float32'),plugin='tifffile')

    #end of function
    return 1

def simPtycho(microscope,
        jobID = 'simPtycho',
        path = '/Users/cphatak/',
        dim = 256,
        del_px = 1.0,
        num_x = 10,
        num_y = 10,
        defocus = 5000.0, #nm
        aperture = 80.0, #nm
        pr_shift = 10.0, #nm
        ObjWave = np.zeros([256,256],dtype='complex'),
        ran_shift = False,
        display = False,
        saveimg = False,
        fullTF = False):

    # This function will take the first argument as the microscope object and additional 
    # parameters for dim, pixel size, num_x and num_y for steps along x and y, defocus of 
    # the probe, aperture size, probe shift. The jobID and path are used as prefix to save
    # all the data (ProbeGuess in CSV, DPs in H5, Probe positions in float32 format).

    #Dimensiond and coordinates
    d2 = dim/2
    line = np.arange(dim) - float(d2)
    [X,Y] = np.meshgrid(line,line)
    th = np.arctan2(Y,X)
    qq = np.sqrt(X**2 + Y**2) / float(dim)

    #Set the aperture
    res = microscope.setAperture(qq, del_px, aperture)

    #Next we set the defocus
    microscope.defocus = defocus
    #Get the transfer function.
    if fullTF:
        Probe_q = microscope.getTransferFunction(qq, del_px)
    else:
        Chiq = microscope.getChiQ(qq, del_px)
        Probe_q = microscope.aperture * (np.cos(Chiq) - 1j * np.sin(Chiq))
    #Calculate the probe in real space.
    Probe = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(Probe_q)))
    Probe_Img = np.abs(Probe)**2

    #Save the Probe in complex format
    fname=path+jobID+'_ProbeGuess'
    np.savetxt(fname+'.csv',Probe,delimiter=',')
    np.savetxt(fname+'_real.txt',Probe.real)
    np.savetxt(fname+'_imag.txt',Probe.imag)

    #Compute Probe size
    pr_img_bin = np.zeros(Probe_Img.shape)
    pr_img_bin[Probe_Img > 1.e-5] = 1.0
    pr_size = np.sqrt(np.sum(pr_img_bin) / np.pi) * 2.0 #diameter

    #Check if ObjFunc is supplied as an argument
    tot = np.sum(ObjWave)
    if (tot == 0):
        #Object is not defined. We use the default ObjSetup
        #Get the Object Amp and Phase to compute Wavefunction
        [Amp, Mphi, Ephi] = Obj_Setup(dim=dim, del_px=del_px)

        #Create ObjWave
        Ephi *= microscope.sigma
        Tphi = Mphi + Ephi
        ObjWave = Amp * (np.cos(Tphi) + 1j * np.sin(Tphi))

        if saveimg:
            skimage_io.imsave(path+jobID+'_Amp.tiff',Amp.astype('float32'),plugin='tifffile')
            skimage_io.imsave(path+jobID+'_Mphi.tiff',Mphi.astype('float32'),plugin='tifffile')
            skimage_io.imsave(path+jobID+'_Ephi.tiff',Ephi.astype('float32'),plugin='tifffile')
            skimage_io.imsave(path+jobID+'_Tphi.tiff',Tphi.astype('float32'),plugin='tifffile')


    if display:
        p.ion()
        fig, (im1,im2) = p.subplots(nrows=1,ncols=2,figsize=(6,3))
        time.sleep(0.05)

    #print out the required data
    print( "Dimensions    (px):", dim)
    print( "Pixel size (nm/px):", del_px)
    print( "Probe Size    (nm):", pr_size * del_px)
    print( "Probe Shift   (nm):", pr_shift)
    print( "Scan Dims  (Nx,Ny):", num_x,num_y)
    print( "Wavelength    (nm):", microscope.lam)
    #save the simulation data to a file as well
    sim_file = open(path+jobID+'_sim_details.txt', 'w')
    sim_file.write("Dimension     (px): %s \n" % dim)
    sim_file.write("Pixel size (nm/px): %s \n" % del_px)
    sim_file.write("Probe Size    (nm): %s \n" % (pr_size * del_px))
    sim_file.write("Probe Shift   (nm): %s \n" % pr_shift)
    sim_file.write("Scan Dims  (Nx,Ny): %s,%s \n" % (num_x,num_y))
    sim_file.write("Wavelength    (nm): %s \n" % microscope.lam)
    sim_file.close()

    #HDF5 file path
    h5_file_path = '/entry/instrument/detector/data'

    #Precompute the interpolates for probe shift
    Probe_a = spline_2d(line,line,Probe.real)
    Probe_b = spline_2d(line,line,Probe.imag)

    #Loop over scan positions
    count = 0
    for ix in range(-num_x/2,num_x/2):
        for iy in range(-num_y/2,num_y/2):
            #Compute interpolated probe position
            Probe_s = np.zeros(Probe.shape, dtype=Probe.dtype)
            xs = ix * pr_shift
            ys = iy * pr_shift
            if ran_shift:
                xs += np.random.randint(-3,4)
                ys += np.random.randint(-3,4)

            #Probe_s.real = Probe_a(line + xs, line + ys)
            #Probe_s.imag = Probe_b(line + xs, line + ys)
            Probe_s = np.roll(np.roll(Probe, int(xs), axis=0), int(ys), axis=1)

            #Save probe image
            Probe_s_Img = np.abs(Probe_s)**2
            skimage_io.imsave(path+jobID+'_ProbePos_'+str(count).zfill(4)+'.tiff',
                    Probe_s_Img.astype('float32'), plugin='tifffile')
            #Compute DP
            Dp_Img_s = np.abs(np.fft.fftshift(np.fft.fftn(ObjWave * Probe_s)))**2
            fname = path+jobID+'_DpScan_'+str(count).zfill(4)+'.h5'
            h5py.File(fname)[h5_file_path] = Dp_Img_s.astype('float32')
            #Increment count
            count += 1

            if display:
                im1.imshow(Probe_s_Img,cmap=p.cm.gray)
                im1.axis('off')
                im1.set_title('ProbePos',fontsize=20)
                im2.imshow(Dp_Img_s,cmap=p.cm.gray)
                im2.axis('off')
                im2.set_title('DiffPatt',fontsize=20)
                fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                        bottom=0.02, left=0.02, right=0.98)
                p.draw()

    #End of function
    return 1

















