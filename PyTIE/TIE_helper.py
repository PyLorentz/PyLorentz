#!/usr/bin/python
#
# NAME: TIE_helper.py
#
# PURPOSE:
# An assortment of helper functions for TIE_reconstruct.py and TIE_template.ipynb
#
# CALLING SEQUENCE:
# Functions are imported and called as needed. 
#
# AUTHOR:
# Arthur McCray, ANL, Summer 2019.
#----------------------------------------------------------------------------------------------------

 
import matplotlib.pyplot as plt 
import numpy as np
import hyperspy.api as hs
from skimage import io
from scipy.ndimage.filters import median_filter
from ipywidgets import interact
import hyperspy # just for checking type in show_stack. 
from copy import deepcopy

from TIE_params import TIE_params




def load_data(path, fls_file, al_file, flip, flip_fls_file = None, filtersize = 3): 
    '''
    Load files in a directory (from a .fls file) using hyperspy. 
    Expects the .fls part of the filename.
    Also pass a aligned stack of images
    Returns (dm3stack, defocus_values)
    
    Might need to load the tifstack too, but remember that the focus order
    isn't the same. Dm3stack is same as fls file: 
    [in focus, -1, -2 ..., -n, +1, +2, ..., +n] 
    while tifstack (as currently instructed to do) is: 
    [-n, ..., -1, in focus, +1, ..., +n]
    '''

    unflip_files = []
    flip_files = []

    if not fls_file.endswith('.fls'):
        fls_file += '.fls'

    if flip_fls_file is None: # one fls file given
        fls = []
        with open(path + fls_file) as file:
            # stip newlines
            for line in file:
                fls.append(line.strip())
        
        num_files = int(fls[0])
        
        if flip: 
            for line in fls[1:num_files+1]:
                unflip_files.append(path + 'unflip/' + line)
            for line in fls[1:num_files+1]:
                flip_files.append(path + 'flip/' + line)
        else:
            for line in fls[1:num_files+1]:
                unflip_files.append(path + 'tfs/' + line)
    
    else: # there are 2 fls files given
        if not flip: 
            print("""You probably made a mistake.
                You're defining a flip fls file but saying there is no full tfs for both unflip and flip.
                If just one tfs use one fls file.\n""")
            return 0 
        if not flip_fls_file.endswith('.fls'):
            flip_fls_file += '.fls'

        fls = []
        flip_fls = []
        with open(path + fls_file) as file:
            for line in file:
                fls.append(line.strip())

        with open(path + flip_fls_file) as file:
            for line in file:
                flip_fls.append(line.strip())

        num_files = int(fls[0])
        for line in fls[1:num_files+1]:
            unflip_files.append(path + 'unflip/' + line)
        for line in flip_fls[1:num_files+1]:
            flip_files.append(path + 'flip/' + line)

    dm3stack = hs.load(unflip_files)
    if flip:
        flip_dm3stack = hs.load(flip_files)
    else:
        flip_dm3stack = []

    # convert scale dimensions to nm
    for sig in dm3stack + flip_dm3stack: 
        sig.axes_manager.convert_units(units = ['nm', 'nm'])

    if unflip_files[0][-4:] != '.dm3' and unflip_files[0][-4:] != '.dm4': 
        # if not dm3's then they generally don't have the title metadata. 
        for sig in dm3stack + flip_dm3stack: 
            sig.metadata.General.title = sig.metadata.General.original_filename

    # load the aligned tifs and update the dm3 data to match
    try:
        al_tifs = io.imread(path + al_file)

    except FileNotFoundError as e:
        print('Incorrect aligned stack filename given.')
        raise e

    
    if flip:
        tot_files = 2*num_files
    else:
        tot_files = num_files 

    for i in range(tot_files):
        # pull slices from correct axis
        if al_tifs.shape[0] < al_tifs.shape[2]:
            im = al_tifs[i]
        elif al_tifs.shape[0] > al_tifs.shape[2]:
            im = al_tifs[:,:,i]
        else:
            print("Bad stack\n Or maybe the second axis is slice axis?")
            print('Loading failed.\n')
            return 0
        
        # then median filter to remove "hot pixels"
        im = median_filter(im, size= filtersize)

        # and assign to appropriate stack 
        if i < num_files:
            print('loading unflip:', unflip_files[i])
            dm3stack[i].data = im
        else: 
            j = i - num_files
            print('loading flip:', flip_files[j])
            flip_dm3stack[j].data = im

    # read the defocus values
    defvals = fls[-(num_files//2):]
    assert num_files == 2*len(defvals) + 1
    defvals = [float(i) for i in defvals] # defocus values +/-

    ptie = TIE_params(dm3stack, flip_dm3stack, defvals, flip, path)

    print('done\n')
    return (dm3stack, flip_dm3stack, ptie)


def select_tifs(i, ptie, long_deriv = False):
    '''
    Returns a list of the images which will be fed into TIE().
    '''
    if long_deriv:
        recon_tifs = []
        for sig in ptie.dm3stack:
            recon_tifs.append(sig.data)
        if ptie.flip:
            for sig in ptie.flip_dm3stack:
                recon_tifs.append(sig.data)

    else:
        num_files = ptie.num_files
        under = num_files//2 - (i+1)
        over = num_files//2 + (i+1)
        dm3stack = ptie.dm3stack
        flip_dm3stack = ptie.flip_dm3stack
        if ptie.flip:
            recon_tifs = [
                dm3stack[under].data,           # +-
                flip_dm3stack[under].data,      # --
                (dm3stack[num_files//2].data + 
                 flip_dm3stack[num_files//2].data)/2,    # 0
                dm3stack[over].data,            # ++
                flip_dm3stack[over].data        # -+
            ]
        else:
            recon_tifs = [
                dm3stack[under].data,           # +-
                dm3stack[num_files//2].data,    # 0
                dm3stack[over].data             # ++
            ]
    try:
        recon_tifs = deepcopy(recon_tifs) ### not sure what the best practice is for this
    except TypeError:
        print("TypeError in select_tifs deepcopy. Proceeding with originals.")
    return recon_tifs


def dist(ny,nx):
    '''
    Implementation of the IDL DIST function. 
    IDL Description: "Returns a rectangular array in which the value of each 
    element is proportional to its frequency." 
    My description: "Creates an array where each value is smallest distance to a 
    corner (measured from upper left corner of pixel)"
    '''
    axisy = np.linspace(-ny//2+1, ny//2, ny)
    axisx = np.linspace(-nx//2+1, nx//2, nx)
    result = np.sqrt(axisx**2 + axisy[:,np.newaxis]**2)
    return np.roll(np.roll(result, ny//2+1, axis=0), nx//2+1, axis = 1)


def scale_stack(imstack):
    '''scale a stack of images so all have the same total intensity 
    and intensities between [0,1]'''
    imstack = deepcopy(imstack)
    for im in imstack: 
        im -= np.min(im)

    tots = np.sum(imstack, axis = (1,2))
    t = max(tots) / tots
    for i in range(len(tots)):
        imstack[i] *= t[i]
    return imstack/np.max(imstack)


def scale_array(array):
    ''' scale an array so all values in [0,1]'''
    array -= np.min(array)
    array /= np.max(array)
    return array



###################################################
# Various display functions, some useful some not # 
###################################################


def show_im(im, title=None):
    ''' Displays an image on a new axis'''
    fig,ax = plt.subplots()
    ax.matshow(im, cmap = 'gray', origin = 'upper')
    if title is not None: 
        ax.set_title(str(title))
    plt.show()


def show_stack(images, ptie = None):
    '''Shows a stack of dm3s or np images with a slider to navigate slice axis'''
    sig = False
    if type(images[0]) == hyperspy._signals.signal2d.Signal2D:
        sig = True
        imstack = []
        defvals = []
        for signal2D in images:
            imstack.append(signal2D.data)
            defvals.append(signal2D.metadata.General.title)
        images = np.array(imstack)
    else:
        images = np.array(images)

    if ptie is None:
        t , b = 0, images[0].shape[0]
        l , r = 0, images[0].shape[1]
    else:
        t = ptie.crop['top']
        b = ptie.crop['bottom']
        l = ptie.crop['left']
        r = ptie.crop['right']

    images = images[:,t:b,l:r]

    fig, ax = plt.subplots()
    N = images.shape[0]
    def view_image(i=0):
        plt.imshow(images[i], cmap='gray', interpolation='nearest', origin='upper')
        if sig:
            plt.title('Defocus: {:}'.format(defvals[i]))
        else:
            plt.title('Stack[{:}]'.format(i))
    interact(view_image, i=(0, N-1))


def show_scaled(im, title = None):
    ''' Shows an image with intensity scaled. 
    Useful for looking at images before they have a median filter applied'''
    mean = np.mean(im)
    std = np.std(im)

    low = max(np.min(im), mean - 8*std)
    high = min(np.max(im), mean + 8*std)

    fig,ax = plt.subplots()
    ax.matshow(im, cmap = 'gray', vmin=low, vmax=high)
    if title is not None:
        ax.set_title(str(title))
    plt.show()


def get_fft(im):
    '''Returns shifted fast forier transform of an image.'''
    return np.fft.fftshift(np.fft.fft2(im))


def get_ifft(fft):
    '''Returns inverse of a shifted fft'''
    return np.real(np.fft.ifft2(np.fft.ifftshift(fft)))


def show_fft(fft, title=None):
    '''Given an fft this displays the log of that fft using matplot lib.'''
    fig, ax = plt.subplots()
    display = np.where(np.abs(fft)!=0,np.log(np.abs(fft)),0)
    ax.matshow(display,cmap='gray')
    if title is not None:
        ax.set_title(str(title))
    plt.show()
