"""Helper functions for TIE. 

PURPOSE:
An assortment of helper functions broadly divided into two sections. First for 
loading images, passing that data, and helping with the reconstruction; second
a set of functions helpful for displaying images. 

AUTHOR:
Arthur McCray, ANL, Summer 2019.
--------------------------------------------------------------------------------
"""
 
import matplotlib.pyplot as plt 
import numpy as np
import hyperspy.api as hs
from skimage import io
from scipy.ndimage.filters import median_filter
from ipywidgets import interact
import hyperspy # just for checking type in show_stack. 
from copy import deepcopy

from TIE_params import TIE_params


#######################################################
# Functions used for loading and passing the TIE data # 
#######################################################

def load_data(path=None, fls_file='', al_file='', flip=None, flip_fls_file=None, filtersize=3): 
    """Load files in a directory (from a .fls file) using hyperspy. 

    For more information on how to organize the directory and load the data, as 
    well as how to setup the .fls file please refer to the README or the 
    TIE_template.ipynb notebook. 

    Args: 
        path: String. Location of data directory. 
        fls_file: String. Name of the .fls file which contains the image names 
            and defocus values. 
        al_file: String. Name of the aligned stack image file. 
        flip: Bool. Is there a flip stack? If false, it will not assume a 
            uniformly thick film and not reconstruct electrostatic phase shift.
    Optional Args: 
        flip_fls_file: String. Name of the .fls file for the flip images if they
            are not named the same as the unflip files. Will only be applied to 
            the /flip/ directory. 
        filtersize: Int. The images are processed with a median filter to remove
            hot pixels which occur in experimental data. This should be set to 0
            for simulated data, though generally one would only use this 
            function for experimental data. 
    
    Returns: 
        dm3stack: array of hyperspy signal2D objects (one per image)
        flip_dm3stack: array of hyperspy signal2D objects, only if flip
        ptie: TIE_params object holding a reference to the dm3stack and many
            useful parameters.
    """

    unflip_files = []
    flip_files = []

    if not fls_file.endswith('.fls'):
        fls_file += '.fls'

    if flip_fls_file is None: # one fls file given
        fls = []
        with open(path + fls_file) as file:
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
            return 1
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

        assert int(fls[0]) == int(flip_fls[0])
        num_files = int(fls[0])
        for line in fls[1:num_files+1]:
            unflip_files.append(path + 'unflip/' + line)
        for line in flip_fls[1:num_files+1]:
            flip_files.append(path + 'flip/' + line)

    # Actually load the data using hyperspy
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
    # The data from the dm3's will be replaced with the aligned image data. 
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
        # pull slices from correct axis, assumes fewer slices than images are tall
        if al_tifs.shape[0] < al_tifs.shape[2]:
            im = al_tifs[i]
        elif al_tifs.shape[0] > al_tifs.shape[2]:
            im = al_tifs[:,:,i]
        else:
            print("Bad stack\n Or maybe the second axis is slice axis?")
            print('Loading failed.\n')
            return 1
        
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

    # Create a TIE_params object
    ptie = TIE_params(dm3stack, flip_dm3stack, defvals, flip, path)
    print('Data loaded successfully.\n')
    return (dm3stack, flip_dm3stack, ptie)


def select_tifs(i, ptie, long_deriv = False):
    """Returns a list of the images which will be used in TIE() or SITIE().

    Uses copy.deepcopy() as the data will be modified in the reconstruction 
    process, and we don't want to change the original data. This method is 
    likely not best practice. 

    In the future this might get moved to the TIE_params class. 

    Args: 
        i: Int. Index of defvals for which to select the tifs. 
        ptie: TIE_params object. 

    
    Returns: 
        List of np arrays, return depends on parameters:
        if long_deriv = True:
            returns all images in dm3stack followed by all images in flip_dm3stack
        if ptie.flip: 
            returns [ +- , -- , 0 , ++ , -+ ]
            first +- is unflip/flip, and second +- is over/underfocus
            0 is averaged infocus image
        else:
            returns [+-, 0, ++]
        For a 3-point derivative the images are returned
    """
    if long_deriv:
        recon_tifs = []
        for sig in ptie.dm3stack:
            recon_tifs.append(sig.data)
        if ptie.flip:
            for sig in ptie.flip_dm3stack:
                recon_tifs.append(sig.data)

    else:
        if i < 0:
            i = len(ptie.defvals)+i
            print('new i: ', i)
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
        recon_tifs = deepcopy(recon_tifs) 
    except TypeError:
        print("TypeError in select_tifs deepcopy. Proceeding with originals.")
    return recon_tifs


def dist(ny,nx):
    """Implementation of the IDL DIST function. 

    Returns a rectangular array in which the value of each element is 
    proportional to its frequency. This is equivalent to an array where each 
    value is smallest distance to a corner (measured from upper left corner of 
    pixel). This is used for Fourier processing the inverse Laplacian operator. 

    Args: 
        ny: Int. Height of array 
        nx: Int. Width of array

    Returns: 
        numpy array of shape (ny, nx). 
    """
    axisy = np.linspace(-ny//2+1, ny//2, ny)
    axisx = np.linspace(-nx//2+1, nx//2, nx)
    result = np.sqrt(axisx**2 + axisy[:,np.newaxis]**2)
    return np.roll(np.roll(result, ny//2+1, axis=0), nx//2+1, axis = 1)


def scale_stack(imstack):
    """Scale a stack of images so all have the same total intensity. 

    A helper function used in TIE_reconstruct. Scales each image in a stack to
    have the same total intensity, with the minimum and maximum across all 
    images being 0 and 1.  
    
    Args: 
        imstack: List. List of 2D arrays. 

    Returns:
        List of same shape as imstack
    """

    imstack = deepcopy(imstack)
    minv = np.min(imstack)
    for im in imstack: 
        im -= minv

    tots = np.sum(imstack, axis = (1,2))
    t = max(tots) / tots
    for i in range(len(tots)):
        imstack[i] *= t[i]
    return imstack/np.max(imstack)



###################################################
#            Various display functions            # 
###################################################
""" Not all of these are used in TIE_reconstruct, but I often find them useful
to have handy when working in Jupyter notebooks."""


def show_im(im, title=None):
    """Display an image on a new axis.
    
    Takes a 2D array and displays the image in grayscale with optional title on 
    a new axis. In general it's nice to have things on their own axes, but if 
    too many are open it's a good idea to close with plt.close('all'). 

    Args: 
        im: 2D array or list. Image to be displayed.
    Keyword Args: 
        title: String. Title of plot. 
    
    Returns:
        Nothing
    """
    fig,ax = plt.subplots()
    ax.matshow(im, cmap = 'gray', origin = 'upper')
    if title is not None: 
        ax.set_title(str(title))
    plt.show()
    return


def show_stack(images, ptie = None):
    """Shows a stack of dm3s or np images with a slider to navigate slice axis. 
    
    Uses ipywidgets.interact to allow user to view multiple images on the same
    axis using a slider. There is likely a better way to do this, but this was 
    the first one I found that works... 

    If a TIE_params object is given, only the regions corresponding to ptie.crop
    will be shown. 

    Args:
        images: List of 2D arrays. Stack of images to be shown. 
    Keyword Args:
        ptie: TIE_params object. Will use ptie.crop to show only the region that
            will be cropped. 

    Returns:
        Nothing. 
    """
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
    return 

def show_scaled(im, title = None):
    """ Shows an image with intensity scaled. 

    Useful for looking at images before they have a median filter applied.

    Args: 
        im: 2D array or list. Image to be displayed.
    Keyword Args: 
        title: String. Title of plot. 
    
    Returns:
        Nothing
    """
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
    """Returns shifted fast forier transform of an image."""
    return np.fft.fftshift(np.fft.fft2(im))


def get_ifft(fft):
    """Returns real portion of inverse of a shifted fft."""
    return np.real(np.fft.ifft2(np.fft.ifftshift(fft)))


def show_fft(fft, title=None):
    """Given an fft this displays the log of that fft."""
    fig, ax = plt.subplots()
    display = np.where(np.abs(fft)!=0,np.log(np.abs(fft)),0)
    ax.matshow(display,cmap='gray')
    if title is not None:
        ax.set_title(str(title))
    plt.show()
    return
