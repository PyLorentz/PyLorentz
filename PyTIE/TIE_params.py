"""Class for TIE reconstruction parameters. 

A class file for holding TIE images and reconstruction parameters.
Also has several methods relating to doing the reconstruction, namely making
masks and interactively cropping the image stacks. 

AUTHOR: 
Arthur McCray, ANL, Summer 2019
--------------------------------------------------------------------------------
"""

import numpy as np
from scipy import constants, ndimage
import hyperspy #just to check signal type
import hyperspy.api as hs
import os
import textwrap


class TIE_params(object):
    """An object for holding the data and parameters for the reconstruction.

    Some params will be obtained directly from the metadata, others given by the
    user. Also holds some other useful values. An instance is created by the 
    load_data() function in TIE_helper.py. 

    For more information on how to organize the directory and load the data, as 
    well as how to setup the .fls file please refer to the README or the 
    TIE_template.ipynb notebook. 

    Attributes: 
        dm3stack: A list of hyperspy Signal2D objects, one per image in the 
            through focus series (tfs). 
        flip_dm3stack: A list of hyperspy Signal2D objects for the flip tfs if
            there is one. 
        defvals: List of defocus values in nm from least to most defocus. This 
            assumes a symmetric defocus over/under, so expects 2 values for a 5
            image tfs. 
        flip: Boolean for whether or not to reconstruct using the flip stack.  
            Even if the flip stack exists the reconstruction will only use the 
            unflip stack if self.flip = False. 
        data_loc: String for location of data folder. 
        sim: Bool. True if data is simulated, minimizes mask. 
        num_files: Equiv to len(self.dm3stack)
        axes: Hyperspy axes_manager from unflip infocus image. Contains useful 
            information such as scale if loaded from dm3. 
        shape: Shape of original image data (y, x) to be consistent with numpy
        infocus: Averaged infocus image between unflip and flip stacks. If no 
            flip stack, just unflip infocus image. 
        qi: inverse laplacian helper, used in TIE_reconstruct
        roi: hyperspy region-of-interest object. Initialized to central quarter 
            of image, but changes interactively with self..select_region() 
            method. Values for the roi are in nm, and continue to change as
            the interactive region is adjusted. 
        crop: Region of interest in pixels used to select the are to be 
            reconstructed. self.roi is converted from nm to pixels with the 
            self..crop_dm3s() method, and does not change as the roi is 
            adjusted. Initialized to full image. 
        mask: Binary mask made form all the images. 1 where all images have
            nonzero data, 0 where any do not. Made by self.make_mask()
    """
    def __init__(self, dm3stack=None, flip_dm3stack=[], defvals=None, flip=None, data_loc=None, sim=False):
        """Inits TIE_params object. dm3stack, defvals must be specified at minimum.

        Args: 
            dm3stack: List of hyperspy Signal2D objects, or numpy arrays containing 
                image data which will be converted to Signal2D. One per image in
                the through focus series (tfs). 
            flip_dm3stack: A list of hyperspy Signal2D objects for the flip tfs 
                if there is one. 
            defvals: List of defocus values in nm from least to most defocus. 
                This assumes a symmetric defocus over/under, e.g. expects 2 
                values for a 5 image tfs. 
            flip: Boolean for whether or not to reconstruct using the flip stack.  
                Even if the flip stack exists the reconstruction will only use 
                the unflip stack if self.flip = False. 
            data_loc: String for location of data folder.
            sim: Bool. True if data is simulated, minimizes mask. 
        """
        if type(dm3stack) == list:
            pass
        else:
            try:
                if np.ndim(dm3stack) != 3:
                    dm3stack = [dm3stack]
            except:
                pass
        self.dm3stack = dm3stack
        self.flip_dm3stack = flip_dm3stack
        if type(defvals) is not list and type(defvals) is not np.ndarray:
            self.defvals = [defvals]
        else:
            self.defvals = defvals # array of the defocus steps.

        self.num_files = len(self.dm3stack)
        if self.num_files == 1:
            assert len(self.defvals) == 1
        else:
            assert self.num_files == 2*len(self.defvals)+1 # confirm they match

        if type(dm3stack[0]) != hyperspy._signals.signal2d.Signal2D: 
            # images loaded are tifs, conver to dm3s
            ndm3stack = []
            for arr in dm3stack:
                ndm3stack.append(hs.signals.Signal2D(arr))
            self.dm3stack = ndm3stack
            if list(flip_dm3stack):
                nflipdm3stack = []
                for arr in flip_dm3stack:
                    nflipdm3stack.append(hs.signals.Signal2D(arr))
                self.flip_dm3stack = nflipdm3stack
            print("Data not given in hyperspy signal class. You likely need to set ptie.scale (nm/pix).")

        infocus = self.dm3stack[self.num_files//2] # unflip infocus dm3
        self.axes = infocus.axes_manager # dm3 axes manager
        self.shape = (self.axes.shape[1], self.axes.shape[0]) # to be consistent with np
        scale_y = self.axes[0].scale # pixel size (nm/pix)
        scale_x = self.axes[1].scale 
        assert scale_y == scale_x
        self.scale = scale_y
        print('Given scale: {:.4f} nm/pix\n'.format(self.dm3stack[0].axes_manager[0].scale))

        if flip is not None:
            self.flip = flip
        elif flip_dm3stack:
            self.flip = True
        else:
            self.flip = False

        if data_loc:
            if not data_loc.endswith('/'):
                data_loc += '/'
            self.data_loc = data_loc
        else:
            self.data_loc = None 

        if flip: 
            assert len(self.dm3stack) == len(self.flip_dm3stack)
            flip_infocus = self.flip_dm3stack[self.num_files//2]
            self.infocus = (infocus.data + flip_infocus.data)/2 
            # An averaged infocus image between the flip/unflip stack.
        else:
            self.infocus = np.copy(infocus.data)

        self.qi = np.zeros(self.shape) # will be inverse Laplacian array
        # Default to central square for ROI
        self.roi = hs.roi.RectangularROI(left= self.shape[1]//4*self.scale, 
                                         right=3*self.shape[1]//4*self.scale, 
                                         top=self.shape[0]//4*self.scale, 
                                         bottom=3*self.shape[0]//4*self.scale)
        # Default to full image for crop, (remember: bottom > top, right > left)
        self.crop = {'top'   : 0,             
                     'bottom': self.shape[0],
                     'left'  : 0,
                     'right' : self.shape[1]}
        if sim: 
            self.mask = np.ones(self.shape)
            self.mask[0,0] = 0 # doesnt deal well with all white mask
        else:
            self.make_mask()


    def pre_Lap(self, pscope, def_step=1):
        """ Scaling prefactor used in the TIE reconstruction.

            Args:
                pscope: Microscope object from microscopes.py
                def_step: The defocus value for which is being reconstructed. If
                    using a longitudinal derivative, def_step should be 1. 
            
            Returns: Float
        """
        return -1 * self.scale**2 / (16 * np.pi**3 * pscope.lam * def_step) 

    def make_mask(self, dm3stack = None, threshhold=0):
        """Sets self.mask to be a binary bounding mask from dm3stack and flip_dm3stack. 
        
        Makes all images binary using a threshhold value, and then 
        multiplies these arrays. Can also take a stack of images that aren't 
        signal2D. 

        The inverse laplacian reconstruction does not deal well with a mask that 
        is all ones, that is accounted for in TIE() function rather than here. 

        Optional Args: 
            dm3stack: List. List of arrays or Signal2D objects from which to make mask
                Default will use self.dm3stack and self.flip_dm3stack
            threshhold: Float. Pixel value with which to threshhold the images. 
                Default is 0. 

        Returns: None
        """
        if len(self.dm3stack) == 1: # SITIE params
            self.mask = np.ones(self.shape)
            return
        if dm3stack is None:
            dm3stack = self.dm3stack + self.flip_dm3stack
        shape = np.shape(dm3stack[0].data)
        mask = np.ones(shape)

        if type(dm3stack[0]) == hyperspy._signals.signal2d.Signal2D:
            for sig in dm3stack:
                im_mask = np.where(sig.data <= threshhold, 0, 1)
                mask *= im_mask
        else: # assume they're images
            for im in dm3stack:
                im_mask = np.where(im <= threshhold, 0, 1)
                mask *= im_mask

        # shrink mask slightly 
        its = min(15, self.shape[0]//250, self.shape[1]//250)
        if its >= 1: # binary_erosion fails if iterations=0
            mask = ndimage.morphology.binary_erosion(mask, iterations = its)
        mask = mask.astype(float, copy = False)
        # apply a light filter to the edges
        mask = ndimage.gaussian_filter(mask,2)
        self.mask = mask
        self.infocus *= mask
        return

    def select_region(self, infocus=True):
        """ Interactively crop dm3stack to smaller size. 

        This method sets self.roi to be the region (square or otherwise) as
        selected by the user. Default is center region.

        Args:
            infocus: Bool. If True, will display the infocus image for selecting
                a sub-region. If False, will display a defocused image (this is 
                useful for datasets which have no in-focus contrast). 

        Returns: None
        """
        if infocus:
            display_sig = self.dm3stack[self.num_files//2].deepcopy()
        else:
            display_sig=self.dm3stack[0].deepcopy()

        dimy, dimx = self.shape
        scale = self.scale
        # reset roi to central square
        roi = hs.roi.RectangularROI(left= dimx//4*scale, right=3*dimx//4*scale, 
                            top=dimy//4*scale, bottom=3*dimy//4*scale)
        display_sig.plot()
        roi2D = roi.interactive(display_sig, color="blue")
        self.roi = roi


    def crop_dm3s(self): 
        """ Sets self.crop in pixels as region to be reconstructed. 

        Converts self.roi (in units of nm) to pixels and asks for user input if
        this an acceptable shape. 

        Input options: 
            y: sets self.crop
            n: does not set self.crop
            reset: sets self.crop to be full image size. (This is the default 
                initialized value.)

        Crops the full dm3 + flip_dm3 stack to the specified shape as defined by
        roi (hyperspy region of interest). Adjusts other axes accordingly. 
        """
        if self.roi is None:
            print('No region previously selected, defaulting to central square.')
            dimy, dimx = self.shape
            scale = self.scale
            self.roi = hs.roi.RectangularROI(left= dimx//4*scale, right=3*dimx//4*scale, 
                                top=dimy//4*scale, bottom=3*dimy//4*scale)

        left = int(self.roi.left/self.scale)
        right = int(self.roi.right/self.scale)
        top = int(self.roi.top/self.scale)
        bottom = int(self.roi.bottom/self.scale) 

        print('The new images will be cropped (in pixels)')
        print(F'left: {left} , right: {right} , top: {top} , bottom: {bottom}')
        print(F"New dimensions will be: ({bottom-top}, {right-left})")
        print()

        proceed = input("""Does this work? (y/n):\nOr you can reset to the original full images (reset):\n""")
        while proceed != 'y':
            if proceed == 'n':
                print("Okay, change the region and run this again.")
                return
            elif proceed == 'reset':
                self.crop['left'] = 0
                self.crop['right'] = self.shape[1]
                self.crop['top'] = 0
                self.crop['bottom'] = self.shape[0]
                print("The region has been returned to the full image.")
                return 
            else:
                proceed = input("Please respond with 'y' , 'n' , or 'reset'.\n")

        self.crop['left'] = left
        self.crop['right'] = right
        self.crop['top'] = top
        self.crop['bottom'] = bottom
        
        print(textwrap.dedent(f"""
            Your images are now shape: ({bottom-top}, {right-left})
            Changes can continue to be made by moving/updating the region,
            but you have to run this again for them to take affect.\n"""))
        return

### End ### 