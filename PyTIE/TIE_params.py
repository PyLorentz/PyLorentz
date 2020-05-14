#!/usr/bin/python
#
# A class file for holding TIE reconstruction parameters.
# Also has several functions relating to doing the reconstruction, namely making
# masks and interactively cropping the image stacks. 
# 
# Arthur McCray, ANL, Summer 2019
#
#-------------------------------------------------------------------------------

import numpy as np
from scipy import constants, ndimage
import hyperspy #just to check signal type
import hyperspy.api as hs
import os

class TIE_params(object):
    '''
    An object for holding all the parameters for the reconstruction. 
    Some params will be obtained directly from the metadata, others given by user. 
    Also holds some other useful values.
    '''
    def __init__(self, dm3stack, flip_dm3stack, defvals, flip = False, data_loc=None, SITIE = False):
        ### Data and related parameters ###
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
        self.SITIE = SITIE 
        if SITIE:
            self.num_files = 1
        else:
            self.num_files = 2*len(self.defvals) + 1 # total number images 

        if type(dm3stack[0]) != hyperspy._signals.signal2d.Signal2D: 
            # hyperspy is slow to load, so it would be great to find another
            # way to check this. 
            ndm3stack = []
            for arr in dm3stack:
                ndm3stack.append(hs.signals.Signal2D(arr))
            self.dm3stack = ndm3stack
            if list(flip_dm3stack):
                nflipdm3stack = []
                for arr in flip_dm3stack:
                    nflipdm3stack.append(hs.signals.Signal2D(arr))
                self.flip_dm3stack = nflipdm3stack
            print("Data not given in hyperspy signal class.")
            print("You likely need to set ptie.scale (nm/pix).")

        infocus = self.dm3stack[self.num_files//2] # unflip infocus dm3
        self.metadata = infocus.metadata 
        self.axes = infocus.axes_manager # dm3 axes manager
        self.shape = (self.axes.shape[1], self.axes.shape[0]) # to be consistent with np, shape of orig data
        scale_y = self.axes[0].scale # pixel size (nm/pix)
        scale_x = self.axes[1].scale 
        assert scale_y == scale_x
        self.scale = scale_y
        print('Given scale: {:.4f} nm/pix'.format(self.dm3stack[0].axes_manager[0].scale))

        self.flip = flip # whether or not a flipped tfs is included
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
        self.roi = None # If smaller area is selected, this is the region (in nm)
        self.crop = {'top'   : 0,             # region that will be reconstructed 
                     'bottom': self.shape[0], # initially full image
                     'left'  : 0,
                     'right' : self.shape[1]} # bottom > top, right > left
        self.make_mask()



    def pre_Lap(self, pscope, def_step):
        'prefactor for laplacian reconstruction' 
        return -1 * self.scale**2 / (16 * np.pi**3 * pscope.lam * def_step) 

    def make_mask(self, dm3stack = None, threshhold=0):
        '''Make a binary bounding mask from dm3stack and flip_dm3stack. 
        Can also take a stack of images. '''
        if self.SITIE:
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
        its = min(15, self.shape[0]//100, self.shape[1]//100)
        mask = ndimage.morphology.binary_erosion(mask, iterations = its)
        mask = mask.astype(float, copy = False)
        # apply a light filter to the edges
        mask = ndimage.gaussian_filter(mask,2)
        self.mask = mask
        if not self.SITIE:
            self.infocus *= mask
        return

    def select_region(self, infocus=True):
        '''interactively crop dm3stack to smaller size. 
        Default is center region.
        infocus = bool -- whether to use an infocus or defocused image for the selection
        '''
        # updating the axes here in case load was from tiffs and scale was set
        # for the tie_params object. Doesn't exactly ned to be here... 
        # for sig in self.dm3stack + self.flip_dm3stack: 
        #     sig.axes_manager[0].scale = self.scale
        #     sig.axes_manager[1].scale = self.scale

        if self.SITIE:
            infocus_im = self.dm3stack[0].deepcopy()
        elif infocus: 
            if len(self.dm3stack) == self.num_files:
                infocus_im = self.dm3stack[self.num_files//2].deepcopy()
            elif len(self.dm3stack) == 2*self.num_files:
                infocus_im = self.dm3stack[self.num_files//4].deepcopy()
            else:
                print("Error in TIE_params select_region()")
                return 1
        else: 
            infocus_im=self.dm3stack[0].deepcopy()

        infocus_im.data = infocus_im.data 
        dimy, dimx = self.shape
        scale = self.scale

        roi = hs.roi.RectangularROI(left= dimx//4*scale, right=3*dimx//4*scale, 
                            top=dimy//4*scale, bottom=3*dimy//4*scale)
        infocus_im.plot()
        roi2D = roi.interactive(infocus_im, color="blue")
        self.roi = roi


    def crop_dm3s(self): 
        ''' 
        Crops the full dm3 + flip_dm3 stack to the specified shape as defined by
        roi (hyperspy region of interest). Adjusts other axes accordingly. 
        '''
        import textwrap
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
        
        # for signal in self.dm3stack:
        #     signal.crop(0, left, right)
        #     signal.crop(1, top, bottom)         
        # if self.flip_dm3stack is not None:         
        #     for signal in self.flip_dm3stack:
        #         signal.crop(0, left, right)
        #         signal.crop(1, top, bottom)

        # # update axes stuff
        # self.axes = self.dm3stack[0].axes_manager
        # self.shape = (self.axes.shape[1], self.axes.shape[0])
        # self.qi = np.zeros(self.shape) 
        # # update mask 
        # mask = self.mask[top:bottom,left:right]
        # # it doesn't handle an all-white mask very well
        # if np.max(mask) == np.min(mask):
        #     mask = ndimage.morphology.binary_erosion(mask)

        # self.mask = mask.astype('float') #not sure if i need to make sure type here


        print(textwrap.dedent(f'''
            Your images are now: ({bottom-top}, {right-left})
            Changes can continue to be made by moving/updating the region,
            but you have to run this again for them to take affect.\n'''))