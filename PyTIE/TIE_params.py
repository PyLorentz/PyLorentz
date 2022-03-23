"""Class for TIE reconstruction parameters.

A class for holding TIE images and reconstruction parameters.
Also has several methods relating to doing the reconstruction, namely making
masks and interactively cropping the image stacks.

AUTHOR:
Arthur McCray, ANL, Summer 2019
"""

import numpy as np
from scipy import constants, ndimage
import hyperspy  # just to check signal type
import hyperspy.api as hs
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
        imstack (list): A list of hyperspy Signal2D objects, one per image in
            the through focus series (tfs).
        flipstack (list): A list of hyperspy Signal2D objects for the flip tfs
            if there is one.
        defvals (list): List of defocus values in nm from least to most defocus.
            This assumes a symmetric defocus over/under, so expects 2 values for
            a 5 image tfs.
        flip (bool): Boolean for whether or not to reconstruct using the flip
            stack.  Even if the flip stack exists the reconstruction will only
            use the unflip stack if self.flip = False.
        data_loc (str): String for location of data folder.
        no_mask (bool): Eliminates mask (generally used with simulated data).
        num_files (int): Equiv to len(self.imstack)
        axes (``hyperspy.signal2D.axes_manager``): Hyperspy axes_manager from unflip
            infocus image. Contains useful information such as scale if loaded
            from dm3.
        shape (tuple): Shape of original image data (y, x)
        scale (float): Scale of images (nm/pixel). Taken from the dm3 metadata
            or set with the ``set_scale()`` method.
        rotation (float, int): The rotation to apply to the image before reconstruction in deg.
        x_transl (int): The x_translation to apply to the image before reconstruction in pix.
        y_transl (int): The y_translation to apply to the image before reconstruction in pix.
        infocus (2D array): Averaged infocus image between unflip and flip
            stacks. If no flip stack, just unflip infocus image.
        qi (2D array): 2D inverse frequency array, possibly modified with
            Tikhonov filter.
        roi (``hyperspy.roi.RectangularROI``): hyperspy region-of-interest object.
            Initialized to central quarter of image, but changes interactively
            with self.select_region() method. Values for the roi are in nm, and
            continue to change as the interactive region is adjusted.
        crop (tuple): Region of interest in pixels used to select the are to be
            reconstructed. self.roi is converted from nm to pixels with the
            self.crop_ims() method, and does not change as the roi is
            adjusted. Initialized to full image.
        mask (2D array): Binary mask made form all the images. 1 where all images have
            nonzero data, 0 where any do not. Made by self.make_mask()
    """

    def __init__(
        self,
        imstack=None,
        flipstack=[],
        defvals=None,
        flip=None,
        data_loc=None,
        no_mask=False,
        v=1,
    ):
        """Constructs TIE_params object. imstack, defvals must be specified at minimum.

        Flipstack, flip, no_mask, and v are optional arguments.
        v (int): Verbosity.
            - 0 : No output
            - 1 : Default output
        """
        vprint = print if v >= 1 else lambda *a, **k: None

        if type(imstack) == list:
            pass
        else:
            try:
                if np.ndim(imstack) != 3:
                    imstack = [imstack]
            except:
                pass
        self.imstack = imstack
        self.flipstack = flipstack
        if type(defvals) is not list and type(defvals) is not np.ndarray:
            self.defvals = [defvals]
        else:
            self.defvals = defvals  # array of the defocus steps.

        self.num_files = len(self.imstack)
        if self.num_files == 1:
            assert len(self.defvals) == 1
        else:
            assert self.num_files == 2 * len(self.defvals) + 1  # confirm they match

        if type(imstack[0]) != hyperspy._signals.signal2d.Signal2D:
            # images loaded are tifs, conver to dm3s
            nimstack = []
            for arr in imstack:
                nimstack.append(hs.signals.Signal2D(arr))
            self.imstack = nimstack
            if list(flipstack):
                nflipimstack = []
                for arr in flipstack:
                    nflipimstack.append(hs.signals.Signal2D(arr))
                self.flipstack = nflipimstack
            vprint(
                "Data not given in hyperspy signal objects. You likely need to set ptie.scale (nm/pix)."
            )

        infocus = self.imstack[self.num_files // 2]  # unflip infocus dm3
        self.axes = infocus.axes_manager  # dm3 axes manager
        self.shape = (
            self.axes.shape[1],
            self.axes.shape[0],
        )  # to be consistent with np
        scale_y = self.axes[0].scale  # pixel size (nm/pix)
        scale_x = self.axes[1].scale
        assert scale_y == scale_x
        self.scale = scale_y
        scale_units = self.axes[0].units
        for sig in (
            self.imstack + self.flipstack
        ):  # just to make sure they all have same scale
            sig.axes_manager[0].units = scale_units
            sig.axes_manager[1].units = scale_units
            sig.axes_manager[0].scale = self.scale
            sig.axes_manager[1].scale = self.scale
        self.rotation, self.x_transl, self.y_transl = (
            0,
            0,
            0,
        )  # The rotation/translation to apply to images.

        vprint(
            "Given scale: {:.4f} nm/pix\n".format(self.imstack[0].axes_manager[0].scale)
        )

        if flip is not None:
            self.flip = flip
        elif flipstack:
            self.flip = True
        else:
            self.flip = False

        if data_loc:
            if not data_loc.endswith("/"):
                data_loc += "/"
            self.data_loc = data_loc
        else:
            self.data_loc = "./"

        if flip:
            assert len(self.imstack) == len(self.flipstack)
            flip_infocus = self.flipstack[self.num_files // 2]
            self.infocus = (infocus.data + flip_infocus.data) / 2
            # An averaged infocus image between the flip/unflip stack.
        else:
            self.infocus = np.copy(infocus.data)

        self.qi = np.zeros(self.shape)  # will be inverse Laplacian array
        # Default to central square for ROI
        self.roi = hs.roi.RectangularROI(
            left=self.shape[1] // 4 * self.scale,
            right=3 * self.shape[1] // 4 * self.scale,
            top=self.shape[0] // 4 * self.scale,
            bottom=3 * self.shape[0] // 4 * self.scale,
        )
        # Default to full image for crop, (remember: bottom > top, right > left)
        self.crop = {
            "top": 0,
            "bottom": self.shape[0],
            "left": 0,
            "right": self.shape[1],
        }
        if no_mask:
            self.mask = np.ones(self.shape)
        else:
            self.make_mask()

    def pre_Lap(self, pscope, def_step=1):
        """Scaling prefactor used in the TIE reconstruction.

        Args:
            pscope (``Microscope`` object): Microscope object from
                microscopes.py
            def_step (float): The defocus value for which is being
                reconstructed. If using a longitudinal derivative, def_step
                should be 1.

        Returns:
            float: Numerical prefactor
        """
        return -1 * self.scale ** 2 / (16 * np.pi ** 3 * pscope.lam * def_step)

    def make_mask(self, imstack=None, threshold=0):
        """Sets self.mask to be a binary bounding mask from imstack and flipstack.

        Makes all images binary using a threshold value, and then
        multiplies these arrays. Can also take a stack of images that aren't
        signal2D.

        The inverse Laplacian reconstruction does not deal well with a mask that
        is all ones, that is accounted for in TIE() function rather than here.

        Args:
            imstack (list): (`optional`) List of arrays or Signal2D objects from
                which to make mask Default will use self.imstack and
                self.flipstack
            threshold (float): (`optional`) Pixel value with which to
                threshold the images. Default is 0.

        Returns:
            None. Assigns result to self.mask()
        """
        if len(self.imstack) == 1:  # SITIE params
            self.mask = np.ones(self.shape)
            return
        if imstack is None:
            imstack = self.imstack + self.flipstack
        shape = np.shape(imstack[0].data)
        mask = np.ones(shape)

        if type(imstack[0]) == hyperspy._signals.signal2d.Signal2D:
            for sig in imstack:
                im_mask = np.where(np.abs(sig.data) <= threshold, 0, 1)
                mask *= im_mask
        else:  # assume they're images
            for im in imstack:
                im_mask = np.where(np.abs(im) <= threshold, 0, 1)
                mask *= im_mask

        # shrink mask slightly
        its = int(min(15, self.shape[0] // 250, self.shape[1] // 250))
        if its >= 1:  # binary_erosion fails if iterations=0
            mask = ndimage.morphology.binary_erosion(mask, iterations=its)
        mask = mask.astype(float, copy=False)
        # apply a light filter to the edges
        mask = ndimage.gaussian_filter(mask, 2)
        self.mask = mask
        self.infocus *= mask
        return

    def select_region(self, infocus=True):
        """Interactively crop imstack to smaller size.

        This method sets self.roi to be the region (square or otherwise) as
        selected by the user. Default is central quarter of image.

        Args:
            infocus (bool): If True, will display the infocus image for
                selecting a sub-region. If False, will display a defocused image
                this is useful for datasets which have no in-focus contrast).

        Returns:
            None
        """
        if infocus:
            display_sig = self.imstack[self.num_files // 2].deepcopy()
        else:
            display_sig = self.imstack[0].deepcopy()

        if self.rotation != 0 or self.x_transl != 0 or self.y_transl != 0:
            rotate, x_shift, y_shift = self.rotation, self.x_transl, self.y_transl
            display_sig.data = ndimage.rotate(
                display_sig.data, rotate, reshape=False, order=0
            )
            display_sig.data = ndimage.shift(
                display_sig.data, (-y_shift, x_shift), order=0
            )

        dimy, dimx = self.shape
        scale = self.scale

        # reset roi to central square
        roi = hs.roi.RectangularROI(
            left=dimx // 4 * scale,
            right=3 * dimx // 4 * scale,
            top=dimy // 4 * scale,
            bottom=3 * dimy // 4 * scale,
        )

        # roi = hs.roi.RectangularROI(left=  59, right=3*dimx//4*scale,
        #                     top= 59, bottom=3*dimy//4*scale)
        display_sig.plot()
        roi2D = roi.interactive(display_sig, color="blue")
        self.roi = roi

    def crop_ims(self):
        """Sets self.crop in pixels as region to be reconstructed.

        Converts self.roi (in units of nm) to pixels and asks for user input if
        this an acceptable shape.

        Input options:

            - "y": sets self.crop
            - "n": does not set self.crop
            - "reset": sets self.crop to be full image size. (This is the default
              initialized value.)

        Crops the full dm3 + flip_dm3 stack to the specified shape as defined by
        roi (hyperspy region of interest). Adjusts other axes accordingly.
        """
        if self.roi is None:
            print("No region previously selected, defaulting to central square.")
            dimy, dimx = self.shape
            scale = self.scale
            self.roi = hs.roi.RectangularROI(
                left=dimx // 4 * scale,
                right=3 * dimx // 4 * scale,
                top=dimy // 4 * scale,
                bottom=3 * dimy // 4 * scale,
            )

        left = int(self.roi.left / self.scale)
        right = int(self.roi.right / self.scale)
        top = int(self.roi.top / self.scale)
        bottom = int(self.roi.bottom / self.scale)

        print("The new images will be cropped (in pixels)")
        print(f"left: {left} , right: {right} , top: {top} , bottom: {bottom}")
        print(f"New dimensions will be: ({bottom-top}, {right-left})")
        print()

        proceed = input(
            """Does this work? (y/n):\nOr you can reset to the original full images (reset):\n"""
        )
        while proceed != "y":
            if proceed == "n":
                print("Okay, change the region and run this again.")
                return
            elif proceed == "reset":
                self.crop["left"] = 0
                self.crop["right"] = self.shape[1]
                self.crop["top"] = 0
                self.crop["bottom"] = self.shape[0]
                print("The region has been returned to the full image.")
                return
            else:
                proceed = input("Please respond with 'y' , 'n' , or 'reset'.\n")

        self.crop["left"] = left
        self.crop["right"] = right
        self.crop["top"] = top
        self.crop["bottom"] = bottom

        print(
            textwrap.dedent(
                f"""
            Your images are now shape: ({bottom-top}, {right-left})
            Changes can continue to be made by moving/updating the region,
            but you have to run this again for them to take affect.\n"""
            )
        )
        return

    def set_scale(self, scale):
        """Change the scale of the images (nm/pix) in the relevant places.

        Args:
            scale (float): Scale of images in nm/pixel

        Returns:
            None
        """
        self.axes[0].scale = scale
        self.axes[1].scale = scale
        for sig in self.imstack + self.flipstack:
            sig.axes_manager[0].units = "nm"
            sig.axes_manager[1].units = "nm"
            sig.axes_manager[0].scale = scale
            sig.axes_manager[1].scale = scale

        self.scale = scale
        self.roi = hs.roi.RectangularROI(
            left=self.shape[1] // 4 * self.scale,
            right=3 * self.shape[1] // 4 * self.scale,
            top=self.shape[0] // 4 * self.scale,
            bottom=3 * self.shape[0] // 4 * self.scale,
        )
        return


### End ###
