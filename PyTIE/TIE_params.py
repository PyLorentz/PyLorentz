"""Class for TIE reconstruction parameters.

A class for holding TIE images and reconstruction parameters.
Also has several methods relating to doing the reconstruction, namely making
masks and interactively cropping the image stacks.

AUTHOR:
Arthur McCray, ANL, Summer 2019
"""

import numpy as np
from scipy import ndimage
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap
import time


class TIE_params(object):
    """An object for holding the data and parameters for the reconstruction.

    Some params will be obtained directly from the metadata, others given by the
    user. Also holds some other useful values. An instance is created by the
    load_data() function in TIE_helper.py.

    For more information on how to organize the directory and load the data, as
    well as how to setup the .fls file please refer to the README or the
    TIE_template.ipynb notebook.

    Attributes:
        imstack (list): A list of numpy arrays, one per image in
            the through focus series (tfs).
        flipstack (list): A list of numpy arrays for the flip tfs if there is one.
        defvals (list): List of defocus values in nm from least to most defocus.
            This assumes a symmetric defocus over/under, so expects 2 values for
            a 5 image tfs.
        flip (bool): Boolean for whether or not to reconstruct using the flip
            stack.  Even if the flip stack exists the reconstruction will only
            use the unflip stack if self.flip = False.
        data_loc (str): String for location of data folder.
        no_mask (bool): Eliminates mask (generally used with simulated data).
        num_files (int): Equiv to len(self.imstack)
        shape (tuple): Shape of original image data (y, x)
        scale (float): Scale of images (nm/pixel). Taken from the dm3/tif metadata
            or set manually.
        rotation (float, int): The rotation to apply to the image before reconstruction in deg.
        x_transl (int): The x_translation to apply to the image before reconstruction in pix.
        y_transl (int): The y_translation to apply to the image before reconstruction in pix.
        infocus (2D array): Averaged infocus image between unflip and flip
            stacks. If no flip stack, just unflip infocus image.
        qi (2D array): 2D inverse frequency array, possibly modified with
            Tikhonov filter.
        crop (dict): Region of interest in pixels used to select the are to be
            reconstructed. Initialized to full image.
        mask (2D array): Binary mask made form all the images. 1 where all images have
            nonzero data, 0 where any do not. Made by self.make_mask()
    """

    def __init__(
        self,
        imstack=None,
        flipstack=[],
        defvals=None,
        scale=None,
        flip=None,
        data_loc=None,
        no_mask=False,
        v=1,
    ):
        """Constructs TIE_params object. imstack, defvals, and scale (nm/pixel) must be
        specified at a minimum.

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
        self.shape = np.shape(imstack)[1:]
        if type(defvals) is not list and type(defvals) is not np.ndarray:
            self.defvals = [defvals]
        else:
            self.defvals = np.array(defvals)  # array of the defocus steps.

        self.num_files = len(self.imstack)
        if self.num_files == 1:
            assert len(self.defvals) == 1
        else:
            assert self.num_files == 2 * len(self.defvals) + 1  # confirm they match
        self.scale = scale  # nm/pixel
        vprint(f"Setting scale: {self.scale:.4f} nm/pix\n")

        # The rotation/translation to apply to images.
        self.rotation, self.x_transl, self.y_transl = (0, 0, 0)

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

        infocus = self.imstack[self.num_files // 2]  # unflip infocus dm3
        if flip:
            assert len(self.imstack) == len(self.flipstack)
            flip_infocus = self.flipstack[self.num_files // 2]
            self.infocus = (infocus + flip_infocus) / 2
            # An averaged infocus image between the flip/unflip stack.
        else:
            self.infocus = np.copy(infocus)

        self.qi = np.zeros(self.shape)  # will be inverse Laplacian array
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
        return -1 * self.scale**2 / (16 * np.pi**3 * pscope.lam * def_step)

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
            imstack = np.concatenate([self.imstack, self.flipstack])
        shape = np.shape(imstack[0])
        mask = np.ones(shape)

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


    def select_ROI(self, infocus=True):
        """Select a rectangular region of interest (ROI) and assign it to ptie.crop

        Args:
            infocus (bool, optional): Whether to select a region from the infocus image.
                For some datasets the infocus image will have no contrast, and therefore set
                infocus=False to select a region from the most underfocused image.
                Defaults to True.
        """
        # needs to take list as input so it can add them
        fig, ax = plt.subplots()
        print(
            "right click to add or move points, 'd' to delete when hovering over a point, 'esc' to exit. "
        )

        image = self.infocus if infocus else self.imstack[0]
        ax.matshow(image, cmap="gray")
        dy, dx = image.shape

        # points = np.array([[-1, -1], [-1, -1]])  # [[y1, x1], [y2, x2]]
        points = np.array(
            [
                [self.crop["top"], self.crop["left"]],
                [self.crop["bottom"], self.crop["right"]],
            ]
        )

        click_pad = 100
        class plotter:
            def __init__(self, points):
                self.scat = None
                self.rect = Rectangle((0, 0), 1, 1, fc="none", ec="red")
                ax.add_patch(self.rect)
                if np.all(points >= 0):
                    self.plotrect(points)
                    self.plot(points)

            def plot(self, points):
                # moving point left/up by 1 if > 0 to prevent plotting outside of window
                if self.scat is not None:
                    self.clear()
                ypoints = points[:, 0][points[:, 0] >= 0]
                xpoints = points[:, 1][points[:, 1] >= 0]
                xpoints = np.where(xpoints == dx, xpoints - 1, xpoints)
                ypoints = np.where(ypoints == dy, ypoints - 1, ypoints)
                self.scat = ax.scatter(xpoints, ypoints, c="r")

            def plotrect(self, points):
                (y0, x0), (y1, x1) = points
                self.rect.set_width(x1 - x0)
                self.rect.set_height(y1 - y0)
                self.rect.set_xy((x0, y0))
                ax.figure.canvas.draw()

            def clear(self):
                self.scat.remove()
                self.scat = None

        def on_click(event):
            # make it move closer point not second one always
            if event.button is MouseButton.RIGHT:
                x, y = event.xdata, event.ydata
                if np.any(points[0] < 0):  # draw point0
                    points[0, 0] = y
                    points[0, 1] = x
                elif np.any(points[1] < 0):  # draw point1
                    points[1, 0] = y
                    points[1, 1] = x
                else:  # redraw closer point
                    dist0 = get_dist(points[0], [y, x])
                    dist1 = get_dist(points[1], [y, x])
                    if dist0 < dist1:  # change point0
                        points[0, 0] = y
                        points[0, 1] = x
                    else:
                        points[1, 0] = y
                        points[1, 1] = x
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)

        def on_key_press(event):
            if event.key == "escape":
                if np.all(points > 0):
                    print(f"saving ROI")
                    plt.disconnect(binding_id)
                    plt.disconnect(binding_id2)
                    self.crop["top"] = points[0, 0]
                    self.crop["left"] = points[0, 1]
                    self.crop["bottom"] = points[1, 0]
                    self.crop["right"] = points[1, 1]
                    print(f"ptie.crop: {self.crop}")
                    print(
                        f"Final image dimensions (h x w): {points[1,0]-points[0,0]} x {points[1,1]-points[0,1]}"
                    )
                    print(
                        "Cropping can be returned to the full image by running ptie.reset_crop()"
                    )
                else:
                    print("One or more points are not well defined.")
                    self.reset_crop()
                plt.close(fig)
                return

            elif event.key == "d":
                x, y = event.xdata, event.ydata
                dist0 = get_dist(points[0], [y, x])
                dist1 = get_dist(points[1], [y, x])
                if dist0 < dist1:  # delete point0
                    if dist0 < click_pad:
                        points[0, 0] = -1
                        points[0, 1] = -1
                else:
                    if dist1 < click_pad:
                        points[1, 0] = -1
                        points[1, 1] = -1
                p.plot(points)

            # elif event.key == "c":
            #     p.clear()

        def on_move(event):
            if np.any(points < 0):  # only drawing if not all points not placed
                if event.xdata is not None and event.ydata is not None:
                    if 0 < event.xdata < dx and 0 < event.ydata < dy:
                        if np.all(points[0] > 0):
                            y0, x0 = points[0]
                        elif np.all(points[1] > 0):
                            y0, x0 = points[1]
                        else:
                            return

                        x1 = event.xdata
                        y1 = event.ydata
                        p.plotrect([[y0, x0], [y1, x1]])

        p = plotter(points)
        binding_id = plt.connect("button_press_event", on_click)
        binding_id2 = plt.connect("motion_notify_event", on_move)
        plt.connect("key_press_event", on_key_press)
        plt.show()

    def reset_crop(self):
        """Reset the ptie.crop() region to the full image.
        """
        print("Resetting ROI to full image.")
        self.crop["left"] = 0
        self.crop["right"] = self.shape[1]
        self.crop["top"] = 0
        self.crop["bottom"] = self.shape[0]


def get_dist(pos1, pos2):
    """Distance between two 2D points

    Args:
        pos1 (list): [y1, x1] point 1
        pos2 (list): [y2, x2] point 2

    Returns:
        float: Euclidean distance between the two points
    """
    squared = (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
    return np.sqrt(squared)


### End ###
