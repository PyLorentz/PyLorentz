"""Utility functions for GUI and alignment.

Contains miscellaneous utility functions that help with GUI event handling,
image handling, and certain file handling for alignment.

AUTHOR:
Timothy Cote, ANL, Fall 2019.
"""

# Standard Library Imports
from io import BytesIO
import os
from warnings import catch_warnings, simplefilter
with catch_warnings() as w:
    simplefilter('ignore')
from PIL import Image, ImageDraw
import subprocess
from sys import platform, path as sys_path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
from cv2 import INTER_AREA, INTER_NEAREST, resize, flip, fillPoly, imwrite
import hyperspy.api as hs
import matplotlib
from matplotlib import cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, zeros, flipud, uint8 as np_uint8
import PySimpleGUI as sg

# Local imports
sys_path.append("../PyTIE/")
from TIE_helper import *
# matplotlib.use('TkAgg')


# Miscellaneous to potentially apply later:
# https://imagejdocu.tudor.lu/plugin/utilities/python_dm3_reader/start
# https://python-forum.io/Thread-Tkinter-createing-a-tkinter-photoimage-from-array-in-python3


# ----------------------------------- #
# ------------- Classes ------------- #
# ----------------------------------- #
class Struct(object):
    """The data structure for saving GUI info, image info, and
     reconstruction info. Attributes are built in PyLorentz init
     funcitons.

     This is the most useful class for the GUI and event handler.
     It contains information for each tab and keeps tracks of files, transformations, threads,
     subprocesses, etc.
     """
    pass


class FileObject(object):
    """An object for holding the file data.

    FileObjects hold information for .fls data or other files that may or may not be images.

    For more information on how to organize images the directory and load the data, as
    well as how to setup the .fls file please refer to the README.

    Attributes:
        path: String of the path name to file.
        shortname: The shortened name of the file,
            only shows relative path not full path.
    """

    def __init__(self, path: str) -> None:
        """Initialize file object.

        Args:
            path: The path to the file object.
        """
        self.path = path
        self.shortname = ''
        self.shorten_name()

    def shorten_name(self) -> None:
        """Creates a string of the path name with only the direct parent
        "image_dir" and the child of "image_dir".
        """
        index = self.path.rfind('/') + 1
        self.shortname = self.path[index:]


class FileImage(FileObject):
    """The Image Class contains data about an individual image.
    This data is encoded into bytes for the TK Canvas.

    For more information on how to organize images the directory and load the data, as
    well as how to setup the .fls file please refer to the README.

    Attributes:
        path: String of the path name to file.
        shortname: The shortened name of the file,
            only shows relative path not full path.
        uint8_data: Dictionary of uint8 data of image.
        flt_data: Dictionary of float data of image.
        x_size: The x-size of the image.
        y_size: The y-size of the image.
        z_size: The z-size of the image.
        lat_dims: The laterial dimensions of the image.
        byte_data: The byte data of the image.
    """

    def __init__(self, uint8_data: Dict, flt_data: Dict,
                 size: Tuple[int, int, int], path: str,
                 float_array: Optional['np.ndarray']=None) -> None:
        """Initialize the FileImage Object.

        Args:
            uint8_data: Ndarray of uint8 data of image.
            flt_data: Ndarray of float data of image.
            size: Tuple of x, y, z size of the image.
            path: The path to the FileImage object.
            float_array: The original numpy array of data. Necessary for certain
                reconstruction images.
        """

        super().__init__(path)
        if uint8_data is not None:
            self.uint8_data = uint8_data                   # Uint8 image data
            self.flt_data = flt_data                       # Numerical image array
            self.x_size, self.y_size, self.z_size = size
            self.lat_dims = self.x_size, self.y_size
            self.byte_data = None                           # Byte data
            self.float_array = float_array


class Stack(FileImage):
    """The Stack Class contains data about an image stack.
    This data is encoded into bytes for the TK Canvas. It
    is a subclass of the Image Class.

    Attributes:
        path: String of the path name to file.
        shortname: The shortened name of the file,
            only shows relative path not full path.
        uint8_data: Ndarray of uint8 data of image.
        flt_data: Ndarray of float data of image.
        x_size: The x-size of the image.
        y_size: The y-size of the image.
        z_size: The z-size of the image.
        lat_dims: The laterial dimensions of the image.
        byte_data: Dictionary of the byte data of the image.
    """

    def __init__(self, uint8_data: 'np.ndarray', flt_data: 'np.ndarray',
                 size: Tuple[int, int, int], path: str):
        """Initialize the Stack Object.

        Args:
            uint8_data: Ndarray of uint8 data of image.
            flt_data: Ndarray of float data of image.
            size: Tuple of x, y, z size of the image.
            path: The path to the FileImage object
        """

        super().__init__(uint8_data, flt_data, size, path)
        self.stack_byte_data()

    def stack_byte_data(self):
        """Create the byte data for all the images in the stack."""

        self.byte_data = {}
        self.rgba_data = {}
        for pic in range(self.z_size):
            self.byte_data[pic] = vis_1_im(self, pic)


# ============================================================= #
#             Miscellaneous Manipulations and Checks.           #
# ============================================================= #
def flatten_order_list(my_list: List[List]) -> List:
    """Flattens and orders a list of 3 lists into a single list.

    Flattens and orders 2D list of lists of items:: 

        [[b , a], [c, d], [e, f]]

    into a 1D list of items::

        [a, b, c, d, e, f]

    Args:
        my_list: A 2D list of list of items.

    Returns:
        flat_list: A 1D flattened/ordered list of items.
    """

    l0, l1, l2 = my_list[0], my_list[1], my_list[2]
    ordered_list = [l0[::-1], l1, l2]
    flat_list = [item for sublist in ordered_list for item in sublist]
    return flat_list


def join(strings: List[str], sep: str = '') -> str:
    """Method joins strings them with a specific separator.

    Strings are joined with the seperator and if the string contains
    double backslashes, it replaces them with a forward slash.

    Args:
        strings: A list of strings to join.
        sep: The character to seperate the string joing.

    Returns:
        final_string: The concatenated string.
    """

    final_string = sep.join(strings)
    final_string = final_string.replace('\\', '/')
    return final_string


def represents_float(s: str) -> bool:
    """Evaluate if the string is a float.

    Args:
        s: A string to check if it wil be a float.

    Returns:
        True if it converts to float, False otherwise.
    """

    try:
        float(s)
        return True
    except ValueError:
        return False


def represents_int_above_0(s: str) -> bool:
    """Evaluate if the string is an integer.

    Args:
        s: A string to check if it wil be a float.

    Returns:
        True if it converts to float, False otherwise.
    """

    try:
        val = int(s)
        if val > 0:
            return True
        else:
            return False
    except ValueError:
        return False


# ============================================================= #
#                  Image Loading & Manipulating.                #
# ============================================================= #
def load_image(img_path: str, graph_size: Tuple[int, int], key: str,
               stack: bool = False, prefix: str = '') -> Tuple[
                Optional[Dict[int, 'np.ndarray[np.uint8]']],
                Optional[Dict[int, 'np.ndarray[np.float64, np.float32]']],
                Optional[Tuple[int, int, int]]]:
    """Load an image file.

    Load an image file if it is a stack, dm3, dm4, or bitmap. As of now,
    Fiji doesn't allow easy loading of dm4's so be warned that alignment
    for dm4 files probably won't work.

    Args:
        img_path: Full path to the location of the image(s).
        graph_size: The size of the graph in (x, y) coords.
        key: The key of the element clicked for loading an image.
        stack: Boolean value if the image is a stack. Default is False.
        prefix: The prefix value for the print statements to the GUI log.
            Default is True.

    Returns:
        tuple: Tuple containing three items: 

            - uint8_data: The uint8 data dictionary with stack image/stack slice key
              and value of uint8 dtype ndarray or None if loading failed.
            - float_data: The float data dictionary with stack image/stack slice key
              and value of float dtype ndarray or None if loading failed.
            - size: The x, y, z size of the data or None if the loading failed.
    """
    try:
        # Check path has correct filetype
        correct_end = False
        for end in ['.tif', '.tiff', '.dm3', '.dm4', '.bmp']:
            if img_path.endswith(end):
                correct_end = True
        if not correct_end:
            if 'Stage' in key or 'Align' in key:
                print(f'{prefix}Trying to load an incorrect filetype. Acceptable values "tif" are "tiff".')
            elif 'FLS' in key:
                print(f'{prefix}Trying to load an incorrect filetype. Acceptable values "tif", "tiff", "dm3", "dm4".')
            raise
        # Load data into numpy arrays for processing in GUI.
        uint8_data, float_data = {}, {}
        z_size = 1
        img = hs.load(img_path)
        img_data = img.data
        shape = img_data.shape
        if stack and len(shape) > 1:
            z_size, y_size, x_size = shape
        elif not stack:
            y_size, x_size = shape
        else:
            print(f'{prefix}Do not try loading a single image to stack.')
            raise
        for z in range(z_size):
            if stack:
                temp_data = img_data[z, :, :]
            else:
                temp_data = img_data
            # Scale data for graph, convert to uint8, and store float data as well
            uint8_data, float_data = convert_float_unint8(temp_data, graph_size, uint8_data, float_data, z)

        # Return dictionary of uint8 data, a scaled float array, and the shape of the image/stack.
        size =  (x_size, y_size, z_size)
        return uint8_data, float_data, size
    # A multitude of errors can cause an image failure, usually with trying to load an incorrect file.
    except (IndexError, TypeError, NameError):
        print(f'{prefix}Error. You may have tried loading a file that is not recognized. Try a different file type', end='')
        return None, None, None
    except:
        return None, None, None


def array_resize(array, new_size):

    resized_array = resize(array, new_size, interpolation=INTER_AREA)
    return resized_array


def convert_float_unint8(float_array: 'np.ndarray[np.float64, np.float32]', graph_size: Tuple[int, int],
                         uint8_data: Optional[Dict] = None, float_data: Optional[Dict] = None,
                         z: int = 0) -> Tuple[Dict[int, 'np.ndarray[np.uint8]'], Dict[int, 'np.ndarray[np.float64, np.float32]']]:
    """Convert float image data to uint8 data, scaling for view in window.

        Images need to be converted to uint8 data for potential future processing and
        loading into the GUI window.

        Args:
            float_array: Ndarray dtype float of a single slice of image data.
            graph_size: The size of the graph in (x, y) coords.
            uint8_data: The dictionary that stores the uint8 image data.
            float_data: The dictionary that stores the scaled float image data.
            z: The stack slice key for the uint8 and float dictionaries.

        Returns:
            uint8_data: The uint8 data dictionary with stack image/stack slice key
                        and value of uint8 dtype ndarray.
            float_data: The float data dictionary with stack image/stack slice key
                        and value of float dtype ndarray.
        """

    if uint8_data is None:
        uint8_data = {}
    if float_data is None:
        float_data = {}
    if float_array is not None:
        scaled_float_array = resize(float_array, graph_size, interpolation=INTER_AREA)
        resized_data = scaled_float_array - scaled_float_array.min()
        maximum = resized_data.max()
        if maximum == 0:
            maximum = 1
        inv_max = 1/maximum
        scaled_float_array = resized_data * inv_max
        uint8_array = scaled_float_array * 255
        uint8_array = uint8_array.astype(np_uint8)
        uint8_data[z] = uint8_array
        float_data[z] = scaled_float_array
    return uint8_data, float_data


def apply_rot_transl(data: 'np.ndarray', d_theta: Union[int, float] = 0, d_x: Union[int, float] = 0,
                           d_y: Union[int, float] = 0, h_flip: Optional[bool] = None):

    # Apply horizontal flip if necessary
    if h_flip:
        data = flip(data, 1)

    # Convert to PIL Image
    rgba_img = Image.fromarray(data).convert('RGBA')

    # Rotate, take note of size change due to expand=True value
    old_size = rgba_img.size
    rgba_img = rgba_img.rotate(d_theta, expand=True)
    new_size = rgba_img.size

    # Translate
    affine_matrix = (1, 0, -d_x, 0, 1, d_y)
    rgba_img = rgba_img.transform(rgba_img.size, Image.AFFINE, affine_matrix)

    # Reshape the expanded array
    old_side, new_side = old_size[0], new_size[0]
    box_diff = (new_side - old_side) // 2
    left, top, right, bottom = box_diff, box_diff, new_side - box_diff, new_side - box_diff
    crop_box = (left, top, right, bottom)
    rgba_img = rgba_img.crop(crop_box)
    return rgba_img


def make_rgba(data: 'np.ndarray[np.uint8]', adjust: bool = False,
              d_theta: Union[int, float] = 0, d_x: Union[int, float] = 0,
              d_y: Union[int, float] = 0, h_flip: Optional[bool] = None,
              color: Optional[bool] = None) -> Image.Image:
    """Create an rgba image from numpy ndarray data of an image.

        Rgba images need to be created from uint8 data so that they can be converted to
        bytes for display in the graph. Additionally a colormap may need to be applied along
        with alpha values to show overlaying images.

        Args:
            data: The ndarray of the image data.
            adjust: Boolean value whether transformations need to be applied to the image.
            d_theta: The angle with which to rotate the image data.
            d_x: The x-translation to move the image data.
            d_y: The y-translation to move the image data.
            h_flip: Boolean value of whether to flip the image horizontally.
            color: Boolean of whether or not a color-map should be applied to the image.
                True if overlaying images, otherwise False.
        Returns:
            rgba_img: The PIL rgba image of the image data.
        """

    if adjust:
        # Apply colormap and convert to uint8 datatype
        if color == 'None':
            data = data * 255
            data = data.astype(np_uint8)
        elif color:
            cm = mpl_cm.get_cmap(color)
            data = cm(data, bytes=True)
        else:
            cm = mpl_cm.get_cmap('Spectral')    # spectral, bwr,twilight, twilight_shifted, hsv shows good contrast
            data = cm(data, bytes=True)

        # Apply transformation  if necessary
        rgba_img = apply_rot_transl(data, d_theta, d_x, d_y, h_flip)
    else:
        # Convert to PIL Image
        rgba_img = Image.fromarray(data).convert('RGBA')
    return rgba_img


def convert_to_bytes(img: Image.Image) -> bytes:
    """Converts a PIL image to bytes.

        The byte 'ppm' type is used to directly insert image data into the GUI window. Thus the
        PIL images need to be converted to this type.

        Args:
            img: The PIL representation of the image data.

        Returns:
            byte_img: The byte representation of the image data.
            """
    byte_img = BytesIO()
    img.save(byte_img, format='ppm')
    byte_img = byte_img.getvalue()
    return byte_img


def overlay_images(image1: 'np.ndarray[np.float32, np.float64]', image2: 'np.ndarray[np.float32, np.float64]',
                   transform: Tuple[Union[int, float], Union[int, float], Union[int, float], bool],
                   image_size: Tuple[int, int], graph_size: Tuple[int, int]) -> bytes:
    """Overlay images.

        Visualize overlaid images in the GUI canvas. Takes an
        Image Object and converts uint8 data into byte data the Tk
        canvas can use.

        Args:
            image1: Float data of image1 in an ndarray.
            image2: Float data of image2 in an ndarray.
            transform: Tuple of the rotation, x-translation, y-translation, and
                horizontal flip to apply to the image.
            image_size: The image size in (x, y) size.
            graph_size: The graph size in (x, y) size.

        Returns:
            return_img: The byte representation of the image data.
        """

    # Pull the transformation data, rounding for ease of use for translations.
    d_theta, d_x, d_y, h_flip = transform
    d_x = round(d_x/image_size*graph_size)
    d_y = round(d_y/image_size*graph_size)
    array1 = image1.flt_data[0]
    array2 = image2.flt_data[0]

    # Convert to rgba images and merge to create the final bytes image.
    source_rgba = make_rgba(array1, adjust=True, d_theta=d_theta, d_x=d_x, d_y=d_y, h_flip=h_flip)
    target_rgba = make_rgba(array2, adjust=True)
    final_img = Image.blend(target_rgba, source_rgba, alpha=.6)
    return_img = convert_to_bytes(final_img)
    return return_img


def adjust_image(data: 'np.ndarray[np.float32, np.float64]',
                 transform: Tuple[Union[int, float], Union[int, float], Union[int, float], bool],
                 image_size: Tuple[int, int], graph_size: Tuple[int, int]) -> bytes:
    """Apply transformations to an image.

        Apply rotations, translations, and/or flippin of image.

        Args:
            data: Float data of image1 in an ndarray.
            transform: Tuple of the rotation, x-translation, y-translation, and
                horizontal flip to apply to the image.
            image_size: The image size in (x, y) size.
            graph_size: The graph size in (x, y) size.

        Returns:
            return_img: The byte representation of the image data.
        """
    # Pull the transformation data, rounding for ease of use for translations.
    d_theta, d_x, d_y, h_flip = transform
    d_x = round(d_x/image_size*graph_size)
    d_y = round(d_y/image_size*graph_size)

    # Create the rgba and the return byte image.
    rgba_image = make_rgba(data, adjust=True, d_theta=d_theta, d_x=d_x, d_y=d_y, h_flip=h_flip, color='None')
    return_img = convert_to_bytes(rgba_image)
    return return_img, rgba_image


def vis_1_im(image: FileImage, layer: int = 0) -> bytes:
    """Visualize one image in the GUI canvas. Takes a
    FileImage Object and converts uint8 data into byte data the Tk
    canvas can use.

    Args:
        image: The image object that holds data and path info about
            the image.
        layer: Default is 0 if viewing single image. If viewing a
            stack, must choose layer slice.
    Returns:
        return_img: A byte representation of the image data
            to represent in TK canvas.
    """

    im_data = image.uint8_data[layer]
    rgba_image = make_rgba(im_data)
    return_img = convert_to_bytes(rgba_image)
    return return_img


def slice(image: 'np.ndarray', slice_size: Tuple[int, int]) -> 'np.ndarray':
    """Slice an image

    Args:
        image: ndarray of image data.
        slice_size: The ends bounds of the image slice in (x, y) coords.

    Returns:
        An ndarray of the sliced image.
    """

    endx, endy = slice_size
    startx, starty = 0, 0
    return image[startx:endx, starty:endy, :]


def add_vectors(mag_x: 'np.ndarray', mag_y: 'np.ndarray', color_np_array: 'np.ndarray', color: bool,
                hsv: bool, arrows: int, length: float, width: float,
                graph_size: Tuple[int, int], GUI_handle: bool = True,
                save: Optional[bool] = None) -> Optional['bytes']:
    """Vectorize the magnetic saturation images for GUI.

    Args:
        mag_x: The x-component of the magnetic induction.
        mag_y: The y-component of the magnetic induction.
        color_np_array: The colorized magnetic saturation array.
        color: The boolean value for a color image (True) or black & white image (False).
        hsv: The boolean value for hsv color image (True) or 4-fold color image (False).
        arrows: The number of arrows to place along the rows and cols of the image.
        length: The inverse length of the arrows. Inverted when passed to show_2D.
        width: The width of the arrows.
        graph_size: The (x, y) size of the GUI display graph.
        GUI_handle: The handle to pass to TIE_helper.show_2D() signalling whether to use GUI
            or matplotlib. This defaults to True for the GUI.
        save: The value to determine saving of the vectorized image.

    Returns:
        Optional: The byte image for the vectorized data.
    """

    fig, ax = show_2D(mag_x, mag_y, a=arrows, l=1/length, w=width, title=None, color=color, hsv=hsv,
                      save=save, GUI_handle=GUI_handle, GUI_color_array=color_np_array)
    if GUI_handle:
        plt.figure(fig.number)
        fig.tight_layout(pad=0)
        plt.axis('off')
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        # Resize with CV
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = resize(data, graph_size, interpolation=INTER_AREA)
        rgba_image = make_rgba(data)
        return_img = convert_to_bytes(rgba_image)
        plt.close('all')
        return return_img
    else:
        plt.close('all')
        return


def apply_crop_to_stack(coords, graph_size, transform, stack, i):

    # Create mask image (np mask)
    mask = zeros(graph_size, np.uint8)
    coords = [[coords[i][0], coords[i][1]] for i in range(len(coords))]
    mask = create_mask(mask, coords, 1)
    mask = flipud(mask)

    # Create transformed image (PIL)
    img = stack.rgba_data[i]
    img_array = np.asarray(img)
    img = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

    # Black out the masked region
    masked_image = np.multiply(img, mask)
    rgba_masked_image = make_rgba(masked_image)
    display_img = convert_to_bytes(rgba_masked_image)
    return display_img, rgba_masked_image


# ============================================================= #
#                   Run Fiji from Command Line                  #
# ============================================================= #
def run_macro(ijm_macro_script: str, key: str,
              image_dir: str, fiji_path: str) -> str:
    """Run Fiji macros.

    This script generates the command to run the Fiji macro from
    the subprocess command.

    Args:
        ijm_macro_script: String of the full Fiji macro (.ijm) to run.
        key: The key of the button element that was clicked for running.
        image_dir: The path to the image direcotry.
        fiji_path: The path to Fiji.

    Returns:
        cmd: The subprocess command to run as a string.
    """

    # Check Fiji path
    if platform.startswith('win'):
        add_on = "/ImageJ-win64"
    elif platform.startswith('darwin'):
        add_on = "/Contents/MacOS/ImageJ-macosx"
    elif platform.startswith('linux'):
        add_on = "/ImageJ-linux64"
    fiji_path = fiji_path + add_on

    if key == '__LS_Run_Align__':
        align_type = 'LS'
    elif key == "__BUJ_Elastic_Align__":
        align_type = 'BUJ'
    elif key == "__BUJ_Unflip_Align__":
        align_type = 'BUJ_unflip_LS'
    elif key == "__BUJ_Flip_Align__":
        align_type = 'BUJ_flip_LS'

    macro_file = f'{image_dir}/macros/{align_type}_macro.ijm'
    if not os.path.exists(f'{image_dir}/macros'):
        os.mkdir(f'{image_dir}/macros')

    with open(macro_file, 'w') as f:
        f.write(ijm_macro_script)
        f.close()

    cmd = join([fiji_path, "--ij2", "--headless", "--console", "-macro ", macro_file], " ")
    return cmd


# ============================================================= #
#                 Mask Interaction on GUI Graph                 #
# ============================================================= #
def draw_mask_points(winfo: Struct, graph: sg.Graph,
                     current_tab: str, double_click: bool = False) -> None:
    """Draw mask markers to appear on graph.

    Args:
        winfo: Struct object that holds all GUI information.
        graph: The selected PSG graph to draw mask points on.
        current_tab: String denoting the current tab the graph
            is on in the GUI
        double_click: Double-click value for terminating the mask drawing.
            Only necessary for bUnwarpJ masks.

    Returns:
        None
    """
    if current_tab == 'bunwarpj_tab':
        num_coords = len(winfo.buj_mask_coords)
        for i in range(num_coords):
            x1, y1 = winfo.buj_mask_coords[i]
            x2, y2 = winfo.buj_mask_coords[(i + 1) % num_coords]
            id_horiz = graph.DrawLine((x1 - 7, y1), (x1 + 7, y1), color='red', width=2)
            id_verti = graph.DrawLine((x1, y1 - 7), (x1, y1 + 7), color='red', width=2)
            end = (i+1) % num_coords
            if end or double_click:
                id_next = graph.DrawLine((x1, y1), (x2, y2), color='red', width=2)
            else:
                id_next = graph.DrawLine((x1, y1), (x1, y1), color='red', width=1)
            winfo.buj_mask_markers.append((id_horiz, id_verti, id_next))
    elif current_tab == 'reconstruct_tab':
        num_coords = len(winfo.rec_mask_coords)
        for i in range(num_coords):
            x1, y1 = winfo.rec_mask_coords[i]
            x2, y2 = winfo.rec_mask_coords[(i + 1) % num_coords]
            id = graph.DrawLine((x1-1, y1), (x2-1, y2), color='red', width=1)
            winfo.rec_mask_markers.append(id)


def erase_marks(winfo: Struct, graph: sg.Graph,
                     current_tab: str, full_erase: bool = False) -> None:
    """Erase markers off graph. Delete stored markers if full_erase enabled.

        Args:
            winfo: Struct object that holds all GUI information.
            graph: The selected PSG graph to draw mask points on.
            current_tab: String denoting the current tab the graph
                is on in the GUI
            full_erase: Value for deleting the figures from the graph.

        Returns:
            None
    """
    if current_tab == 'bunwarpj_tab':
        for marks in winfo.buj_mask_markers:
            for line in marks:
                graph.DeleteFigure(line)
        if full_erase:
            winfo.buj_mask_coords = []
            winfo.buj_mask_markers = []
            winfo.buj_graph_double_click = False
    elif current_tab == 'reconstruct_tab':
        winfo.rec_mask_coords = []
        for line in winfo.rec_mask_markers:
            graph.DeleteFigure(line)


def create_mask(img, mask_coords, color):

    pts = array(mask_coords)
    img = fillPoly(img, pts=np.int32([pts]), color=color)
    return img


def save_mask(winfo: Struct, filenames: List[str], ref_img: FileImage):
    """Save created mask(s) as .bmp with a given filename(s).

        Args:
            winfo: Struct object that holds all GUI information.
            filenames: The filenames for the masks.
            ref_img: The reference FileImage for getting the dimensions of the image.

        Returns:
            None
        """

    # Choose the graph
    graph = winfo.window['__BUJ_Graph__']
    coords = winfo.buj_mask_coords

    # Get mask coordinates and create cv.fillPoly image
    graph_size = graph.CanvasSize
    img = zeros(graph_size, np_uint8)
    img = create_mask(img, coords, (255, 255, 255))

    # Resize the mask and flip due to inverted y-axis for graph compared to saved file
    orig_sizex, orig_sizey = ref_img.lat_dims
    img = resize(img, (orig_sizex, orig_sizey), interpolation=INTER_NEAREST)
    img = flipud(img)
    for filename in filenames:
        imwrite(f"{filename}", img)


def draw_square_mask(winfo: Struct, graph: sg.Graph) -> None:
    """Create the square mask for the REC graph.

        Args:
            winfo: Struct object that holds all GUI information.
            graph: The selected PSG graph to draw mask points on.

        Returns:
            None
    """

    # Get the size of the mask and the graph
    mask_percent = winfo.rec_mask[0] / 100
    graph_x, graph_y = graph.get_size()
    center_x, center_y = winfo.rec_mask_center
    winfo.rec_mask_coords = []
    left_bounds, right_bounds = False, False
    top_bounds, bottom_bounds = False, False

    # Specific handling of figuring out the coordinates of the GUI mask. Be careful
    # not to change this or check how this may change with future PySimpleGUI updates.
    width, height = round(graph_x * mask_percent), round(graph_x * mask_percent)
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1
    if center_x <= width//2:
        left_bounds = True
    if center_x >= graph_x - width//2:
        right_bounds = True
    if graph_y - center_y <= height//2:
        top_bounds = True
    if graph_y - center_y >= graph_y - height//2:
        bottom_bounds = True

    if not left_bounds and not right_bounds:
        x_left = center_x - width//2
        x_right = center_x + width//2
    elif left_bounds and right_bounds:
        x_left = 0
        x_right = graph_x
    elif right_bounds:
        x_left = graph_x - width
        x_right = graph_x
    elif left_bounds:
        x_left = 0
        x_right = width

    if not top_bounds and not bottom_bounds:
        y_top = center_y - height//2
        y_bottom = center_y + height//2
    elif top_bounds and bottom_bounds:
        y_top = 0
        y_bottom = graph_y
    elif bottom_bounds:
        y_top = 0
        y_bottom = height
    elif top_bounds:
        y_top = graph_y - height
        y_bottom = graph_y

    winfo.rec_mask_coords = [(x_left, y_top), (x_left, y_bottom), (x_right, y_bottom), (x_right, y_top)]
