"""Utility functions for GUI.
Contains miscellaneous utility functions that help with GUI event handling,
image handling, and certain file handling for alignment.
AUTHOR:
Timothy Cote, ANL, Fall 2019.
"""

# Standard Library Imports
from io import BytesIO
import dask.array as da
import os
from os import path as os_path
from warnings import catch_warnings, simplefilter

with catch_warnings() as w:
    simplefilter("ignore")
from PIL import Image, ImageDraw
import subprocess
from sys import platform, path as sys_path
from typing import Any, Dict, List, Optional, Tuple, Union
import gui_styling

# Third-Party Imports
import hyperspy.api as hs

# Set TkAgg to show if using a drawing computer so vector image gets displayed
import matplotlib
matplotlib.use("agg")
from matplotlib import cm as mpl_cm, pyplot as plt
import numpy as np
from numpy import array, zeros, flipud, uint8 as np_uint8
from skimage import transform, draw
import PySimpleGUI as sg

# Local imports
sys_path.append("../PyTIE/")
from TIE_helper import *


# ----------------------------------- #
# ------------- Classes ------------- #
# ----------------------------------- #
class Struct(object):
    """The data structure responsible saving GUI info, image info, and
    reconstruction info. Attributes are built in PyLorentz 'init'
    functions.
    This is the most useful class for the GUI and event handler.
    It contains information for each tab and keeps tracks of files, transformations,
    threads, subprocesses, etc.
    """

    pass


class FileObject(object):
    """An object for holding the file data.
    FileObjects hold information for .fls data or other files that may or may not be images.
    For more information on how to organize images the directory and load the data, as
    well as how to setup the .fls file please refer to the README or GUI manual.
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
        self.shortname = ""
        self.shorten_name()

    def shorten_name(self) -> None:
        """Creates a string of the path name with only the direct parent
        "image_dir" and the child of "image_dir".
        """
        index = self.path.rfind("/") + 1
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
        byte_data: The dictionary of the byte data of the image.
    """

    def __init__(
        self,
        uint8_data: Dict,
        flt_data: Dict,
        size: Tuple[int, int, int],
        path: str,
        float_array: Optional["np.ndarray"] = None,
    ) -> None:
        """Initialize the FileImage Object.
        Args:
            uint8_data: Dictionary of uint8 data of image.
            flt_data: Dictionary of float data of image.
            size: Tuple of x, y, z size of the image.
            path: The path to the FileImage object.
            float_array: The original numpy array of data. Necessary for certain
                reconstruction images.
        """

        super().__init__(path)
        if uint8_data is not None:
            self.uint8_data = uint8_data  # Uint8 image data
            self.flt_data = flt_data  # Numerical image array
            self.x_size, self.y_size, self.z_size = size
            self.lat_dims = self.x_size, self.y_size
            self.byte_data = None  # Byte data
            self.float_array = float_array


class Stack(FileImage):
    """The Stack Class contains data about an image stack.
    This data is encoded into bytes for the TK Canvas. It
    is a subclass of the Image Class.
    Attributes:
        path: String of the full path of the stack.
        shortname: The shortened name of the stack,
            only shows relative path not full path.
        uint8_data: Dictionary of uint8 data of stack. Each
            key is associated with a slice of the stack.
        flt_data: Dictionary of float data of stack. Each
            key is associated with a slice of the stack.
        x_size: The x-size of the stack.
        y_size: The y-size of the stack.
        z_size: The z-size of the stack.
        lat_dims: The lateral dimensions of the stack.
        byte_data: Dictionary of the byte data of the stack.
    """

    def __init__(
        self, uint8_data: Dict, flt_data: Dict, size: Tuple[int, int, int], path: str
    ):
        """Initialize the Stack Object.
        Args:
            uint8_data: Dictionary of uint8 data of stack. Each
                slice is a key for an ndarray.
            flt_data: Dictionary of float data of stack. Each
                slice is a key for an ndarray.
            size: Tuple of x, y, z size of the stack.
            path: The path to the file represented by the
                FileImage object.
        """

        super().__init__(uint8_data, flt_data, size, path)
        self.rgba_data = {}
        self.byte_data = {}
        self.stack_byte_data()

    def stack_byte_data(self):
        """Create the byte data for all the images in the stack."""

        for pic in range(self.z_size):
            self.byte_data[pic] = vis_1_im(self, pic)


# ============================================================= #
#             Miscellaneous Manipulations and Checks.           #
# ============================================================= #
def join(strings: List[str], sep: str = "") -> str:
    """Method joins strings with a specific separator.
    Strings are joined with the separator and if the string contains
    double backslashes, it replaces them with a forward slash.
    Args:
        strings: A list of strings to join.
        sep: The character to separate the string joining.
    Returns:
        final_string: The concatenated string.
    """

    final_string = sep.join(strings)
    final_string = final_string.replace("\\", "/")
    return final_string


def represents_float(s: str) -> bool:
    """Returns value evaluating if a string is a float.
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
    """Returns value evaluating if a string is an integer > 0.
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
#                    Manipulating FLS Files                     #
# ============================================================= #

#                     Declare Original Types                    #
Check_Setup = Tuple[
    bool, Optional[str], Optional[str], Optional[List[str]], Optional[List[str]]
]


def flatten_order_list(my_list: List[List]) -> List:
    """Flattens and orders a list of 3 lists into a single list.
    Flattens and orders 2D list of lists of items:
        [[b , a], [c, d], [e, f]]
    into a 1D list of items:
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


def pull_image_files(fls_file: str, check_align: bool = False) -> List[List[str]]:
    """Use .fls file to return ordered images for alignment.
    Initially it will read in .fls data, pull the number of files
    from the first line, and then locate the in-focus image. Then
    it separates the overfocus and underfocus images.
    If the check alignment is set, the returned images are the infocus image,
    and the nearest under-focused/over-focused on either side of the
    infocus image. Otherwise all image files are returned. The files are
    ordered from
                    [
                    [smallest underfocus,  ... , largest underfocus]
                    [infocus]
                    [smallest overfocus, ... , largest overfocus]
                    ]
    Args:
        fls_file: The filename for the fls file.
        check_align: Option for full alignment or parameter test.
    Returns:
        filenames: 2D list of under/in/over-focus images.
    """

    # Read data.
    with open(fls_file) as fls_text:
        fls_text = fls_text.read()

    # Find focus line demarcations in .fls file
    fls_lines = fls_text.splitlines()
    num_files = int(fls_lines[0])
    under_split = 1
    focus_split = num_files // 2 + under_split
    over_split = num_files // 2 + focus_split

    # Grab infocus image and defocus images.
    # If checking parameters, only grab one
    # file on either side of the infocus image.
    focus_file = [fls_lines[focus_split]]
    if check_align:
        under_files = [fls_lines[focus_split - 1]]
        over_files = [fls_lines[focus_split + 1]]
    else:
        under_files = fls_lines[under_split:focus_split]
        over_files = fls_lines[focus_split + 1 : over_split + 1]

    # Reverse underfocus files due to how ImageJ opens images
    filenames = [under_files[::-1], focus_file, over_files]
    return filenames


def grab_fls_data(
    fls1: str, fls2: str, tfs_value: str, fls_value: str, check_sift: bool
) -> Tuple[List[str], List[str]]:
    """Grab image data from .fls file.
    Given the FLS files for the flip/unflip/single images,
    return the image filenames depending on the fls_value and
    through-focal-series (tfs) value.
    Examples:
        - 1 FLS, Unflip/FLip -> files1 : populated, files2 : populated
        - 1 FLS, Single      -> files1 : populated, files2 : empty
        - 2 FLS, Unflip/FLip -> files1 : populated, files2 : populated
    Args:
        fls1: Filename for 1st FLS file.
        fls2: Filename for 2nd FLS file.
        tfs_value: Value for type of through-focal series,
            Single or Unflip/Flip.
        fls_value: String of number of FLS files used.
        check_sift: Option to check sift alignment.
    Returns:
        Tuple of image filenames for scenarios above
            - files1: List of image filenames
                according to examples above.
            - files2: List of image filenames.
                according to examples above.
    """

    # Read image data from .fls files and store in flip/unflip lists
    if fls_value == "One":
        files1 = pull_image_files(fls1, check_sift)
        if tfs_value == "Unflip/Flip":
            files2 = pull_image_files(fls1, check_sift)
        else:
            files2 = []
    elif fls_value == "Two":
        files1 = pull_image_files(fls1, check_sift)
        files2 = pull_image_files(fls2, check_sift)
    return files1, files2


def read_fls(
    path1: Optional[str],
    path2: Optional[str],
    fls_files: List[str],
    tfs_value: str,
    fls_value: str,
    check_sift: bool = False,
) -> Tuple[List[str], List[str]]:
    """Read image files from .fls files and returns their paths.
    The images are read from the FLS files and the files are returned
    depending on the through-focal series value and the fls value. Once
    image filenames are pulled from the FLS file, they are joined to
    the paths (directories) the images are stored in. Those resulting
    full path names are returned in files1 and files2.
    Args:
        path1: The first unflip/flip/single path/directory. Optional
        path2: The first unflip/flip/single path/directory. Optional
        fls_files: A list of the FLS filenames.
        tfs_value: The through-focal series option.
                Options: Unflip/FLip, Single
        fls_value: The FLS option.
                Options: One, Two
        check_sift: Option for checking SIFT alignment.
    Returns:
        (files1, files2): A tuple of the lists of image paths corresponding
            to path1 and path2.
            - files1: List of full image paths.
            - files2: List of full image paths or empty list.
    """

    # Find paths and .fls files
    fls1, fls2 = fls_files[0], fls_files[1]

    # Read image data from .fls files and store in flip/unflip lists
    files1, files2 = grab_fls_data(fls1, fls2, tfs_value, fls_value, check_sift)

    # Check same number of files between fls
    if tfs_value != "Single":
        if len(flatten_order_list(files1)) != len(flatten_order_list(files2)):
            return
    # Check if image path exists and break if any path is nonexistent
    if path1 is None and path2 is None:
        return
    for file in flatten_order_list(files1):
        full_path = join([path1, file], "/")
        if not os_path.exists(full_path):
            print(full_path, " doesn't exist!")
            return
    if files2:
        for file in flatten_order_list(files2):
            full_path = join([path2, file], "/")
            if not os_path.exists(full_path):
                print(full_path, " doesn't exist!")
                return
    return files1, files2


def check_setup(
    datafolder: str,
    tfs_value: str,
    fls_value: str,
    fls_files: List[str],
    prefix: str = "",
) -> Check_Setup:
    """Check to see all images filenames in .fls exist in datafolder.
    Args:
        datafolder: Datafolder path.
        tfs_value: The through-focal series option.
                Options: Unflip/FLip, Single
        fls_value: The FLS option.
                Options: One, Two
        fls_files: A list of the FLS filenames.
        prefix: The prefix to prepend for print statements in GUI.
    Returns:
        vals: Will return images filenames and paths to those files
            and their parent directories if all images pulled from FLS exist.
            - vals[0]: Corresponds to process success.
            - vals[1]: 1st parent directory path or None.
            - vals[2]: 2nd parent directory path or None.
            - vals[3]: 1st path list of ordered image filenames.
            - vals[4]: 2nd path list of ordered image filenames.
    """

    # Find paths and .fls files
    if tfs_value == "Unflip/Flip":
        path1 = join([datafolder, "unflip"], "/")
        path2 = join([datafolder, "flip"], "/")
    elif tfs_value == "Single":
        path1 = join([datafolder, "tfs"], "/")
        if not os_path.exists(path1):
            path1 = join([datafolder, "unflip"], "/")
            if not os_path.exists(path1):
                path1 = None
        path2 = None

    # Grab the files that exist in the flip and unflip dirs.
    file_result = read_fls(
        path1, path2, fls_files, tfs_value, fls_value, check_sift=False
    )

    vals = (False, None, None, None, None)
    if isinstance(file_result, tuple):
        files1, files2 = file_result
        flattened_files1 = flatten_order_list(files1)
        flattened_files2 = None
        if files2:
            flattened_files2 = flatten_order_list(files2)
        vals = (True, path1, path2, flattened_files1, flattened_files2)
    # Prints if task failed
    else:
        print(
            f"{prefix}Task failed because the number of files extracted from the directory",
            end=" ",
        )
        print(f"does not match the number of files expected from the .fls file.")
        print(f"{prefix}Check that filenames in the flip, unflip, or tfs", end=" ")
        print(f"path match and all files exist in the right directories.")
    return vals


# ============================================================= #
#                  Image Loading & Manipulating.                #
# ============================================================= #
def load_image(
    img_path: str,
    graph_size: Tuple[int, int],
    key: str,
    stack: bool = False,
    prefix: str = "",
) -> Tuple[
    Optional[Dict[int, "np.ndarray[np.uint8]"]],
    Optional[Dict[int, "np.ndarray[np.float64, np.float32]"]],
    Optional[Tuple[int, int, int]],
]:
    """Loads an image file.
    Load an image file if it is a stack, dm3, dm4, or bitmap. As of now,
    Fiji doesn't allow easy loading of dm4's so be warned that alignment
    for dm4 files probably won't work.
    Args:
        img_path: Full path to the location of the image.
        graph_size: The size of the graph in (x, y) coords.
        key: The key of the element clicked for loading an image.
        stack: Boolean value if the image is a stack. Default is False.
        prefix: The prefix value for the print statements to the GUI log.
            Default is True.
    Returns:
        tuple: Tuple containing three items:
            - uint8_data: The uint8 data dictionary with image/stack slice key
              and value of uint8 dtype ndarray or None if loading failed.
            - float_data: The float data dictionary with image/stack slice key
              and value of float dtype ndarray or None if loading failed.
            - size: The x, y, z size of the data or None if the loading failed.
    """
    try:
        # Check path has correct filetype
        correct_end = False
        for end in [".tif", ".tiff", ".dm3", ".dm4", ".bmp"]:
            if img_path.endswith(end):
                correct_end = True
        if not correct_end:
            if "Stage" in key or "Align" in key:
                print(
                    f'{prefix}Trying to load an incorrect file type. Acceptable values "tif" are "tiff".'
                )
            elif "FLS" in key:
                print(
                    f'{prefix}Trying to load an incorrect file type. Acceptable values "tif", "tiff", "dm3", "dm4".'
                )
            raise
        # Load data into numpy arrays for processing in GUI.
        uint8_data, float_data = {}, {}
        z_size = 1
        img = hs.load(img_path)
        img_data = img.data
        shape = img_data.shape

        # Differetiate between loading of stack and single image, raise an exception if
        # loading a single image for a stack.
        if stack and len(shape) > 1:
            z_size, y_size, x_size = shape
        elif not stack:
            y_size, x_size = shape
        else:
            print(f"{prefix}Do not try loading a single image to stack.")
            raise
        for z in range(z_size):
            if stack:
                temp_data = img_data[z, :, :]
            else:
                temp_data = img_data
            # Scale data for graph, convert to uint8, and store float data as well
            uint8_data, float_data = convert_float_unint8(
                temp_data, graph_size, uint8_data, float_data, z
            )

        # Return dictionary of uint8 data, a scaled float array, and the shape of the image/stack.
        size = (x_size, y_size, z_size)
        return uint8_data, float_data, size
    # A multitude of errors can cause an image failure, usually with trying to load an incorrect file.
    except (IndexError, TypeError, NameError):
        print(
            f"{prefix}Error. You may have tried loading a file that is not recognized. Try a different file type",
            end="",
        )

        return None, None, None
    # If any other exception just return Nones
    # Usually file might be too big
    except:
        print(f"{prefix}Error. File might be too big. Usually has to be <= 2GB", end="")
        return None, None, None



def array_resize(array: "np.ndarray", new_size: Tuple[int, int]) -> "np.ndarray":
    """Resize numpy arrays.
    Args:
        array: Full path to the location of the image.
        new_size: The new size of array in (x, y) coords, generally size of display.
    Returns:
        resized_array: The resized numpy array.
    """

    resized_array = transform.resize(array, new_size)
    return resized_array


def convert_float_unint8(
    float_array: "np.ndarray[np.float64, np.float32]",
    graph_size: Tuple[int, int],
    uint8_data: Optional[Dict] = None,
    float_data: Optional[Dict] = None,
    z: int = 0,
) -> Tuple[
    Dict[int, "np.ndarray[np.uint8]"], Dict[int, "np.ndarray[np.float64, np.float32]"]
]:
    """Convert float image data to uint8 data, scaling for view in display.
    Images need to be converted to uint8 data for future processing and
    loading into the GUI window.
    Args:
        float_array: Ndarray dtype float of a single slice of image data.
        graph_size: The size of the graph in (x, y) coords.
        uint8_data: The dictionary that stores the uint8 image data.
        float_data: The dictionary that stores the scaled float image data.
        z: The slice key for the uint8 and float dictionaries.
    Returns:
        uint8_data: The uint8 data dictionary with image/stack slice key
                    and value of uint8 dtype ndarray.
        float_data: The float data dictionary with image/stack slice key
                    and value of float dtype ndarray.
    """

    # Initialize data dictionaries if none were passed.
    if uint8_data is None:
        uint8_data = {}
    if float_data is None:
        float_data = {}
    # Scale data so that minimum value will be black and maximum value would be white
    if float_array is not None:
        scaled_float_array = array_resize(float_array, graph_size)
        # Subtract minimum to get a 0 value as min and scale max value to 1.
        # Mutliply by 255 for conversion to uint8.
        resized_data = scaled_float_array - scaled_float_array.min()
        maximum = resized_data.max()
        if maximum == 0:
            maximum = 1
        inv_max = 1 / maximum
        scaled_float_array = resized_data * inv_max
        uint8_array = scaled_float_array * 255
        uint8_array = uint8_array.astype(np_uint8)
        uint8_data[z] = uint8_array
        float_data[z] = scaled_float_array
    return uint8_data, float_data


def apply_rot_transl(
    data: "np.ndarray",
    d_theta: Union[int, float] = 0,
    d_x: Union[int, float] = 0,
    d_y: Union[int, float] = 0,
    h_flip: Optional[bool] = None,
) -> Image.Image:
    """Apply any rotations and translations to an image array. Takes an array of data
    and converts it to PIL
    Images need to be converted to uint8 data for future processing and
    loading into the GUI window.
    Args:
        data: The ndarray of the image data.
        d_theta: The angle with which to rotate the image data.
        d_x: The x-translation to move the image data.
        d_y: The y-translation to move the image data.
        h_flip: Boolean value of whether to flip the image horizontally.
    Returns:
        rgba_img: The PIL rgba image of the image data.
    """

    # Apply horizontal flip if necessary
    if h_flip:
        data = np.fliplr(data)

    # Convert to PIL Image
    rgba_img = Image.fromarray(data).convert("RGBA")

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
    left, top, right, bottom = (
        box_diff,
        box_diff,
        new_side - box_diff,
        new_side - box_diff,
    )
    crop_box = (left, top, right, bottom)
    rgba_img = rgba_img.crop(crop_box)
    return rgba_img


def make_rgba(
    data: "np.ndarray",
    adjust: bool = False,
    d_theta: Union[int, float] = 0,
    d_x: Union[int, float] = 0,
    d_y: Union[int, float] = 0,
    h_flip: Optional[bool] = None,
    color: Optional[bool] = None,
) -> Image.Image:
    """Create an rgba image from numpy ndarray data of an image.
    Rgba images need to be created from uint8 data so that they can be converted to
    bytes for display in the graph. Additionally a colormap may need to be applied along
    with alpha values to show overlaying images.
    Args:
        data: The ndarray of the image data. Can be float or uint8.
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
        if color == "None":
            data = data * 255
            data = data.astype(np_uint8)
        elif color:
            cm = mpl_cm.get_cmap(color)
            data = cm(data, bytes=True)
        else:
            cm = mpl_cm.get_cmap(
                "Spectral"
            )  # spectral, bwr,twilight, twilight_shifted, hsv shows good contrast
            data = cm(data, bytes=True)

        # Apply transformation  if necessary
        rgba_img = apply_rot_transl(data, d_theta, d_x, d_y, h_flip)
    else:
        # Convert to PIL Image
        try:
            rgba_img = Image.fromarray(data).convert("RGBA")
        except:
            rgba_img = Image.fromarray((data * 255).astype(np.uint8))
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
    img.save(byte_img, format="ppm")
    byte_img = byte_img.getvalue()
    return byte_img


def adjust_image(
    data: "np.ndarray[np.float32, np.float64]",
    transform: Tuple[Union[int, float], Union[int, float], Union[int, float], bool],
    image_size: Tuple[int, int],
    graph_size: Tuple[int, int],
) -> Tuple["bytes", Image.Image]:
    """Apply transformations to an image given by some float data.
    Apply rotations, translations, and/or flipping of image. Generally used for stack slices.
    Args:
        data: Float data of image1 in an ndarray.
        transform: Tuple of the rotation, x-translation, y-translation, and
            horizontal flip to apply to the image.
        image_size: The image size in (x, y) size.
        graph_size: The graph size in (x, y) size.
    Returns:
        return_img: The byte image for the data.
        rgba_image: The PIL rgba image of the data.
    """

    # Pull the transformation data, rounding for ease of use for translations.
    d_theta, d_x, d_y, h_flip = transform
    d_x = round(d_x / image_size * graph_size)
    d_y = round(d_y / image_size * graph_size)

    # Create the rgba and the return byte image.
    rgba_image = make_rgba(
        data,
        adjust=True,
        d_theta=d_theta,
        d_x=d_x,
        d_y=d_y,
        h_flip=h_flip,
        color="None",
    )
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


def slice_im(
    image: "np.ndarray", slice_size: Tuple[int, int, int, int]
) -> "np.ndarray":
    """Slice an image
    Args:
        image: ndarray of image data.
        slice_size: The bounds of the image slice in (y_start, x_start, y_end, x_end) coords.
    Returns:
        new_image: An ndarray of the sliced image. Can be 3 or 2 dimensions
    """

    start_y, start_x, end_y, end_x = slice_size
    try:
        new_image = image[start_y:end_y, start_x:end_x, :]
    except:
        new_image = image[start_y:end_y, start_x:end_x]
    return new_image


def add_vectors(mag_x: 'np.ndarray', mag_y: 'np.ndarray', color_np_array: 'np.ndarray', color: bool,
                hsv: bool, arrows: int, length: float, width: float,
                graph_size: Tuple[int, int], pad_info: Tuple[Any, Any],
                GUI_handle: bool = True,
                save: Optional[bool] = None) -> Optional['bytes']:

    """Add a vector plot for the magnetic saturation images to be shown in the GUI.
    Args:
        window: The main GUI window.
        mag_x: The x-component of the magnetic induction.
        mag_y: The y-component of the magnetic induction.
        color_np_array: The colorized magnetic saturation array.
        color: The boolean value for a color image (True) or black & white image (False).
        hsv: The boolean value for hsv color image (True) or 4-fold color image (False).
        arrows: The number of arrows to place along the rows and cols of the image.
        length: The inverse length of the arrows. Inverted when passed to show_2D.
        width: The width of the arrows.
        graph_size: The (y, x) size of the GUI display graph.
        pad_info: The (axis, pad_size) datat. (None, 0, 1) for axes and int for pad_size.
        GUI_handle: The handle to pass to TIE_helper.show_2D() signalling whether to use GUI
            or matplotlib. This defaults to True for the GUI.
        save: The value to determine saving of the vectorized image.
    Returns:
        Optional: The byte image for the vectorized data.
    """

    # Retrieve the image with the added vector plot
    fig, ax = show_2D(
        mag_x,
        mag_y,
        a=arrows,
        l=1 / length,
        w=width,
        title=None,
        color=color,
        hsv=hsv,
        save=save,
        GUI_handle=GUI_handle,
        GUI_color_array=color_np_array,
    )

    if GUI_handle and not save:
        # Get figure and remove any padding
        plt.figure(fig.number)
        fig.tight_layout(pad=0)
        plt.axis("off")
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        # Resize and return byte image suitable for Graph
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if pad_info[0] is not None:
            max_val = max(pad_info[2:])
            pad_side = pad_info[1]
            data = array_resize(data, (max_val, max_val))
            if pad_info[0] == 1:
                npad = ((0, 0), (int(pad_side), int(pad_side)), (0, 0))
                start_x, end_x = pad_side, max_val - pad_side
                start_y, end_y = 0, max_val
            elif pad_info[0] == 0:
                npad = ((int(pad_side), int(pad_side)), (0, 0), (0, 0))
                start_x, end_x = 0, max_val
                start_y, end_y = pad_side, max_val - pad_side
            data = slice_im(data, (start_y, start_x, end_y, end_x))
            data = np.pad(data, pad_width=npad, mode="constant", constant_values=0)
        data = array_resize(data, graph_size)

        # This has the final shape of the graph
        plt.close('all')

        rgba_image = make_rgba(data)
        return_img = convert_to_bytes(rgba_image)

        return return_img
    else:
        plt.close("all")
        return


def apply_crop_to_stack(
    coords: Tuple[int, int, int, int], graph_size: Tuple[int, int], stack: Stack, i: int
) -> Tuple["bytes", Image.Image]:
    """When an ROI mask is selected in the GUI, apply that ROI mask to a slices of image/stack.
    Args:
        coords: The tuple of the corners of the square ROI.
        graph_size: The tuple of the graph size (x, y).
        stack: Stack image object representing the loaded stack.
        i: The slice value of the stack to apply the crop to.
    Returns:
        display_img: The byte image for the data.
        rgba_masked_image: The PIL rgba image of the data.
    """

    # Create mask image (np mask)
    mask = zeros(graph_size, np.uint8)
    coords = [[coords[i][0], coords[i][1]] for i in range(len(coords))]
    mask = create_mask(mask, coords, False)
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
#                 Mask Interaction on GUI Graph                 #
# ============================================================= #
def draw_mask_points(
    winfo: Struct, graph: sg.Graph, current_tab: str, double_click: bool = False
) -> None:
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

    if current_tab == "bunwarpj_tab":
        num_coords = len(winfo.buj_mask_coords)
        for i in range(num_coords):
            x1, y1 = winfo.buj_mask_coords[i]
            x2, y2 = winfo.buj_mask_coords[(i + 1) % num_coords]
            id_horiz = graph.DrawLine((x1 - 7, y1), (x1 + 7, y1), color="red", width=2)
            id_verti = graph.DrawLine((x1, y1 - 7), (x1, y1 + 7), color="red", width=2)
            end = (i + 1) % num_coords
            if end or double_click:
                id_next = graph.DrawLine((x1, y1), (x2, y2), color="red", width=2)
            else:
                id_next = graph.DrawLine((x1, y1), (x1, y1), color="red", width=1)
            winfo.buj_mask_markers.append((id_horiz, id_verti, id_next))
    elif current_tab == "reconstruct_tab":
        num_coords = len(winfo.rec_mask_coords)
        for i in range(num_coords):
            x1, y1 = winfo.rec_mask_coords[i]
            x2, y2 = winfo.rec_mask_coords[(i + 1) % num_coords]
            id_num = graph.DrawLine((x1 - 1, y1), (x2 - 1, y2), color="red", width=1)
            winfo.rec_mask_markers.append(id_num)


def erase_marks(
    winfo: Struct, graph: sg.Graph, current_tab: str, full_erase: bool = False
) -> None:
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

    if current_tab == "bunwarpj_tab":
        for marks in winfo.buj_mask_markers:
            for line in marks:
                graph.DeleteFigure(line)
    elif current_tab == "reconstruct_tab":
        winfo.rec_mask_coords = []
        for line in winfo.rec_mask_markers:
            graph.DeleteFigure(line)



def create_mask(img: 'np.ndarray', mask_coords: Tuple[int],
                bmp: bool = False) -> 'np.ndarray':
    """Create a mask image utilizing corner coordinates and a fill color.
    Args:
        img: The numpy array of the image data.
        mask_coords: The tuple of the corner coordinates of the mask.
        bmp: If a bmp is chosen, create file with 255 color. Else use 1 for fill.

    Returns:
        img: The return mask image numpy array.
    """

    mask_coords = np.asarray(mask_coords)
    rr, cc = draw.polygon(mask_coords[:, 1], mask_coords[:, 0], img.shape)
    mask_img = np.zeros(img.shape, dtype=np.uint8)
    if bmp:
        mask_img[rr, cc] = 255
    else:
        mask_img[rr, cc] = 1
    return mask_img


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
    if center_x <= width // 2:
        left_bounds = True
    if center_x >= graph_x - width // 2:
        right_bounds = True
    if graph_y - center_y <= height // 2:
        top_bounds = True
    if graph_y - center_y >= graph_y - height // 2:
        bottom_bounds = True

    if not left_bounds and not right_bounds:
        x_left = center_x - width // 2
        x_right = center_x + width // 2
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
        y_top = center_y - height // 2
        y_bottom = center_y + height // 2
    elif top_bounds and bottom_bounds:
        y_top = 0
        y_bottom = graph_y
    elif bottom_bounds:
        y_top = 0
        y_bottom = height
    elif top_bounds:
        y_top = graph_y - height
        y_bottom = graph_y

    winfo.rec_mask_coords = [
        (x_left, y_top),
        (x_left, y_bottom),
        (x_right, y_bottom),
        (x_right, y_top),
    ]
