import hyperspy.api as hs
from io import BytesIO
from matplotlib import cm as mpl_cm
from warnings import catch_warnings, simplefilter
with catch_warnings() as w:
    simplefilter('ignore')
from PIL import Image
from sys import platform as sys_platform
import subprocess
from align import join
from numpy import uint8 as np_uint8
from numpy import array, zeros, flipud
from cv2 import INTER_AREA, INTER_NEAREST, resize, flip, fillPoly, imwrite

# Miscellaneous to potentially apply later
# https://imagejdocu.tudor.lu/plugin/utilities/python_dm3_reader/start
# https://python-forum.io/Thread-Tkinter-createing-a-tkinter-photoimage-from-array-in-python3


# ------ Image Loading and Visualization ------ #
def load_image(img_path, graph_size, stack=False, prefix=''):
    """Load in an image"""

    try:
        # Path has correct value
        correct_end = False
        for end in ['.tif', '.tiff', '.dm3', '.dm4', '.bmp']:
            if img_path.endswith(end):
                correct_end = True
        if not correct_end:
            print(f'{prefix}Trying to load an incorrect filetype. Acceptable values "tif", "tiff", "dm3", "dm4", and "bmp".')
            raise
        uint8_data, float_data = {}, {}
        # Load data
        img = hs.load(img_path)
        img_data = img.data
        z_size = 1
        shape = img_data.shape
        if stack and len(shape) > 1:
            z_size, x_size, y_size = shape
        elif not stack:
            x_size, y_size = shape
        else:
            print(f'{prefix}Do not try loading a single image to stack.')
            raise
        for z in range(z_size):
            if stack:
                temp_data = img_data[z, :, :]
            else:
                temp_data = img_data
            # Scale data for graph, convert to uint8
            uint8_data, float_data = convert_float_unint8(temp_data, graph_size, uint8_data, float_data, z)

        # Return dictionary of uint8 data, a scaled float array, and the shape of the image/stack
        return uint8_data, float_data, (x_size, y_size, z_size)
    except (IndexError, TypeError, NameError):
        print(f'{prefix}You tried loading a file that is not recognized. Try a different file type.')
        return None, None, None
    except:
        return None, None, None


def convert_float_unint8(float_array, graph_size, uint8_data=None, float_data=None, z=0):
    """Convert float data to uint8 data, scaling for view in window."""

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


def make_rgba(data, adjust=False, d_theta=0, d_x=0, d_y=0, h_flip=None, color=None):

    # print("Before data shape: ", data.shape)
    # print("Before data dtype: ", data.dtype)
    # print("Before data type: ", type(data))
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
        box_diff = (new_side-old_side) // 2
        left, top, right, bottom = box_diff, box_diff, new_side-box_diff, new_side-box_diff
        crop_box = (left, top, right, bottom)
        rgba_img = rgba_img.crop(crop_box)
    else:
        # Convert to PIL Image
        rgba_img = Image.fromarray(data).convert('RGBA')
    return rgba_img


def convert_to_bytes(img):
    byte_img = BytesIO()
    img.save(byte_img, format='ppm')
    byte_img = byte_img.getvalue()
    return byte_img


def overlay_images(image1, image2, transform, image_size, graph_size):
    """Visualize overlayed images in the GUI canvas. Takes an
    Image Object and converts uint8 data into byte data the Tk
    canvas can use."""
    d_theta, d_x, d_y, h_flip = transform
    d_x = round(d_x/image_size*graph_size)
    d_y = round(d_y/image_size*graph_size)
    array1 = image1.flt_data[0]
    array2 = image2.flt_data[0]

    source_rgba = make_rgba(array1, adjust=True, d_theta=d_theta, d_x=d_x, d_y=d_y, h_flip=h_flip)
    target_rgba = make_rgba(array2, adjust=True)
    final_img = Image.blend(target_rgba, source_rgba, alpha=.6)

    return_img = convert_to_bytes(final_img)
    return return_img


def adjust_image(data, transform, img_size, graph_size):

    d_theta, d_x, d_y, h_flip = transform
    d_x = round(d_x/img_size*graph_size)
    d_y = round(d_y/img_size*graph_size)
    rgba_image = make_rgba(data, adjust=True, d_theta=d_theta, d_x=d_x, d_y=d_y, h_flip=h_flip, color='None')
    return_img = convert_to_bytes(rgba_image)
    return return_img


def vis_1_im(image, layer=0):
    """Visualize one image in the GUI canvas. Takes an
    Image Object and converts uint8 data into byte data the Tk
    canvas can use.

    Parameters
    ----------
    image : Image Object
        The image object that holds data and path info about
        the image.
    layer : Int
        Default is 0 if viewing single image. If viewing a
        stack, must choose layer slice.
    Returns
    -------
    return_img : bytes
        A byte representation of the image data
        to represent in TK canvas.
    """

    im_data = image.uint8_data[layer]
    rgba_image = make_rgba(im_data)
    return_img = convert_to_bytes(rgba_image)
    return return_img


def slice(image, size):
    """Slice an image"""

    endx, endy = size
    startx, starty = 0, 0
    return image[startx:endx, starty:endy, :]


# ------ Run FIJI Macros ------ #
def run_macro(ijm_macro_script, image_dir, fiji_path):
    # check fiji path
    if sys_platform.startswith('win'):
        add_on = "/ImageJ-win64"
    elif sys_platform.startswith('darwin'):
        add_on = "/Contents/MacOS/ImageJ-macosx"
    elif sys_platform.startswith('linux'):
        add_on = "/ImageJ-linux64"
    fiji_path = fiji_path + add_on

    macro_file = f'{image_dir}/macro.ijm'
    with open(macro_file, 'w') as f:
        f.write(ijm_macro_script)
        f.close()

    cmd = join([fiji_path, "--ij2", "--headless", "--console", "-macro ", macro_file], " ")
    # cmd = [fiji_path, "--ij2", "--headless", "--console", "-macro ", macro_file]

    return cmd



# ------ Mask Interaction on Graph ------ #
def draw_mask_points(winfo, graph, current_tab, double_click=False):
    """Draw markers to appear on BUJ graph"""
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


def erase_marks(winfo, graph, current_tab, full_erase=False):
    """Erase markers off graph. Delete stored markers if full_erase enabled."""
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


def create_mask(winfo, filenames, ref_img):
    """ Save created mask as bmp with a given filename """

    graph = winfo.window['__BUJ_Graph__']
    sizex, sizey = graph.CanvasSize
    coords = winfo.buj_mask_coords
    img = zeros((sizex, sizey), np_uint8)
    pts = array(coords)
    img = fillPoly(img, pts=[pts], color=(255, 255, 255))

    orig_sizex, orig_sizey = ref_img.lat_dims

    img = resize(img, (orig_sizex, orig_sizey), interpolation=INTER_NEAREST)
    img = flipud(img)
    for filename in filenames:
        imwrite(f"{filename}", img)


def draw_square_mask(winfo, graph):

    mask_percent = winfo.rec_mask[0] / 100
    graph_x, graph_y = graph.get_size()

    center_x, center_y = winfo.rec_mask_center
    winfo.rec_mask_coords = []
    left_bounds, right_bounds = False, False
    top_bounds, bottom_bounds = False, False

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


def represents_float(s):
    """Evaluate if the string is an integer.

    Parameters
    ----------
    s : str
        The string to check if float.

    Returns
    -------
    Boolean
    """
    try:
        float(s)
        return True
    except ValueError:
        return False