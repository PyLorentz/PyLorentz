import io
import os
import matplotlib as mpl
import hyperspy.api as hs
from PIL import Image
import sys
import subprocess
from align import join
import numpy as np
# import time
import cv2 as cv

# Miscellaneous to potentially apply later
# https://imagejdocu.tudor.lu/plugin/utilities/python_dm3_reader/start
# https://python-forum.io/Thread-Tkinter-createing-a-tkinter-photoimage-from-array-in-python3


# ------ Image Loading and Visualization ------ #
# Prepare images for transformations
def load_image(img_path, stack=False, stack_names=None):
    try:
        # Load data
        data = {}
        img = hs.load(img_path)
        img_data = img.data
        img_array = None
        z_size = 1
        cm = mpl.cm.get_cmap('gray')
        if stack:
            z_size, x_size, y_size = img_data.shape
        else:
            x_size, y_size = img_data.shape
        for z in range(z_size):
            if stack:
                temp_data = img_data[z, :, :]
            else:
                temp_data = img_data
            # Resize data to fit graph window
            resized_data = cv.resize(temp_data, (512, 512), interpolation=cv.INTER_AREA)
            # Scale for uint8 conversion
            resized_data = resized_data / resized_data.max()

            # Convert to uint8 datatype
            img_array = resized_data
            resized_data = cm(resized_data, bytes=True)
            data[z] = resized_data
        return data, img_array, (x_size, y_size, z_size)
    except (IndexError, TypeError, NameError):
        print('You tried loading a file that is not recognized. Try a different file type.')
        return None, None, None


def make_rgba(data, adjust=False, d_theta=0, d_x=0, d_y=0, h_flip=None, color=None):
    if adjust:
        # Apply colormap and convert to uint8 datatype
        if color:
            cm = mpl.cm.get_cmap(color)

        else:
            cm = mpl.cm.get_cmap('Spectral')    # spectral, bwr,twilight, twilight_shifted, hsv shows good contrast
                                            # tab20c, gnuplot2, terrain, nipy_spectral, gist_stern, jet, PuRd
        data = cm(data, bytes=True)

        # print("Adjusting data shape: ", data.shape)
        # print("Adjusting data dtype: ", data.dtype)
        # print("Adjusting data type: ", type(data))

        # Apply horizontal flip if necessary
        if h_flip:
            data = cv.flip(data, 1)

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
    byte_img = io.BytesIO()
    img.save(byte_img, format='ppm')
    byte_img = byte_img.getvalue()
    return byte_img


def overlay_images(image1, image2, transform):
    """Visualize overlayed images in the GUI canvas. Takes an
    Image Object and converts uint8 data into byte data the Tk
    canvas can use."""

    d_theta, d_x, d_y, h_flip = transform
    array1 = image1.img_array
    array2 = image2.img_array

    source_rgba = make_rgba(array1, adjust=True, d_theta=d_theta, d_x=d_x, d_y=d_y, h_flip=h_flip)
    target_rgba = make_rgba(array2, adjust=True)
    final_img = Image.blend(target_rgba, source_rgba, alpha=.6)

    return_img = convert_to_bytes(final_img)
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

    im_data = image.data[layer]
    rgba_image = make_rgba(im_data)
    return_img = convert_to_bytes(rgba_image)
    return return_img


# ------ Run FIJI Macros ------ #
def run_macro(ijm_macro_script, image_dir, fiji_path):
    # check fiji path
    if sys.platform.startswith('win'):
        add_on = "/ImageJ-win64"
    elif sys.platform.startswith('darwin'):
        add_on = "/Contents/MacOS/ImageJ-macosx"
    elif sys.platform.startswith('linux'):
        add_on = "/ImageJ-linux64"
    fiji_path = fiji_path + add_on

    macro_file = f'{image_dir}/macro.ijm'
    with open(macro_file, 'w') as f:
        f.write(ijm_macro_script)
        f.close()

    cmd = join([fiji_path, "--ij2", "--headless", "--console", "-macro ", macro_file], " ")
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True) #stdout=subprocess.PIPE
    return proc
    # # output = None
    # while True:
    #     continue


# ------ Mask Interaction on Graph ------ #
def draw_mask_points(winfo, graph, double_click=False):
    """Draw markers to appear on BUJ graph"""
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


def erase_marks(winfo, graph, full_erase=False):
    """Erase markers off graph. Delete stored markers if full_erase enabled."""
    for marks in winfo.buj_mask_markers:
        for line in marks:
            graph.DeleteFigure(line)

    if full_erase:
        winfo.buj_mask_coords = []
        winfo.buj_mask_markers = []
        winfo.buj_graph_double_click = False


def create_mask(winfo, filenames, ref_img):
    """ Save created mask as bmp with a given filename """

    graph = winfo.window['__BUJ_Graph__']
    sizex, sizey = graph.CanvasSize
    coords = winfo.buj_mask_coords
    img = np.zeros((sizex, sizey), np.uint8)
    pts = np.array(coords)
    img = cv.fillPoly(img, pts=[pts], color=(255, 255, 255))

    orig_sizex, orig_sizey = ref_img.lat_dims

    img = cv.resize(img, (orig_sizex, orig_sizey), interpolation=cv.INTER_NEAREST)
    img = np.flipud(img)
    for filename in filenames:
        cv.imwrite(f"{filename}", img)

