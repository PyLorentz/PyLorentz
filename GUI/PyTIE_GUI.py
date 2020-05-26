import warnings
import PySimpleGUI as sg
import os
from io import StringIO
import align
import gui_help as g_help
from gui_styling import WindowStyle
from gui_layout import window_ly, keys, save_window_ly
import numpy as np
from matplotlib import colors
import itertools
import sys
import time
sys.path.append("../PyTIE/")
from microscopes import Microscope
from TIE_helper import *
from TIE_reconstruct import TIE, SITIE
from colorwheel import colorwheel_HSV, colorwheel_RGB

# print = sg.Print
# sg.main()

# !!!!!!!!!!!!!!!!!! CHANGE DEFAULTS BELOW !!!!!!!!!!!!!!!!!! #
DEFAULTS = {'fiji_dir': '/Applications/Fiji.app',
            'browser_dir': '/Users/timothycote/Box/dataset1_tim'}
# !!!!!!!!!!!!!!!!!! CHANGE DEFAULTS ABOVE !!!!!!!!!!!!!!!!!! #

# ---------------- Styling (styling.py) -------- #
style = WindowStyle()


# ------------------------------- Window Functionality and Event Handling --------------------------------- #
# ------------- Classes ------------- #
class Struct(object):
    """The data structure for saving GUI info, image info, and
     reconstruction info."""
    pass


class File_Object(object):
    """The File_Object Class contains data pathname
    and shortname data for file."""

    def __init__(self, path):
        self.path = path
        self.shortname = ''
        self.shorten_name()

    def shorten_name(self):
        """Creates a string of the path name with only the direct parent
        "image_dir" and the child of "image_dir".

        """
        index = self.path.rfind('/') + 1
        self.shortname = self.path[index:]


class Image(File_Object):
    """The Image Class contains data about an individual image.
    This data is encoded into bytes for the TK Canvas."""

    def __init__(self, uint8_data, flt_data, size, path):
        super().__init__(path)
        if uint8_data is not None:
            self.uint8_data = uint8_data                   # Uint8 image data
            self.flt_data = flt_data                       # Numerical image array
            self.x_size, self.y_size, self.z_size = size
            self.lat_dims = self.x_size, self.y_size
            self.byte_data = None                           # Byte data


class Stack(Image):
    """The Stack Class contains data about an image stack.
    This data is encoded into bytes for the TK Canvas. It
    is a subclass of the Image Class.

    uint8 data is a dictionary where each key is a slice of a stack
    flt_array is the original np array of all the image data in z, y, x
    format
    size is the size of the float array but relayed as x, y, z"""

    def __init__(self, uint8_data, flt_data, size, path):
        super().__init__(uint8_data, flt_data, size, path)
        self.stack_byte_data()

    def stack_byte_data(self):
        self.byte_data = {}
        for pic in range(self.z_size):
            self.byte_data[pic] = g_help.vis_1_im(self, pic)


# ------------- Initialize and reset ------------- #
def init_ls(winfo):
    """Initialize Linear Sift Tab variables

    Parameters
    ----------
    winfo - Struct Class
        A data structure that holds a information about images and
        window.

    Returns
    -------
    None
    """
    # Declare image directory and image storage
    winfo.ls_image_dir = ''
    winfo.ls_images = {}

    # Declare transformation variables
    winfo.ls_transform = (0, 0, 0, 1)
    winfo.ls_past_transform = (0, 0, 0, 1)

    # --- Set up loading files --- #
    winfo.ls_file_queue = {}
    winfo.ls_queue_disable_list = []


def init_buj(winfo):
    """Initialize bUnwarpJ Tab variables

    Parameters
    ----------
    winfo - Struct Class
        A data structure that holds a information about images and
        window.

    Returns
    -------
    None
    """
    # Declare image path and image storage
    winfo.buj_image_dir = ''
    winfo.buj_images = {}

    # --- Set up loading files --- #
    winfo.buj_file_queue = {}
    winfo.buj_queue_disable_list = []

    # Stack selection
    winfo.buj_last_stack_choice = None

    # Declare transformation timers and related variables
    winfo.buj_transform = (0, 0, 0, 1)
    winfo.buj_past_transform = (0, 0, 0, 1)

    # Graph and mask making
    winfo.buj_graph_double_click = False
    winfo.buj_mask_coords = []
    winfo.buj_mask_markers = []


def init_rec(winfo, window):
    """Initialize Reconstructuion Tab variables

    Parameters
    ----------
    winfo - Struct Class
        A data structure that holds a information about images and
        window.

    Returns
    -------
    None
    """
    # Declare image path and image storage
    winfo.rec_image_dir = ''
    winfo.rec_images = {}
    winfo.rec_fls_files = [None, None]
    winfo.rec_ptie = None
    winfo.rec_microscope = None
    winfo.rec_colorwheel = None

    # --- Set up loading files --- #
    winfo.rec_file_queue = {}
    winfo.rec_queue_disable_list = []

    # Declare transformation timers and related variables
    winfo.rec_transform = (0, 0, 0, None)
    winfo.rec_past_transform = (0, 0, 0, None)
    winfo.rec_mask_timer = (0,)
    winfo.rec_mask = (100,)
    winfo.rec_past_mask = (100,)
    winfo.img_scale = None

    graph_size = window['__REC_Graph__'].metadata['size']
    winfo.rec_mask_center = ((graph_size[0]-2)/2, (graph_size[1])/2)

    # Image selection
    winfo.rec_last_image_choice = None

    # Graph and mask making
    winfo.rec_graph_double_click = False
    winfo.rec_mask_coords = []
    winfo.rec_mask_markers = []


def init(winfo, window):
    """Initialize all window and event variables

    Parameters
    ----------
    winfo - Struct Class
        A data structure that holds a information about images and
        window.
    window - PySimpleGUI Window Element
        The python representation of the window GUI.

    Returns
    -------
    None
    """
    # --- Set up window and tabs --- #
    winfo.window = window
    winfo.invis_graph = window.FindElement("__invisible_graph__")
    winfo.tabnames = ["Home", "Registration", "Linear Stack Alignment with SIFT", "bUnwarpJ", "Phase Reconstruction"]
    winfo.pages = "pages_tabgroup"
    winfo.current_tab = "home_tab"

    # --- Set up FIJI/ImageJ --- #
    winfo.fiji_path = ""

    # --- Set up FIJI/ImageJ --- #
    winfo.rotxy_timers = 0, 0, 0

    # --- Set up linear SIFT tab --- #
    init_ls(winfo)

    # --- Set up bUnwarpJ tab --- #
    init_buj(winfo)

    # --- Set up bUnwarpJ tab --- #
    init_rec(winfo, window)

    # --- Set up event handling and bindings --- #
    winfo.true_element = None
    winfo.window.bind("<Button-1>", 'Window Click')
    winfo.window['__BUJ_Graph__'].bind('<Double-Button-1>', 'Double Click')
    winfo.window['__REC_Graph__'].bind('<Double-Button-1>', 'Double Click')
    for key in list(itertools.chain(keys['input'], keys['radio'], keys['graph'], keys['combo'],
                                    keys['checkbox'], keys['slider'], keys['button'],
                                    keys['multiline'], keys['listbox'])):
        winfo.window[key].bind("<Enter>", '+HOVER+')
        winfo.window[key].bind("<Leave>", '+STOP_HOVER+')


def reset(winfo, window, current_tab):
    """Reset the current tab values to be empty or defaults

    Parameters
    ----------
    winfo : Struct Class
        A data structure that holds a information about images and
        window.
    window : PySimpleGUI Window Element
        The python representation of the window GUI.
    current_tab : str
        The key of the current tab being viewed in the window.

    Returns
    -------
    None
    """
    # Reset timers
    winfo.rotxy_timers = 0, 0, 0

    if current_tab == "ls_tab":
        graph = window['__LS_Graph__']
        graph.Erase()
        metadata_change(window, ['__LS_Image1__', '__LS_Image2__', '__LS_Stack__'], reset=True)
        toggle(window, ['__LS_Image1__', '__LS_Image2__', '__LS_Stack__',
                        '__LS_Adjust__', '__LS_View_Stack__', '__LS_Set_Img_Dir__'], state='Def')
        update_values(window, [('__LS_Image_Dir_Path__', ""),
                               ('__LS_transform_x__', '0'), ('__LS_transform_y__', '0'), ('__LS_transform_rot__', "0"),
                               ('__LS_horizontal_flip__', True)])
        update_slider(window, [('__LS_Stack_Slider__', {"value": 0, "slider_range": (0, 2)})])
        change_visibility(window, [('__LS_Stack_Slider__', False)])

        # Declare image path and related variables
        init_ls(winfo)

    elif current_tab == 'bunwarpj_tab':
        graph = window['__BUJ_Graph__']
        graph.Erase()
        metadata_change(window, ['__BUJ_Image1__', '__BUJ_Image2__', '__BUJ_Stack__',
                                 '__BUJ_Flip_Stack_Inp__', '__BUJ_Unflip_Stack_Inp__',
                                 '__BUJ_Unflip_Mask_Inp__', '__BUJ_Flip_Mask_Inp__'], reset=True)
        toggle(window, ['__BUJ_Image1__', '__BUJ_Image2__', '__BUJ_Stack__',
                        '__BUJ_Adjust__', '__BUJ_View_Stack__', '__BUJ_Make_Mask__',
                        '__BUJ_Flip_Stack_Inp__', '__BUJ_Unflip_Stack_Inp__',
                        '__BUJ_Unflip_Mask_Inp__', '__BUJ_Flip_Mask_Inp__',
                        '__BUJ_Set_Img_Dir__'], state='Def')
        update_values(window, [('__BUJ_Image_Dir_Path__', ""),
                               ('__BUJ_transform_x__', '0'), ('__BUJ_transform_y__', '0'),
                               ('__BUJ_transform_rot__', "0"), ('__BUJ_horizontal_flip__', True)])
        change_visibility(window, [('__BUJ_Stack_Slider__', False)])

        # Re-init bUnwarpJ
        init_buj(winfo)

    elif current_tab == 'reconstruct_tab':
        graph = window['__REC_Graph__']
        colorwheel_graph = window['__REC_Colorwheel_Graph__']
        graph.Erase()
        colorwheel_graph.Erase()
        metadata_change(window, ['__REC_Stack__', '__REC_FLS1__', '__REC_FLS2__'], reset=True)
        toggle(window, ['__REC_Set_Img_Dir__', '__REC_FLS_Combo__', '__REC_TFS_Combo__',
                        '__REC_Stack__', '__REC_FLS1__', '__REC_FLS2__',
                        '__REC_Mask__', '__REC_View__'], state='Def')
        update_values(window, [('__REC_Image_Dir_Path__', ""),
                               ('__REC_transform_x__', '0'), ('__REC_transform_y__', '0'),
                               ('__REC_transform_rot__', "0"), ('__REC_Mask_Size__', '50')])

        # Re-init bUnwarpJ
        init_rec(winfo, window)


# ------------- Path and Fiji Helper Functions ------------- #
def load_ls_sift_params(vals):
    """ Convert the values of the GUI inputs for the Linear
    SIFT alignment from strings into FIJI's values to read
    into macro.

    Parameters:
    vals - dict
        The dictionary for key-value pairs for the Linear
        SIFT Alignment of FIJI.

    Returns:
    sift_params : dict
        The converted dictionary of ints, floats, strs for
        Linear SIFT Alignment.
    """
    sift_params = {'igb': float(vals['__LS_igb__']),
                   'spso': int(vals['__LS_spso__']),
                   'min_im': int(vals['__LS_min_im__']),
                   'max_im': int(vals['__LS_max_im__']),
                   'fds': int(vals['__LS_fds__']),
                   'fdob': int(vals['__LS_fdob__']),
                   'cnc': float(vals['__LS_cncr__']),
                   'max_align_err': float(vals['__LS_max_al_err__']),
                   'inlier_rat': float(vals['__LS_inlier_rat__']),
                   'exp_transf': vals['__LS_exp_transf__'],
                   'interpolate': vals['__LS_interp__']}
    return sift_params


def load_buj_ls_sift_params(vals):
    """ Convert the values of the GUI inputs for the bUnwarpJ
    procedure Linear SIFT alignment from strings into
    FIJI's values to read into macro.

    Parameters:
    vals - dict
        The dictionary for key-value pairs for the
        Linear SIFT alignment procedure for bUnwarpJ
        of FIJI.

    Returns:
    sift_params : dict
        The converted dictionary of ints, floats, strs for
        Lin. SIFT Alignment for bUnwarpJ procedure
    """
    sift_params = {'igb': float(vals['__BUJ_LS_igb__']),
                   'spso': int(vals['__BUJ_LS_spso__']),
                   'min_im': int(vals['__BUJ_LS_min_im__']),
                   'max_im': int(vals['__BUJ_LS_max_im__']),
                   'fds': int(vals['__BUJ_LS_fds__']),
                   'fdob': int(vals['__BUJ_LS_fdob__']),
                   'cnc': float(vals['__BUJ_LS_cncr__']),
                   'max_align_err': float(vals['__BUJ_LS_max_al_err__']),
                   'inlier_rat': float(vals['__BUJ_LS_inlier_rat__']),
                   'exp_transf': vals['__BUJ_LS_exp_transf__'],
                   'interpolate': vals['__BUJ_LS_interp__']}
    return sift_params


def load_buj_feat_ext_params(vals):
    """ Convert the values of the GUI inputs for the bUnwarpJ
    feature extraction parameters from strings into
    FIJI's values to read into macro.

    Parameters:
    vals - dict
        The dictionary for key-value pairs for the
        bUnwarpJ parameters.

    Returns:
    sift_params : dict
        The converted dictionary of ints, floats, strs for
        for feature extraction bUnwarpJ procedure.
    """
    sift_params = {'igb': float(vals['__BUJ_igb__']),
                   'spso': int(vals['__BUJ_spso__']),
                   'min_im': int(vals['__BUJ_min_im__']),
                   'max_im': int(vals['__BUJ_max_im__']),
                   'fds': int(vals['__BUJ_fds__']),
                   'fdob': int(vals['__BUJ_fdob__']),
                   'cnc': float(vals['__BUJ_cncr__']),
                   'filter_param': vals['__BUJ_filter__'],
                   'max_align_err': float(vals['__BUJ_max_al_err__']),
                   'inlier_rat': float(vals['__BUJ_inlier_rat__']),
                   'min_num_inls': int(vals['__BUJ_min_num_inlier__']),
                   'exp_transf': vals['__BUJ_exp_transf__']}
    return sift_params


def load_buj_params(vals):
    """ Convert the values of the GUI inputs for the bUnwarpJ
    main parameters from strings into
    FIJI's values to read into macro.

    Parameters:
    vals - dict
        The dictionary for key-value pairs for the
        bUnwarpJ parameters.

    Returns:
    sift_params : dict
        The converted dictionary of ints, floats, strs for
        for main bUnwarpJ procedure.
    """
    buj_params = {'reg_mode': vals['__BUJ_reg_mode__'],
                  'img_sub_factor': int(vals['__BUJ_img_subsf__']),
                  'init_def': vals['__BUJ_init_def__'],
                  'final_def': vals['__BUJ_final_def__'],
                  'div_weight': float(vals['__BUJ_div_w__']),
                  'curl_weight': float(vals['__BUJ_curl_w__']),
                  'landmark_weight': float(vals['__BUJ_land_w__']),
                  'img_weight': float(vals['__BUJ_img_w__']),
                  'cons_weight': float(vals['__BUJ_cons_w__']),
                  'stop_thresh': float(vals['__BUJ_stop_thresh__'])}
    return buj_params


# ------------- Window Helper Functions ------------- #
def get_open_tab(winfo, tabgroup):
    """Recursively determine which tab is open.

    Parameters
    ----------
    winfo : Struct Class
        The data structure holding all information about
        windows and loaded images.
    tabgroup : str
        The key of the tabgroup.

    Returns
    -------
    tab_key : str
        The key for the current open tab.
    """
    # Recursively go through tabgroups and tabs to find
    # current tab.
    tab_key = winfo.window[tabgroup].Get()
    tab = winfo.window[tab_key]
    tab_dict = tab.metadata
    child_tabgroup = tab_dict["child_tabgroup"]
    if child_tabgroup:
        tab_key = get_open_tab(winfo, child_tabgroup)
        return tab_key
    return tab_key


def get_orientation(window, pref):
    """Get the current orientation value for the
    current window.

    Parameters
    ----------
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    pref : str
        The prefix for the the key of
        the orientation for the window.

    Returns
    -------
    orientation : str
        The orientation the current image should be.
    """
    if window[f'__{pref}_unflip_reference__'].Get():
        orientation = 'unflip'
    elif window[f'__{pref}_flip_reference__'].Get():
        orientation = 'flip'
    return orientation


def get_mask_transform(winfo, window, current_tab):

    timers = winfo.rec_mask_timer
    old_transform = winfo.rec_past_mask
    new_transform = [window['__REC_Mask_Size__'].Get()]
    transf_list = [(new_transform[0], timers[0], 0)]
    transform = retrieve_transform(winfo, window, current_tab, transf_list,
                                   old_transform, new_transform, mask=True)
    return transform


def retrieve_transform(winfo, window, current_tab, transf_list,
                       old_transform, new_transform, mask=False):
    """Return transformation to apply to image based off correct
    inputs and timers.

    Parameters
    ----------
    winfo : Struct Class
        The data structure holding all information about
        windows and loaded images.
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    current_tab : str
        The key representing the current main tab of the
        window.
    rotxy_list : list of tuples
        The list containing the tuple that has values and
        timers for each of rotation, x, and y inputs.
    old_transform : tuple of ints, floats
        The previous transformation that was applied to img.
    new_transform : tuple of ints, floats
        The next transformation to potentially apply to
        img.

    Returns
    -------
    transform : tuple of ints, floats
        The transformation to apply to the img.
    """
    # Set transform to old_transform in case no changes made
    transform = old_transform

    # Cycle through each transformation: rotation, x-shift, y-shift
    timer_triggered, val_triggered = False, False
    if mask:
        timers = [0]
    else:
        timers = [0, 0, 0]
    timer_cutoff = 15  # timeout

    # Loop through rotation, x, and y values
    val_set = False
    for val, timer, i in transf_list:
        # If not int, "", or "-", don't increase timer
        if not g_help.represents_float(val) and not mask:
            val_triggered = True
            timer += 1
            if val not in ["", "-", '.']:
                val = '0'
                val_set = True
                timer = timer_cutoff
        # Don't increase timer for mask size if '' or "."
        elif not g_help.represents_float(val) and mask:
            val_triggered = True
            timer += 1
            if val not in ["", '.']:
                val = '50'
                val_set = True
                timer = timer_cutoff
        else:
            if float(val) > 100 and mask:
                val = '100'
                val_set = True
                timer = timer_cutoff
            else:
                timer = 0

        # Timer triggered
        if timer == timer_cutoff:
            timer_triggered = True
            timer = 0
            if not val_set:
                val = '0'

        timers[i], new_transform[i] = timer, val

    # Update timers
    if not mask:
        winfo.rotxy_timers = tuple(timers)
    else:
        winfo.rec_mask_timer = tuple(timers)

    # Check if triggers are set
    if (timer_triggered or not val_triggered) and not mask:
        transform = update_rotxy(winfo, window, current_tab, tuple(new_transform))
    elif (timer_triggered or not val_triggered) and mask:
        transform = update_mask_size(winfo, window, tuple(new_transform))
    return transform


def get_transformations(winfo, window, current_tab):
    """ Gets transformations from the event window.
    Timers give user a limited amount of time before
    the rotation or shift is cleared.

    Parameters
    ----------
    winfo : Struct Class
        A data structure that holds a information about images and
        window.
    window : PySimpleGUI Window Element
        The python representation of the window GUI.
    current_tab : str
        The key of the current tab being viewed in the window.

    Returns
    -------
    transform : tuple of ints, floats
        A tuple of the transformation variables for
        rotation, x-translate, y-translate, and flip.
    """

    # Grab the timers for the inputs
    timers = winfo.rotxy_timers

    # Grab the transformation for the open tab
    if current_tab == "ls_tab":
        pref = "LS"
        old_transform = winfo.ls_past_transform
    elif current_tab == "bunwarpj_tab":
        pref = "BUJ"
        old_transform = winfo.buj_past_transform
    elif current_tab == 'reconstruct_tab':
        pref = "REC"
        old_transform = winfo.rec_past_transform

    # Get the current values of the potential transformation
    hflip = None
    if current_tab in ["ls_tab", "bunwarpj_tab"]:
        hflip = window[f'__{pref}_horizontal_flip__'].Get()
    new_transform = [window[f'__{pref}_transform_rot__'].Get(),
                     window[f'__{pref}_transform_x__'].Get(),
                     window[f'__{pref}_transform_y__'].Get(),
                     hflip]

    # Create list of transform input values and timers to cycle through and change
    rotxy_list = [(new_transform[i], timers[i], i) for i in range(len(timers))]
    transform = retrieve_transform(winfo, window, current_tab, rotxy_list,
                                   old_transform, new_transform)
    return transform


def load_file_queue(winfo, window, current_tab):
    """Loop through unloaded images and check whether they
    exist. If they do, load that file and remove it from the
    queue. FIFO loading preferred.

    Parameters
    ----------
    winfo : Struct Class
        A data structure that holds a information about images and
        window.
    window : PySimpleGUI Window Element
        The python representation of the window GUI.
    current_tab : str
        The key for the current tab where the function is running.

    Returns
    -------
    None"""

    # Loop through items in the queue, checking if they exist
    # If they do load image and save data
    delete_indices = []
    disable_elem_list = []
    if current_tab == "ls_tab":
        queue = winfo.ls_file_queue
    elif current_tab == "bunwarpj_tab":
        queue = winfo.buj_file_queue
    elif current_tab == "reconstruct_tab":
        queue = winfo.rec_file_queue
    for key in queue:
        filename, image_key, target_key, current_tab, keys, proc = queue[key]
        if proc is None:
            poll = True
        else:
            poll = proc.poll()
        # Does file exist?
        if os.path.exists(filename) and poll is not None:
            with warnings.catch_warnings():
                try:
                    # Is file loading correctly?
                    warnings.filterwarnings('error')
                    if current_tab == 'reconstruct_tab':
                        graph = window['__REC_Graph__']
                        graph_size = graph.get_size()
                        uint8_data, flt_data, size = g_help.load_image(filename, graph_size, stack=True)
                    elif current_tab == 'bunwarpj_tab':
                        graph = window['__BUJ_Graph__']
                        graph_size = graph.get_size()
                        uint8_data, flt_data, size = g_help.load_image(filename, graph_size, stack=True)
                    elif current_tab == 'ls_tab':
                        graph = window['__LS_Graph__']
                        graph_size = graph.get_size()
                        uint8_data, flt_data, size = g_help.load_image(filename, graph_size, stack=True)
                    if uint8_data:
                        stack = Stack(uint8_data, flt_data, size, filename)
                        if current_tab == "ls_tab":
                            winfo.ls_images[image_key] = stack
                        elif current_tab == "bunwarpj_tab":
                            winfo.buj_images[image_key] = stack
                        elif current_tab == "reconstruct_tab":
                            winfo.rec_images[image_key] = stack
                        delete_indices.append(key)
                        metadata_change(window, [(target_key, stack.shortname)])
                        toggle(window, [target_key], state="Set")
                        print(f'The file {stack.shortname} was loaded.')
                    else:
                        # Incorrect file loaded, don't keep iterating through it
                        delete_indices.append(key)
                except ValueError:
                    print('Value Error')
                    raise
                except UserWarning:
                    disable_elem_list = disable_elem_list + keys
        # Process finished but nothing made
        elif poll is not None:
            delete_indices.append(key)
            print('FIJI did not complete its task successfully!')
        else:
            disable_elem_list = disable_elem_list + keys

    # Delete any indices that have been successfully loaded
    # Add load and create stack buttons to disable list
    if current_tab == "ls_tab":
        for key in delete_indices:
            del winfo.ls_file_queue[key]
        winfo.ls_queue_disable_list = disable_elem_list
    elif current_tab == "bunwarpj_tab":
        for key in delete_indices:
            del winfo.buj_file_queue[key]
        winfo.buj_queue_disable_list = disable_elem_list
    elif current_tab == "reconstruct_tab":
        for key in delete_indices:
            del winfo.rec_file_queue[key]
        winfo.rec_queue_disable_list = disable_elem_list


# ------------- Changing Element Values ------------- #
def update_values(window, elem_val_list):
    """ Take a list of element key, value tuple pairs
    and update value of the element.

    Parameters
    ----------
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    elem_val_list : list of PySimpleGUI Elements
        The list of elements whose state is to be changed.

    Returns
    -------
    None
    """

    for elem_key, value in elem_val_list:
        if elem_key in keys['button']:
            window[elem_key].Update(text=value)
        elif elem_key in keys['multiline']:
            window[elem_key].Update(value=value, append=True)
        else:
            window[elem_key].Update(value=value)


def metadata_change(window, elem_val_list, reset=False):
    """Change the metadata of the element to update between
    the default value and the user set value.

    Parameters
    ----------
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    elem_val_list : list of tuple(PySimpleGUI Elements, str)
        The list of tuples made of PySimpleGUI elements
        along with the value that the metadata of the
        element state 'Set' will change to.
    reset : Boolean
        If true, the 'Set' value is reset to 'Def'.
        Otherwise the value will be 'Set' as defined
        by the user.

    Returns
    -------
    None
    """

    if reset:
        for elem in elem_val_list:
            window[elem].metadata['Set'] = window[elem].metadata['Def']
    else:
        for elem, val in elem_val_list:
            window[elem].metadata['Set'] = val
            if window[elem].metadata['State'] == 'Set':
                update_values(window, [(elem, val)])


def toggle(window, elem_list, state=None):
    """Toggle between the default state and set state
    of an elements metadata.

    Parameters
    ----------
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    elem_list : list of PySimpleGUI Elements
        The list of elements whose state is to be changed.
    state : None or str
        If the state is None, the state is changed
        from Set -> Def or Def -> Set.
        If the state is specified, that state will
        be activated.

    Returns
    -------
    None
    """

    for elem in elem_list:
        if state == 'Def':
            new_state = window[elem].metadata['State'] = 'Def'
        elif state == 'Set':
            new_state = window[elem].metadata['State'] = 'Set'
        else:
            state = window[elem].metadata['State']
            if state == 'Def':
                new_state = 'Set'
            elif state == 'Set':
                new_state = 'Def'
            window[elem].metadata['State'] = new_state
        if new_state in window[elem].metadata:
            value = window[elem].metadata[new_state]
            update_values(window, [(elem, value)])


def update_slider(window, slider_list):
    """ Updates sliders based off passing a list
    with element, dictionary pairs. The dictionary
    contains all values to update.

    Parameters
    ----------
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    slider_list : list of tuple(PySimpleGUI Slider Element, dict)
        List of slider, dictionary tuple pairs where the dictionary
        contains the values to update.

    Returns
    -------
    None
    """

    for slider_key, d in slider_list:
        slider = window[slider_key]
        for key in d:
            if key == "value":
                update_values(window, [(slider_key, d[key])])
            elif key == "slider_range":
                slider_range = d[key]
                slider.metadata["slider_range"] = slider_range
                window[slider_key].Update(range=slider_range)


def update_rotxy(winfo, window, current_tab, new_transform):
    """Update the rotation, x-trans, y-trans, and
    flip coordinates for the transform to apply to
    series of images.

    Parameters
    ----------
    winfo : Struct Class
        The data structure holding all information about
        windows and loaded images.
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    current_tab : str
        The key representing the current main tab of the
        window.
    new_transform : tuple of ints, floats
        The next transformation to potentially apply to
        img.

    Returns
    -------
    transform : tuple of ints, floats
        The transformation to apply to the img.
    """

    rot_val, x_val, y_val, h_flip = new_transform
    transform = float(rot_val), float(x_val), float(y_val), h_flip
    if current_tab == "ls_tab":
        pref = 'LS'
        winfo.ls_transform = transform
    elif current_tab == "bunwarpj_tab":
        pref = 'BUJ'
        winfo.buj_transform = transform
    elif current_tab == "reconstruct_tab":
        pref = 'REC'
        winfo.rec_transform = transform
    rot_key, x_key, y_key = (f'__{pref}_transform_rot__',
                             f'__{pref}_transform_x__',
                             f'__{pref}_transform_y__')
    elem_val_list = [(rot_key, str(rot_val)), (x_key, str(x_val)), (y_key, str(y_val))]
    update_values(window, elem_val_list)
    return transform


def update_mask_size(winfo, window, new_transform):
    """Update the rotation, x-trans, y-trans, and
      flip coordinates for the transform to apply to
      series of images.

      Parameters
      ----------
      winfo : Struct Class
          The data structure holding all information about
          windows and loaded images.
      window : PySimpleGUI Window Element
          The element representing the main GUI window.
      current_tab : str
          The key representing the current main tab of the
          window.
      new_transform : tuple of ints, floats
          The next transformation to potentially apply to
          img.

      Returns
      -------
      transform : tuple of ints, floats
          The transformation to apply to the img.
      """

    mask_size = new_transform[0]
    mask_transform = (float(mask_size), )
    winfo.rec_mask = mask_transform
    mask_size_key = '__REC_Mask_Size__'
    elem_val_list = [(mask_size_key, str(mask_size))]
    update_values(window, elem_val_list)
    return mask_transform


# ------------- Visualizing Elements ------------- #
def set_pretty_focus(winfo, window, event):
    """ Sets the focus to reduce unwanted placements of
    cursor or focus within the GUI. This is done by
    setting unwanted focus to an invisible graph.

    Parameters
    ----------
    winfo : Struct Class
        The data structure holding all information about
        windows and loaded images.
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    event : str
        The key for the values dictionary that represents
        an event in the window.

    Returns
    -------
    None
    """

    # Set the 'true element' to be the one the
    # cursor is hovering over.
    if "+HOVER+" in event and not winfo.true_element:
        winfo.true_element = event.replace("+HOVER+", "")
    elif "+STOP_HOVER+" in event:
        winfo.true_element = None
    # Window click will never set focus on button
    elif event == "Window Click":
        if winfo.true_element and winfo.true_element not in keys['button']:
            window[winfo.true_element].SetFocus()
        else:
            winfo.invis_graph.SetFocus(force=True)


def redraw_graph(graph, display_image):
    """Redraw graph.

    Parameters
    ----------
    graph : PySimpleGUI Graph Element
        The graph element in the window.
    display_image : bytes or None
        If None, the graph is erased
        Else, bytes representation of the image

    Returns
    -------
    None
    """

    graph.Erase()
    if display_image:
        x, y = graph.get_size()
        graph.DrawImage(data=display_image, location=(0, y-1))


def change_visibility(window, elem_val_list):
    """ Take a list of element keys and change
    visibility of the element.

    Parameters
    ----------
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    elem_list : list of PySimpleGUI Elements
        The list of elements whose state is to be changed.

    Returns
    -------
    None
    """
    for elem_key, val in elem_val_list:
        window[elem_key].Update(visible=val)


def disable_elements(window, elem_list):
    """ Take a list of element keys and disable the element.

    Parameters
    ----------
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    elem_list : list of PySimpleGUI Elements
        The list of elements whose state is to be changed.

    Returns
    -------
    None
    """

    for elem_key in elem_list:
        window[elem_key].Update(disabled=True)


def enable_elements(window, elem_list):
    """ Take a list of element keys and enable the element.

    Parameters
    ----------
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    elem_list : list of PySimpleGUI Elements
        The list of elements whose state is to be changed.

    Returns
    -------
    None
    """

    for elem_key in elem_list:
        window[elem_key].Update(disabled=False)


# -------------- Home Tab Event Handler -------------- #
def run_home_tab(winfo, window, event, values):
    """Run events associated with the home tab.

    Parameters
    ----------
    winfo : Struct Class
        The data structure holding all information about
        windows and loaded images.
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    event : str
        The key for the values dictionary that represents
        an event in the window.
    values : dict
        A dictionary where every value is paired with
        a key represented by an event in the window.

    Returns
    -------
    None
    """

    # Get directories for Fiji and images
    if event == '__Fiji_Set__':
        winfo.fiji_path = values['__Fiji_Path__']
        if not os.path.exists(winfo.fiji_path):
            print('This Fiji path is incorrect, try again.')
        else:
            print('Fiji path is set, you may now proceed to registration.')
            disable_elements(window, ['__Fiji_Path__', '__Fiji_Set__', '__Fiji_Browse__'])
            enable_elements(window, ['align_tab'])
    elif event == '__Fiji_Reset__':
        update_values(window, [('__Fiji_Path__', '')])
        winfo.fiji_path = values['__Fiji_Path__']
        enable_elements(window, ['__Fiji_Path__', '__Fiji_Set__', '__Fiji_Browse__'])
        disable_elements(window, ['align_tab'])


# -------------- Linear SIFT Tab Event Handler -------------- #
def run_ls_tab(winfo, window, current_tab, event, values):
    """Run events associated with the linear sift tab.

    Parameters
    ----------
    winfo : Struct Class
        The data structure holding all information about
        windows and loaded images.
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    current_tab : str
        The key representing the current main tab of the
        window. Ex. '
    event : str
        The key for the values dictionary that represents
        an event in the window.
    values : dict
        A dictionary where every value is paired with
        a key represented by an event in the window.

    Returns
    -------
    None
    """
    # ------------- Visualizing Elements ------------- #
    def special_enable_disable(window, adjust_button, view_stack_button, images):
        enable_list = []
        active_keys = ['__LS_View_Stack__', '__LS_Run_Align__', '__LS_Load_Stack__',
                       '__LS_Adjust__', '__LS_unflip_reference__', '__LS_flip_reference__',
                       '__LS_Image_Dir_Path__', '__LS_Set_Img_Dir__', '__LS_Image_Dir_Browse__']
        # Don't view/load any images accidentally when adjusting images
        if adjust_button.metadata['State'] == 'Def':
            enable_list.extend(['__LS_unflip_reference__', '__LS_flip_reference__'])
            if images and 'stack' in images:
                enable_list.append('__LS_View_Stack__')
            # Don't enable load stack when viewing stack
            if view_stack_button.metadata['State'] == 'Def':
                if '__LS_Load_Stack__' not in winfo.ls_queue_disable_list:
                    enable_list.append('__LS_Load_Stack__')
                if window['__LS_Set_Img_Dir__'].metadata['State'] == 'Set':
                    if '__LS_Run_Align__' not in winfo.ls_queue_disable_list:
                        enable_list.append('__LS_Run_Align__')
        if view_stack_button.metadata['State'] == 'Def':
            if window['__LS_Set_Img_Dir__'].metadata['State'] == 'Set':
                enable_list.append('__LS_Adjust__')
        # If image dir is not set
        if window['__LS_Set_Img_Dir__'].metadata['State'] == 'Def':
            enable_list.extend(['__LS_Image_Dir_Path__',
                                '__LS_Set_Img_Dir__',
                                '__LS_Image_Dir_Browse__'])
        disable_list = np.setdiff1d(active_keys, enable_list)
        enable_elements(window, enable_list)
        disable_elements(window, disable_list)

    # Get rotations, shifts and orientation
    transform = get_transformations(winfo, window, current_tab)
    orientation = get_orientation(window, "LS")

    # Grab important elements
    graph = window['__LS_Graph__']
    adjust_button = window['__LS_Adjust__']
    view_stack_button = window['__LS_View_Stack__']

    # Pull in image data from struct object
    image_dir = winfo.ls_image_dir
    images = winfo.ls_images
    display_img = None

    # Import event handler names (overlaying, etc.)
    overlay = adjust_button.metadata['State'] == 'Set' and winfo.ls_past_transform != transform
    scroll = (event in ['MouseWheel:Up', 'MouseWheel:Down']
              and window['__LS_View_Stack__'].metadata['State'] == 'Set'
              and winfo.true_element == "__LS_Graph__")
    change_ref = (event in ['__LS_unflip_reference__', '__LS_flip_reference__'] and
                  view_stack_button.metadata['State'] == 'Def')

    # Set image directory and load in-focus image
    if event == '__LS_Set_Img_Dir__':
        image_dir = values['__LS_Image_Dir_Path__']
        if os.path.exists(image_dir):
            # Check if files match fls file in image_dir
            check = align.check_setup(image_dir)
            if check:
                # Set the image dir button
                toggle(window, ['__LS_Set_Img_Dir__'], state='Set')

                # Prepare reference data
                unflip_ref, flip_ref, unflip_files, flip_files, unflip_path, flip_path = align.collect_image_data(image_dir)

                # Load image data as numpy arrays for uint8, numerical val, and size
                u_uint8, u_flt_data, u_size = g_help.load_image(unflip_ref, graph.get_size())
                f_uint8, f_flt_data, f_size = g_help.load_image(flip_ref, graph.get_size())
                if u_uint8 and f_uint8:
                    # Create image instances and store byte data for TK Canvas
                    unflip_image = Image(u_uint8, u_flt_data, u_size, unflip_ref)
                    flip_image = Image(f_uint8, f_flt_data, f_size, flip_ref)
                    unflip_image.byte_data = g_help.vis_1_im(unflip_image)
                    flip_image.byte_data = g_help.vis_1_im(flip_image)

                    # Display ref filename and load display data
                    if orientation == 'unflip':
                        image = unflip_image
                    elif orientation == 'flip':
                        image = flip_image

                    # Update window only if view stack not set
                    if view_stack_button.metadata['State'] == 'Def':
                        metadata_change(window, [('__LS_Image1__', image.shortname)])
                        toggle(window, ['__LS_Image1__'])
                        display_img = image.byte_data

                    # Push data to winfo
                    winfo.ls_images['unflip'] = unflip_image
                    winfo.ls_images['flip'] = flip_image
                    winfo.ls_image_dir = image_dir
                    print('Directory properly set-up.')
            else:
                print('Look at Help Tab for correct file setup.')
        else:
            print('This pathname is incorrect.')

    # Change reference state between flip/unflip images
    elif change_ref:
        if image_dir:
            image = images[orientation]
            display_img = image.byte_data
            metadata_change(window, [('__LS_Image1__', image.shortname)])

    # Load flipped image for adjustment
    elif event == '__LS_Adjust__':
        if window['__LS_Set_Img_Dir__'].metadata['State'] == 'Set':
            # Quit flip adjustment
            if adjust_button.metadata['State'] == 'Set':
                display_img = images[orientation].byte_data
                toggle(window, ['__LS_Adjust__', '__LS_Image2__'], state='Def')

            # Begin flip adjustment
            elif adjust_button.metadata['State'] == 'Def':
                if orientation == 'unflip':
                    img_1 = images['flip']
                    img_2 = images['unflip']
                elif orientation == 'flip':
                    img_1 = images['unflip']
                    img_2 = images['flip']
                display_img = g_help.overlay_images(img_1, img_2, transform)
                metadata_change(window, [('__LS_Image2__', img_1.shortname)])
                toggle(window, ['__LS_Adjust__', '__LS_Image2__'], state='Set')

        else:
            print('No flip data to adjust, make sure to set your working directory.')

    # Run Linear SIFT alignment
    elif event == '__LS_Run_Align__':
        if window['__LS_Set_Img_Dir__'].metadata['State'] == 'Set':
            param_test = values['__LS_param_test__']
            sift_params = load_ls_sift_params(values)
            rot, x_shift, y_shift, horizontal = transform
            transform_params = rot, x_shift, -y_shift, horizontal

            # Decide whether file should be created
            save, overwrite_signal = True, []
            if param_test:
                filename = align.join([image_dir, "Param_Test.tif"], '/')
            else:
                filename, overwrite_signal = run_save_window(winfo, event, image_dir)
                save = overwrite_signal[0]
                if filename == 'close' or not filename or not save:
                    print('Exited save screen without saving image.')
                    save = False
                else:
                    filename = filename[0]

            # Create files
            if save:
                if os.path.exists(filename):
                    os.remove(filename)
                ijm_macro_script = align.run_ls_align(image_dir, orientation, param_test,
                                                      sift_params, transform_params, filename)
                proc = g_help.run_macro(ijm_macro_script, image_dir, winfo.fiji_path)

                # Load the stack when ready
                target_key = '__LS_Stack__'
                image_key = 'stack'
                align_keys = ['__LS_Load_Stack__', '__LS_Run_Align__']
                winfo.ls_file_queue[event] = (filename, image_key, target_key, current_tab, align_keys, proc)
        else:
            print('A valid directory has not been set.')

    # Load an image stack to be viewed when view stack pressed
    elif event == '__LS_Staging_Load__':
        # Create stack
        filename = values['__LS_Staging_Load__']
        update_values(window, [(event, 'None')])
        target_key = '__LS_Stack__'
        image_key = 'stack'
        align_keys = ['__LS_Load_Stack__', '__LS_Run_Align__']
        winfo.ls_file_queue[event] = (filename, image_key, target_key, current_tab, align_keys, None)

    # View the loaded image stack
    elif event == '__LS_View_Stack__':
        if view_stack_button.metadata['State'] == 'Def':
            # Get stack information
            stack = images['stack']
            slider_val = 0
            slider_range = (0, stack.z_size - 1)
            display_img = stack.byte_data[slider_val]

            # Update window
            metadata_change(window, [('__LS_Image1__', f'Image {slider_val+1}')])
            toggle(window, ['__LS_Adjust__'], state='Def')
            toggle(window, ['__LS_Image1__', '__LS_View_Stack__'], state='Set')
            update_slider(window, [('__LS_Stack_Slider__', {"value": slider_val, "slider_range": slider_range})])
            change_visibility(window, [('__LS_Stack_Slider__', True)])
        elif view_stack_button.metadata['State'] == 'Set':
            # Update window
            if window['__LS_Set_Img_Dir__'].metadata['State'] == 'Set':
                image = images[orientation]
                metadata_change(window, [('__LS_Image1__', image.shortname)])
                display_img = image.byte_data
            else:
                graph.Erase()
                metadata_change(window, ['__LS_Image1__'], reset=True)
                toggle(window, ['__LS_Image1__'])
            toggle(window, ['__LS_View_Stack__'])
            change_visibility(window, [('__LS_Stack_Slider__', False)])

    # Change the slider
    elif event == '__LS_Stack_Slider__':
        # Get image from stack
        stack = images['stack']
        slider_val = int(values["__LS_Stack_Slider__"])

        # Update window
        display_img = stack.byte_data[slider_val]
        metadata_change(window, [('__LS_Image1__', f'Image {slider_val+1}')])

    # Scroll through stacks in the graph area
    elif scroll:
        stack = images['stack']
        slider_val = int(values["__LS_Stack_Slider__"])
        max_slider_val = stack.z_size - 1
        # Scroll up or down
        if event == 'MouseWheel:Down':
            slider_val = min(max_slider_val, slider_val+1)
        elif event == 'MouseWheel:Up':
            slider_val = max(0, slider_val-1)

        # Update the window
        display_img = stack.byte_data[slider_val]
        update_slider(window, [('__LS_Stack_Slider__', {"value": slider_val})])
        metadata_change(window, [('__LS_Image1__', f'Image {slider_val+1}')])

    # Apply any immediate changes
    if overlay:
        if orientation == 'unflip':
            img_1 = images['flip']
            img_2 = images['unflip']
        elif orientation == 'flip':
            img_1 = images['unflip']
            img_2 = images['flip']
        display_img = g_help.overlay_images(img_1, img_2, transform)
    winfo.ls_past_transform = transform

    # Check to see if any files need loading
    if len(winfo.ls_file_queue) > 0:
        load_file_queue(winfo, window, current_tab)

    # Reset the image directory to nothing
    if event == '__LS_Reset_Img_Dir__':
        reset(winfo, window, current_tab)
        images = winfo.ls_images

    # Make sure certain events have happened for buttons to be enabled
    special_enable_disable(window, adjust_button, view_stack_button, images)

    # Redraw graph
    if display_img:
        redraw_graph(graph, display_img)


# -------------- bUnwarpJ Tab Event Handler -------------- #
def run_bunwarpj_tab(winfo, window, current_tab, event, values):
    """Run events associated with the bUnwarpJ tab.
    Parameters
    ----------
    winfo : Struct Class
        The data structure holding all information about
        windows and loaded images.
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    current_tab : str
        The key representing the current main tab of the
        window. Ex. '
    event : str
        The key for the values dictionary that represents
        an event in the window.
    values : dict
        A dictionary where every value is paired with
        a key represented by an event in the window.

    Returns
    -------
    None
    """
    # ------------- Visualizing Elements ------------- #
    def special_enable_disable(window, adjust_button, view_stack_button, make_mask_button,
                               images):
        enable_list = []
        active_keys = ['__BUJ_View_Stack__', '__BUJ_Flip_Align__', '__BUJ_Unflip_Align__',
                       '__BUJ_Elastic_Align__', '__BUJ_Load_Flip_Stack__', '__BUJ_Load_Unflip_Stack__',
                       '__BUJ_Load_Stack__', '__BUJ_Image_Dir_Path__', '__BUJ_Set_Img_Dir__',
                       '__BUJ_Image_Dir_Browse__', '__BUJ_Adjust__', '__BUJ_Make_Mask__',
                       '__BUJ_unflip_reference__', '__BUJ_flip_reference__',
                       '__BUJ_Clear_Unflip_Mask__', '__BUJ_Clear_Flip_Mask__',
                       '__BUJ_Load_Mask__', '__BUJ_Reset_Mask__', '__BUJ_Make_Mask__']
        if adjust_button.metadata['State'] == 'Def' and make_mask_button.metadata['State'] == 'Def':
            enable_list.extend(['__BUJ_unflip_reference__', '__BUJ_flip_reference__'])
            if ('BUJ_flip_stack' in images or 'BUJ_unflip_stack' in images or 'BUJ_stack' in images):
                enable_list.append('__BUJ_View_Stack__')
            if view_stack_button.metadata['State'] == 'Def':
                if '__BUJ_Load_Stack__' not in winfo.buj_queue_disable_list:
                    enable_list.append('__BUJ_Load_Stack__')
                if '__BUJ_Load_Flip_Stack__' not in winfo.buj_queue_disable_list:
                    enable_list.append('__BUJ_Load_Flip_Stack__')
                if '__BUJ_Load_Unflip_Stack__' not in winfo.buj_queue_disable_list:
                    enable_list.append('__BUJ_Load_Unflip_Stack__')
                if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set':
                    if ('__BUJ_Elastic_Align__' not in winfo.buj_queue_disable_list and
                            '__BUJ_Flip_Align__' not in winfo.buj_queue_disable_list and
                            '__BUJ_Unflip_Align__' not in winfo.buj_queue_disable_list):
                        enable_list.append('__BUJ_Flip_Align__')
                        enable_list.append('__BUJ_Unflip_Align__')
                        if 'BUJ_flip_stack' in images and 'BUJ_unflip_stack' in images:
                            enable_list.append('__BUJ_Elastic_Align__')
        if view_stack_button.metadata['State'] == 'Def' and make_mask_button.metadata['State'] == 'Def':
            if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set':
                enable_list.append('__BUJ_Adjust__')
        if view_stack_button.metadata['State'] == 'Def' and adjust_button.metadata['State'] == 'Def':
            enable_list.extend(['__BUJ_Load_Mask__', '__BUJ_Clear_Unflip_Mask__', '__BUJ_Clear_Flip_Mask__'])
            if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set':
                enable_list.extend(['__BUJ_Reset_Mask__', '__BUJ_Make_Mask__'])
        if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Def':
            enable_list.extend(['__BUJ_Image_Dir_Path__', '__BUJ_Set_Img_Dir__',
                                '__BUJ_Image_Dir_Browse__'])

        disable_list = np.setdiff1d(active_keys, enable_list)
        enable_elements(window, enable_list)
        disable_elements(window, disable_list)

    # Get rotations and shifts to apply to image (only positive rotations)
    transform = get_transformations(winfo, window, current_tab)
    orientation = get_orientation(window, "BUJ")

    # Grab important elements
    graph = window['__BUJ_Graph__']
    adjust_button = window['__BUJ_Adjust__']
    view_stack_button = window['__BUJ_View_Stack__']
    make_mask_button = window['__BUJ_Make_Mask__']

    # Pull in image data from struct object
    image_dir = winfo.buj_image_dir
    images = winfo.buj_images
    display_img = None
    draw_mask_points, draw_mask_polygon = False, False

    # Import event handler names (overlaying, etc.)
    overlay = adjust_button.metadata['State'] == 'Set' and winfo.buj_past_transform != transform
    change_ref = (event in ['__BUJ_unflip_reference__', '__BUJ_flip_reference__'] and
                  view_stack_button.metadata['State'] == 'Def')
    scroll = (event in ['MouseWheel:Up', 'MouseWheel:Down']
              and window['__BUJ_View_Stack__'].metadata['State'] == 'Set'
              and winfo.true_element == "__BUJ_Graph__")

    # Set the working directory
    if event == '__BUJ_Set_Img_Dir__':
        image_dir = values['__BUJ_Image_Dir_Path__']
        if os.path.exists(image_dir):
            check = align.check_setup(image_dir)
            if check:
                # Set image button
                toggle(window, ['__BUJ_Set_Img_Dir__'], state='Set')

                # Prepare reference data
                unflip_ref, flip_ref, unflip_files, flip_files, unflip_path, flip_path = align.collect_image_data(image_dir)

                # Load image data as numpy arrays, in byte format, and in rgba for display
                u_uint8, u_flt_data, u_size = g_help.load_image(unflip_ref, graph.get_size())
                f_uint8, f_flt_data, f_size = g_help.load_image(flip_ref, graph.get_size())
                if u_uint8 and f_uint8:
                    # Create image instances and store byte data for TK Canvas
                    unflip_image = Image(u_uint8, u_flt_data, u_size, unflip_ref)
                    flip_image = Image(f_uint8, f_flt_data, f_size, flip_ref)
                    unflip_image.byte_data = g_help.vis_1_im(unflip_image)
                    flip_image.byte_data = g_help.vis_1_im(flip_image)

                    # Display ref filename and load display data
                    if orientation == 'unflip':
                        image = unflip_image
                    elif orientation == 'flip':
                        image = flip_image

                    # Update window
                    if view_stack_button.metadata['State'] == 'Def':
                        metadata_change(window, [('__BUJ_Image1__', image.shortname)])
                        toggle(window, ['__BUJ_Image1__'])
                        display_img = image.byte_data

                    # Push data to winfo
                    winfo.buj_images['unflip'] = unflip_image
                    winfo.buj_images['flip'] = flip_image
                    winfo.buj_image_dir = image_dir
                    print('Directory properly set-up.')
            else:
                print('Look at Help Tab for correct file setup.')
        else:
            print('This pathname is incorrect.')

    # Change reference state between flip/unflip images
    elif change_ref:
        if image_dir:
            image = images[orientation]
            display_img = image.byte_data
            metadata_change(window, [('__BUJ_Image1__', image.shortname)])

    # Load image for rotation/translation adjustment
    elif event == '__BUJ_Adjust__':
        if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set':
            # Quit flip adjustment
            if adjust_button.metadata['State'] == 'Set':
                display_img = images[orientation].byte_data
                toggle(window, ['__BUJ_Adjust__', '__BUJ_Image2__'], state='Def')

            elif adjust_button.metadata['State'] == 'Def':
                if orientation == 'unflip':
                    img_1 = images['flip']
                    img_2 = images['unflip']
                elif orientation == 'flip':
                    img_1 = images['unflip']
                    img_2 = images['flip']
                display_img = g_help.overlay_images(img_1, img_2, transform)
                metadata_change(window, [('__BUJ_Image2__', img_1.shortname)])
                toggle(window, ['__BUJ_Adjust__', '__BUJ_Image2__'], state='Set')
        else:
            print('Unable to adjust, make sure to set your working directory.')

    # Run Linear SIFT alignments
    elif event in ['__BUJ_Flip_Align__', '__BUJ_Unflip_Align__']:
        if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set':
            sift_params = load_buj_ls_sift_params(values)
            if event == '__BUJ_Unflip_Align__':
                orient = 'unflip'
            elif event == '__BUJ_Flip_Align__':
                orient = 'flip'
            filename, overwrite_signal = run_save_window(winfo, event, image_dir, [orient])
            save = overwrite_signal[0]
            if filename == 'close' or not filename or not save:
                print('Exited save screen without saving image.')
                save = False

            # Create the file
            if save:
                # Delete file if it supposed to be overwritten
                filename = filename[0]
                if os.path.exists(filename):
                    os.remove(filename)

                # Execute fiji macro
                ijm_macro_script = align.run_single_ls_align(image_dir, orient, sift_params, filename)
                proc = g_help.run_macro(ijm_macro_script, image_dir, winfo.fiji_path)

                # Load file
                if event == '__BUJ_Unflip_Align__':
                    target_key = '__BUJ_Unflip_Stack_Inp__'
                    align_keys = ['__BUJ_Unflip_Align__', '__BUJ_Load_Unflip_Stack__', '__BUJ_Elastic_Align__']
                elif event == '__BUJ_Flip_Align__':
                    target_key = '__BUJ_Flip_Stack_Inp__'
                    align_keys = ['__BUJ_Flip_Align__', '__BUJ_Load_Flip_Stack__', '__BUJ_Elastic_Align__']
                image_key = f'BUJ_{orient}_stack'
                winfo.buj_file_queue[event] = (filename, image_key, target_key, current_tab, align_keys, proc)
        else:
            print('A valid directory has not been set.')

    # Load in the stacks from Unflip, Flip, or General
    elif event in ['__BUJ_Unflip_Stage_Load__', '__BUJ_Stack_Stage_Load__', '__BUJ_Flip_Stage_Load__']:
        # Load in stacks
        filename = values[event]
        update_values(window, [(event, 'None')])
        if event == '__BUJ_Unflip_Stage_Load__':
            target_key = '__BUJ_Unflip_Stack_Inp__'
            align_keys = ['__BUJ_Unflip_Align__', '__BUJ_Flip_Align__',
                          '__BUJ_Load_Unflip_Stack__', '__BUJ_Elastic_Align__']
            image_key = 'BUJ_unflip_stack'
        elif event == '__BUJ_Flip_Stage_Load__':
            target_key = '__BUJ_Flip_Stack_Inp__'
            align_keys = ['__BUJ_Unflip_Align__', '__BUJ_Flip_Align__',
                          '__BUJ_Load_Flip_Stack__', '__BUJ_Elastic_Align__']
            image_key = 'BUJ_flip_stack'
        elif event == '__BUJ_Stack_Stage_Load__':
            target_key = '__BUJ_Stack__'
            align_keys = ['__BUJ_Unflip_Align__', '__BUJ_Flip_Align__',
                          '__BUJ_Load_Stack__', '__BUJ_Elastic_Align__']
            image_key = 'BUJ_stack'
        winfo.buj_file_queue[event] = (filename, image_key, target_key, current_tab, align_keys, None)

    # View the image stack created from alignment
    elif event == '__BUJ_View_Stack__':
        # Look at which stack to view
        stack_choice = window['__BUJ_Stack_Choice__'].Get()
        if stack_choice == 'Unflip LS':
            stack_key = 'BUJ_unflip_stack'
            disabled = '__BUJ_Load_Unflip_Stack__' in winfo.buj_file_queue
        elif stack_choice == 'Flip LS':
            stack_key = 'BUJ_flip_stack'
            disabled = '__BUJ_Load_Flip_Stack__' in winfo.buj_file_queue
        elif stack_choice == 'bUnwarpJ':
            stack_key = 'BUJ_stack'
            disabled = '__BUJ_Load_Stack__' in winfo.buj_file_queue

        if view_stack_button.metadata['State'] == 'Def':
            if stack_key in images and not disabled:
                stack = images[stack_key]
                slider_val = 0
                slider_range = (0, stack.z_size - 1)
                display_img = stack.byte_data[slider_val]

                # Update window
                metadata_change(window, [('__BUJ_Image1__', f'Image {slider_val + 1}')])
                toggle(window, ['__BUJ_Adjust__'], state='Def')
                toggle(window, ['__BUJ_Image1__', '__BUJ_View_Stack__'], state='Set')
                update_slider(window, [('__BUJ_Stack_Slider__', {"value": slider_val, "slider_range": slider_range})])
                change_visibility(window, [('__BUJ_Stack_Slider__', True)])
            else:
                print("Tried loading unavailable stack, you must perform an alignment.")

        elif view_stack_button.metadata['State'] == 'Set':
            # Update window
            if image_dir:
                image = images[orientation]
                metadata_change(window, [('__BUJ_Image1__', image.shortname)])
                display_img = image.byte_data
            else:
                graph.Erase()
                metadata_change(window, ['__BUJ_Image1__'], reset=True)
                toggle(window, ['__BUJ_Image1__'])
            toggle(window, ['__BUJ_View_Stack__'])
            change_visibility(window, [('__BUJ_Stack_Slider__', False)])
        winfo.buj_last_stack_choice = stack_choice

    # Change the slider
    elif event == '__BUJ_Stack_Slider__':
        stack_choice = window['__BUJ_Stack_Choice__'].Get()
        if stack_choice == 'Unflip LS':
            stack_key = 'BUJ_unflip_stack'
        elif stack_choice == 'Flip LS':
            stack_key = 'BUJ_flip_stack'
        elif stack_choice == 'bUnwarpJ':
            stack_key = 'BUJ_stack'
        stack = images[stack_key]
        slider_val = int(values["__BUJ_Stack_Slider__"])

        # Update window
        display_img = stack.byte_data[slider_val]
        metadata_change(window, [('__BUJ_Image1__', f'Image {slider_val + 1}')])

    # Scroll through stacks in the graph area
    elif scroll:
        stack_choice = window['__BUJ_Stack_Choice__'].Get()
        if stack_choice == 'Unflip LS':
            stack_key = 'BUJ_unflip_stack'
        elif stack_choice == 'Flip LS':
            stack_key = 'BUJ_flip_stack'
        elif stack_choice == 'bUnwarpJ':
            stack_key = 'BUJ_stack'
        stack = images[stack_key]
        slider_val = int(values["__BUJ_Stack_Slider__"])
        max_slider_val = stack.z_size - 1
        # Scroll up or down
        if event == 'MouseWheel:Down':
            slider_val = min(max_slider_val, slider_val+1)
        elif event == 'MouseWheel:Up':
            slider_val = max(0, slider_val-1)

        # Update the window
        display_img = stack.byte_data[slider_val]
        update_slider(window, [('__BUJ_Stack_Slider__', {"value": slider_val})])
        metadata_change(window, [('__BUJ_Image1__', f'Image {slider_val+1}')])

    # Changing view stack combo
    elif event == '__BUJ_Stack_Choice__' and window['__BUJ_View_Stack__'].metadata['State'] == 'Set':
        stack_choice = window['__BUJ_Stack_Choice__'].Get()
        if stack_choice == 'Unflip LS':
            stack_key = 'BUJ_unflip_stack'
        elif stack_choice == 'Flip LS':
            stack_key = 'BUJ_flip_stack'
        elif stack_choice == 'bUnwarpJ':
            stack_key = 'BUJ_stack'
        if stack_key in images:
            stack = images[stack_key]
            slider_val = 0
            slider_range = (0, stack.z_size - 1)

            # Update window
            metadata_change(window, [('__BUJ_Image1__', f'Image {slider_val + 1}')])
            display_img = stack.byte_data[slider_val]
            update_slider(window, [('__BUJ_Stack_Slider__', {"value": slider_val, "slider_range": slider_range})])
        else:
            stack_choice = winfo.buj_last_stack_choice
            update_values(window, [('__BUJ_Stack_Choice__', stack_choice)])
            print("Stack is not available to view. Must load or create alignment.")

    # Start making bunwarpJ masks
    elif event == '__BUJ_Make_Mask__':
        if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set':
            if make_mask_button.metadata['State'] == 'Def':
                mask_choice = window['__BUJ_Mask_View__'].Get()
                change_visibility(window, [('__BUJ_Reset_Mask__', True),
                                           ('__BUJ_Load_Mask_Col__', False)])
                if mask_choice == 'Unflip' or mask_choice == 'Flip':
                    mask_choice = mask_choice.lower()
                    display_img = images[mask_choice].byte_data
                    shortname = images[mask_choice].shortname
                elif mask_choice == 'Overlay':
                    if orientation == 'unflip':
                        img_1 = images['flip']
                        img_2 = images['unflip']
                    elif orientation == 'flip':
                        img_1 = images['unflip']
                        img_2 = images['flip']
                    display_img = g_help.overlay_images(img_1, img_2, transform)
                    shortname = 'overlay'
                toggle(window, ['__BUJ_Make_Mask__'])
                toggle(window, ['__BUJ_Image2__'], state='Def')
                metadata_change(window, [('__BUJ_Image1__', shortname)])
                disable_elements(window, ['__BUJ_transform_x__', '__BUJ_transform_y__',
                                          '__BUJ_transform_rot__', '__BUJ_horizontal_flip__'])
            # Quit mask making make_mask_button
            elif make_mask_button.metadata['State'] == 'Set':
                image = images[orientation]
                display_img = image.byte_data
                toggle(window, ['__BUJ_Make_Mask__'])
                metadata_change(window, [('__BUJ_Image1__', image.shortname)])
                enable_elements(window, ['__BUJ_transform_x__', '__BUJ_transform_y__',
                                         '__BUJ_transform_rot__', '__BUJ_horizontal_flip__'])
                change_visibility(window, [('__BUJ_Reset_Mask__', False),
                                           ('__BUJ_Load_Mask_Col__', True)])
                if winfo.buj_graph_double_click:
                    stack_choice = window['__BUJ_Mask_View__'].Get()
                    if stack_choice in ['Unflip', 'Flip']:
                        orientations = [stack_choice.lower()]
                        if stack_choice == 'Unflip':
                            flag = (True, False)
                        elif stack_choice == 'Flip':
                            flag = (False, True)
                    elif stack_choice == 'Overlay':
                        orientations = ['flip', 'unflip']
                        flag = (True, True)

                    filenames, overwrite_signs = run_save_window(winfo, event, image_dir, orientations)
                    if filenames == 'close':
                        return filenames
                    elif filenames:
                        g_help.create_mask(winfo, filenames, image)
                        if flag == (True, False):
                            image = Image(None, None, None, filenames[0])
                            images['BUJ_unflip_mask'] = image
                            metadata_change(window, [('__BUJ_Unflip_Mask_Inp__', image.shortname)])
                            toggle(window, ['__BUJ_Unflip_Mask_Inp__'], state='Set')
                        elif flag == (False, True):
                            image = Image(None, None, None, filenames[0])
                            images['BUJ_flip_mask'] = image
                            metadata_change(window, [('__BUJ_Flip_Mask_Inp__', image.shortname)])
                            toggle(window, ['__BUJ_Flip_Mask_Inp__'], state='Set')
                        elif flag == (True, True):
                            image1 = Image(None, None, None, filenames[0])
                            image2 = Image(None, None, None, filenames[1])
                            images['BUJ_flip_mask'] = image1
                            images['BUJ_unflip_mask'] = image2
                            metadata_change(window, [('__BUJ_Unflip_Mask_Inp__', image2.shortname)])
                            metadata_change(window, [('__BUJ_Flip_Mask_Inp__', image1.shortname)])
                            toggle(window, ['__BUJ_Flip_Mask_Inp__'], state='Set')
                            toggle(window, ['__BUJ_Unflip_Mask_Inp__'], state='Set')
                        if flag[0]: print(f'Successfully saved unflip mask!')
                        if flag[1]: print(f'Successfully saved flip mask!')
                        else: print(f'No masks were saved!')
                    else:
                        print(f'Exited without saving files!')
                else:
                    print('Mask was not finished, make sure to double-click and close mask.')
                g_help.erase_marks(winfo, graph, current_tab, full_erase=True)
        else:
            print('No flip data to adjust, make sure to set your working directory.')

    # Loading a mask
    elif event == '__BUJ_Mask_Stage_Load__':
        # Choose which masks should be loaded
        choice = window['__BUJ_Mask_View__'].Get()
        path = window['__BUJ_Mask_Stage_Load__'].Get()
        update_values(window, [('__BUJ_Mask_Stage_Load__', 'None')])
        image = Image(None, None, None, path)
        if choice == 'Unflip':
            images['BUJ_unflip_mask'] = image
            metadata_change(window, [('__BUJ_Unflip_Mask_Inp__', image.shortname)])
            toggle(window, ['__BUJ_Unflip_Mask_Inp__'], state="Set")
        elif choice == 'Flip':
            images['BUJ_flip_mask'] = image
            metadata_change(window, [('__BUJ_Flip_Mask_Inp__', image.shortname)])
            toggle(window, ['__BUJ_Flip_Mask_Inp__'], state="Set")
        else:
            images['BUJ_unflip_mask'] = image
            images['BUJ_flip_mask'] = image
            metadata_change(window, [('__BUJ_Unflip_Mask_Inp__', image.shortname),
                                     ('__BUJ_Flip_Mask_Inp__', image.shortname)])
            toggle(window, ['__BUJ_Unflip_Mask_Inp__', '__BUJ_Flip_Mask_Inp__'], state="Set")

    # Alternate between views for making mask points
    elif event == '__BUJ_Mask_View__' and make_mask_button.metadata['State'] == 'Set':
        mask_choice = window['__BUJ_Mask_View__'].Get()
        if mask_choice == 'Unflip' or mask_choice == 'Flip':
            mask_choice = mask_choice.lower()
            display_img = images[mask_choice].byte_data
            shortname = images[mask_choice].shortname
        elif mask_choice == 'Overlay':
            if orientation == 'unflip':
                img_1 = images['flip']
                img_2 = images['unflip']
            elif orientation == 'flip':
                img_1 = images['unflip']
                img_2 = images['flip']
            display_img = g_help.overlay_images(img_1, img_2, transform)
            shortname = 'overlay'
        toggle(window, ['__BUJ_Image2__'], state='Def')
        metadata_change(window, [('__BUJ_Image1__', shortname)])

        # Has double click been executed
        if winfo.buj_graph_double_click:
            draw_mask_polygon = True
        else:
            draw_mask_points = True

    # Clicking on graph and making markers for mask
    elif (event == '__BUJ_Graph__' and make_mask_button.metadata['State'] == 'Set' and
          not winfo.buj_graph_double_click):

        # Erase any previous marks
        g_help.erase_marks(winfo, graph, current_tab)

        # Draw new marks
        value = values['__BUJ_Graph__']
        winfo.buj_mask_coords.append([value[0], value[1]])
        draw_mask_points = True

    # Finishing markers for mask
    elif event == "__BUJ_Graph__Double Click" and make_mask_button.metadata['State'] == 'Set':
        g_help.erase_marks(winfo, graph, current_tab)
        if len(winfo.buj_mask_markers) >= 3:
            # Draw complete mask polygon
            winfo.buj_graph_double_click = True
            draw_mask_polygon = True
        else:
            print("Not enough vertices to close mask.")

    # Remove all mask coordinates from the graph and mask file
    elif event == '__BUJ_Reset_Mask__':
        # Erase any previous marks
        g_help.erase_marks(winfo, graph, current_tab, full_erase=True)

    # Generate the bUnwarpJ transformation file
    elif event == '__BUJ_Elastic_Align__':
        if (window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set'
                and images['BUJ_flip_stack'] and images['BUJ_unflip_stack']):
            # Get mask info
            mask_files = [None, None]
            if 'BUJ_unflip_mask' in images:
                mask_files[0] = images['BUJ_unflip_mask'].path
            elif 'BUJ_flip_mask' in images:
                mask_files[1] = images['BUJ_flip_mask'].path

            im_size = images['BUJ_flip_stack'].lat_dims
            sift_params, buj_params = load_buj_feat_ext_params(values), load_buj_params(values)

            # Decide whether file should be created
            filenames, overwrite_signals = run_save_window(winfo, event, image_dir)
            save = True
            save1, save2 = overwrite_signals[0], overwrite_signals[1]
            if filenames == 'close' or not filenames or not (save1 and save2):
                print('Exited save screen without saving image.')
                save = False
            if save:
                src1, src2 = filenames[0], filenames[1]
                for src in [src1, src2]:
                    if os.path.exists(src):
                        os.remove(src)
                stackpaths = images['BUJ_flip_stack'].path, images['BUJ_unflip_stack'].path
                macro = align.run_bUnwarp_align(image_dir, mask_files, orientation, transform, im_size,
                                                     stackpaths, sift_FE_params=sift_params,
                                                     buj_params=buj_params, savenames=(src1, src2))
                proc = g_help.run_macro(macro, image_dir, winfo.fiji_path)

                # Load the stack when ready
                target_key = '__BUJ_Stack__'
                image_key = 'BUJ_stack'
                align_keys = ['__BUJ_Unflip_Align__', '__BUJ_Load_Flip_Stack__', '__BUJ_Load_Unflip_Stack__',
                              '__BUJ_Elastic_Align__', '__BUJ_Load_Stack__']
                winfo.buj_file_queue[event] = (src2, image_key, target_key, current_tab, align_keys, proc)
            else:
                print(f'Exited without saving files!')
        else:
            print("Both unflip and flip stacks are not loaded")

    # If clear mask, remove from dictionaries
    elif event in ["__BUJ_Clear_Flip_Mask__", "__BUJ_Clear_Unflip_Mask__"]:
        if event == '__BUJ_Clear_Flip_Mask__' and 'BUJ_flip_mask' in images:
            del images['BUJ_flip_mask']
            metadata_change(window, ['__BUJ_Flip_Mask_Inp__'], reset=True)
            toggle(window, ['__BUJ_Flip_Mask_Inp__'], state='Def')
        elif event == '__BUJ_Clear_Unflip_Mask__' and 'BUJ_unflip_mask' in images:
            del images['BUJ_unflip_mask']
            metadata_change(window, ['__BUJ_Unflip_Mask_Inp__'], reset=True)
            toggle(window, ['__BUJ_Unflip_Mask_Inp__'], state='Def')

    # Update any image adjustments
    if overlay:
        if orientation == 'unflip':
            img_1 = images['flip']
            img_2 = images['unflip']
        elif orientation == 'flip':
            img_1 = images['unflip']
            img_2 = images['flip']
        display_img = g_help.overlay_images(img_1, img_2, transform)
    winfo.buj_past_transform = transform

    # Check to see if any files need loading
    if len(winfo.buj_file_queue) > 0:
        load_file_queue(winfo, window, current_tab)

    # Reset page
    if event == "__BUJ_Reset_Img_Dir__":
        reset(winfo, window, current_tab)

    # Enable any elements if need be
    special_enable_disable(window, adjust_button, view_stack_button, make_mask_button,
                           winfo.buj_images)

    # Redraw all
    if display_img:
        redraw_graph(graph, display_img)
    if draw_mask_polygon:
        g_help.draw_mask_points(winfo, graph, current_tab, double_click=True)
    elif draw_mask_points:
        g_help.draw_mask_points(winfo, graph, current_tab)


# -------------- Reconstruct Tab Event Handler -------------- #
def run_reconstruct_tab(winfo, window, current_tab, event, values):
    """Run events associated with the reconstruct tab.
    Parameters
    ----------
    winfo : Struct Class
        The data structure holding all information about
        windows and loaded images.
    window : PySimpleGUI Window Element
        The element representing the main GUI window.
    current_tab : str
        The key representing the current main tab of the
        window. Ex. '
    event : str
        The key for the values dictionary that represents
        an event in the window.
    values : dict
        A dictionary where every value is paired with
        a key represented by an event in the window.

    Returns
    -------
    None
    """
    # ------------- Visualizing Elements ------------- #
    def special_enable_disable(window):
        enable_list = []
        active_keys = ['__REC_Image_Dir_Path__', '__REC_Set_Img_Dir__', '__REC_Image_Dir_Browse__',
                       '__REC_FLS_Combo__', '__REC_Load_FLS1__', '__REC_Set_FLS__',
                       '__REC_Load_FLS2__', '__REC_Load_Stack__', '__REC_View__', '__REC_Image_List__',
                       '__REC_Def_Combo__', '__REC_QC_Input__',
                       "__REC_Reset_Stack__", "__REC_Reset_FLS__", "__REC_TFS_Combo__",
                       '__REC_Mask_Size__', '__REC_Mask__', "__REC_Erase_Mask__",
                       "__REC_transform_y__", "__REC_transform_x__",
                       "__REC_transform_rot__", '__REC_Run_TIE__', '__REC_Save_TIE__',
                       "__REC_Slider__", "__REC_Colorwheel__", "__REC_Derivative__"]

        if window['__REC_Set_Img_Dir__'].metadata['State'] == 'Set':
            if window['__REC_Set_FLS__'].metadata['State'] == 'Def':
                enable_list.extend(['__REC_FLS_Combo__', "__REC_TFS_Combo__"])
                if (window['__REC_FLS_Combo__'].Get() == 'Two' and
                        window['__REC_FLS2__'].metadata['State'] == 'Def'):
                    enable_list.extend(['__REC_Load_FLS2__'])
                if window['__REC_FLS1__'].metadata['State'] == 'Def':
                    enable_list.extend(['__REC_Load_FLS1__'])
                if window['__REC_Stack__'].metadata['State'] == 'Def':
                    enable_list.append('__REC_Load_Stack__')
                if (window['__REC_Stack__'].metadata['State'] == 'Set' and
                        window['__REC_FLS1__'].metadata['State'] == 'Set' and
                        window['__REC_FLS2__'].metadata['State'] == 'Set' and
                        len(winfo.rec_file_queue) == 0):
                    enable_list.extend(['__REC_Set_FLS__'])
            elif (window['__REC_Set_FLS__'].metadata['State'] == 'Set' and
                  window['__REC_View__'].metadata['State'] == 'Def'):
                enable_list.extend(['__REC_Mask__', '__REC_Mask_Size__',
                                    "__REC_transform_y__", "__REC_transform_x__",
                                    "__REC_transform_rot__"])
                if window['__REC_Mask__'].metadata['State'] == 'Def':
                    enable_list.extend(['__REC_Def_Combo__', '__REC_QC_Input__',
                                        '__REC_Run_TIE__', '__REC_Save_TIE__',
                                        "__REC_Colorwheel__", "__REC_Derivative__"])
                else:
                    enable_list.extend(['__REC_Erase_Mask__'])
            if (window['__REC_Stack__'].metadata['State'] == 'Set' and
                    window['__REC_Mask__'].metadata['State'] == 'Def'):
                enable_list.extend(['__REC_View__', "__REC_Image_List__"])

            if ((window['__REC_View__'].metadata['State'] == 'Set' and
                 window['__REC_Image_List__'].get()[0] == 'Stack') or
                 window['__REC_Mask__'].metadata['State'] == 'Set'):
                enable_list.extend(["__REC_Slider__"])
            elif (window['__REC_View__'].metadata['State'] == 'Def' and
                    window['__REC_Mask__'].metadata['State'] == 'Def'):
                enable_list.extend(["__REC_Reset_Stack__", "__REC_Reset_FLS__"])

        elif window['__REC_Set_Img_Dir__'].metadata['State'] == 'Def':
            enable_list.extend(['__REC_Image_Dir_Path__', '__REC_Set_Img_Dir__',
                                '__REC_Image_Dir_Browse__'])

        disable_list = np.setdiff1d(active_keys, enable_list)
        enable_elements(window, enable_list)
        disable_elements(window, disable_list)

    # Get rotations and shifts to apply to image (only positive rotations)
    transform = get_transformations(winfo, window, current_tab)
    mask_transform = get_mask_transform(winfo, window, current_tab)

    # Grab important elements
    graph = window['__REC_Graph__']
    colorwheel_graph =  window['__REC_Colorwheel_Graph__']
    view_button = window['__REC_View__']
    mask_button = window['__REC_Mask__']

    # Pull in image data from struct object
    image_dir = winfo.rec_image_dir
    images = winfo.rec_images
    # fls_files = winfo.rec_fls_files

    # if 'TIMEOUT' not in event and "HOVER" not in event:
    #     print(event)

    view_image_button = window['__REC_View__']
    display_img = None
    display_img2 = None

    # Import event handler names (overlaying, etc.)
    adjust = mask_button.metadata['State'] == 'Set' and (winfo.rec_past_transform != transform or
                                                         winfo.rec_past_mask != mask_transform)
    change_img = (winfo.rec_last_image_choice !=
                  window['__REC_Image_List__'].get()[0])
    scroll = (event in ['MouseWheel:Up', 'MouseWheel:Down']
              and (window['__REC_View__'].metadata['State'] == 'Set' or
                   window['__REC_Mask__'].metadata['State'] == 'Set')
              and winfo.true_element == "__REC_Graph__")
    draw_mask = mask_button.metadata['State'] == 'Set'

    # Set the working directory
    if event == '__REC_Set_Img_Dir__':
        image_dir = values['__REC_Image_Dir_Path__']
        if os.path.exists(image_dir):
            winfo.rec_image_dir = image_dir
            toggle(window, ['__REC_Set_Img_Dir__'], state='Set')
            print(f'The path is set: {image_dir}.')
        else:
            print('This pathname is incorrect.')

    # Load Stack
    elif event == '__REC_Stack_Staging__':
        stack_path = window['__REC_Stack_Staging__'].Get()
        update_values(window, [('__REC_Stack_Staging__', 'None')])
        if os.path.exists(stack_path):
            target_key = '__REC_Stack__'
            load_keys = ["__REC_Load_Stack__"]
            image_key = 'REC_Stack'
            winfo.rec_file_queue[event] = (stack_path, image_key, target_key,
                                           current_tab, load_keys, None)
        else:
            print('This pathname is incorrect.')

    # Reset Stack
    elif event == '__REC_Reset_Stack__':
        metadata_change(window, ['__REC_Stack__'], reset=True)
        toggle(window, ['__REC_Stack__'], state='Def')
        print('The stack was reset.')

    # Set number of FLS files to use
    elif event == '__REC_FLS_Combo__' or event == '__REC_TFS_Combo__':
        fls_value = window['__REC_FLS_Combo__'].Get()
        tfs_value = window['__REC_TFS_Combo__'].Get()
        metadata_change(window, ['__REC_FLS2__', '__REC_FLS1__'], reset=True)
        toggle(window, ['__REC_FLS2__', '__REC_FLS1__'], state='Def')
        # FLS Combo Chosen
        if event == '__REC_FLS_Combo__':
            # If one fls file is to be used
            metadata_change(window, [('__REC_FLS_Combo__', fls_value)])
            if fls_value == 'One':
                toggle(window, ['__REC_FLS_Combo__', '__REC_FLS2__'], state='Set')
                if tfs_value == 'Unflip/Flip':
                    val = 'Both'
                elif tfs_value == 'Single':
                    val = tfs_value
            # If two fls file is to be used
            elif fls_value == 'Two':
                val = fls_value
                metadata_change(window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__'], reset=True)
                toggle(window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__',
                                '__REC_FLS2__'], state='Def')
        # TFS Combo Chosen
        elif event == '__REC_TFS_Combo__':
            metadata_change(window, [('__REC_TFS_Combo__', tfs_value)])
            if tfs_value == 'Unflip/Flip':
                val = 'Two'
                metadata_change(window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__'], reset=True)
                toggle(window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__',
                                '__REC_FLS2__'], state='Def')
            elif tfs_value == 'Single':
                val = tfs_value
                metadata_change(window, [('__REC_FLS_Combo__', 'One')])
                toggle(window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__',
                                '__REC_FLS2__'], state='Set')
        window['__REC_FLS1_Text__'].update(value=window['__REC_FLS1_Text__'].metadata[val])
        window['__REC_FLS2_Text__'].update(value=window['__REC_FLS2_Text__'].metadata[val])

    # Load FLS files
    elif event == '__REC_FLS1_Staging__' or event == '__REC_FLS2_Staging__':
        if 'FLS1' in event:
            fls_path = window['__REC_FLS1_Staging__'].Get()
            update_values(window, [('__REC_FLS1_Staging__', 'None')])
            target_key = '__REC_FLS1__'
        elif 'FLS2' in event:
            fls_path = window['__REC_FLS2_Staging__'].Get()
            update_values(window, [('__REC_FLS2_Staging__', 'None')])
            target_key = '__REC_FLS2__'
        if os.path.exists(fls_path) and fls_path.endswith('.fls'):
            fls = File_Object(fls_path)
            if 'FLS1' in event:
                winfo.rec_fls_files[0] = fls
            elif 'FLS2' in event:
                winfo.rec_fls_files[1] = fls
            metadata_change(window, [(target_key, fls.shortname)])
            toggle(window, [target_key], state='Set')
        else:
            print('Pathname is incorrect or none selected.')

    # Set number of FLS files to use
    elif event == '__REC_Reset_FLS__':
        winfo.rec_fls_files = {}
        metadata_change(window, ['__REC_FLS_Combo__', '__REC_FLS2__', '__REC_FLS1__'], reset=True)
        toggle(window, ['__REC_FLS2__', '__REC_FLS1__'], state='Def')
        toggle(window, ['__REC_FLS_Combo__'], state='Def')
        window['__REC_FLS1_Text__'].update(value=window['__REC_FLS1_Text__'].metadata['Two'])
        window['__REC_FLS2_Text__'].update(value=window['__REC_FLS2_Text__'].metadata['Two'])

    # Set which image you will be working with FLS files
    elif event == '__REC_Set_FLS__':
        # Get PYTIE loading params
        path = image_dir + '/'
        stack_name = images['REC_Stack'].shortname

        # Get FLS value information
        fls_value = window['__REC_FLS_Combo__'].Get()
        if fls_value == 'One':
            fls_1 = winfo.rec_fls_files[0]
            fls_2 = winfo.rec_fls_files[1]
        elif fls_value == 'Two':
            fls_1 = winfo.rec_fls_files[0]
            fls_2 = winfo.rec_fls_files[1]
        fls1_path = fls_1.shortname
        print(fls1_path)
        if fls_2:
            fls2_path = fls_2.shortname
        else:
            fls2_path = None
        print(fls1_path)
        print(fls2_path)



        # Is this single series or flipped/unflipped series
        if window['__REC_TFS_Combo__'].metadata['State'] == 'Def':
            flip = True
        else:
            flip = False

        # Load ptie params
        try:
            dm3stack1, dm3stack2, ptie = load_data(path, fls1_path, stack_name, flip, fls2_path)
            string_vals = []
            window['__REC_Def_Multi__'].update(value="")
            for def_val in ptie.defvals:
                val = str(def_val)
                window['__REC_Def_Multi__'].update(value=val+'\n', append=True)
                string_vals.append(val)
            window['__REC_Def_Combo__'].update(values=string_vals)

            accel_volt = float(window['__REC_M_Volt__'].get()) * 1e3
            winfo.rec_ptie = ptie
            winfo.rec_microscope = Microscope(E=accel_volt, Cs=200.0e3, theta_c=0.01e-3, def_spr=80.0)
            toggle(window, elem_list=['__REC_Set_FLS__'])

        except:
            print('Something went wrong loading in image data.')
            raise

    # View the image stack created from alignment
    elif event == '__REC_View__':
        # Look at which image to view
        image_choice = window['__REC_Image_List__'].get()[0]
        disabled = False
        if image_choice == 'Stack':
            image_key = 'REC_Stack'
            disabled = '__REC_Load_Stack__' in winfo.rec_file_queue
        elif image_choice == 'MagX':
            image_key = 'bxt'
        elif image_choice == 'MagY':
            image_key = 'byt'
        elif image_choice == 'Mag':
             image_key = 'bbt'
        elif image_choice == 'Mag. Phase':
             image_key = 'phase_m'
        elif image_choice == 'Electr. Phase':
             image_key = 'phase_e'
        elif image_choice == 'Mag. Deriv.':
             image_key = 'dIdZ_m'
        elif image_choice == 'Electr. Deriv.':
             image_key = 'dIdZ_e'
        elif image_choice == 'Color':
             image_key = 'color_b'
        elif image_choice == 'In Focus':
             image_key = 'inf_im'
        if view_button.metadata['State'] == 'Def':
            slider_val = int(values["__REC_Slider__"])
            if image_key in images and not disabled and image_key == 'REC_Stack':
                stack = images[image_key]
                slider_range = (0, stack.z_size - 1)
                display_img = stack.byte_data[slider_val]
                colorwheel_graph.Erase()

                # Update window
                metadata_change(window, [('__REC_Image__', f'Image {slider_val + 1}')])
                toggle(window, ['__REC_Image__', '__REC_View__'], state='Set')
                update_slider(window, [('__REC_Slider__', {"value": slider_val, "slider_range": slider_range})])
                winfo.rec_last_image_choice = image_choice

            elif image_key in images and not disabled:
                toggle(window, ['__REC_Image__', '__REC_View__'], state='Set')
                image = images[image_key]
                display_img = image.byte_data
                if image_key == 'color_b':
                    display_img2 = winfo.rec_colorwheel
                else:
                    colorwheel_graph.Erase()
                update_slider(window, [('__REC_Slider__', {"value": slider_val})])

                # Update window
                metadata_change(window, [('__REC_Image__', image_choice)])
                winfo.rec_last_image_choice = image_choice
            else:
                print("Tried loading unavailable image.")
        elif view_button.metadata['State'] == 'Set':
            graph.Erase()
            colorwheel_graph.Erase()
            metadata_change(window, ['__REC_Image__'], reset=True)
            toggle(window, ['__REC_Image__'])
            toggle(window, ['__REC_View__'])
            # update_slider(window, [('__REC_Slider__', {"value": 0})])

        winfo.rec_last_image_choice = image_choice

    # Change the slider
    elif event == '__REC_Slider__':
        stack_choice = window['__REC_Image_List__'].get()[0]
        if stack_choice == 'Stack':
            stack_key = 'REC_Stack'
        stack = images[stack_key]
        slider_val = int(values["__REC_Slider__"])

        # Update window
        if window['__REC_View__'].metadata['State'] == 'Set':
            display_img = stack.byte_data[slider_val]
        elif window['__REC_Mask__'].metadata['State'] == 'Set':
            display_img = g_help.adjust_image(stack.flt_data[slider_val], transform)
        metadata_change(window, [('__REC_Image__', f'Image {slider_val + 1}')])

    # Scroll through stacks in the graph area
    elif scroll:
        stack_choice = window['__REC_Image_List__'].get()[0]
        if stack_choice == 'Stack':
            stack_key = 'REC_Stack'
            stack = images[stack_key]
            slider_val = int(values["__REC_Slider__"])
            max_slider_val = stack.z_size - 1
            # Scroll up or down
            if event == 'MouseWheel:Down':
                slider_val = min(max_slider_val, slider_val+1)
            elif event == 'MouseWheel:Up':
                slider_val = max(0, slider_val-1)

            # Update the window
            if window['__REC_View__'].metadata['State'] == 'Set':
                display_img = stack.byte_data[slider_val]
            elif window['__REC_Mask__'].metadata['State'] == 'Set':
                display_img = g_help.adjust_image(stack.flt_data[slider_val], transform)

            update_slider(window, [('__REC_Slider__', {"value": slider_val})])
            metadata_change(window, [('__REC_Image__', f'Image {slider_val+1}')])

    # Changing view stack combo
    elif change_img and window['__REC_View__'].metadata['State'] == 'Set':
        image_choice = window['__REC_Image_List__'].get()[0]
        if image_choice == 'Stack':
            image_key = 'REC_Stack'
        elif image_choice == 'Color':
            image_key = 'color_b'
        elif image_choice == 'MagX':
            image_key = 'bxt'
        elif image_choice == 'MagY':
            image_key = 'byt'
        elif image_choice == 'Mag':
             image_key = 'bbt'
        elif image_choice == 'Mag. Phase':
             image_key = 'phase_m'
        elif image_choice == 'Electr. Phase':
             image_key = 'phase_e'
        elif image_choice == 'Mag. Deriv.':
             image_key = 'dIdZ_m'
        elif image_choice == 'Electr. Deriv.':
             image_key = 'dIdZ_e'
        elif image_choice == 'In Focus':
             image_key = 'inf_im'
        # Stack set
        if image_key in images and image_choice == 'Stack':
            stack = images[image_key]
            slider_val = 0
            slider_range = (0, stack.z_size - 1)

            # Update window
            metadata_change(window, [('__REC_Image__', f'Image {slider_val + 1}')])
            display_img = stack.byte_data[slider_val]
            colorwheel_graph.Erase()
            update_slider(window, [('__REC_Slider__', {"value": slider_val, "slider_range": slider_range})])
        # Other image set
        elif image_key in images:
            image = images[image_key]
            display_img = image.byte_data
            if image_key == 'color_b':
                display_img2 = winfo.rec_colorwheel
            else:
                colorwheel_graph.Erase()
            metadata_change(window, [('__REC_Image__', f'{image_choice}')])
        else:
            list_values = window['__REC_Image_List__'].GetListValues()
            image_choice = winfo.rec_last_image_choice
            index = list_values.index(image_choice)
            window['__REC_Image_List__'].update(set_to_index=index, scroll_to_index=index)
            print("Image is not available to view. Must run PYTIE.")
        winfo.rec_last_image_choice = image_choice

    # Run PyTIE
    elif event == '__REC_Run_TIE__':
        ptie = winfo.rec_ptie
        microscope = winfo.rec_microscope
        def_val = float(window['__REC_Def_Combo__'].Get())
        def_ind = ptie.defvals.index(def_val)
        dataname = 'example'
        save = False

        sym = window['__REC_Symmetrize__'].Get()
        qc = window['__REC_QC_Input__'].Get()
        qc_passed = True
        if g_help.represents_float(qc):
            qc = float(qc)
            if qc < 0:
                qc_passed = False
            elif qc == 0:
                qc = None
        else:
            qc_passed = False

        # Longitudinal deriv
        deriv_val = window['__REC_Derivative__'].get()
        if deriv_val == 'Longitudinal Deriv.':
            longitudinal_deriv = True
        elif deriv_val == 'Central Diff.':
            longitudinal_deriv = False

        # Set crop data
        # bottom, top, left, right = None, None, None, None
        # for i in range(len(winfo.rec_mask_coords)):
        #     x, y = winfo.rec_mask_coords[i]
        #     x, y = round(x), round(y)
        #     if right is None or x > right:
        #         right = x
        #     if left is None or x < left:
        #         left = x
        #     if bottom is None or y > bottom:
        #         bottom = y
        #     if top is None or y < top:
        #         top = y
        #
        # reg_width, reg_height = images['REC_Stack'].lat_dims
        # crop_width, crop_height = (right - left), (bottom - top)
        # scale_x, scale_y = reg_width/crop_width, reg_height/crop_height
        # print('crops: ', left, right, top, bottom)
        # print('scaled_crop: ', left*scale_x, right*scale_x, top*scale_y, bottom*scale_y)
        # ptie.crop['right'], ptie.crop['left'] = left*scale_x, right*scale_x
        # ptie.crop['bottom'], ptie.crop['top'] = bottom*scale_y, top*scale_y,

        if not qc_passed:
            print('QC value should be an integer or float and not negative. Change value.')
            update_values(window, [('__REC_QC_Input__', '0.00')])
        else:
            try:
                print(f'Reconstructing for defocus value: {ptie.defvals[def_ind]} nm ')
                results = TIE(def_ind, ptie, microscope,
                              dataname, sym, qc, save,
                              longitudinal_deriv, v=False)

                # This will need to consider like the cropping region
                x_size, y_size = images['REC_Stack'].lat_dims
                for key in results:
                    # print(key,  results[key].shape, results[key].dtype, results[key])
                    float_array = results[key]
                    if key == 'color_b':
                        float_array = g_help.slice(float_array, (x_size, y_size))
                        colorwheel_type = window['__REC_Colorwheel__'].get()
                        rad1, rad2 = colorwheel_graph.get_size()
                        if colorwheel_type == 'HSV':
                            cwheel_hsv = colorwheel_HSV(rad1, background='black')
                            cwheel = colors.hsv_to_rgb(cwheel_hsv)
                        elif colorwheel_type == '4-Fold':
                            cwheel = colorwheel_RGB(rad1)
                        uint8_colorwheel, float_colorwheel = g_help.convert_float_unint8(cwheel, (rad1, rad2))
                        rgba_colorwheel = g_help.make_rgba(uint8_colorwheel[0])
                        winfo.rec_colorwheel = g_help.convert_to_bytes(rgba_colorwheel)
                    uint8_data, float_data = {}, {}
                    uint8_data, float_data = g_help.convert_float_unint8(float_array, graph.get_size(),
                                                                         uint8_data, float_data)
                    image = Image(uint8_data, float_data, (x_size, y_size, 1), f'/{key}')

                    image.byte_data = g_help.vis_1_im(image)
                    winfo.rec_images[key] = image
            except:
                print('There was an error when running TIE.')
                raise

    # Start making reconstruct subregion
    elif event == '__REC_Mask__':
        if mask_button.metadata['State'] == 'Def':
            # Get the stack to view
            stack = images['REC_Stack']
            slider_range = (0, stack.z_size - 1)
            slider_val = int(values["__REC_Slider__"])
            display_img = g_help.adjust_image(stack.flt_data[slider_val], transform)

            # Update window
            metadata_change(window, [('__REC_Image__', f'Image {slider_val + 1}')])
            toggle(window, ['__REC_Mask__', '__REC_Image__'], state='Set')
            update_slider(window, [('__REC_Slider__', {"value": slider_val, "slider_range": slider_range})])

            # winfo.rec_mask = float(window['__REC_Mask_Size__'].Get())
            draw_mask = True
            g_help.draw_square_mask(winfo, graph)

        # Quit mask making make_mask_button
        elif mask_button.metadata['State'] == 'Set':
            graph.Erase()
            toggle(window, ['__REC_Mask__', '__REC_Image__'], state='Def')
            metadata_change(window, ['__REC_Image__'], reset=True)
            draw_mask = False

    # Clicking on graph and making markers for mask
    elif event in ['__REC_Graph__', '__REC_Graph__+UP'] and mask_button.metadata['State'] == 'Set':

        # Erase any previous marks
        g_help.erase_marks(winfo, graph, current_tab)

        # # Draw new marks
        value = values['__REC_Graph__']
        winfo.rec_mask_center = value
        g_help.draw_square_mask(winfo, graph)
        draw_mask = True

    # Remove all mask coordinates from the graph and mask file
    elif event == '__REC_Erase_Mask__':
        # Erase any previous marks
        g_help.erase_marks(winfo, graph, current_tab, full_erase=True)
        graph_size = graph.get_size()
        winfo.rec_mask_center = ((graph_size[0] - 2) / 2, (graph_size[1]) / 2)
        winfo.rec_mask = (50,)
        mask_transform = (50,)
        transform = (0, 0, 0, None)
        adjust = True
        draw_mask = True
        update_values(window, [('__REC_transform_x__', '0'), ('__REC_transform_y__', '0'),
                               ('__REC_transform_rot__', "0"), ('__REC_Mask_Size__', '50')])

    # Adjust stack and related variables
    if adjust:
        if winfo.rec_past_transform != transform:
            stack = images['REC_Stack']
            slider_val = int(values["__REC_Slider__"])
            display_img = g_help.adjust_image(stack.flt_data[slider_val], transform)
        if winfo.rec_past_mask != mask_transform:
            g_help.erase_marks(winfo, graph, current_tab, full_erase=True)
            g_help.draw_square_mask(winfo, graph)
    winfo.rec_past_transform = transform
    winfo.rec_past_mask = mask_transform

    # Check to see if any files need loading
    if len(winfo.rec_file_queue) > 0:
        load_file_queue(winfo, window, current_tab)

    # Reset page
    if event == "__REC_Reset_Img_Dir__":
        reset(winfo, window, current_tab)

    # Enable any elements if need be
    special_enable_disable(window)

    # Redraw all
    if display_img:
        redraw_graph(graph, display_img)
    if display_img2:
        redraw_graph(colorwheel_graph, display_img2)
    if draw_mask:
        g_help.draw_mask_points(winfo, graph, current_tab)


# -------------- Save Window --------------#
def check_overwrite(save_win, true_paths, orientations, im_type):
    """Check whether the paths listed in the log box for
    each image will be overwritten.

    Parameters
    ----------
    save_win : PySimpleGUI window element
        The save window element
    true_paths : list
        A list of path names that will be
        checked if they exist.
    orientations :  list of str
        A list of strings that represent
        the orientations of the image ('flip',
        'unflip', 'stack', etc.)
    im_type : str
        Image type (.bmp, .tiff, etc.)
    """
    # If file exists notify user and give option to change name
    update_values(save_win, [('__save_win_log__', '')])
    overwrite_signals = [None]*len(true_paths)
    save_enable = True
    for i in range(len(true_paths)):
        text = ''
        overwrite_box = save_win[f'__save_win_overwrite{i+1}__'].Get()
        exists = os.path.exists(true_paths[i])
        # If no orientation, this removes extra space in insertion for log
        if orientations[i]:
            insertion = f'{orientations[i]}'
        else:
            insertion = im_type
        # Exists but not overwrite
        if exists and not overwrite_box:
            text = f'''The {insertion} file already exists. Check overwrite checkbox if you want to save anyway.'''
            overwrite_signals[i] = False
            save_enable = False
        # File already exists but will be overwritten
        elif exists and overwrite_box:
            text = f'The {insertion} file will be overwritten.'
            overwrite_signals[i] = True
        # Doesn't exist, don't overwrite
        elif not exists:
            text = f'The {insertion} file will be saved.'
            overwrite_signals[i] = True

        # Update save window
        current_log_text = save_win['__save_win_log__'].Get()
        new_log_text = current_log_text + text
        update_values(save_win, [('__save_win_log__', new_log_text.strip())])
    if save_enable:
        enable_elements(save_win, ['__save_win_save__'])
    else:
        disable_elements(save_win, ['__save_win_save__'])
    return overwrite_signals


def save_window_values(save_win, num_paths):
    """Sets ups the save window layout.

    Parameters
    ----------
    save_win : PySimpleGUI Window Element
        The representation of the save window.
    num_paths : int
        The number of paths, to create the number
        of overwrite checkboxes and true_path
        input elements.


    Returns
    -------
    true_paths : list of str
        The list containing the full path names.
    """
    # Comb through all input fields and pull current path name
    # overwrite = True
    true_paths, overwrite_boxes = [], []
    for i in range(1, num_paths + 1):
        true_paths.append(save_win[f'__save_win_filename{i}__'].Get())
    return true_paths


def run_save_window(winfo, event, image_dir, orientations=None):
    """Executes the save window.

    Parameters
    __________
    winfo : Struct Class
        The data structure holding all information about
        windows and loaded images.
    event : str
        The key for the values dictionary that represents
        an event in the window.
    image_dir : str
        The working directory where image will be saved
    orientations : None or list of str
        List of the orientations to categorize the saved
        file ('flip', 'unflip', 'stack', '').

    Returns
    -------
    filenames : list of str
        The list of filenames to give the saved images.
    """
    # Create layout of save window
    window_layout, im_type, file_paths, orientations = save_window_ly(event, image_dir, orientations)
    save_win = sg.Window('Save Window', window_layout,
                         grab_anywhere=True).Finalize()
    winfo.save_win = save_win
    winfo.window.Hide()
    true_paths = save_window_values(save_win, len(file_paths))
    # Run event handler
    overwrite_signals = []
    while True:
        ev2, vals2 = save_win.Read(timeout=400)
        filenames = []
        if ev2 and ev2 != 'Exit':
            true_paths = save_window_values(save_win, len(file_paths))

        # Exit or save pressed
        if not ev2 or ev2 in ['Exit', '__save_win_save__']:
            winfo.window.UnHide()
            save_win.Close()
            if ev2 == '__save_win_save__':
                for path in true_paths:
                    filenames.append(path)
            break
        overwrite_signals = check_overwrite(save_win, true_paths, orientations, im_type)
    return filenames, overwrite_signals


# -------------- Main Event Handler and run GUI --------------#
def event_handler(winfo, window):
    """ The event handler handles all button presses, mouse clicks, etc.
    that can take place in the app. It takes the SG window and the struct
    containing all window data as parameters.

    Parameters
    ----------
    winfo : Struct Class
        The data structure holding all information about
        windows and loaded images.
    window : PySimpleGUI Window Element
        The element representing the main GUI window.

    Returns
    -------
    None
    """
    # Finalize window
    window.finalize()

    # Prepare file save window
    winfo.window_active = True

    # Initialize window, bindings, and event variables
    init(winfo, window)

    # Run event loop
    bound_click = True
    close = None
    while True:
        # Capture events
        event, values = window.Read(timeout=200)

        # Break out of event loop
        if event is None or close == 'close':  # always,  always give a way out!
            window.close()
            break

        # (turn this into it's own function in BUJ')
        # Disable window clicks if creating mask or setting subregion
        if ((winfo.true_element == '__BUJ_Graph__' and bound_click and
             window['__BUJ_Make_Mask__'].metadata['State'] == 'Set') or
            (winfo.true_element == '__REC_Graph__' and bound_click and
             window['__REC_Mask__'].metadata['State'] == 'Set')):
            window.TKroot.unbind("<Button-1>")
            bound_click = False
        elif ((not bound_click and winfo.true_element != '__BUJ_Graph__') and
              (not bound_click and winfo.true_element != '__REC_Graph__')):
            winfo.window.bind("<Button-1>", 'Window Click')
            bound_click = True

        # Check which tab is open and execute events regarding that tab
        current_tab = winfo.current_tab = get_open_tab(winfo, winfo.pages)
        if current_tab == "home_tab":
            run_home_tab(winfo, window, event, values)
        elif current_tab == "ls_tab":
            run_ls_tab(winfo, window, current_tab, event, values)
        elif current_tab == "bunwarpj_tab":
            run_bunwarpj_tab(winfo, window, current_tab, event, values)
        elif current_tab == "reconstruct_tab":
            run_reconstruct_tab(winfo, window, current_tab, event, values)

        # Set the focus of the GUI to reduce interferences
        set_pretty_focus(winfo, window, event)


def run_GUI(style, DEFAULTS):
    """Main run function. Takes in the style and defaults for GUI.

    Parameters
    ----------
    style : WindowStyle Class
        Class that controls the styling of certain
        elements within the GUI.
    DEFAULTS : dict
        Dictionary of the default values for certain
        values of the window.

    Returns
    -------
    None
    """

    # Create the layout
    window = window_ly(style, DEFAULTS)

    # Create data structure to hold variables about GUI, alignment and reconstruction.
    winfo = Struct()

    # Event handling
    event_handler(winfo, window)


run_GUI(style, DEFAULTS)
