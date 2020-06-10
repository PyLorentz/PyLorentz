import warnings
import PySimpleGUI as sg
from os import path as os_path, remove as os_remove, mknod as os_mknod
import subprocess
import multiprocessing
from platform import system as platform
from io import StringIO
from align import check_setup, join as al_join, run_bUnwarp_align, run_ls_align, run_single_ls_align
import gui_help as g_help
from gui_styling import WindowStyle, window_scaling #get_icon,
from gui_layout import window_ly, keys, save_window_ly, output_ly
from numpy import setdiff1d
from matplotlib import colors
# import itertools
from sys import path as sys_path
from contextlib import redirect_stdout
import sys
import shlex
from threading import Thread
from queue import Queue, Empty
sys_path.append("../PyTIE/")
from microscopes import Microscope
from TIE_helper import *
from TIE_reconstruct import TIE, SITIE, save_results
from colorwheel import colorwheel_HSV, colorwheel_RGB, color_im

# print = sg.Print
# sg.main()

# !!!!!!!!!!!!!!!!!! CHANGE DEFAULTS BELOW !!!!!!!!!!!!!!!!!! #
DEFAULTS = {'fiji_dir': '/Applications/Fiji.app',
            'browser_dir': '/Users/timothycote/Box/dataset1_tim'}
# !!!!!!!!!!!!!!!!!! CHANGE DEFAULTS ABOVE !!!!!!!!!!!!!!!!!! #


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
    winfo.ls_fls_files = [None, None]
    winfo.ls_files1 = None
    winfo.ls_files2 = None

    # Declare transformation variables
    winfo.ls_transform = (0, 0, 0, 1)
    winfo.ls_past_transform = (0, 0, 0, 1)


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
    winfo.buj_fls_files = [None, None]
    winfo.buj_files1 = None
    winfo.buj_files2 = None


    # --- Set up loading files --- #
    winfo.buj_file_queue = []

    # Stack selection
    winfo.buj_last_image_choice = None

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
    winfo.rec_files1 = None
    winfo.rec_files2 = None
    winfo.rec_ptie = None
    winfo.rec_sym = None
    winfo.rec_qc = None
    winfo.rec_microscope = None
    winfo.rec_colorwheel = None

    # --- Set up loading files --- #
    winfo.rec_defocus_slider_set = 0
    winfo.rec_image_slider_set = 6
    winfo.rec_image_slider_dict = {'Stack': 0, 'Color': 1,
                                   'MagX': 2, 'MagY': 3, 'Mag': 4,
                                   'Electr. Phase': 5, 'Mag. Phase': 6,
                                   'Electr. Deriv.': 7, 'Mag. Deriv.': 8,
                                   'In Focus': 9, 'Default Stack': 10}

    # Declare transformation timers and related variables
    winfo.rec_transform = (0, 0, 0, None)
    winfo.rec_past_transform = (0, 0, 0, None)
    winfo.rec_mask_timer = (0,)
    winfo.rec_mask = (50,)
    winfo.rec_past_mask = (50,)
    winfo.graph_slice = (None, None)

    graph_size = window['__REC_Graph__'].metadata['size']
    winfo.rec_mask_center = ((graph_size[0])/2, (graph_size[1])/2)

    # Image selection
    winfo.rec_last_image_choice = None
    winfo.rec_last_colorwheel_choice = None
    winfo.rec_tie_results = None
    winfo.rec_def_val = None

    # Graph and mask making
    winfo.rec_graph_double_click = False
    winfo.rec_mask_coords = []
    winfo.rec_mask_markers = []


def init(winfo, window, output_window):
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
    winfo.output_window = output_window
    winfo.window_active = True
    winfo.output_window_active = False
    winfo.invis_graph = window.FindElement("__invisible_graph__")
    winfo.tabnames = ["Home", "Registration", "Linear Stack Alignment with SIFT", "bUnwarpJ", "Phase Reconstruction"]
    winfo.pages = "pages_tabgroup"
    winfo.current_tab = "home_tab"
    winfo.buf = None
    winfo.ptie_init_thread = None
    winfo.ptie_recon_thread = None

    # --- Set up FIJI/ImageJ --- #
    winfo.fiji_path = ""
    winfo.fiji_queue = []
    winfo.fiji_log_line = -1
    winfo.fiji_thread_queue = None
    winfo.fiji_thread = None
    winfo.proc = None
    winfo.kill_proc = []


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
    big_list = keys['input'] + keys['radio'] + keys['graph'] + keys['combo'] + keys['checkbox'] + keys['slider'] + keys['button'] + keys['listbox'] #keys['multiline'],
    for key in big_list:
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
                        '__LS_Adjust__', '__LS_View_Stack__', '__LS_Set_Img_Dir__',
                        '__LS_FLS1__', '__LS_FLS2__', '__LS_Set_FLS__', '__LS_FLS_Combo__',
                        '__LS_TFS_Combo__',], state='Def')
        update_values(window, [('__LS_Image_Dir_Path__', ""),('__LS_FLS1_Staging__', ''), ('__LS_FLS2_Staging__', ''),
                               ('__LS_transform_x__', '0'), ('__LS_transform_y__', '0'), ('__LS_transform_rot__', "0"),
                               ('__LS_horizontal_flip__', True)])
        update_slider(window, [('__LS_Stack_Slider__', {"value": 0, "slider_range": (0, 0)})])
        window['__LS_unflip_reference__'].ResetGroup()
        # Declare image path and related variables
        init_ls(winfo)

    elif current_tab == 'bunwarpj_tab':
        graph = window['__BUJ_Graph__']
        graph.Erase()
        metadata_change(window, ['__BUJ_Image1__', '__BUJ_Image2__', '__BUJ_Stack__',
                                 '__BUJ_Flip_Stack_Inp__', '__BUJ_Unflip_Stack_Inp__',
                                 '__BUJ_Unflip_Mask_Inp__', '__BUJ_Flip_Mask_Inp__'], reset=True)
        toggle(window, ['__BUJ_Image1__', '__BUJ_Image2__', '__BUJ_Stack__',
                        '__BUJ_FLS1__', '__BUJ_FLS2__', '__BUJ_Set_FLS__', '__BUJ_FLS_Combo__',
                        '__BUJ_Adjust__', '__BUJ_View__', '__BUJ_Make_Mask__',
                        '__BUJ_Flip_Stack_Inp__', '__BUJ_Unflip_Stack_Inp__',
                        '__BUJ_Unflip_Mask_Inp__', '__BUJ_Flip_Mask_Inp__',
                        '__BUJ_Set_Img_Dir__'], state='Def')
        update_values(window, [('__BUJ_Image_Dir_Path__', ""),
                               ('__BUJ_transform_x__', '0'), ('__BUJ_transform_y__', '0'),
                               ('__BUJ_transform_rot__', "0"), ('__BUJ_horizontal_flip__', True),
                               ('__BUJ_unflip_reference__', True)])
        update_slider(window, [('__BUJ_Stack_Slider__', {"value": 0, "slider_range": (0, 0)})])

        # Re-init bUnwarpJ
        init_buj(winfo)

    elif current_tab == 'reconstruct_tab':
        graph = window['__REC_Graph__']
        colorwheel_graph = window['__REC_Colorwheel_Graph__']
        graph.Erase()
        colorwheel_graph.Erase()
        metadata_change(window, ['__REC_Stack__', '__REC_Image__', '__REC_FLS1__', '__REC_FLS2__'], reset=True)
        toggle(window, ['__REC_Set_Img_Dir__', '__REC_FLS_Combo__', '__REC_TFS_Combo__',
                        '__REC_Stack__', '__REC_FLS1__', '__REC_FLS2__', '__REC_Set_FLS__',
                        '__REC_Mask__', '__REC_Image__'], state='Def')
        window['__REC_Def_Combo__'].update(values=['None'])
        window['__REC_Def_List__'].update(values=['None'])
        window['__REC_Symmetrize__'].update(value=False)
        window['__REC_FLS1_Text__'].update(value=window['__REC_FLS1_Text__'].metadata['Two'])
        window['__REC_FLS2_Text__'].update(value=window['__REC_FLS2_Text__'].metadata['Two'])
        window['__REC_FLS1_Text__'].metadata['State'] = 'Two'
        window['__REC_FLS2_Text__'].metadata['State'] = 'Two'
        change_list_ind_color(window, current_tab, [('__REC_Image_List__', [])])
        update_values(window, [('__REC_Image_Dir_Path__', ""), ('__REC_Image__', 'None'),
                               ('__REC_transform_x__', '0'), ('__REC_transform_y__', '0'),
                               ('__REC_transform_rot__', "0"), ('__REC_Mask_Size__', '50'),
                               ('__REC_Stack_Staging__', ''), ('__REC_FLS1_Staging__', ''),
                               ('__REC_FLS2_Staging__', ''), ('__REC_Colorwheel__', 'HSV'),
                               ('__REC_Def_Combo__', 'None'), ('__REC_QC_Input__', '0.00'),
                               ('__REC_Data_Prefix__', 'Example'), ('__REC_M_Volt__', '200')])

        # Re-init reconstruct
        init_rec(winfo, window)
        update_slider(window, [('__REC_Defocus_Slider__', {'value':winfo.rec_defocus_slider_set,
                                                           'slider_range':(0, 0)}),
                               ('__REC_Slider__', {'value': 0,
                                                   'slider_range': (0, 0)}),
                               ('__REC_Image_Slider__', {'value': winfo.rec_image_slider_set})])
        window['__REC_Image_List__'].update(set_to_index=0, scroll_to_index=0)
        window['__REC_Def_List__'].update(set_to_index=0, scroll_to_index=0)


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
def shorten_name(path, ind=1):
    """Creates a string of the path name with only the direct parent
    "image_dir" and the child of "image_dir".

    """
    check_string = path
    for i in range(ind):
        index = check_string.rfind('/') - 1
        check_string = check_string[:index]
    shortname = path[index+2:]
    return shortname


def get_open_tab(winfo, tabgroup, event):
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
        tab_key = get_open_tab(winfo, child_tabgroup, event)
        return tab_key
    if 'tab' in event:
        if tab_key == "home_tab":
            tab = 'Home'
        elif tab_key == "ls_tab":
            tab = 'Lin. SIFT'
        elif tab_key == "bunwarpj_tab":
            tab = 'bUnwarpJ'
        elif tab_key == "reconstruct_tab":
            tab = 'Reconstruct'
        print(f'*** Current Tab: {tab} ***')
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
    else:
        orientation = 'tfs'
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


def file_loading(winfo, window, filename, active_key, image_key, target_key,
                 conflict_keys, num_files, disable_elem_list):

    remove = False
    # Path exists
    if os_path.exists(filename):
        with warnings.catch_warnings():
            try:
                # Is file loading correctly?
                warnings.filterwarnings('error')
                # Load images and convert to uint8 using numpy and hyperspy,
                if active_key.startswith('__REC'):
                    graph_size_key, reset_key, prefix = '__REC_Graph__', '__REC_Set_Img_Dir__', 'LS: '
                elif active_key.startswith('__BUJ'):
                    graph_size_key, reset_key, prefix = '__BUJ_Graph__', '__BUJ_Set_Img_Dir__', 'BUJ: '
                elif active_key.startswith('__LS'):
                    graph_size_key, reset_key, prefix = '__LS_Graph__', '__LS_Set_Img_Dir__', 'REC: '
                graph_size = window[graph_size_key].get_size()
                uint8_data, flt_data, size = g_help.load_image(filename, graph_size, stack=True, prefix=prefix)
                reset = not window[reset_key].metadata['State'] == 'Set'

                # Check if data was successfully converted to uint8
                # Save the stack in the correct image dictionary
                if (uint8_data and (num_files is None or num_files == len(uint8_data.keys()))
                        and not reset):
                    stack = Stack(uint8_data, flt_data, size, filename)
                    if active_key.startswith('__LS'):
                        winfo.ls_images[image_key] = stack
                    elif active_key.startswith('__BUJ'):
                        winfo.buj_images[image_key] = stack
                    elif active_key.startswith('__REC'):
                        winfo.rec_images[image_key] = stack
                    metadata_change(window, [(target_key, stack.shortname)])
                    toggle(window, [target_key], state="Set")
                    print(f'{prefix}The file {stack.shortname} was loaded.')
                else:
                    # Incorrect file loaded, don't keep iterating through it
                    print(f'{prefix}An incorrect file was loaded. Either there was a file type error', end=' ')
                    print('or, if a stack, the number of files may not equal that expected from the fls.')
                remove = True
            except ValueError:
                print(f'{prefix}Value Error, had to remove item from queue.')
                remove = True
            # This warning captures the case when a file might be present after
            # creation but hasn't fully loaded
            except UserWarning:
                disable_elem_list = disable_elem_list + conflict_keys
    # Path doesn't exist, remove item from queue
    else:
        print(f'{prefix}There is no valid image name here.')
        remove = True
    return remove, disable_elem_list


def readlines(process, queue):
    while process.poll() is None:
        queue.put(process.stdout.readline())


def tie_readlines(process, queue):
    pass


def load_file_queue(winfo, window):
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
    # If they do load image and save data and remove loading data from queue
    disable_elem_list = []
    delete_indices = []
    for i in range(len(winfo.fiji_queue)):
        filename, active_key, image_key, target_key, conflict_keys, cmd = winfo.fiji_queue[i]
        if active_key.startswith('__LS'):
            prefix = 'LS: '
        elif active_key.startswith('__BUJ'):
            prefix = 'BUJ: '
        elif active_key.startswith('__REC'):
            prefix = 'REC: '

        # Zero-eth alignment and no process running
        if i == 0 and winfo.proc is None:
            if winfo.kill_proc:
                for item in winfo.kill_proc:
                    if item in active_key:
                        delete_indices.append(i)
                if i in delete_indices:
                    print(f'{prefix}The {active_key} process was removed.')
            if i not in delete_indices:
                winfo.proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE,
                                              universal_newlines=True, bufsize=1)
                winfo.fiji_thread_queue = Queue()
                winfo.fiji_thread = Thread(target=readlines, args=(winfo.proc, winfo.fiji_thread_queue), daemon=True)
                winfo.fiji_thread.start()
                disable_elem_list = disable_elem_list + conflict_keys

                if active_key in ["__LS_Run_Align__", "__BUJ_Unflip_Align__", "__BUJ_Flip_Align__",]:
                    align_type = 'Linear SIFT'
                elif active_key in [ "__BUJ_Elastic_Align__"]:
                    align_type = 'bUnwarpJ'
                print('--- Opening FIJI ---')
                prompt = f'--- Starting {align_type} Alignment ---'
                print(prefix, prompt)
                winfo.output_window['FIJI_OUTPUT'].update(value=f'{prompt}\n', append=True)

        # Zero-eth alignment and process currently running
        elif i == 0 and winfo.proc is not None:
            # Poll the process
            poll = winfo.proc.poll()
            # Get information from the FIJI thread
            try:
                line = winfo.fiji_thread_queue.get_nowait()  # False for non-blocking, raises Empty if empty
                winfo.output_window['FIJI_OUTPUT'].update(value=f'{line}', append=True)
            except Empty:
                pass

            # Kill the process for whatever reason
            remove = False
            if winfo.kill_proc:
                for item in winfo.kill_proc:
                    if item in active_key:
                        try:
                            # Terminate subprocess
                            winfo.proc.terminate()
                        except OSError:
                            pass
                        remove = True
                        poll = True
                if remove:
                    print(f'{prefix}The {active_key} process was removed.')
            # Process is finished, file was made and saved
            if os_path.exists(filename) and poll is not None and not remove:
                num_files = None
                remove, disable_elem_list = file_loading(winfo, window, filename, active_key, image_key,
                                                         target_key, conflict_keys, num_files, disable_elem_list)
            # Process finished but nothing made
            elif poll is not None:
                remove = True
                print(f'{prefix}FIJI did not complete its task successfully!')
            # Decide whether the alignment should be removed from the queue
            if remove:
                delete_indices.append(i)
                # Reset the process that is finished
                try:
                    if not winfo.fiji_thread.is_alive():
                        winfo.fiji_thread.join()
                    winfo.proc.close()
                except:
                    pass
                winfo.proc = None
                winfo.fiji_thread = None
                winfo.fiji_thread_queue = None
                prompt = '--- FIJI has closed ---'
                print(prefix, prompt)
                winfo.output_window['FIJI_OUTPUT'].update(value=f'{prompt}\n\n', append=True)
            else:
                disable_elem_list = disable_elem_list + conflict_keys
        elif i > 0:
            next_active = winfo.fiji_queue[i][1]
            if winfo.kill_proc:
                for item in winfo.kill_proc:
                    if item in next_active:
                        delete_indices.append(i)
                if i in delete_indices:
                    print(f'{prefix}The {next_active} process was removed.')
            if i not in delete_indices:
                disable_elem_list = disable_elem_list + conflict_keys
    if delete_indices:
        rev_sort_del_items = list(set(delete_indices))
        rev_sort_del_items.sort(reverse=True)
        for item in rev_sort_del_items:
            winfo.fiji_queue.pop(item)
    winfo.kill_proc = []

    # new_buj_queue = winfo.buj_file_queue
    for j in range(len(winfo.buj_file_queue)):
        filename, active_key, image_key, target_key, conflict_keys, num_files = winfo.buj_file_queue[j]
        remove2, disable_elem_list = file_loading(winfo, window, filename, active_key, image_key,
                                                  target_key, conflict_keys, num_files, disable_elem_list)
    winfo.buj_file_queue = []

    disable_elements(window, disable_elem_list)


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
        # elif elem_key in keys['multiline']:
        #     window[elem_key].Update(value=value, append=True)
        else:
            window[elem_key].Update(value=value)


def change_list_ind_color(window, current_tab, elem_ind_val_list):

    if current_tab == 'reconstruct_tab':
        num_list = list(range(11))
    if current_tab == 'bunwarpj_tab':
        num_list = list(range(3))

    for listbox_key, green_choices in elem_ind_val_list:
        listbox = window[listbox_key]
        grey_choices = setdiff1d(num_list, green_choices)
        for index in green_choices:
            listbox.Widget.itemconfig(index, fg='black', bg="light green")
        for index in grey_choices:
            listbox.Widget.itemconfig(index, fg='light grey', bg=sg.theme_input_background_color())


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


def rec_get_listbox_ind_from_key(key_list):
    indices = [0, 10]
    for key in key_list:
        if key == 'color_b':
            ind = 1
        if key == 'bxt':
            ind = 2
        elif key == 'byt':
            ind = 3
        elif key == 'bbt':
            ind = 4
        elif key == 'phase_e':
            ind = 5
        elif key == 'phase_m':
            ind = 6
        elif key == 'dIdZ_e':
            ind = 7
        elif key == 'dIdZ_m':
            ind = 8
        elif key == 'inf_im':
            ind = 9
        indices.append(ind)
    return indices


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
        if elem_key in keys['combo']:
            window[elem_key].Update(readonly=True)
        else:
            window[elem_key].Update(disabled=False)


def ptie_init_thread(winfo, path, fls1_path, fls2_path, stack_name,
                     files1, files2, flip, tfs_value):
    try:
        if float(winfo.window['__REC_M_Volt__'].get()) > 0:
            accel_volt = float(winfo.window['__REC_M_Volt__'].get()) * 1e3
        else:
            print(f'REC: Error with Voltage.')
            raise

        stack1, stack2, ptie = load_data(path, fls1_path, stack_name, flip, fls2_path)
        string_vals = []
        for def_val in ptie.defvals:
            val = str(def_val)
            string_vals.append(val)

        if tfs_value == 'Single':
            prefix = 'tfs'
        else:
            prefix = 'unflip'
        im_name = files1[0]
        # display_img = images['REC_Stack'].byte_data[0]
        metadata_change(winfo.window, [('__REC_Image__', f'{prefix}/{im_name}')])

        length_slider = len(string_vals)
        winfo.window['__REC_Def_Combo__'].update(values=string_vals)
        winfo.window['__REC_Def_List__'].update(ptie.defvals, set_to_index=0,
                                          scroll_to_index=0)
        winfo.window['__REC_Def_List__'].metadata['length'] = length_slider
        update_slider(winfo.window, [('__REC_Defocus_Slider__', {"slider_range": (0, max(length_slider - 3, 0)),
                                                           "value": 0})])
        winfo.rec_defocus_slider_set = 0
        winfo.rec_ptie = ptie
        winfo.rec_microscope = Microscope(E=accel_volt, Cs=200.0e3, theta_c=0.01e-3, def_spr=80.0)
        winfo.rec_files1 = files1
        winfo.rec_files2 = files2
        enable_elements(winfo.window, ["__REC_Reset_FLS__", "__REC_Reset_Img_Dir__"])
        toggle(winfo.window, elem_list=['__REC_Set_FLS__'])
    except:
        print(f'REC: Something went wrong loading in image data.')
        print(f'REC: 1. Check to make sure the fls file(s) match the aligned file chosen.', end=' ')
        print('Otherwise PYTIE will search the wrong directories.')
        print(f'REC: 2. Check to see voltage is numerical and above 0.')
        raise
    enable_elements(winfo.window, ["__REC_Reset_FLS__", "__REC_Reset_Img_Dir__"])
    winfo.ptie_init_thread = None
    print('--- Exited PTIE Initialization ---')

def ptie_recon_thread(winfo, window, graph, colorwheel_graph, images, current_tab):
    ptie = winfo.rec_ptie
    microscope = winfo.rec_microscope
    def_val = float(window['__REC_Def_Combo__'].Get())
    def_ind = ptie.defvals.index(def_val)
    dataname = 'example'
    hsv = window['__REC_Colorwheel__'].get() == 'HSV'
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
    bottom, top, left, right = None, None, None, None
    for i in range(len(winfo.rec_mask_coords)):
        x, y = winfo.rec_mask_coords[i]
        if right is None or x > right:
            right = x
        if left is None or x < left:
            left = x
        if bottom is None or graph.get_size()[1] - y > bottom:
            bottom = graph.get_size()[1] - y
        if top is None or graph.get_size()[1] - y < top:
            top = graph.get_size()[1] - y
    if (bottom, top, left, right) == (None, None, None, None):
        bottom, top, left, right = graph.get_size()[1], 0, 0, graph.get_size()[0]

    # Scaling the image from the graph region to the regular sized image
    reg_width, reg_height = images['REC_Stack'].lat_dims
    scale_x, scale_y = reg_width / graph.get_size()[0], reg_height / graph.get_size()[1]

    # Make sure the image is square
    if round(right * scale_x) - round(left * scale_x) != round(bottom * scale_y) - round(top * scale_y):
        print(f'REC: The crop region was not square. Fixing.')
        if round(right * scale_x) - round(left * scale_x) < round(bottom * scale_y) - round(top * scale_y):
            if (round(right * scale_x) - round(left * scale_x)) % 2 != 0:
                if right == reg_width:
                    left -= 1
                else:
                    right += 1
            elif (round(bottom * scale_y) - round(top * scale_y)) % 2 != 0:
                bottom -= 1
        if round(right * scale_x) - round(left * scale_x) > round(bottom * scale_y) - round(top * scale_y):
            if (round(right * scale_x) - round(left * scale_x)) % 2 != 0:
                right -= 1
                if right == reg_width:
                    left -= 1
                else:
                    right += 1
            elif (round(bottom * scale_y) - round(top * scale_y)) % 2 != 0:
                if bottom == reg_height:
                    top -= 1
                else:
                    bottom += 1

    # Set ptie crop
    ptie.crop['right'], ptie.crop['left'] = round(right * scale_x), round(left * scale_x)
    ptie.crop['bottom'], ptie.crop['top'] = round(bottom * scale_y), round(top * scale_y)

    if not qc_passed:
        print(f'REC: QC value should be an integer or float and not negative. Change value.')
        update_values(window, [('__REC_QC_Input__', '0.00')])
    else:
        try:
            print(f'REC: Reconstructing for defocus value: {ptie.defvals[def_ind]} nm')
            rot, x_trans, y_trans = (winfo.rec_transform[0], winfo.rec_transform[1], winfo.rec_transform[2])
            x_trans, y_trans = x_trans * scale_x, y_trans * scale_y
            ptie.rotation, ptie.x_transl, ptie.y_transl = rot, int(x_trans), int(y_trans)
            results = TIE(def_ind, ptie, microscope,
                          dataname, sym, qc, hsv, save,
                          longitudinal_deriv, v=0)

            # This will need to consider like the cropping region
            winfo.rec_tie_results = results
            winfo.rec_def_val = def_val
            winfo.rec_sym = sym
            winfo.rec_qc = qc
            winfo.graph_slice = (round(right * scale_x) - round(left * scale_x),
                                 round(bottom * scale_x) - round(top * scale_x))

            loaded_green_list = []
            for key in results:
                float_array = results[key]
                if key == 'color_b':
                    float_array = g_help.slice(float_array, winfo.graph_slice)
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
                if uint8_data:
                    image = Image(uint8_data, float_data, (winfo.graph_slice[0], winfo.graph_slice[1], 1), f'/{key}')
                    image.byte_data = g_help.vis_1_im(image)
                    winfo.rec_images[key] = image
                    loaded_green_list.append(key)
                else:
                    winfo.rec_images[key] = None

            # Update window
            list_color_indices = rec_get_listbox_ind_from_key(loaded_green_list)
            change_list_ind_color(window, current_tab, [('__REC_Image_List__', list_color_indices)])
            redraw_graph(graph, winfo.rec_images['color_b'].byte_data)
            redraw_graph(colorwheel_graph, winfo.rec_colorwheel)
            metadata_change(window, [('__REC_Image__', f'{def_val} Color')])
            toggle(window, ['__REC_Image__'], state='Set')
            update_slider(window, [('__REC_Image_Slider__', {"value": 6 - 1})])
            window['__REC_Image_List__'].update(set_to_index=1, scroll_to_index=1)
            winfo.rec_image_slider_set = 6 - 1
            winfo.rec_last_image_choice = 'Color'
            winfo.rec_ptie = ptie
        except:
            print(f'REC: There was an error when running TIE.')
            raise
    enable_elements(winfo.window, ["__REC_Reset_FLS__", "__REC_Reset_Img_Dir__"])
    winfo.ptie_recon_thread = None
    print('--- Exited Reconstruction ---')


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
    prefix = 'HOM: '
    # Get directories for Fiji and images
    if event == '__Fiji_Set__':
        winfo.fiji_path = values['__Fiji_Path__']
        if not os_path.exists(winfo.fiji_path):
            print(f'{prefix}This Fiji path is incorrect, try again.')
        else:
            print(f'{prefix}Fiji path is set, you may now proceed to registration.')
            disable_elements(window, ['__Fiji_Path__', '__Fiji_Set__', '__Fiji_Browse__'])
            window.Refresh()
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
        active_keys = ['__LS_View_Stack__', '__LS_Run_Align__', '__LS_Reset_Img_Dir__',
                       '__LS_Adjust__', '__LS_unflip_reference__', '__LS_flip_reference__',
                       '__LS_Image_Dir_Path__', '__LS_Set_Img_Dir__', '__LS_Image_Dir_Browse__',
                       '__LS_FLS_Combo__', '__LS_Load_FLS1__', '__LS_Set_FLS__',
                       '__LS_Load_FLS2__', "__LS_Reset_FLS__", "__LS_TFS_Combo__",
                       '__LS_transform_rot__', '__LS_transform_x__', '__LS_transform_y__',
                       '__LS_horizontal_flip__']
        # Don't view/load any images accidentally when adjusting images
        if window['__LS_Set_Img_Dir__'].metadata['State'] == 'Set':
            if window['__LS_Set_FLS__'].metadata['State'] == 'Def':
                enable_list.extend(['__LS_FLS_Combo__', "__LS_TFS_Combo__"])
                if (window['__LS_FLS_Combo__'].Get() == 'Two' and
                        window['__LS_FLS2__'].metadata['State'] == 'Def'):
                    enable_list.extend(['__LS_Load_FLS2__'])
                if window['__LS_FLS1__'].metadata['State'] == 'Def':
                    enable_list.extend(['__LS_Load_FLS1__'])
                if (window['__LS_FLS1__'].metadata['State'] == 'Set' and
                        window['__LS_FLS2__'].metadata['State'] == 'Set'):
                    enable_list.extend(['__LS_Set_FLS__'])
            elif window['__LS_Set_FLS__'].metadata['State'] == 'Set':
                if adjust_button.metadata['State'] == 'Def':
                    if window['__LS_TFS_Combo__'].Get() != 'Single':
                        enable_list.extend(['__LS_unflip_reference__', '__LS_flip_reference__'])
                    if images and 'stack' in images:
                        enable_list.append('__LS_View_Stack__')
                    # Don't enable load stack when viewing stack
                    if view_stack_button.metadata['State'] == 'Def':
                        if window['__LS_Set_Img_Dir__'].metadata['State'] == 'Set':
                            enable_list.append('__LS_Run_Align__')
                if (view_stack_button.metadata['State'] == 'Def' and
                        window['__LS_Set_Img_Dir__'].metadata['State'] == 'Set' and
                        window['__LS_TFS_Combo__'].Get() != 'Single'):
                    enable_list.extend(['__LS_Adjust__'])
            if window['__LS_Adjust__'].metadata['State'] == 'Def':
                if images and 'stack' in images:
                    enable_list.append('__LS_View_Stack__')
            elif window['__LS_Adjust__'].metadata['State'] == 'Set':
                enable_list.extend(['__LS_transform_rot__', '__LS_transform_x__',
                                    '__LS_transform_y__',  '__LS_horizontal_flip__'])
            if (window['__LS_Adjust__'].metadata['State'] == 'Def' and
                    view_stack_button.metadata['State'] == 'Def'):
                enable_list.extend(["__LS_Reset_FLS__"])
        # If image dir is not set
        elif window['__LS_Set_Img_Dir__'].metadata['State'] == 'Def':
            enable_list.extend(['__LS_Image_Dir_Path__',
                                '__LS_Set_Img_Dir__',
                                '__LS_Image_Dir_Browse__'])
        enable_list.extend(['__LS_Reset_Img_Dir__'])
        disable_list = setdiff1d(active_keys, enable_list)
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
    prefix = 'LS: '

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
        if os_path.exists(image_dir):
            winfo.ls_image_dir = image_dir
            toggle(window, ['__LS_Set_Img_Dir__'], state='Set')
            print(f'{prefix}The path is set: {image_dir}.')
        else:
            print(f'{prefix}This pathname is incorrect.')

    elif event == '__LS_FLS_Combo__' or event == '__LS_TFS_Combo__':
        winfo.ls_fls_files = [None, None]
        fls_value = window['__LS_FLS_Combo__'].Get()
        tfs_value = window['__LS_TFS_Combo__'].Get()
        metadata_change(window, ['__LS_FLS2__', '__LS_FLS1__'], reset=True)
        toggle(window, ['__LS_FLS2__', '__LS_FLS1__'], state='Def')
        # FLS Combo Chosen
        if event == '__LS_FLS_Combo__':
            # If one fls file is to be used
            metadata_change(window, [('__LS_FLS_Combo__', fls_value)])
            if fls_value == 'One':
                toggle(window, ['__LS_FLS_Combo__', '__LS_FLS2__'], state='Set')
                if tfs_value == 'Unflip/Flip':
                    if window['__LS_Set_FLS__'].metadata['State'] == 'Set':
                        window['__LS_unflip_reference__'].update(True)
                    val = 'Both'
                elif tfs_value == 'Single':
                    window['__LS_unflip_reference__'].ResetGroup()
                    val = tfs_value
            # If two fls file is to be used
            elif fls_value == 'Two':
                val = fls_value
                if window['__LS_Set_FLS__'].metadata['State'] == 'Set':
                    window['__LS_unflip_reference__'].update(True)
                metadata_change(window, ['__LS_FLS_Combo__', '__LS_TFS_Combo__'], reset=True)
                toggle(window, ['__LS_FLS_Combo__', '__LS_TFS_Combo__',
                                '__LS_FLS2__'], state='Def')
        # TFS Combo Chosen
        elif event == '__LS_TFS_Combo__':
            metadata_change(window, [('__LS_TFS_Combo__', tfs_value)])
            if tfs_value == 'Unflip/Flip':
                val = 'Two'
                if window['__LS_Set_FLS__'].metadata['State'] == 'Set':
                    window['__LS_unflip_reference__'].update(True)
                metadata_change(window, ['__LS_FLS_Combo__', '__LS_TFS_Combo__'], reset=True)
                toggle(window, ['__LS_FLS_Combo__', '__LS_TFS_Combo__',
                                '__LS_FLS2__'], state='Def')
            elif tfs_value == 'Single':
                val = tfs_value
                window['__LS_unflip_reference__'].ResetGroup()
                metadata_change(window, [('__LS_FLS_Combo__', 'One')])
                toggle(window, ['__LS_FLS_Combo__', '__LS_TFS_Combo__',
                                '__LS_FLS2__'], state='Set')
        window['__LS_FLS1_Text__'].update(value=window['__LS_FLS1_Text__'].metadata[val])
        window['__LS_FLS2_Text__'].update(value=window['__LS_FLS2_Text__'].metadata[val])

    # Load FLS files
    elif event == '__LS_FLS1_Staging__' or event == '__LS_FLS2_Staging__':
        tfs_value = window['__LS_TFS_Combo__'].Get()
        fls_value = window['__LS_FLS_Combo__'].Get()
        if 'FLS1' in event:
            fls_path = window['__LS_FLS1_Staging__'].Get()
            update_values(window, [('__LS_FLS1_Staging__', 'None')])
            target_key = '__LS_FLS1__'
        elif 'FLS2' in event:
            fls_path = window['__LS_FLS2_Staging__'].Get()
            update_values(window, [('__LS_FLS2_Staging__', 'None')])
            target_key = '__LS_FLS2__'
        if os_path.exists(fls_path) and fls_path.endswith('.fls'):
            fls = File_Object(fls_path)
            if 'FLS1' in event:
                winfo.ls_fls_files[0] = fls
                if tfs_value == 'Unflip/Flip' and fls_value == 'One':
                    winfo.ls_fls_files[1] = fls
            elif 'FLS2' in event:
                winfo.ls_fls_files[1] = fls
            metadata_change(window, [(target_key, fls.shortname)])
            toggle(window, [target_key], state='Set')
        else:
            print(f'{prefix}Pathname is incorrect or none selected.')

    # Reset FLS
    elif event == '__LS_Reset_FLS__':
        # Reset FLS but don't reset loaded stack
        winfo.ls_images = {}
        winfo.ls_fls_files = [None, None]
        winfo.ls_files1 = None
        winfo.ls_files2 = None
        winfo.kill_proc.append('LS')


        # --- Set up loading files --- #

        graph.Erase()
        metadata_change(window, ['__LS_FLS1__', '__LS_FLS2__',
                                 '__LS_Image1__', '__LS_Image2__'], reset=True)
        toggle(window, ['__LS_FLS_Combo__', '__LS_TFS_Combo__', '__LS_Adjust__',
                        '__LS_View_Stack__', '__LS_Image1__', '__LS_Image2__',
                        '__LS_FLS1__', '__LS_FLS2__', '__LS_Set_FLS__'], state='Def')

        window['__LS_FLS1_Text__'].update(value=window['__LS_FLS1_Text__'].metadata['Two'])
        window['__LS_FLS2_Text__'].update(value=window['__LS_FLS2_Text__'].metadata['Two'])
        window['__LS_FLS1_Text__'].metadata['State'] = 'Two'
        window['__LS_FLS2_Text__'].metadata['State'] = 'Two'
        update_values(window, [('__LS_FLS1_Staging__', ''), ('__LS_FLS2_Staging__', ''),
                               ('__LS_transform_x__', '0'), ('__LS_transform_y__', '0'),
                               ('__LS_transform_rot__', "0"), ('__LS_horizontal_flip__', True)
                               ])
        window['__LS_unflip_reference__'].ResetGroup()

        # Re-init reconstruct
        update_slider(window, [('__LS_Stack_Slider__', {'value': 0,
                                                        'slider_range': (0, 0)})])
        print(f'{prefix}FLS reset.')

    # Set the fls files and load in images
    elif event == '__LS_Set_FLS__':
        tfs_value = window['__LS_TFS_Combo__'].Get()
        fls_value = window['__LS_FLS_Combo__'].Get()
        if tfs_value == 'Unflip/Flip':
            orientation = 'unflip'
            fls_file_names = [winfo.ls_fls_files[0].path, winfo.ls_fls_files[1].path]
        else:
            fls_file_names = [winfo.ls_fls_files[0].path, None]
        check = check_setup(image_dir, tfs_value, fls_value, fls_file_names, prefix=' LS:')
        if check:
            # Set the image dir button
            path1, path2, files1, files2 = check[1:]
            toggle(window, ['__LS_Set_Img_Dir__', '__LS_Set_FLS__'], state='Set')

            # Prepare reference data
            ref1 = files1[len(files1)//2]
            ref1_path = al_join([path1, ref1], '/')
            uint8_1, flt_data_1, size_1 = g_help.load_image(ref1_path, graph.get_size(), prefix=' LS: ')
            uint8_2 = None
            if tfs_value == 'Single':
                ref2_path = None
            else:
                ref2 = files2[len(files2)//2]
                ref2_path = al_join([path2, ref2], '/')
                uint8_2, flt_data_2, size_2 = g_help.load_image(ref2_path, graph.get_size(), prefix=' LS: ')

            # Load image data as numpy arrays for uint8, numerical val, and size
            if uint8_1:
                # Create image instances and store byte data for TK Canvas
                image1 = Image(uint8_1, flt_data_1, size_1, ref1_path)
                image1.byte_data = g_help.vis_1_im(image1)
                if uint8_2:
                    image2 = Image(uint8_2, flt_data_2, size_2, ref2_path)
                    image2.byte_data = g_help.vis_1_im(image2)
                else:
                    image2 = None

                # Display ref filename and load display data
                if tfs_value == 'Unflip/Flip':
                    window['__LS_unflip_reference__'].update(True)

                # Update window only if view stack not set
                if view_stack_button.metadata['State'] == 'Def':
                    metadata_change(window, [('__LS_Image1__', al_join([orientation, image1.shortname], '/'))])
                    toggle(window, ['__LS_Image1__'])
                    display_img = image1.byte_data

                # Push data to winfo
                winfo.ls_images['image1'] = image1
                winfo.ls_images['image2'] = image2
                winfo.ls_files1 = files1
                winfo.ls_files2 = files2
                print(f'{prefix}Directory properly set-up.')
        else:
            print(f'{prefix}Look at Help Tab for correct file setup.')

    # Change reference state between flip/unflip images
    elif change_ref:
        if image_dir:
            if orientation in ['unflip', 'tfs']:
                image = images['image1']
            elif orientation == 'flip':
                image = images['image2']
            display_img = image.byte_data
            metadata_change(window, [('__LS_Image1__', al_join([orientation, image.shortname], '/'))])

    # Load flipped image for adjustment
    elif event == '__LS_Adjust__':
        if window['__LS_Set_Img_Dir__'].metadata['State'] == 'Set':
            # Quit flip adjustment
            if adjust_button.metadata['State'] == 'Set':
                if orientation == 'unflip':
                    image = 'image1'
                elif orientation == 'flip':
                    image = 'image2'
                display_img = images[image].byte_data
                toggle(window, ['__LS_Adjust__', '__LS_Image2__'], state='Def')

            # Begin flip adjustment
            elif adjust_button.metadata['State'] == 'Def':
                if orientation == 'unflip':
                    img_1 = images['image2']
                    img_2 = images['image1']
                    img2_orientation = 'flip'
                elif orientation == 'flip':
                    img_1 = images['image1']
                    img_2 = images['image2']
                    img2_orientation = 'unflip'
                display_img = g_help.overlay_images(img_1, img_2, transform, img_1.x_size, graph.get_size()[0])
                metadata_change(window, [('__LS_Image2__', al_join([img2_orientation, img_1.shortname], '/'))])
                toggle(window, ['__LS_Adjust__', '__LS_Image2__'], state='Set')

        else:
            print(f'{prefix}No flip data to adjust, make sure to set your working directory.')

    # Run Linear SIFT alignment
    elif event == '__LS_Run_Align__':
        if window['__LS_Set_Img_Dir__'].metadata['State'] == 'Set':
            param_test = values['__LS_param_test__']
            sift_params = load_ls_sift_params(values)
            tfs_value = window['__LS_TFS_Combo__'].Get()
            fls_value = window['__LS_FLS_Combo__'].Get()
            transform_params = None, None, None, None
            if tfs_value != 'Single':
                rot, x_shift, y_shift, horizontal = transform
                transform_params = rot, x_shift, -y_shift, horizontal

            # Decide whether file should be created
            save, overwrite_signal = True, []
            if param_test:
                filename = al_join([image_dir, "Param_Test.tif"], '/')
            else:
                filename, overwrite_signal = run_save_window(winfo, event, image_dir, tfs=tfs_value)
                save = overwrite_signal[0]
                if filename == 'close' or not filename or not save:
                    print(f'{prefix}Exited save screen without saving image.')
                    save = False
                else:
                    filename = filename[0]

            # Create files
            if save:
                if os_path.exists(filename):
                    os_remove(filename)
                if tfs_value == 'Unflip/Flip':
                    orientation = 'unflip'
                    fls_file_names = [winfo.ls_fls_files[0].path, winfo.ls_fls_files[1].path]
                else:
                    fls_file_names = [winfo.ls_fls_files[0].path, None]
                ijm_macro_script = run_ls_align(image_dir, orientation, param_test,
                                                      sift_params, transform_params, filename,
                                                      tfs_value=tfs_value, fls_value=fls_value,
                                                      fls_files=fls_file_names)
                cmd = g_help.run_macro(ijm_macro_script, image_dir, winfo.fiji_path)

                # Load the stack when ready
                target_key = '__LS_Stack__'
                image_key = 'stack'
                conflict_keys = ['__LS_Run_Align__']
                winfo.fiji_queue.append((filename, event, image_key, target_key,
                                         conflict_keys, cmd))
        else:
            print(f'{prefix}A valid directory has not been set.')

    # View the loaded image stack
    elif event == '__LS_View_Stack__':
        if view_stack_button.metadata['State'] == 'Def':
            # Get stack information
            tfs_value = window['__LS_TFS_Combo__'].Get()
            stack = images['stack']
            slider_val = 0
            slider_range = (0, stack.z_size - 1)
            display_img = stack.byte_data[slider_val]

            # Update window
            name = window['__LS_Stack__'].get()
            if tfs_value == 'Single':
                prefix = 'tfs'
            else:
                prefix = 'unflip'
            if 'Param' in name:
                im_name = winfo.ls_files1[len(winfo.ls_files1)//2-1]
            else:
                im_name = winfo.ls_files1[slider_val]
            metadata_change(window, [('__LS_Image1__', f'{prefix}/{im_name}')])
            toggle(window, ['__LS_Adjust__'], state='Def')
            toggle(window, ['__LS_Image1__', '__LS_View_Stack__'], state='Set')
            update_slider(window, [('__LS_Stack_Slider__', {"value": slider_val, "slider_range": slider_range})])
        elif view_stack_button.metadata['State'] == 'Set':
            # Update window
            if (window['__LS_Set_Img_Dir__'].metadata['State'] == 'Set' and
                    window['__LS_Set_FLS__'].metadata['State'] == 'Set'):
                if orientation in ['unflip', 'tfs']:
                    image_key = 'image1'
                else:
                    image_key = 'image2'
                image = images[image_key]
                metadata_change(window, [('__LS_Image1__', al_join([orientation, image.shortname], '/'))])
                display_img = image.byte_data
            else:
                graph.Erase()
                metadata_change(window, ['__LS_Image1__'], reset=True)
                toggle(window, ['__LS_Image1__'])
            toggle(window, ['__LS_View_Stack__'])

    # Change the slider
    elif event == '__LS_Stack_Slider__':
        # Get image from stack
        if 'stack' in images:
            stack = images['stack']
            tfs_value = window['__LS_TFS_Combo__'].Get()
            slider_val = int(values["__LS_Stack_Slider__"])

            # Update window
            display_img = stack.byte_data[slider_val]
            if tfs_value == 'Single':
                prefix = 'tfs'
            else:
                prefix = 'unflip'
            name = window['__LS_Stack__'].get()
            if 'Param' in name:
                if slider_val < 3:
                    slider_val = slider_val % 3 - 1
                    im_name = winfo.ls_files1[len(winfo.ls_files1) // 2 + slider_val]
                    if prefix != 'tfs':
                        prefix = 'unflip'
                elif slider_val >= 3:
                    slider_val = slider_val % 3 - 1
                    im_name = winfo.ls_files2[len(winfo.ls_files2) // 2 + slider_val]
                    if prefix != 'tfs':
                        prefix = 'flip'
            else:
                if slider_val < len(winfo.ls_files1):
                    im_name = winfo.ls_files1[slider_val]
                    if prefix != 'tfs':
                        prefix = 'unflip'
                elif slider_val >= len(winfo.ls_files1):
                    slider_val = slider_val % len(winfo.ls_files2)
                    im_name = winfo.ls_files2[slider_val]
                    if prefix != 'tfs':
                        prefix = 'flip'
            metadata_change(window, [('__LS_Image1__', f'{prefix}/{im_name}')])

    # Scroll through stacks in the graph area
    elif scroll:
        stack = images['stack']
        tfs_value = window['__LS_TFS_Combo__'].Get()
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
        name = window['__LS_Stack__'].get()
        prefix = ''
        if tfs_value == 'Single':
            prefix = 'tfs'
        if 'Param' in name:
            if slider_val < 3:
                slider_val = slider_val % 3 - 1
                im_name = winfo.ls_files1[len(winfo.ls_files1) // 2 + slider_val]
                if prefix != 'tfs':
                    prefix = 'unflip'
            elif slider_val >= 3:
                slider_val = slider_val % 3 - 1
                im_name = winfo.ls_files2[len(winfo.ls_files2) // 2 + slider_val]
                if prefix != 'tfs':
                    prefix = 'flip'
        else:
            if slider_val < len(winfo.ls_files1):
                im_name = winfo.ls_files1[slider_val]
                if prefix != 'tfs':
                    prefix = 'unflip'
            elif slider_val >= len(winfo.ls_files1):
                slider_val = slider_val % len(winfo.ls_files2)
                im_name = winfo.ls_files2[slider_val]
                if prefix != 'tfs':
                    prefix = 'flip'
        metadata_change(window, [('__LS_Image1__', f'{prefix}/{im_name}')])

    # Apply any immediate changes
    if overlay:
        if orientation == 'unflip':
            img_1 = images['image2']
            img_2 = images['image1']
        elif orientation == 'flip':
            img_1 = images['image1']
            img_2 = images['image2']
        display_img = g_help.overlay_images(img_1, img_2, transform, img_1.x_size, graph.get_size()[0])
    winfo.ls_past_transform = transform

    # Reset the image directory to nothing
    if event == '__LS_Reset_Img_Dir__':
        reset(winfo, window, current_tab)
        winfo.kill_proc.append('LS')
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
    def special_enable_disable(window, adjust_button, view_stack_button, make_mask_button, images):
        enable_list = []
        active_keys = ['__BUJ_View__', '__BUJ_Flip_Align__', '__BUJ_Unflip_Align__',
                       '__BUJ_Elastic_Align__', '__BUJ_Load_Flip_Stack__', '__BUJ_Load_Unflip_Stack__',
                       '__BUJ_Image_Dir_Path__', '__BUJ_Set_Img_Dir__', '__BUJ_Reset_Img_Dir__',

                       '__BUJ_Load_FLS1__', '__BUJ_Set_FLS__', '__BUJ_FLS_Combo__',
                       '__BUJ_Load_FLS2__', "__BUJ_Reset_FLS__", "__BUJ_TFS_Combo__",
                       '__BUJ_transform_rot__', '__BUJ_transform_x__', '__BUJ_transform_y__',
                       '__BUJ_horizontal_flip__',

                       '__BUJ_Image_Dir_Browse__', '__BUJ_Adjust__', '__BUJ_Make_Mask__',
                       '__BUJ_unflip_reference__', '__BUJ_flip_reference__', '__BUJ_Image_Choice__',
                       '__BUJ_Clear_Unflip_Mask__', '__BUJ_Clear_Flip_Mask__',
                       '__BUJ_Load_Mask__', '__BUJ_Reset_Mask__', '__BUJ_Make_Mask__',
                       ]
        if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set':
            if window['__BUJ_Set_FLS__'].metadata['State'] == 'Def':
                enable_list.extend(['__BUJ_FLS_Combo__', "__BUJ_TFS_Combo__"])
                if (window['__BUJ_FLS_Combo__'].Get() == 'Two' and
                        window['__BUJ_FLS2__'].metadata['State'] == 'Def'):
                    enable_list.extend(['__BUJ_Load_FLS2__'])
                if window['__BUJ_FLS1__'].metadata['State'] == 'Def':
                    enable_list.extend(['__BUJ_Load_FLS1__'])
                if (window['__BUJ_FLS1__'].metadata['State'] == 'Set' and
                        window['__BUJ_FLS2__'].metadata['State'] == 'Set'):
                    enable_list.extend(['__BUJ_Set_FLS__'])
            elif window['__BUJ_Set_FLS__'].metadata['State'] == 'Set':
                if adjust_button.metadata['State'] == 'Def' and make_mask_button.metadata['State'] == 'Def':
                    enable_list.extend(['__BUJ_unflip_reference__', '__BUJ_flip_reference__'])
                    if ('BUJ_flip_stack' in images or 'BUJ_unflip_stack' in images or 'BUJ_stack' in images):
                        enable_list.append('__BUJ_View__')
                    if view_stack_button.metadata['State'] == 'Def':
                        enable_list.append('__BUJ_Load_Flip_Stack__')
                        enable_list.append('__BUJ_Load_Unflip_Stack__')
                        enable_list.append('__BUJ_Flip_Align__')
                        enable_list.append('__BUJ_Unflip_Align__')
                        enable_list.append('__BUJ_Elastic_Align__')
                if view_stack_button.metadata['State'] == 'Def' and make_mask_button.metadata['State'] == 'Def':
                    if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set':
                        enable_list.extend(['__BUJ_Adjust__'])
                if view_stack_button.metadata['State'] == 'Def' and adjust_button.metadata['State'] == 'Def':
                    enable_list.extend(['__BUJ_Load_Mask__', '__BUJ_Clear_Unflip_Mask__', '__BUJ_Clear_Flip_Mask__'])
                    if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set':
                        enable_list.extend(['__BUJ_Reset_Mask__', '__BUJ_Make_Mask__'])
            if (window['__BUJ_Adjust__'].metadata['State'] == 'Def' and
                    view_stack_button.metadata['State'] == 'Def'):
                enable_list.extend(["__BUJ_Reset_FLS__"])
            if window['__BUJ_Adjust__'].metadata['State'] == 'Set':
                enable_list.extend(['__BUJ_transform_rot__', '__BUJ_transform_x__',
                                    '__BUJ_transform_y__',  '__BUJ_horizontal_flip__'])
            for elem_key in ['__BUJ_Unflip_Stack_Inp__', '__BUJ_Flip_Stack_Inp__', '__BUJ_Stack__']:
                if window[elem_key].metadata['State'] == 'Set' and '__BUJ_Image_Choice__' not in enable_list:
                    enable_list.extend(['__BUJ_Image_Choice__'])
        if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Def':
            enable_list.extend(['__BUJ_Image_Dir_Path__', '__BUJ_Set_Img_Dir__',
                                '__BUJ_Image_Dir_Browse__'])
        enable_list.extend(['__BUJ_Reset_Img_Dir__'])
        disable_list = setdiff1d(active_keys, enable_list)
        enable_elements(window, enable_list)
        disable_elements(window, disable_list)

    # Get rotations and shifts to apply to image (only positive rotations)
    transform = get_transformations(winfo, window, current_tab)
    orientation = get_orientation(window, "BUJ")

    # Grab important elements
    graph = window['__BUJ_Graph__']
    adjust_button = window['__BUJ_Adjust__']
    view_stack_button = window['__BUJ_View__']
    make_mask_button = window['__BUJ_Make_Mask__']

    # Pull in image data from struct object
    image_dir = winfo.buj_image_dir
    images = winfo.buj_images
    display_img = None
    prefix = 'BUJ: '
    draw_mask_points, draw_mask_polygon = False, False

    # Import event handler names (overlaying, etc.)
    overlay = adjust_button.metadata['State'] == 'Set' and winfo.buj_past_transform != transform
    change_ref = (event in ['__BUJ_unflip_reference__', '__BUJ_flip_reference__'] and
                  view_stack_button.metadata['State'] == 'Def')
    change_img = winfo.buj_last_image_choice != values['__BUJ_Image_Choice__'][0]
    scroll = (event in ['MouseWheel:Up', 'MouseWheel:Down']
              and window['__BUJ_View__'].metadata['State'] == 'Set'
              and winfo.true_element == "__BUJ_Graph__")

    # if 'TIMEOUT' not in event and 'HOVER' not in event:
    #     print(event)

    # Set the working directory
    # Set image directory and load in-focus image
    if event == '__BUJ_Set_Img_Dir__':
        image_dir = values['__BUJ_Image_Dir_Path__']
        if os_path.exists(image_dir):
            winfo.buj_image_dir = image_dir
            toggle(window, ['__BUJ_Set_Img_Dir__'], state='Set')
            print(f'{prefix}The path is set: {image_dir}.')
        else:
            print(f'{prefix}This pathname is incorrect.')

    elif event == '__BUJ_FLS_Combo__':
        winfo.buj_fls_files = [None, None]
        fls_value = window['__BUJ_FLS_Combo__'].Get()
        metadata_change(window, ['__BUJ_FLS2__', '__BUJ_FLS1__'], reset=True)
        toggle(window, ['__BUJ_FLS2__', '__BUJ_FLS1__'], state='Def')
        # FLS Combo Chosen
        if event == '__BUJ_FLS_Combo__':
            # If one fls file is to be used
            metadata_change(window, [('__BUJ_FLS_Combo__', fls_value)])
            if fls_value == 'One':
                toggle(window, ['__BUJ_FLS_Combo__', '__BUJ_FLS2__'], state='Set')
                val = 'Both'
            # If two fls file is to be used
            elif fls_value == 'Two':
                val = fls_value
                metadata_change(window, ['__BUJ_FLS_Combo__'], reset=True)
                toggle(window, ['__BUJ_FLS_Combo__', '__BUJ_FLS2__'], state='Def')
        window['__BUJ_FLS1_Text__'].update(value=window['__BUJ_FLS1_Text__'].metadata[val])
        window['__BUJ_FLS2_Text__'].update(value=window['__BUJ_FLS2_Text__'].metadata[val])

    # Load FLS files
    elif event == '__BUJ_FLS1_Staging__' or event == '__BUJ_FLS2_Staging__':
        tfs_value = window['__BUJ_TFS_Combo__'].Get()
        fls_value = window['__BUJ_FLS_Combo__'].Get()
        if 'FLS1' in event:
            fls_path = window['__BUJ_FLS1_Staging__'].Get()
            update_values(window, [('__BUJ_FLS1_Staging__', 'None')])
            target_key = '__BUJ_FLS1__'
        elif 'FLS2' in event:
            fls_path = window['__BUJ_FLS2_Staging__'].Get()
            update_values(window, [('__BUJ_FLS2_Staging__', 'None')])
            target_key = '__BUJ_FLS2__'
        if os_path.exists(fls_path) and fls_path.endswith('.fls'):
            fls = File_Object(fls_path)
            if 'FLS1' in event:
                winfo.buj_fls_files[0] = fls
                if tfs_value == 'Unflip/Flip' and fls_value == 'One':
                    winfo.buj_fls_files[1] = fls
            elif 'FLS2' in event:
                winfo.buj_fls_files[1] = fls
            metadata_change(window, [(target_key, fls.shortname)])
            toggle(window, [target_key], state='Set')
        else:
            print(f'{prefix}Pathname is incorrect or none selected.')

    # Set the fls files and load in images
    elif event == '__BUJ_Set_FLS__':
        fls_value = window['__LS_FLS_Combo__'].Get()
        tfs_value = 'Unflip/Flip'

        fls_file_names = [winfo.buj_fls_files[0].path, winfo.buj_fls_files[1].path]
        check = check_setup(image_dir, tfs_value, fls_value, fls_file_names, prefix='BUJ: ')
        if check:
            # Set the image dir button
            path1, path2, files1, files2 = check[1:]
            toggle(window, ['__BUJ_Set_Img_Dir__', '__BUJ_Set_FLS__'], state='Set')

            # Prepare reference data
            ref1 = files1[len(files1)//2]
            ref1_path = al_join([path1, ref1], '/')
            uint8_1, flt_data_1, size_1 = g_help.load_image(ref1_path, graph.get_size(), prefix=' LS: ')

            ref2 = files2[len(files2)//2]
            ref2_path = al_join([path2, ref2], '/')
            uint8_2, flt_data_2, size_2 = g_help.load_image(ref2_path, graph.get_size(), prefix=' LS: ')

            # Load image data as numpy arrays for uint8, numerical val, and size
            if uint8_1 and uint8_2:
                # Create image instances and store byte data for TK Canvas
                image1 = Image(uint8_1, flt_data_1, size_1, ref1_path)
                image1.byte_data = g_help.vis_1_im(image1)
                image2 = Image(uint8_2, flt_data_2, size_2, ref2_path)
                image2.byte_data = g_help.vis_1_im(image2)

                # Display ref filename and load display data
                # Update window only if view stack not set
                if view_stack_button.metadata['State'] == 'Def':
                    metadata_change(window, [('__BUJ_Image1__', al_join([orientation, image1.shortname], '/'))])
                    toggle(window, ['__BUJ_Image1__'])
                    display_img = image1.byte_data

                # Push data to winfo
                winfo.buj_images['image1'] = image1
                winfo.buj_images['image2'] = image2
                winfo.buj_files1 = files1
                winfo.buj_files2 = files2
                print(f'{prefix}Directory properly set-up.')
        else:
            print(f'{prefix}Look at Help Tab for correct file setup.')

    # Set the fls files and load in images
    elif event == '__BUJ_Reset_FLS__':
        # Reset FLS but don't reset loaded stack
        winfo.buj_images = {}
        winfo.buj_fls_files = [None, None]
        winfo.buj_files1 = None
        winfo.buj_files2 = None
        winfo.buj_file_queue = []
        winfo.kill_proc.append('BUJ')

        # --- Set up loading files --- #
        winfo.buj_graph_double_click = False
        winfo.buj_mask_coords = []
        winfo.buj_mask_markers = []

        graph.Erase()
        metadata_change(window, ['__BUJ_FLS1__', '__BUJ_FLS2__',
                                 '__BUJ_Image1__', '__BUJ_Image2__'], reset=True)
        toggle(window, ['__BUJ_FLS_Combo__', '__BUJ_Adjust__',
                        '__BUJ_View__', '__BUJ_Image1__', '__BUJ_Image2__',
                        '__BUJ_FLS1__', '__BUJ_FLS2__', '__BUJ_Set_FLS__',
                        '__BUJ_Make_Mask__',
                        '__BUJ_Flip_Stack_Inp__', '__BUJ_Unflip_Stack_Inp__',
                        '__BUJ_Unflip_Mask_Inp__', '__BUJ_Flip_Mask_Inp__',
                        ], state='Def')

        window['__BUJ_FLS1_Text__'].update(value=window['__BUJ_FLS1_Text__'].metadata['Two'])
        window['__BUJ_FLS2_Text__'].update(value=window['__BUJ_FLS2_Text__'].metadata['Two'])
        window['__BUJ_FLS1_Text__'].metadata['State'] = 'Two'
        window['__BUJ_FLS2_Text__'].metadata['State'] = 'Two'
        update_values(window, [('__BUJ_FLS1_Staging__', ''), ('__BUJ_FLS2_Staging__', ''),
                               ('__BUJ_transform_x__', '0'), ('__BUJ_transform_y__', '0'),
                               ('__BUJ_transform_rot__', "0"), ('__BUJ_horizontal_flip__', True),
                               ('__BUJ_unflip_reference__', True)])
        # window['__BUJ_unflip_reference__'].update(True)

        # Re-init reconstruct
        update_slider(window, [('__BUJ_Stack_Slider__', {'value': 0,
                                                         'slider_range': (0, 0)})])
        print(f'{prefix}FLS reset.')

    # Change reference state between flip/unflip images
    elif change_ref:
        if image_dir:
            if orientation == 'unflip':
                image_key = 'image1'
            elif orientation == 'flip':
                image_key = 'image2'
            image = images[image_key]
            display_img = image.byte_data
            metadata_change(window, [('__BUJ_Image1__', al_join([orientation, image.shortname], '/'))])

    # Load image for rotation/translation adjustment
    elif event == '__BUJ_Adjust__':
        if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set':
            # Quit flip adjustment
            if adjust_button.metadata['State'] == 'Set':
                if orientation == 'unflip':
                    image_key = 'image1'
                elif orientation == 'flip':
                    image_key = 'image2'
                display_img = images[image_key].byte_data
                toggle(window, ['__BUJ_Adjust__', '__BUJ_Image2__'], state='Def')

            elif adjust_button.metadata['State'] == 'Def':
                if orientation == 'unflip':
                    orient2 = 'flip'
                    img_1 = images['image2']
                    img_2 = images['image1']
                elif orientation == 'flip':
                    orient2 = 'unflip'
                    img_1 = images['image1']
                    img_2 = images['image2']
                display_img = g_help.overlay_images(img_1, img_2, transform, img_1.x_size, graph.get_size()[0])
                metadata_change(window, [('__BUJ_Image2__', al_join([orient2, img_1.shortname], '/'))])
                toggle(window, ['__BUJ_Adjust__', '__BUJ_Image2__'], state='Set')
        else:
            print(f'{prefix}Unable to adjust, make sure to set your working directory.')

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
                print(f'{prefix}Exited save screen without saving image.')
                save = False

            # Create the file
            if save:
                # Delete file if it supposed to be overwritten
                filename = filename[0]
                if os_path.exists(filename):
                    os_remove(filename)

                # Execute fiji macro
                fls_file_names = [winfo.buj_fls_files[0].path, winfo.buj_fls_files[1].path]
                ijm_macro_script = run_single_ls_align(image_dir, orient, sift_params, filename, fls_file_names)
                cmd = g_help.run_macro(ijm_macro_script, image_dir, winfo.fiji_path)


                # Load file
                if event == '__BUJ_Unflip_Align__':
                    target_key = '__BUJ_Unflip_Stack_Inp__'
                    conflict_keys = ['__BUJ_Unflip_Align__', '__BUJ_Load_Unflip_Stack__',
                                     '__BUJ_Flip_Align__', '__BUJ_Elastic_Align__']
                elif event == '__BUJ_Flip_Align__':
                    target_key = '__BUJ_Flip_Stack_Inp__'
                    conflict_keys = ['__BUJ_Unflip_Align__', '__BUJ_Flip_Align__',
                                     '__BUJ_Load_Flip_Stack__', '__BUJ_Elastic_Align__']
                image_key = f'BUJ_{orient}_stack'
                winfo.fiji_queue.append((filename, event, image_key, target_key, conflict_keys, cmd))
        else:
            print(f'{prefix}A valid directory has not been set.')

    # Load in the stacks from Unflip or Flip
    elif event in ['__BUJ_Unflip_Stage_Load__', '__BUJ_Flip_Stage_Load__']:
        # Load in stacks
        filename = values[event]
        update_values(window, [(event, 'None')])
        files1 = winfo.buj_files1
        files2 = winfo.buj_files2
        num_files = [len(files1), len(files2)]
        if event == '__BUJ_Unflip_Stage_Load__':
            num_files = len(files1)
            target_key = '__BUJ_Unflip_Stack_Inp__'
            conflict_keys = ['__BUJ_Unflip_Align__', '__BUJ_Load_Unflip_Stack__',
                             '__BUJ_Elastic_Align__']
            image_key = 'BUJ_unflip_stack'
        elif event == '__BUJ_Flip_Stage_Load__':
            num_files = len(files2)
            target_key = '__BUJ_Flip_Stack_Inp__'
            conflict_keys = ['__BUJ_Flip_Align__', '__BUJ_Load_Flip_Stack__',
                             '__BUJ_Elastic_Align__']
            image_key = 'BUJ_flip_stack'
        winfo.buj_file_queue.append((filename, event, image_key, target_key, conflict_keys, num_files))

    # View the image stack created from alignment
    elif event == '__BUJ_View__':
        # Look at which stack to view
        stack_choice = values['__BUJ_Image_Choice__'][0]
        if stack_choice == 'Unflip LS':
            stack_key = 'BUJ_unflip_stack'
            disabled = False
        elif stack_choice == 'Flip LS':
            stack_key = 'BUJ_flip_stack'
            disabled = False
        elif stack_choice == 'bUnwarpJ':
            stack_key = 'BUJ_stack'
            disabled = False

        if view_stack_button.metadata['State'] == 'Def':
            if stack_key in images and not disabled:
                stack = images[stack_key]
                slider_val = 0
                slider_range = (0, stack.z_size - 1)
                display_img = stack.byte_data[slider_val]

                if stack_choice in ['Unflip LS', 'bUnwarpJ']:
                    prefix = 'unflip'
                    im_name = winfo.buj_files1[slider_val]
                elif stack_choice == 'Flip LS':
                    prefix = 'flip'
                    im_name = winfo.buj_files2[slider_val]

                # Update window
                metadata_change(window, [('__BUJ_Image1__', f'{prefix}/{im_name}')])
                toggle(window, ['__BUJ_Adjust__'], state='Def')
                toggle(window, ['__BUJ_Image1__', '__BUJ_View__'], state='Set')
                update_slider(window, [('__BUJ_Stack_Slider__', {"value": slider_val, "slider_range": slider_range})])
            else:
                print(f"{prefix}Tried loading unavailable stack, you must perform an alignment.")

        elif view_stack_button.metadata['State'] == 'Set':
            # Update window
            if orientation == 'unflip':
                image_key = 'image1'
            elif orientation == 'flip':
                image_key = 'image2'
            image = images[image_key]
            metadata_change(window, [('__BUJ_Image1__', al_join([orientation, image.shortname], '/'))])
            display_img = image.byte_data
            toggle(window, ['__BUJ_View__'])
        winfo.buj_last_image_choice = stack_choice

    # Change the slider
    elif event == '__BUJ_Stack_Slider__':
        stack_choice = values['__BUJ_Image_Choice__'][0]
        if stack_choice == 'Unflip LS':
            stack_key = 'BUJ_unflip_stack'
        elif stack_choice == 'Flip LS':
            stack_key = 'BUJ_flip_stack'
        elif stack_choice == 'bUnwarpJ':
            stack_key = 'BUJ_stack'

        if stack_key in images:
            stack = images[stack_key]
            slider_val = int(values["__BUJ_Stack_Slider__"])

            # Update window
            choice = values['__BUJ_Image_Choice__'][0]
            if choice == 'Unflip LS':
                prefix = 'unflip'
                im_name = winfo.buj_files1[slider_val]
            elif choice == 'Flip LS':
                prefix = 'flip'
                im_name = winfo.buj_files2[slider_val]
            elif choice == 'bUnwarpJ':
                if slider_val < len(winfo.buj_files1):
                    prefix = 'unflip'
                    im_name = winfo.buj_files1[slider_val]
                elif slider_val >= len(winfo.buj_files1):
                    prefix = 'flip'
                    im_name = winfo.buj_files2[slider_val % len(winfo.buj_files1)]

            display_img = stack.byte_data[slider_val]
            metadata_change(window, [('__BUJ_Image1__', f'{prefix}/{im_name}')])

    # Scroll through stacks in the graph area
    elif scroll:
        stack_choice = values['__BUJ_Image_Choice__'][0]
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

        choice = values['__BUJ_Image_Choice__'][0]
        if choice == 'Unflip LS':
            prefix = 'unflip'
            im_name = winfo.buj_files1[slider_val]
        elif choice == 'Flip LS':
            prefix = 'flip'
            im_name = winfo.buj_files2[slider_val]
        elif choice == 'bUnwarpJ':
            if slider_val < len(winfo.buj_files1):
                prefix = 'unflip'
                im_name = winfo.buj_files1[slider_val]
            elif slider_val >= len(winfo.buj_files1):
                prefix = 'flip'
                im_name = winfo.buj_files2[slider_val % len(winfo.buj_files1)]

        # Update the window
        display_img = stack.byte_data[slider_val]
        update_slider(window, [('__BUJ_Stack_Slider__', {"value": slider_val})])
        metadata_change(window, [('__BUJ_Image1__', f'{prefix}/{im_name}')])

    # Changing view stack combo
    elif change_img:
        stack_choice = values['__BUJ_Image_Choice__'][0]
        if stack_choice == 'Unflip LS':
            stack_key = 'BUJ_unflip_stack'
        elif stack_choice == 'Flip LS':
            stack_key = 'BUJ_flip_stack'
        elif stack_choice == 'bUnwarpJ':
            stack_key = 'BUJ_stack'

        if stack_key in images:
            winfo.buj_last_image_choice = stack_choice
            if window['__BUJ_View__'].metadata['State'] == 'Set':
                stack = images[stack_key]
                slider_val = 0
                slider_range = (0, stack.z_size - 1)
                if stack_choice in ['Unflip LS', 'bUnwarpJ']:
                    prefix = 'unflip'
                    im_name = winfo.buj_files1[slider_val]
                elif stack_choice == 'Flip LS':
                    prefix = 'flip'
                    im_name = winfo.buj_files2[slider_val]

                # Update window
                metadata_change(window, [('__BUJ_Image1__', f'{prefix}/{im_name}')])
                display_img = stack.byte_data[slider_val]
                update_slider(window, [('__BUJ_Stack_Slider__', {"value": slider_val, "slider_range": slider_range})])
        else:
            stack_choice = winfo.buj_last_image_choice
            if stack_choice is not None:
                list_vals = window['__BUJ_Image_Choice__'].GetListValues()
                ind = list_vals.index(stack_choice)
                window['__BUJ_Image_Choice__'].update(set_to_index=ind)
                print(f"{prefix}Stack is not available to view. Must load or create alignment.")

    # Start making bunwarpJ masks
    elif event == '__BUJ_Make_Mask__':
        if window['__BUJ_Set_Img_Dir__'].metadata['State'] == 'Set':
            if make_mask_button.metadata['State'] == 'Def':
                mask_choice = window['__BUJ_Mask_View__'].Get()
                change_visibility(window, [('__BUJ_Reset_Mask__', True),
                                           ('__BUJ_Load_Mask_Col__', False)])
                if mask_choice == 'Unflip' or mask_choice == 'Flip':
                    mask_choice = mask_choice.lower()
                    if mask_choice == 'unflip':
                        image_key = 'image1'
                    elif mask_choice == 'flip':
                        image_key = 'image2'
                    display_img = images[image_key].byte_data
                    shortname = images[image_key].shortname
                elif mask_choice == 'Overlay':
                    if orientation == 'unflip':
                        img_1 = images['image2']
                        img_2 = images['image1']
                    elif orientation == 'flip':
                        img_1 = images['image1']
                        img_2 = images['image2']
                    display_img = g_help.overlay_images(img_1, img_2, transform, img_1.x_size, graph.get_size()[0])
                    shortname = 'overlay'
                toggle(window, ['__BUJ_Make_Mask__'])
                toggle(window, ['__BUJ_Image2__'], state='Def')
                metadata_change(window, [('__BUJ_Image1__', shortname)])
                disable_elements(window, ['__BUJ_transform_x__', '__BUJ_transform_y__',
                                          '__BUJ_transform_rot__', '__BUJ_horizontal_flip__'])
            # Quit mask making make_mask_button
            elif make_mask_button.metadata['State'] == 'Set':
                if orientation == 'unflip':
                    image_key = 'image1'
                elif orientation == 'flip':
                    image_key = 'image2'
                image = images[image_key]
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
                        if flag[0] or flag[1]:
                            if flag[0]:
                                print(f'{prefix}Successfully saved unflip mask!')
                            if flag[1]:
                                print(f'{prefix}Successfully saved flip mask!')
                        else:
                            print(f'{prefix}No masks were saved!')
                    else:
                        print(f'{prefix}Exited without saving files!')
                else:
                    print(f'{prefix}Mask was not finished, make sure to double-click and close mask.')
                g_help.erase_marks(winfo, graph, current_tab, full_erase=True)
        else:
            print(f'{prefix}No flip data to adjust, make sure to set your working directory.')

    # Loading a mask
    elif event == '__BUJ_Mask_Stage_Load__':
        # Choose which masks should be loaded
        choice = window['__BUJ_Mask_View__'].Get()
        path = window['__BUJ_Mask_Stage_Load__'].Get()
        update_values(window, [('__BUJ_Mask_Stage_Load__', 'None')])
        image = Image(None, None, None, path)
        if choice == 'Unflip':
            if 'BUJ_unflip_mask' in images and path == 'None':
                image = images['BUJ_unflip_mask']
            else:
                images['BUJ_unflip_mask'] = image
            metadata_change(window, [('__BUJ_Unflip_Mask_Inp__', image.shortname)])
            toggle(window, ['__BUJ_Unflip_Mask_Inp__'], state="Set")
        elif choice == 'Flip':
            if 'BUJ_flip_mask' in images and path == 'None':
                image = images['BUJ_flip_mask']
            else:
                images['BUJ_flip_mask'] = image
            metadata_change(window, [('__BUJ_Flip_Mask_Inp__', image.shortname)])
            toggle(window, ['__BUJ_Flip_Mask_Inp__'], state="Set")
        else:
            if 'BUJ_unflip_mask' in images and path == 'None':
                image1 = images['BUJ_unflip_mask']
            else:
                images['BUJ_unflip_mask'] = image
                image1 = image
            if 'BUJ_flip_mask' in images and path == 'None':
                image2 = images['BUJ_flip_mask']
            else:
                images['BUJ_flip_mask'] = image
                image2 = image
            metadata_change(window, [('__BUJ_Unflip_Mask_Inp__', image1.shortname),
                                     ('__BUJ_Flip_Mask_Inp__', image2.shortname)])
            toggle(window, ['__BUJ_Unflip_Mask_Inp__', '__BUJ_Flip_Mask_Inp__'], state="Set")
        winfo.buj_images = images

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
            display_img = g_help.overlay_images(img_1, img_2, transform, img_1.x_size, graph.get_size()[0])
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
            print(f"{prefix}Not enough vertices to close mask.")

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
                print(f'{prefix}Exited save screen without saving image.')
                save = False
            if save:
                src1, src2 = filenames[0], filenames[1]
                for src in [src1, src2]:
                    if os_path.exists(src):
                        os_remove(src)
                stackpaths = images['BUJ_flip_stack'].path, images['BUJ_unflip_stack'].path
                fls_file_names = [winfo.buj_fls_files[0].path, winfo.buj_fls_files[1].path]
                macro = run_bUnwarp_align(image_dir, mask_files, orientation, transform, im_size,
                                                stackpaths, sift_FE_params=sift_params,
                                                buj_params=buj_params, savenames=(src1, src2),
                                                fls_files=fls_file_names)
                cmd = g_help.run_macro(macro, image_dir, winfo.fiji_path)

                # Load the stack when ready
                target_key = '__BUJ_Stack__'
                image_key = 'BUJ_stack'
                conflict_keys = ['__BUJ_Unflip_Align__', '__BUJ_Flip_Align__', '__BUJ_Load_Flip_Stack__',
                                 '__BUJ_Load_Unflip_Stack__', '__BUJ_Elastic_Align__']
                winfo.fiji_queue.append((src2, event, image_key, target_key, conflict_keys, cmd))
            else:
                print(f'{prefix}Exited without saving files!')
        else:
            print(f"{prefix}Both unflip and flip stacks are not loaded")

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
            img_1 = images['image2']
            img_2 = images['image1']
        elif orientation == 'flip':
            img_1 = images['image1']
            img_2 = images['image2']
        display_img = g_help.overlay_images(img_1, img_2, transform, img_1.x_size, graph.get_size()[0])
    winfo.buj_past_transform = transform

    # Check to see if any files need loading

    # Reset page
    if event == "__BUJ_Reset_Img_Dir__":
        reset(winfo, window, current_tab)
        winfo.kill_proc.append('BUJ')


    # Show which stacks/images are loaded
    inps = ["__BUJ_Unflip_Stack_Inp__", "__BUJ_Flip_Stack_Inp__", "__BUJ_Stack__"]
            #"__BUJ_Unflip_Mask_Inp__", "__BUJ_Flip_Mask_Inp__"
    indices = []
    for i in range(len(inps)):
        if window[inps[i]].metadata['State'] == 'Set':
            indices.append(i)
    change_list_ind_color(window, current_tab, [('__BUJ_Image_Choice__', indices)])

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
    def special_enable_disable(winfo, window):
        enable_list = []
        active_keys = ['__REC_Image_Dir_Path__', '__REC_Set_Img_Dir__', '__REC_Image_Dir_Browse__',
                       '__REC_FLS_Combo__', '__REC_Load_FLS1__', '__REC_Set_FLS__',
                       '__REC_Load_FLS2__', '__REC_Load_Stack__', '__REC_Image_List__',
                       '__REC_M_Volt__', '__REC_Def_Combo__', '__REC_QC_Input__',
                       "__REC_Reset_FLS__", "__REC_TFS_Combo__",
                       '__REC_Mask_Size__', '__REC_Mask__', "__REC_Erase_Mask__",
                       "__REC_transform_y__", "__REC_transform_x__", "__REC_transform_rot__",
                       '__REC_Data_Prefix__', '__REC_Run_TIE__', '__REC_Save_TIE__',
                       "__REC_Slider__", "__REC_Colorwheel__", "__REC_Derivative__",
                       '__REC_Reset_Img_Dir__']

        if window['__REC_Set_Img_Dir__'].metadata['State'] == 'Set':
            if winfo.ptie_recon_thread is None and winfo.ptie_init_thread is None:
                enable_list.extend(["__REC_Reset_FLS__"])
            if window['__REC_Set_FLS__'].metadata['State'] == 'Def':
                enable_list.extend(['__REC_FLS_Combo__', "__REC_TFS_Combo__", '__REC_M_Volt__'])
                if (window['__REC_FLS_Combo__'].Get() == 'Two' and
                        window['__REC_FLS2__'].metadata['State'] == 'Def'):
                    enable_list.extend(['__REC_Load_FLS2__'])
                if window['__REC_FLS1__'].metadata['State'] == 'Def':
                    enable_list.extend(['__REC_Load_FLS1__'])
                if window['__REC_Stack__'].metadata['State'] == 'Def':
                    enable_list.append('__REC_Load_Stack__')
                if (window['__REC_Stack__'].metadata['State'] == 'Set' and
                        window['__REC_FLS1__'].metadata['State'] == 'Set' and
                        window['__REC_FLS2__'].metadata['State'] == 'Set'):
                    enable_list.extend(['__REC_Set_FLS__'])
            elif window['__REC_Set_FLS__'].metadata['State'] == 'Set':
                enable_list.extend(['__REC_Mask__', '__REC_Erase_Mask__',
                                    '__REC_Data_Prefix__', '__REC_Def_Combo__',
                                    '__REC_QC_Input__', "__REC_Derivative__",
                                    "__REC_Colorwheel__"])
                if window['__REC_Mask__'].metadata['State'] == 'Def':
                    enable_list.extend(['__REC_Run_TIE__'])
                else:
                    enable_list.extend(['__REC_Mask_Size__', "__REC_transform_y__",
                                        "__REC_transform_x__",  "__REC_transform_rot__"])
            if winfo.rec_tie_results is not None:
                enable_list.extend(['__REC_Save_TIE__'])
            if (window['__REC_Stack__'].metadata['State'] == 'Set' and
                    window['__REC_Mask__'].metadata['State'] == 'Def'):
                enable_list.extend(["__REC_Image_List__"])

            if (window['__REC_Image_List__'].get()[0] in ['Stack', 'Default Stack'] or
                window['__REC_Mask__'].metadata['State'] == 'Set'):
                enable_list.extend(["__REC_Slider__"])

        elif window['__REC_Set_Img_Dir__'].metadata['State'] == 'Def':
            enable_list.extend(['__REC_Image_Dir_Path__', '__REC_Set_Img_Dir__',
                                '__REC_Image_Dir_Browse__'])
        if winfo.ptie_recon_thread is None and winfo.ptie_init_thread is None:
            enable_list.extend(['__REC_Reset_Img_Dir__'])

        disable_list = setdiff1d(active_keys, enable_list)
        enable_elements(window, enable_list)
        disable_elements(window, disable_list)

    # Get rotations and shifts to apply to image (only positive rotations)
    transform = get_transformations(winfo, window, current_tab)
    mask_transform = get_mask_transform(winfo, window, current_tab)

    # Grab important elements
    graph = window['__REC_Graph__']
    colorwheel_graph =  window['__REC_Colorwheel_Graph__']
    mask_button = window['__REC_Mask__']

    # Pull in image data from struct object
    image_dir = winfo.rec_image_dir
    images = winfo.rec_images
    colorwheel_choice = window['__REC_Colorwheel__'].Get()[0]

    # if 'TIMEOUT' not in event:
    #     print(event)

    prefix = 'REC: '
    display_img = None
    display_img2 = None

    # Import event handler names (overlaying, etc.)
    adjust = mask_button.metadata['State'] == 'Set' and (winfo.rec_past_transform != transform or
                                                         winfo.rec_past_mask != mask_transform)
    change_img = winfo.rec_last_image_choice != values['__REC_Image_List__'][0]
    change_colorwheel = winfo.rec_last_colorwheel_choice != colorwheel_choice
    scroll = (event in ['MouseWheel:Up', 'MouseWheel:Down']
              and (values['__REC_Image_List__'][0] in ['Stack', 'Default Stack'] or
                   window['__REC_Mask__'].metadata['State'] == 'Set')
              and winfo.true_element == "__REC_Graph__")
    scroll_defocus = (event in event in ['MouseWheel:Up', 'MouseWheel:Down'] and
                      winfo.true_element == '__REC_Def_List__' or
                      event == '__REC_Defocus_Slider__')
    scroll_images = (event in event in ['MouseWheel:Up', 'MouseWheel:Down'] and
                     winfo.true_element == '__REC_Image_List__' or
                     event == '__REC_Image_Slider__')
    draw_mask = mask_button.metadata['State'] == 'Set'

    # Set the working directory
    if event == '__REC_Set_Img_Dir__':
        image_dir = values['__REC_Image_Dir_Path__']
        if os_path.exists(image_dir):
            winfo.rec_image_dir = image_dir
            toggle(window, ['__REC_Set_Img_Dir__'], state='Set')
            print(f'{prefix}The path is set: {image_dir}.')
        else:
            print(f'{prefix}This pathname is incorrect.')

    # Load Stack
    elif event == '__REC_Stack_Staging__':
        stack_path = window['__REC_Stack_Staging__'].Get()
        update_values(window, [('__REC_Stack_Staging__', 'None')])
        if os_path.exists(stack_path):
            image_key = 'REC_Stack'
            graph = window['__REC_Graph__']
            graph_size = graph.get_size()
            uint8_data, flt_data, size = g_help.load_image(stack_path, graph_size, stack=True, prefix=' LS: ')
            if uint8_data:
                stack = Stack(uint8_data, flt_data, size, stack_path)
                slider_range = (0, stack.z_size - 1)
                slider_val = 0
                winfo.rec_images[image_key] = stack
                display_img = g_help.adjust_image(stack.flt_data[slider_val], transform, stack.x_size,
                                                  graph.get_size()[0])
                metadata_change(window, [('__REC_Stack__', stack.shortname)])
                toggle(window, ['__REC_Stack__', '__REC_Image__'], state="Set")
                update_slider(window, [('__REC_Slider__', {"value": slider_val, "slider_range": slider_range})])
                winfo.rec_last_image_choice = 'Stack'
                change_list_ind_color(window, current_tab, [('__REC_Image_List__', [0, 10])])
                if winfo.rec_files1:
                    if winfo.rec_files1 and winfo.rec_files2:
                        if slider_val < len(winfo.rec_files1):
                            prefix = 'unflip'
                            im_name = winfo.rec_files1[slider_val]
                        elif slider_val >= len(winfo.rec_files1):
                            prefix = 'flip'
                            im_name = winfo.rec_files2[slider_val % len(winfo.rec_files1)]
                    else:
                        prefix = ''
                        im_name = winfo.rec_files1[slider_val]
                    metadata_change(window, [('__REC_Image__', f'{prefix}/{im_name}')])
                else:
                    metadata_change(window, [('__REC_Image__', f'Image {slider_val + 1}')])
                print(f'{prefix}The file {stack.shortname} was loaded.')
        else:
            print(f'{prefix}This pathname is incorrect.')

    # Set number of FLS files to use
    elif event == '__REC_FLS_Combo__' or event == '__REC_TFS_Combo__':
        fls_value = window['__REC_FLS_Combo__'].Get()
        tfs_value = window['__REC_TFS_Combo__'].Get()
        winfo.rec_fls_files = [None, None]
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
        tfs_value = window['__REC_TFS_Combo__'].Get()
        fls_value = window['__REC_FLS_Combo__'].Get()
        if 'FLS1' in event:
            fls_path = window['__REC_FLS1_Staging__'].Get()
            update_values(window, [('__REC_FLS1_Staging__', 'None')])
            target_key = '__REC_FLS1__'
        elif 'FLS2' in event:
            fls_path = window['__REC_FLS2_Staging__'].Get()
            update_values(window, [('__REC_FLS2_Staging__', 'None')])
            target_key = '__REC_FLS2__'
        if os_path.exists(fls_path) and fls_path.endswith('.fls'):
            fls = File_Object(fls_path)
            if 'FLS1' in event:
                winfo.rec_fls_files[0] = fls
                if tfs_value == 'Unflip/Flip' and fls_value == 'One':
                    winfo.rec_fls_files[1] = fls
            elif 'FLS2' in event:
                winfo.rec_fls_files[1] = fls
            metadata_change(window, [(target_key, fls.shortname)])
            toggle(window, [target_key], state='Set')
        else:
            print(f'{prefix}Pathname is incorrect or none selected.')

    # Set number of FLS files to use
    elif event == '__REC_Reset_FLS__':
        # Reset FLS but don't reset loaded stack
        if winfo.ptie_init_thread is not None and winfo.ptie_init_thread.is_alive():
            winfo.ptie_init_thread.join()
            winfo.ptie_init_thread = None
        if winfo.ptie_recon_thread is not None and winfo.ptie_recon_thread.is_alive():
            winfo.ptie_recon_thread.join()
            winfo.ptie_recon_thread = None
        winfo.rec_images = {}
        winfo.rec_fls_files = [None, None]
        winfo.rec_ptie = None
        winfo.rec_file_queue = []

        # --- Set up loading files --- #
        winfo.rec_defocus_slider_set = 0
        winfo.rec_image_slider_set = 6

        # Image selection
        winfo.rec_tie_results = None
        winfo.rec_def_val = None

        # Graph and mask making
        winfo.graph_slice = (None, None)
        winfo.rec_graph_double_click = False
        winfo.rec_mask_coords = []
        winfo.rec_mask_markers = []

        graph.Erase()
        colorwheel_graph.Erase()
        metadata_change(window, ['__REC_FLS1__', '__REC_FLS2__', '__REC_Stack__'], reset=True)
        toggle(window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__', '__REC_Stack__',
                        '__REC_FLS1__', '__REC_FLS2__', '__REC_Set_FLS__', '__REC_Image__',
                        '__REC_Mask__'], state='Def')
        window['__REC_Def_Combo__'].update(values=['None'])
        window['__REC_Def_List__'].update(values=['None'])
        window['__REC_FLS1_Text__'].update(value=window['__REC_FLS1_Text__'].metadata['Two'])
        window['__REC_FLS2_Text__'].update(value=window['__REC_FLS2_Text__'].metadata['Two'])
        window['__REC_FLS1_Text__'].metadata['State'] = 'Two'
        window['__REC_FLS2_Text__'].metadata['State'] = 'Two'
        update_values(window, [('__REC_FLS1_Staging__', ''), ('__REC_FLS2_Staging__', ''),
                               ('__REC_Stack_Staging__', ''), ('__REC_Def_Combo__', 'None'),
                               ('__REC_Image__', 'None')])
        change_list_ind_color(window, current_tab, [('__REC_Image_List__', [])])
        # Re-init reconstruct
        update_slider(window, [('__REC_Defocus_Slider__', {'value': winfo.rec_defocus_slider_set,
                                                           'slider_range': (0, 0)}),
                               ('__REC_Slider__', {'value': 0,
                                                   'slider_range': (0, 0)}),
                               ('__REC_Image_Slider__', {'value': winfo.rec_image_slider_set})])
        window['__REC_Image_List__'].update(set_to_index=0, scroll_to_index=0)
        window['__REC_Def_List__'].update(set_to_index=0, scroll_to_index=0)
        print(f'{prefix}FLS and reconstruct data reset.')

    # Set which image you will be working with FLS files
    elif event == '__REC_Set_FLS__':
        # Get PYTIE loading params
        path = image_dir + '/'
        stack_name = images['REC_Stack'].shortname

        # Get FLS value information
        tfs_value = window['__REC_TFS_Combo__'].Get()
        fls_value = window['__REC_FLS_Combo__'].Get()
        if tfs_value == 'Unflip/Flip':
            fls_file_names = [winfo.rec_fls_files[0].path, winfo.rec_fls_files[1].path]
        else:
            fls_file_names = [winfo.rec_fls_files[0].path, None]
        check = check_setup(image_dir, tfs_value, fls_value, fls_file_names, prefix='REC: ')
        if check:
            path1, path2, files1, files2 = check[1:]
            fls_1 = winfo.rec_fls_files[0]
            fls_2 = winfo.rec_fls_files[1]
            fls1_path = fls_1.shortname
            if tfs_value != 'Single':
                fls2_path = fls_2.shortname
            else:
                fls2_path = None

            # Is this single series or flipped/unflipped series
            if window['__REC_TFS_Combo__'].metadata['State'] == 'Def':
                flip = True
            else:
                flip = False

            # Load ptie params
            if ((2*len(files1) == images['REC_Stack'].z_size and tfs_value == 'Unflip/Flip') or
                    (len(files1) == images['REC_Stack'].z_size and tfs_value == 'Single')):

                winfo.ptie_init_thread = Thread(target=ptie_init_thread,
                                                args=(winfo, path, fls1_path, fls2_path, stack_name,
                                                      files1, files2, flip, tfs_value),
                                                daemon=True)
                print('--- Start PTIE Initialization ---')
                winfo.ptie_init_thread.start()
            else:
                print(f'{prefix}The number of expected files does not match the', end=' ')
                print('current stack.')
        else:
            print(f'{prefix}There was an incompatibility between the fls contents and the', end=' ')
            print('files within the directories.')

    # Change the slider
    elif event == '__REC_Slider__':
        stack_choice = window['__REC_Image_List__'].get()[0]
        if stack_choice == 'Stack':
            adjustment = False
        elif stack_choice == 'Default Stack':
            adjustment = True
        stack_key = 'REC_Stack'
        stack = images[stack_key]
        slider_val = int(values["__REC_Slider__"])

        # Update window
        if not adjustment:
            display_img = stack.byte_data[slider_val]
        elif adjustment:
            display_img = g_help.adjust_image(stack.flt_data[slider_val], transform, stack.x_size, graph.get_size()[0])

        if winfo.rec_files1:
            if winfo.rec_files1 and winfo.rec_files2:
                if slider_val < len(winfo.rec_files1):
                    prefix = 'unflip'
                    im_name = winfo.rec_files1[slider_val]
                elif slider_val >= len(winfo.rec_files1):
                    prefix = 'flip'
                    im_name = winfo.rec_files2[slider_val % len(winfo.rec_files1)]
            else:
                prefix = ''
                im_name = winfo.rec_files1[slider_val]
            metadata_change(window, [('__REC_Image__', f'{prefix}/{im_name}')])
        else:
            metadata_change(window, [('__REC_Image__', f'Image {slider_val+1}')])

    # Scroll through stacks in the graph area
    elif scroll:
        stack_choice = window['__REC_Image_List__'].get()[0]
        if stack_choice == 'Stack':
            adjustment = True
        elif stack_choice == 'Default Stack':
            adjustment = False
        if stack_choice in ['Stack', 'Default Stack'] or window['__REC_Mask__'].metadata['State'] == 'Set':
            stack = images['REC_Stack']
            slider_val = int(values["__REC_Slider__"])
            max_slider_val = stack.z_size - 1
            # Scroll up or down
            if event == 'MouseWheel:Down':
                slider_val = min(max_slider_val, slider_val+1)
            elif event == 'MouseWheel:Up':
                slider_val = max(0, slider_val-1)

            # Update the window
            if not adjustment:
                display_img = stack.byte_data[slider_val]
            elif window['__REC_Mask__'].metadata['State'] == 'Set' or adjustment:
                display_img = g_help.adjust_image(stack.flt_data[slider_val], transform,
                                                  stack.x_size, graph.get_size()[0])

            update_slider(window, [('__REC_Slider__', {"value": slider_val})])
            if winfo.rec_files1:
                if winfo.rec_files1 and winfo.rec_files2:
                    if slider_val < len(winfo.rec_files1):
                        prefix = 'unflip'
                        im_name = winfo.rec_files1[slider_val]
                    elif slider_val >= len(winfo.rec_files1):
                        prefix = 'flip'
                        im_name = winfo.rec_files2[slider_val % len(winfo.rec_files1)]
                else:
                    prefix = 'tfs'
                    im_name = winfo.rec_files1[slider_val]
                metadata_change(window, [('__REC_Image__', f'{prefix}/{im_name}')])
            else:
                metadata_change(window, [('__REC_Image__', f'Image {slider_val+1}')])

    # Scroll through image options
    elif scroll_images:
        max_slider_val = 6
        if event in ['MouseWheel:Down', 'MouseWheel:Up']:
            slider_set = winfo.rec_image_slider_set
            if event == 'MouseWheel:Up':
                slider_val = min(max_slider_val, slider_set + 1)
            elif event == 'MouseWheel:Down':
                slider_val = max(0, slider_set - 1)
        elif event == "__REC_Image_Slider__":
            slider_val = int(values["__REC_Image_Slider__"])
        update_slider(window, [('__REC_Image_Slider__', {"value": slider_val})])
        window['__REC_Image_List__'].update(scroll_to_index=max_slider_val-slider_val)
        winfo.rec_image_slider_set = slider_val

    # Scroll through image options
    elif scroll_defocus:
        max_slider_val = max(window['__REC_Def_List__'].metadata['length'] - 3, 0)
        if event in ['MouseWheel:Down', 'MouseWheel:Up']:
            slider_set = winfo.rec_defocus_slider_set
            if event == 'MouseWheel:Down':
                slider_val = min(max_slider_val, slider_set + 1)
            elif event == 'MouseWheel:Up':
                slider_val = max(0, slider_set - 1)
        elif event == "__REC_Defocus_Slider__":
            slider_val = int(values["__REC_Defocus_Slider__"])

        update_slider(window, [('__REC_Defocus_Slider__', {"value": slider_val})])
        window['__REC_Def_List__'].update(scroll_to_index=slider_val)
        winfo.rec_defocus_slider_set = slider_val

    # Changing view stack combo
    elif change_img:
        list_values = window['__REC_Image_List__'].GetListValues()
        last_index = None
        if winfo.rec_last_image_choice is not None:
            last_index = list_values.index(winfo.rec_last_image_choice)
        image_choice = values['__REC_Image_List__'][0]
        if image_choice == 'Stack':
            image_key = 'REC_Stack'
            adjustment = True
        elif image_choice == 'Default Stack':
            image_key = 'REC_Stack'
            adjustment = False
        elif image_choice == 'Color':
            image_key = 'color_b'
            im_name = 'Color'
        elif image_choice == 'MagX':
            image_key = 'bxt'
            im_name = 'X-Comp. of Mag. Induction'
        elif image_choice == 'MagY':
            image_key = 'byt'
            im_name = 'Y-Comp. of Mag. Induction'
        elif image_choice == 'Mag':
            image_key = 'bbt'
            im_name = 'Magnitude of Mag. Induction'
        elif image_choice == 'Mag. Phase':
            image_key = 'phase_m'
            im_name = 'Magnetic Phase Shift (radians)'
        elif image_choice == 'Electr. Phase':
            image_key = 'phase_e'
            im_name = 'Electrostatic Phase Shift (radians)'
        elif image_choice == 'Mag. Deriv.':
            image_key = 'dIdZ_m'
            im_name = 'Intensity Deriv. for Mag. Phase'
        elif image_choice == 'Electr. Deriv.':
            image_key = 'dIdZ_e'
            im_name = 'Intensity Deriv. for Electr. Phase'
        elif image_choice == 'In Focus':
            image_key = 'inf_im'
            im_name = 'In-focus image'
        if values['__REC_TFS_Combo__'][0] == 'Single' and image_key in ['dIdZ_e', 'phase_e']:
            window['__REC_Image_List__'].update(set_to_index=last_index)
            print(f'{prefix}Electric information not available for single TFS.')
        else:
            if image_key in images and image_choice in ['Stack', 'Default Stack'] and images[image_key] is not None:
                stack = images[image_key]
                slider_val = 0
                slider_range = (0, stack.z_size - 1)

                # Update window
                if winfo.rec_files1:
                    if winfo.rec_files1 and winfo.rec_files2:
                        if slider_val < len(winfo.rec_files1):
                            prefix = 'unflip'
                            im_name = winfo.rec_files1[slider_val]
                        elif slider_val >= len(winfo.rec_files1):
                            prefix = 'flip'
                            im_name = winfo.rec_files2[slider_val % len(winfo.rec_files1)]
                    else:
                        prefix = ''
                        im_name = winfo.rec_files1[slider_val]
                    metadata_change(window, [('__REC_Image__', f'{prefix}/{im_name}')])
                else:
                    metadata_change(window, [('__REC_Image__', f'Image {slider_val + 1}')])
                if adjustment:
                    display_img = g_help.adjust_image(stack.flt_data[slider_val], transform, stack.x_size, graph.get_size()[0])
                else:
                    display_img = stack.byte_data[slider_val]
                colorwheel_graph.Erase()
                update_slider(window, [('__REC_Slider__', {"value": slider_val, "slider_range": slider_range}),
                                       ])
            # Other image set
            elif image_key in images and images[image_key] is not None:
                image = images[image_key]
                display_img = image.byte_data
                if image_key == 'color_b':
                    display_img2 = winfo.rec_colorwheel
                else:
                    colorwheel_graph.Erase()
                metadata_change(window, [('__REC_Image__', f'{winfo.rec_def_val} {im_name}')])
            elif last_index is not None:
                window['__REC_Image_List__'].update(set_to_index=last_index)
                print(f"{prefix}Image is not available to view. Check PYTIE is run.")
                if values['__REC_TFS_Combo__'] == 'Single':
                    print(f"{prefix}For a single TFS, electric deriv. and phase are not available.")

        winfo.rec_last_image_choice = image_choice

    # Start making reconstruct subregion
    elif event == '__REC_Mask__':
        stack = images['REC_Stack']
        slider_range = (0, stack.z_size - 1)
        slider_val = int(values["__REC_Slider__"])
        display_img = g_help.adjust_image(stack.flt_data[slider_val], transform, stack.x_size, graph.get_size()[0])
        if winfo.rec_files1:
            if winfo.rec_files1 and winfo.rec_files2:
                if slider_val < len(winfo.rec_files1):
                    prefix = 'unflip'
                    im_name = winfo.rec_files1[slider_val]
                elif slider_val >= len(winfo.rec_files1):
                    prefix = 'flip'
                    im_name = winfo.rec_files2[slider_val % len(winfo.rec_files1)]
            else:
                prefix = 'tfs'
                im_name = winfo.rec_files1[slider_val]
            metadata_change(window, [('__REC_Image__', f'{prefix}/{im_name}')])
        if mask_button.metadata['State'] == 'Def':
            toggle(window, ['__REC_Mask__'], state='Set')
            update_slider(window, [('__REC_Slider__', {"value": slider_val, "slider_range": slider_range})])

            # winfo.rec_mask = float(window['__REC_Mask_Size__'].Get())
            draw_mask = True
            g_help.draw_square_mask(winfo, graph)

        # Quit mask making make_mask_button
        elif mask_button.metadata['State'] == 'Set':
            toggle(window, ['__REC_Mask__'], state='Def')
            draw_mask = False
        colorwheel_graph.Erase()
        window['__REC_Image_List__'].update(set_to_index=0, scroll_to_index=0)
        update_slider(window, [('__REC_Image_Slider__', {"value": 6})])
        winfo.rec_last_image_choice = 'Stack'
        winfo.rec_image_slider_set = 6

    # Clicking on graph and making markers for mask
    elif event in ['__REC_Graph__', '__REC_Graph__+UP'] and mask_button.metadata['State'] == 'Set':

        # Erase any previous marks
        g_help.erase_marks(winfo, graph, current_tab)

        # # Draw new marks
        value = values['__REC_Graph__']
        winfo.rec_mask_center = round(value[0]), round(value[1])
        g_help.draw_square_mask(winfo, graph)
        draw_mask = True

    # Remove all mask coordinates from the graph and mask file
    elif event == '__REC_Erase_Mask__':
        # Erase any previous marks
        g_help.erase_marks(winfo, graph, current_tab, full_erase=True)
        graph_size = graph.get_size()
        draw_mask = True
        adjust = True
        if mask_button.metadata['State'] == 'Def':
            winfo.rec_mask_coords = []
            winfo.rec_mask_markers = []
            draw_mask = False
            adjust = False
        winfo.rec_mask_center = (graph_size[0] / 2, graph_size[1] / 2)
        winfo.rec_mask = (50,)
        mask_transform = (50,)
        transform = (0, 0, 0, None)
        update_values(window, [('__REC_transform_x__', '0'), ('__REC_transform_y__', '0'),
                               ('__REC_transform_rot__', "0"), ('__REC_Mask_Size__', '50')])

    # Run PyTIE
    elif event == '__REC_Run_TIE__':
        winfo.ptie_recon_thread = Thread(target=ptie_recon_thread,
                                         args=(winfo, window, graph, colorwheel_graph, images, current_tab),
                                         daemon=True)
        print('--- Starting Reconstruction ---')
        winfo.ptie_recon_thread.start()

    # Save PyTIE
    elif event == '__REC_Save_TIE__':
        if winfo.rec_tie_results:
            tfs = values['__REC_TFS_Combo__']
            prefix = window['__REC_Data_Prefix__'].get()
            filenames, overwrite_signals, prefix, save_tie, im_dir = run_save_window(winfo, event, image_dir,
                                                                                     orientations=prefix,
                                                                                     defocus=winfo.rec_def_val,
                                                                                     tfs=tfs)
            save = overwrite_signals[0]
            if filenames == 'close' or not filenames or not save or not save_tie:
                print(f'{prefix}Exited without saving files!')
            elif save:
                save_results(winfo.rec_def_val, winfo.rec_tie_results, winfo.rec_ptie,
                             prefix, winfo.rec_sym, winfo.rec_qc, save=save_tie, v=2,
                             directory=im_dir, long_deriv=False)
        else:
            print(f"{prefix}Reconstruction results haven't been generated.")

    # Adjust stack and related variables
    if adjust:
        if winfo.rec_past_transform != transform:
            stack = images['REC_Stack']
            slider_val = int(values["__REC_Slider__"])
            display_img = g_help.adjust_image(stack.flt_data[slider_val], transform, stack.x_size, graph.get_size()[0])
        # if winfo.rec_past_mask != mask_transform or mask_transform=:
        g_help.erase_marks(winfo, graph, current_tab, full_erase=True)
        g_help.draw_square_mask(winfo, graph)
    winfo.rec_past_transform = transform
    winfo.rec_past_mask = mask_transform

    if change_colorwheel:
        if winfo.rec_tie_results:
            colorwheel_type = window['__REC_Colorwheel__'].get()
            rad1, rad2 = colorwheel_graph.get_size()
            if colorwheel_type == 'HSV':
                cwheel_hsv = colorwheel_HSV(rad1, background='black')
                cwheel = colors.hsv_to_rgb(cwheel_hsv)
                hsvwheel = True
            elif colorwheel_type == '4-Fold':
                cwheel = colorwheel_RGB(rad1)
                hsvwheel = False
            uint8_colorwheel, float_colorwheel = g_help.convert_float_unint8(cwheel, (rad1, rad2))
            rgba_colorwheel = g_help.make_rgba(uint8_colorwheel[0])
            winfo.rec_colorwheel = g_help.convert_to_bytes(rgba_colorwheel)
            results = winfo.rec_tie_results

            results['color_b'] = color_im(results['bxt'], results['byt'],
                                   hsvwheel=hsvwheel, background='black')
            float_array = g_help.slice(results['color_b'], winfo.graph_slice)
            uint8_data, float_data = {}, {}
            uint8_data, float_data = g_help.convert_float_unint8(float_array, graph.get_size(),
                                                                 uint8_data, float_data)
            image = Image(uint8_data, float_data, (winfo.graph_slice[0], winfo.graph_slice[1], 1), 'color_b')
            image.byte_data = g_help.vis_1_im(image)
            winfo.rec_images['color_b'] = image
            if window['__REC_Image_List__'].get()[0] == 'Color':
                display_img = image.byte_data
                display_img2 = winfo.rec_colorwheel
    winfo.rec_last_colorwheel_choice = colorwheel_choice

    # Reset page
    if event == "__REC_Reset_Img_Dir__":
        if winfo.ptie_init_thread is not None and winfo.ptie_init_thread.is_alive():
            winfo.ptie_init_thread.join()
            winfo.ptie_init_thread = None
        if winfo.ptie_recon_thread is not None and winfo.ptie_recon_thread.is_alive():
            winfo.ptie_recon_thread.join()
            winfo.ptie_recon_thread = None
        reset(winfo, window, current_tab)

    # Enable any elements if need be
    special_enable_disable(winfo, window)

    # Redraw all
    if display_img:
        redraw_graph(graph, display_img)
    if display_img2:
        redraw_graph(colorwheel_graph, display_img2)
    if draw_mask:
        g_help.draw_mask_points(winfo, graph, current_tab)
    if winfo.rec_mask_coords and mask_button.metadata['State'] == 'Def':
        text = 'Set'
        mask_color = 'green'
        font = 'Times 18 bold'
    else:
        text = 'Unset'
        mask_color = 'black'
        font = 'Times 17'
    window['__REC_Mask_Text__'].update(value=text, text_color=mask_color, font=font)


# -------------- Save Window --------------#
def check_overwrite(save_win, true_paths, orientations, im_type, event, tfs):
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
    if event == '__REC_Save_TIE__':
        overwrite_box = save_win[f'__save_win_overwrite1__'].Get()
    rec_tie_dont_overwrite_state = False
    rec_tie_dont_overwrite_text = 'Some files already exist. Check overwrite box or change name.'
    for i in range(len(true_paths)):
        text = ''
        exists = os_path.exists(true_paths[i])
        # If no orientation, this removes extra space in insertion for log
        if event != '__REC_Save_TIE__':
            overwrite_box = save_win[f'__save_win_overwrite{i+1}__'].Get()
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

        elif event == '__REC_Save_TIE__':
            if tfs != 'Single' or (tfs == 'Single' and 'phase_e' not in true_paths[i] and 'dIdZ_e' not in true_paths[i]):
                index = true_paths[i].rfind('/')
                insertion = true_paths[i][index+1:]
                if exists and not overwrite_box:
                    rec_tie_dont_overwrite_state = True
                    save_enable = False
                    overwrite_signals = [False]
                elif exists and overwrite_box:
                    text = f'''The {insertion} file will be overwritten.'''
                    overwrite_signals = [True]
                elif not exists:
                    text = f'The {insertion} file will be saved.'
                    overwrite_signals = [True]


        # Update save window
        current_log_text = save_win['__save_win_log__'].Get()
        new_log_text = current_log_text + text
        update_values(save_win, [('__save_win_log__', new_log_text.strip())])

    if rec_tie_dont_overwrite_state:
        update_values(save_win, [('__save_win_log__', rec_tie_dont_overwrite_text.strip())])
    if save_enable:
        enable_elements(save_win, ['__save_win_save__'])
    else:
        disable_elements(save_win, ['__save_win_save__'])
    return overwrite_signals


def save_window_values(save_win, num_paths, event, orientations, defocus=None):
    """Sets ups the save window layout.

    Parameters
    ----------
    save_win : PySimpleGUI Window Element
        The representation of the save window.
    num_paths : int
        The number of paths, to create the number
        of overwrite checkboxes and true_path
        input elements.
    event : str
        The save event from the main GUI window.


    Returns
    -------
    true_paths : list of str
        The list containing the full path names.
    """
    # Comb through all input fields and pull current path name
    true_paths = []
    if event != '__REC_Save_TIE__':
        for i in range(1, num_paths + 1):
            true_paths.append(save_win[f'__save_win_filename{i}__'].Get())
    elif event == '__REC_Save_TIE__':
        save_choice = save_win['__save_rec_combo__'].Get()
        working_directory = save_win[f'__save_win_wd__'].Get()
        image_save_directory = save_win[f'__save_win_filename1__'].Get()
        prefix = save_win[f'__save_win_prefix__'].Get()
        path = al_join([working_directory, image_save_directory, prefix], "/")
        if save_choice == 'Color':
            stop = 2
        elif save_choice == 'Full Save':
            stop = 10
        elif save_choice == 'Mag. & Color':
            stop = 4
        elif save_choice == 'No Save':
            stop = 0
        for i in range(stop):
            true_paths.append(al_join([path, str(defocus), orientations[i]], '_'))
    return true_paths


def run_save_window(winfo, event, image_dir, orientations=None, defocus=None, tfs='Unflip/Flip'):
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
    window_layout, im_type, file_paths, orientations = save_window_ly(event, image_dir, orientations, tfs=tfs)
    save_win = sg.Window('Save Window', window_layout, finalize=True) #icon=get_icon(),
    winfo.save_win = save_win
    winfo.window.Hide()
    true_paths = save_window_values(save_win, len(file_paths), event, orientations, defocus)
    # Run event handler
    overwrite_signals = []
    ev2 = 'Initiate'
    while True:
        if ev2 != 'Initiate':
            ev2, vals2 = save_win.Read(timeout=400)
        if event == '__REC_Save_TIE__':
            prefix = save_win[f'__save_win_prefix__'].Get()
            index = save_win['__save_win_filename1__'].Get().rfind('/')
            im_dir = save_win['__save_win_filename1__'].Get()[index + 1:]
            save_choice = save_win['__save_rec_combo__'].Get()
            if save_choice == 'Color':
                save_tie = 'color'
            elif save_choice == 'Full Save':
                save_tie = True
            elif save_choice == 'Mag. & Color':
                save_tie = 'b'
            elif save_choice == 'No Save':
                save_tie = False

        filenames = []
        if ev2 and ev2 != 'Exit':
            true_paths = save_window_values(save_win, len(file_paths), event, orientations, defocus)

        if ev2 and 'TIMEOUT' not in ev2:
            overwrite_signals = check_overwrite(save_win, true_paths, orientations, im_type, event, tfs)

        # Exit or save pressed
        if not ev2 or ev2 in ['Exit', '__save_win_save__']:
            winfo.window.UnHide()
            save_win.Close()
            if ev2 == '__save_win_save__':
                for path in true_paths:
                    filenames.append(path)
            break
        if ev2 == 'Initiate':
            ev2 = None

    if event != '__REC_Save_TIE__':
        return filenames, overwrite_signals
    elif event == '__REC_Save_TIE__':
        return filenames, overwrite_signals, prefix, save_tie, im_dir


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
    # Create output_window
    output_window = output_ly()
    output_window.Hide()

   # Initialize window, bindings, and event variables
    init(winfo, window, output_window)

    # fiji_log = '../GUI/fiji_log.txt'
    # window['home_graph'].DrawImage(filename='../GUI/PyLo_Icon-1.png', location=(0, 299))

    # Set up the output log window and redirecting std_out
    log_line = 0
    with StringIO() as winfo.buf, StringIO() as winfo.fiji_buf:
        with redirect_stdout(winfo.buf):

            # Run event loop
            bound_click = True
            close = None
            while True:
                # Capture events
                event, values = window.Read(timeout=200)
                if event == 'Show::Log' and not winfo.output_window_active:
                    winfo.output_window_active = True
                    output_window.Reappear()
                    output_window.UnHide()
                elif event == 'Show::Log' and winfo.output_window_active:
                    output_window.BringToFront()
                elif event == 'Hide::Log' and winfo.output_window_active:
                    winfo.output_window_active = False
                    output_window.Hide()
                    output_window.Disappear()

                # Break out of event loop
                if event is None or close == 'close':  # always,  always give a way out!
                    winfo.kill_proc = ['LS', 'BUJ']
                    load_file_queue(winfo, window)
                    if winfo.ptie_init_thread is not None:
                        winfo.ptie_init_thread = None
                        if winfo.ptie_init_thread.is_alive():
                            winfo.ptie_init_thread.join()
                        print('--- Stopped Loading PTIE Params ---')
                    if winfo.ptie_recon_thread is not None:
                        if winfo.ptie_recon_thread.is_alive():
                            winfo.ptie_recon_thread.join()
                        winfo.ptie_recon_thread = None
                        print('--- Exited Reconstruction ---')
                    window.close()
                    output_window.close()
                    break

                # if event != '__TIMEOUT__' and 'HOVER' not in event:
                #     print(event)

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

                # Make sure input element that just display names can't be typed in
                if event in keys['read_only_inputs']:
                    state = window[event].metadata['State']
                    text = window[event].metadata[state]
                    window[event].update(value=text)

                # Check which tab is open and execute events regarding that tab
                current_tab = winfo.current_tab = get_open_tab(winfo, winfo.pages, event)
                if current_tab == "home_tab":
                    run_home_tab(winfo, window, event, values)
                elif current_tab == "ls_tab":
                    run_ls_tab(winfo, window, current_tab, event, values)
                elif current_tab == "bunwarpj_tab":
                    run_bunwarpj_tab(winfo, window, current_tab, event, values)
                elif current_tab == "reconstruct_tab":
                    run_reconstruct_tab(winfo, window, current_tab, event, values)

                # Load files and run sub-processes
                load_file_queue(winfo, window)
                # Redirecting fiji stdout

                # Set the focus of the GUI to reduce interferences
                set_pretty_focus(winfo, window, event)

                # Change color of any loaded images/files
                for key in keys['read_only_inputs']:
                    if window[key].metadata['State'] == 'Set':
                        window[key].update(text_color='green')
                    elif window[key].metadata['State'] == 'Def':
                        window[key].update(text_color='black')


                sys.stdout.flush()
                output_txt = winfo.buf.getvalue().split('\n')
                # output_window['MAIN_OUTPUT'].update(value=f'[{len(output_txt)}, {log_line}]\n', append=True)
                i = 0
                for line in output_txt:
                    if log_line <= i:
                        log_line = i
                        if not(line.isspace() or line.strip() == ''):
                            if (not (line.startswith('REC') or line.startswith('LS') or
                                      line.startswith('BUJ') or line.startswith('HOM') or
                                      line.startswith('***')) and line != '\n'):
                                line = 'REC: ' + line
                            output_window['MAIN_OUTPUT'].update(value=f'{line}\n', append=True)
                    i += 1




def run_GUI(DEFAULTS):
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

    # Create the layouts
    style = WindowStyle()
    window = window_ly(style, DEFAULTS)

    if platform() == 'Darwin':
        subprocess.call(["/usr/bin/osascript", "-e", 'tell app "Finder" to set frontmost of process "Finder" to true'])
        subprocess.call(["/usr/bin/osascript", "-e", 'tell app "Finder" to set frontmost of process "python" to true'])

    # Create data structure to hold variables about GUI, alignment and reconstruction.
    winfo = Struct()

    # Event handling
    event_handler(winfo, window)


run_GUI(DEFAULTS)
