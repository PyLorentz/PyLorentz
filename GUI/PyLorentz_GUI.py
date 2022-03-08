"""Functions for GUI event handling.

Contains most functions for handling events in GUI. This includes controlling user events,
calls to PyTIE, alignment calls to FIJI, saving images, creating image masks,
and image manipulation for GUI display.

AUTHOR:
Timothy Cote, ANL, Fall 2019.
"""

# Standard library imports
import collections
from contextlib import redirect_stdout
from io import StringIO
from os import path as os_path, remove as os_remove, mkdir
from platform import system as platform
from queue import Queue, Empty
import subprocess
import shlex
from sys import path as sys_path, stdout as sys_stdout
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union
import webbrowser
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Third-party imports
from numpy import setdiff1d
import PySimpleGUI as sg
import matplotlib
matplotlib.use('agg')
from matplotlib import colors

# Local imports
sys_path.append("../PyTIE/")

from gui_layout import window_ly, file_choice_ly, save_window_ly, output_ly, element_keys
from gui_styling import WindowStyle, get_icon
from colorwheel import colorwheel_HSV, colorwheel_RGB, color_im
from microscopes import Microscope
from TIE_helper import *
from TIE_reconstruct import TIE, SITIE, save_results
from tkinter import INSERT as tk_insert
import util
from util import Struct, check_setup
# import faulthandler
# faulthandler.enable()


# ============================================================= #
#      Setting defaults for FIJI and the working Directory.      #
# ============================================================= #
def defaults() -> Dict[str, str]:
    """Load the default Fiji and working directory if any is set.

    Returns:
        DEFAULTS: Dictionary of the working directory paths.
    """
    GUI_dir = os.path.dirname(os.path.abspath(__file__))
    default_txt = f'{GUI_dir}/defaults.txt'
    DEFAULTS = {'browser_dir': '', 'fiji_dir': ''}
    if not os_path.exists(default_txt):
        with open(default_txt, 'w+') as f:
            f.write('// File contains the default paths to FIJI (ignore) and the browser working directory for GUI.\n')
            f.write('FIJI Directory,\n')
            f.write('Browser Directory,\n')
    else:
        try:
            new_mode = os.stat(my_file).st_mode | 0o777
            os.chmod(default_txt, new_mode)
        except:
            pass

        with open(default_txt, 'r') as f:
            for line in f.readlines():
                if not line.startswith('//'):
                    items = line.split(',')
                    key, value = items[0], items[1]
                    value = value.strip()
                    if key == 'FIJI Directory':
                        DEFAULTS['fiji_dir'] = value
                    elif key == 'Browser Directory':
                        DEFAULTS['browser_dir'] = value

    return DEFAULTS


# ============================================================= #
# ========== Window Functionality and Event Handling ========== #
# ============================================================= #
# ------------- Initialize and reset ------------- #
def init_rec(winfo: Struct, window: sg.Window, mask_reset: bool = True,
             arrow_reset: bool = True) -> None:
    """Initialize Reconstruction Tab variables.

    Initializes winfo arguments for the reconstruction tab this includes:
        - The image working directory
        - Loaded image/stack dictionary
        - The last enabled/disabled element list to see if enabling/disabling needs to change,
            this helps to prevent constantly enabling/disabling on every window call.
        - The list of linear stack files
            * ls_files1 is for unflip or tfs folder
            * ls_files2 is for flip folder (when applicable)
        - The fls files
        - Tracking setting for the PYTIE parameters and object instances
        - Trackers for the threads that are run or currently running for PYTIE
            initialization and reconstruction
        - The last selected stack/image choice for viewing.
        - Trackers for the stack and image choice sliders
        - Dictionary of image choices available to view
        - The transformation adjustments to apply to the non-reference image
        - Trackers for the vector maps on the magnetic images
        - Timers to update the transformation adjustment
        - Arguments for tracking mask/ROI making and the corners/location of the mask/ROI.

    Args:
        winfo: A data structure that holds a information about
            window and GUI.
        window: The main element that represents the GUI window.
        mask_reset: Whether to reset the region select parameters.
        arrow_reset: Whether to reset the arrow parameters.

    Returns:
        None
    """

    # Declare image path and image storage
    winfo.rec_image_dir = ''
    winfo.rec_images = {}
    winfo.last_rec_disable, winfo.last_rec_enable = None, None
    winfo.rec_fls_files = [None, None]
    winfo.rec_files1 = None
    winfo.rec_files2 = None

    # PTIE parameters and elements
    winfo.rec_ptie = None
    winfo.rec_sym = None
    winfo.rec_qc = None
    winfo.rec_microscope = None
    winfo.rec_colorwheel = None
    winfo.rec_past_recon_thread = None

    # --- Set up loading files --- #
    winfo.rec_defocus_slider_set = 0
    winfo.rec_image_slider_set = 7
    winfo.rec_image_slider_dict = {'Stack': 0, 'Color': 1, 'Vector Im.': 2,
                                   'MagX': 3, 'MagY': 4, 'Mag': 5,
                                   'Electr. Phase': 6, 'Mag. Phase': 7,
                                   'Electr. Deriv.': 8, 'Mag. Deriv.': 9,
                                   'In Focus': 10, 'Loaded Stack': 11}

    # Declare transformation timers and related variables
    if arrow_reset:
        winfo.rec_past_arrow_transform = (15, 1, 1, 'On')

    # Image selection
    winfo.rec_last_image_choice = None
    winfo.rec_last_colorwheel_choice = None
    winfo.rec_tie_results = None
    winfo.rec_def_val = None

    # Transformations
    # Graph and mask making
    graph_size = window['__REC_Graph__'].metadata['size']
    winfo.rec_mask_center = ((graph_size[0]) / 2, (graph_size[1]) / 2)

    # Declare rectangular selection of reconstruction region
    winfo.rec_rotxy_timers = (0, 0, 0)
    winfo.rec_corner1 = None
    winfo.rec_corner2 = None
    winfo.new_selection = True

    winfo.graph_slice = (None, None)
    winfo.rec_graph_double_click = False
    winfo.rec_pad_info = (None, 0, 0, 0)
    winfo.rec_mask_coords = []
    winfo.rec_mask_markers = []

    if mask_reset:
        winfo.rec_transform = (0, 0, 0, None)
        winfo.rec_past_transform = (0, 0, 0, None)
        winfo.rec_mask_timer = (0,)
        winfo.rec_mask = (50,)
        winfo.rec_past_mask = (50,)


def init(winfo: Struct, window: sg.Window, output_window: sg.Window) -> None:
    """The main element and window initialization. Creates all initial bindings.

    Initializes winfo arguments for the main GUI, includeing:
        - The window element
        - The output log window element
        - Arguments for tracking the active window
        - The keys available for elements in the window
        - Tracks which tab is open
        - Holds the buffer for printing output to log
        - Managers for reconstruction threads and FIJI threads (and all processes)
        - Tracks what paths are stored for the users defaults
        - Managers for which loading icons should be displayed
        - Tracks which element should have focus in the window

    Args:
        winfo: A data structure that holds a information about
            window and GUI.
        window: The main element that represents the GUI window.
        output_window: The main element that represents the Log
            output Window.

    Returns:
        None
    """
    # --- Set up window and tabs --- #
    winfo.window = window
    winfo.output_window = output_window
    winfo.window_active = True
    winfo.output_window_active = False
    winfo.output_focus_active = False
    winfo.active_output_focus_el = None
    winfo.last_browser_color = None
    keys = element_keys()
    winfo.keys = keys
    winfo.invis_graph = window.FindElement("__invisible_graph__")
    winfo.output_invis_graph = output_window.FindElement("__output_invis_graph__")
    winfo.tabnames = ["Home", "Phase Reconstruction"]
    winfo.pages = "pages_tabgroup"
    winfo.current_tab = "home_tab"
    winfo.buf = None
    winfo.ptie_init_thread = None
    winfo.ptie_recon_thread = None
    winfo.rec_tie_prefix = 'Example'

    winfo.ptie_init_spinner_active = False
    winfo.ptie_recon_spinner_active = False

    # --- Set up bUnwarpJ tab --- #
    init_rec(winfo, window)

    # --- Set up event handling and bindings --- #
    winfo.true_element = None
    winfo.true_input_element = None
    winfo.cursor_pos = 'end'
    winfo.window.bind("<Button-1>", 'Window Click')
    winfo.output_window.bind("<Button-1>", 'Log Click')

    winfo.window['__REC_Graph__'].bind('<Double-Button-1>', 'Double Click')
    winfo.window.bind("<Control-l>", 'Show Log')
    winfo.window.bind("<Control-h>", 'Hide Log')
    winfo.output_window.bind("<Control-h>", 'Output Hide Log')

    big_list = keys['input'] + keys['radio'] + keys['graph'] + keys['combo'] + \
               keys['checkbox'] + keys['slider'] + keys['button'] + keys['listbox']
    for key in big_list:
        winfo.window[key].bind("<Enter>", '+HOVER+')
        winfo.window[key].bind("<Leave>", '+STOP_HOVER+')
    for key in ['MAIN_OUTPUT']:
        winfo.output_window[key].bind("<FocusIn>", '+FOCUS_IN+')
        winfo.output_window[key].bind("<FocusOut>", '+FOCUS_OUT+')


def reset(winfo: Struct, window: sg.Window, current_tab: str) -> None:
    """Reset the current tab elements to default values.

    Args:
        winfo: A data structure that holds the information about the
            window and GUI.
        window: The main representation of the GUI window.
        current_tab:  The key of the current tab being viewed in the window.

    Returns:
        None
    """

    # Reset tabs
    if current_tab == 'reconstruct_tab':
        graph = window['__REC_Graph__']
        colorwheel_graph = window['__REC_Colorwheel_Graph__']
        graph.Erase()
        colorwheel_graph.Erase()
        metadata_change(winfo, window, ['__REC_Stack__', '__REC_Image__', '__REC_FLS1__', '__REC_FLS2__'], reset=True)
        toggle(winfo, window, ['__REC_Set_Img_Dir__', '__REC_FLS_Combo__', '__REC_TFS_Combo__',
                               '__REC_Stack__', '__REC_FLS1__', '__REC_FLS2__', '__REC_Set_FLS__',
                               '__REC_Mask__', '__REC_Image__'], state='Def')
        window['__REC_Def_Combo__'].update(values=['None'])
        window['__REC_Def_List__'].update(values=['None'])
        window['__REC_FLS1_Text__'].update(value=window['__REC_FLS1_Text__'].metadata['Two'])
        window['__REC_FLS2_Text__'].update(value=window['__REC_FLS2_Text__'].metadata['Two'])
        window['__REC_FLS1_Text__'].metadata['State'] = 'Two'
        window['__REC_FLS2_Text__'].metadata['State'] = 'Two'
        change_list_ind_color(window, current_tab, [('__REC_Image_List__', [])])
        change_inp_readonly_bg_color(window, ['__REC_Stack__', '__REC_FLS1__',
                                              '__REC_FLS2__',
                                              '__REC_QC_Input__', '__REC_Arrow_Num__',
                                              '__REC_Arrow_Len__', '__REC_Arrow_Wid__'], 'Readonly')
        update_values(winfo, window, [('__REC_Image_Dir_Path__', ""), ('__REC_Image__', 'None'),
                                      ('__REC_Stack_Stage__', ''), ('__REC_FLS1_Staging__', ''),
                                      ('__REC_FLS2_Staging__', ''),
                                      ('__REC_Def_Combo__', 'None')])
        init_rec(winfo, window, mask_reset=False, arrow_reset=False)
        update_slider(winfo, window, [('__REC_Defocus_Slider__', {'value': winfo.rec_defocus_slider_set,
                                                                  'slider_range': (0, 0)}),
                                      ('__REC_Slider__', {'value': 0, 'slider_range': (0, 0)}),
                                      ('__REC_Image_Slider__', {'value': winfo.rec_image_slider_set})])
        window['__REC_Image_List__'].update(set_to_index=0, scroll_to_index=0)
        window['__REC_Def_List__'].update(set_to_index=0, scroll_to_index=0)


# ------------- Window Helper Functions ------------- #
def shorten_name(path: str, ind: int = 1) -> str:
    """Creates a string of the path name with only the direct parent
    "image_dir" and the child of "image_dir".

    Args:
        path: The full_path to be shortened.
        ind: The index for checking how many '/' to check in path.

    Returns:
        shortname: The shortened pathname for window display.
    """

    check_string = path
    for i in range(ind):
        index = check_string.rfind('/') - 1
        check_string = check_string[:index]
    shortname = path[index+2:]
    return shortname


def get_open_tab(winfo: Struct, tabgroup: str, event: str) -> str:
    """Recursively determine which tab is open.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        tabgroup: The key of the tabgroup.
        event: The event key.

    Returns:
        tab_key: The key for the current open tab.
    """

    # Recursively go through tabgroups and tabs to find
    # current tab.
    tab_key = winfo.window[tabgroup].Get()
    tab = winfo.window[tab_key]
    tab_dict = tab.metadata
    child_tabgroup = None
    if tab_dict is not None:
        child_tabgroup = tab_dict["child_tabgroup"]
    if child_tabgroup is not None:
        tab_key = get_open_tab(winfo, child_tabgroup, event)
        return tab_key
    if 'tab' in event:
        if tab_key == "home_tab":
            tab = 'Home'
        elif tab_key == "reconstruct_tab":
            tab = 'Reconstruct'
        if event not in ['output_tabgroup', 'param_tabgroup']:
            print(f'*** Current Tab: {tab} ***')
    return tab_key


def get_orientation(winfo: Struct, window: sg.Window, pref: str) -> str:
    """Get the current orientation value for the
    current window.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        pref: The prefix for the the key of the orientation for the window.

    Returns:
        orientation: The orientation the current image should be.
    """

    if window[f'__{pref}_unflip_reference__'].Get():
        orientation = 'unflip'
    elif window[f'__{pref}_flip_reference__'].Get():
        orientation = 'flip'
    else:
        orientation = 'tfs'
        if not os_path.exists(util.join([image_dir, orientation], '/')):
            orientation = 'unflip'
    return orientation


def skip_save(filenames: List[str], image_dir: str) -> bool:
    """Returns the flag on whether to skip saving a file.

    If the path is not the intended path where the file will be saved, it will
    skip saving the file. This makes it so the files are stored where they are intended
    for use in reconstruction. The user may move them after they are saved at their
    own precaution.

    Args:
        filenames: The list of filenames to save.
        image_dir: The path of the current working directory for images.

    Returns:
        The flag for whether to save the files.
    """

    skip_save_flag = False
    for filename in filenames:
        string = filename
        index = 0
        while index != -1:
            last_index = index
            index = string.rfind('/')
            string = string[index + 1:]
        folder = filename[:last_index]
        if folder != image_dir:
            skip_save_flag = True
    return skip_save_flag


def get_arrow_transform(window: sg.Window) -> Tuple[str, str, str, str]:
    """Get the mask transformation of the REC window.

        Args:
            window: The element representing the main GUI window.

        Returns:
            new_transform: The transformation to apply to REC mask, a list of strs.
        """

    new_transform = [window['__REC_Arrow_Num__'].Get(),
                     window['__REC_Arrow_Len__'].Get(),
                     window['__REC_Arrow_Wid__'].Get(),
                     window['__REC_Arrow_Color__'].Get()]
    return new_transform


def get_mask_transform(winfo: Struct, window: sg.Window,
                       current_tab: str) -> Tuple[Union[float, int], Union[float, int], Union[float, int]]:
    """Get the mask transformation of the REC window.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        current_tab: The current tab of the window.

    Returns:
        transform: The transformation to apply to REC mask
    """

    timers = winfo.rec_mask_timer
    old_transform = winfo.rec_past_mask
    new_transform = [window['__REC_Mask_Size__'].Get()]
    transf_list = [(new_transform[0], timers[0], 0)]
    transform = retrieve_transform(winfo, window, current_tab, transf_list,
                                   old_transform, new_transform, mask=True)
    return transform


def retrieve_transform(winfo: Struct, window: sg.Window, current_tab: str,
                       transf_list: List[Tuple], old_transform: Tuple,
                       new_transform: Tuple, mask: bool = False) -> Tuple:
    """Return transformation to apply to image based off correct
    inputs and timers.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        current_tab: The key representing the current main tab of the
            window.
        transf_list: The list containing the tuple that has values and
            timers for each of rotation, x, and y inputs.
        old_transform: The previous transformation that was applied to img.
        new_transform: The next transformation to potentially apply to
            img.
        mask: The boolean value whether the transformation is for a mask or not.

    Returns:
        transform: The transformation to apply to the img.
    """

    # Set transform to old_transform in case no changes made
    transform = old_transform

    # Cycle through each transformation: rotation, x-shift, y-shift
    timer_triggered, val_triggered = False, False
    if mask:
        timers = [0]
    else:
        timers = [0, 0, 0]
    timer_cutoff = 45  # timeout

    # Loop through rotation, x, and y values (or size values if mask)
    val_set = False
    for val, timer, i in transf_list:
        # If not float, "", or "-", don't increase timer
        if not util.represents_float(val) and not mask:
            val_triggered = True
            timer += 1
            if val not in ["", "-", '.', "-."]:
                val = '0'
                val_set = True
                timer = timer_cutoff
        # Don't increase timer for mask size if '' or "."
        elif not util.represents_float(val) and mask:
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
            elif float(val) < 1 and mask:
                val = '1'
                val_set = True
                timer = timer_cutoff
            else:
                timer = 0

        # Timer triggered
        if timer == timer_cutoff:
            timer_triggered = True
            timer = 0
            if not val_set and not mask:
                val = '0'
            elif not val_set and mask:
                val = '50'

        timers[i], new_transform[i] = timer, val

    # Update timers
    if not mask:
        if current_tab == 'reconstruct_tab':
            winfo.rec_rotxy_timers = tuple(timers)
    else:
        winfo.rec_mask_timer = tuple(timers)

    # Check if triggers are set
    if (timer_triggered or not val_triggered) and not mask:
        transform = update_rotxy(winfo, window, current_tab, tuple(new_transform))
    elif (timer_triggered or not val_triggered) and mask:
        transform = update_mask_size(winfo, window, tuple(new_transform))
    return transform


def get_transformations(winfo: Struct, window: sg.Window,
                        current_tab: str) -> Tuple:
    """ Gets transformations from the event window.
    Timers give user a limited amount of time before
    the rotation or shift is cleared.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        current_tab: The key of the current tab being viewed in the window.

    Returns:
        transform: A tuple of the transformation variables for
            rotation, x-translate, y-translate, and flip.
    """

    # Grab the transformation for the open tab
    if current_tab == 'reconstruct_tab':
        timers = winfo.rec_rotxy_timers
        pref = "REC"
        old_transform = winfo.rec_past_transform

    # Get the current values of the potential transformation
    hflip = None
    new_transform = [window[f'__{pref}_transform_rot__'].Get(),
                     window[f'__{pref}_transform_x__'].Get(),
                     window[f'__{pref}_transform_y__'].Get(),
                     hflip]

    # Create list of transform input values and timers to cycle through and change
    rotxy_list = [(new_transform[i], timers[i], i) for i in range(len(timers))]
    transform = retrieve_transform(winfo, window, current_tab, rotxy_list,
                                   old_transform, new_transform)
    return transform


def file_loading(winfo: Struct, window: sg.Window, filename: str, active_key: str,
                 image_key: str, target_key: str, conflict_keys: List[str],
                 num_files: int, disable_elem_list: List[str]) -> Tuple[bool, List[str]]:
    """
    The function for loading stacks and other image files.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        filename: The name of the file being loaded.
        active_key: The key of the element that is active process.
        image_key: The key of the image in the image dictionary file should be
            loaded to.
        target_key: The target key that should be updated when image is loaded.
        conflict_keys: All the keys that should be disabled if they are in
            conflict with the active key.
        num_files: The number of files that should be loaded, 1 if not stack.
        disable_elem_list: The list of elements to disable based off active key
            or next active keys.

    Returns:
        remove: Boolean value if active key should be removed
        disable_elem_list: The list of elements to disable based off active key
        or next active keys.
    """

    remove = False
    # Path exists
    if active_key.startswith('__REC'):
        prefix = 'REC: '
    if os_path.exists(filename):
        with warnings.catch_warnings():
            try:
                # Is file loading correctly?
                warnings.filterwarnings('error')
                # Load images and convert to uint8 using numpy and hyperspy,
                if active_key.startswith('__REC'):
                    graph_size_key, reset_key, fls_reset_key = ('__REC_Graph__', '__REC_Set_Img_Dir__', False)
                graph_size = window[graph_size_key].get_size()
                uint8_data, flt_data, size = util.load_image(filename, graph_size, active_key,
                                                               stack=True, prefix=prefix)
                reset = (window[reset_key].metadata['State'] == 'Def' or
                         (not fls_reset_key or
                          window[fls_reset_key].metadata['State'] == 'Def'))

                # Check if data was successfully converted to uint8
                # Save the stack in the correct image dictionary
                if (uint8_data and (num_files is None or num_files == len(uint8_data.keys()))
                        and not reset):
                    stack = util.Stack(uint8_data, flt_data, size, filename)
                    if active_key.startswith('__REC'):
                        winfo.rec_images[image_key] = stack
                    metadata_change(winfo, window, [(target_key, stack.shortname)])
                    toggle(winfo, window, [target_key], state="Set")
                    # Show which stacks/images are loaded
                    print(f'{prefix}The file {stack.shortname} was loaded.')
                else:
                    # Incorrect file loaded, don't keep iterating through it
                    print(f'{prefix}An incorrect file was loaded. Either there was a file type error', end=' ')
                    print('or, if a stack, the number of files may not equal that expected from the FLS.')
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
        if len(filename) != 0:
            print(f'{prefix}There is no valid image name here.')
        remove = True
    return remove, disable_elem_list


def readlines(process: 'subprocess.Popen', queue:  'queue.Queue') -> None:
    """Reads output that is passed to the queue from the running process.

    Args:
        process: The running process.
        queue: The queue of the output stream.

    Returns:
        None
    """

    while process.poll() is None:
        queue.put(process.stdout.readline())


# ------------- Changing Element Values ------------- #
def update_values(winfo: Struct, window: sg.Window,
                  elem_val_list: List[Tuple[sg.Element, Any]]) -> None:
    """ Take a list of element key, value tuple pairs
    and update value of the element.

    Args:
        winfo: The data structure holding all information about
                windows and GUI.
        window: The element representing the main GUI window.
        elem_val_list: The list of elements, value paris to update.

    Returns:
        None
    """

    for elem_key, value in elem_val_list:
        if elem_key in winfo.keys['button']:
            window[elem_key].Update(text=value)
        elif elem_key in ["__REC_transform_y__", "__REC_transform_x__", "__REC_transform_rot__",
                          "__REC_Mask_Size__"]:
            window[elem_key].Update(value=value, move_cursor_to=winfo.cursor_pos)
        else:
            window[elem_key].Update(value=value)


def change_list_ind_color(window: sg.Window, current_tab: str,
                          elem_ind_val_list: List[int]) -> None:
    """Change the listbox index color based off what images are loaded.

    Args:
        window: The element representing the main GUI window.
        current_tab: The key for the current tab.
        elem_ind_val_list: The list of tuples made of PySimpleGUI elements
            along with the value that the metadata of the
            element state 'Set' will change to.

    Returns:
        None
    """

    if current_tab == 'reconstruct_tab':
        num_list = list(range(11))

    for listbox_key, green_choices in elem_ind_val_list:
        listbox = window[listbox_key]
        grey_choices = setdiff1d(num_list, green_choices)
        for index in green_choices:
            listbox.Widget.itemconfig(index, fg='black', bg="light green")
        for index in grey_choices:
            listbox.Widget.itemconfig(index, fg='light grey', bg=sg.theme_input_background_color())


def change_inp_readonly_bg_color(window: sg.Window, elem_list: List[sg.Element],
                                 val: str):
    """Change the readonly input background color.

    Args:
        window: The element representing the main GUI window.
        elem_list: The list of elements whose color will change
        val: The value to change the color to.

    Returns:
        None
    """

    for elem in elem_list:
        if val == 'Default':
            window[elem].Widget.config(readonlybackground=sg.theme_input_background_color())
        elif val == 'Readonly':
            window[elem].Widget.config(readonlybackground='#A7A7A7')


def metadata_change(winfo: Struct, window: sg.Window,
                    elem_val_list: List[Tuple[sg.Element, str]],
                    reset: bool = False) -> None:
    """Change the metadata of the element to update between
    the default value and the user set value.

    Args:
        winfo: The data structure holding all information about
              windows and GUI.
        window: The element representing the main GUI window.
        elem_val_list: The list of tuples made of PySimpleGUI elements
            along with the value that the metadata of the
            element state 'Set' will change to.
        reset: If true, the 'Set' value is reset to 'Def'.
            Otherwise the value will be 'Set' as defined
            by the user.

    Returns:
        None
    """

    if reset:
        for elem in elem_val_list:
            window[elem].metadata['Set'] = window[elem].metadata['Def']
    else:
        for elem, val in elem_val_list:
            window[elem].metadata['Set'] = val
            if window[elem].metadata['State'] == 'Set':
                update_values(winfo, window, [(elem, val)])
                if elem in winfo.keys['input'] + winfo.keys['read_only_inputs']:
                    window[elem].update(move_cursor_to="end")
                    window[elem].Widget.xview_moveto(1)


def toggle(winfo: Struct, window: sg.Window,
           elem_list: List[sg.Element], state: Optional[str] = None) -> None:
    """Toggle between the default state and set state
    of an elements metadata.

    Parameters:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        elem_list: The list of elements whose state is to be changed.
        state: If the state is None, the state is changed
            from Set -> Def or Def -> Set.
            If the state is specified, that state will
            be activated.

    Returns:
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
            update_values(winfo, window, [(elem, value)])


def update_slider(winfo: Struct, window: sg.Window,
                  slider_list: List[Tuple[sg.Element, Dict]]) -> None:
    """ Updates sliders.

    Updates sliders based off passing a list
    with element, dictionary pairs. The dictionary
    contains all values to update.

    Args:
        winfo: The data structure holding all information about
          windows and GUI.
        window: The element representing the main GUI window.
            new_transform: The next mask size to apply for REC graph.
        slider_list : List of slider, dictionary tuple pairs where the dictionary
            contains the values to update.

    Returns:
        None
    """

    for slider_key, d in slider_list:
        slider = window[slider_key]
        for key in d:
            if key == "value":
                update_values(winfo, window, [(slider_key, d[key])])
            elif key == "slider_range":
                slider_range = d[key]
                slider.metadata["slider_range"] = slider_range
                window[slider_key].Update(range=slider_range)


def update_rotxy(winfo: Struct, window: sg.Window,
                 current_tab: str,
                 new_transform: Tuple[Union[int, float], Union[int, float], Union[int, float], bool]) -> (
                     Tuple[Union[int, float], Union[int, float], Union[int, float], bool]):
    """Update the rotation, x-trans, y-trans, and
    flip coordinates for the transform to apply to
    series of images.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        current_tab: The key representing the current main tab of the
            window.
        new_transform: The next transformation to potentially apply to
            img.

    Returns:
        transform: The transformation to apply to the img.
    """

    rot_val, x_val, y_val, h_flip = new_transform
    transform = float(rot_val), float(x_val), float(y_val), h_flip
    if current_tab == "reconstruct_tab":
        pref = 'REC'
        winfo.rec_transform = transform
    rot_key, x_key, y_key = (f'__{pref}_transform_rot__',
                             f'__{pref}_transform_x__',
                             f'__{pref}_transform_y__')
    elem_val_list = [(rot_key, str(rot_val)), (x_key, str(x_val)), (y_key, str(y_val))]
    update_values(winfo, window, elem_val_list)
    return transform


def update_mask_size(winfo: Struct, window: sg.Window,
                     new_transform: Tuple[Union[int, float]]) -> Tuple[float]:
    """Update the mask size.

      Args:
          winfo: The data structure holding all information about
              windows and GUI.
          window: The element representing the main GUI window.
          new_transform: The next mask size to apply for REC graph.

      Returns:
          mask_transform: The float of the mask size.
      """

    # Make sure the mask is a tuple so it works with retrieve_transform()
    mask_size = new_transform[0]
    mask_transform = (float(mask_size), )
    winfo.rec_mask = mask_transform
    mask_size_key = '__REC_Mask_Size__'
    elem_val_list = [(mask_size_key, str(mask_size))]
    update_values(winfo, window, elem_val_list)
    return mask_transform


def set_crop_data(winfo: Struct, graph: sg.Graph, images: Dict, ptie: TIE_params) -> None:
    """Set the ptie crop data for reconstruction.

          Args:
              winfo: The data structure holding all information about
                  windows and GUI.
              graph: The graph of the reconstruction tab.
              images: The dictionary of the loaded reconstruction images.
              ptie: The TIE_params object for reconstruction.

          Returns:
              mask_transform: The float of the mask size.
          """

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

    # Take care of odd number of pixels or if hitting boundaries
    if round(right * scale_x) - round(left * scale_x) != round(bottom * scale_y) - round(top * scale_y):
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

    # Make sure lengths of sides even boundaries and set scaled indices
    scaled_left, scaled_right = round(left * scale_x), round(right * scale_x)
    scaled_bottom, scaled_top = round(bottom * scale_y), round(top * scale_y)
    if scaled_left % 2 != 0:
        scaled_left = max(scaled_left - 1, 0)
    if scaled_right % 2 != 0:
        scaled_right = min(scaled_right + 1, reg_width)
    if scaled_top % 2 != 0:
        scaled_top = max(scaled_top - 1, 0)
    if scaled_bottom % 2 != 0:
        scaled_bottom = min(scaled_bottom + 1, reg_height)

    # Set ptie crop
    ptie.crop['right'], ptie.crop['left'] = scaled_right, scaled_left
    ptie.crop['bottom'], ptie.crop['top'] = scaled_bottom, scaled_top
    winfo.graph_slice = (scaled_bottom - scaled_top, scaled_right - scaled_left)  # y, x, z
    winfo.rec_pad_info = (None, 0, 0, 0)


def erase_mask_data(winfo: Struct, graph: sg.Graph, current_tab: str, display_img: bytes,
                    values: dict) -> Tuple[bool, bool, bytes, Tuple, Tuple]:
    """Erase the masks on the region selection planes

          Args:
              winfo: The data structure holding all information about
                  windows and GUI.
              graph: The graph of the reconstruction tab.
              current_tab: The current selected tab in the GUI.
              display_img: The current display image.
              values: Dictionary of GUI window values associated with element keys.

          Returns:
              Tuple that contains booleans of whether to draw/adjust mask, along with transformation
              and resulting image.
          """

    util.erase_marks(winfo, graph, current_tab, full_erase=True)
    graph_size = graph.get_size()
    draw_mask = True
    adjust = True
    if winfo.window['__REC_Mask__'].metadata['State'] == 'Def':
        winfo.rec_mask_coords = []
        winfo.rec_mask_markers = []

        draw_mask = False
        adjust = False
        stack = winfo.rec_images['REC_Stack']
        slider_val = int(values["__REC_Slider__"])
        transform = (0, 0, 0, False)
        resized_mask = util.array_resize(winfo.rec_ptie.mask, winfo.window['__REC_Graph__'].get_size())
        for i in range(stack.z_size):
            stack.uint8_data[i] = np.multiply(stack.uint8_data[i], resized_mask)
            stack.flt_data[i] = np.multiply(stack.flt_data[i], resized_mask)
            stack.byte_data[i], stack.rgba_data[i] = util.adjust_image(stack.flt_data[i], transform, stack.x_size,
                                                                       winfo.window['__REC_Graph__'].get_size()[
                                                                           0])
        image_choice = winfo.window['__REC_Image_List__'].get()[0]
        if image_choice == 'Stack':
            display_img = util.convert_to_bytes(stack.rgba_data[slider_val])

    winfo.rec_corner1 = None
    winfo.rec_corner2 = None
    winfo.new_selection = True
    winfo.rec_pad_info = (None, 0, 0, 0)

    winfo.rec_mask_center = (graph_size[0] / 2, graph_size[1] / 2)
    winfo.rec_mask = (50,)
    mask_transform = (50,)
    transform = (0, 0, 0, False)
    update_values(winfo, winfo.window, [('__REC_transform_x__', '0'), ('__REC_transform_y__', '0'),
                                        ('__REC_transform_rot__', "0"), ('__REC_Mask_Size__', '50')])
    return draw_mask, adjust, display_img, mask_transform, transform


# ------------- Visualizing Elements ------------- #
def set_pretty_focus(winfo: Struct, window: sg.Window, event: str) -> None:
    """ Sets the focus to reduce unwanted placements of
    cursor or focus within the GUI.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        event: The key for the values dictionary that represents
            an event in the window.

    Returns:
        None
    """

    # Set the 'true element' to be the one the cursor is hovering over.
    if "+HOVER+" in event and not winfo.true_element:
        winfo.true_element = event.replace("+HOVER+", "")
    elif "+STOP_HOVER+" in event:
        winfo.true_element = None
    # Window click will never set focus on button
    elif event == "Window Click":
        if winfo.true_element is not None and winfo.true_element not in winfo.keys['button']:
            window[winfo.true_element].SetFocus()
        else:
            winfo.invis_graph.SetFocus(force=True)

        if winfo.true_element in ["__REC_transform_x__", "__REC_transform_y__", "__REC_transform_rot__", "__REC_Mask_Size__"]:
            winfo.true_input_element = winfo.true_element

    # Set pretty focus for log.
    if event == 'Log Click':
        winfo.output_invis_graph.SetFocus(force=True)


def rec_get_listbox_ind_from_key(key_list: List[str]) -> List[int]:
    """Get the listbox indices from key list to color once images have loaded.

    Args:
        key_list: The list of keys of items in the rec listbox.

    Returns:
        indices: The indices of the keys that were in the key_list.
    """

    indices = [0, 2, 11]
    for key in key_list:
        if key == 'color_b':
            ind = 1
        elif key == 'bxt':
            ind = 3
        elif key == 'byt':
            ind = 4
        elif key == 'bbt':
            ind = 5
        elif key == 'phase_e':
            ind = 6
        elif key == 'phase_b':
            ind = 7
        elif key == 'dIdZ_e':
            ind = 8
        elif key == 'dIdZ_m':
            ind = 9
        elif key == 'inf_im':
            ind = 10
        indices.append(ind)
    return indices


def activate_spinner(window: sg.Window, elem: sg.Element) -> None:
    """Activate loading spinner.

    Args:
        window: The element representing the passed GUI window.
        elem: The element who's spinner should be disabled

    Returns:
        None
    """

    spinner_fn = window[elem].metadata['Set']
    window[elem].metadata['State'] = 'Set'
    window[elem].Update(spinner_fn)


def deactivate_spinner(window: sg.Window, elem: sg.Element) -> None:
    """Deactivate loading spinner.

    Args:
        window: The element representing the passed GUI window.
        elem: The element who's spinner should be disabled

    Returns:
        None
    """

    background = window[elem].metadata['Def']
    window[elem].metadata['State'] = 'Def'
    window[elem].Update(filename=background)


def redraw_graph(graph: sg.Graph, display_image: Optional[bytes]) -> None:
    """Redraw graph.

    Args:
        graph: The graph element in the window.
        display_image : If None, the graph is erased
            Else, bytes representation of the image.

    Returns:
        None
    """

    graph.Erase()
    if display_image:
        x, y = graph.get_size()
        graph.DrawImage(data=display_image, location=(0, y-1))


def change_visibility(window: sg.Window, elem_val_list: List[Tuple[sg.Element, Any]]) -> None:
    """ Take a list of element keys and change
    visibility of the element.

    Args:
        window : The element representing the main GUI window.
        elem_val_list : The list of elements with values whose
            state is to be changed.

    Returns:
        None
    """
    for elem_key, val in elem_val_list:
        window[elem_key].Update(visible=val)


def disable_elements(window: sg.Window, elem_list: List[sg.Window]) -> None:
    """ Take a list of element keys and disable the element.

    Args:
        window: The element representing the passed GUI window.
        elem_list: The list of elements whose state is to be changed.

    Returns:
        None
    """

    for elem_key in elem_list:
        window[elem_key].Update(disabled=True)


def enable_elements(winfo: Struct, window: sg.Window, elem_list: List[sg.Window]) -> None:
    """ Take a list of element keys and enable the element.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the passed GUI window.
        elem_list: The list of elements whose state is to be changed.

    Returns:
        None
    """

    for elem_key in elem_list:
        if elem_key in winfo.keys['combo']:
            window[elem_key].Update(readonly=True)
        else:
            window[elem_key].Update(disabled=False)


def ptie_init_thread(winfo: Struct, path: str, fls1_path: str, fls2_path: str,
                     stack_name: str, files1: List[str], files2: List[str],
                     single: bool, tfs_value: str) -> None:
    """ Create the PYTIE initialization thread.

    Function initializes the parameters for PYTIE. See load_data from
    TIE_helper for more information on initialized parameters.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        path: The path to datafolder.
        fls1_path: Path to the first .fls file.
        fls2_path: Path to the second .fls file.
        stack_name: Name of the stack to perform reconstruction on.
        files1: The files in the unflip or tfs folder.
        files2: The files in the flip folder or None.
        single: Boolean value if a single tfs series or flipped tfs.
        tfs_value: The defined through focal series value by the user,
            whether it is a single series or unflip/flip series.

    Returns:
        None
    """

    try:
        # Make sure there is not error with the accelerationg voltage val.
        assert (float(winfo.window['__REC_M_Volt__'].get()) > 0)
        accel_volt = float(winfo.window['__REC_M_Volt__'].get()) * 1e3

        # Load in stack data and ptie data, pulling image filenames
        string = stack_name
        index = 0
        # Make sure the stack is directly in the current working image directory.
        while index != -1:
            last_index = index
            index = string.rfind('/')
            string = string[index + 1:]
        folder = stack_name[:last_index]
        assert(folder == path[:-1])

        stack1, stack2, ptie = load_data_GUI(path, fls1_path, fls2_path, stack_name, single)
        string_vals = []
        for def_val in ptie.defvals:
            val = str(def_val)
            string_vals.append(val)
        # If single reconstruction, look for 'tfs' dir first and then try using 'unflip'
        if tfs_value == 'Single':
            prefix = 'tfs'
            path1 = util.join([path, prefix], '/')
            if not os.path.exists(path1):
                prefix = 'unflip'
        else:
            prefix = 'unflip'
        im_name = files1[0]

        # Apply ptie_mask to stack
        stack = winfo.rec_images['REC_Stack']
        transform = (0, 0, 0, False)
        resized_mask = util.array_resize(ptie.mask, winfo.window['__REC_Graph__'].get_size())
        for i in range(stack.z_size):
            stack.uint8_data[i] = np.multiply(stack.uint8_data[i], resized_mask)
            stack.flt_data[i] = np.multiply(stack.flt_data[i], resized_mask)
            stack.byte_data[i], stack.rgba_data[i] = util.adjust_image(stack.flt_data[i], transform, stack.x_size,
                                                                         winfo.window['__REC_Graph__'].get_size()[0])

        # Change the appearance and values in the GUI
        metadata_change(winfo, winfo.window, [('__REC_Image__', f'{prefix}/{im_name}')])
        length_slider = len(string_vals)
        winfo.window['__REC_Def_Combo__'].update(value=string_vals[0], values=string_vals)
        winfo.window['__REC_Def_List__'].update(ptie.defvals, set_to_index=0, scroll_to_index=0)
        winfo.window['__REC_Def_List__'].metadata['length'] = length_slider
        toggle(winfo, winfo.window, elem_list=['__REC_Set_FLS__'])
        update_slider(winfo, winfo.window, [('__REC_Defocus_Slider__', {"slider_range": (0, max(length_slider - 3, 0)),
                                             "value": 0})])
        update_slider(winfo, winfo.window, [('__REC_Slider__', {"value": 0, "slider_range": (0, stack.z_size-1)})])
        enable_elements(winfo, winfo.window, ['__REC_Def_Combo__', '__REC_QC_Input__',
                                              '__REC_Mask__', "__REC_Erase_Mask__", '__REC_Run_TIE__',
                                              "__REC_Slider__", "__REC_Colorwheel__", "__REC_Derivative__"])
        disable_elements(winfo.window, ['__REC_Stack__', '__REC_FLS1__',  '__REC_FLS2__', '__REC_M_Volt__'])
        change_inp_readonly_bg_color(winfo.window, ['__REC_Stack__', '__REC_FLS1__',  '__REC_FLS2__',
                                                    '__REC_M_Volt__'], 'Readonly')
        change_inp_readonly_bg_color(winfo.window, ['__REC_QC_Input__'], 'Default')
        values = winfo.window['__REC_Image_List__'].GetListValues()
        index = winfo.window['__REC_Image_List__'].GetIndexes()
        selected = values[index[0]]
        if selected == 'Stack':
            redraw_graph(winfo.window['__REC_Graph__'], stack.byte_data[0])

        # Load all relevant PTIE data into winfo
        winfo.rec_defocus_slider_set = 0
        winfo.rec_ptie = ptie
        winfo.rec_microscope = Microscope(E=accel_volt, Cs=200.0e3, theta_c=0.01e-3, def_spr=80.0)
        winfo.rec_files1 = files1
        winfo.rec_files2 = files2
    except:
        print(f'REC: Something went wrong during initialization.')
        print(f'REC: 1. Check to make sure aligned file is in cwd and not somewhere else.')
        print(f'REC: 2. Check to make sure the fls file(s) match the aligned file chosen.', end=' ')
        print('Otherwise PYTIE will search the wrong directories.')
        print(f'REC: 3. Check to see voltage is numerical and above 0.')
        raise

    enable_elements(winfo, winfo.window, ["__REC_Reset_FLS__",  '__REC_Reset_Img_Dir__'])
    winfo.ptie_init_thread = None
    print('--- Exited PTIE Initialization ---')


def ptie_recon_thread(winfo: Struct, window: sg.Window, graph: sg.Graph,
                      colorwheel_graph: sg.Graph, images: Dict,
                      current_tab: str) -> None:
    """ Create the PYTIE reconstruction thread.

    Function initializes the thread that runs the PYTIE reconstruction. For more
    information on the reconstruction, see TIE_reconstruct.py.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        graph: The reconstruction graph canvas.
        colorwheel_graph: The graph in the window where to place the colorwheel.
        images: The dictionary of images and their values.
        current_tab: The key representing the current main tab of the
            window.'

    Returns:
        None
    """

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
    if util.represents_float(qc):
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
    set_crop_data(winfo, graph, images, ptie)

    if not qc_passed:
        print(f'REC: QC value should be an integer or float and not negative. Change value.')
        update_values(winfo, window, [('__REC_QC_Input__', '0.00')])
    else:
        try:
            print(f'REC: Reconstructing for defocus value: {ptie.defvals[def_ind]} nm')
            rot, x_trans, y_trans = (winfo.rec_transform[0], winfo.rec_transform[1], winfo.rec_transform[2])
            ptie.rotation, ptie.x_transl, ptie.y_transl = float(rot), int(x_trans), int(y_trans)
            results = TIE(def_ind, ptie, microscope, dataname, sym, qc, save, hsv, longitudinal_deriv, v=0)

            winfo.rec_tie_results = results
            winfo.rec_def_val = def_val
            winfo.rec_sym = sym
            winfo.rec_qc = qc

            # Load the color image immediately after reconstruction
            loaded_green_list = []
            for key in results:
                float_array = results[key]
                # If single stack, just gloss over results that might be None.
                if float_array is None:
                        continue
                if key == 'color_b':
                    float_array = util.slice_im(float_array, (0, 0, winfo.graph_slice[0], winfo.graph_slice[1]))
                    colorwheel_type = window['__REC_Colorwheel__'].get()
                    rad1, rad2 = colorwheel_graph.get_size()
                    if colorwheel_type == 'HSV':
                        cwheel_hsv = colorwheel_HSV(rad1, background='black')
                        cwheel = colors.hsv_to_rgb(cwheel_hsv)
                    elif colorwheel_type == '4-Fold':
                        cwheel = colorwheel_RGB(rad1)
                    uint8_colorwheel, float_colorwheel = util.convert_float_unint8(cwheel, (rad1, rad2))
                    rgba_colorwheel = util.make_rgba(uint8_colorwheel[0])
                    winfo.rec_colorwheel = util.convert_to_bytes(rgba_colorwheel)

                if float_array.shape[0] != float_array.shape[1]:
                    if float_array.shape[0] > float_array.shape[1]:
                        pad_side = (float_array.shape[0] - float_array.shape[1]) // 2
                        axis = 1
                        try:
                            if float_array.shape[2] > 0:
                                npad = ((0, 0), (int(pad_side), int(pad_side)), (0, 0))
                        except:
                            npad = ((0, 0), (int(pad_side), int(pad_side)))

                    elif float_array.shape[0] < float_array.shape[1]:
                        pad_side = (float_array.shape[1] - float_array.shape[0]) // 2
                        axis = 0
                        try:
                            if float_array.shape[2] > 0:
                                npad = ((int(pad_side), int(pad_side)), (0, 0), (0, 0))
                        except:
                            npad = ((int(pad_side), int(pad_side)), (0, 0))
                    float_array = np.pad(float_array, pad_width=npad, mode='constant', constant_values=0)
                    winfo.rec_pad_info = (axis, pad_side, float_array.shape[0], float_array.shape[1])

                uint8_data, float_data = {}, {}
                uint8_data, float_data = util.convert_float_unint8(float_array, graph.get_size(),
                                                                     uint8_data, float_data)
                if uint8_data:
                    image = util.FileImage(uint8_data, float_data, (winfo.graph_slice[0], winfo.graph_slice[1], 1), f'/{key}',
                                             float_array=float_array)
                    image.byte_data = util.vis_1_im(image)
                    winfo.rec_images[key] = image
                    loaded_green_list.append(key)
                else:
                    winfo.rec_images[key] = None

            # Update window
            list_color_indices = rec_get_listbox_ind_from_key(loaded_green_list)
            change_list_ind_color(window, current_tab, [('__REC_Image_List__', list_color_indices)])
            redraw_graph(graph, winfo.rec_images['color_b'].byte_data)
            redraw_graph(colorwheel_graph, winfo.rec_colorwheel)
            metadata_change(winfo, window, [('__REC_Image__', f'{def_val} Color')])
            toggle(winfo, window, ['__REC_Image__'], state='Set')
            update_slider(winfo, window, [('__REC_Image_Slider__', {"value": 7 - 1})])
            window['__REC_Image_List__'].update(set_to_index=1, scroll_to_index=1)
            change_inp_readonly_bg_color(window, ['__REC_QC_Input__'], 'Default')
            winfo.rec_image_slider_set = 7 - 1
            winfo.rec_last_image_choice = 'Color'
            winfo.rec_ptie = ptie
        except:
            print(f'REC: There was an error when running TIE.')
            raise

    winfo.ptie_recon_thread = None
    print('--- Exited Reconstruction ---')


def ptie_save(winfo: Struct, window: sg.Window, cwd: str, images: Dict,
              filenames: List[str], pref: str, im_dir: str,
              save_tie: Union[str, bool],
              ) -> None:
    """ Save the current images of PYTIE.

    Function saves the images of PYTIE, see TIE_reconstruct.py for more info.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        cwd: The current working directory that stores images and stacks.
        images: The dictionary of images and their values.
        filenames: List of filenames that will be saved.
        pref: The string denoting the prefix name for labeling the images.
        im_dir: The 'image directory for saving.
        save_tie: The value for which save to apply to the current PYTIE images.
            Can be a full save, color image save, or saving of x/y magnetizations
            along with color.

    Returns:
        None
    """

    # Check to see where the results should be saved
    if not os_path.exists(f'{cwd}/images'):
        os.mkdir(f'{cwd}/images')
    winfo.rec_tie_prefix = pref
    if save_tie != 'manual':
        names = None
    else:
        names = filenames
    save_results(winfo.rec_def_val, winfo.rec_tie_results, winfo.rec_ptie,
                 pref, winfo.rec_sym, winfo.rec_qc, save=save_tie, v=2,
                 directory=im_dir, long_deriv=False, filenames=names)
    try:
        if save_tie in [True, 'b'] or save_tie == 'manual':
            if save_tie == 'manual':
                arrow_filenames = []
                for name in filenames:
                    if 'arrow' in name:
                        arrow_filenames.append(name)
            else:
                arrow_filenames = filenames[-2:]
            if arrow_filenames:
                hsv = window['__REC_Colorwheel__'].get() == "HSV"
                color_float_array = images['color_b'].float_array
                mag_x, mag_y = images['bxt'].float_array, images['byt'].float_array
                v_num, v_len, v_wid = winfo.rec_past_arrow_transform[:3]
                graph_size = window['__REC_Graph__'].get_size()
                for i in range(len(arrow_filenames)):
                    name = arrow_filenames[i]
                    if i == 0:
                        v_color = True
                    else:
                        v_color = False

                    if winfo.rec_pad_info[0] is not None:
                        max_val = max(winfo.rec_pad_info[2:])
                        pad_side = winfo.rec_pad_info[1]
                        if winfo.rec_pad_info[0] == 1:
                            start_x, end_x = pad_side, max_val - pad_side
                            start_y, end_y = 0, max_val
                        elif winfo.rec_pad_info[0] == 0:
                            start_x, end_x = 0, max_val
                            start_y, end_y = pad_side, max_val - pad_side
                        color_float_array = util.slice_im(color_float_array, (start_y, start_x, end_y, end_x))
                        mag_x = util.slice_im(mag_x, (start_y, start_x, end_y, end_x))
                        mag_y = util.slice_im(mag_y, (start_y, start_x, end_y, end_x))

                    util.add_vectors(mag_x, mag_y, color_float_array, v_color, hsv, v_num, v_len,
                                     v_wid, graph_size, winfo.rec_pad_info, save=name)
    except:
        print('Did not save images correctly')
        raise

    winfo.ptie_recon_thread = None
    print('--- Exited Saving ---')


# -------------- Home Tab Event Handler -------------- #
def run_home_tab(winfo: Struct, window: sg.Window,
                 event: str, values: Dict) -> None:
    """Run events associated with the Home tab.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        event: The key for the values dictionary that represents
            an event in the window.
        values: A dictionary where every value is paired with
            a key represented by an event in the window.

    Returns:
        None
    """

    prefix = 'HOM: '
    # Get directories for Fiji and image directory
    python_dir = os.path.dirname(os.path.abspath(__file__))
    # chmod
    default_txt = f'{python_dir}/defaults.txt'
    with open(default_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Browser Directory'):
                items = line.split(',')
                key, value = items[0], items[1]
                value = value.strip()
                if value:
                    color_def = window['__Browser_Set__'].metadata['Def']
                    color_res = window['__Browser_Reset__'].metadata['Set']
                else:
                    color_def = window['__Browser_Set__'].metadata['Set']
                    color_res = window['__Browser_Reset__'].metadata['Def']
                if winfo.last_browser_color != color_def:
                    window['__Browser_Set__'].update(button_color=color_def)
                    window['__Browser_Reset__'].update(button_color=color_res)
                    winfo.last_browser_color = color_def

    if event == '__Browser_Set__':
        python_dir = os.path.dirname(os.path.abspath(__file__))
        default_txt = f'{python_dir}/defaults.txt'
        with open(default_txt, 'r') as f:
            lines = f.readlines()
        with open(default_txt, 'w+') as fnew:
            for line in lines:
                if not line.startswith('//'):
                    items = line.split(',')
                    key, value = items[0], items[1]
                    if key == 'Browser Directory' and event == '__Browser_Set__':
                        filename = window["__Browser_Path__"].Get()
                        if os_path.exists(filename):
                            fnew.write(f'Browser Directory,{filename}\n')
                            print(f'{prefix}Browser working directory default was set.')
                            window['__REC_Image_Dir_Browse__'].InitialFolder = filename
                        else:
                            fnew.write(line)
                            print(f'{prefix}Directory does not exist, try again.')
                    elif key == 'Browser Directory' and event != '__Browser_Set__':
                        fnew.write(line)
                    else:
                        fnew.write(line)
                elif line.startswith('//'):
                    fnew.write(line)

    elif event in ['__Browser_Reset__']:
        python_dir = os.path.dirname(os.path.abspath(__file__))
        default_txt = f'{python_dir}/defaults.txt'
        with open(default_txt, 'r') as f:
            lines = f.readlines()
        with open(default_txt, 'w') as fnew:
            for line in lines:
                if line.startswith('Browser Directory') and event == '__Browser_Reset__':
                    update_values(winfo, window, [('__Browser_Path__', '')])
                    fnew.write('Browser Directory, \n')
                    print(f'{prefix}Browser working directory default was reset.')
                    window['__REC_Image_Dir_Browse__'].InitialFolder = ''
                elif line.startswith('FIJI Directory'):
                    fnew.write('FIJI Directory, \n')
                else:
                    fnew.write(line)


# -------------- Reconstruct Tab Event Handler -------------- #
def run_reconstruct_tab(winfo: Struct, window: sg.Window,
                        current_tab: str, event: str, values: Dict) -> None:
    """Run events associated with the reconstruct tab.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.
        current_tab: The key representing the current main tab of the
            window. Ex. '
        event: The key for the values dictionary that represents
            an event in the window.
        values: A dictionary where every value is paired with
            a key represented by an event in the window.

    Returns:
        None
    """

    # ------------- Visualizing Elements ------------- #
    def special_enable_disable(winfo: Struct, window: sg.Window) -> None:
        """Determine enabling and disabling of elements based off loaded buttons and active processes.

        Args:
            winfo: The data structure holding all information about
                windows and GUI.
            window: The element representing the main GUI window.

        Returns:
            None
        """

        enable_list = []
        active_keys = ['__REC_Image_Dir_Path__', '__REC_Set_Img_Dir__', '__REC_Image_Dir_Browse__',
                       '__REC_FLS_Combo__', '__REC_Load_FLS1__', '__REC_Set_FLS__',
                       '__REC_Load_FLS2__', '__REC_Load_Stack__', '__REC_Image_List__',
                       '__REC_M_Volt__', '__REC_Def_Combo__', '__REC_QC_Input__',
                       "__REC_Reset_FLS__", "__REC_TFS_Combo__", "__REC_Arrow_Num__",
                       '__REC_Arrow_Wid__', '__REC_Arrow_Len__', '__REC_Arrow_Color__',
                       '__REC_Mask_Size__', '__REC_Mask__', "__REC_Erase_Mask__",
                       "__REC_transform_y__", "__REC_transform_x__", "__REC_transform_rot__",
                       '__REC_Run_TIE__', '__REC_Save_TIE__', "__REC_Square_Region__", "__REC_Rectangle_Region__",
                       "__REC_Slider__", "__REC_Colorwheel__", "__REC_Derivative__",
                       '__REC_Reset_Img_Dir__', "__REC_Arrow_Set__"]

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
                        window['__REC_FLS2__'].metadata['State'] == 'Set' and
                        winfo.ptie_init_thread is None):
                    enable_list.extend(['__REC_Set_FLS__'])
            elif window['__REC_Set_FLS__'].metadata['State'] == 'Set' and winfo.ptie_recon_thread is None:
                enable_list.extend(['__REC_Erase_Mask__',
                                    '__REC_Def_Combo__', "__REC_Colorwheel__",
                                    '__REC_QC_Input__', "__REC_Derivative__"])
                if winfo.rec_rotxy_timers == (0, 0, 0) and winfo.rec_mask_timer == (0,):
                    enable_list.extend(['__REC_Mask__'])
                if window['__REC_Mask__'].metadata['State'] == 'Def':
                    enable_list.extend(['__REC_Run_TIE__'])
                    enable_list.extend(["__REC_Arrow_Num__", '__REC_Arrow_Color__',
                                        '__REC_Arrow_Wid__', '__REC_Arrow_Len__']),
                    if 'color_b' in winfo.rec_images:
                        enable_list.extend(["__REC_Arrow_Set__"])
                else:
                    if window['__REC_Square_Region__'].Get():
                        enable_list.extend(['__REC_Mask_Size__'])
                    enable_list.extend(["__REC_transform_y__", "__REC_Square_Region__", "__REC_Rectangle_Region__",
                                        "__REC_transform_x__",  "__REC_transform_rot__"])
            if winfo.rec_tie_results is not None and winfo.ptie_recon_thread is None:
                enable_list.extend(['__REC_Save_TIE__'])
            if (window['__REC_Stack__'].metadata['State'] == 'Set' and
                    window['__REC_Mask__'].metadata['State'] == 'Def'):
                enable_list.extend(["__REC_Image_List__"])
            if (window['__REC_Image_List__'].get()[0] in ['Stack', 'Loaded Stack'] or
                    window['__REC_Mask__'].metadata['State'] == 'Set'):
                enable_list.extend(["__REC_Slider__"])
        elif window['__REC_Set_Img_Dir__'].metadata['State'] == 'Def':
            enable_list.extend(['__REC_Image_Dir_Path__', '__REC_Set_Img_Dir__',
                                '__REC_Image_Dir_Browse__'])
        if winfo.ptie_recon_thread is None and winfo.ptie_init_thread is None:
            enable_list.extend(['__REC_Reset_Img_Dir__'])

        disable_list = setdiff1d(active_keys, enable_list)
        if ((winfo.last_rec_disable is None or winfo.last_rec_enable is None) or
                (collections.Counter(disable_list) != collections.Counter(winfo.last_rec_disable) and
                 collections.Counter(enable_list) != collections.Counter(winfo.last_rec_enable))):

            disable_elements(window, disable_list)
            enable_elements(winfo, window, enable_list)
            winfo.last_rec_enable = enable_list
            winfo.last_rec_disable = disable_list

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

    if winfo.ptie_init_thread is not None and not winfo.ptie_init_thread.is_alive():
        winfo.ptie_init_thread = None
    if winfo.ptie_recon_thread is not None and not winfo.ptie_recon_thread.is_alive():
        winfo.ptie_recon_thread = None

    if winfo.ptie_recon_thread is not None:
        winfo.rec_past_recon_thread = 'alive'
    elif winfo.ptie_recon_thread is None and winfo.rec_past_recon_thread is not None:
        if 'color_b' in images:
            # Add the vector image
            hsv = window['__REC_Colorwheel__'].get() == 'HSV'
            color_float_array = images['color_b'].float_array
            mag_x, mag_y = images['bxt'].float_array, images['byt'].float_array
            vector_color = window['__REC_Arrow_Color__'].get()
            if vector_color == 'On':
                vector_color = True
            elif vector_color == 'Off':
                vector_color = False
            vector_num = int(window['__REC_Arrow_Num__'].get())
            vector_len, vector_wid = float(window['__REC_Arrow_Len__'].get()), float(window['__REC_Arrow_Wid__'].get())
            graph_size = graph.get_size()
            byte_img = util.add_vectors(mag_x, mag_y, color_float_array,
                                          vector_color, hsv, vector_num, vector_len,
                                          vector_wid, graph_size, winfo.rec_pad_info, save=None)

            shape = color_float_array.shape
            im = util.FileImage(np.empty(shape), np.empty(shape),
                                (winfo.graph_slice[0], winfo.graph_slice[1], 1), '/vector')
            im.byte_data = byte_img
            winfo.rec_images['vector'] = im
        winfo.rec_past_recon_thread = None

    prefix = 'REC: '
    display_img = None
    display_img2 = None

    # Import event handler names (overlaying, etc.)
    adjust = mask_button.metadata['State'] == 'Set' and (winfo.rec_past_transform != transform or
                                                         winfo.rec_past_mask != mask_transform)
    image_list = window['__REC_Image_List__'].get()
    try:
        change_img = winfo.rec_last_image_choice != image_list[0]
    except:
        list_values = window['__REC_Image_List__'].GetListValues()
        last_index = list_values.index(winfo.rec_last_image_choice)
        window['__REC_Image_List__'].update(set_to_index=last_index)
        change_img = True
    change_colorwheel = winfo.rec_last_colorwheel_choice != colorwheel_choice
    scroll = (event in ['MouseWheel:Up', 'MouseWheel:Down']
              and (window['__REC_Image_List__'].get()[0] in ['Stack', 'Loaded Stack'] or
                   window['__REC_Mask__'].metadata['State'] == 'Set')
              and winfo.rec_images
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
            toggle(winfo, window, ['__REC_Set_Img_Dir__'], state='Set')
            for item in ["__REC_Load_FLS2__", "__REC_Load_FLS1__", "__REC_Load_Stack__"]:
                window[item].InitialFolder = image_dir
            change_inp_readonly_bg_color(window, ['__REC_Stack__', '__REC_FLS1__', '__REC_FLS2__'], 'Default')
            print(f'{prefix}The path is set: {image_dir}.')
        else:
            print(f'{prefix}This pathname is incorrect.')

    # Load Stack
    elif event == '__REC_Stack_Stage__':
        stack_path = window['__REC_Stack_Stage__'].Get()
        update_values(winfo, window, [('__REC_Stack_Stage__', 'None')])
        if os_path.exists(stack_path) and (stack_path.endswith('.tif') or stack_path.endswith('.tiff')):
            graph = window['__REC_Graph__']
            graph_size = graph.get_size()
            uint8_data, flt_data, size = util.load_image(stack_path, graph_size,  event, stack=True, prefix='REC: ')
            if uint8_data:
                stack = util.Stack(uint8_data, flt_data, size, stack_path)
                stack_def = util.Stack(uint8_data, flt_data, size, stack_path)
                slider_range = (0, stack.z_size - 1)
                slider_val = 0
                winfo.rec_images['REC_Stack'] = stack
                winfo.rec_images['REC_Def_Stack'] = stack_def
                for i in range(stack.z_size):
                    stack.byte_data[i], stack.rgba_data[i] = util.adjust_image(stack.flt_data[i], transform,
                                                                                 stack.x_size, graph.get_size()[0])
                    if i == slider_val:
                        display_img = stack.byte_data[i]
                metadata_change(winfo, window, [('__REC_Stack__', stack.shortname)])
                toggle(winfo, window, ['__REC_Stack__', '__REC_Image__'], state="Set")
                update_slider(winfo, window, [('__REC_Slider__', {"value": slider_val, "slider_range": slider_range})])
                winfo.rec_last_image_choice = 'Stack'
                change_list_ind_color(window, current_tab, [('__REC_Image_List__', [0, 11])])
                change_inp_readonly_bg_color(window, ['__REC_Stack__'], 'Readonly')
                metadata_change(winfo, window, [('__REC_Image__', f'Image {slider_val + 1}')])
                print(f'{prefix}The file {stack.shortname} was loaded.')
        else:
            if len(stack_path) != 0 and stack_path != "None":
                print(f'{prefix}Stack path is not valid.')

    # Set number of FLS files to use
    elif event == '__REC_FLS_Combo__' or event == '__REC_TFS_Combo__':
        fls_value = window['__REC_FLS_Combo__'].Get()
        tfs_value = window['__REC_TFS_Combo__'].Get()
        winfo.rec_fls_files = [None, None]
        metadata_change(winfo, window, ['__REC_FLS2__', '__REC_FLS1__'], reset=True)
        toggle(winfo, window, ['__REC_FLS2__', '__REC_FLS1__'], state='Def')
        change_inp_readonly_bg_color(window, ['__REC_FLS1__', '__REC_FLS2__'], 'Default')
        # FLS Combo Chosen
        if event == '__REC_FLS_Combo__':
            # If one fls file is to be used
            metadata_change(winfo, window, [('__REC_FLS_Combo__', fls_value)])
            if fls_value == 'One':
                toggle(winfo, window, ['__REC_FLS_Combo__', '__REC_FLS2__'], state='Set')
                if tfs_value == 'Unflip/Flip':
                    val = 'Both'
                elif tfs_value == 'Single':
                    val = tfs_value
                change_inp_readonly_bg_color(window, ['__REC_FLS2__'], 'Readonly')
            # If two fls file is to be used
            elif fls_value == 'Two':
                val = fls_value
                metadata_change(winfo, window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__'], reset=True)
                toggle(winfo, window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__',
                                       '__REC_FLS2__'], state='Def')
        # TFS Combo Chosen
        elif event == '__REC_TFS_Combo__':
            metadata_change(winfo, window, [('__REC_TFS_Combo__', tfs_value)])
            if tfs_value == 'Unflip/Flip':
                val = 'Two'
                metadata_change(winfo, window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__'], reset=True)
                toggle(winfo, window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__',
                                       '__REC_FLS2__'], state='Def')
            elif tfs_value == 'Single':
                val = tfs_value
                metadata_change(winfo, window, [('__REC_FLS_Combo__', 'One')])
                toggle(winfo, window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__',
                                       '__REC_FLS2__'], state='Set')
                change_inp_readonly_bg_color(window, ['__REC_FLS2__'], 'Readonly')
        window['__REC_FLS1_Text__'].update(value=window['__REC_FLS1_Text__'].metadata[val])
        window['__REC_FLS2_Text__'].update(value=window['__REC_FLS2_Text__'].metadata[val])

    # Load FLS files
    elif event == '__REC_FLS1_Staging__' or event == '__REC_FLS2_Staging__':
        tfs_value = window['__REC_TFS_Combo__'].Get()
        fls_value = window['__REC_FLS_Combo__'].Get()
        if 'FLS1' in event:
            fls_path = window['__REC_FLS1_Staging__'].Get()
            update_values(winfo, window, [('__REC_FLS1_Staging__', 'None')])
            target_key = '__REC_FLS1__'
        elif 'FLS2' in event:
            fls_path = window['__REC_FLS2_Staging__'].Get()
            update_values(winfo, window, [('__REC_FLS2_Staging__', 'None')])
            target_key = '__REC_FLS2__'
        if os_path.exists(fls_path) and fls_path.endswith('.fls'):
            fls = util.FileObject(fls_path)
            if 'FLS1' in event:
                winfo.rec_fls_files[0] = fls
                if tfs_value == 'Unflip/Flip' and fls_value == 'One':
                    winfo.rec_fls_files[1] = fls
            elif 'FLS2' in event:
                winfo.rec_fls_files[1] = fls
            metadata_change(winfo, window, [(target_key, fls.shortname)])
            toggle(winfo, window, [target_key], state='Set')
            change_inp_readonly_bg_color(window, [target_key], 'Readonly')
        else:
            if len(fls_path) != 0 and fls_path != "None":
                print(f'{prefix}File is not read as an fls file.')

    # Set number of FLS files to use
    elif event == '__REC_Reset_FLS__':
        winfo.rec_images = {}
        winfo.rec_fls_files = [None, None]
        winfo.rec_ptie = None

        # --- Set up loading files --- #
        winfo.rec_defocus_slider_set = 0
        winfo.rec_image_slider_set = 7

        # Image selection
        winfo.rec_tie_results = None
        winfo.rec_def_val = None

        # Graph and mask making
        winfo.graph_slice = (None, None)
        winfo.rec_graph_double_click = False
        winfo.rec_mask_coords = []
        winfo.rec_mask_markers = []
        winfo.rec_pad_info = (None, 0, 0, 0)

        graph.Erase()
        colorwheel_graph.Erase()
        metadata_change(winfo, window, ['__REC_FLS1__', '__REC_FLS2__', '__REC_Stack__'], reset=True)
        toggle(winfo, window, ['__REC_FLS_Combo__', '__REC_TFS_Combo__', '__REC_Stack__',
                               '__REC_FLS1__', '__REC_FLS2__', '__REC_Set_FLS__', '__REC_Image__',
                               '__REC_Mask__'], state='Def')
        window['__REC_Def_Combo__'].update(value='None', values=['None'])
        window['__REC_Def_List__'].update(values=['None'])
        window['__REC_FLS1_Text__'].update(value=window['__REC_FLS1_Text__'].metadata['Two'])
        window['__REC_FLS2_Text__'].update(value=window['__REC_FLS2_Text__'].metadata['Two'])
        window['__REC_FLS1_Text__'].metadata['State'] = 'Two'
        window['__REC_FLS2_Text__'].metadata['State'] = 'Two'
        update_values(winfo, window, [('__REC_FLS1_Staging__', ''), ('__REC_FLS2_Staging__', ''),
                                      ('__REC_Stack_Stage__', ''), ('__REC_Def_Combo__', 'None'),
                                      ('__REC_Image__', 'None')])
        change_list_ind_color(window, current_tab, [('__REC_Image_List__', [])])
        change_inp_readonly_bg_color(window, ['__REC_Stack__', '__REC_FLS1__',
                                              '__REC_FLS2__'], 'Default')
        change_inp_readonly_bg_color(window, ['__REC_QC_Input__'], 'Readonly')
        # Re-init reconstruct
        update_slider(winfo, window, [('__REC_Defocus_Slider__', {'value': winfo.rec_defocus_slider_set,
                                                                  'slider_range': (0, 0)}),
                                      ('__REC_Slider__', {'value': 0, 'slider_range': (0, 0)}),
                                      ('__REC_Image_Slider__', {'value': winfo.rec_image_slider_set})])
        window['__REC_Image_List__'].update(set_to_index=0, scroll_to_index=0)
        window['__REC_Def_List__'].update(set_to_index=0, scroll_to_index=0)
        print(f'{prefix}FLS and reconstruct data reset.')

    # Set which image you will be working with FLS files
    elif event == '__REC_Set_FLS__':
        # Get PYTIE loading params
        path = image_dir + '/'
        stack_name = images['REC_Stack'].path

        # Get FLS value information
        tfs_value = window['__REC_TFS_Combo__'].Get()
        fls_value = window['__REC_FLS_Combo__'].Get()
        if tfs_value == 'Unflip/Flip':
            fls_file_names = [winfo.rec_fls_files[0].path, winfo.rec_fls_files[1].path]
        else:
            fls_file_names = [winfo.rec_fls_files[0].path, None]

        check = check_setup(image_dir, tfs_value, fls_value, fls_file_names, prefix='REC: ')

        # The resulting check.
        if check and check[1] is not None:
            path1, path2, files1, files2 = check[1:]
            fls_1 = winfo.rec_fls_files[0]
            fls_2 = winfo.rec_fls_files[1]
            fls1_path = fls_1.path
            if tfs_value != 'Single':
                fls2_path = fls_2.path
            else:
                fls2_path = None

            # Is this single series or flipped/unflipped series
            if tfs_value != 'Single':
                single = False
            else:
                single = True

            # Load ptie params
            if ((2*len(files1) == images['REC_Stack'].z_size and tfs_value == 'Unflip/Flip') or
                    (len(files1) == images['REC_Stack'].z_size and tfs_value == 'Single')):

                winfo.ptie_init_thread = Thread(target=ptie_init_thread,
                                                 args=(winfo, path, fls1_path, fls2_path, stack_name,
                                                       files1, files2, single, tfs_value),
                                                 daemon=True)
                print('--- Start PTIE Initialization ---')
                winfo.ptie_init_thread.start()
            else:
                print(f'{prefix}The number of expected files does not match the', end=' ')
                print('current stack.')
        else:
            print(f'{prefix}There was an incompatibility between the fls contents and the', end=' ')
            print('files within the directories. Check to make sure the folder you loaded', end=' ')
            print('the image stack from is the set working directory.', end=' ')

    # Change the slider
    elif event == '__REC_Slider__':
        stack_choice = window['__REC_Image_List__'].get()[0]
        if stack_choice == 'Stack':
            stack_key = 'REC_Stack'
        elif stack_choice == 'Loaded Stack':
            stack_key = 'REC_Def_Stack'
        stack = images[stack_key]
        slider_val = int(values["__REC_Slider__"])

        # Update window
        if stack_key == 'REC_Def_Stack':
            display_img = stack.byte_data[slider_val]
        elif stack_key == 'REC_Stack':
            if window['__REC_Mask__'].metadata['State'] == 'Set':
                display_img, stack.rgba_data[slider_val] = util.adjust_image(stack.flt_data[slider_val], transform,
                                                                      stack.x_size, graph.get_size()[0])
            else:
                display_img = util.convert_to_bytes(stack.rgba_data[slider_val])

        if winfo.rec_files1:
            if winfo.rec_files1 and winfo.rec_files2:
                if slider_val < len(winfo.rec_files1):
                    pref = 'unflip'
                    im_name = winfo.rec_files1[slider_val]
                elif slider_val >= len(winfo.rec_files1):
                    pref = 'flip'
                    im_name = winfo.rec_files2[slider_val % len(winfo.rec_files1)]
            else:
                if os_path.exists(f'{image_dir}/tfs/'):
                    pref = 'tfs'
                else:
                    pref = 'unflip'
                im_name = winfo.rec_files1[slider_val]
            metadata_change(winfo, window, [('__REC_Image__', f'{pref}/{im_name}')])
        else:
            metadata_change(winfo, window, [('__REC_Image__', f'Image {slider_val+1}')])

    # Scroll through stacks in the graph area
    elif scroll:
        stack_choice = window['__REC_Image_List__'].get()[0]

        if stack_choice in ['Stack'] or window['__REC_Mask__'].metadata['State'] == 'Set':
            stack = images['REC_Stack']
        elif stack_choice == 'Loaded Stack':
            stack = images['REC_Def_Stack']

        slider_val = int(values["__REC_Slider__"])
        max_slider_val = stack.z_size - 1
        # Scroll up or down
        if event == 'MouseWheel:Down':
            slider_val = min(max_slider_val, slider_val+1)
        elif event == 'MouseWheel:Up':
            slider_val = max(0, slider_val-1)

        # Update the window
        if stack_choice == 'Stack':
            if window['__REC_Mask__'].metadata['State'] == 'Set':
                display_img, stack.rgba_data[slider_val] = util.adjust_image(stack.flt_data[slider_val], transform,
                                                                             stack.x_size, graph.get_size()[0])
            else:
                display_img = util.convert_to_bytes(stack.rgba_data[slider_val])

        elif stack_choice == 'Loaded Stack':
            display_img = stack.byte_data[slider_val]

        update_slider(winfo, window, [('__REC_Slider__', {"value": slider_val})])
        if winfo.rec_files1:
            if winfo.rec_files1 and winfo.rec_files2:
                if slider_val < len(winfo.rec_files1):
                    pref = 'unflip'
                    im_name = winfo.rec_files1[slider_val]
                elif slider_val >= len(winfo.rec_files1):
                    pref = 'flip'
                    im_name = winfo.rec_files2[slider_val % len(winfo.rec_files1)]
            else:
                if os_path.exists(f'{image_dir}/tfs/'):
                    pref = 'tfs'
                else:
                    pref = 'unflip'
                im_name = winfo.rec_files1[slider_val]
            metadata_change(winfo, window, [('__REC_Image__', f'{pref}/{im_name}')])
        else:
            metadata_change(winfo, window, [('__REC_Image__', f'Image {slider_val+1}')])

    # Scroll through image options
    elif scroll_images:
        max_slider_val = 7
        if event in ['MouseWheel:Down', 'MouseWheel:Up']:
            slider_set = winfo.rec_image_slider_set
            if event == 'MouseWheel:Up':
                slider_val = min(max_slider_val, slider_set + 1)
            elif event == 'MouseWheel:Down':
                slider_val = max(0, slider_set - 1)
        elif event == "__REC_Image_Slider__":
            slider_val = int(values["__REC_Image_Slider__"])
        update_slider(winfo, window, [('__REC_Image_Slider__', {"value": slider_val})])
        window['__REC_Image_List__'].update(scroll_to_index=max_slider_val-slider_val)
        winfo.rec_image_slider_set = slider_val

    # Scroll through defocus options
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

        update_slider(winfo, window, [('__REC_Defocus_Slider__', {"value": slider_val})])
        window['__REC_Def_List__'].update(scroll_to_index=slider_val)
        winfo.rec_defocus_slider_set = slider_val

    # Changing view stack combo
    elif change_img:
        list_values = window['__REC_Image_List__'].GetListValues()
        last_index = None
        if winfo.rec_last_image_choice is not None:
            last_index = list_values.index(winfo.rec_last_image_choice)
        image_choice = window['__REC_Image_List__'].get()[0]
        if image_choice == 'Stack':
            image_key = 'REC_Stack'
        elif image_choice == 'Loaded Stack':
            image_key = 'REC_Def_Stack'
        elif image_choice == 'Color':
            image_key = 'color_b'
            im_name = 'Color'
        elif image_choice == 'Vector Im.':
            image_key = 'vector'
            im_name = 'Vectorized Msat'
        elif image_choice == 'MagX':
            image_key = 'bxt'
            im_name = 'X-Comp. of Mag. Induction'
        elif image_choice == 'MagY':
            image_key = 'byt'
            im_name = 'Y-Comp. of Mag. Induction'
        elif image_choice == 'Mag. Magnitude':
            image_key = 'bbt'
            im_name = 'Magnitude of Mag. Induction'
        elif image_choice == 'Mag. Phase':
            image_key = 'phase_b'
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
            if image_key in images and image_choice in ['Stack', 'Loaded Stack'] and images[image_key] is not None:
                stack = images[image_key]
                slider_val = 0
                slider_range = (0, stack.z_size - 1)

                # Update window
                if winfo.rec_files1:
                    if winfo.rec_files1 and winfo.rec_files2:
                        if slider_val < len(winfo.rec_files1):
                            pref = 'unflip'
                            im_name = winfo.rec_files1[slider_val]
                        elif slider_val >= len(winfo.rec_files1):
                            pref = 'flip'
                            im_name = winfo.rec_files2[slider_val % len(winfo.rec_files1)]
                    else:
                        if os_path.exists(f'{image_dir}/tfs/'):
                            pref = 'tfs'
                        else:
                            pref = 'unflip'
                        im_name = winfo.rec_files1[slider_val]
                    metadata_change(winfo, window, [('__REC_Image__', f'{pref}/{im_name}')])
                else:
                    metadata_change(winfo, window, [('__REC_Image__', f'Image {slider_val + 1}')])
                if image_key == 'REC_Stack':
                    display_img = util.convert_to_bytes(stack.rgba_data[slider_val])
                elif image_key == 'REC_Def_Stack':
                    display_img = stack.byte_data[slider_val]
                colorwheel_graph.Erase()
                update_slider(winfo, window, [('__REC_Slider__', {"value": slider_val, "slider_range": slider_range})])
            # Other image set
            elif image_key in images and images[image_key] is not None:
                image = images[image_key]
                display_img = image.byte_data
                if image_key == 'color_b' or (image_key == 'vector' and window['__REC_Arrow_Color__'].Get() == 'On'):
                    display_img2 = winfo.rec_colorwheel
                else:
                    colorwheel_graph.Erase()
                metadata_change(winfo, window, [('__REC_Image__', f'{winfo.rec_def_val} {im_name}')])
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
            metadata_change(winfo, window, [('__REC_Image__', f'{prefix}/{im_name}')])

        # Start mask making make_mask_button
        if mask_button.metadata['State'] == 'Def':
            toggle(winfo, window, ['__REC_Mask__'], state='Set')
            update_slider(winfo, window, [('__REC_Slider__', {"value": slider_val, "slider_range": slider_range})])
            draw_mask = True
            display_img, rgba_img = util.adjust_image(stack.flt_data[slider_val],
                                                        transform, stack.x_size, graph.get_size()[0])
            if window['__REC_Square_Region__'].Get():
                util.draw_square_mask(winfo, graph)

        # Quit mask making make_mask_button
        elif mask_button.metadata['State'] == 'Set':
            if winfo.rec_mask_coords and ((abs(winfo.rec_mask_coords[2][0] - winfo.rec_mask_coords[0][0]) > 16) and
                    (abs(winfo.rec_mask_coords[2][1] - winfo.rec_mask_coords[0][1]) > 16)):
                toggle(winfo, window, ['__REC_Mask__'], state='Def')
                draw_mask = False

                # Apply cropping to all images
                coords = winfo.rec_mask_coords
                graph_size = graph.CanvasSize
                for i in range(stack.z_size):
                    temp_img, stack.rgba_data[i] = util.adjust_image(stack.flt_data[i],
                                                                       transform, stack.x_size, graph.get_size()[0])
                    temp_img, stack.rgba_data[i] = util.apply_crop_to_stack(coords, graph_size, stack, i)
                    if i == slider_val:
                        display_img = temp_img
            else:
                print('Must choose a larger mask size.')

        colorwheel_graph.Erase()
        window['__REC_Image_List__'].update(set_to_index=0, scroll_to_index=0)
        update_slider(winfo, window, [('__REC_Image_Slider__', {"value": 7})])
        winfo.rec_last_image_choice = 'Stack'
        winfo.rec_image_slider_set = 7

    # Draw the square mask if the region is set
    elif event == '__REC_Square_Region__' and mask_button.metadata['State'] == 'Set':
        util.erase_marks(winfo, graph, current_tab)
        draw_mask = True
        util.draw_square_mask(winfo, graph)

    # Erase graph and draw the rectangular region if the region is set
    elif event == '__REC_Rectangular_Region__' and mask_button.metadata['State'] == 'Set':
        util.erase_marks(winfo, graph, current_tab)
        winfo.rec_mask_coords = []
        winfo.rec_mask_markers = []
        winfo.rec_pad_info = (None, 0, 0, 0)
        winfo.rec_corner1 = None
        winfo.rec_corner2 = None
        winfo.new_selection = True

    # Clicking on graph and making markers for mask
    elif event in ['__REC_Graph__', '__REC_Graph__+UP'] and mask_button.metadata['State'] == 'Set':

        # Erase any previous marks
        util.erase_marks(winfo, graph, current_tab)

        # Draw new marks
        value = values['__REC_Graph__']
        if window['__REC_Square_Region__'].Get():
            winfo.rec_mask_center = round(value[0]), round(value[1])
            util.draw_square_mask(winfo, graph)
        elif window['__REC_Rectangle_Region__'].Get():
            if winfo.new_selection:
                winfo.rec_corner1 = round(value[0]), round(value[1])
                winfo.rec_corner2 = None
                winfo.new_selection = False
            elif not winfo.new_selection:
                winfo.rec_corner2 = round(value[0]), round(value[1])

                x_left = min(winfo.rec_corner1[0], winfo.rec_corner2[0])
                x_right = max(winfo.rec_corner1[0], winfo.rec_corner2[0])
                y_top = max(winfo.rec_corner1[1], winfo.rec_corner2[1])
                y_bottom = min(winfo.rec_corner1[1], winfo.rec_corner2[1])

                winfo.rec_mask_coords = [(x_left, y_top), (x_left, y_bottom), (x_right, y_bottom), (x_right, y_top)]

        if event == '__REC_Graph__+UP':
            winfo.new_selection = True
        draw_mask = True

    # Remove all mask coordinates from the graph and mask file
    elif event == '__REC_Erase_Mask__':
        # Erase any previous marks
        draw_mask, adjust, display_img, mask_transform, transform = erase_mask_data(winfo, graph, current_tab,
                                                                                    display_img, values)

    # Run PyTIE
    elif event == '__REC_Run_TIE__':
        # Make sure stack still exists before trying to run PyTIE
        stack_path = window['__REC_Stack__'].Get()
        if os_path.exists(util.join([image_dir, stack_path], '/')):
            change_inp_readonly_bg_color(window, ['__REC_QC_Input__'], 'Readonly')
            winfo.ptie_recon_thread = Thread(target=ptie_recon_thread,
                                             args=(winfo, window, graph, colorwheel_graph, images, current_tab),
                                             daemon=True)
            print('--- Starting Reconstruction ---')
            winfo.ptie_recon_thread.start()
        else:
            print('The stack has been deleted since it has been loaded. You must restart.')

    # Save PyTIE
    elif event == '__REC_Save_TIE__':
        if winfo.rec_tie_results:
            tfs = values['__REC_TFS_Combo__']
            filenames, overwrite_signals, additional_vals = run_save_window(winfo, event, image_dir,
                                                                            orientations=prefix,
                                                                            defocus=winfo.rec_def_val, tfs=tfs)
            pref, save_tie, im_dir = additional_vals
            if len(overwrite_signals) > 0:
                save = overwrite_signals[0]
            else:
                save = False
            if filenames == 'close' or not filenames or not save or not save_tie:
                print(f'{prefix}Exited without saving files!\n')
            elif save:
                winfo.ptie_recon_thread = Thread(target=ptie_save,
                                                 args=(winfo, window, image_dir, images, filenames,
                                                       pref, im_dir, save_tie),
                                                 daemon=True)
                print('--- Starting Saving ---')
                winfo.ptie_recon_thread.start()
        else:
            print(f"{prefix}Reconstruction results haven't been generated.")

    # Update the arrow images
    elif event == "__REC_Arrow_Set__":
        arrow_transform = get_arrow_transform(window)
        if (util.represents_int_above_0(arrow_transform[0]) and
                util.represents_float(arrow_transform[1]) and
                util.represents_float(arrow_transform[2]) and
                arrow_transform[0] not in [''] and
                arrow_transform[1] not in ['', '.'] and
                arrow_transform[2] not in ['', '.'] and
                float(arrow_transform[1]) > 0 and float(arrow_transform[2]) > 0):

            # Change the vector image
            arrow_transform = (int(arrow_transform[0]),
                               float(arrow_transform[1]),
                               float(arrow_transform[2]),
                               arrow_transform[3])
            winfo.rec_past_arrow_transform = arrow_transform
            hsv = window['__REC_Colorwheel__'].get() == "HSV"

            color_float_array = images['color_b'].float_array
            mag_x, mag_y = images['bxt'].float_array, images['byt'].float_array
            v_num, v_len, v_wid, v_color = arrow_transform
            v_color = v_color == 'On'
            graph_size = graph.get_size()

            byte_img = util.add_vectors(mag_x, mag_y, color_float_array,
                                          v_color, hsv, v_num, v_len,
                                          v_wid, graph_size, winfo.rec_pad_info, save=None)
            shape = color_float_array.shape

            im = util.FileImage(np.empty(shape), np.empty(shape),
                                  (winfo.graph_slice[0], winfo.graph_slice[1], 1), '/vector')

            im.byte_data = byte_img
            winfo.rec_images['vector'] = im
            image_choice = window['__REC_Image_List__'].get()[0]
            if image_choice == 'Vector Im.':
                display_img = im.byte_data
                display_img2 = winfo.rec_colorwheel
        else:
            print(f'{pref}Some of the arrow values are incorrect. Check to make sure', end=' '),
            print(f'that the number of arrows is an integer > 0, & the len/width are floats > 0.')

    # Adjust stack and related variables
    if adjust:
        if winfo.rec_past_transform != transform:
            stack = images['REC_Stack']
            slider_val = int(values["__REC_Slider__"])
            display_img, stack.rgba_data[slider_val] = util.adjust_image(stack.flt_data[slider_val],
                                                                           transform, stack.x_size,
                                                                           graph.get_size()[0])
        util.erase_marks(winfo, graph, current_tab, full_erase=True)
        if window['__REC_Square_Region__']:
            util.draw_square_mask(winfo, graph)
        elif window['__REC_Rectangle_Region__']:
            pass

    winfo.rec_past_transform = transform
    winfo.rec_past_mask = mask_transform

    # Change the colorwheel
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
            uint8_colorwheel, float_colorwheel = util.convert_float_unint8(cwheel, (rad1, rad2))
            rgba_colorwheel = util.make_rgba(uint8_colorwheel[0])
            winfo.rec_colorwheel = util.convert_to_bytes(rgba_colorwheel)
            results = winfo.rec_tie_results
            results['color_b'] = color_im(results['bxt'], results['byt'],
                                          hsvwheel=hsvwheel, background='black')

            float_array = util.slice_im(results['color_b'], (0, 0, winfo.graph_slice[0], winfo.graph_slice[1]))
            uint8_data, float_data = {}, {}
            uint8_data, float_data = util.convert_float_unint8(float_array, graph.get_size(),
                                                                 uint8_data, float_data)
            image = util.FileImage(uint8_data, float_data, (winfo.graph_slice[0], winfo.graph_slice[1], 1), 'color_b',
                                     float_array=float_array)
            image.byte_data = util.vis_1_im(image)
            winfo.rec_images['color_b'] = image

            # Add the vector image
            color_float_array = float_array
            mag_x, mag_y = images['bxt'].float_array, images['byt'].float_array
            vector_color = window['__REC_Arrow_Color__'].get()

            vector_num = int(window['__REC_Arrow_Num__'].get())
            vector_len, vector_wid = int(window['__REC_Arrow_Len__'].get()), int(window['__REC_Arrow_Wid__'].get())
            graph_size = graph.get_size()
            byte_img = util.add_vectors(mag_x, mag_y, color_float_array,
                                        True, hsvwheel, vector_num, vector_len,
                                        vector_wid, graph_size, winfo.rec_pad_info, save=None)
            shape = float_array.shape
            im = util.FileImage(np.empty(shape), np.empty(shape),
                                (winfo.graph_slice[0], winfo.graph_slice[1], 1), '/vector')
            im.byte_data = byte_img
            winfo.rec_images['vector'] = im

            if window['__REC_Image_List__'].get()[0] == 'Color':
                display_img = image.byte_data
                display_img2 = winfo.rec_colorwheel
            elif window['__REC_Image_List__'].get()[0] == 'Vector Im.' and vector_color == 'On':
                display_img = im.byte_data
                display_img2 = winfo.rec_colorwheel
    winfo.rec_last_colorwheel_choice = colorwheel_choice

    # Reset page
    if event == "__REC_Reset_Img_Dir__":
        if winfo.ptie_init_thread is not None and winfo.ptie_init_thread.is_alive():
            winfo.ptie_init_thread = None
        if winfo.ptie_recon_thread is not None and winfo.ptie_recon_thread.is_alive():
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
        util.draw_mask_points(winfo, graph, current_tab)
    if winfo.rec_mask_coords and mask_button.metadata['State'] == 'Def':
        text = ' Set '
        mask_color = 'green'
        font = 'Times 18 bold'
    else:
        text = 'Unset'
        mask_color = 'black'
        font = 'Times 17'
    window['__REC_Mask_Text__'].update(value=text, text_color=mask_color, font=font)


# -------------- Save Window --------------#
def check_overwrite(winfo: Struct, save_win: sg.Window, true_paths: List[str],
                    orientations: List[str], image_dir: str,
                    im_type: str, event: str, tfs) -> List[bool]:
    """Check whether the paths listed in the log box for
    each image will be overwritten.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        save_win: The save window element
        true_paths: A list of path names that will be
            checked if they exist.
        orientations:  A list of strings that represent
            the orientations of the image ('flip',
            'unflip', 'stack', etc.)
        image_dir: The path of the current working directory of images
        im_type:  Image type (.bmp, .tiff, etc.)
        event: The key for the values dictionary that represents
            an event in the window.
        tfs: The through focal series value determined by the user for BUJ or LS.

    Returns:
        overwrite_signals: The boolean values for overwriting the files.
    """

    # If file exists notify user and give option to change name
    update_values(winfo, save_win, [('__save_win_log__', '')])
    overwrite_signals = [None]*len(true_paths)
    save_enable = True
    if event == '__REC_Save_TIE__':
        overwrite_box = save_win[f'__save_win_overwrite1__'].Get()
    rec_tie_dont_overwrite_state = False
    rec_tie_dont_overwrite_text = 'Some files already exist. Check overwrite box or change name.'
    for i in range(len(true_paths)):
        text = ''
        if event == '__REC_Save_TIE__':
            path = f'{image_dir}/images/{true_paths[i]}'
        else:
            path = f'{image_dir}/{true_paths[i]}'
        exists = os_path.exists(path)

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
            if tfs != 'Single' or (tfs == 'Single' and 'phase_e' not in true_paths[i] and
                                   'dIdZ_e' not in true_paths[i]):
                if exists and not overwrite_box:
                    rec_tie_dont_overwrite_state = True
                    save_enable = False
                    overwrite_signals = [False]
                elif exists and overwrite_box:
                    text = f'''The {true_paths[i]} file will be overwritten.'''
                    overwrite_signals = [True]
                elif not exists:
                    text = f'The {true_paths[i]} file will be saved.'
                    overwrite_signals = [True]

        # Update save window
        current_log_text = save_win['__save_win_log__'].Get()
        new_log_text = current_log_text + text
        update_values(winfo, save_win, [('__save_win_log__', new_log_text.strip())])

    if rec_tie_dont_overwrite_state:
        update_values(winfo, save_win, [('__save_win_log__', rec_tie_dont_overwrite_text.strip())])
    if save_enable:
        enable_elements(winfo, save_win, ['__save_win_save__'])
    else:
        disable_elements(save_win, ['__save_win_save__'])
    return overwrite_signals


def save_window_values(save_win: sg.Window, num_paths: int, event: str,
                       orientations: List[str], defocus: Optional[str] = None, tfs: Optional[str] = None,
                       true_paths: Optional[List] = None, file_choices: Optional[List] = []) -> List[str]:
    """Sets ups the save window layout.

    Args:
        save_win: The representation of the save window.
        num_paths: The number of paths, to create the number
            of overwrite checkboxes and true_path
            input elements.
        event: The save event from the main GUI window.
        orientations:  A list of strings that represent
            the orientations of the image ('flip',
            'unflip', 'stack', etc.)
        defocus: The defocus value for the image if its REC.
        tfs: The selected through focal series.
        true_paths: The true_paths for the files to be saved.
        file_choices: The files selected for ptie save, only necessary for manual.

    Returns:
        true_paths: The list containing the full path names.
    """

    # Comb through all input fields and pull current path name
    if event != '__REC_Save_TIE__':
        true_paths = []
        for i in range(1, num_paths + 1):
            true_paths.append(save_win[f'__save_win_filename{i}__'].Get())
    elif event == '__REC_Save_TIE__':
        save_choice = save_win['__save_rec_combo__'].Get()
        pref = save_win[f'__save_win_prefix__'].Get()
        if save_choice != '----':
            true_paths = []
            if save_choice == 'Color':
                stop = 2
            elif save_choice == 'Full Save':
                stop = 10
            elif save_choice == 'Mag. & Color':
                stop = 4
            elif save_choice == 'No Save':
                stop = 0
            elif save_choice == 'Manual':
                stop = None

            # Batch choose files to save
            if stop is not None:
                for i in range(stop):
                    true_paths.append(util.join([pref, str(defocus), orientations[i]], '_'))
            # Manually select the files to save and pass the orientations into true pqths
            else:
                save_win.Hide()
                file_choice_layout = file_choice_ly(tfs)
                file_choice_win = sg.Window('File Choice Window', file_choice_layout, size=(220, 300), element_justification="left",
                                            finalize=True, icon=get_icon())
                while True:
                    ev3, vals3 = file_choice_win.Read(timeout=400)
                    if ev3 == 'Exit' or ev3 is None or ev3 in ['fc_win_submit', 'fc_win_close']:
                        file_choices = [0]
                        item_list = ['color_b', 'byt', 'bxt',
                                     'bbt', 'dIdZ_e', 'dIdZ_m', 'inf_im',
                                     'phase_e', 'phase_b', 'arrow_colormap',
                                     'bw_arrow_colormap']
                        for key in item_list:
                            if tfs == 'Unflip/Flip' or (tfs == 'Single' and 'phase_e' not in key and 'dIdZ_e' not in key):
                                if file_choice_win[key].Get():
                                    file_choices.append(item_list.index(key) + 1)
                        if ev3 == 'fc_win_close' or len(file_choices) == 1:
                            file_choices = []
                        file_choice_win.Close()
                        save_win['__save_rec_combo__'].Update(value='----')
                        break

                for i in file_choices:
                    true_paths.append(f"{pref}_{defocus}_{orientations[i]}")
                save_win.UnHide()

            if save_choice in ['Mag. & Color', 'Full Save']:
                true_paths.append(util.join([pref, str(defocus), orientations[10]], '_'))
                true_paths.append(util.join([pref, str(defocus), orientations[11]], '_'))
        elif save_choice == '----':
            true_paths = []
            for i in file_choices:
                true_paths.append(f"{pref}_{defocus}_{orientations[i]}")

    return true_paths, file_choices


def run_save_window(winfo: Struct, event: str, image_dir: str,
                    orientations: Optional[Union[str, List[str]]] = None,
                    defocus: Optional[str] = None,
                    tfs: str = 'Unflip/Flip') -> Tuple[List[str], List[bool], Optional[Tuple[str, bool, str]]]:
    """Executes the save window.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        event: The key for the values dictionary that represents
            an event in the window.
        image_dir: The working directory where image will be saved
        orientations: List of the orientations or filetypes to categorize the saved
            file ('flip', 'unflip', 'stack', '').
        defocus: The value for the defocus if running reconstruction.
        tfs: The through focal series value chosen for LS or BUJ.

    Returns:
        filenames: The list of filenames to give the saved images.
        overwrite_signals: List of booleans of whether to overwrite files.
        prefix: The prefix for the tab to print to.
        save_tie: Boolean value of whether to execute saving of reconstructed images.
        im_dir: The image directory to save reconstructed images to.
    """

    # Create layout of save window
    file_choices = []
    vals = save_window_ly(event, image_dir, orientations, tfs=tfs, tie_prefix=winfo.rec_tie_prefix)
    window_layout, im_type, file_paths = vals[0:3]
    orientations, inputs = vals[3:]
    save_win = sg.Window('Save Window', window_layout, finalize=True, icon=get_icon())
    for key in inputs:
        save_win[key].Update(move_cursor_to='end')
        save_win[key].Widget.xview_moveto(1)

    winfo.save_win = save_win
    winfo.window.Hide()
    if winfo.output_window_active:
        winfo.output_window.Disappear()
        winfo.output_window.Hide()
    if event == '__REC_Save_TIE__':
        true_paths, file_choices = save_window_values(save_win, len(file_paths), event, orientations, defocus, tfs,
                                                      file_choices=file_choices)
    else:
        true_paths, file_choices = save_window_values(save_win, len(file_paths), event, orientations, defocus,
                                                      file_choices=file_choices)

    # Run save window event handler
    # Initiate event allows successful creation and reading of window
    overwrite_signals = []
    ev2 = 'Initiate'
    while True:
        if ev2 != 'Initiate':
            ev2, vals2 = save_win.Read(timeout=400)
        # Saving TIE images
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
            elif save_choice in ['Manual', '----']:
                save_tie = 'manual'

        # Getting full paths to the images and checking if they need to be overwritten
        filenames = []
        if ev2 and ev2 != 'Exit':
            true_paths, file_choices = save_window_values(save_win, len(file_paths), event, orientations, defocus, tfs,
                                                          file_choices=file_choices)
        if ev2 and 'TIMEOUT' not in ev2:
            overwrite_signals = check_overwrite(winfo, save_win, true_paths, orientations,
                                                image_dir, im_type, event, tfs)

        # Exit or save pressed
        if not ev2 or ev2 in ['Exit', '__save_win_save__']:
            if winfo.output_window_active:
                winfo.output_window.Reappear()
                winfo.output_window.UnHide()
            winfo.window.UnHide()
            save_win.Close()
            if ev2 == '__save_win_save__':
                for i in range(len(true_paths)):
                    if event == '__REC_Save_TIE__':
                        path = f'{image_dir}/images/{true_paths[i]}'
                    else:
                        path = f'{image_dir}/{true_paths[i]}'
                    filenames.append(path)
            break
        if ev2 == 'Initiate':
            ev2 = None

    # Return values based off saving reconstructed images or not.
    if event != '__REC_Save_TIE__':
        return filenames, overwrite_signals, None
    elif event == '__REC_Save_TIE__':
        return filenames, overwrite_signals, (prefix, save_tie, im_dir)


# -------------- Main Event Handler and run GUI --------------#
def event_handler(winfo: Struct, window: sg.Window) -> None:
    """ The event handler handles all button presses, mouse clicks, etc.
    that can take place in the app. It takes the SG window and the struct
    containing all window data as parameters.

    Args:
        winfo: The data structure holding all information about
            windows and GUI.
        window: The element representing the main GUI window.

    Returns:
        None
    """
    # Create output_window
    output_window = output_ly()
    output_window.Hide()

    # Initialize window, bindings, and event variables
    init(winfo, window, output_window)
    for key in winfo.keys['input']:
        window[key].Update(move_cursor_to='end')
        window[key].Widget.xview_moveto(1)
    set_pretty_focus(winfo, window, 'Window Click')

    # Set up the output log window and redirecting std_out
    log_line = 0
    with StringIO() as winfo.buf, StringIO() as winfo.fiji_buf:
        with redirect_stdout(winfo.buf):

            # Run event loop
            bound_click = True
            # bound_scroll = False
            close = None
            while True:
                # Capture events
                event, values = window.Read(timeout=50)

                # Break out of event loop
                if event is None or close == 'close' or event == 'Exit::Exit1':  # always,  always give a way out!

                    if winfo.ptie_init_thread is not None:
                        if winfo.ptie_init_thread.is_alive():
                            winfo.ptie_init_thread.join(0.1)
                    if winfo.ptie_recon_thread is not None:
                        if winfo.ptie_recon_thread.is_alive():
                            winfo.ptie_recon_thread.join(0.1)
                        winfo.ptie_recon_thread = None
                    output_window.close()
                    window.close()
                    break

                if event in ['Show (Control-l)::Log', 'Show Log'] and not winfo.output_window_active:
                    winfo.output_window_active = True
                    output_window.Reappear()
                    output_window.UnHide()
                elif event in ['Show (Control-l)::Log', 'Show Log'] and winfo.output_window_active:
                    output_window.BringToFront()
                elif event in ['Hide (Control-h)::Log', 'Hide Log'] and winfo.output_window_active:
                    winfo.output_window_active = False
                    output_window.Hide()
                    output_window.Disappear()
                    set_pretty_focus(winfo, window, 'Window Click')

                # About section was opened
                if event == 'Documentation::Documentation':
                    try:
                        webbrowser.open('https://pylorentztem.readthedocs.io/en/latest/')
                    except:
                        print('*** ATTEMPT TO ACCESS ABOUT PAGE FAILED ***')
                        print('*** CHECK INTERNET CONNECTION ***')

                # Disable window clicks if creating mask or setting subregion
                if ((winfo.true_element == '__REC_Graph__' and bound_click and
                     window['__REC_Mask__'].metadata['State'] == 'Set')):
                    window.TKroot.unbind("<Button-1>")
                    bound_click = False
                elif (not bound_click and winfo.true_element != '__REC_Graph__'):
                    winfo.window.bind("<Button-1>", 'Window Click')
                    bound_click = True

                # Make sure input element that just display names can't be typed in
                if event in winfo.keys['read_only_inputs']:
                    state = window[event].metadata['State']
                    text = window[event].metadata[state]
                    window[event].update(value=text)

                # Set cursor for the transformation variables
                if window['__REC_Mask__'].metadata['State'] == 'Set':
                    # Get focus if the cursor is in an element
                    chosen_element = winfo.true_input_element
                    if chosen_element in ["__REC_transform_y__", "__REC_transform_x__", "__REC_transform_rot__",
                                          "__REC_Mask_Size__"]:
                        winfo.cursor_pos = window[chosen_element].Widget.index(tk_insert)
                    else:
                        winfo.cursor_pos = 'end'

                # Check which tab is open and execute events regarding that tab
                current_tab = winfo.current_tab = get_open_tab(winfo, winfo.pages, event)
                if current_tab == "home_tab":
                    run_home_tab(winfo, window, event, values)
                elif current_tab == "reconstruct_tab":
                    run_reconstruct_tab(winfo, window, current_tab, event, values)

                # Show loading spinners if necessary def init
                if winfo.ptie_init_thread is not None:
                    if (window['__REC_FLS_Spinner__'].metadata['State'] == 'Def' and
                            not winfo.ptie_init_spinner_active):
                        activate_spinner(window, '__REC_FLS_Spinner__')
                        winfo.ptie_init_spinner_active = True
                elif winfo.ptie_init_thread is None:
                    if window['__REC_FLS_Spinner__'].metadata['State'] == 'Set':
                        deactivate_spinner(window, '__REC_FLS_Spinner__')
                        winfo.ptie_init_spinner_active = False
                if winfo.ptie_recon_thread is not None:
                    if (window['__REC_PYTIE_Spinner__'].metadata['State'] == 'Def' and
                            not winfo.ptie_recon_spinner_active):
                        activate_spinner(window, '__REC_PYTIE_Spinner__')
                        winfo.ptie_recon_spinner_active = True
                elif winfo.ptie_recon_thread is None:
                    if window['__REC_PYTIE_Spinner__'].metadata['State'] == 'Set':
                        deactivate_spinner(window, '__REC_PYTIE_Spinner__')
                        winfo.ptie_recon_spinner_active = False

                # Update loading spinners spinners
                active_spinners = []
                if winfo.ptie_init_spinner_active:
                    active_spinners.append('__REC_FLS_Spinner__')
                if winfo.ptie_recon_spinner_active:
                    active_spinners.append('__REC_PYTIE_Spinner__')
                for spinner_key in active_spinners:
                    spinner_fn = window[spinner_key].metadata['Set']
                    window[spinner_key].UpdateAnimation(spinner_fn)

                # Set the focus of the GUI to reduce interferences
                set_pretty_focus(winfo, window, event)

                # Copying and pasting text from output windows
                if winfo.output_window_active:
                    output_event, output_values = output_window.Read(timeout=0)
                    if output_event in ['MAIN_OUTPUT+FOCUS_IN+', 'FIJI_OUTPUT+FOCUS_IN+']:
                        key = output_event[:-10]
                        widget = winfo.output_window[key].Widget
                        widget.bind("<1>", widget.focus_set())
                        disable_elements(output_window, [key])
                        winfo.output_focus_active = True
                        winfo.active_output_focus_el = key
                    elif output_event in ['MAIN_OUTPUT+FOCUS_OUT+', 'FIJI_OUTPUT+FOCUS_OUT+']:
                        key = output_event[:-11]
                        widget = winfo.output_window[key].Widget
                        widget.unbind("<1>")
                        enable_elements(winfo, output_window, [key])
                        winfo.output_focus_active = False
                        winfo.active_output_focus_el = None
                        if event != '__TIMEOUT__' and 'HOVER' not in event:
                            winfo.invis_graph.SetFocus()
                    elif output_event in ['output_tabgroup'] and winfo.output_focus_active:
                        key = winfo.active_output_focus_el
                        widget = winfo.output_window[key].Widget
                        widget.unbind("<1>")
                        enable_elements(winfo, output_window, [key])
                        winfo.output_focus_active = False
                        winfo.active_output_focus_el = None
                    if output_event in ['MAIN_OUTPUT_AUTOSCROLL']:
                        autoscroll_state = output_window[output_event].get()
                        if 'MAIN_OUTPUT' in output_event:
                            key = 'MAIN_OUTPUT'
                        output_window[key].Update(autoscroll=autoscroll_state)
                    if output_event in ['MAIN_OUTPUT_HIDE', 'Output Hide Log']:
                        winfo.output_window_active = False
                        output_window.Hide()
                        output_window.Disappear()
                        set_pretty_focus(winfo, window, 'Window Click')
                    elif '__TIMEOUT__' not in output_event:
                        set_pretty_focus(winfo, output_window, output_event)

                sys_stdout.flush()
                output_txt = winfo.buf.getvalue().split('\n')
                i = 0
                for line in output_txt:
                    if log_line <= i:
                        log_line = i
                        if not(line.isspace() or line.strip() == ''):
                            if (not (line.startswith('REC') or line.startswith('HOM') or
                                      line.startswith('***')) and line != '\n'):
                                line = 'REC: ' + line
                            output_window['MAIN_OUTPUT'].update(value=f'{line}\n', append=True)
                    i += 1


def run_GUI() -> None:
    """Main run function. Takes in the style and defaults for GUI."""

    # Create the layouts
    DEFAULTS = defaults()
    sg.theme('BlueMono')
    background_color = sg.theme_background_color()
    sg.SetOptions(margins=(0, 0), element_padding=((0, 0), (0, 0)),
                  border_width=0, font=('Times New Roman', '16'))
    sys_layout = [[sg.Text('')]]
    scaling_window = sg.Window('Window Title', sys_layout, alpha_channel=0, no_titlebar=True,
                                finalize=True, icon=get_icon())
    scaling_window.TKroot.tk.call('tk', 'scaling', 1)
    scaling_window.close()
    window = window_ly(background_color, DEFAULTS)

    # This snippet allows use of menu bar without having to switch windows first on Mac
    if platform() == 'Darwin':
        subprocess.call(["/usr/bin/osascript", "-e", 'tell app "Finder" to set frontmost of process "Finder" to true'])
        subprocess.call(["/usr/bin/osascript", "-e", 'tell app "Finder" to set frontmost of process "python" to true'])

    # Create data structure to hold variables about GUI and reconstruction.
    winfo = Struct()

    # Event handling
    event_handler(winfo, window)


if __name__ == '__main__':
    run_GUI()

