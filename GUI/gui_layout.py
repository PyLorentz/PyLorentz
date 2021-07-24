"""Functions for GUI layout building.

These functions build the layout of the GUI. It controls what elements are available and
where they are placed in the GUI. Styling controls the overall look and feel of the
elements within this space.

AUTHOR:
Timothy Cote, ANL, Fall 2019.
"""

import os
from sys import platform
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import PySimpleGUI as sg
from PySimpleGUI import Menu, Tab, Window
from util import join
from gui_styling import pad, WindowStyle, get_icon


# ---------------------------------------------------- #
#                      Element Keys                    #
# ---------------------------------------------------- #
def element_keys() -> Dict[str, List[str]]:
    """Keep track and return the element keys that need bindings to them in the GUI"""

    button_keys = ["__Browser_Browse__", "__Browser_Set__", "__Browser_Reset__",

                   "__REC_Image_Dir_Browse__", "__REC_Set_Img_Dir__", "__REC_Reset_Img_Dir__",
                   "__REC_Save_TIE__", "__REC_Run_TIE__",
                   "__REC_Erase_Mask__", "__REC_Mask__", "__REC_Arrow_Set__",
                   "__REC_Load_FLS2__", "__REC_Load_FLS1__", "__REC_Reset_FLS__", "__REC_Load_Stack__",
                   "__REC_Set_FLS__"
                   ]
    checkbox_keys = ["__REC_Symmetrize__"]
    combo_keys = ["__REC_Def_Combo__", "__REC_FLS_Combo__", "__REC_TFS_Combo__",
                  "__REC_Derivative__", "__REC_Colorwheel__", "__REC_Arrow_Color__"]
    graph_keys = ["__REC_Graph__", "__REC_Colorwheel_Graph__",
                  "__invisible_graph__"]
    input_keys = ["__Browser_Path__",

                  "__REC_Image_Dir_Path__", "__REC_Image__", "__REC_QC_Input__",
                  "__REC_transform_y__", "__REC_transform_x__",
                  "__REC_transform_rot__", "__REC_Mask_Size__", "__REC_Arrow_Num__",
                  "__REC_Arrow_Len__", "__REC_Arrow_Wid__",
                  "__REC_FLS1__", "__REC_FLS2__", "__REC_Stack__", "__REC_Stack_Stage__",
                  "__REC_FLS1_Staging__", "__REC_FLS2_Staging__", "__REC_M_Volt__"
                  ]
    listbox_keys = ["__REC_Image_List__", "__REC_Def_List__"]
    radio_keys = ["__REC_Square_Region__", "__REC_Rectangle_Region__"
                  ]
    slider_keys = ["__REC_Slider__", "__REC_Image_Slider__", "__REC_Defocus_Slider__"]
    tab_keys = ["reconstruct_tab", "home_tab"]
    tabgroup_keys = ["pages_tabgroup"]
    text_keys = ["home_title", "home_version", "home_authors", "home_readme", "home_contact"
                 ,
                 '__REC_FLS2_Text__', '__REC_FLS2_Text__', '__REC_Mask_Text__']
    read_only_inputs_keys = ["__REC_Image__"
                             ]
    keys = {'button': button_keys,
            'checkbox': checkbox_keys,
            'combo': combo_keys,
            'graph': graph_keys,
            'input': input_keys,
            'read_only_inputs': read_only_inputs_keys,
            'listbox': listbox_keys,
            'radio': radio_keys,
            'slider': slider_keys,
            'tab': tab_keys,
            'tabgroup': tabgroup_keys,
            'text': text_keys}

    return keys


# ---------------------------------------------------- #
#                        Menu Bar                      #
# ---------------------------------------------------- #
def menu_bar() -> Menu:
    """Return the menu bar layout."""

    menu_def = [['PyLorentz', ['Documentation::Documentation', 'Exit::Exit1']],
                ['Log', ['Show (Control-l)::Log', 'Hide (Control-h)::Log']]]
    return sg.Menu(menu_def, font='Times 15')


# ---------------------------------------------------- #
#                        Home Tab                      #
# ---------------------------------------------------- #
def home_tab(style: WindowStyle, DEFAULTS: Dict) -> Tab:
    """The home tab layout.

    Args:
        style: The class that holds all style data for the window.
            Look at gui_styling.py for more info.
        DEFAULTS: The default values for certain style elements
            such as font

    Returns:
        tab: The layout for the hometab.
    """

    title = '''PyLorentz Phase Reconstruction'''
    version = '''Version 1.2.0'''
    authors = '''Authors: Tim Cote, Arthur McCray, CD Phatak'''
    readme = '''PyLorentz is a codebase designed for analyzing Lorentz Transmission Electron Microscopy (LTEM) data. There are two primary features and functions:

* PyTIE – Reconstructing the magnetic induction from LTEM images using the Transport of Intensity Equation (TIE).

* SimLTEM – Simulating phase shift and LTEM images from a given magnetization. (Unavailable in GUI)
                
                
This GUI should be used as an alternative to the Jupyter Notebook for some improvments in region selection and control over saving images in PyTIE.

'''
    run_color_bg = '#0b7a29'
    no_run_color_bg = '#b00707'
    fg_color = '#f5f5f5'

    browser_button = sg.FolderBrowse("Browse", key="__Browser_Browse__", pad=pad(164, 0, 10, 0),
                                     target='__Browser_Path__')
    browser_def_text = sg.Text('You can set the default browser working directory below.', pad=pad(230, 0, 10, 0))
    browser_wd_input = sg.Input(DEFAULTS['browser_dir'], key='__Browser_Path__', pad=pad(16, 0, 10, 0), size=(30, 1))
    if DEFAULTS['browser_dir']:
        browser_set_color = (fg_color, no_run_color_bg)
        browser_reset_color = (fg_color, run_color_bg)
    else:
        browser_set_color = (fg_color, run_color_bg)
        browser_reset_color = (fg_color, no_run_color_bg)
    browser_def_set = sg.Button('Set as Default', key='__Browser_Set__', pad=pad(10, 0, 10, 0),
                                button_color=browser_set_color,
                                metadata={'Set': (fg_color, run_color_bg), 'Def': (fg_color, no_run_color_bg)})
    browser_def_reset = sg.Button('Reset Default',  key='__Browser_Reset__', pad=pad(10, 0, 10, 0),
                                  button_color=browser_reset_color,
                                  metadata={'Set': (fg_color, run_color_bg), 'Def': (fg_color, no_run_color_bg)})

    layout = [[sg.Col([[sg.T(title, **style.styles('home_title'))],
                       [sg.T(version, **style.styles('home_version'))],
                       [sg.T(authors, **style.styles('home_authors'))],
                       [sg.Multiline(readme, **style.styles('home_readme'))],
                       [browser_def_text],
                       [browser_button, browser_wd_input, browser_def_set, browser_def_reset]])]]
    metadata = {"parent_tabgroup": "pages_tabgroup",
                "child_tabgroup": None}

    # Set the size of all tabs here as well!
    home_layout = [[sg.Col(layout, size=style.tab_size)]]
    tab = sg.Tab('Home', home_layout, font=style.fonts['tab'], key="home_tab", metadata=metadata)
    return tab


# ---------------------------------------------------- #
#                    Reconstruct Tab                   #
# ---------------------------------------------------- #
def reconstruct_tab(style: WindowStyle, DEFAULTS: Dict) -> Tab:
    """The reconstruction tab layout.

    Args:
        style: The class that holds all style data for the window.
            Look at gui_styling.py for more info.
        DEFAULTS: The default values for certain style elements
            such as font.

    Returns:
        tab: The layout for the reconstruction tab.
    """

    def layout_reconstruct_tab() -> List[List[Any]]:
        """The overall reconstruction tab layout.

        Returns:
            The layout for the overall reconstruction tab.
        """

        load_data_frame = [[sg.Text('Stack:', pad=((5, 0), (5, 0))),
                            sg.Input('None', **style.styles('__REC_Stack_Stage__')),
                            sg.Input('None', **style.styles('__REC_Stack__')),
                            sg.FileBrowse("Load", **style.styles('__REC_Load_Stack__'))],
                            [sg.Text("FLS Files:", pad=((21, 0), (10, 0))),
                             sg.Combo(['Two', 'One'], **style.styles('__REC_FLS_Combo__')),
                             sg.Text("TFS:", pad=((35, 0), (10, 0))),
                             sg.Combo(['Unflip/Flip', 'Single'], **style.styles('__REC_TFS_Combo__'))
                             ],
                           [sg.FileBrowse("Load", **style.styles('__REC_Load_FLS1__')),
                            sg.Input('None', **style.styles('__REC_FLS1_Staging__')),
                            sg.Input("None", **style.styles('__REC_FLS1__')),
                            sg.Text("Unflip FLS", **style.styles('__REC_FLS1_Text__'))
                            ],
                           [sg.FileBrowse("Load", **style.styles('__REC_Load_FLS2__')),
                            sg.Input('None', **style.styles('__REC_FLS2_Staging__')),
                            sg.Input("None",  **style.styles('__REC_FLS2__')),
                            sg.Text("Flip FLS", **style.styles('__REC_FLS2_Text__'))],
                           [sg.Col([[sg.Text('Defocus Values:', pad=((35, 0), (8, 0)))],
                                   [sg.Listbox(['None'], **style.styles('__REC_Def_List__')),
                                    sg.Slider(**style.styles('__REC_Defocus_Slider__'))]]),
                            sg.Col([[sg.Text('Microscope Voltage:', pad=((20, 0), (8, 0)))],
                                    [sg.Input('200', **style.styles('__REC_M_Volt__')),
                                     sg.Text('kV', pad=((13, 0), (10, 0)), font="Times 36")]]),
                            ],
                           [sg.Button('Set', **style.styles('__REC_Set_FLS__')),
                            sg.Button('Reset', **style.styles('__REC_Reset_FLS__')),
                            sg.Image(**style.styles('__REC_FLS_Spinner__'))
                            ]]

        subregion_frame = [[sg.Text('Unset', key='__REC_Mask_Text__', background_color='#F1FDFC',
                                    justification='center', font='Times 19', pad=((10, 0), (4, 4))),
                            sg.Button('Select Region', **style.styles('__REC_Mask__')),
                            sg.Button('Reset', **style.styles('__REC_Erase_Mask__')),
                            ],
                           [sg.Col([[sg.Text('Rotation:', pad=((50, 0), (0, 0))),
                                     sg.Input('0', **style.styles('__REC_transform_rot__')),
                                     sg.Text(u'\N{DEGREE SIGN}')],
                                    [sg.Text('X shift:', pad=((61, 0), (0, 0))),
                                     sg.Input('0', **style.styles('__REC_transform_x__')),
                                     sg.Text('px')],
                                    [sg.Text('Y shift:', pad=((61, 0), (0, 0))),
                                     sg.Input('0', **style.styles('__REC_transform_y__')),
                                     sg.Text('px')]]),
                            sg.Col([[sg.Text('Type:           Size:', pad=((40, 29), (5, 0)))],
                                    [sg.Radio('Square', 'region_select', **style.styles('__REC_Square_Region__'),
                                               pad=((20, 0), (0, 0))),
                                     sg.Input('50', **style.styles('__REC_Mask_Size__')),
                                     sg.Text('%', pad=((2, 0), (0, 0)))],
                                    [sg.Radio('Rectangle', 'region_select', **style.styles('__REC_Rectangle_Region__'),
                                     pad=((20, 0), (0, 0)))]
                                    ])
                            ]]

        TIE_menu = [[sg.Col([[sg.Text('Defocus:', pad=((30, 0), (4, 0))),
                              sg.Combo(['None'], **style.styles('__REC_Def_Combo__'))],
                             [sg.Text("QC value:", pad=((22, 0), (5, 0))),
                              sg.Input('0.00', **style.styles('__REC_QC_Input__')),
                              sg.Text('Symmetrize:', pad=((12, 0), (5, 0))),
                              sg.Checkbox('', **style.styles('__REC_Symmetrize__'))],
                             [sg.Text('Derivative:', pad=((16, 0), (5, 0))),
                              sg.Combo(['Central Diff.', 'Longitudinal Deriv.'],
                                        **style.styles('__REC_Derivative__'))]]),
                     sg.Col([[sg.Button('Run', **style.styles('__REC_Run_TIE__'))],
                             [sg.Button('Save', **style.styles('__REC_Save_TIE__'))]]),
                     sg.Col([[sg.Image(**style.styles('__REC_PYTIE_Spinner__'))]])],
                             [sg.Text('Colorwheel:', pad=((8, 0), (7, 0))),
                              sg.Combo(['HSV', '4-Fold'], **style.styles('__REC_Colorwheel__')),
                              sg.Button('Set Vector Im.', key='__REC_Arrow_Set__', pad=((60, 0), (7, 0)))],
                             [sg.Text('Vector Im. |', pad=((2, 0), (7, 0))),
                              sg.Text('Arrows:', pad=((0, 0), (7, 0))),
                              sg.Input('15', **style.styles('__REC_Arrow_Num__')),
                              sg.Text('Color:', pad=((4, 0), (7, 0))),
                              sg.Combo(['On', 'Off'], **style.styles('__REC_Arrow_Color__')),
                              sg.Text('L:', pad=((2, 0), (7, 0))),
                              sg.Input('1', **style.styles('__REC_Arrow_Len__')),
                              sg.Text('W:', pad=((2, 0), (7, 0))),
                              sg.Input('1', **style.styles('__REC_Arrow_Wid__'))],
                             [sg.HorizontalSeparator(color='black', pad=(5, 5))],
                             [sg.Text('Images:', pad=((64, 0), (0, 0))),
                              sg.Listbox(['Stack', 'Color', 'Vector Im.', 'MagX', 'MagY', 'Mag. Magnitude',
                                          'Electr. Phase', 'Mag. Phase', 'Electr. Deriv.', 'Mag. Deriv.',
                                          'In Focus', 'Loaded Stack'],
                                          **style.styles('__REC_Image_List__')),
                              sg.Slider(**style.styles('__REC_Image_Slider__'))]]

        reconstruct_graph = [[sg.Text("Image Directory:",  pad=((70, 0), (5, 0))),
                              sg.Input(DEFAULTS['browser_dir'], **style.styles('__REC_Image_Dir_Path__', DEFAULTS['browser_dir'])),
                              sg.FolderBrowse("Browse", **style.styles('__REC_Image_Dir_Browse__')),
                              sg.Button('Set', **style.styles('__REC_Set_Img_Dir__')),
                              sg.Button('Reset', **style.styles('__REC_Reset_Img_Dir__'))],
                             [sg.Slider((0, 0), **style.styles('__REC_Slider__')),
                              sg.Graph((672, 672), (0, 0), (671, 671), **style.styles('__REC_Graph__')),
                              sg.Graph((70, 70), (0, 0), (69, 69), **style.styles('__REC_Colorwheel_Graph__'))
                              ]]

        files_column = [[sg.Input('None', **style.styles('__REC_Image__'))]]

        right_panel = sg.Col([[sg.Column(reconstruct_graph)],
                               [sg.Column(files_column)]])
        left_panel = sg.Col([[sg.Frame("Load Data", load_data_frame, relief=sg.RELIEF_SUNKEN,
                                       pad=((8, 0), (1, 1)), font=('Times New Roman', 18))],
                             [sg.Frame("Region Select", subregion_frame, relief=sg.RELIEF_SUNKEN,
                                       pad=((8, 0), (1, 1)), font=('Times New Roman', 18))],
                             [sg.Frame("TIE", TIE_menu, relief=sg.RELIEF_SUNKEN,
                                       pad=((8, 0), (1, 1)), font=('Times New Roman', 18))]],
                            key='__REC_Scrollable_Column__',
                            scrollable=False)

        return [[left_panel, right_panel]]

    reconstruct_layout = layout_reconstruct_tab()
    metadata = {"parent_tabgroup": "pages_tabgroup",
                "child_tabgroup": None}
    tab = sg.Tab('Phase Reconstruction', reconstruct_layout, font=style.fonts['tab'], key="reconstruct_tab",
                 metadata=metadata)
    return tab


# ---------------------------------------------------- #
#                         Windows                      #
# ---------------------------------------------------- #
def window_ly(background_color: str, DEFAULTS: Dict) -> Window:
    """The full window layout.

    Args:
        background_color: The background color for the window.
        DEFAULTS: The default values for certain style elements
            such as font.

    Returns:
        window: The window element of the window.
    """

    style = WindowStyle(background_color)

    menu = menu_bar()
    home_pg = home_tab(style, DEFAULTS)
    reconstruct_pg = reconstruct_tab(style, DEFAULTS)
    pages = sg.TabGroup([[home_pg, reconstruct_pg]], tab_location='topleft', theme=sg.THEME_CLASSIC,
                        enable_events=True, key="pages_tabgroup")
    invisible_graph = sg.Graph((0, 0), (0, 0), (0, 0), visible=True, key="__invisible_graph__")

    icon = get_icon()
    window_layout = [[menu], [invisible_graph, pages]]
    window = sg.Window('PyLorentz', window_layout, return_keyboard_events=True, default_element_size=(12, 1),
                       resizable=True, size=(style.window_width, style.window_height), use_default_focus=False,
                       finalize=True, icon=icon)
    return window


def save_window_ly(event: str, image_dir: str,
                   orientations: Optional[Union[List[str], str]],
                   tfs: Optional[str] = None, tie_prefix: str = 'Example') -> List[List[Any]]:
    """Initializes save window.

    Args:
        event: The event key that was passed to the save window.
        image_dir: The image parent directory/current working directory.
        orientations: List of the strings of the unflip or flip or tfs orientations.
            Otherwise may be the prefix for REC or None.
        tfs: The string of the tfs selected value.
        tie_prefix: The prefix label for the name of the images to
            be saved from the reconstruction.

    Returns:
        layout: The layout for the save window.
        im_type: The image file type.
        file_paths: List of the file paths.
        orientations: List of strings of the orientations or image names.
        inputs: The keys of the inputs.
    """

    # Change parameters to suit what is being saved
    if orientations is None:
        orientations = ['']
    if event == '__REC_Save_TIE__':
        im_type = 'Reconstructed Images'
        orientations = ['recon_params.txt', 'color_b.tiff', 'byt.tiff', 'bxt.tiff',
                        'bbt.tiff', 'dIdZ_e.tiff', 'dIdZ_m.tiff', 'inf_im.tiff',
                        'phase_e.tiff', 'phase_b.tiff', 'arrow_colormap.png',
                        'bw_arrow_colormap.png']
        file_paths = ['images']

    # Define the layout for the save window
    if event != '__REC_Save_TIE__':
        col1 = [[sg.Text(f'Choose your {im_type} filename(s): ', pad=((5, 0), (10, 0)))]]
    elif event == '__REC_Save_TIE__':
        col1 = [[sg.Text(f'Choose your {im_type} prefix: ', pad=((5, 0), (10, 0)))]]
    col2 = [[sg.Text('Overwrite', pad=((0, 10), (10, 0)))]]

    # Add extra padding for specific variables
    size = (70, 1)
    for i in range(1, len(file_paths)+1):
        orient = orientations[i-1]
        if not orient:
            orient = 'file'
        if orient == 'file':
            pad = ((10, 0), (10, 0))
            x_pad = 47
        elif orient == 'flip':
            pad = ((21, 0), (10, 0))
            x_pad = 59
        elif orient == 'unflip':
            pad = ((5, 0), (10, 0))
            x_pad = 59
        elif im_type == 'Reconstructed Images':
            pad1 = ((5, 0), (10, 0))
            pad2 = ((22, 0), (10, 0))
            x_pad = 138
            size = (70, 10)
        else:
            x_pad = 49
            pad = ((5, 0), (10, 0))

        # Create on input fields based off of which files are being saved.
        inputs = []
        if event != '__REC_Save_TIE__':
            inp_key = f'__save_win_filename{i}__'
            col1 += [[sg.Text(f'{orient}:', pad=pad),
                      sg.Input(f'{file_paths[i-1]}', key=inp_key, enable_events=True,
                               size=(70, 1), pad=((5, 10), (10, 0)))]]
            col2 += [[sg.Checkbox('', key=f'__save_win_overwrite{i}__', pad=((28, 0), (10, 0)), enable_events=True)]]
            inputs.append(inp_key)
        elif event == '__REC_Save_TIE__':
            col1 += [[sg.Text('Working Directory:', pad=pad1),
                      sg.Input(f'{image_dir}', key=f'__save_win_wd__', size=(65, 1),
                               use_readonly_for_disable=True, disabled=True,
                               disabled_readonly_background_color='#A7A7A7',
                               pad=((0, 10), (10, 0)))],
                     [sg.Text('Image Directory:', pad=pad2),
                      sg.Input(f'images', key=f'__save_win_filename1__', size=(65, 1),
                               enable_events=True, use_readonly_for_disable=True, disabled=True,
                               disabled_readonly_background_color='#A7A7A7',
                               pad=((0, 10), (10, 0)))]
                     ]
            inputs.extend(['__save_win_wd__', '__save_win_filename1__'])
            col1 += [[sg.Text(f'prefix:', pad=((89, 0), (10, 0))),
                      sg.Input(f'{tie_prefix}', key=f'__save_win_prefix__', size=(30, 1), enable_events=True,
                               pad=((0, 0), (10, 0))),
                      sg.Combo(['Manual', 'Color', 'Full Save', 'Mag. & Color', 'No Save', '----'], key='__save_rec_combo__',
                                enable_events=True, size=(12, 1), default_value='Color',
                                readonly=True, pad=((20, 0), (10, 0)))]]
            inputs.extend(['__save_win_prefix__'])
            col2 += [[sg.Checkbox('', key=f'__save_win_overwrite{i}__', pad=((28, 0), (10, 0)), default=True,
                                  enable_events=True)]]
    # Create buttons to define whether to check if paths exists, exit, or save info
    col1 += [[sg.Button('Exit', pad=((x_pad, 0), (10, 5))),
              sg.Button('Save', key='__save_win_save__', pad=((5, 0), (10, 5)), disabled=True)],
             [sg.Multiline('', visible=True, key='__save_win_log__', size=size, disabled=True,
                           pad=((x_pad, 0), (0, 15)))]]
    layout = [[sg.Col(col1), sg.Col(col2)]]
    return layout, im_type, file_paths, orientations, inputs


def file_choice_ly(tfs: str) -> List[List[Any]]:
    """Creates the file choice window for saving reconstructed windows.

    Args:
        tfs: The type of through focal series (Single or Unflip/FLip)

    Returns:
        window: The layout for the file choice window."""

    if tfs == 'Unflip/Flip':
        orientations = ['color_b', 'byt', 'bxt',
                        'bbt', 'dIdZ_e', 'dIdZ_m', 'inf_im',
                        'phase_e', 'phase_b', 'arrow_colormap',
                        'bw_arrow_colormap']
    elif tfs == 'Single':
        orientations = ['color_b', 'byt', 'bxt',
                        'bbt', 'dIdZ_m', 'inf_im',
                        'phase_b', 'arrow_colormap',
                        'bw_arrow_colormap']
    file_choice_window = []
    for orientation in orientations:
        file_choice_window.append([sg.Checkbox(orientation, key=orientation)])

    file_choice_window.append([sg.Button('Submit', key='fc_win_submit', pad=(20, 10), enable_events=True),
                               sg.Button('Close', key='fc_win_close', pad=(5, 10), enable_events=True)])
    return file_choice_window


def output_ly() -> Window:
    """Creates the output log layout.

    Returns:
        window: The PySimpleGUI window element of the output window."""

    invisible_graph = sg.Graph((0, 0), (0, 0), (0, 0), visible=True, key="__output_invis_graph__")
    main_output = sg.Tab('Main', [[sg.Multiline('', key='MAIN_OUTPUT',
                                                write_only=True, size=(600, 16), autoscroll=True
                                                )],
                                   [sg.Text('Autoscroll', pad=(20, 2)),
                                    sg.Checkbox('', key='MAIN_OUTPUT_AUTOSCROLL', default=True,
                                                enable_events=True),
                                    sg.Button('Hide', pad=(20, 2), key='MAIN_OUTPUT_HIDE')]])
    pages = sg.TabGroup([[main_output]], tab_location='topleft',
                          theme=sg.THEME_CLASSIC,
                          enable_events=True, key="output_tabgroup")
    window_layout = [[invisible_graph], [pages]]
    icon = get_icon()
    window = sg.Window('Log', window_layout, default_element_size=(12, 1), disable_close=True,
                       resizable=True, size=(600, 350), use_default_focus=False, alpha_channel=0,
                       finalize=True, icon=icon)
    return window
