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

    button_keys = ["__Fiji_Browse__", "__Browser_Browse__", "__Fiji_Set__", "__Fiji_Reset__",
                   "__Fiji_Def_Set__", "__Fiji_Def_Reset__",

                   "__LS_Image_Dir_Browse__", "__LS_Load_FLS1__", "__LS_Load_FLS2__",
                   "__LS_Set_Img_Dir__", "__LS_Reset_Img_Dir__", "__LS_Run_Align__", "__LS_Adjust__",
                   "__LS_View_Stack__", "__LS_Set_FLS__",
                   "__LS_Reset_FLS__",

                   "__BUJ_Image_Dir_Browse__", "__BUJ_Set_FLS__", "__BUJ_Load_FLS1__",
                   "__BUJ_Load_FLS2__", "__BUJ_Reset_FLS__",
                   "__BUJ_Set_Img_Dir__", "__BUJ_Reset_Img_Dir__",
                   "__BUJ_Load_Unflip_Stack__", "__BUJ_Load_Flip_Stack__",
                   "__BUJ_Elastic_Align__", "__BUJ_Unflip_Align__", "__BUJ_Flip_Align__",
                   "__BUJ_Adjust__", "__BUJ_View__", "__BUJ_Make_Mask__", "__BUJ_Reset_Mask__",
                   "__BUJ_Load_Mask__",
                   "__BUJ_Clear_Unflip_Mask__", "__BUJ_Clear_Flip_Mask__",

                   "__REC_Image_Dir_Browse__", "__REC_Set_Img_Dir__", "__REC_Reset_Img_Dir__",
                   "__REC_Save_TIE__", "__REC_Run_TIE__",
                   "__REC_Erase_Mask__", "__REC_Mask__", "__REC_Arrow_Set__",
                   "__REC_Load_FLS2__", "__REC_Load_FLS1__", "__REC_Reset_FLS__", "__REC_Load_Stack__",
                   "__REC_Set_FLS__"
                   ]
    checkbox_keys = ["__LS_horizontal_flip__", "__LS_interp__",
                     "__BUJ_horizontal_flip__", "__BUJ_LS_interp__", "__BUJ_filter__",
                     "__REC_Symmetrize__"]
    combo_keys = ["__LS_exp_transf__","__LS_FLS_Combo__", "__LS_TFS_Combo__",
                  "__BUJ_reg_mode__", "__BUJ_init_def__", "__BUJ_final_def__",
                  "__BUJ_FLS_Combo__", "__BUJ_TFS_Combo__",
                  "__BUJ_LS_exp_transf__", "__BUJ_exp_transf__", "__BUJ_Mask_View__",
                  "__REC_Def_Combo__", "__REC_FLS_Combo__", "__REC_TFS_Combo__",
                  "__REC_Derivative__", "__REC_Colorwheel__", "__REC_Arrow_Color__"]
    column_keys = ["__BUJ_Load_Mask_Col__"]
    graph_keys = ["__LS_Graph__",
                  "__BUJ_Graph__",
                  "__REC_Graph__", "__REC_Colorwheel_Graph__",
                  "__invisible_graph__"]
    input_keys = ["__Fiji_Path__", "__Browser_Path__", "__LS_Image_Dir_Path__", "__LS_igb__",
                  "__LS_spso__", "__LS_min_im__", "__LS_max_im__", "__LS_fds__", "__LS_fdob__",
                  "__LS_cncr__", "__LS_max_al_err__", "__LS_inlier_rat__", "__LS_transform_x__",
                  "__LS_transform_y__", "__LS_transform_rot__",
                  "__LS_FLS1__", "__LS_FLS1_Staging__", "__LS_FLS2_Staging__", "__LS_FLS2__",
                  "__LS_Image1__", "__LS_Image2__", "__LS_Stack__",

                  "__BUJ_Unflip_Stage_Load__", "__BUJ_Flip_Stage_Load__",
                  "__BUJ_LS_igb__", "__BUJ_LS_spso__", "__BUJ_LS_min_im__", "__BUJ_LS_max_im__",
                  "__BUJ_LS_fds__", "__BUJ_LS_fdob__", "__BUJ_LS_cncr__", "__BUJ_LS_max_al_err__",
                  "__BUJ_LS_inlier_rat__", "__BUJ_igb__", "__BUJ_spso__", "__BUJ_min_im__",
                  "__BUJ_max_im__", "__BUJ_fds__", "__BUJ_fdob__", "__BUJ_cncr__",
                  "__BUJ_max_al_err__", "__BUJ_inlier_rat__", "__BUJ_min_num_inlier__",
                  "__BUJ_FLS1__", "__BUJ_FLS1_Staging__", "__BUJ_FLS2_Staging__", "__BUJ_FLS2__",
                  "__BUJ_Image_Dir_Path__", "__BUJ_div_w__", "__BUJ_curl_w__",
                  "__BUJ_land_w__", "__BUJ_img_w__", "__BUJ_cons_w__", "__BUJ_stop_thresh__",
                  "__BUJ_transform_x__", "__BUJ_transform_y__", "__BUJ_transform_rot__",
                  "__BUJ_Image1__", "__BUJ_Image2__", "__BUJ_Stack__",
                  "__BUJ_Unflip_Stack_Inp__", "__BUJ_Flip_Stack_Inp__", "__BUJ_Unflip_Mask_Inp__", "__BUJ_Flip_Mask_Inp__",

                  "__REC_Image_Dir_Path__", "__REC_Image__", "__REC_QC_Input__",
                  "__REC_transform_y__", "__REC_transform_x__",
                  "__REC_transform_rot__", "__REC_Mask_Size__", "__REC_Arrow_Num__",
                  "__REC_Arrow_Len__", "__REC_Arrow_Wid__",
                  "__REC_FLS1__", "__REC_FLS2__", "__REC_Stack__", "__REC_Stack_Stage__",
                  "__REC_FLS1_Staging__", "__REC_FLS2_Staging__", "__REC_M_Volt__"
                  ]
    listbox_keys = ["__BUJ_Image_Choice__",
                    "__REC_Image_List__", "__REC_Def_List__"]
    radio_keys = ["__LS_full_align__", "__LS_param_test__", '__LS_unflip_reference__', '__LS_flip_reference__',
                  "__BUJ_unflip_reference__", "__BUJ_flip_reference__"]
    slider_keys = ["__LS_Stack_Slider__",
                   "__BUJ_Stack_Slider__", "__BUJ_img_subsf__",
                   "__REC_Slider__", "__REC_Image_Slider__", "__REC_Defocus_Slider__"]
    tab_keys = ["ls_tab", "bunwarpj_tab", "reconstruct_tab", "home_tab", "align_tab"]
    tabgroup_keys = ["align_tabgroup", "pages_tabgroup"]
    text_keys = ["home_title", "home_version", "home_authors", "home_readme", "home_contact",
                 "__LS_FLS1_Text__", "__LS_FLS2_Text__", "__BUJ_FLS1_Text__", "__BUJ_FLS2_Text__",
                 '__REC_FLS2_Text__', '__REC_FLS2_Text__', '__REC_Mask_Text__']
    read_only_inputs_keys = ["__LS_Image1__", "__LS_Image2__", "__LS_Stack__",
                             "__BUJ_Image1__", "__BUJ_Image2__", "__BUJ_Stack__",
                             "__BUJ_Unflip_Stack_Inp__", "__BUJ_Flip_Stack_Inp__",
                             "__REC_Image__"
                             ]
    keys = {'button': button_keys,
        'checkbox': checkbox_keys,
        'combo': combo_keys,
        'colum': column_keys,
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

    menu_def = [['PyLo', ['About::About', 'Exit::Exit1']],
                ['Log', ['Show (Control-l)::Log', 'Hide (Control-h)::Log']],
                ['Procedure Help', ['Open Manual::Manual']]]
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

    Returns
    -------
    tab: The layout for the hometab.
    """

    title = '''PyLorentz Phase Reconstruction'''
    version = '''Version 1.0.0'''
    authors = '''Authors: Tim Cote, Arthur McCray, CD Phatak'''
    readme = '''PyLorentz is a codebase designed for analyzing Lorentz Transmission Electron Microscopy (LTEM) data. There are three primary features and functions:

* PyTIE – Reconstructing the magnetic induction from LTEM images using the Transport of Intensity Equation (TIE).

* SimLTEM – Simulating phase shift and LTEM images from a given magnetization. (Unavailable in GUI)

* GUI/Align – Aligning raw LTEM images so they can be reconstructed and providing a GUI for PyTIE.
                
                
This GUI should be used as an alternative to the Jupyter Notebook for some improvments in region selection and control over saving images in PYTIE along with a means of faster alignment of LTEM images utilizing Fiji alignment protocols.

'''
    contact = 'Contact: tcote@anl.gov, amccray@anl.gov'

    if platform in ['win32', 'Windows']:
        fiji_button = sg.FolderBrowse("Browse", **style.styles('__Fiji_Browse__'))
    else:
        fiji_button = sg.FileBrowse("Browse", **style.styles('__Fiji_Browse__'))

    arrow1 = sg.Text("\u2193", pad=pad(240, 0, 20, 0))
    arrow2 = sg.Text("\u2193", pad=pad(0, 0, 20, 0))
    fiji_install_text = sg.Text(" Select Fiji Installation to access 'Registration' tab. ", pad=pad(0, 0, 15, 0))
    fiji_input = sg.Input(DEFAULTS['fiji_dir'], **style.styles('__Fiji_Path__'))
    fiji_set = sg.Button('Set Fiji', **style.styles('__Fiji_Set__'))
    fiji_reset = sg.Button('Reset Fiji', **style.styles('__Fiji_Reset__'))
    fiji_def_set = sg.Button('Set as Default', key='__Fiji_Def_Set__', pad=pad(10, 0, 10, 0))
    fiji_def_reset = sg.Button('Reset Default',  key='__Fiji_Def_Reset__', pad=pad(10, 0, 10, 0))
    browser_button = sg.FolderBrowse("Browse", key="__Browser_Browse__", pad=pad(164, 0, 10, 0),
                                     target='__Browser_Path__')
    browser_def_text = sg.Text('You can set the default browser working directory below.', pad=pad(230, 0, 10, 0))
    browser_wd_input = sg.Input(DEFAULTS['browser_dir'], key='__Browser_Path__', pad=pad(16, 0, 10, 0), size=(30, 1))
    browser_def_set = sg.Button('Set as Default', key='__Browser_Set__', pad=pad(10, 0, 10, 0))
    browser_def_reset = sg.Button('Reset Default',  key='__Browser_Reset__', pad=pad(10, 0, 10, 0))

    layout = [[sg.Col([[sg.T(title, **style.styles('home_title'))],
                       [sg.T(version, **style.styles('home_version'))],
                       [sg.T(authors, **style.styles('home_authors'))],
                       [sg.Multiline(readme, **style.styles('home_readme'))],
                       [sg.T(contact, **style.styles('home_contact'))],
                       [arrow1, fiji_install_text, arrow2],
                       [fiji_button,  fiji_input, sg.Col([[fiji_set, fiji_reset],
                                                          [fiji_def_set, fiji_def_reset]])],
                       [browser_def_text],
                       [browser_button, browser_wd_input, browser_def_set, browser_def_reset]]),
               ]] #sg.Graph((300, 300), (0, 0), (299, 299), background_color='grey', key='home_graph',
                        # pad=pad(80, 0, 110, 0))
    metadata = {"parent_tabgroup": "pages_tabgroup",
                "child_tabgroup": None}

    # Set the size of all tabs here as well!
    home_layout = [[sg.Col(layout, size=style.tab_size)]]
    tab = sg.Tab('Home', home_layout, font=style.fonts['tab'], key="home_tab", metadata=metadata)
    return tab


# ---------------------------------------------------- #
#                       Align Tab                      #
# ---------------------------------------------------- #
def align_tab(style: WindowStyle, DEFAULTS: Dict) -> Tab:
    """The align tab layout.

    Args:
        style: The class that holds all style data for the window.
            Look at gui_styling.py for more info.
        DEFAULTS: The default values for certain style elements
            such as font

    Returns
    -------
        tab: The layout for the align tab.
    """

    # ----- SIFT Tab ----- #
    def lin_sift_tab() -> List[List[Any]]:
        """The linear sift tab layout.

        Returns:
            tab: The layout for the linear SIFT alignment tab.
        """

        # --- FLS File Frame --- #
        def ls_fls_frame() -> List[List[Any]]:
            """The linear sift FlS frame.

            Returns:
                layout: List of lists of PySimpleGUI elements.
            """

            layout = [[sg.Text("FLS Files:", pad=((10, 0), (10, 0))),
                       sg.Combo(['Two', 'One'], **style.styles('__LS_FLS_Combo__')),
                       sg.Text("TFS:", pad=((10, 0), (10, 0))),
                       sg.Combo(['Unflip/Flip', 'Single'], **style.styles('__LS_TFS_Combo__'))
                       ],
                      [sg.FileBrowse("Load", **style.styles('__LS_Load_FLS1__')),
                       sg.Input('None', **style.styles('__LS_FLS1_Staging__')),
                       sg.Input("None", **style.styles('__LS_FLS1__')),
                       sg.Text("Unflip FLS", **style.styles('__LS_FLS1_Text__'))
                       ],
                      [sg.FileBrowse("Load", **style.styles('__LS_Load_FLS2__')),
                       sg.Input('None', **style.styles('__LS_FLS2_Staging__')),
                       sg.Input("None", **style.styles('__LS_FLS2__')),
                       sg.Text("Flip FLS", **style.styles('__LS_FLS2_Text__'))],
                      [sg.Text('Choose Reference Image:', pad=((70, 0), (2, 2)))],
                      [sg.Radio("Unflip", 'ReferenceRadio', **style.styles('__LS_unflip_reference__')),
                       sg.Radio("Flip", 'ReferenceRadio', **style.styles('__LS_flip_reference__'))],
                      [sg.Button('Set', **style.styles('__LS_Set_FLS__')),
                       sg.Button('Reset', **style.styles('__LS_Reset_FLS__')),
                       ]]
            return layout

        # --- SIFT Parameter Frame --- #
        def ls_sift_p_frame() -> List[List[Any]]:
            """The linear sift parameter frame layout.

            Returns:
                ls_parameters : The layout for the linear sift parameter frame.
            """
            # Scale Invariant Interest Point Detector parameters
            scale_inv_ipd = [sg.Text('Scale Invariant Interest Point Detector:', pad=((20, 10), (2, 2)))]
            init_gauss_blur = [sg.Text('initial gaussian blur:', pad=((95, 0), (0, 10))),
                               sg.Input('1.60', **style.styles('__LS_igb__')),
                               sg.Text('px', pad=((5, 0), (0, 10)))]
            steps_per_so = [sg.Text('steps per scale octave:', pad=((83, 0), (0, 10))),
                            sg.Input('3', **style.styles('__LS_spso__'))]
            min_im_size = [sg.Text('minimum image size:', pad=((87, 0), (0, 10))),
                           sg.Input('48', **style.styles('__LS_min_im__')),
                           sg.Text('px', pad=((5, 0), (0, 10)))]
            max_im_size = [sg.Text('maximum image size:', pad=((84, 0), (0, 10))),
                           sg.Input('1200', **style.styles('__LS_max_im__')),
                           sg.Text('px', pad=((5, 0), (0, 10)))]

            # Feature Descriptor parameters
            feature_desc = [sg.Text('Feature Descriptor:', pad=((20, 10), (2, 2)))]
            fds = [sg.Text('feature descriptor size:', pad=((80, 0), (0, 10))),
                   sg.Input('8', **style.styles('__LS_fds__'))]
            fdob = [sg.Text('feature descriptor orientation bins:', pad=((5, 0), (0, 10))),
                    sg.Input('8', **style.styles('__LS_fdob__'))]
            cncr = [sg.Text('closest/next closest ratio:', pad=((65, 0), (0, 10))),
                    sg.Input('0.95', **style.styles('__LS_cncr__'))]

            # Geometric Consensus Filter parameters
            geom_c_filter = [sg.Text('Geometric Consensus Filter:', pad=((20, 10), (2, 2)))]
            max_align_err = [sg.Text('maximum alignment error:', pad=((54, 0), (0, 10))),
                             sg.Input('5', **style.styles('__LS_max_al_err__')),
                             sg.Text('px', font=style.fonts['body'], pad=((5, 0), (0, 6)))]
            inlier_ratio = [sg.Text('inlier ratio:', pad=((155, 0), (0, 6))),
                            sg.Input('0.05', **style.styles('__LS_inlier_rat__'))]
            exp_transf = [sg.Text('expected transformation:', pad=((52, 0), (0, 0))),
                          sg.Combo(('Rigid', 'Translation', 'Similarity', 'Affine'), **style.styles('__LS_exp_transf__'))]

            # Output parameters
            output = [sg.Text('Output:', pad=((20, 10), (2, 2)))]
            interpolate = [sg.Checkbox('Interpolate', **style.styles('__LS_interp__'))]

            # Layout
            ls_parameters = [scale_inv_ipd,
                             init_gauss_blur,
                             steps_per_so,
                             min_im_size,
                             max_im_size,
                             feature_desc,
                             fds,
                             fdob,
                             cncr,
                             geom_c_filter,
                             max_align_err,
                             inlier_ratio,
                             exp_transf,
                             output, interpolate]
            return ls_parameters

        # --- Alignment Frame --- #
        def ls_al_frame() -> List[List[Any]]:
            """The linear sift alignment frame layout.

            Returns:
                align_frame: The layout for the linear alignment frame.
            """
            sift_test = [
                         [sg.Text('Image Transformation', font=style.fonts['heading'], pad=((0, 0), (70, 0)))],
                         [sg.Text('Rotation:'),
                          sg.Input('0', **style.styles('__LS_transform_rot__')),
                          sg.Text(u'\N{DEGREE SIGN}')],
                         [sg.Text('X shift:', pad=((11, 0), (0, 0))),
                          sg.Input('0', **style.styles('__LS_transform_x__')),
                          sg.Text('px')],
                         [sg.Text('Y shift:', pad=((11, 0), (0, 0))),
                          sg.Input('0', **style.styles('__LS_transform_y__')),
                          sg.Text('px')],
                         [sg.Checkbox('Horizontal Flip', **style.styles('__LS_horizontal_flip__'))],
                         [sg.Button('Adjust Image', **style.styles('__LS_Adjust__'))],
                         [sg.Text('Registration', pad=((43, 5), (120, 2)), font=style.fonts['heading'])],
                         [sg.Radio('Full Align', 'AlignRadio', **style.styles('__LS_full_align__'))],
                         [sg.Radio('Parameter Test', 'AlignRadio', **style.styles('__LS_param_test__'))],
                         [sg.HorizontalSeparator(color='black', pad=((0, 10), (0, 0)))],
                         [sg.Button('Run Align', **style.styles('__LS_Run_Align__')),
                          sg.Image(**style.styles('__LS_Align_Spinner__'))]]
            image_column = [[sg.Text("Image Directory:", pad=((10, 0), (0, 0))),
                             sg.Input(DEFAULTS['browser_dir'], **style.styles('__LS_Image_Dir_Path__')),
                             sg.FolderBrowse("Browse", **style.styles('__LS_Image_Dir_Browse__')),
                             sg.Button('Set', **style.styles('__LS_Set_Img_Dir__')),
                             sg.Button('Reset', **style.styles('__LS_Reset_Img_Dir__'))],
                            [sg.Graph((512, 512), (0, 0), (511, 511), **style.styles('__LS_Graph__'))]]

            align_frame = [[sg.Column(image_column),
                            sg.Column(sift_test)],
                           [sg.HorizontalSeparator(color='black', pad=(5, 5))],
                           [sg.Text('Image: 1.', pad=((10, 0), (0, 0))),
                            sg.Input(**style.styles('__LS_Image1__')),
                            sg.Button('View Stack', **style.styles('__LS_View_Stack__'))],
                           [sg.Text('2.', pad=((59, 0), (0, 0))),
                            sg.Input(**style.styles('__LS_Image2__')),
                            sg.Slider(**style.styles('__LS_Stack_Slider__'))],
                           [sg.Text('Stack:', pad=((12, 0), (0, 5))),
                            sg.Input(**style.styles('__LS_Stack__'))]]
            return align_frame

        # --- Full Linear SIFT Layout --- #
        def layout_ls_tab() -> List[List[Any]]:
            """The linear sift tab layout.

            Returns:
                lin_sift_layout: The layout for the linear sift tab.
            """
            fls_frame = sg.Frame(layout=ls_fls_frame(), title='FLS Files',
                                 relief=sg.RELIEF_SUNKEN, pad=((55, 0), (12, 0)), font=('Times New Roman', 19))
            sift_p_frame = sg.Frame(layout=ls_sift_p_frame(), title='Linear SIFT Parameters',  # title_color='purple'
                                    relief=sg.RELIEF_SUNKEN, pad=((55, 0), (12, 0)), font=('Times New Roman', 19))
            align_frame = sg.Frame(layout=ls_al_frame(), title='Alignment',
                                   relief=sg.RELIEF_SUNKEN, pad=((10, 0), (12, 30)), font=('Times New Roman', 19))
            # Layout
            page_layout = [[sg.Column([[fls_frame],
                                       [sift_p_frame]]), align_frame]]
            lin_sift_layout = [[sg.Column(page_layout, size=style.small_tab_size)]]
            return lin_sift_layout

        tab = layout_ls_tab()
        return tab

    # ----- bUnwarpJ Tab ----- #
    def bunwarp_tab() -> List[List[Any]]:
        """The bunwarpJ tab layout.

        Returns:
            tab: The layout for the bunwarpj tab.
        """

        def bunwarp_fls_frame() -> List[List[Any]]:
            """The bunwarpJ fls frame layout.

            Returns:
                layout: The layout for the bunwarpj tab.
            """
            layout = [[sg.Text("FLS Files:", pad=((10, 0), (10, 0))),
                       sg.Combo(['Two', 'One'], **style.styles('__BUJ_FLS_Combo__')),
                       sg.Text("TFS:", pad=((30, 0), (10, 0))),
                       sg.Combo(['Unflip/Flip'], **style.styles('__BUJ_TFS_Combo__'))
                       ],
                      [sg.FileBrowse("Load", **style.styles('__BUJ_Load_FLS1__')),
                       sg.Input('None', **style.styles('__BUJ_FLS1_Staging__')),
                       sg.Input("None", **style.styles('__BUJ_FLS1__')),
                       sg.Text("Unflip FLS", **style.styles('__BUJ_FLS1_Text__'))
                       ],
                      [sg.FileBrowse("Load", **style.styles('__BUJ_Load_FLS2__')),
                       sg.Input('None', **style.styles('__BUJ_FLS2_Staging__')),
                       sg.Input("None", **style.styles('__BUJ_FLS2__')),
                       sg.Text("Flip FLS", **style.styles('__BUJ_FLS2_Text__'))],
                      [sg.Text('Choose reference image:', pad=((90, 5), (3, 2)))],
                      [sg.Radio('Unflip', 'FlipRadio', **style.styles('__BUJ_unflip_reference__')),
                       sg.Radio('Flip', 'FlipRadio', **style.styles('__BUJ_flip_reference__'))],
                      [sg.Button('Set', **style.styles('__BUJ_Set_FLS__')),
                       sg.Button('Reset', **style.styles('__BUJ_Reset_FLS__')),
                       ]]
            return layout

        # --- SIFT Parameter Frame --- #
        def bunwarp_ls_sift_p_frame() -> List[List[Any]]:
            """The bunwarpJ linear sift parameter frame layout.

            Returns:
                ls_parameters: The layout for the bunwarpj ls_parameters.
            """
            # Scale Invariant Interest Point Detector parameters
            scale_inv_ipd = [sg.Text('Scale Invariant Interest Point Detector:', pad=((20, 10), (2, 2)))]
            init_gauss_blur = [sg.Text('initial gaussian blur:', pad=((95, 0), (0, 10))),
                               sg.Input('1.60', **style.styles('__BUJ_LS_igb__')),
                               sg.Text('px', pad=((5, 0), (0, 10)))]
            steps_per_so = [sg.Text('steps per scale octave:', pad=((83, 0), (0, 10))),
                            sg.Input('3', **style.styles('__BUJ_LS_spso__'))]
            min_im_size = [sg.Text('minimum image size:', pad=((87, 0), (0, 10))),
                           sg.Input('48', **style.styles('__BUJ_LS_min_im__')),
                           sg.Text('px', pad=((5, 0), (0, 10)))]
            max_im_size = [sg.Text('maximum image size:', pad=((84, 0), (0, 10))),
                           sg.Input('1200', **style.styles('__BUJ_LS_max_im__')),
                           sg.Text('px', pad=((5, 0), (0, 10)))]

            # Feature Descriptor parameters
            feature_desc = [sg.Text('Feature Descriptor:', pad=((20, 10), (2, 2)))]
            fds = [sg.Text('feature descriptor size:', pad=((80, 0), (0, 10))),
                   sg.Input('8', **style.styles('__BUJ_LS_fds__'))]
            fdob = [sg.Text('feature descriptor orientation bins:', pad=((5, 0), (0, 10))),
                    sg.Input('8', **style.styles('__BUJ_LS_fdob__'))]
            cncr = [sg.Text('closest/next closest ratio:', pad=((65, 0), (0, 10))),
                    sg.Input('0.95', **style.styles('__BUJ_LS_cncr__'))]

            # Geometric Consensus Filter parameters
            geom_c_filter = [sg.Text('Geometric Consensus Filter:', pad=((20, 10), (2, 2)))]
            max_align_err = [sg.Text('maximum alignment error:', pad=((53, 0), (0, 10))),
                             sg.Input('5', **style.styles('__BUJ_LS_max_al_err__')),
                             sg.Text('px', font=style.fonts['body'], pad=((5, 0), (0, 6)))]
            inlier_ratio = [sg.Text('inlier ratio:', pad=((154, 0), (0, 6))),
                            sg.Input('0.05', **style.styles('__BUJ_LS_inlier_rat__'))]
            exp_transf = [sg.Text('expected transformation:', pad=((52, 0), (0, 0))),
                          sg.Combo(('Rigid', 'Translation', 'Similarity', 'Affine'), **style.styles('__BUJ_LS_exp_transf__'))]

            # Output parameters
            output = [sg.Text('Output:', pad=((20, 10), (2, 2)))]
            interpolate = [sg.Checkbox('Interpolate', **style.styles('__BUJ_LS_interp__'))]

            # Layout
            ls_parameters = [scale_inv_ipd,
                             init_gauss_blur,
                             steps_per_so,
                             min_im_size,
                             max_im_size,
                             feature_desc,
                             fds,
                             fdob,
                             cncr,
                             geom_c_filter,
                             max_align_err,
                             inlier_ratio,
                             exp_transf,
                             output, interpolate]
            return ls_parameters

        # --- Bunwarpj SIFT Parameter Frame --- #
        def buj_feat_ext_p_frame() -> List[List[Any]]:
            """The bunwarpJ linear sift feature extraction parameter
            frame layout.

            Returns:
                feat_extr_parameters: The layout for the bunwarpj feat_extr_parameters.
            """
            # Scale Invariant Interest Point Detector parameters
            scale_inv_ipd = [sg.Text('Scale Invariant Interest Point Detector:', pad=((20, 10), (2, 2)))]
            init_gauss_blur = [sg.Text('initial gaussian blur:', pad=((95, 0), (0, 10))),
                               sg.Input('1.60', **style.styles('__BUJ_igb__')),
                               sg.Text('px', pad=((5, 0), (0, 10)))]
            steps_per_so = [sg.Text('steps per scale octave:', pad=((83, 0), (0, 10))),
                            sg.Input('3', **style.styles('__BUJ_spso__'))]
            min_im_size = [sg.Text('minimum image size:', pad=((87, 0), (0, 10))),
                           sg.Input('48', **style.styles('__BUJ_min_im__')),
                           sg.Text('px', pad=((5, 0), (0, 10)))]
            max_im_size = [sg.Text('maximum image size:', pad=((84, 0), (0, 10))),
                           sg.Input('1200', **style.styles('__BUJ_max_im__')),
                           sg.Text('px', pad=((5, 0), (0, 10)))]

            # Feature Descriptor parameters
            feature_desc = [sg.Text('Feature Descriptor:', pad=((20, 10), (2, 2)))]
            fds = [sg.Text('feature descriptor size:', pad=((80, 0), (0, 10))),
                   sg.Input('8', **style.styles('__BUJ_fds__'))]
            fdob = [sg.Text('feature descriptor orientation bins:', pad=((5, 0), (0, 10))),
                    sg.Input('8', **style.styles('__BUJ_fdob__'))]
            cncr = [sg.Text('closest/next closest ratio:', pad=((65, 0), (0, 10))),
                    sg.Input('0.95', **style.styles('__BUJ_cncr__'))]

            # Geometric Consensus Filter parameters
            filter_p = [sg.Checkbox('Filter matches by geometric consensus', **style.styles('__BUJ_filter__'))]
            geom_c_filter = [sg.Text('Geometric Consensus Filter:', pad=((20, 10), (2, 2)))]
            max_align_err = [sg.Text('maximum alignment error:', pad=((53, 0), (0, 10))),
                             sg.Input('5', **style.styles('__BUJ_max_al_err__')),
                             sg.Text('px', font=style.fonts['body'], pad=((5, 0), (0, 6)))]
            inlier_ratio = [sg.Text('minimal inlier ratio:', pad=((97, 0), (0, 6))),
                            sg.Input('0.05', **style.styles('__BUJ_inlier_rat__'))]
            min_num_inliers = [sg.Text('minimum number of inliers:', pad=((45, 0), (0, 6))),
                               sg.Input('7', **style.styles('__BUJ_min_num_inlier__'))]
            exp_transf = [sg.Text('expected transformation:', pad=((58, 0), (0, 0))),
                          sg.Combo(('Rigid', 'Translation', 'Similarity', 'Affine', 'Perspective'), **style.styles('__BUJ_exp_transf__'))]

            # Layout
            feat_extr_parameters = [scale_inv_ipd,
                                    init_gauss_blur,
                                    steps_per_so,
                                    min_im_size,
                                    max_im_size,
                                    feature_desc,
                                    fds,
                                    fdob,
                                    cncr,
                                    geom_c_filter,
                                    filter_p,
                                    max_align_err,
                                    inlier_ratio,
                                    min_num_inliers,
                                    exp_transf]
            return feat_extr_parameters

        # --- bUnwarp Parameter Frame --- #
        def bunwarp_p_frame() -> List[List[Any]]:
            """The bunwarpJ parameter frame layout.

            Returns:
                bunwarp_parameters: The layout for the bunwarpj parameter frame.
            """
            # bUnwarpJ parameters
            reg_mode = [sg.Text('Registration mode:', pad=((41, 10), (15, 0))),
                        sg.Combo(('Accurate', 'Fast', 'Mono'), **style.styles('__BUJ_reg_mode__'))]
            img_sub_factor_txt = [sg.Text('Image Subsample Factor:', pad=((68, 0), (0, 0)))]
            img_sub_factor = [sg.Slider(**style.styles('__BUJ_img_subsf__'))]

            init_def = [sg.Text('Initial Deformation:', pad=((42, 0), (0, 0))),
                        sg.Combo(('Very Coarse', 'Coarse', 'Fine', 'Very Fine'), **style.styles('__BUJ_init_def__'))]
            final_def = [sg.Text('Final Deformation:', pad=((47, 0), (0, 0))),
                         sg.Combo(('Very Coarse', 'Coarse', 'Fine', 'Very Fine', 'Super Fine'),
                                  **style.styles('__BUJ_final_def__'))]

            div_weight = [sg.Text('Divergence Weight:', pad=((43, 0), (0, 10))),
                          sg.Input('0.1', **style.styles('__BUJ_div_w__'))]
            curl_weight = [sg.Text('Curl Weight:', pad=((89, 0), (0, 10))),
                           sg.Input('0.1', **style.styles('__BUJ_curl_w__'))]

            landmark_weight = [sg.Text('Landmark Weight:', pad=((51, 0), (0, 10))),
                               sg.Input('0', **style.styles('__BUJ_land_w__'))]
            img_weight = [sg.Text('Image Weight:', pad=((77, 0), (0, 10))),
                          sg.Input('0.1', **style.styles('__BUJ_img_w__'))]
            cons_weight = [sg.Text('Consistency Weight:', pad=((39, 0), (0, 10))),
                           sg.Input('0.1', **style.styles('__BUJ_cons_w__'))]
            stop_thresh = [sg.Text('Stop Threshold:', pad=((70, 0), (0, 10))),
                           sg.Input('0.1', **style.styles('__BUJ_stop_thresh__'))]

            # Layout
            bunwarp_parameters = [reg_mode,
                                  img_sub_factor_txt,
                                  img_sub_factor,
                                  init_def,
                                  final_def,
                                  div_weight,
                                  curl_weight,
                                  landmark_weight,
                                  img_weight,
                                  cons_weight,
                                  stop_thresh]
            return bunwarp_parameters

        # --- Alignment Frame --- #
        def bunwarp_al_frame() -> List[List[Any]]:
            """The bunwarpJ alignment frame layout.

            Returns:
                align_frame: The layout for the bunwarpj alignment frame.
            """
            bunwarp_graph = [[sg.Text("1. Image Directory:", pad=((5, 0), (0, 0))),
                              sg.Input(DEFAULTS['browser_dir'], **style.styles('__BUJ_Image_Dir_Path__')),
                              sg.FolderBrowse("Browse", **style.styles('__BUJ_Image_Dir_Browse__')),
                              sg.Button('Set', **style.styles('__BUJ_Set_Img_Dir__')),
                              sg.Button('Reset', **style.styles('__BUJ_Reset_Img_Dir__'))],
                             [sg.Graph((512, 512), (0, 0), (511, 511), **style.styles('__BUJ_Graph__'))]]

            bunwarp_test = [[sg.Text('2b. Linear Stack SIFT Alignment', font=style.fonts['heading'],
                                     pad=((0, 0), (40, 5)))],

                            [sg.Text('Unflip:'),
                             sg.Button('Create New', **style.styles('__BUJ_Unflip_Align__')),
                             sg.Text(' |'),
                             sg.FileBrowse('Load Stack', **style.styles('__BUJ_Load_Unflip_Stack__')),
                             sg.Input(**style.styles('__BUJ_Unflip_Stage_Load__')),
                             sg.Image(**style.styles('__BUJ_Unflip_Spinner__'))],
                            [sg.Text('Flip:', pad=((16, 0), (5, 0))),
                             sg.Button('Create New', **style.styles('__BUJ_Flip_Align__')),
                             sg.Text(' |'),
                             sg.FileBrowse('Load Stack', **style.styles('__BUJ_Load_Flip_Stack__')),
                             sg.Input(**style.styles('__BUJ_Flip_Stage_Load__')),
                             sg.Image(**style.styles('__BUJ_Flip_Spinner__'))],

                            [sg.HorizontalSeparator(color='black', pad=((0, 10), (5, 5)))],

                            [sg.Text('3. Image Transformation', font=style.fonts['heading'], pad=((23, 0), (0, 0)))],
                            [sg.Text('Rotation:', pad=((35, 0), (0, 0))),
                             sg.Input('0', **style.styles('__BUJ_transform_rot__'), enable_events=True),
                             sg.Text(u'\N{DEGREE SIGN}')],
                            [sg.Text('X shift:', pad=((46, 0), (0, 0))),
                             sg.Input('0', **style.styles('__BUJ_transform_x__'), enable_events=True),
                             sg.Text('px')],
                            [sg.Text('Y shift:', pad=((46, 0), (0, 0))),
                             sg.Input('0', **style.styles('__BUJ_transform_y__'), enable_events=True),
                             sg.Text('px')],
                            [sg.Checkbox('Horizontal Flip', **style.styles('__BUJ_horizontal_flip__'))],
                            [sg.Button('Adjust Image', **style.styles('__BUJ_Adjust__'))],
                            [sg.HorizontalSeparator(color='black', pad=((0, 10), (5, 5)))],
                            [sg.Input("", **style.styles('__BUJ_Mask_Stage_Load__'))],
                            [sg.Text('4. Image Mask (opt.)', font=style.fonts['heading'], pad=((45, 5), (0, 2)))],
                            [sg.Button('Create New', **style.styles('__BUJ_Make_Mask__')),
                             sg.Text('|'),
                             sg.Col([[sg.FileBrowse("Load Mask", **style.styles('__BUJ_Load_Mask__'))]],
                                    key="__BUJ_Load_Mask_Col__"),
                             sg.Button("Erase", **style.styles('__BUJ_Reset_Mask__'))],
                            [sg.Text("Choose:", pad=((35, 0), (7, 0))),
                             sg.Combo(('Unflip', 'Flip', 'Overlay'),  **style.styles('__BUJ_Mask_View__'))],

                            [sg.Text('Unflip:', pad=((0, 0), (5, 0))),
                             sg.Input(**style.styles("__BUJ_Unflip_Mask_Inp__")),
                             sg.Button('Clear', pad=((4, 0), (5, 0)), key='__BUJ_Clear_Unflip_Mask__')],
                            [sg.Text('Flip:', pad=((16, 0), (3, 0))),
                             sg.Input(**style.styles("__BUJ_Flip_Mask_Inp__")),
                             sg.Button('Clear', pad=((4, 0), (3, 0)), key='__BUJ_Clear_Flip_Mask__')],
                            [sg.HorizontalSeparator(color='black', pad=((0, 10), (5, 5)))],
                            [sg.Text('5c. Registration', pad=((54, 5), (0, 0)), font=style.fonts['heading'])],
                            [sg.Button('bUnwarpJ Alignment', **style.styles('__BUJ_Elastic_Align__')),
                             sg.Image(**style.styles('__BUJ_Elastic_Spinner__'))]]

            files_column = [[sg.Text('Image:  1.', pad=((5, 0), (0, 0))),
                             sg.Input(**style.styles('__BUJ_Image1__'))],
                            [sg.Text('2.', pad=((57, 0), (0, 0))),
                             sg.Input(**style.styles('__BUJ_Image2__'))],
                            [sg.Text('Stacks:', pad=((5, 0), (0, 0)))],
                            [sg.Text('Unflip.', pad=((24, 0), (0, 0))),
                             sg.Input(**style.styles('__BUJ_Unflip_Stack_Inp__'))],
                            [sg.Text('Flip.', pad=((40, 0), (0, 0))),
                             sg.Input(**style.styles('__BUJ_Flip_Stack_Inp__'))],
                            [sg.Text('BUJ.', pad=((37, 0), (0, 5))),
                             sg.Input(**style.styles('__BUJ_Stack__'))]]

            stack_view_column = [[sg.Text(" ", pad=((20, 0), (10, 0))), sg.Slider(**style.styles('__BUJ_Stack_Slider__'))],
                                 [sg.Button('View Image', **style.styles('__BUJ_View__')),
                                  sg.Text(':'),
                                  sg.Listbox(['Unflip LS', 'Flip LS', 'bUnwarpJ'], **style.styles('__BUJ_Image_Choice__'))]]
            align_frame = [[sg.Column(bunwarp_graph), sg.Column(bunwarp_test)],
                           [sg.HorizontalSeparator(color='black', pad=(5, 5))],
                           [sg.Column(files_column), sg.Column(stack_view_column)]]
            return align_frame

        # --- Full bunwarpj Layout --- #
        def layout_bunwarp_tab() -> List[List[Any]]:
            """The bunwarpJ tab layout.

            Returns:
            bunwarp_layout: The layout for the bunwarpj tab.
            """
            fls_frame = sg.Frame(layout=bunwarp_fls_frame(), title='FLS Files',
                                 relief=sg.RELIEF_SUNKEN, pad=((4, 0), (0, 0)), font=('Times New Roman', 19))
            sift_par_tab = sg.Tab("2a. Linear SIFT", layout=bunwarp_ls_sift_p_frame(), font=style.fonts['tab'])
            feat_ext_par_tab = sg.Tab("5a. Feature Extract", layout=buj_feat_ext_p_frame(), font=style.fonts['tab'])
            bunwarp_par_tab = sg.Tab("5b. bUnwarpJ", layout=bunwarp_p_frame(), font=style.fonts['tab'])
            align_frame = sg.Frame(layout=bunwarp_al_frame(), title='Alignment',
                                   relief=sg.RELIEF_SUNKEN, pad=((0, 5), (0, 2)), font=('Times New Roman', 19))
            param_tabgroup = [[sg.TabGroup([[sift_par_tab, feat_ext_par_tab, bunwarp_par_tab]], tab_location='topleft',
                                           theme=sg.THEME_CLASSIC, enable_events=True, key="param_tabgroup")]]
            param_frame = sg.Frame(layout=param_tabgroup, title='Registration Parameters', pad=((4, 0), (2, 0)),
                                   relief=sg.RELIEF_SUNKEN, font=('Times New Roman', 19))
            # Layout
            page_layout = [[sg.Column([[fls_frame],
                                       [param_frame]]), align_frame]]

            bunwarp_layout = [[sg.Column(page_layout)]]
            return bunwarp_layout

        tab = layout_bunwarp_tab()
        return tab

    # ----- Overall Alignment Tab ----- #
    def layout_align_tab() -> List[List[Any]]:
        """The overall alignment tab layout.

        Returns:
        align_layout: The layout for the overall alignment tab.
        """
        # Align sub-tabs
        ls_metadata = {"parent_tabgroup": "align_tabgroup",
                       "child_tabgroup": None}
        ls_tab = sg.Tab('Linear Stack Alignment with SIFT', lin_sift_tab(), font=style.fonts['tab'],
                        key="ls_tab", metadata=ls_metadata)
        buj_metadata = {"parent_tabgroup": "align_tabgroup",
                        "child_tabgroup": None}
        bunwarpj_tab = sg.Tab('bUnwarpJ', bunwarp_tab(), font=style.fonts['tab'],
                              key="bunwarpj_tab", metadata=buj_metadata)
        align_layout = [[sg.TabGroup([[ls_tab, bunwarpj_tab]], tab_location='topleft', theme=sg.THEME_CLASSIC,
                                     enable_events=True, key="align_tabgroup")]]
        return align_layout

    metadata = {"parent_tabgroup": "pages_tabgroup",
                "child_tabgroup": "align_tabgroup"}
    tab = sg.Tab('Registration', layout_align_tab(),
                 disabled=True, font=style.fonts['tab'],
                 key="align_tab", metadata=metadata)
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
            The layout for the overall alignment tab.
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

        subregion_frame = [[sg.Text('Unset', key='__REC_Mask_Text__', font='Times 19', pad=((10, 0), (4, 4))),
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
                            sg.Col([[sg.Text('Mask Size:', pad=((40, 71), (12, 0)))],
                                    [sg.Input('50', **style.styles('__REC_Mask_Size__')),
                                      sg.Text('%', pad=((4, 0), (10, 0)))]
                                    ])
                            ]]

        TIE_menu = [[sg.Col([[sg.Text('Defocus:', pad=((30, 0), (4, 0))),
                              sg.Combo(['None'], **style.styles('__REC_Def_Combo__'))],
                             [sg.Text("QC value:", pad=((22, 0), (5, 0))),
                              sg.Input('0.00', **style.styles('__REC_QC_Input__')),
                              sg.Text('Symmetrize:', pad=((12, 0), (5, 0))),
                              sg.Checkbox('', **style.styles('__REC_Symmetrize__'))],
                             [sg.Text('Derivative:', pad=((16, 0), (5, 0))),
                              sg.Combo(['Central Diff.'], #, 'Longitudinal Deriv.'
                                        **style.styles('__REC_Derivative__'))]]),
                     sg.Col([[sg.Button('Run', **style.styles('__REC_Run_TIE__')),
                              sg.Image(**style.styles('__REC_PYTIE_Spinner__'))],
                             [sg.Button('Save', **style.styles('__REC_Save_TIE__'))]])],
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
                              sg.Input(DEFAULTS['browser_dir'], **style.styles('__REC_Image_Dir_Path__')),
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
    align_pg = align_tab(style, DEFAULTS)
    reconstruct_pg = reconstruct_tab(style, DEFAULTS)
    pages = sg.TabGroup([[home_pg, align_pg, reconstruct_pg]], tab_location='topleft', theme=sg.THEME_CLASSIC,
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

    Returns:
        layout: The layout for the save window.
        im_type: The image filetype.
        file_paths: List of the filepaths.
        orientations: List of strings of the orientations or image names.
        inputs: The keys of the inputs.
    """

    # Change parameters to suit what is being saved
    if orientations is None:
        orientations = ['']
    if event == '__BUJ_Make_Mask__':
        im_type = 'mask'
        names = [orientation + '_mask.bmp' for orientation in orientations]
        file_paths = names
    elif event == '__BUJ_Flip_Align__' or event == '__BUJ_Unflip_Align__':
        im_type = 'Linear Sift stack'
        names = [orientations[0] + '_aligned_ls_stack.tif']
        file_paths = names
    elif event == '__BUJ_Elastic_Align__':
        im_type = 'bUnwarpJ transform and stack'
        orientations = ['bunwarp transform', 'bunwarp stack']
        names = ['buj_transform.txt', 'aligned_buj_stack.tif']
        file_paths = names
    elif event == '__LS_Run_Align__':
        im_type = 'Linear Sift stack'
        if tfs == 'Unflip/Flip':
            names = ['aligned_ls_stack.tif']
        elif tfs == 'Single':
            if os.path.exists(join([image_dir, 'tfs'], '/')):
                names = ['tfs_aligned_ls_stack.tif']
            elif os.path.exists(join([image_dir, 'unflip'], '/')):
                names = ['unflip_aligned_ls_stack.tif']
        file_paths = names
    elif event == '__REC_Save_TIE__':
        im_type = 'Reconstructed Images'
        prefix = orientations
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
        elif orient == 'bunwarp stack':
            pad = ((34, 0), (10, 0))
            x_pad = 143
        elif orient == 'bunwarp transform':
            pad = ((5, 0), (10, 0))
            x_pad = 143
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
                      sg.Combo(['Color', 'Full Save', 'Mag. & Color', 'No Save'], key='__save_rec_combo__',
                                enable_events=True, size=(12, 1), default_value='Color',
                                readonly=True, pad=((20, 0), (10, 0)))]]
            inputs.extend(['__save_win_prefix__'])
            col2 += [[sg.Checkbox('', key=f'__save_win_overwrite{i}__', pad=((28, 0), (10, 0)), default=True,
                                  enable_events=True)]]
    # Create buttons to define whether to check if paths exists, exit, or save info
    col1 += [[sg.Button('Exit', pad=((x_pad, 0), (10, 5))),
              sg.Button('Save', key='__save_win_save__', pad=((5, 0), (10, 5)), disabled=True)],
             [sg.Multiline('', visible=True, key='__save_win_log__', size=size,
                           pad=((x_pad, 0), (0, 15)))]]
    layout = [[sg.Col(col1), sg.Col(col2)]]
    return layout, im_type, file_paths, orientations, inputs


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
    fiji_output = sg.Tab('FIJI', [[sg.Multiline('', key='FIJI_OUTPUT', write_only=True,
                                                size=(600, 16), autoscroll=True)],
                                  [sg.Text('Autoscroll', pad=(20, 2)),
                                   sg.Checkbox('', key='FIJI_OUTPUT_AUTOSCROLL', default=True,
                                               enable_events=True),
                                   sg.Button('Hide', pad=(20, 2), key='FIJI_OUTPUT_HIDE')]])
    pages = sg.TabGroup([[main_output, fiji_output]], tab_location='topleft',
                          theme=sg.THEME_CLASSIC,
                          enable_events=True, key="output_tabgroup")
    window_layout = [[invisible_graph], [pages]]
    icon = get_icon()
    window = sg.Window('Log', window_layout, default_element_size=(12, 1), disable_close=True,
                       resizable=True, size=(600, 350), use_default_focus=False, alpha_channel=0,
                       finalize=True, icon=icon)
    return window
