import time
import os

########################################################
# Path manipulation and converting strings into macros #
########################################################
def join(strings, sep=''):
    """Takes in strings and joins them with a specific separator. Replaces double
     backslash with forward slash to take care of path compatibilities between systems."""

    final_string = sep.join(strings)
    final_string = final_string.replace('\\', '/')
    return final_string


def fijify_macro(unready_macro):
    """Converts macro into FIJI format and returns macro"""

    fiji_ready_macro = ''
    for line in unready_macro.splitlines():
        # Make strings without ';' appear on same line.
        # Remove excess whitespace
        line = line.strip()
        whitespace = ''
        if ';' in line:
            whitespace = '\n'
        elif line and not line.isspace():
            whitespace = ' '
        fiji_ready_macro = join([fiji_ready_macro, line, whitespace])
    return fiji_ready_macro


def flatten_list(l):
    """Flattens a list of lists into a single list"""

    flat_list = [item for sublist in l for item in sublist]
    return flat_list


def format_GUI_string(s):
    """Formats strings for GUI to allow easier writing in code."""

    num = 1
    new_s = ""
    for line in s.splitlines():
        # Remove first newline
        if num == 1:
            add_str = ""
            num += 1
        # Remove and trailing spaces
        else:
            add_str = line.strip() + "\n"
        new_s = join([new_s, add_str])
    return new_s.strip()


##########################
# Manipulating fls files #
##########################
def grab_fls_data(unflip_fls, flip_fls, single_fls, check_sift):
    """Grab image data from .fls file."""

    # Read image data from .fls files and store in flip/unflip lists
    if single_fls:
        flip_files = pull_image_files(single_fls, check_sift)
        unflip_files = pull_image_files(single_fls, check_sift)
    else:
        flip_files = pull_image_files(flip_fls, check_sift)
        unflip_files = pull_image_files(unflip_fls, check_sift)
    return flip_files, unflip_files


def read_fls(datafolder, unflip_path, flip_path, check_sift):
    """Read image files from a .fls file. Checks if the image files exist, returning
       them if they do."""

    # Find paths and .fls files
    unflip_fls, flip_fls, single_fls = return_fls(datafolder)

    # Read image data from .fls files and store in flip/unflip lists
    flip_files, unflip_files = grab_fls_data(unflip_fls, flip_fls, single_fls, check_sift)

    # Check if image path exists and break if any path is nonexistent
    for file in flatten_list(unflip_files):
        full_path = join([unflip_path, file], '/')
        if not os.path.exists(full_path):
            return
    for file in flatten_list(flip_files):
        full_path = join([flip_path, file], '/')
        if not os.path.exists(full_path):
            return
    return flip_files, unflip_files


def return_fls(datafolder):
    """Return fls file(s) located in the datafolder."""

    unflip_fls, flip_fls, single_fls = None, None, None
    for file in os.listdir(datafolder):
        if '.fls' in file and 'unflip' in file:
            unflip_fls = join([datafolder, file], '/')
        elif '.fls' in file and 'flip' in file:
            flip_fls = join([datafolder, file], '/')
        elif '.fls' in file:
            single_fls = join([datafolder, file], '/')
    return unflip_fls, flip_fls, single_fls


def pull_image_files(fls_file, check_align=False):
    """Use .fls file to return ordered image files for alignment."""

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
        over_files = fls_lines[focus_split + 1: over_split + 1]

    # Reverse underfocus files due to how ImageJ opens images
    filenames = [under_files[::-1], focus_file, over_files]
    return filenames


def check_setup(datafolder):
    """Check to see all images filenames in .fls exists in
    the datafolder"""

    # Find paths and .fls files
    unflip_path = join([datafolder, 'unflip'], '/')
    flip_path = join([datafolder, 'flip'], '/')

    # Grab the files that exist in the flip and unflip dirs.
    file_result = read_fls(datafolder, unflip_path, flip_path, check_sift=False)
    if isinstance(file_result, tuple):
        return True, unflip_path, flip_path
    else:
        print('Task failed because the number of files extracted from the')
        print('directory does not match the number of files expected from')
        print('the .fls file. Check that filenames in the flip or unflip')
        print('path match and all files exist in the right directories.')
        print("\nTask failed!")
        return False


def collect_image_data(datafolder):
    """ Loads images files located in the "flip" and "unflip directories" of
    the image_dir
    """

    unflip_path = join([datafolder, 'unflip'], '/')
    flip_path = join([datafolder, 'flip'], '/')
    unflip_fls, flip_fls, single_fls = return_fls(datafolder)
    if single_fls:
        flip_files = pull_image_files(single_fls)
        unflip_files = pull_image_files(single_fls)
    else:
        flip_files = pull_image_files(flip_fls)
        unflip_files = pull_image_files(unflip_fls)

    unflip_ref = unflip_files[1]
    unflip_ref = join([unflip_path, unflip_ref[0]], '/')
    flip_ref = flip_files[1]
    flip_ref = join([flip_path, flip_ref[0]], '/')
    return unflip_ref, flip_ref, unflip_files, flip_files, unflip_path, flip_path


def pre_ls_alignement(datafolder, reference, check_sift):
    """ Pre-alignment file manipulations """

    # Check setup of datafolder
    check = check_setup(datafolder)
    if check:
        correct_setup, unflip_path, flip_path = check
        flip_files, unflip_files = read_fls(datafolder, unflip_path, flip_path, check_sift)
    else:
        return check

    # Get reference image to align to.
    ref = ''
    if reference == 'unflip':
        ref_list = unflip_files[1].pop()
        ref = join([unflip_path, ref_list], '/')
    elif reference == 'flip':
        ref_list = flip_files[1].pop()
        ref = join([flip_path, ref_list], '/')
    all_files = [unflip_files, flip_files]

    return ref, (all_files, flip_files, unflip_files), (flip_path, unflip_path)


#############################
# FIJI/ImageJ manipulations #
#############################
def ls_macro(sift_params):
    """Create macro for running SIFT alignment. Read 'feature
    extraction' on ImageJ for more information."""

    s_p = sift_params
    interpolate = ''
    if s_p['interpolate']:
        interpolate = 'interpolate' #'\n' +
    macro = f'''
            run("Linear Stack Alignment with SIFT",
            "initial_gaussian_blur={s_p['igb']}
            steps_per_scale_octave={s_p['spso']}
            minimum_image_size={s_p['min_im']}
            maximum_image_size={s_p['max_im']}
            feature_descriptor_size={s_p['fds']}
            feature_descriptor_orientation_bins={s_p['fdob']}
            closest/next_closest_ratio={s_p['cnc']}
            maximal_alignment_error={s_p['max_align_err']}
            inlier_ratio={s_p['inlier_rat']}
            expected_transformation={s_p['exp_transf']}
            {interpolate}");
            '''
    return macro


def get_shift_rot_macro(transformation, stack="False"):

    rot, x_shift, y_shift, horizontal = transformation
    apply_rotation, apply_translation, apply_hflip = '', '', ''
    run_flip_stack = ''
    run_stack = ''
    if stack:
        run_flip_stack = ', "stack"'
        run_stack = ' stack'
    if horizontal:
        apply_hflip = f'run("Flip Horizontally"{run_flip_stack});'
    if rot != 0:
        apply_rotation = f'''
                             run("Rotate... ", "angle={-rot} grid=1 interpolation=Bilinear{run_stack}");
                          '''
    if x_shift != 0 or y_shift != 0:
        apply_translation = f'''
                                run("Translate...", "x={x_shift} y={y_shift} interpolation=None{run_stack}");
                             '''
    apply_transform = f'''{apply_hflip}
                          {apply_rotation}
                          {apply_translation}'''
    return apply_transform


# noinspection PyUnboundLocalVariable
def apply_transform(flip, orient_path, fnames, transform_params):
    """Create macro that applies transformations for pre-alignment"""

    open_files, apply_translation = '', ''
    num_files = len(fnames) + 1
    for file in fnames:
        file = join([orient_path, file], '/')
        trans = get_shift_rot_macro(transform_params)
        apply_transform = f'open("{file}");'
        if flip:
            apply_transform = f'''open("{file}");
                              {trans}
                              '''
        open_files = join([open_files, apply_transform], "\n")
    transformed_files_macro = open_files
    return num_files, transformed_files_macro


def set_shortnames(files, place):
    """Return the filenames along with shortened names for
       ImageJ window"""

    # Set list of filenames based off orientation and focus.
    # Create shortnames for ImageJ windows.
    if place == (0, 0):
        fnames, short = files[0][0], 'u_-'
    elif place == (0, 1):
        fnames, short = files[0][1], 'u_0'
    elif place == (0, 2):
        fnames, short = files[0][2], 'u_+'
    elif place == (1, 0):
        fnames, short = files[1][0], 'f_-'
    elif place == (1, 1):
        fnames, short = files[1][1], 'f_0'
    elif place == (1, 2):
        fnames, short = files[1][2], 'f_+'
    return fnames, short


def trim_and_reverse_stacks(filenames, shortname, ref, pos):
    "Removes excess images and reverses images after initial alignment"

    # Trim excess images and reverse stack if necessary. Exceptions are if only 1 other
    # file or if overfocus images are of the reference orientation.
    # Reverse order of stack if images are underfocused.
    delete, reverse = '', ''
    if (len(filenames) > 1 and ((shortname != 'u_+' and 'unflip' in ref) or
       (shortname != 'f_+' and 'flip' in ref and 'unflip' not in ref))):
        delete = 'run("Delete Slice");'
        if not pos:
            reverse = 'run("Reverse");'
    return delete, reverse


def trim_and_reverse_single_stack(filenames, shortname):
    "Removes excess images and reverses images after initial alignment"

    # Trim excess images and reverse stack if necessary. Exceptions are if only 1 other
    # file or if overfocus images are of the reference orientation.
    # Reverse order of stack if images are underfocused.
    delete, reverse = '', ''
    if len(filenames) > 1 and shortname == '-under-':
        delete = 'run("Delete Slice");'
        reverse = 'run("Reverse");'
    return delete, reverse


def write_ls_macro(orient_path, ref, files, count, place, flip,
                   pos, sift_params, transform_params):
    """Create the macro for specific orientation, sign, and focus.
    """

    # Grab filenames and create shortened name for window.
    filenames, shortname = set_shortnames(files, place)
    if not filenames:
        return '', shortname

    # Create LS macro and apply transformations to images.
    sift = ls_macro(sift_params)
    num_files, open_files = apply_transform(flip, orient_path, filenames, transform_params)

    # Trim excess images and reverse stack if necessary. Exceptions are if only 1 other
    delete, reverse = trim_and_reverse_stacks(filenames, shortname, ref, pos)

    # Write macro
    macro = f'''open("{ref}");
                {open_files}
                run("Images to Stack", "name={count} title=[] use");
                {sift}
                selectWindow("{count}");
                close();
                selectWindow("Aligned {num_files} of {num_files}");
                rename("{shortname}");
                {delete}
                {reverse}
                '''
    return macro, shortname


def write_single_ls_macro(ref, path, files, window, sift_params):
    """Create the macro for specific orientation, sign, and focus.
    """

    # Grab filenames and create shortened name for window.
    if window == 1:
        shortname = "-under-"
    elif window == 2:
        shortname = "+over+"

    # Create LS macro
    sift = ls_macro(sift_params)

    # Trim excess images and reverse stack if necessary. Exceptions are if only 1 other
    delete, reverse = trim_and_reverse_single_stack(files, shortname)
    num_files = len(files) + 1

    open_files = ""    #apply_transform
    for file in files:
        file = join([path, file], '/')
        apply_transform = f'''open("{file}");
                          '''
        open_files = join([open_files, apply_transform], '\n')

    # Write macro
    ref = join([path, ref], '/')
    macro = f'''open("{ref}");
                {open_files}
                run("Images to Stack", "name={window} title=[] use");
                {sift}
                selectWindow("{window}");
                close();
                selectWindow("Aligned {num_files} of {num_files}");
                rename("{shortname}");
                {delete}
                {reverse}
                '''
    return macro, shortname


def set_default_sift():
    """Create the sub-macro for specific orientation and focus.

    Default SIFT parameters
        {'igb': 1.6,
        'spso': 3,
        'min_im': 48,
        'max_im': 1200,
        'fds': 4,
        'fdob': 8,
        'cnc': 0.95,
        'max_align_err': 5,
        'inlier_rat': 0.05,
        'exp_transf': 'Affine',
        'interpolate': True}

    Returns
    -------
    sift_params : Dict[str, Union[str, float, int, bool]]
        All necessary Linear Stack Align with SIFT parameters.
    """

    sift_params = {'igb': 1.6,
                   'spso': 3,
                   'min_im': 48,
                   'max_im': 1200,
                   'fds': 4,
                   'fdob': 8,
                   'cnc': 0.95,
                   'max_align_err': 5,
                   'inlier_rat': 0.05,
                   'min_num_inls': 7,
                   'exp_transf': 'Affine',
                   'interpolate': True,
                   'filter_param': True}
    return sift_params


def set_default_bUnwarp():
    bUnwarp_params = {'reg_mode': 'Fast',
                      'img_sub_factor': 0,
                      'init_def': 'Very Coarse',
                      'final_def': 'Very Fine',
                      'div_weight': 0.1,
                      'curl_weight': 0.1,
                      'landmark_weight': 1.0,
                      'img_weight': 1.0,
                      'cons_weight': 10,
                      'stop_thresh': 0.01}
    return bUnwarp_params


def check_image_flip(window, unflip_path, flip_path, ref):
    """Check if image requires a horizontal flip"""

    # If 'unflip' is reference, flip the 'flipped' images.
    # If 'flip' is reference, flip the 'unflipped' images.
    # From setup -> windows 1-3: unflipped, windows 4-6: flipped
    flip, path = None, None
    if window <= 3:
        path = unflip_path
        if 'unflip' in ref:
            flip = False
        elif 'flip' in ref:
            flip = True
    elif window > 3:
        path = flip_path
        if 'unflip' in ref:
            flip = True
        elif 'flip' in ref:
            flip = False
    return flip, path


def determine_window_focus(window):
    """Determine if image window is underfocused,
       in-focus, or overfocused."""

    # From setup -> images [2, 3, 5, 6] in/overfocused (+)
    #            -> images [1, 4] underfocus (-)
    pos = None
    if window % 3 >= 2:
        pos = True
    elif window % 3 == 1:
        pos = False
    orientation_num = (window - 1) // 3
    focus_num = (window - 1) % 3
    place = (orientation_num, focus_num)
    return place, pos


def ls_alignment(unflip_path, flip_path, sift_params, transform_params, ref, all_files):
    """Create ImageJ macros for each defocus and orientation
     other than the reference infocus image."""

    # Set sift param and image transformation defaults
    if not sift_params:
        sift_params = set_default_sift()
    if not transform_params:
        transform_params = (0, 0, 0, False)

    # Initialize parameters
    shortnames, macros = [], []
    window = 1
    while window <= 6:
        # Determine if the image window requires horizontal flip.
        flip, path = check_image_flip(window, unflip_path, flip_path, ref)
        # Determine image window's focus
        place, pos = determine_window_focus(window)
        # Write alignment macro and append to list. Write shortnames.
        macro, short = write_ls_macro(path, ref, all_files, window, place, flip=flip,
                                      pos=pos, sift_params=sift_params,
                                      transform_params=transform_params)
        macros.append(macro)
        shortnames.append(short)
        window += 1
    return macros, shortnames


def single_ls_alignment(sift_params, files, path):

    # Set sift param and image transformation defaults
    if not sift_params:
        sift_params = set_default_sift()

    # Initialize parameters
    ref = files[1][0]
    under, over = files[0][:], files[2][:]
    shortnames, macros = [], []
    window = 1
    for filenames in [under, over]:
        # Write single alignment macro and append to list. Write shortnames.
        macro, short = write_single_ls_macro(ref, path, filenames, window, sift_params)
        macros.append(macro)
        shortnames.append(short)
        window += 1
    return macros, shortnames


def save_stack_macro(savename):
    """Create macro to save and close all remaining windows."""

    macro = f'''saveAs("Tiff", "{savename}");'''
    return macro


def close_all_windows_macro():
    """ Create macro to close all remaining windows."""

    macro = '''while (nImages>0) {{selectImage(nImages); 
               close();}};'''
    return macro


def delete_excess_images_macro(files, short_names, ref):
    """Create macro to trim excess images appearing in aligned stacks."""

    # Initialize strings, set save title, and remove
    # ref image if necessary.
    full_img, window_select, trimmed_macro = '', '', ''
    index = 0
    for i in range(len(files)):
        for j in range(len(files[i])):
            delete = ""
            # Delete excess reference images except from overfocus
            # stack of the same orientation of reference image.
            if (len(files[i][j]) == 1 and short_names[index] != 'u_+' and
                    'unflip' in ref):
                delete = f'''selectWindow("{short_names[index]}");
                             run("Delete Slice");
                             '''
            elif (len(files[i][j]) == 1 and short_names[index] != 'f_+' and
                  'flip' in ref and 'unflip' not in ref):
                delete = f'''selectWindow("{short_names[index]}");
                             run("Delete Slice");
                             '''
            trimmed_macro = join([trimmed_macro, delete], "\n")
            index += 1
    return trimmed_macro


def order_windows_for_selection_macro(shortnames, ref):
    """Select windows in the order they will be combined."""

    img_num = 1
    num_windows = len(shortnames)
    window_order, concat_list = '', ''
    for window in range(num_windows):
        # Build list for concatenation later
        img = f' image{img_num}={shortnames[window]}'
        if (('unflip' in ref and shortnames[window] != 'u_0') or
            ('flip' in ref and 'unflip' not in ref and shortnames[window] != 'f_0')):
            new_select = f'selectWindow("{shortnames[window]}");'
            concat_list = join([concat_list, img])
            window_order = join([window_order, new_select], '\n')
            img_num += 1
        elif ('-under-' or '+over+') in shortnames[window]:
            new_select = f'selectWindow("{shortnames[window]}");'
            concat_list = join([concat_list, img])
            window_order = join([window_order, new_select], '\n')
            img_num += 1
    return window_order, concat_list


def concatenate_stacks_macro(stack_name, concat_list):
    """Create macro to concatenate all files into one large stack and return."""

    concatanation = f' run("Concatenate...", "  title={stack_name}{concat_list}");'
    return concatanation


def post_ls_alignment(macros, short_names, files, stack_name, ref):
    """Runs all saving, concatenation, deleting of excess files."""

    # Remove excess images
    trimmed_stacks = delete_excess_images_macro(files, short_names, ref)

    # Order windows and concatenate them together.
    window_order, concat_list = order_windows_for_selection_macro(short_names, ref)
    concatenated_stacks = concatenate_stacks_macro(short_names, concat_list)

    # Save all stacks and close windows, return names for display in GUI.
    save = save_stack_macro(stack_name)
    close_windows = close_all_windows_macro()

    # Return list of all macros and saved stack name
    macros = macros + [trimmed_stacks, window_order, concatenated_stacks, save, close_windows]
    return macros


def post_single_ls_alignment(macros, shortnames, files, stack_name, ref):

    # Remove excess images if necessary
    trimmed_macro = ''
    if len(files[0]) == 1:
        trimmed_macro = f'''selectWindow("{shortnames[0]}");
                            run("Delete Slice");
                         '''

    # Order windows and concatenate them together.
    window_order, concat_list = order_windows_for_selection_macro(shortnames, ref)
    concatenated_stacks = concatenate_stacks_macro(shortnames, concat_list)

    # Save all stacks and close windows, return names for display in GUI.
    save = save_stack_macro(stack_name)
    close_windows = close_all_windows_macro()

    # Return list of all macros and saved stack name
    macros = macros + [trimmed_macro, window_order, concatenated_stacks, save, close_windows]
    return macros


def format_macro(all_macros):
    """Format the full macro so it will be ready for use in FIJI."""

    joined_macro = join(all_macros, '\n')
    full_macro = fijify_macro(joined_macro)
    return full_macro


def order_slices(datafolder, all_files, ref, reference):
    """ Order filenames in a stack for display in the GUI."""

    ordered_files = []
    for i in range(len(all_files)):
        if i == 0:
            flip = 'unflip'
        elif i == 1:
            flip = 'flip'
        for j in range(len(all_files[i])):
            # Add if unflip reference and unflip overfocus
            if j == 1:
                if reference == flip:
                    ordered_files.append(ref)
                else:
                    name = all_files[i][j][0]
                    name = join([datafolder, flip, name], '/')
                    ordered_files.append(name)
            elif j == 0:
                for name in all_files[i][j][::-1]:
                    name = join([datafolder, flip, name], '/')
                    ordered_files.append(name)
            # Add if overfocused
            elif j == 2:
                for name in all_files[i][j]:
                    name = join([datafolder, flip, name], '/')
                    ordered_files.append(name)
    return ordered_files


def order_single_slices(files, path):

    ordered_files = []
    sorted_files = files[0][::-1] + files[1][:] + files[2][:]
    for file in sorted_files:
        name = join([path, file], '/')
        ordered_files.append(name)
    return ordered_files


def run_ls_align(datafolder, reference='unflip', check_sift=False, sift_params=None, transform_params=None,
                 stack_name='aligned_ls_stack.tif'):
    """ Aligns all 'dm3' files in the 'datafolder' and saves an aligned Tiff
    stack in the datafolder.
    """

    # Acknowledge if parameters are being tested.
    if check_sift:
        print('Checking SIFT parameters ...\n')

    # Open files, rotate, and apply transformations before alignment (pre-alignment)
    ref, image_files, image_paths = pre_ls_alignement(datafolder, reference, check_sift)
    all_files, flip_files, unflip_files = image_files
    flip_path, unflip_path = image_paths

    # Generate the Fiji macro for each alignment procedure.
    all_macros, shortnames = ls_alignment(unflip_path, flip_path, sift_params, transform_params, ref, all_files)

    # Post-alignment processing for saving the stack
    all_macros = post_ls_alignment(all_macros, shortnames, all_files, stack_name, ref)

    # Format macro to run in FIJI
    full_ls_macro = format_macro(all_macros)

    # # Run macro with PyimageJ
    # # print(full_ls_macro, '\n')
    # ij.py.run_macro(full_ls_macro)

    # Return list of ordered filenames (to display in GUI)
    # ordered_slice_names = order_slices(datafolder, all_files, ref, reference)
    return full_ls_macro  #, ordered_slice_names


def run_single_ls_align(datafolder, reference='', sift_params=None,
                        stack_name='test_ls_align.tif'):
    """ Aligns all 'dm3' files in the 'datafolder' and saves an aligned Tiff
    stack in the datafolder.
    """

    # Initiate timing and pre-alignment processing.
    # time_start = time.time()

    # Grab image data
    unflip_ref, flip_ref, unflip_files, flip_files, unflip_path, flip_path = collect_image_data(datafolder)

    # Generate the Fiji macro for each alignment procedure.
    if reference == "unflip":
        ref = unflip_ref
        files = unflip_files
        path = unflip_path
    elif reference == 'flip':
        ref = flip_ref
        files = flip_files
        path = flip_path

    # Single alignement
    all_macros, shortnames = single_ls_alignment(sift_params, files, path)

    # Post-alignment processing for saving the stack
    all_macros = post_single_ls_alignment(all_macros, shortnames, files, stack_name, ref)

    # Format macro to run in FIJI
    full_ls_macro = format_macro(all_macros)

    # Run macro with PyimageJ
    # print(full_ls_macro, '\n')
    # ij.py.run_macro(full_ls_macro)

    # Return list of ordered filenames (to display in GUI)
    # ordered_slice_names = order_single_slices(files, path)
    # time_stop = time.time()
    # print(f'Completed task. Aligned in {round(time_stop - time_start)} seconds!')
    return full_ls_macro

# import imagej
# fiji = imagej.init('/Applications/Fiji.app')
# fiji = 'hi'

# =======
# image_folder = '/Users/timothycote/Box/dataset1_tim'
# print(run_ls_align(image_folder, fiji, reference='unflip', check_sift=True,
#                   sift_params=None, transform_params=(0, 0, 0, True)))


def run_bUnwarp_align(datafolder, mask_files, reference, transformation, im_size,
                      stack_paths, sift_FE_params=None, buj_params=None,
                      savenames=("test.txt", "test.tif")):

    # Initiate pre-alignment processing.
    if savenames == ("test.txt", "test.tif"):
        transf_savename = join([datafolder, savenames[0]], '/')
        stack_savename = join([datafolder, savenames[1]], '/')
    else:
        transf_savename, stack_savename = savenames

    # Grab image data and get path names
    unflip_ref, flip_ref, unflip_files, flip_files, unflip_path, flip_path = collect_image_data(datafolder)

    # Pre-alignment (open image files, apply transformations, open masks)
    open_macro, src_img, target_img, masks = pre_bUnwarp_align(unflip_ref, flip_ref, mask_files,
                                                               reference, transformation)

    # Pre-alignment (open image files, apply transformations, open masks)
    SIFT_macro = extract_SIFT_landmarks_macro(src_img, target_img, sift_FE_params)

    # Alignment (grab bunwarp parameters, run alignment, save transformation)
    bUnwarp_macro = bUnwarp_align(src_img, target_img, masks, im_size,
                                  buj_params, transf_savename)
    close_windows = close_all_windows_macro()
    create_trasf_macro = fijify_macro(join([open_macro, SIFT_macro, bUnwarp_macro, close_windows], '\n'))
    apply_transf = apply_buj_trasf2stack(flip_ref, unflip_ref, stack_paths,
                                         flip_files, unflip_files, reference,
                                         transf_savename, transformation, stack_savename)
    apply_buj_macro = fijify_macro(join([apply_transf, close_windows], '\n'))
    full_macro = fijify_macro(join([create_trasf_macro, apply_buj_macro], '\n'))
    return full_macro


def apply_buj_trasf2stack(flip_ref, unflip_ref, stack_paths, flip_files, unflip_files, reference,
                          transform_path, transformation, savename):

    flip_stack_path, unflip_stack_path = stack_paths
    concat_macro = ''
    if reference == 'unflip':
        target_img = unflip_ref
        target_stack = unflip_stack_path
        source_stack = flip_stack_path
        stack_len = len(flatten_list(flip_files))
        concat_macro = f'run("Concatenate...", "title=merged_stack image1=target_stack image2=source_stack");'
    elif reference == 'flip':
        target_img = flip_ref
        target_stack = flip_stack_path
        source_stack = unflip_stack_path
        stack_len = len(flatten_list(unflip_files))
        concat_macro = f'run("Concatenate...", "title=merged_stack image1=source_stack image2=target_stack");'
    open_macro = f'''open("{target_img}");
                     rename("target_img");
                     open("{source_stack}");
                     rename("source_stack");'''
    transf_macro = get_shift_rot_macro(transformation, stack=True)
    buj_macro = ""
    for i in range(stack_len):
        apply_buj_macro = f'''call("bunwarpj.bUnwarpJ_.loadElasticTransform", "{transform_path}", 
                                                                              "target_img", 
                                                                              "source_stack");
                              print("Applying transform to slice {i+1}");'''
        if i != stack_len - 1:
            next_slice = 'run("Next Slice [>]");'
            apply_buj_macro = join([apply_buj_macro, next_slice], '\n')
        buj_macro = join([buj_macro, apply_buj_macro], '\n')
    target_stack_macro = f'''open("{target_stack}");
                             rename("target_stack");
                          '''

    save_macro = f'saveAs("TIFF","{savename}");'
    macro = join([open_macro, transf_macro, buj_macro,
                  target_stack_macro, concat_macro, save_macro], '\n')
    return macro


def bUnwarp_align(src_img, target_img, mask_fn, im_size, buj_params, transf_savename):

    # Grab all bUnwarpJ parameters
    if not buj_params:
        buj_params = set_default_bUnwarp()
    src_mask, tgt_mask = "", ""
    mask_macro = ""
    open_blank_mask = False
    for i in range(len(mask_fn)):
        if mask_fn[i]:
            if i == 0:
                open_mask = f'''open("{mask_fn[i]}");
                                rename("source_mask");
                             '''
                src_mask = "srcmask=source_mask"
            elif i == 1:
                open_mask = f'''open("{mask_fn[i]}");
                                rename("target_mask");
                             '''
                tgt_mask = "tgtmask=target_mask"
        else:
            x, y = im_size
            if not open_blank_mask:
                open_mask = f'newImage("No_Mask", "8-bit white", {x}, {y}, 1);'
                open_blank_mask = True
            if i == 0:
                src_mask = f"srcmask=No_Mask"
            elif i == 1:
                tgt_mask = f"tgtmask=No_Mask"
        mask_macro = join([mask_macro, open_mask], '\n')
    bUnwarp_macro = f"""
                {mask_macro}
                run("bUnwarpJ 2 images",
                "src={src_img}
                tgt={target_img}
                {src_mask}
                {tgt_mask}
                modechoice={buj_params['reg_mode']}
                subsamplefactor={buj_params['img_sub_factor']}
                minscalechoice=[{buj_params['init_def']}]
                maxscalechoice=[{buj_params['final_def']}]
                divweight={buj_params['div_weight']}
                curlweight={buj_params['curl_weight']}
                landmarkweight={buj_params['landmark_weight']}
                imageweight={buj_params['img_weight']}
                consistencyweight={buj_params['cons_weight']}
                stopthreshold={buj_params['stop_thresh']}
                savetransf=true
                filename={transf_savename}" 
                );
                """
    return bUnwarp_macro


def extract_SIFT_landmarks_macro(src_img, target_img, SIFT_params):

    # Grab all SIFT parameters
    if not SIFT_params:
        SIFT_params = set_default_sift()
    if SIFT_params['filter_param']:
        filter_p = 'filter' #'\n' +
    SIFT_macro = f"""
                run("Extract SIFT Correspondences",
                "source_image={src_img}
                target_image={target_img}
                initial_gaussian_blur={SIFT_params['igb']}
                steps_per_scale_octave={SIFT_params['spso']}
                minimum_image_size={SIFT_params['min_im']}
                maximum_image_size={SIFT_params['max_im']}
                feature_descriptor_size={SIFT_params['fds']}
                feature_descriptor_orientation_bins={SIFT_params['fdob']}
                closest/next_closest_ratio={SIFT_params['cnc']}
                {filter_p} 
                maximal_alignment_error={SIFT_params['max_align_err']}
                minimal_inlier_ratio={SIFT_params['inlier_rat']}
                minimal_number_of_inliers={SIFT_params['min_num_inls']}
                expected_transformation={SIFT_params['exp_transf']}"
                );
                """
    return SIFT_macro





def pre_bUnwarp_align(unflip_ref, flip_ref, mask_files, reference, transformation):

    # Grab reference images
    if reference == 'unflip':
        target_path, src_path = unflip_ref, flip_ref
        target_img = f'target_unflip_image'
        src_img = f'source_flip_image'
        src_mask, target_mask = mask_files[1], mask_files[0]
    elif reference == 'flip':
        target_path, src_path = flip_ref, unflip_ref
        target_img = f'target_flip_image'
        src_img = f'source_unflip_image'
        src_mask, target_mask = mask_files[0], mask_files[1]

    # Open files
    macro = f'''
                open("{target_path}");
                rename("{target_img}");
                open("{src_path}");
                rename("{src_img}");
             '''

    # Apply transformations to source image
    apply_transform = get_shift_rot_macro(transformation)
    macro = join([macro, apply_transform], '\n')

    # Open masks
    masks = [src_mask, target_mask]
    return macro, src_img, target_img, masks

# import imagej
# fiji = imagej.init('/Applications/Fiji.app')
# # fiji = 'hi'
# image_folder = '/Users/timothycote/Box/dataset1_tim'
# mask_files = [None, "/Users/timothycote/Box/dataset1_tim/flip_mask.bmp"]
# # ij = "hi"
# transformation = (1, 2, 1, True)
# im_size = (2048, 2048)
# reference = 'unflip'
# sift_params = None
# buj_params = None
# run_bUnwarp_align(image_folder, fiji, mask_files, reference,
#                   transformation, im_size, sift_params,
#                   buj_params)
