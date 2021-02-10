"""Functions for FIJI and alignment.

There are different alignment procedures for FIJI for Linear Stack Alignment with Sift
as well as bUnwarpJ alignemnt. These function build the FIJI macro to be run
in FIJI and return a stack of aligned images.

Author:
Tim Cote, ANL, Fall 2019.
"""

import os
from os import path
from util import join, flatten_order_list
from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================================= #
#                     Declare Original Types                    #
# ============================================================= #
Check_Setup = Tuple[bool, Optional[str], Optional[str], Optional[List[str]], Optional[List[str]]]
Pre_LS_Align = Tuple[str, Tuple[List[List[str]], Optional[str], Optional[List[str]]]]


# ============================================================= #
#     Path manipulation and converting strings into macros      #
# ============================================================= #
def fijify_macro(unready_macro: str) -> str:
    """Converts macro into FIJI format and returns macro.

    The Fiji format uses ';' as end lines, and sometimes contains
    additional whitespace. This function accounts for that, edits
    the macro string, and returns it in the correct format.

    Args:
        unready_macro: The un-formatted FIJI macro.

    Returns:
        fiji_ready_macro: The correctly formatted FIJI macro.
    """

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


def format_macro(all_macros: List[str]) -> str:
    """Format the full macro so it will be ready for use in FIJI.

    Args:
        all_macros: All formatted FIJI macros for alignment.

    Returns:
        fiji_ready_macro: The finalized FIJI macro."""

    joined_macro = join(all_macros, '\n')
    full_macro = fijify_macro(joined_macro)
    return full_macro


# ============================================================= #
#                    Manipulating FLS Files                     #
# ============================================================= #
def pull_image_files(fls_file: str,
                     check_align: bool = False) -> List[List[str]]:
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
        over_files = fls_lines[focus_split + 1: over_split + 1]

    # Reverse underfocus files due to how ImageJ opens images
    filenames = [under_files[::-1], focus_file, over_files]
    return filenames


def grab_fls_data(fls1: str, fls2: str, tfs_value: str,
                  fls_value: str, check_sift: bool) -> Tuple[List[str], List[str]]:
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
    if fls_value == 'One':
        files1 = pull_image_files(fls1, check_sift)
        if tfs_value == 'Unflip/Flip':
            files2 = pull_image_files(fls1, check_sift)
        else:
            files2 = []
    elif fls_value == 'Two':
        files1 = pull_image_files(fls1, check_sift)
        files2 = pull_image_files(fls2, check_sift)
    return files1, files2


def read_fls(path1: Optional[str], path2: Optional[str], fls_files: List[str],
             tfs_value: str, fls_value: str,
             check_sift: bool = False) -> Tuple[List[str], List[str]]:
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
    fls1, fls2 =  fls_files[0], fls_files[1]

    # Read image data from .fls files and store in flip/unflip lists
    files1, files2 = grab_fls_data(fls1, fls2, tfs_value, fls_value, check_sift)

    # Check same number of files between fls
    if tfs_value != 'Single':
        if len(flatten_order_list(files2)) != len(flatten_order_list(files1)):
            return
    # Check if image path exists and break if any path is nonexistent
    if path1 is None and path2 is None:
        return
    for file in flatten_order_list(files1):
        full_path = join([path1, file], '/')
        if not path.exists(full_path):
            print(full_path, " doesn't exist!")
            return
    if files2:
        for file in flatten_order_list(files2):
            full_path = join([path2, file], '/')
            if not path.exists(full_path):
                print(full_path, " doesn't exist!")
                return
    return files1, files2


def check_setup(datafolder: str, tfs_value: str,
                fls_value: str, fls_files: List[str],
                prefix: str = '') -> Check_Setup:
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
    if tfs_value == 'Unflip/Flip':
        path1 = join([datafolder, 'unflip'], '/')
        path2 = join([datafolder, 'flip'], '/')
    elif tfs_value == 'Single':
        path1 = join([datafolder, 'tfs'], '/')
        if not os.path.exists(path1):
            path1 = join([datafolder, 'unflip'], '/')
            if not os.path.exists(path1):
                path1 = None
        path2 = None

    # Grab the files that exist in the flip and unflip dirs.
    file_result = read_fls(path1, path2, fls_files,
                           tfs_value, fls_value, check_sift=False)

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
        print(f'{prefix}Task failed because the number of files extracted from the directory', end=' ')
        print(f'does not match the number of files expected from the .fls file.')
        print(f'{prefix}Check that filenames in the flip or unflip', end=' ')
        print(f'path match and all files exist in the right directories.')
    return vals


# ============================================================= #
#                 Fiji Pre-Alignment Protocols                  #
# ============================================================= #
def pre_ls_alignment(reference: str, check_sift: bool, path1: str, path2: str,
                     fls_files: List[str], tfs_value: str, fls_value: str) -> Pre_LS_Align:
    """ Pre-alignment file manipulations for linear stack alignment with SIFT.

    Check setup, get reference image to align to, and get all image filenames.

    Args:
        reference: The type of reference to dictate alignment.
                Options: 'tfs', 'unflip', 'flip'
        check_sift (bool): Option for checking SIFT alignment.
        path1: The first unflip/flip/single path/directory.
        path2: The first unflip/flip/single path/directory.
        fls_files: A list of the FLS filenames.
        tfs_value: The through-focal series option.
                Options: Unflip/FLip, Single
        fls_value: The FLS option.
                Options: One, Two
    Returns:
        vals: A tuple of the reference file as well as all other filenames.

            - vals[0]: The reference filename.
            - vals[1][0]: 2D list of all files
            - vals[1][1]: 2D list of ordered image files for path1
            - vals[1][2]: 2D list of ordered image files for path2
    """

    # Check setup of datafolder
    files1, files2 = read_fls(path1, path2, fls_files, tfs_value, fls_value, check_sift)

    # Get reference image to align to.
    ref = ''
    if reference == 'tfs' or reference == 'unflip':
        ref_list = files1[1].pop()
        ref = join([path1, ref_list], '/')
    elif reference == 'flip':
        ref_list = files2[1].pop()
        ref = join([path2, ref_list], '/')
    all_files = [files1, files2]
    vals = ref, (all_files, files1, files2)
    return vals


def get_shift_rot_macro(transformation: Tuple[float, float, float, bool],
                        stack: bool = False) -> str:
    """ Creates macro for applying shifts and rotations for images .

    Args:
        transformation: Tuple of transformation values.
        stack: Option of whether image is a stack or single image

    Returns:
        apply_transform: The macro for shift and rotation transformations
    """

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
        y_shift = -y_shift
        apply_translation = f'''
                                run("Translate...", "x={x_shift} y={y_shift} interpolation=None{run_stack}");
                             '''
    apply_transform = f'''{apply_hflip}
                          {apply_rotation}
                          {apply_translation}'''
    return apply_transform


def apply_transform(flip: bool, orient_path: str,  fnames: List[str],
                    transform_params: Tuple[float, float, float, bool]) -> Tuple[int, str]:
    """Create macro that applies transformations for pre-alignment.

    Args:
        flip (bool): Option for whether the files are flipped or not.
        orient_path (str): The orientation path (flip, unflip, tfs).
        fnames (List[str]): List of filenames.
        transform_params (Tuple[float, float, float, bool]):
            Tuple of transformation parameters.

    Returns:
        Tuple of number of images and shifted rotated macro
            - num_files: The number of images files being manipulated.
            - transformed_files_macro: The shifted/rotated macro.
    """

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


def set_shortnames(files: List[List[str]],
                   place: Tuple[int, int]) -> Tuple[List[str], str]:
    """Return the filenames along with shortened names for
       FIJI window.

    Args:
        files: List of list of image filenames.
        place: The indices to be checking in the 'files' list.

    Returns:
        List of image filenames and shortnames
            - fnames: List of image filenames.
            - short: The shortened name for the files to be aligning
                - first character refers to flip/unlip,
                - last character to under/in/over-focus.
    """

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


def set_default_sift() -> Dict[str, Any]:
    """Set the default sift parameters.

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

    Returns:
        sift_params:All necessary Linear Stack Align with SIFT parameters.
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


def set_default_bUnwarp() -> Dict[str, Any]:
    """Create the default bUnwarpJ parameters.

        Default bUnwarpJ parameters
            {'reg_mode': 'Fast',
              'img_sub_factor': 0,
              'init_def': 'Very Coarse',
              'final_def': 'Very Fine',
              'div_weight': 0.1,
              'curl_weight': 0.1,
              'landmark_weight': 1.0,
              'img_weight': 1.0,
              'cons_weight': 10,
              'stop_thresh': 0.01}

        Returns:
            bUnwarp_params:All necessary bUnwarpJ parameters.
        """

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


def check_image_flip(window: int, path1: str,
                     path2: str, ref: str) -> Tuple[bool, str]:
    """Check if image requires a horizontal flip.

    Args:
        window: The number of the window being looked at
        path1: The 1st pathname for the orientation directory,
            depending on unflip/flip reference.
        path2: The 2nd pathname for the orientation directory,
            depending on unflip/flip reference.
        ref: The selected reference image of 'unflip' or 'flip'.

    Returns:
        Boolean value for if image needs to be flipped and the pathname
            of that image.
        """

    # If 'unflip' is reference, flip the 'flipped' images.
    # If 'flip' is reference, flip the 'unflipped' images.
    # From setup -> windows 1-3: unflipped, windows 4-6: flipped
    flip, path = None, None
    if window <= 3:
        path = path1
        if 'unflip' in ref:
            flip = False
        elif 'flip' in ref:
            flip = True
    elif window > 3:
        path = path2
        if 'unflip' in ref:
            flip = True
        elif 'flip' in ref:
            flip = False
    return flip, path


def determine_window_focus(window: int) -> Tuple[Tuple[int, int], bool]:
    """Determine if image window is underfocused, infocus, or overfocused.

    Args:
        window: The number of the window.
            - images [2, 3, 5, 6] in/overfocused (+)
            - images [1, 4] underfocus (-)

    Returns:
        Tuple of place in file list to choose focus during
            macro creation later and truth value for whether
            image is in/overfocus (True) or underfocus (False).
    """

    pos = None
    if (window - 1) % 3 > 0:
        pos = True
    elif (window - 1) % 3 == 0:
        pos = False
    orientation_num = (window - 1) // 3
    focus_num = (window - 1) % 3
    place = (orientation_num, focus_num)
    return place, pos


# ============================================================= #
#                   Fiji Alignment Protocols                    #
# ============================================================= #
def ls_macro(sift_params: Dict[str, Any]) -> str:
    """Create macro for running SIFT alignment. Read 'feature
    extraction' on ImageJ for more information.

    Args:
        sift_params: Dictionary of sift parameter values.

    Returns:
        marco: The macro for running the linear stack alignment
            with SIFT.
    """

    s_p = sift_params
    interpolate = ''
    if s_p['interpolate']:
        interpolate = 'interpolate'
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


def ls_alignment(path1: str, path2: str, sift_params: Dict[str, Any],
                 transform_params: Tuple[float, float, float, bool],
                 ref: str, all_files: List[List[str]]) -> Tuple[List[str], List[str]]:
    """Create ImageJ macros for each defocus and orientation other than infocus.

    Args:
         path1: The first unflip/flip path/directory.
         path2: The first unflip/flip path/directory.
         sift_params: The Linear SIFT params.
         transform_params: The shifting and rotation transformation params.
         ref: The reference image for flip/unflip.
         all_files: All ordered filenames in under
            [[underfocus], [infocus] [overfocus]]

    Returns:
        Tuple of list of both macros and shortnames.
    """

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
        flip, path = check_image_flip(window, path1, path2, ref)
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


def single_ls_alignment(sift_params: Dict[str, Any], files: List[List[str]],
                        path: str, param_test: bool = False,
                        ref: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Create ImageJ macros for each defocus and orientation
     other than the reference infocus image for a single stack.

     Args:
         sift_params: The Linear SIFT params.
         files: All ordered filenames in under [[underfocus], [infocus] [overfocus]]
         path: The single path/directory.
         param_test: Boolean for whether LS alignment is being checked
         ref: The reference filename.

    Returns:
        Tuple of list of both macros and shortnames.
    """

    # Set sift param and image transformation defaults
    if not sift_params:
        sift_params = set_default_sift()

    # Initialize parameters
    if ref is None:
        ref = files[1][0]
    if param_test:
        under, over = files[0], files[2]
    else:
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


def write_ls_macro(orient_path: str, ref: str, files: List[List[str]],
                   window: int, place: Tuple[int, int], flip: bool,
                   pos: bool, sift_params: Dict[str, Any],
                   transform_params: Tuple[float, float, float, bool]) -> Tuple[str, str]:
    """Create the linear stack alignment with SIFT (LS) macro for
    specific orientation, sign, and focus.

    Takes the orientation of flip/unflip/tfs and focus level of
    overfocus/underfocus/infocus parameters to write the specific
    LS macro.

    Args:
        orient_path: The orientation path (flip, unflip).
        ref: The reference infocus image filename.
        files: List of list of image filenames.
        window: Placeholder for name of new stack being made for bookkeeping
            when combining all stacks later.
        place: The indices to be checking in the 'files' list.
        flip: Option for whether the files are flipped or not.
        pos: Option for whether the image is under or overfocus to reverse
            the image stack.
        sift_params: Dictionary of sift parameter values.
        transform_params: Tuple of transformation parameters.

    Returns:
        Macro for specific orientation and focus and its shortname.
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


def write_single_ls_macro(ref: str, path: str, files: List[List[str]],
                          window: int, sift_params: Dict[str, Any]) -> Tuple[str, str]:
    """Create the linear stack alignment with SIFT (LS) macro for
    specific orientation, sign, and focus.

    Similar to write_ls_macro but for a single through focal series (tfs).

    Args:
        ref: The reference infocus image filename.
        path: The orientation path (tfs).
        files: List of list of image filenames.
        window: Placeholder for name of new stack being made for bookkeeping
            when combining all stacks later.
        sift_params: Dictionary of sift parameter values.

    Returns:
        Macro for specific orientation and focus and its shortname.
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


def extract_SIFT_landmarks_macro(src_img: str, target_img: str,
                                 SIFT_params: Optional[Dict[str, Any]]) -> str:
    """ Extracts SIFT features for bUnwarpJ alignment.

        Args:
            src_img: The infocus image to transform.
            target_img: The infocus image to align to.
            SIFT_params: Dictionary of bunwarpJ parameters.

        Returns:
            The bUnwarpJ macro for SIFT feature extraction for bUnwarpJ alignment.
        """

    # Grab all SIFT parameters
    if not SIFT_params:
        SIFT_params = set_default_sift()
    if SIFT_params['filter_param']:
        filter_p = 'filter'
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


def pre_bUnwarp_align(unflip_ref: str, flip_ref: str,
                      mask_files: List[Optional[str]], reference: str,
                      transformation: Tuple[float, float, float, bool]) -> Tuple[str, str, str, List[Optional[str]]]:
    """ Precursor to bUnwarpJ alignment to collect files, references, and masks..

        Determine the source and target images for bUnwarpJ. Apply any
        pre-transformations necessary to help with bUnwarpJ alignment. Open and
        return the masks for source and target.

        Args:
            unflip_ref: The unflip infocus image path.
            flip_ref: The flip infocus image path.
            mask_files: The filenames for the mask files, can be [None, None].
            reference: The reference value for 'unflip'/'flip'.
            transformation: The pre-shift/rotation to align the infocus images.

        Returns:
            The bUnwarpJ macro for carrying out the bUnwarpJ alignment. Additionally
                returns the source and target path names, as well as
                the associated masks for each image.
        """

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


def bUnwarp_align(src_img: str, target_img: str, mask_fn: List[Optional[str]],
                  im_size: int, buj_params: Optional[Dict[str, Any]],
                  transf_savename: str) -> str:
    """ Carries out the bUnwarpJ alignment.

        Aligns the flip and unflip infocus images, defined as the source or target.
        Mask files can be utilized to select specific regions for aligning.

        Args:
            src_img: The infocus image to transform.
            target_img: The infocus image to align to.
            mask_fn: The filenames for the mask files, can be [None, None].
            im_size: The pixel size of the sides of the square image.
            buj_params: Dictionary of bunwarpJ parameters.
            transf_savename: The path to the transformation file created by bUnwarpJ.

        Returns:
            The bUnwarpJ macro for carrying out the bUnwarpJ alignment.
        """

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


# ============================================================= #
#                Fiji Post-Alignment Protocols                  #
# ============================================================= #
def trim_and_reverse_stacks(filenames: List[List[str]], shortname: str,
                            ref: str, pos: bool) -> Tuple[str, str]:
    """Removes excess images and reverses images after initial SIFT alignment.

    The images for linear stack alignment with SIFT produce the best results by
    aligning images to an infocus image. However the stacks that are made in FIJI
    require the first image be the infocus, so when concatenating multiple stacks
    together, there are extra infocus images that may need to be removed. This function
    takes care of that. Images with only 2 images wait to be trimmed as you can't
    concatenate a single image with a stack in FIJI.

    Args:
        filenames: List of image filenames.
        shortname: The shortened name for the files to be aligning
            - first character refers to flip/unlip,
            - last character to under/in/overfocus.
        ref: The reference infocus image filename.
        pos: Option for whether the image is under or overfocus to reverse
            the image stack.

    Returns:
        Macros for deleting slices and reversing a stack.
            - delete: Macro for deleting slices from stack.
            - reverse: Macro for reversing an image stack.
    """
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


def trim_and_reverse_single_stack(filenames: List[List[str]], shortname: str) -> Tuple[str, str]:
    """Removes excess images for a single stack.

    Similar to trim_and_reverse_stacks but for a single tfs stack.

    Args:
        filenames: List of image filenames.
        shortname: The shortened name for the files to be aligning
            - first character refers to flip/unlip,
            - last character to under/in/over-focus.

    Returns:
        Macros for deleting slices and reversing a stack.
            - delete: Macro for deleting slices from stack.
            - reverse: Macro for reversing an image stack.
    """

    # Trim excess images and reverse stack if necessary. Exceptions are if only 1 other
    # file or if overfocus images are of the reference orientation.
    # Reverse order of stack if images are underfocused.
    delete, reverse = '', ''
    if len(filenames) > 1 and shortname == '-under-':
        delete = 'run("Delete Slice");'
        reverse = 'run("Reverse");'
    return delete, reverse


def save_stack_macro(savename: str) -> str:
    """Create macro to save and close all remaining windows."""

    macro = f'''saveAs("Tiff", "{savename}");'''
    return macro


def close_all_windows_macro() -> str:
    """Create macro to close all remaining windows."""

    macro = '''while (nImages>0) {{selectImage(nImages); 
               close();}};'''
    return macro


def delete_excess_images_macro(files: List[List[str]], short_names: List[List[str]],
                               ref: str) -> str:
    """Create macro to trim excess images appearing after stack is aligned.

    Args:
        files: 2D list of all of the files.
        short_names: The list of shortnames for the aligned stacks.
        ref: The reference infocus image.

    Returns:
        The macro from trimming excess images from stacks.
    """

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


def order_windows_for_selection_macro(shortnames: List[List[str]],
                                      ref: str) -> Tuple[str, str]:
    """Select windows in the order they will be combined.

    Args:
        shortnames: The list of shortnames for the aligned stacks.
        ref: The reference infocus image.

    Returns:
        Window order macro and the macro for concatenating stacks.
    """

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
        elif '-under-' in shortnames[window] or '+over+' in shortnames[window]:
            new_select = f'selectWindow("{shortnames[window]}");'
            concat_list = join([concat_list, img])
            window_order = join([window_order, new_select], '\n')
            img_num += 1
    return window_order, concat_list


def concatenate_stacks_macro(stack_names: str, concat_list: str) -> str:
    """Create macro to concatenate all files into one large stack and return.

    Args:
        stack_names: The shortnames of the stacks to concatenate.
        concat_list: The order for concatenating the stacks.

    Returns:
        The macro for concatenating stacks.
    """

    concatanation = f' run("Concatenate...", "  title={stack_names}{concat_list}");'
    return concatanation


def post_ls_alignment(macros: List[str], short_names: List[List[str]],
                      files: List[List[str]], stack_name:str,
                      ref: str) -> List[str]:
    """Runs all saving, concatenation, deleting of excess files.

    Args:
        macros: The current list of macros to feed into FIJI.
        short_names: The list of stack shortnames
        files: The 2D list of all image files.
        stack_name: The name to give the final fully aligned stack.
        ref: The reference infocus filename.

    Returns:
        List of macros for post LS alignment procedure.
    """

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


def post_single_ls_alignment(macros: List[str], shortnames: List[List[str]],
                             files: List[List[str]], stack_name:str,
                             ref: str) -> List[str]:
    """Runs all saving, concatenation, deleting of excess files for single tfs.

    Args:
        macros: The current list of macros to feed into FIJI.
        shortnames: The list of stack shortnames
        files: The 2D list of all image files.
        stack_name: The name to give the final fully aligned stack.
        ref: The reference infocus filename.

    Returns:
        List of macros for post LS alignment procedure.
    """

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


# ============================================================= #
#                        Fiji Run Calls                         #
# ============================================================= #
def run_ls_align(datafolder: str, reference: str = 'unflip', check_sift: bool = False,
                 sift_params: Optional[Dict[str, Any]] = None,
                 transform_params: Optional[Tuple[float, float, float, bool]] = None,
                 stack_name: str = 'uf_aligned_ls_stack.tif',
                 tfs_value: str = 'Unflip/Flip',
                 fls_value: str = 'Two',
                 fls_files: Optional[List[str]] = None) -> str:
    """ Aligns all 'dm3' files in the 'datafolder' and saves an aligned Tiff
    stack in the datafolder.

    Args:
        datafolder: The datafolder that contains the paths to the dm3 files.
        reference: The reference tag that will be used for aligning.
        check_sift: Option for full align (False) or check SIFT params align (True).
        sift_params: Dictionary of SIFT params.
        transform_params: Tuple of image shift and rotation params.
        stack_name: The filename for saving the aligned stack.
        tfs_value: Value for the type of tfs.
        fls_value: Value for the number of fls files.
        fls_files: The list of fls files.

    Returns:
        The full run LS align macro.
    """

    if tfs_value == 'Unflip/Flip':
        path1 = join([datafolder, 'unflip'], '/')
        path2 = join([datafolder, 'flip'], '/')
    elif tfs_value == 'Single':
        path1 = join([datafolder, reference], '/')
        path2 = None

    # Open files, rotate, and apply transformations before alignment (pre-alignment)
    ref, image_files = pre_ls_alignment(reference, check_sift, path1, path2, fls_files, tfs_value, fls_value)
    all_files, files1, files2 = image_files

    # Generate the Fiji macro for each alignment procedure.
    if tfs_value != 'Single':
        all_macros, shortnames = ls_alignment(path1, path2, sift_params, transform_params, ref, all_files)
        all_macros = post_ls_alignment(all_macros, shortnames, all_files, stack_name, ref)
    else:
        ref_name = ref[ref.rfind('/')+1:]
        all_macros, shortnames = single_ls_alignment(sift_params, files1, path1, param_test=check_sift, ref=ref_name)
        all_macros = post_single_ls_alignment(all_macros, shortnames, files1, stack_name, ref)

    # Format macro to run in FIJI
    full_ls_macro = format_macro(all_macros)

    # Return list of ordered filenames (to display in GUI)
    return full_ls_macro


def run_single_ls_align(datafolder: str, reference:  str = '',
                        sift_params: Optional[Dict[str, Any]] = None,
                        stack_name: str = 'test_ls_align.tif',
                        fls_files: Optional[List[str]] = None) -> str:
    """ Aligns all 'dm3' files in the 'datafolder' and saves an aligned tiff
    stack in the datafolder for a single tfs.

    Args:
        datafolder: The datafolder that contains the paths to the dm3 files.
        reference: The reference tag that will be used for aligning.
        sift_params: Dictionary of SIFT params.
        stack_name: The filename for saving the aligned stack.
        fls_files: The list of fls files

    Returns:
        The full run LS align macro for single tfs..
    """

    # Grab image data
    path1 = join([datafolder, 'unflip'], '/')
    path2 = join([datafolder, 'flip'], '/')
    unflip_files = pull_image_files(fls_files[0])
    flip_files = pull_image_files(fls_files[1])

    # Generate the Fiji macro for each alignment procedure.
    if reference == "unflip":
        ref = unflip_files[1]
        files = unflip_files
        path = path1
    elif reference == 'flip':
        ref = flip_files[1]
        files = flip_files
        path = path2

    # Single alignement
    all_macros, shortnames = single_ls_alignment(sift_params, files, path)

    # Post-alignment processing for saving the stack
    all_macros = post_single_ls_alignment(all_macros, shortnames, files, stack_name, ref)

    # Format macro to run in FIJI
    full_ls_macro = format_macro(all_macros)

    return full_ls_macro


def run_bUnwarp_align(datafolder: str, mask_files: List[Optional[str]],
                      reference: str, transformation: Tuple[float, float, float, bool],
                      im_size: int, stack_paths: List[Optional[str]],
                      sift_FE_params: Optional[Dict[str, Any]] = None,
                      buj_params: Optional[Dict[str, Any]] = None,
                      savenames: Tuple[str, str] = ("default.txt", "default.tif"),
                      fls_files: Optional[List[str]] = None) -> str:
    """ Uses bUnwarpJ to align all 'dm3' files in the 'datafolder' and saves an aligned Tiff
        stack in the datafolder.

        Args:
            datafolder: The datafolder that contains the paths to the dm3 files.
            mask_files: A list of the names of the mask_files.
                - This can be a list of [None, None] if no masks are used.
            reference: The reference tag that will be used for aligning.
            transformation: Tuple of image shift and rotation params.
            im_size: Integer pixel size of the image.
            stack_paths: [flip_path, unflip_path]
            sift_FE_params: Dictionary of SIFT feature extract params.
            buj_params: Dictionary of bUnwarpJ params.
            savenames: The filenames for saving the aligned stacks.
            fls_files: The list of fls files

        Returns:
            The full run bUnwarpJ align macro.
        """

    # Initiate pre-alignment processing.
    if savenames == ("default.txt", "default.tif"):
        transf_savename = join([datafolder, savenames[0]], '/')
        stack_savename = join([datafolder, savenames[1]], '/')
    else:
        transf_savename, stack_savename = savenames

    # Grab image data and get path names
    unflip_files = pull_image_files(fls_files[0])
    flip_files = pull_image_files(fls_files[1])
    for i in range(len(unflip_files)):
        for j in range(len(unflip_files[i])):
            unflip_files[i][j] = join([datafolder, 'unflip', unflip_files[i][j]], '/')
            flip_files[i][j] = join([datafolder, 'flip', flip_files[i][j]], '/')

    unflip_ref, flip_ref = unflip_files[1][0], flip_files[1][0]

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


def apply_buj_trasf2stack(flip_ref: str, unflip_ref: str,
                          stack_paths: List[str],
                          flip_files: List[List[str]], unflip_files: List[List[str]],
                          reference: str, transform_path: str,
                          transformation: Tuple[float, float, float, bool],
                          savename: str) -> str:
    """ Applies the transformation produced by bUnwarpJ for the unflip<->flip infocus
    images to all images of the opposite stack.

    Once the flip/unflip stack series are generated using the Linear Stack with SIFT alignment,
    the infocus images of each series are aligned. One of these infocus images is the target
    to align to, the other image is the source that is aligning to the target. Once these have
    been aligned with bUnwarpJ, the transformation is applied to all images that are in
    the stack series of the source image (source stack).

        Args:
            flip_ref: The datafolder that contains the paths to the dm3 files.
            unflip_ref: A list of the names of the mask_files.

            stack_paths: [flip_path, unflip_path]
            flip_files: List of all the flip files.
            unflip_files: List of all the unflip files.
            reference: The reference value of 'flip' or 'unflip'.
            transform_path: The path to the transformation file created by bUnwarpJ.
            transformation: Tuple of image shift and rotation params.
            savename: The filename for saving the aligned stack.

        Returns:
            The bUnwarpJ macro for applying the bUnwarpJ transformation.
        """

    flip_stack_path, unflip_stack_path = stack_paths
    concat_macro = ''
    if reference == 'unflip':
        target_img = unflip_ref
        target_stack = unflip_stack_path
        source_stack = flip_stack_path
        stack_len = len(flatten_order_list(flip_files))
        concat_macro = f'run("Concatenate...", "title=merged_stack image1=target_stack image2=source_stack");'
    elif reference == 'flip':
        target_img = flip_ref
        target_stack = flip_stack_path
        source_stack = unflip_stack_path
        stack_len = len(flatten_order_list(unflip_files))
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
