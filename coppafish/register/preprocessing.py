import os
import pickle
import itertools
import numpy as np
from tqdm import tqdm
from skimage.filters import sobel
from coppafish.setup import NotebookPage
from coppafish.utils.npy import load_tile


def load_reg_data(nbp_file: NotebookPage, nbp_basic: NotebookPage, config: dict):
    """
    Function to load in pkl file of previously obtained registration data if it exists.
    Args:
        nbp_file: File Names notebook page
        nbp_basic: Basic info notebook page
        config: register page of config dictionary
    Returns:
        registration_data: dictionary with the following keys
        * round_registration (dict) with keys:
            * completed (list)
            * position ( (z_subvols x y subvols x x_subvols) x 3 ) ndarray
            * round_shift ( n_tiles x n_rounds x (z_subvols x y subvols x x_subvols) x 3 ) ndarray
            * round_shift_corr ( n_tiles x n_rounds x (z_subvols x y subvols x x_subvols) ) ndarray
            * round_transform_raw (n_tiles x n_rounds x 3 x 4) ndarray
            * round_transform (n_tiles x n_rounds x 3 x 4) ndarray
        * channel_registration (dict) with keys:
            * completed (list)
            * channel_transform (n_tiles x n_channels x 3 x 4) ndarray
            * channel_shift_corr ( n_tiles x n_channels x (z_subvols x y subvols x x_subvols) ) ndarray
            * reference_round (n_tiles) ndarray
        * initial_transform (n_tiles x n_rounds x n_channels x 3 x 4) ndarray

    """
    # Check if the registration data file exists
    if os.path.isfile(os.path.join(nbp_file.output_dir, 'registration_data.pkl')):
        with open(os.path.join(nbp_file.output_dir, 'registration_data.pkl'), 'rb') as f:
            registration_data = pickle.load(f)
    else:
        n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels
        z_subvols, y_subvols, x_subvols = config['z_subvols'], config['y_subvols'], config['x_subvols']
        round_registration = {'tiles_completed': [], 'position': np.zeros((z_subvols * y_subvols * x_subvols, 3)),
                              'round_shift': np.zeros((n_tiles, n_rounds, z_subvols * y_subvols * x_subvols, 3)),
                              'round_shift_corr': np.zeros((n_tiles, n_rounds, z_subvols * y_subvols * x_subvols)),
                              'round_transform_raw': np.zeros((n_tiles, n_rounds, 3, 4)),
                              'round_transform': np.zeros((n_tiles, n_rounds, 3, 4))}
        channel_registration = {'channel_transform': np.zeros((nbp_basic.n_tiles, nbp_basic.n_channels, 3, 4)),
                                'cam_mse': np.zeros((len(set(nbp_basic.channel_camera)), 50))}
        registration_data = {'round_registration': round_registration,
                             'channel_registration': channel_registration,
                             'initial_transform': np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
                             }
    return registration_data


def save_compressed_image(output_dir: str, image: np.ndarray, t: int, r: int, c: int, filter: bool = False):
    """
    Save low quality cropped images for reg diagnostics

    Args:
        output_dir: string of output directory
        image: zyx image to be saved in compressed form
        t: tile
        r: round
        c: channel

    Returns:
        N/A
    """

    # Check directory exists otherwise create it
    if not os.path.isdir(os.path.join(output_dir, 'reg_images')):
        os.makedirs(os.path.join(output_dir, 'reg_images'))
    if filter:
        image = sobel(image)

    mid_z, mid_y, mid_x = image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2
    # save a small subset for reg diagnostics
    small_im = image[mid_z - 5: mid_z + 5, mid_y - 250: mid_y + 250, mid_x - 250: mid_x + 250]
    small_im = (small_im * 255 / np.max(small_im)).astype(np.uint8)
    np.save(os.path.join(output_dir, 'reg_images/') + 't' + str(t) + 'r' + str(r) + 'c' + str(c), small_im)


def replace_scale(transform: np.ndarray, scale: np.ndarray):
    """
    Function to replace the diagonal of transform with new scales
    Args:
        transform: n_tiles x n_rounds x 3 x 4 or n_tiles x n_channels x 3 x 4 of zyx affine transforms
        scale: 3 xn_tiles x n_rounds or 3 x n_tiles x n_channels of zyx scales

    Returns:
        transform: n_tiles x n_rounds x 3 x 4 or n_tiles x n_channels x 3 x 4 of zyx affine transforms
    """
    # Loop through dimensions i: z = 0, y = 1, x = 2
    for i in range(3):
        transform[:, :, i, i] = scale[i]

    return transform


def populate_full(sublist_1, list_1, sublist_2, list_2, array):
    """
    Function to convert array from len(sublist1) x len(sublist2) to len(list1) x len(list2), listting elems not in sublists
    as 0
    Args:
        sublist_1: sublist in the 0th dim
        list_1: entire list in 0th dim
        sublist_2: sublist in the 1st dim
        list_2: entire list in 1st dim
        array: array to be converted (dimensions len(sublist 1) x len(sublist2) x 3 x 4)

    Returns:
        full_array: len(list1) x len(list2) x 3 x 4 ndarray
    """
    full_array = np.zeros((len(list_1), len(list_2), 3, 4))
    for i in range(len(sublist_1)):
        for j in range(len(sublist_2)):
            full_array[sublist_1[i], sublist_2[j]] = array[i, j]
    return full_array


def yxz_to_zyx(image: np.ndarray):
    """
    Function to convert image from yxz to zyx
    Args:
        image: yxz image

    Returns:
        image_new: zyx image
    """
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    return image


def n_matches_to_frac_matches(n_matches: np.ndarray, spot_no: np.ndarray):
    """
    Function to convert n_matches to fraction of matches
    Args:
        n_matches: n_rounds x n_channels_use x n_iters
        spot_no: n_rounds x n_channels_use

    Returns:
        frac_matches: n_tiles x n_rounds x n_channels x n_iters
    """
    frac_matches = np.zeros_like(n_matches)

    for r in range(frac_matches.shape[0]):
        for c in range(frac_matches.shape[1]):
            frac_matches[r, c] = n_matches[r, c] / spot_no[r, c]

    return frac_matches


def split_3d_image(image, z_subvolumes, y_subvolumes, x_subvolumes, z_box, y_box, x_box):
    """
    Splits a 3D image into y_subvolumes * x_subvolumes * z_subvolumes subvolumes.

    Parameters
    ----------
    image : ndarray
        The 3D image to be split.
    y_subvolumes : int
        The number of subvolumes to split the image into in the y dimension.
    x_subvolumes : int
        The number of subvolumes to split the image into in the x dimension.
    z_subvolumes : int
        The number of subvolumes to split the image into in the z dimension.

    Returns
    -------
    subvolume : ndarray
        An array of subvolumes. The first three dimensions index the subvolume, the rest store the actual data.
    position: ndarray
        (y_subvolumes * x_subvolumes * z_sub_volumes) x 3 The middle coord of each subtile
    """
    z_image, y_image, x_image = image.shape

    # Allow 0.5 of a box either side and then split the middle with subvols evenly spaced points, ie into subvols - 1
    # intervals. Then use integer division. e.g actual unit distance is 12.5, this gives a unit distance of 12 so
    # should never overshoot
    if z_subvolumes > 1:
        z_unit = (z_image - z_box) // (z_subvolumes - 1)
    else:
        z_unit = 0
    y_unit = (y_image - y_box) // (y_subvolumes - 1)
    x_unit = (x_image - x_box) // (x_subvolumes - 1)

    # Create an array to store the subvolumes in
    subvolume = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, z_box, y_box, x_box))

    # Create an array to store the positions in
    position = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, 3))

    # Split the image into subvolumes and store them in the array
    for z in range(z_subvolumes):
        for y in range(y_subvolumes):
            for x in range(x_subvolumes):
                z_centre, y_centre, x_centre = z_box//2 + z * z_unit, y_box//2 + y * y_unit, x_box//2 + x * x_unit
                z_start, z_end = z_centre - z_box//2, z_centre + z_box//2
                y_start, y_end = y_centre - y_box//2, y_centre + y_box//2
                x_start, x_end = x_centre - x_box//2, x_centre + x_box//2

                subvolume[z, y, x] = image[z_start:z_end, y_start:y_end, x_start:x_end]
                position[z, y, x] = np.array([(z_start + z_end)//2, (y_start + y_end)//2, (x_start + x_end)//2])

    # Reshape the position array
    position = np.reshape(position, (z_subvolumes * y_subvolumes * x_subvolumes, 3))

    return subvolume, position


def compose_affine(A1, A2):
    """
    Function to compose 2 affine transfroms
    Args:
        A1: 3 x 4 affine transform
        A2: 3 x 4 affine transform
    Returns:
        A1 * A2: Composed Affine transform
    """
    # Add Final row, compose and then get rid of final row
    A1 = np.vstack((A1, np.array([0, 0, 0, 1])))
    A2 = np.vstack((A2, np.array([0, 0, 0, 1])))

    composition = (A1 @ A2)[:3, :4]

    return composition


def invert_affine(A):
    """
    Function to invert affine transform.
    Args:
        A: 3 x 4 affine transform

    Returns:
        inverse: 3 x 4 affine inverse transform
    """
    inverse = np.zeros((3, 4))

    inverse[:3, :3] = np.linalg.inv(A[:3, :3])
    inverse[:, 3] = -np.linalg.inv(A[:3, :3]) @ A[:, 3]

    return inverse


def yxz_to_zyx_affine(A: np.ndarray, new_origin: np.ndarray = np.array([0, 0, 0])):
    """
        Function to convert 4 x 3 matrix in y, x, z coords into a 3 x 4 matrix of z, y, x coords.

        Args:
            A: Original transform in old format (4 x 3)
            new_origin: Origin of new coordinate system in z, y, x coords

        Returns:
            A_reformatted: 3 x 4 transform with associated changes
            """
    # convert A to 3 x 4
    A = A.T

    # Append a bottom row to A
    A = np.vstack((A, np.array([0, 0, 0, 1])))

    # Now get the change of basis matrix to go from yxz to zyx. This is just obtained by rolling first 3 rows + cols
    # of the identity matrix right by 1
    C = np.eye(4)
    C[:3, :3] = np.roll(C[:3, :3], 1, axis=1)

    # Change basis and remove the final row
    A = (np.linalg.inv(C) @ A @ C)[:3, :4]

    # Add new origin conversion for zyx shift, need to do this after changing basis so that the matrix is in zyx coords
    A[:, 3] += (A[:3, :3] - np.eye(3)) @ new_origin

    return A


def zyx_to_yxz_affine(A: np.ndarray, new_origin: np.ndarray = np.array([0, 0, 0])):
    """
    Function to convert 3 x 4 matrix in z, y, x coords into a 4 x 3 matrix of y, x, z coords

    Args:
        A: Original transform in old format (3 x 4)
        new_origin: new origin to use for the transform (zyx)

    Returns:
        A_reformatted: 4 x 3 transform with associated changes

    """
    # Add new origin conversion for zyx shift, need to do this before changing basis
    A[:, 3] += (A[:3, :3] - np.eye(3)) @ new_origin

    # Append a bottom row
    A = np.vstack((A, np.array([0, 0, 0, 1])))

    # First, change basis to yxz
    C = np.eye(4)
    C[:3, :3] = np.roll(C[:3, :3], -1, axis=1)

    # compute the matrix in the new basis, remove the final row and transpose to get 4 x 3
    A = (np.linalg.inv(C) @ A @ C)[:3, :4].T

    return A


def reformat_array(A, nbp_basic, round):
    """
    Reformatting function to send A from (n_tiles * n_rounds) x z_subvol x x_subvol x y_subvol x 3 array to a new array
    with dimensions n_tiles x n_rounds x z_subvol x x_subvol x y_subvol x 3 and equivalently with channels
    Args:
        A: round, pos arrays for shift and position need to be reformatted
        nbp_basic: basic info page of notebook
        round: boolean option (True if working with round_shifts, False if working with channel_shifts)

    Returns:
        B: Reformatted array
    """

    n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels
    use_rounds, use_channels = nbp_basic.use_rounds, nbp_basic.use_channels
    c_ref = nbp_basic.anchor_channel
    counter = 0

    if round:
        B = np.zeros((n_tiles, n_rounds, A.shape[1], A.shape[2], A.shape[3], A.shape[4]))
        for t in range(n_tiles):
            for r in use_rounds:
                B[t, r] = A[counter]
                counter += 1
    else:
        # This scenario arises if we don't save the transforms for the anchor channel (identity)
        use_channels.remove(c_ref)
        B = np.zeros((n_tiles, n_channels, A.shape[1], A.shape[2], A.shape[3], A.shape[4]))
        for t in range(n_tiles):
            for c in use_channels:
                B[t, c] = A[counter]
                counter += 1
        use_channels.append(c_ref)

    return B


def custom_shift(array: np.ndarray, offset: np.ndarray, constant_values=0):
    """
    Custom-built function to compute array shifted by a certain offset
    Args:
        array: array to be shifted
        offset: shift value (must be int)
        constant_values: by default this is 0

    Returns:
        new_array: array shifted by offset with constant value 0
    """
    array = np.asarray(array)
    offset = np.atleast_1d(offset)
    assert len(offset) == array.ndim
    new_array = np.empty_like(array)

    def slice1(o):
        return slice(o, None) if o >= 0 else slice(0, o)

    new_array[tuple(slice1(o) for o in offset)] = (
        array[tuple(slice1(-o) for o in offset)])

    for axis, o in enumerate(offset):
        new_array[(slice(None),) * axis +
                  (slice(0, o) if o >= 0 else slice(o, None),)] = constant_values

    return new_array


def merge_subvols(position, subvol):
    """
    Function to merge subvolumes together into one image
    Args:
        position: n_subvols x 3 array of positions of bottom left of subvols (zyx)
        subvol: n_subvols x z_box x y_box x x_box array of subvols

    Returns:
        merged: merged image (size will depend on amount of overlap)
    """
    # set min values to 0
    position -= np.min(position, axis=0)
    z_box, y_box, x_box = subvol.shape[1:]
    # Get the min and max values of the position, use this to get the size of the merged image and initialise it
    max_pos = np.max(position, axis=0)
    merged = np.zeros((max_pos + subvol.shape[1:]).astype(int))

    # Loop through the subvols and add them to the merged image at the correct position. If there is overlap, take the
    # final value
    for i in range(position.shape[0]):
        merged[position[i, 0]:position[i, 0] + z_box, position[i, 1]:position[i, 1] + y_box,
               position[i, 2]:position[i, 2] + x_box] = subvol[i]

    return merged


def generate_channel_reg_images(nb):
    """
    Function to generate the channel registration images
    Args:
        nb: Notebook
    """
    non_anchor_channels = [c for c in nb.basic_info.use_channels if c != nb.basic_info.anchor_channel]
    use_tiles = nb.basic_info.use_tiles
    # Instead of doing a double loop over tiles and non-anchor channels, we can do a single, combine these into a single
    # array and then loop over this
    with tqdm(total=len(use_tiles) * len(non_anchor_channels)) as pbar:
        for t, c in itertools.product(use_tiles, non_anchor_channels):
            pbar.set_description(f"Generating channel registration image for tile {t} and channel {c}")
            # TODO: Get this working with nb.register_Debug.reference_round
            # Get the image for the tile and channel
            im = yxz_to_zyx(load_tile(nb.file_names, nb.basic_info, t, 3, c))
            save_compressed_image(nb.file_names.output_dir, im, t, 3, c, True)
            pbar.update(1)
