import os
import torch
import pickle
import skimage
import numpy as np
from scipy import signal
from scipy.ndimage import affine_transform
from skimage.transform import warp
from itertools import product
from tqdm import tqdm
import numpy.typing as npt
from typing import Union

from .. import log
from ..setup import NotebookPage, Notebook
from ..utils import tiles_io


def offset_pixels_by(image: npt.NDArray[np.uint16], tile_pixel_value_shift: int) -> npt.NDArray[np.int32]:
    """
    Apply an integer, negative shift to every image pixel and convert datatype from uint16 to int32.

    Args:
        image (`ndarray[uint16]`): image to shift.
        tile_pixel_value_shift (int): shift.

    Returns:
        `ndarray[int32]`: shifted image.
    """
    assert tile_pixel_value_shift <= 0, "Cannot shift by a positive number"
    return image.astype(np.int32) + tile_pixel_value_shift


def load_reg_data(nbp_file: NotebookPage, nbp_basic: NotebookPage):
    """
    Function to load in pkl file of previously obtained registration data if it exists.
    Args:
        nbp_file: File Names notebook page
        nbp_basic: Basic info notebook page
    Returns:
        registration_data: dictionary with the following keys
        * round_registration (dict) with keys:
            * completed (list)
            * warp_directory (str)
        * channel_registration (dict) with keys:
            * transform (n_channels x 4 x 3) ndarray of affine transforms (zyx)
    """
    # Check if the registration data file exists
    if os.path.isfile(os.path.join(nbp_file.output_dir, "registration_data.pkl")):
        with open(os.path.join(nbp_file.output_dir, "registration_data.pkl"), "rb") as f:
            registration_data = pickle.load(f)
    else:
        n_tiles, n_rounds, n_channels = (
            nbp_basic.n_tiles,
            nbp_basic.n_rounds + nbp_basic.n_extra_rounds,
            nbp_basic.n_channels,
        )
        round_registration = {"flow_dir": os.path.join(nbp_file.output_dir, "flow")}
        channel_registration = {"transform": np.zeros((n_channels, 4, 3))}
        registration_data = {
            "round_registration": round_registration,
            "channel_registration": channel_registration,
            "blur": False,
        }
    return registration_data


def replace_scale(transform: np.ndarray, scale: np.ndarray):
    """
    Replace the diagonal of transform with new scales
    Args:
        transform: n_tiles x n_rounds x 3 x 4 or n_tiles x n_channels x 3 x 4 of zyx affine transforms
        scale: 3 x n_tiles x n_rounds or 3 x n_tiles x n_channels of zyx scales

    Returns:
        transform: n_tiles x n_rounds x 3 x 4 or n_tiles x n_channels x 3 x 4 of zyx affine transforms
    """
    # Loop through dimensions i: z = 0, y = 1, x = 2
    for i in range(3):
        transform[:, :, i, i] = scale[i]

    return transform


def populate_full(sublist_1, list_1, sublist_2, list_2, array):
    """
    Function to convert array from len(sublist1) x len(sublist2) to len(list1) x len(list2), listing elems not in
    sub-lists as 0
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


def zyx_to_yxz(image: np.ndarray):
    """
    Function to convert image from zyx to yxz
    Args:
        image: zyx image

    Returns:
        image_new: yxz image
    """
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 0, 1)
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
    frac_matches = np.zeros_like(n_matches, dtype=np.float32)

    for r in range(frac_matches.shape[0]):
        for c in range(frac_matches.shape[1]):
            frac_matches[r, c] = n_matches[r, c] / spot_no[r, c]

    return frac_matches


def split_3d_image(image, z_subvolumes, y_subvolumes, x_subvolumes, z_box, y_box, x_box):
    """
    Splits a 3D image into y_subvolumes * x_subvolumes * z_subvolumes subvolumes. z_box, y_box and x_box must be even
    numbers.

    Parameters
    ----------
    image : (nz x ny x nx) ndarray or (ny x nx x nz) ndarray
        The 3D image to be split.
    y_subvolumes : int
        The number of subvolumes to split the image into in the y dimension.
    x_subvolumes : int
        The number of subvolumes to split the image into in the x dimension.
    z_subvolumes : int
        The number of subvolumes to split the image into in the z dimension.
    z_box : int
        The size of the subvolume in the z dimension.
    y_box : int
        The size of the subvolume in the y dimension.
    x_box : int
        The size of the subvolume in the x dimension.

    Returns
    -------
    subvolume : ((z_subvols * y_subvols * x_subvols) x z_box x y_box x z_box) ndarray
        An array of subvolumes. The first three dimensions index the subvolume, the rest store the actual data.
    position: ndarray
        (y_subvolumes * x_subvolumes * z_sub_volumes) x 3 The middle coord of each subtile
    """
    # Convert image to zyx
    if np.argmin(image.shape) == 2:
        image = yxz_to_zyx(image)

    # Make sure that box dims are even
    assert y_box % 2 == 0 and x_box % 2 == 0, "Box dimensions must be even numbers!"
    z_image, y_image, x_image = image.shape

    # Allow 0.5 of a box either side and then split the middle with subvols evenly spaced points, ie into subvols - 1
    # intervals.
    while (y_image - y_box) % (y_subvolumes - 1) != 0 or y_box % 2 != 0:
        y_box += 1
    while (x_image - x_box) % (x_subvolumes - 1) != 0 or x_box % 2 != 0:
        x_box += 1
    # define the unit spacing between centres for y and x
    y_unit = (y_image - y_box) // (y_subvolumes - 1)
    x_unit = (x_image - x_box) // (x_subvolumes - 1)

    # 2 cases for z, if z_subvolumes = 1, then z_box = z_image and z_unit = 0, else, deal with z_box and z_unit
    if z_subvolumes == 1:
        z_box = z_image
        z_unit = 0
    else:
        z_unit = (z_image - z_box) // (z_subvolumes - 1)
        while (z_image - z_box) % (z_subvolumes - 1) != 0 or z_box % 2 != 0:
            z_box += 1

    # Create an array to store the subvolumes in
    subvolume = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, z_box, y_box, x_box))

    # Create an array to store the positions in
    position = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, 3))

    # Split the image into subvolumes and store them in the array
    for z, y, x in np.ndindex(z_subvolumes, y_subvolumes, x_subvolumes):
        z_centre, y_centre, x_centre = z_box // 2 + z * z_unit, y_box // 2 + y * y_unit, x_box // 2 + x * x_unit
        y_start, y_end = y_centre - y_box // 2, y_centre + y_box // 2
        x_start, x_end = x_centre - x_box // 2, x_centre + x_box // 2
        if z_subvolumes == 1:
            z_start, z_end = 0, z_image + 1
        else:
            z_start, z_end = z_centre - z_box // 2, z_centre + z_box // 2

        subvolume[z, y, x] = image[z_start:z_end, y_start:y_end, x_start:x_end]
        position[z, y, x] = np.array([(z_start + z_end) // 2, (y_start + y_end) // 2, (x_start + x_end) // 2])

    # Reshape the position array
    position = np.reshape(position, (z_subvolumes * y_subvolumes * x_subvolumes, 3))

    return subvolume.astype(np.float32), position


def compose_affine(A1, A2):
    """
    Function to compose 2 affine transforms. A1 comes before A2.
    Args:
        A1: 3 x 4 affine transform
        A2: 3 x 4 affine transform
    Returns:
        A1 * A2: Composed Affine transform
    """
    assert A1.shape == (3, 4)
    assert A2.shape == (3, 4)
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
    # convert A to 4 x 3
    A = A.T

    # Append a right column to A
    A = np.vstack((A.T, np.array([0, 0, 0, 1]))).T

    # First, change basis to yxz
    C = np.eye(4)
    C[:3, :3] = np.roll(C[:3, :3], -1, axis=1)

    # compute the matrix in the new basis and remove the final column
    A = (np.linalg.inv(C) @ A @ C)[:4, :3]

    # Add new origin conversion for yxz shift, need to do this after changing basis so that the matrix is in yxz coords
    A[3, :] += (A[:3, :3] - np.eye(3)) @ new_origin

    return A


def custom_shift(array: np.ndarray, offset: np.ndarray, constant_values=0):
    """
    Compute array shifted by a certain offset.

    Args:
        array: array to be shifted.
        offset: shift value (must be int).
        constant_values: This is the value used for points outside the boundaries after shifting.

    Returns:
        new_array: array shifted by offset.
    """
    array = np.asarray(array)
    offset = np.atleast_1d(offset)
    assert len(offset) == array.ndim
    new_array = np.empty_like(array)

    def slice1(o):
        return slice(o, None) if o >= 0 else slice(0, o)

    new_array[tuple(slice1(o) for o in offset)] = array[tuple(slice1(-o) for o in offset)]

    for axis, o in enumerate(offset):
        new_array[(slice(None),) * axis + (slice(0, o) if o >= 0 else slice(o, None),)] = constant_values

    return new_array


def merge_subvols(position, subvol):
    """
    Suppose we have a known volume V split into subvolumes. The position of the subvolume corner
    in the coords of the initial volume is given by position. However, these subvolumes may have been shifted
    so position may be slightly different. This function finds the minimal volume containing all these
    shifted subvolumes.

    If regions overlap, we take the values from the later subvolume.
    Args:
        position: n_subvols x 3 array of positions of bottom left of subvols (zyx)
        subvol: n_subvols x z_box x y_box x x_box array of subvols

    Returns:
        merged: merged image (size will depend on amount of overlap)
    """
    position = position.astype(int)
    # set min values to 0
    position -= np.min(position, axis=0)
    z_box, y_box, x_box = subvol.shape[1:]
    centre = position + np.array([z_box // 2, y_box // 2, x_box // 2])
    # Get the min and max values of the position, use this to get the size of the merged image and initialise it
    max_pos = np.max(position, axis=0)
    merged = np.zeros((max_pos + subvol.shape[1:]).astype(int))
    neighbour_im = np.zeros_like(merged)
    # Loop through the subvols and add them to the merged image at the correct position.
    for i in range(position.shape[0]):
        subvol_i_mask = np.ix_(
            range(position[i, 0], position[i, 0] + z_box),
            range(position[i, 1], position[i, 1] + y_box),
            range(position[i, 2], position[i, 2] + x_box),
        )
        neighbour_im[subvol_i_mask] += 1
        merged[subvol_i_mask] = subvol[i]

    # identify overlapping regions
    overlapping_pixels = np.argwhere(neighbour_im > 1)
    if len(overlapping_pixels) == 0:
        return merged
    centre_dist = np.linalg.norm(overlapping_pixels[:, None, :] - centre[None, :, :], axis=2)
    # get the index of the closest centre
    closest_centre = np.argmin(centre_dist, axis=1)
    # now loop through subvols and assign overlapping pixels to the closest centre
    for i in range(position.shape[0]):
        subvol_i_pixel_ind = np.where(closest_centre == i)[0]
        subvol_i_pixel_coords_global = np.array([overlapping_pixels[j] for j in subvol_i_pixel_ind])
        subvol_i_pixel_coords_local = subvol_i_pixel_coords_global - position[i]
        z_global, y_global, x_global = subvol_i_pixel_coords_global.T
        z_local, y_local, x_local = subvol_i_pixel_coords_local.T
        merged[z_global, y_global, x_global] = subvol[i, z_local, y_local, x_local]

    return merged


def generate_reg_images(nb: Notebook):
    """
    Function to generate registration images. These are `[500 x 500 x min(10, n_planes)]` images centred in the middle
    of the tile and saved as uint8. They are saved as .npy files in the reg_images folder in the output directory.

    Args:
        nb: notebook.
    """
    use_tiles, use_rounds, use_channels = (
        nb.basic_info.use_tiles.copy(),
        nb.basic_info.use_rounds.copy(),
        nb.basic_info.use_channels.copy(),
    )
    if nb.basic_info.pre_seq_round is not None:
        use_rounds += [nb.basic_info.pre_seq_round]
    anchor_round, anchor_channel, dapi_channel = (
        nb.basic_info.anchor_round,
        nb.basic_info.anchor_channel,
        nb.basic_info.dapi_channel,
    )
    yx_centre = nb.basic_info.tile_centre.astype(int)[:2]
    yx_radius = np.min([250, nb.basic_info.tile_sz // 2])
    z_central_index = int(np.median(np.arange(len(nb.basic_info.use_z))))
    if len(nb.basic_info.use_z) <= 10:
        z_planes = np.arange(len(nb.basic_info.use_z))
    else:
        z_planes = np.arange(z_central_index - 5, z_central_index + 5)

    tile_centre = np.array([yx_centre[0], yx_centre[1]])
    yxz = [
        np.arange(tile_centre[0] - yx_radius, tile_centre[0] + yx_radius),
        np.arange(tile_centre[1] - yx_radius, tile_centre[1] + yx_radius),
        z_planes,
    ]

    # Create the reg_images directory if it doesn't exist
    reg_images_dir = os.path.join(nb.file_names.output_dir, "reg_images")
    if not os.path.isdir(os.path.join(nb.file_names.output_dir, "reg_images")):
        os.makedirs(reg_images_dir)
    # Create the t directories if they don't exist, within these create the round reg and channel reg directories
    for t in use_tiles:
        if not os.path.isdir(os.path.join(reg_images_dir, f"t{t}")):
            os.makedirs(os.path.join(reg_images_dir, f"t{t}"))
        if not os.path.isdir(os.path.join(reg_images_dir, f"t{t}", "round")):
            os.makedirs(os.path.join(reg_images_dir, f"t{t}", "round"))
        if not os.path.isdir(os.path.join(reg_images_dir, f"t{t}", "channel")):
            os.makedirs(os.path.join(reg_images_dir, f"t{t}", "channel"))

    # Get the anchor round and active channels
    anchor_round_active_channels = [dapi_channel, anchor_channel]
    for t, c in tqdm(product(use_tiles, anchor_round_active_channels), desc="Anchor Images", total=len(use_tiles) * 2):
        im = load_transformed_image(
            nb.basic_info,
            nb.file_names,
            nb.extract,
            nb.register,
            nb.register_debug,
            t=t,
            r=anchor_round,
            c=c,
            yxz=yxz,
            reg_type="none",
        )
        sub_dir = "round" if c == dapi_channel else "channel"
        file_name = os.path.join(reg_images_dir, f"t{t}", sub_dir, "anchor.npy")
        save_reg_image(im=im, file_path=file_name)

    # get the round images, apply optical flow, apply icp + optical flow, concatenate and save
    for t, r in tqdm(product(use_tiles, use_rounds), desc="Round Images", total=len(use_tiles) * len(use_rounds)):
        im = load_transformed_image(
            nb.basic_info,
            nb.file_names,
            nb.extract,
            nb.register,
            nb.register_debug,
            t=t,
            r=r,
            c=dapi_channel,
            yxz=yxz,
            reg_type="none",
        )
        im_flow = load_transformed_image(
            nb.basic_info,
            nb.file_names,
            nb.extract,
            nb.register,
            nb.register_debug,
            t=t,
            r=r,
            c=dapi_channel,
            yxz=yxz,
            reg_type="flow",
        )
        im_flow_icp = load_transformed_image(
            nb.basic_info,
            nb.file_names,
            nb.extract,
            nb.register,
            nb.register_debug,
            t=t,
            r=r,
            c=dapi_channel,
            yxz=yxz,
            reg_type="flow_icp",
        )
        im_concat = np.concatenate([im[None], im_flow[None], im_flow_icp[None]], axis=0)
        file_name = os.path.join(reg_images_dir, f"t{t}", "round", f"r{r}.npy")
        save_reg_image(im=im_concat, file_path=file_name)

    # get the channel images, save, apply optical flow, save, apply icp, save
    r_mid = 3
    for t, c in tqdm(product(use_tiles, use_channels), desc="Channel Images", total=len(use_tiles) * len(use_channels)):
        im = load_transformed_image(
            nb.basic_info,
            nb.file_names,
            nb.extract,
            nb.register,
            nb.register_debug,
            t=t,
            r=r_mid,
            c=c,
            yxz=yxz,
            reg_type="none",
        )
        im_flow = load_transformed_image(
            nb.basic_info,
            nb.file_names,
            nb.extract,
            nb.register,
            nb.register_debug,
            t=t,
            r=r_mid,
            c=c,
            yxz=yxz,
            reg_type="flow",
        )
        im_flow_icp = load_transformed_image(
            nb.basic_info,
            nb.file_names,
            nb.extract,
            nb.register,
            nb.register_debug,
            t=t,
            r=r_mid,
            c=c,
            yxz=yxz,
            reg_type="flow_icp",
        )
        im_concat = np.concatenate([im[None], im_flow[None], im_flow_icp[None]], axis=0)
        file_name = os.path.join(reg_images_dir, f"t{t}", "channel", f"c{c}.npy")
        save_reg_image(im=im_concat, file_path=file_name)


def load_transformed_image(
    nbp_basic_info: NotebookPage,
    nbp_file: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_register: NotebookPage,
    nbp_register_debug: NotebookPage,
    t: int,
    r: int,
    c: int,
    yxz: Union[list, None] = None,
    reg_type: str = "none",
) -> np.ndarray:
    """
    Load the image from tile t, round r, channel c, apply the relevant registration and return the image.
    Args:
        nbp_basic_info (NotebookPage)
        nbp_file_names (NotebookPage)
        nbp_register (NotebookPage)
        nbp_register_debug (NotebookPage)
        nb: Notebook (must have register and register_debug page)
        t: tile (int)
        r: round (int)
        c: channel (int)
        yxz: [np.arange(y), np.arange(x), np.arange(z)] (list). If None, load the entire transformed image.
        reg_type: str, 'none', 'flow' or 'flow_icp'
            - none: no registration
            - flow: apply channel correction (due to fluorescent beads) followed by optical flow
            - flow_icp: apply affine correction (due to icp) followed by optical flow

    Returns:
        im: np.ndarray, image
    """
    assert reg_type in ["none", "flow", "flow_icp"], "reg_type must be 'none', 'flow' or 'flow_icp'"
    suffix = "_raw" if r == nbp_basic_info.pre_seq_round else ""
    im = tiles_io.load_image(nbp_file, nbp_basic_info, nbp_extract.file_type, t, r, c, yxz=yxz, suffix=suffix).astype(
        np.float32
    )
    # anchor round has no flow or affine correction so can return early
    if reg_type == "none" or r == nbp_basic_info.anchor_round:
        return im

    # If we get this far, we will either be doing flow or flow icp, and we will not be in the anchor round.
    # These differ only by the affine correction we apply before.
    if yxz is not None:
        new_origin = np.array([yxz[0][0], yxz[1][0], yxz[2][0]])
    else:
        new_origin = np.zeros(3, dtype=int)
    affine_correction = np.eye(4, 3)
    if "reg_type" == "flow":
        if c != nbp_basic_info.dapi_channel:
            affine_correction = nbp_register_debug.channel_correction[t, c].copy()
    elif reg_type == "flow_icp":
        if c == nbp_basic_info.dapi_channel:
            affine_correction = nbp_register.icp_correction[t, r, nbp_basic_info.anchor_channel].copy()
        if c != nbp_basic_info.dapi_channel:
            affine_correction = nbp_register.icp_correction[t, r, c].copy()
    # adjust the affine correction for the new origin
    affine_correction = adjust_affine(affine=affine_correction, new_origin=new_origin)
    if yxz is not None:
        flow_indices = np.ix_(
            np.arange(3),
            np.arange(yxz[0][0], yxz[0][-1] + 1),
            np.arange(yxz[1][0], yxz[1][-1] + 1),
            np.arange(yxz[2][0], yxz[2][-1] + 1),
        )
    else:
        flow_indices = None
    flow_dir = os.path.join(nbp_register.flow_dir, "smooth", f"t{t}_r{r}.npy")
    im = transform_im(im=im, affine=affine_correction, flow_dir=flow_dir, flow_ind=flow_indices)

    return im


def transform_im(im: np.ndarray, affine: np.ndarray, flow_dir: str, flow_ind: tuple) -> np.ndarray:
    """
    Function to apply affine and flow transformations to an image.
    Args:
        im: image to transform
        affine: 3 x 4 affine transform
        flow_dir: directory containing the flow file
        flow_ind: indices to take from the flow file. If None, return the entire flow file.
    """
    # Apply the affine transform
    im = affine_transform(im, affine, order=1, mode="constant", cval=0)
    # Apply the flow transform
    if flow_ind is None:
        flow = -np.load(flow_dir, mmap_mode=None)
    else:
        flow = np.load(flow_dir, mmap_mode="r")
        flow = -(flow[flow_ind].astype(np.float32))
    # Flow's shape changes (3, im_y, im_x, im_z) -> (im_y, im_x, im_z, 3).
    flow = flow.transpose((1, 2, 3, 0))
    norm_half_pixel_0, norm_half_pixel_1, norm_half_pixel_2 = [1 / im.shape[i] for i in range(3)]
    flow[..., 0] *= norm_half_pixel_0 / 2
    flow[..., 1] *= norm_half_pixel_1 / 2
    flow[..., 2] *= norm_half_pixel_2 / 2
    grid_0, grid_1, grid_2 = torch.meshgrid(
        torch.linspace(norm_half_pixel_0 - 1, 1 - norm_half_pixel_0, im.shape[0]),
        torch.linspace(norm_half_pixel_1 - 1, 1 - norm_half_pixel_1, im.shape[1]),
        torch.linspace(norm_half_pixel_2 - 1, 1 - norm_half_pixel_2, im.shape[2]),
        indexing="ij",
    )
    grid = torch.cat((grid_2[None, :, :, :, None], grid_1[None, :, :, :, None], grid_0[None, :, :, :, None]), dim=4)
    im_warped = torch.asarray(im).float()[np.newaxis, np.newaxis]
    im_warped = torch.nn.functional.grid_sample(
        im_warped, grid + flow[np.newaxis], mode="bilinear", align_corners=False
    )[0, 0]
    return im_warped.numpy()


def adjust_affine(affine: np.ndarray, new_origin: np.ndarray) -> np.ndarray:
    """
    adjusts the affine transform for a new origin, then converts from 4 x 3 to 3 x 4 format.
    Args:
        affine: 4 x 3 affine transform (y x z)
        new_origin: (y, x, z) origin to adjust for

    Returns:
        affine: 3 x 4 affine transform (y x z)
    """
    assert affine.shape == (4, 3), "Affine must be 4 x 3"
    affine = affine.T
    affine[:, 3] += (affine[:3, :3] - np.eye(3)) @ new_origin
    return affine


def save_reg_image(im: np.ndarray, file_path: str) -> None:
    """
    takes in a small image and saves it as a uint8 image in the output directory
    Args:
        im: image to save (y, x, z) or 3( y, x, z) (usually (500 x 500 x 10) or (3, 500, 500, 10))
        file_path: str, path to save the image

    """
    im_min, im_max = np.min(im), np.max(im)
    im = im - im_min
    # Save the image as uint8
    if im_max != 0:
        im = im / np.max(im) * 255  # Scale to 0-255
    im = im.astype(np.uint8)
    np.save(file_path, im)


def window_image(image: np.ndarray) -> np.ndarray:
    """
    Window the image by a hann window in y and x and a Tukey window in z.

    Args:
        image: image to be windowed. (z, y, x)

    Returns:
        image: windowed image.
    """
    window_yx = skimage.filters.window("hann", image.shape[1:])
    window_z = signal.windows.tukey(image.shape[0], alpha=0.33)
    if (window_z == 0).all():
        window_z[...] = 1
    window = window_z[:, None, None] * window_yx[None, :, :]
    image = image * window
    return image


def flow_zyx_to_yxz(flow_zyx: np.ndarray) -> np.ndarray:
    """
    Convert a flow from zyx to yxz.
    Args:
        flow_zyx: np.ndarray of shape (3, nz, ny, nx) of flows in zyx coords.

    Returns:
        flow_yxz: np.ndarray of shape (3, ny, nx, nz)
    """
    flow_yxz = np.moveaxis(flow_zyx, 1, -1)
    flow_yxz = np.roll(flow_yxz, -1, axis=0)
    return flow_yxz


def apply_flow(flow: np.ndarray, points: np.ndarray, ignore_oob: bool = True, round_to_int: bool = True) -> np.ndarray:
    """
    Apply a flow to a set of points. Note that this is applying forward warping, meaning that the points are moved to
    their location in the warp array.

    Args:
        flow (np.ndarray): flow to apply. (3 x ny x nx x nz). In our case, this flow will always be the inverse flow,
        so we need to apply the negative of the flow to the points.
        points: integer points to apply the warp to. (n_points x 3 in yxz coords)
        ignore_oob: remove points that go out of bounds. Default: True.

    Returns:
        new_points: new points.
    """
    # have to subtract the flow from the points as we are applying the inverse warp
    y_indices, x_indices, z_indices = points.T
    new_points = points - flow[:, y_indices, x_indices, z_indices].T
    if round_to_int:
        new_points = np.round(new_points).astype(int)
    ny, nx, nz = flow.shape[1:]
    if ignore_oob:
        oob = (
            (new_points[:, 0] < 0)
            | (new_points[:, 0] >= ny)
            | (new_points[:, 1] < 0)
            | (new_points[:, 1] >= nx)
            | (new_points[:, 2] < 0)
            | (new_points[:, 2] >= nz)
        )
        new_points = new_points[~oob]
    return new_points
