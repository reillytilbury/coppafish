from itertools import product
import os
import pickle
from typing import Optional, Union, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import affine_transform
import skimage
from skimage.transform import warp
from tqdm import tqdm
import zarr

from .. import spot_colors
from ..setup import Notebook, NotebookPage


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
        _, _, n_channels = (
            nbp_basic.n_tiles,
            nbp_basic.n_rounds + nbp_basic.n_extra_rounds,
            nbp_basic.n_channels,
        )
        round_registration = {"flow_dir": os.path.join(nbp_file.output_dir, "flow")}
        channel_registration = {"transform": np.zeros((n_channels, 4, 3))}
        registration_data = {
            "round_registration": round_registration,
            "channel_registration": channel_registration,
        }
    return registration_data


def split_image(im: np.ndarray, subvols_yx: int, overlap: float = 0.2) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Function to split an image into yx subvolumes with overlap. If the image does not divide evenly into subvolumes with
    the given overlap, the overlap will be increased to ensure that it does.
    Args:
        im: np.ndarray (n_y, n_x, n_z) image
        subvols_yx: int number of subvolumes in y and x
        overlap: float overlap fraction between subvolumes (0 <= overlap < 1)

    Returns:
        im_split: np.ndarray (subvols_yx**2, n_y, n_x, n_z) image
        positions: np.ndarray (subvols_yx**2, 2) yx positions of bottom left of subvolumes
        overlap_adjusted: float adjusted overlap fraction
    """
    im_size_yx, im_size_z = im.shape[0], im.shape[2]
    subvol_size_yx = int(im_size_yx / (subvols_yx * (1 - overlap) + overlap))
    # rounding errors will cause this overlap to be slightly different. Let's recompute given the integer subvol size
    overlap = (subvols_yx - im_size_yx / subvol_size_yx) / (subvols_yx - 1)

    # define im_split and positions
    im_split = np.zeros((subvols_yx, subvols_yx, subvol_size_yx, subvol_size_yx, im_size_z), dtype=im.dtype)
    positions = np.zeros((subvols_yx, subvols_yx, 2), dtype=int)

    # loop through subvolumes and populate im_split and positions
    for i, j in product(range(subvols_yx), range(subvols_yx)):
        y_start = int(i * (subvol_size_yx * (1 - overlap)))
        x_start = int(j * (subvol_size_yx * (1 - overlap)))
        y_end = y_start + subvol_size_yx
        x_end = x_start + subvol_size_yx
        im_split[i, j] = im[y_start:y_end, x_start:x_end]
        positions[i, j] = np.array([y_start, x_start])

    # flatten dims 0 and 1 of im_split and positions
    im_split = im_split.reshape((subvols_yx**2, subvol_size_yx, subvol_size_yx, im_size_z))
    positions = positions.reshape((subvols_yx**2, 2))

    return im_split, positions, overlap


def merge_subvols(im_split: np.ndarray, positions: np.ndarray, overlap: float,
                  output_shape: Union[list, tuple]) -> np.ndarray:
    """
    Function to merge subvolumes back into a single image. In cases of overlap, the subvolume with closest centre to the
    pixel will be used.
    Args:
        im_split: np.ndarray (subvols_yx**2, n_y, n_x, n_z) image
        positions: np.ndarray (subvols_yx**2, 2) yx positions of bottom left of subvolumes
        overlap: float overlap fraction between subvolumes (0 <= overlap < 1)
        output_shape: shape of the output image (n_y, n_x, n_z).

    Returns:
        im: np.ndarray (n_y, n_x, n_z) image
    """
    # initialise variables
    subvol_size_yx = im_split.shape[1]
    subvols_yx = int(np.sqrt(im_split.shape[0]))
    im = np.zeros(output_shape, dtype=im_split.dtype)

    # reshape im_split and positions
    im_split = im_split.reshape((subvols_yx, subvols_yx, subvol_size_yx, subvol_size_yx, im_split.shape[3]))
    positions = positions.reshape((subvols_yx, subvols_yx, 2))

    # taper the subvolumes where there is overlap so that we can add them together without double counting
    n_overlap = int(subvol_size_yx * overlap)
    taper_array_1d = np.linspace(0, 1, n_overlap)
    for i, j in product(range(subvols_yx), range(subvols_yx)):
        # does tile have northern neighbour?
        if i > 0:
            im_split[i, j, :n_overlap] *= taper_array_1d[:, None, None]
        # does tile have southern neighbour?
        if i < subvols_yx - 1:
            im_split[i, j, -n_overlap:] *= taper_array_1d[::-1, None, None]
        # does tile have western neighbour?
        if j > 0:
            im_split[i, j, :, :n_overlap] *= taper_array_1d[None, :, None]
        # does tile have eastern neighbour?
        if j < subvols_yx - 1:
            im_split[i, j, :, -n_overlap:] *= taper_array_1d[None, ::-1, None]

    # loop through subvolumes and populate im
    for i, j in product(range(subvols_yx), range(subvols_yx)):
        y_start, x_start = positions[i, j]
        y_end, x_end = y_start + subvol_size_yx, x_start + subvol_size_yx
        im[y_start:y_end, x_start:x_end] += im_split[i, j]

    return im


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


def generate_reg_images(
    nbp_basic: NotebookPage,
    nbp_file: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_register: NotebookPage,
    nbp_register_debug: NotebookPage,
):
    """
    Function to generate and save registration images. These are `[500 x 500 x min(10, n_planes)]` images centred in
    the middle of the tile and saved as uint8. They are saved as .npy files in the reg_images folder in the output
    directory.

    Args:
        nbp_basic: `basic_info` notebook page.
        nbp_file: `file_names` notebook page.
        nbp_extract: `extract` notebook page.
        nbp_register: unfinished `register` notebook page.
        nbp_register_debug: unfinished `register_debug` notebook page.
    """
    # initialise index variables
    use_tiles, use_rounds, use_channels = (
        list(nbp_basic.use_tiles),
        list(nbp_basic.use_rounds),
        list(nbp_basic.use_channels),
    )
    anchor_round, anchor_channel, dapi_channel = (
        nbp_basic.anchor_round,
        nbp_basic.anchor_channel,
        nbp_basic.dapi_channel,
    )

    # get the yxz coords for the central 500 x 500 x 10 region
    yx_centre = nbp_basic.tile_centre.astype(int)[:2]
    yx_radius = min(250, nbp_basic.tile_sz // 2)
    z_central_index = int(np.median(np.arange(len(nbp_basic.use_z))))
    if len(nbp_basic.use_z) <= 10:
        z_planes = np.arange(len(nbp_basic.use_z))
    else:
        z_planes = np.arange(z_central_index - 5, z_central_index + 5)

    tile_centre = (int(yx_centre[0]), int(yx_centre[1]))
    yxz_min = (tile_centre[0] - yx_radius, tile_centre[1] - yx_radius, int(z_planes[0]))
    yxz_max = (tile_centre[0] + yx_radius, tile_centre[1] + yx_radius, int(z_planes[-1]))
    yxz_coords = np.meshgrid(
        np.arange(yxz_min[0], yxz_max[0]),
        np.arange(yxz_min[1], yxz_max[1]),
        np.arange(yxz_min[2], yxz_max[2]),
        indexing="ij",
    )
    yxz_coords = np.array(yxz_coords).reshape((3, -1)).T
    image_shape = tuple([yxz_max[i] - yxz_min[i] for i in range(3)])

    # initialise zarr arrays to store the images
    anchor_images = zarr.open_array(
        os.path.join(nbp_file.output_dir, "anchor_reg_images.zarr"),
        dtype=np.uint8,
        shape=(max(use_tiles) + 1, 2) + image_shape,
        chunks=(1, 1) + image_shape,
    )
    round_images = zarr.open_array(
        os.path.join(nbp_file.output_dir, "round_reg_images.zarr"),
        dtype=np.uint8,
        shape=(max(use_tiles) + 1, max(use_rounds) + 1, 3) + image_shape,
        chunks=(1, 1, 1) + image_shape,
    )
    channel_images = zarr.open_array(
        os.path.join(nbp_file.output_dir, "channel_reg_images.zarr"),
        dtype=np.uint8,
        shape=(max(use_tiles) + 1, max(use_channels) + 1, 3) + image_shape,
        chunks=(1, 1, 1) + image_shape,
    )

    anchor_round_active_channels = [dapi_channel, anchor_channel]
    for t, c in tqdm(product(use_tiles, anchor_round_active_channels), desc="Anchor Images", total=len(use_tiles) * 2):
        im = nbp_filter.images[
            t, anchor_round, c, yxz_min[0] : yxz_max[0], yxz_min[1] : yxz_max[1], yxz_min[2] : yxz_max[2]
        ]
        im = fill_to_uint8(im)
        sub_index = 0 if c == dapi_channel else 1
        anchor_images[t, sub_index] = im
    nbp_register.anchor_images = anchor_images

    # get the round images, apply optical flow, apply icp + optical flow, concatenate and save
    for t, r in tqdm(product(use_tiles, use_rounds), desc="Round Images", total=len(use_tiles) * len(use_rounds)):
        im_tr = nbp_filter.images[
            t, r, dapi_channel, yxz_min[0] : yxz_max[0], yxz_min[1] : yxz_max[1], yxz_min[2] : yxz_max[2]
        ]
        # TODO: The below code doesn't work (seems to return blank image) - need to debug
        im_tr_flow = spot_colors.base.get_spot_colours_new(
            nbp_filter.images,
            nbp_register.flow,
            nbp_register.icp_correction,
            nbp_register_debug.channel_correction,
            nbp_basic.use_channels,
            nbp_basic.dapi_channel,
            t,
            r,
            dapi_channel,
            yxz=yxz_coords,
            registration_type="flow",
        ).reshape((1,) + image_shape)
        im_tr_flow_icp = spot_colors.base.get_spot_colours_new(
            nbp_filter.images,
            nbp_register.flow,
            nbp_register.icp_correction,
            nbp_register_debug.channel_correction,
            nbp_basic.use_channels,
            nbp_basic.dapi_channel,
            t,
            r,
            dapi_channel,
            yxz=yxz_coords,
            registration_type="flow_and_icp",
        ).reshape((1,) + image_shape)
        im_tr_concat = np.concatenate([im_tr[None], im_tr_flow, im_tr_flow_icp], axis=0)
        im_tr_concat = fill_to_uint8(im_tr_concat)
        round_images[t, r] = im_tr_concat
    nbp_register.round_images = round_images

    # get the channel images, save, apply optical flow, save, apply icp, save
    r_mid = 3
    for t, c in tqdm(product(use_tiles, use_channels), desc="Channel Images", total=len(use_tiles) * len(use_channels)):
        im_tc = nbp_filter.images[t, r_mid, c, yxz_min[0] : yxz_max[0], yxz_min[1] : yxz_max[1], yxz_min[2] : yxz_max[2]]
        # TODO: The below code doesn't work (seems to return blank image) - need to debug
        im_tc_flow = spot_colors.base.get_spot_colours_new(
            nbp_filter.images,
            nbp_register.flow,
            nbp_register.icp_correction,
            nbp_register_debug.channel_correction,
            nbp_basic.use_channels,
            nbp_basic.dapi_channel,
            t,
            r_mid,
            c,
            yxz=yxz_coords,
            registration_type="flow",
        ).reshape((1,) + image_shape)
        im_tc_flow_icp = spot_colors.base.get_spot_colours_new(
            nbp_filter.images,
            nbp_register.flow,
            nbp_register.icp_correction,
            nbp_register_debug.channel_correction,
            nbp_basic.use_channels,
            nbp_basic.dapi_channel,
            t,
            r_mid,
            c,
            yxz=yxz_coords,
            registration_type="flow_and_icp",
        ).reshape((1,) + image_shape)
        im_tc_concat = np.concatenate([im_tc[None], im_tc_flow, im_tc_flow_icp], axis=0)
        im_tc_concat = fill_to_uint8(im_tc_concat)
        channel_images[t, c] = im_tc_concat
    nbp_register.channel_images = channel_images


# TODO: Get rid of this function
def load_transformed_image(
    nb: Notebook,
    t: int,
    r: int,
    c: int,
    yxz: Optional[list] = None,
    reg_type: str = "none",
) -> np.ndarray:
    """
    Load the image from tile t, round r, channel c, apply the relevant registration and return the image.

    Args:
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
    im = nb.filter.images[t, r, c].astype(np.float32)
    # anchor round has no flow or affine correction so can return early
    if reg_type == "none" or r == nb.basic_info.anchor_round:
        return im

    # If we get this far, we will either be doing flow or flow icp, and we will not be in the anchor round.
    # These differ only by the affine correction we apply before.
    if yxz is not None:
        new_origin = np.array([yxz[0][0], yxz[1][0], yxz[2][0]])
    else:
        new_origin = np.zeros(3, dtype=int)
    affine_correction = np.eye(4, 3)
    if "reg_type" == "flow":
        if c != nb.basic_info.dapi_channel:
            affine_correction = nb.register_debug.channel_correction[t, c].copy()
    elif reg_type == "flow_icp":
        if c == nb.basic_info.dapi_channel:
            affine_correction = nb.register.icp_correction[t, r, nb.basic_info.anchor_channel].copy()
        if c != nb.basic_info.dapi_channel:
            affine_correction = nb.register.icp_correction[t, r, c].copy()
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
    im = transform_im(im=im, affine=affine_correction, flow=nb.register.flow[t, r], flow_ind=flow_indices)

    return im


# TODO: Get rid of this function
def transform_im(im: np.ndarray, affine: np.ndarray, flow: zarr.Array, flow_ind: Union[tuple, None]) -> np.ndarray:
    """
    Function to apply affine and flow transformations to an image.

    Args:
        im: image to transform
        affine: 3 x 4 affine transform
        flow: flow as zarr array
        flow_ind: indices to take from the flow file. If None, use the entire flow file.
    """
    assert type(flow) is zarr.Array or type(flow) is np.ndarray

    im = affine_transform(im, affine, order=1, mode="constant", cval=0)
    if flow_ind is not None:
        flow = flow[flow_ind].astype(np.float32)
    else:
        flow = flow.astype(np.float32)
    coords = np.meshgrid(
        np.arange(im.shape[0], dtype=np.float32),
        np.arange(im.shape[1], dtype=np.float32),
        np.arange(im.shape[2], dtype=np.float32),
        indexing="ij",
    )
    im = warp(im, coords + flow, order=1, mode="constant", cval=0, preserve_range=True)
    return im


def adjust_affine(affine: np.ndarray, new_origin: np.ndarray) -> np.ndarray:
    """
    Adjusts the affine transform for a new origin, then converts from 4 x 3 to 3 x 4 format.

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


def fill_to_uint8(array: np.ndarray) -> np.ndarray[np.uint8]:
    """
    Take a numpy array, scale/shift the array to take up the uint8 range. If the array is a single pixel, then that
    pixel is set to 0.

    Args:
        array: array to shift and scale.

    Returns:
        (ndarray[uint8]): shifted/scaled array.
    """
    assert array.size > 0, "Given array cannot be empty"

    im_min, im_max = np.min(array), np.max(array)
    array = array - im_min
    # Save the image as uint8
    if im_max != 0:
        array = array / np.max(array) * 255  # Scale to 0-255
    array = array.astype(np.uint8)
    return array


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
