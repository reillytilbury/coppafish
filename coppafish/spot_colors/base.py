from typing import Any, List, Optional, Tuple, Union
from ..setup import NotebookPage
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm
import zarr


def apply_transform(
    yxz: np.ndarray, flow: np.ndarray, icp_correction: np.ndarray, tile_sz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This transforms the coordinates yxz based on the flow and icp_correction.
    E.g. to find coordinates of spots on the same tile but on a different round and channel.

    Args:
        yxz: ```int [n_spots x 3]```.
            ```yxz[i, :2]``` are the non-centered yx coordinates in ```yx_pixels``` for spot ```i```.
            ```yxz[i, 2]``` is the non-centered z coordinate in ```z_pixels``` for spot ```i```.
            E.g. these are the coordinates stored in ```nb['find_spots']['spot_details']```.
        flow: '''float mem-map [3 x n_pixels_y x n_pixels_x x n_pixels_z]''' of shifts for each pixel.
        icp_correction: '''float32 [4 x 3]'''. The affine transform to apply to the coordinates after the flow.
        tile_sz: ```int16 [3]```.
            YXZ dimensions of tile

    Returns:
        ```int [n_spots x 3]```.
            ```yxz_transform``` such that
            ```yxz_transform[i, [1,2]]``` are the transformed non-centered yx coordinates in ```yx_pixels```
            for spot ```i```.
            ```yxz_transform[i, 2]``` is the transformed non-centered z coordinate in ```z_pixels``` for spot ```i```.
        - ```in_range``` - ```bool [n_spots]```.
            Whether spot s was in the bounds of the tile when transformed to round `r`, channel `c`.
    """
    if flow is not None:
        # load in shifts for each pixel
        y_indices, x_indices, z_indices = yxz.T
        # apply shifts to each pixel
        yxz_shifts = (-flow[:, y_indices, x_indices, z_indices].T).astype(np.float32)
        yxz = np.asarray(yxz + yxz_shifts)
    # apply icp correction
    yxz_transform = np.pad(yxz, ((0, 0), (0, 1)), constant_values=1)
    yxz_transform = np.round(yxz_transform @ icp_correction).astype(np.int16)
    in_range = np.logical_and(
        (yxz_transform >= np.array([0, 0, 0])).all(axis=1), (yxz_transform < tile_sz).all(axis=1)
    )  # set color to nan if out range
    return yxz_transform, in_range


def get_spot_colours_new(
    all_images: Union[zarr.Array, np.ndarray],
    flow: zarr.Array,
    icp_correction: np.ndarray,
    channel_correction: np.ndarray,
    use_channels: List[int],
    dapi_channel: int,
    tile: int,
    round: int,
    channels: Optional[Union[Tuple[int], int]] = None,
    yxz: npt.NDArray[np.int_] = None,
    registration_type: str = "flow_and_icp",
    dtype: np.dtype = np.float32,
    force_cpu: bool = True,
) -> npt.NDArray[Any]:
    """
    Load and return registered sequence image(s) colours. Image(s) are correctly centred around zero.

    Args:
        all_images (`(n_tiles x n_rounds x n_channels x im_y x im_x x im_z) zarray or ndarray`): all filtered images.
            all_images[t][r][c] is for tile t, round r, channel c.
        flow (`(n_tiles x n_rounds x 3 x im_y x im_x x im_z) zarray or ndarray`): optical flow shifts that take the
            anchor image to the tile/round.
        icp_correction (`(n_tiles x n_rounds x n_channels x 4 x 3) ndarray[float]`): affine correction applied to
            anchor image after optical flow correction.
        channel_correction (`(n_tiles x n_channels x 4 x 3) ndarray[float]`): affine channel correction applied to
            anchor image after optical flow correction.
        use_channels (list of int): all sequencing channels.
        dapi_channel (int): the dapi channel index. Only the channel correction affine is used on the DAPI channel.
        tile (int): tile index.
        round (int): round index.
        channels (tuple of ints or int): sequence channel indices (index) to load. Default: sequencing channels.
        yxz (`(n_points x 3) ndarray[int]`): specific points to retrieve relative to the reference image (the anchor
            round/channel). Default: the entire image in the order such that
            `image.reshape((len(channels) x im_y x im_x x im_z))` will return the entire image.
        registration_type (str): registration method to apply up to. Can be 'flow' or 'flow_and_icp'. Default:
            'flow_and_icp', the completed registration, after optical flow and ICP corrections.
        dtype (dtype): data type to return the images into. This must support negative numbers. An integer dtype will
            cause rounding errors. Default: np.float32.
        force_cpu (bool): only use a CPU to run computations on. Default: true.

    Returns:
        `(len(channels) x n_points) ndarray[dtype]`) image: registered image intensities. Any out of bounds values are
            set to nan.

    Notes:
        - If you are planning to load in multiple channels, this function is faster if called once.
        - float32 precision is used throughout intermediate steps. If you do not have sufficient memory, then yxz can
            be used to run on fewer points at once.
    """
    assert type(all_images) is np.ndarray or type(all_images) is zarr.Array
    assert all_images.ndim == 6
    image_shape = all_images.shape[3:]
    assert type(flow) is zarr.Array or type(flow) is np.ndarray
    assert flow.shape[2:] == (3,) + image_shape, f"{flow.shape=} and {all_images.shape=}"
    assert type(icp_correction) is np.ndarray
    assert icp_correction.shape[3:] == (4, 3)
    assert type(channel_correction) is np.ndarray
    assert channel_correction.shape[2:] == (4, 3)
    assert type(use_channels) is list
    assert type(dapi_channel) is int
    assert type(tile) is int, f"Type of tile is {type(tile)}, but should be int."
    assert type(round) is int
    if channels is None:
        channels = tuple(use_channels)
    if type(channels) is int:
        channels = (channels,)
    assert type(channels) is tuple
    assert len(channels) > 0
    if yxz is None:
        yxz = [np.linspace(0, image_shape[i], image_shape[i], endpoint=False) for i in range(3)]
        yxz = np.array(np.meshgrid(*yxz, indexing="ij")).reshape((3, -1)).astype(np.int32).T
    assert type(yxz) is np.ndarray
    assert yxz.shape[0] > 0
    assert yxz.shape[1] == 3
    assert registration_type in ("flow", "flow_and_icp")

    run_on = torch.device("cpu")
    if not force_cpu and torch.cuda.is_available():
        run_on = torch.device("cuda")
    n_points = yxz.shape[0]
    # Half a pixel in the pytorch grid_sample world is represented by this small distance. This is used to convert
    # yxz positions into the pytorch positions later on.
    half_pixels = [1 / image_shape[i] for i in range(3)]

    def get_yxz_bounds(from_yxz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the minimum and maximum yxz positions from pytorch world coordinates. This is used to help gather only
        # a subset of the images when disk loading.
        assert from_yxz.ndim == 3
        assert from_yxz.shape[-1] == 3
        yxz_mins = from_yxz.min(dim=0)[0].min(dim=0)[0] + 1
        yxz_mins = torch.floor(((yxz_mins / torch.asarray(half_pixels)) - 1) * 0.5)
        yxz_mins -= torch.asarray([5, 5, 1])
        yxz_mins = torch.clamp(yxz_mins, torch.zeros(3), torch.asarray(image_shape)).int()
        yxz_maxs = from_yxz.max(dim=0)[0].max(dim=0)[0] + 1
        yxz_maxs = torch.ceil(((yxz_maxs / torch.asarray(half_pixels)) - 1) * 0.5)
        yxz_maxs += torch.asarray([5, 5, 1])
        yxz_maxs = torch.clamp(yxz_maxs, torch.zeros(3), torch.asarray(image_shape)).int()
        assert yxz_mins.shape == (3,)
        assert yxz_maxs.shape == (3,)
        return yxz_mins, yxz_maxs

    # 1: Affine transform every pixel position, keep them as floating points.
    yxz_registered = torch.zeros((0, n_points, 3), dtype=torch.float32)
    for c in channels:
        affine = np.eye(4, 3)
        if registration_type == "flow_and_icp" and c != dapi_channel:
            affine = icp_correction[tile, round, c].copy()
        elif registration_type == "flow":
            affine = channel_correction[tile, c].copy()
        assert affine.shape == (4, 3)
        affine.flags.writeable = True
        affine = torch.asarray(affine).float()
        yxz_affine_c = torch.asarray(yxz).float()
        yxz_affine_c = torch.nn.functional.pad(yxz_affine_c, (0, 1, 0, 0), "constant", 1)
        yxz_affine_c = yxz_affine_c.to(run_on)
        affine = affine.to(run_on)
        yxz_affine_c = torch.matmul(yxz_affine_c, affine)
        yxz_affine_c = yxz_affine_c.cpu()
        affine = affine.cpu()
        assert yxz_affine_c.shape == (n_points, 3)
        for i in range(3):
            yxz_affine_c[:, i] = (2 * yxz_affine_c[:, i] + 1) * half_pixels[i]
            yxz_affine_c[:, i] -= 1
        yxz_registered = torch.cat((yxz_registered, yxz_affine_c[np.newaxis]), dim=0).float()
        del yxz_affine_c, affine
    del yxz

    if registration_type == "flow_and_icp":
        # 2: Gather the optical flow shifts using interpolation from the affine positions.
        # (3, 1, im_y, im_x, im_z).
        flow_image = torch.zeros((3, 1) + image_shape).float()
        yxz_minimums, yxz_maximums = get_yxz_bounds(yxz_registered)
        flow_image[
            :,
            0,
            yxz_minimums[0] : yxz_maximums[0],
            yxz_minimums[1] : yxz_maximums[1],
            yxz_minimums[2] : yxz_maximums[2],
        ] = torch.asarray(
            flow[
                tile,
                round,
                :,
                yxz_minimums[0] : yxz_maximums[0],
                yxz_minimums[1] : yxz_maximums[1],
                yxz_minimums[2] : yxz_maximums[2],
            ]
        )
        del yxz_minimums, yxz_maximums
        # The flow image takes the anchor image -> tile/round image so must invert the shift.
        flow_image = torch.negative(flow_image)
        # (1, 1, len(channels), n_points, 3). yxz becomes zxy to use the grid_sample function correctly.
        yxz_registered = yxz_registered[np.newaxis, np.newaxis, :, :, [2, 1, 0]]
        # The affine must be applied to each optical flow shift direction.
        yxz_registered = yxz_registered.repeat_interleave(3, dim=0)
        optical_flow_shifts = torch.nn.functional.grid_sample(
            flow_image, yxz_registered, mode="bilinear", align_corners=False
        )[:, 0, 0]
        yxz_registered = yxz_registered[0, :, :, :, [2, 1, 0]]
        del flow_image
        # (len(channels), n_points, 3)
        optical_flow_shifts = optical_flow_shifts.movedim(0, 1).movedim(1, 2)
        assert optical_flow_shifts.shape == (len(channels), n_points, 3)
        # Convert optical flow pixel shifts to grid sized shifts based on pytorch's grid_sample function.
        for i in range(3):
            optical_flow_shifts[:, :, i] *= half_pixels[i] * 2
        # 3: Apply optical flow shifts to the yxz affine positions for final registered positions.
        yxz_registered = yxz_registered[0, 0] + optical_flow_shifts
        del optical_flow_shifts

    # 4: Gather all unregistered channel image data.
    images = torch.zeros((len(channels),) + image_shape, dtype=torch.float32)
    for c_i, c in enumerate(channels):
        image_c = torch.zeros(image_shape).float()
        yxz_minimums, yxz_maximums = get_yxz_bounds(yxz_registered)
        yxz_subset = tuple([(yxz_minimums[i].item(), yxz_maximums[i].item()) for i in range(3)])
        image_c_subset = all_images[
            tile,
            round,
            c,
            yxz_subset[0][0] : yxz_subset[0][1],
            yxz_subset[1][0] : yxz_subset[1][1],
            yxz_subset[2][0] : yxz_subset[2][1],
        ]
        image_c[
            yxz_subset[0][0] : yxz_subset[0][1],
            yxz_subset[1][0] : yxz_subset[1][1],
            yxz_subset[2][0] : yxz_subset[2][1],
        ] = torch.asarray(image_c_subset).float()
        del yxz_minimums, yxz_maximums, yxz_subset, image_c_subset
        images[c_i] = torch.asarray(image_c).float()
        del image_c

    # 5: Use the yxz registered positions to gather from the images through interpolation.
    # (len(channels), 1, im_y, im_x, im_z)
    images = images[:, np.newaxis]
    # (len(channels), 1, 1, n_points, 3)
    yxz_registered = yxz_registered[:, np.newaxis, np.newaxis, :, [2, 1, 0]]
    out_of_bounds = torch.logical_or(yxz_registered < -1, yxz_registered > 1).any(dim=4)
    pixel_intensities = torch.nn.functional.grid_sample(images, yxz_registered, mode="bilinear", align_corners=False)
    pixel_intensities[out_of_bounds[:, np.newaxis]] = torch.nan
    # (len(channels), n_points)
    pixel_intensities = pixel_intensities[:, 0, 0, 0].numpy().astype(dtype)

    assert pixel_intensities.shape == (len(channels), n_points)
    return pixel_intensities


def remove_background(spot_colours: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes background from spot colours
    Args:
        spot_colours: 'float [n_spots x n_rounds x n_channels_use]' spot colours to remove background from.
    Returns:
        'spot_colours: [n_spots x n_rounds x n_channels_use]' spot colours with background removed.
        background_noise: [n_spots x n_channels_use]' background noise for each spot and channel.
    """
    n_spots = spot_colours.shape[0]
    background_noise = np.percentile(spot_colours, 25, axis=1)
    # Loop through all channels and remove the background from each channel.
    for c in tqdm(range(spot_colours.shape[2])):
        background_code = np.zeros(spot_colours[0].shape)
        background_code[:, c] = 1
        # Remove the component of the background from the spot colour for each spot
        spot_colours -= background_noise[:, c][:, None, None] * np.repeat(background_code[None], n_spots, axis=0)

    return spot_colours, background_noise


def get_spot_colors(
    yxz_base: np.ndarray,
    t: int,
    nbp_basic: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_register: NotebookPage,
    use_rounds: Optional[List[int]] = None,
    use_channels: Optional[List[int]] = None,
    return_in_bounds: bool = False,
    output_dtype: np.dtype = np.float32,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Takes some spots found on the reference round, and computes the corresponding spot intensity
    in specified imaging rounds/channels.
    By default, will run on `nbp_basic.use_rounds` and `nbp_basic.use_channels`.

    Args:
        yxz_base: `int16 [n_spots x 3]`.
            Local yxz coordinates of spots found in the reference round/reference channel of tile `t`
            yx coordinates are in units of `yx_pixels`. z coordinates are in units of `z_pixels`.
        t: `int`. Tile number.
        nbp_basic: `basic_info` notebook page.
        nbp_register: `register` notebook page.
        use_rounds: `int [n_use_rounds]`.
            Rounds you would like to find the `spot_color` for.
            Error will raise if transform is zero for particular round.
            If `None`, all rounds in `nbp_basic.use_rounds` are used.
        use_channels: `int [n_use_channels]`.
            Channels you would like to find the `spot_color` for.
            Error will raise if transform is zero for particular channel.
            If `None`, all channels in `nbp_basic.use_channels` used.
        return_in_bounds: `bool`. If 'True' will only return spots that are in the bounds of the tile.
        output_dtype (`np.dtype`): return the resulting colours in this data type.

    Returns:
        - `spot_colors` - `int32 [n_spots x n_rounds_use x n_channels_use]` or
            `int32 [n_spots_in_bounds x n_rounds_use x n_channels_use]`.
            `spot_colors[s, r, c]` is the spot color for spot `s` in round `use_rounds[r]`, channel `use_channels[c]`.
        - `yxz_base` - `int16 [n_spots_in_bounds x 3]` or `int16 [n_spots x 3]`.
        - `bg_colours` - `int32 [n_spots_in_bounds x n_rounds_use x n_channels_use]` or
        `int32 [n_spots x n_rounds_use x n_channels_use]`. (only returned if `bg_scale` is not `None`).
        - in_bounds - `bool [n_spots]`. Whether spot s was in the bounds of the tile when transformed to round `r`,
        channel `c`. (only returned if `return_in_bounds` is `True`).
    """
    if use_rounds is None:
        use_rounds = nbp_basic.use_rounds
    if use_channels is None:
        use_channels = nbp_basic.use_channels

    n_spots = yxz_base.shape[0]
    no_verbose = n_spots < 10_000
    # note using nan means can't use integer even though data is integer
    n_use_rounds = len(use_rounds)
    n_use_channels = len(use_channels)
    # spots outside tile bounds on particular r/c will initially be set to 0.
    spot_colors = np.zeros((n_spots, n_use_rounds, n_use_channels), dtype=np.float32) * np.nan
    if not nbp_basic.is_3d:
        # use numpy not jax.numpy as reading images outputs to numpy.
        tile_sz = np.asarray([nbp_basic.tile_sz, nbp_basic.tile_sz, 1], dtype=np.int16)
    else:
        tile_sz = np.asarray([nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)], dtype=np.int16)

    with tqdm(total=n_use_rounds * n_use_channels, disable=no_verbose) as pbar:
        pbar.set_description(f"Reading {n_spots} spot_colors")
        for i, r in enumerate(use_rounds):
            flow_tr = nbp_register.flow[t, r]
            for j, c in enumerate(use_channels):
                transform_rc = nbp_register.icp_correction[t, r, c]
                pbar.set_postfix({"round": r, "channel": c})
                if transform_rc[0, 0] == 0:
                    raise ValueError(f"Transform for tile {t}, round {r}, channel {c} is zero:" f"\n{transform_rc}")

                yxz_transform, in_range = apply_transform(yxz_base, flow_tr, transform_rc, tile_sz)
                yxz_transform, in_range = np.asarray(yxz_transform), np.asarray(in_range)
                yxz_transform = yxz_transform[in_range]
                # if no spots in range, then continue
                if yxz_transform.shape[0] == 0:
                    pbar.update(1)
                    continue

                # Read in the shifted float16 colours here, and remove shift later.
                spot_colors[in_range, i, j] = nbp_filter.images[t, r, c][tuple(yxz_transform.T)]
                spot_colors[in_range, i, j] += nbp_basic.tile_pixel_value_shift
                pbar.update(1)
                print(f"Round {r}, Channel {c} done.")
            del flow_tr

    # remove invalid spots (spots where any round or channel is nan)
    colours_valid = np.all(~np.isnan(spot_colors), axis=(1, 2))

    if return_in_bounds:
        spot_colors = spot_colors[colours_valid]
        yxz_base = yxz_base[colours_valid]

    output_tuple = (spot_colors.astype(output_dtype), yxz_base)
    if not return_in_bounds:
        output_tuple += (colours_valid,)

    return output_tuple