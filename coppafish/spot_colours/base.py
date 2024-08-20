from typing import Any, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm
import zarr


def apply_flow(yxz: Union[np.ndarray, torch.Tensor],
               flow: Union[np.ndarray, torch.Tensor],
               top_left: Union[np.ndarray, torch.Tensor] = np.array([0, 0, 0])) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply a flow to a set of points. Note that this is applying forward warping, meaning that
        new_points = points + flow.
    The flow we pass in may be cropped, so if this is the case, to sample the points correctly, we need to
        make the points relative to the top left corner of the cropped image.
    Args:
        yxz: integer points to apply the warp to. (n_points x 3 in yxz coords) (UNSHIFTED)
        flow: flow to apply to the points. (3 x cube_size_y x cube_size_x x cube_size_z) (SHIFTED)
        top_left: the top left corner of the cube in the flow_image. (3 in yxz coords) Default: [0, 0, 0]

    Returns:
        yxz_flow: (float) new points. (n_points x 3 in yxz coords)
    """
    # First, make yxz coordinates relative to the top left corner of the flow image, so that we can sample the shifts
    yxz_relative = yxz - top_left
    y_indices_rel, x_indices_rel, z_indices_rel = yxz_relative.T
    # sample the shifts relative to the top left corner of the flow image
    yxz_shifts = np.array([flow[i, y_indices_rel, x_indices_rel, z_indices_rel] for i in range(3)]).astype(np.float32).T
    # if original coords are torch, make the shifts torch
    if type(yxz) is torch.Tensor:
        yxz_shifts = torch.tensor(yxz_shifts)
    # apply the shifts to the original points
    yxz_flow = yxz + yxz_shifts
    return yxz_flow


def apply_affine(yxz: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """
    This transforms the coordinates yxz based on the affine transform alone.
    E.g. to find coordinates of spots on the same tile but on a different round and channel.

    Args:
        yxz: ```int [n_spots x 3]```.
            ```yxz[i, :2]``` are the non-centered yx coordinates in ```yx_pixels``` for spot ```i```.
            ```yxz[i, 2]``` is the non-centered z coordinate in ```z_pixels``` for spot ```i```.
        affine: '''float32 [4 x 3]'''. The affine transform to apply to the coordinates.

    Returns:
        ```int [n_spots x 3]```.
            ```yxz_transform``` such that
            ```yxz_transform[i, [1,2]]``` are the transformed non-centered yx coordinates in ```yx_pixels```
            for spot ```i```.
            ```yxz_transform[i, 2]``` is the transformed non-centered z coordinate in ```z_pixels``` for spot ```i```.
    """
    # apply icp correction
    yxz_pad = torch.cat([yxz, torch.ones(yxz.shape[0], 1)], dim=1).float()
    yxz_transform = yxz_pad @ affine
    return yxz_transform


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
        yxz_mins -= torch.asarray([1, 1, 1])
        yxz_mins = torch.clamp(yxz_mins, torch.zeros(3), torch.asarray(image_shape)).int()
        yxz_maxs = from_yxz.max(dim=0)[0].max(dim=0)[0] + 1
        yxz_maxs = torch.ceil(((yxz_maxs / torch.asarray(half_pixels)) - 1) * 0.5)
        yxz_maxs += torch.asarray([1, 1, 1])
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


def get_spot_colours(
    image: Union[np.ndarray, zarr.Array],
    flow: Union[np.ndarray, zarr.Array],
    affine_correction: Union[np.ndarray, torch.Tensor],
    yxz_base: Union[np.ndarray, torch.Tensor],
    output_dtype: torch.dtype = torch.float32,
    fill_value: float = float('nan'),
    tile: int = None,
    use_channels: List[int] = None,
) -> np.ndarray:
    """
    Takes some spots found on the reference round, and computes the corresponding spot intensity
    in specified imaging rounds/channels. The algorithm goes as follows:
    Loop over rounds:
        Apply flow: yxz_flow = yxz_base + flow
        Loop over channels:
            Apply ICP correction: yxz_flow_and_affine = yxz_flow @ icp_correction
            Interpolate spot intensities: spot_colours[:, r, c] = grid_sample(image[r, c], yxz_flow_and_affine)
    The code has been profiled, and any time-consuming operations have been passed to PyTorch and can be run on a GPU.

    - Note: Although yxz is a list of n_spots x 3 and does not need to be made up of intervals, we load the bounding
    box of the image to speed up the loading process and help interpolate points. This means that accessing many random
    points will be slower than accessing a subset of the image at once.

    Args:
        image: 'float16 memmap [n_tiles x n_rounds x n_channels x im_y x im_x x im_z]' unregistered image data.
        flow: 'float16 memmap [n_tiles x n_rounds x 3 x im_y x im_x x im_z]' flow data.
        affine_correction: 'float32 [n_tiles x n_rounds x n_channels x 4 x 3]' affine correction data, or
        [n_rounds x n_channels x 4 x 3] for the tile of interest or
        [n_channels x 4 x 3] if round independent.
        yxz_base: 'int [n_spots x 3]' spot coordinates, or tuple
        output_dtype: 'dtype' dtype of the output spot colours.
        fill_value: 'float' value to fill in for out of bounds spots.
        tile: 'int' tile index to run on.
        use_channels: 'List[int]' channels to run on.

    Returns:
        spot_colours: 'output_dtype [n_spots x n_rounds x n_channels]' spot colours.
    """
    # deal with none values
    if tile is None:
        tile = 0
    if use_channels is None:
        use_channels = list(range(image.shape[2]))
    if type(affine_correction) is np.ndarray:
        affine_correction = torch.tensor(affine_correction, dtype=torch.float32)
    if type(yxz_base) is np.ndarray:
        yxz_base = torch.tensor(yxz_base, dtype=torch.float32)
    if np.ndim(affine_correction) == 3:
        # repeat n_rounds times
        affine_correction = affine_correction[None].repeat(image.shape[1], 1, 1, 1)
        # repeat n_tiles times
        affine_correction = affine_correction[None].repeat(image.shape[0], 1, 1, 1, 1)

    # initialize variables
    n_spots, n_use_rounds, n_use_channels = yxz_base.shape[0], flow.shape[1], len(use_channels)
    use_rounds = list(np.arange(n_use_rounds))
    tile_size = torch.tensor(image.shape[3:])
    pad_size = torch.tensor([100, 100, 5])
    spot_colours = torch.full((n_spots, n_use_rounds, n_use_channels), fill_value, dtype=output_dtype)

    # load slices of the images rather than sampling coordinates directly.
    yxz_min, yxz_max = yxz_base.min(axis=0).values.int(), yxz_base.max(axis=0).values.int()
    # pad to ensure that we are able to interpolate the points even if the shifts are large.
    yxz_min, yxz_max = (torch.maximum(yxz_min - pad_size, torch.tensor([0, 0, 0])),
                        torch.minimum(yxz_max + pad_size, tile_size))
    cube_size = yxz_max - yxz_min
    # load the sliced images for each round and channel (from yxz_min to yxz_max)
    image = np.array(
        [[image[tile, r, c, yxz_min[0]:yxz_max[0], yxz_min[1]:yxz_max[1], yxz_min[2]:yxz_max[2]] for c in use_channels]
         for r in use_rounds]
    )
    flow = np.array(
        [flow[tile, r, :, yxz_min[0]:yxz_max[0], yxz_min[1]:yxz_max[1], yxz_min[2]:yxz_max[2]] for r in use_rounds]
    )
    # convert to torch tensor
    image = torch.tensor(image, dtype=torch.float32)

    # begin the loop over rounds and channels
    for r in tqdm(range(n_use_rounds), total=n_use_rounds, desc="Round Loop"):
        # initialize the coordinates for the round
        yxz_round_r = torch.zeros((n_use_channels, n_spots, 3), dtype=torch.float32)
        # the flow is the same for all channels in the same round. Therefore, only need to read it once.
        # Since flow is cropped, pass the top left corner to this function, so it reads the coords relative to yxz_min.
        yxz_flow = apply_flow(yxz=yxz_base.int(), flow=flow[r], top_left=yxz_min)
        for i, c in enumerate(use_channels):
            # apply the affine transform to the spots
            yxz_round_r[i] = apply_affine(yxz=yxz_flow, affine=affine_correction[tile, r, c])
            # Since image has top left corner yxz_min, must make the sampling points relative to this.
            yxz_round_r[i] -= yxz_min
            # convert tile coordinates [0, cube_size] to coordinates [0, 2]
            yxz_round_r[i] = 2 * yxz_round_r[i] / (cube_size - 1)
            # convert coordinates [0, 2] to coordinates [-1, 1]
            yxz_round_r[i] -= 1
        zxy_round_r = yxz_round_r[:, :, [2, 1, 0]]

        # grid_sample expects image to be input as [N, M, D, H, W] where
        # N = batch size: We set this to n_use_channels,
        # M = number of images to be sampled at the same grid locations: We set this to 1,
        # D = depth, H = height, W = width: We set these to n_y, n_x and n_z respectively.

        # grid_sample expects grid to be input as [N, D', H', W', 3] where
        # N = batch size: We set this to n_use_channels,
        # D' = depth out, H' = height out, W' = width out: We set these to n_spots, 1, 1
        # 3 = 3D coordinates of the points to sample (NOTE: These must be in the order z, x, y).
        # This is NOT included in the documentation, but is inferred from the source code.
        round_r_colours = torch.nn.functional.grid_sample(
            input=image[r, :, None, :, :, :],
            grid=zxy_round_r[:, :, None, None, :],
            mode='bilinear',
            align_corners=False,
        )

        # grid_sample gives output as [N, M, D', H', W'] as defined above.
        round_r_colours = round_r_colours[:, 0, :, 0, 0]
        spot_colours[:, r, :] = round_r_colours.T

    return spot_colours.numpy()
