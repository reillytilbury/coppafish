from typing import List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
import zarr


def apply_flow(
    yxz: Union[np.ndarray, torch.Tensor],
    flow: Union[np.ndarray, torch.Tensor],
    top_left: Union[np.ndarray, torch.Tensor] = np.array([0, 0, 0]),
) -> Union[np.ndarray, torch.Tensor]:
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
    tile: int,
    output_dtype: torch.dtype = torch.float32,
    fill_value: float = float("nan"),
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
        - image: 'float16 memmap [n_tiles x n_rounds x n_channels x im_y x im_x x im_z]' unregistered image data.
        - flow: 'float16 memmap [n_tiles x n_rounds x 3 x im_y x im_x x im_z]' flow data.
        - affine_correction: 'float32 [n_tiles x n_rounds x n_channels x 4 x 3]' affine correction data
        - yxz_base: 'int [n_spots x 3]' spot coordinates, or tuple
        - tile: 'int' tile index to run on.
        - output_dtype: 'dtype' dtype of the output spot colours.
        - fill_value: 'float' value to fill in for out of bounds spots.
        - use_channels: 'List[int]' channels to run on.

    Returns:
        spot_colours: 'output_dtype [n_spots x n_rounds x n_channels]' spot colours.
    """
    # Deal with default values.
    if use_channels is None:
        use_channels = list(range(image.shape[2]))
    if type(affine_correction) is np.ndarray:
        affine_correction = torch.tensor(affine_correction, dtype=torch.float32)
    if type(yxz_base) is np.ndarray:
        yxz_base = torch.tensor(yxz_base, dtype=torch.float32)
    n_tiles, n_rounds, n_channels = image.shape[0], flow.shape[1], image.shape[2]
    assert affine_correction.shape[1:] == (n_rounds, n_channels, 4, 3), \
        f"Expected shape {(n_tiles, n_rounds, n_channels, 4, 3)}, got {affine_correction.shape}"

    # initialize variables
    n_spots, n_use_rounds, n_use_channels = yxz_base.shape[0], flow.shape[1], len(use_channels)
    use_rounds = list(np.arange(n_use_rounds))
    tile_size = torch.tensor(image.shape[3:])
    pad_size = torch.tensor([100, 100, 5])
    spot_colours = torch.full((n_spots, n_use_rounds, n_use_channels), fill_value, dtype=output_dtype)

    # load slices of the images rather than sampling coordinates directly.
    yxz_min, yxz_max = yxz_base.min(axis=0).values.int(), yxz_base.max(axis=0).values.int()
    # pad to ensure that we are able to interpolate the points even if the shifts are large.
    yxz_min, yxz_max = (
        torch.maximum(yxz_min - pad_size, torch.tensor([0, 0, 0])),
        torch.minimum(yxz_max + pad_size, tile_size),
    )
    cube_size = yxz_max - yxz_min
    # load the sliced images for each round and channel (from yxz_min to yxz_max)
    image = np.array(
        [
            [
                image[tile, r, c, yxz_min[0] : yxz_max[0], yxz_min[1] : yxz_max[1], yxz_min[2] : yxz_max[2]]
                for c in use_channels
            ]
            for r in use_rounds
        ]
    )
    flow = np.array(
        [
            flow[tile, r, :, yxz_min[0] : yxz_max[0], yxz_min[1] : yxz_max[1], yxz_min[2] : yxz_max[2]]
            for r in use_rounds
        ]
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
            mode="bilinear",
            align_corners=True,
            padding_mode="border"
        )

        # grid_sample gives output as [N, M, D', H', W'] as defined above.
        round_r_colours = round_r_colours[:, 0, :, 0, 0]
        spot_colours[:, r, :] = round_r_colours.T

        # Any out of bound grid sample retrievals are set to fill_value.
        is_out_of_bounds = torch.logical_or(zxy_round_r < -1, zxy_round_r > 1).any(dim=2).T
        spot_colours[:, r, :][is_out_of_bounds] = fill_value

    return spot_colours.numpy()
