from functools import partial
import numpy as np
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import os

from ..setup.notebook import NotebookPage
from ..utils import tiles_io
from .. import logging


def apply_transform(yxz: jnp.ndarray, flow: jnp.ndarray, icp_correction: jnp.ndarray,
                    tile_sz: jnp.ndarray) -> Tuple[np.ndarray, jnp.ndarray]:
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
    # load in shifts for each pixel
    y_indices, x_indices, z_indices = yxz.T
    # apply shifts to each pixel
    yxz_shifts = -flow[:, y_indices, x_indices, z_indices].T
    yxz_transform = jnp.asarray(yxz + yxz_shifts)
    # apply icp correction
    yxz_transform = jnp.pad(yxz_transform, ((0, 0), (0, 1)), constant_values=1)
    yxz_transform = jnp.round(yxz_transform @ icp_correction).astype(np.int16)
    in_range = jnp.logical_and((yxz_transform >= jnp.array([0, 0, 0])).all(axis=1),
                              (yxz_transform < tile_sz).all(axis=1))  # set color to nan if out range
    return yxz_transform, in_range


def get_spot_colors(yxz_base: jnp.ndarray, t: jnp.ndarray, transform: jnp.ndarray, bg_scale: jnp.ndarray, file_type: str,
                    nbp_file: NotebookPage, nbp_basic: NotebookPage, use_rounds: Optional[List[int]] = None,
                    use_channels: Optional[List[int]] = None, return_in_bounds: bool = False,
                    ) -> Union[np.ndarray, Tuple[np.ndarray, jnp.ndarray]]:
    """
    Takes some spots found on the reference round, and computes the corresponding spot intensity
    in specified imaging rounds/channels.
    By default, will run on `nbp_basic.use_rounds` and `nbp_basic.use_channels`.

    Args:
        yxz_base: `int16 [n_spots x 3]`.
            Local yxz coordinates of spots found in the reference round/reference channel of tile `t`
            yx coordinates are in units of `yx_pixels`. z coordinates are in units of `z_pixels`.
        t: `int`. Tile number.
        transform: `float32 [n_tiles x n_rounds x n_channels x 4 x 3]`.
        bg_scale: `float32 [n_tiles x n_rounds x n_channels]` scale factors to apply to background images before
        subtraction.if 'None', no background subtraction will be performed.
        file_type: `str`. Type of file to read in. E.g. '.zarr' or '.npy'.
        nbp_file: `file_names` notebook page.
        nbp_basic: `basic_info` notebook page.
        use_rounds: `int [n_use_rounds]`.
            Rounds you would like to find the `spot_color` for.
            Error will raise if transform is zero for particular round.
            If `None`, all rounds in `nbp_basic.use_rounds` used.
        use_channels: `int [n_use_channels]`.
            Channels you would like to find the `spot_color` for.
            Error will raise if transform is zero for particular channel.
            If `None`, all channels in `nbp_basic.use_channels` used.
        return_in_bounds: `bool`. If 'True' will only return spots that are in the bounds of the tile.

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
        use_rounds = nbp_basic.use_rounds + [nbp_basic.pre_seq_round] * nbp_basic.use_preseq
    if use_channels is None:
        use_channels = nbp_basic.use_channels

    n_spots = yxz_base.shape[0]
    no_verbose = n_spots < 10000
    # note using nan means can't use integer even though data is integer
    n_use_rounds = len(use_rounds)
    n_use_channels = len(use_channels)
    # spots outside tile bounds on particular r/c will initially be set to 0.
    spot_colors = jnp.zeros((n_spots, n_use_rounds, n_use_channels), dtype=np.int32)
    if not nbp_basic.is_3d:
        # use numpy not jax.numpy as reading in tiff is done in numpy.
        tile_sz = jnp.asarray([nbp_basic.tile_sz, nbp_basic.tile_sz, 1], dtype=np.int16)
    else:
        tile_sz = jnp.asarray([nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)], dtype=np.int16)

    with tqdm(total=n_use_rounds * n_use_channels, disable=no_verbose) as pbar:
        pbar.set_description(f"Reading {n_spots} spot_colors, from {file_type} files")
        for r in range(n_use_rounds):
            flow_r = jnp.load(os.path.join(nbp_file.output_dir, 'flow', 'smooth', f't{t}_r{use_rounds[r]}.npy'),
                             mmap_mode='r')
            for c in range(n_use_channels):
                transform_rc = transform[t, r, use_channels[c]]
                pbar.set_postfix({'round': use_rounds[r], 'channel': use_channels[c]})
                if transform_rc[0, 0] == 0:
                    raise ValueError(
                        f"Transform for tile {t}, round {use_rounds[r]}, channel {use_channels[c]} is zero:"
                        f"\n{transform_rc}")

                yxz_transform, in_range = apply_transform(yxz_base, flow_r, transform_rc, tile_sz)
                yxz_transform, in_range = jnp.asarray(yxz_transform), jnp.asarray(in_range)
                yxz_transform = yxz_transform[in_range]
                # if no spots in range, then continue
                if yxz_transform.shape[0] == 0:
                    pbar.update(1)
                    continue

                # Read in the shifted uint16 colors here, and remove shift later.
                spot_colors[in_range, r, c] = tiles_io.load_image(nbp_file, nbp_basic, file_type, t,
                                                                  use_rounds[r], use_channels[c], yxz_transform,
                                                                  apply_shift=False)
                pbar.update(1)
            del flow_r

    # Remove shift so now spots outside bounds have color equal to - nbp_basic.tile_pixel_shift_value.
    # It is impossible for any actual spot color to be this due to clipping at the extract stage.
    spot_colors = spot_colors - nbp_basic.tile_pixel_value_shift
    colours_valid = (spot_colors > -nbp_basic.tile_pixel_value_shift).all(axis=(1, 2))
    
    if return_in_bounds:
        spot_colors = spot_colors[colours_valid]
        yxz_base = yxz_base[colours_valid]

    # if we are using bg colours, address that here
    if bg_scale is not None:
        bg_colours = jnp.repeat(spot_colors[:, -1, :][:, None, :], n_use_rounds - 1, axis=1)
        bg_colours *= bg_scale[t][np.ix_(use_rounds[:-1], use_channels)][None, :, :]
        spot_colors = spot_colors[:, :-1, :]
        spot_colors = spot_colors - bg_colours

    output_tuple = (spot_colors, yxz_base)
    if bg_scale is not None:
        output_tuple += (bg_colours,)
    if not return_in_bounds:
        output_tuple += (colours_valid,)

    return output_tuple


def all_pixel_yxz(y_size: int, x_size: int, z_planes: Union[List, int, np.ndarray]) -> jnp.ndarray:
    """
    Returns the yxz coordinates of all pixels on the indicated z-planes of an image.

    Args:
        y_size: number of pixels in y direction of image.
        x_size: number of pixels in x direction of image.
        z_planes: `int [n_z_planes]` z_planes, coordinates are desired for.

    Returns:
        `int16 [y_size * x_size * n_z_planes, 3]`
            yxz coordinates of all pixels on `z_planes`.
    """
    if isinstance(z_planes, int):
        z_planes = jnp.array([z_planes])
    elif isinstance(z_planes, list):
        z_planes = jnp.array(z_planes)
    return jnp.array(jnp.meshgrid(jnp.arange(y_size), jnp.arange(x_size), z_planes), dtype=jnp.int16).T.reshape(-1, 3)
