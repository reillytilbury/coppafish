import numpy as np
from ..spot_colors import base as spot_colors_base
from ..call_spots import base as call_spots_base
from .. import find_spots as fs
from .. import log
from ..setup import NotebookPage


def get_reference_spots(
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_find_spots: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_register: NotebookPage,
    tile_origin: np.ndarray,
    icp_correction: np.ndarray,
) -> NotebookPage:
    """
    This takes each spot found on the reference round/channel and computes the corresponding intensity
    in each of the imaging rounds/channels.

    See `'ref_spots'` section of `notebook_comments.json` file
    for description of the variables in the page.
    The following variables:

    * `gene_no`
    * `score`
    * `score_diff`
    * `intensity`

    will be set to `None` so the page can be added to a *Notebook*. `call_reference_spots` should then be run
    to give their actual values. This is so if there is an error in `call_reference_spots`,
    `get_reference_spots` won't have to be re-run.

    Args:
        nbp_file: `file_names` notebook page.
        nbp_basic: `basic_info` notebook page.
        nbp_find_spots: 'find_spots' notebook page.
            Here we will use find_spots, spot_no and isolated_spots variables from this page
        nbp_extract: `extract` notebook page.
        nbp_filter: `filter` notebook page.
        tile_origin: `float [n_tiles x 3]`.
            `tile_origin[t,:]` is the bottom left yxz coordinate of tile `t`.
            yx coordinates in `yx_pixels` and z coordinate in `z_pixels`.
            This is saved in the `stitch` notebook page i.e. `nb.stitch.tile_origin`.
        icp_correction: `float [n_tiles x n_rounds x n_channels x 4 x 3]`.

    Returns:
        `NotebookPage[ref_spots]` - Page containing intensity of each reference spot on each imaging round/channel.
    """
    # We create a notebook page for ref_spots which stores information like local coords, isolated info, tile_no of each
    # spot and much more.
    nbp = NotebookPage("ref_spots")
    # The code is going to loop through all tiles, as we expect some anchor spots on each tile but r and c should stay
    # fixed as the value of the reference round and reference channel
    r = nbp_basic.anchor_round
    c = nbp_basic.anchor_channel
    log.debug("Get ref spots started")
    use_tiles, use_rounds, use_channels = (
        np.array(nbp_basic.use_tiles),
        list(nbp_basic.use_rounds),
        list(nbp_basic.use_channels),
    )

    # all means all spots found on the reference round / channel
    all_local_yxz = np.zeros((0, 3), dtype=np.int16)
    all_isolated = np.zeros(0, dtype=bool)
    all_local_tile = np.zeros(0, dtype=np.int16)

    # Now we start looping through tiles and recording the local_yxz spots on this tile and the isolated status of each
    # We then append this to our all_local_yxz, ... arrays
    for t in nbp_basic.use_tiles:
        t_local_yxz = fs.spot_yxz(nbp_find_spots.spot_yxz, t, r, c, nbp_find_spots.spot_no)
        t_isolated = fs.spot_isolated(nbp_find_spots.isolated_spots, t, r, c, nbp_find_spots.spot_no)
        # np.shape(t_local_yxz)[0] is the number of spots found on this tile. If there's a nonzero number of spots
        # found then we append the local_yxz info and isolated info to our arrays.
        # The all_local_tiles array SHOULD be the same length (ie have same number of elements as all_local_yxz has
        # rows) as all_local_yxz
        if np.shape(t_local_yxz)[0] > 0:
            all_local_yxz = np.append(all_local_yxz, t_local_yxz, axis=0)
            all_isolated = np.append(all_isolated, t_isolated.astype(bool), axis=0)
            all_local_tile = np.append(all_local_tile, np.ones_like(t_isolated, dtype=np.int16) * t)

    # find duplicate spots as those detected on a tile which is not tile centre they are closest to
    not_duplicate = call_spots_base.get_non_duplicate(
        tile_origin, list(nbp_basic.use_tiles), nbp_basic.tile_centre, all_local_yxz, all_local_tile
    )

    # nd means all spots that are not duplicate
    nd_local_yxz = all_local_yxz[not_duplicate]
    nd_local_tile = all_local_tile[not_duplicate]
    nd_isolated = all_isolated[not_duplicate]
    invalid_value = -nbp_basic.tile_pixel_value_shift

    # Only save used rounds/channels initially
    n_use_rounds, n_use_channels, n_use_tiles = len(use_rounds), len(use_channels), len(use_tiles)
    spot_colours = np.zeros((0, n_use_rounds, n_use_channels), dtype=np.int32)
    local_yxz = np.zeros((0, 3), dtype=np.int16)
    bg_colours = np.zeros_like(spot_colours)
    isolated = np.zeros(0, dtype=bool)
    tile = np.zeros(0, dtype=np.int16)
    transform = np.asarray(icp_correction)
    log.info("Reading in spot_colors for ref_round spots")
    for t in nbp_basic.use_tiles:
        in_tile = nd_local_tile == t
        if np.sum(in_tile) == 0:
            continue
        log.info(f"Tile {np.where(use_tiles==t)[0][0]+1}/{n_use_tiles}")
        colour_tuple = spot_colors_base.get_spot_colors(
            yxz_base=nd_local_yxz[in_tile],
            t=t,
            transform=transform,
            bg_scale=nbp_register.bg_scale,
            file_type=nbp_extract.file_type,
            nbp_file=nbp_file,
            nbp_basic=nbp_basic,
        )
        valid = colour_tuple[-1]
        spot_colours = np.append(spot_colours, colour_tuple[0][valid], axis=0)
        local_yxz = np.append(local_yxz, colour_tuple[1][valid], axis=0)
        isolated = np.append(isolated, nd_isolated[in_tile][valid], axis=0)
        tile = np.append(tile, np.ones(np.sum(valid), dtype=np.int16) * t)
        if nbp_basic.use_preseq:
            bg_colours = np.append(bg_colours, colour_tuple[2][valid], axis=0)

    # add in un-used rounds and channels with invalid_value
    n_spots, n_rounds, n_channels = len(local_yxz), nbp_basic.n_rounds, nbp_basic.n_channels
    spot_colours_full = np.full((n_spots, n_rounds, n_channels), invalid_value, dtype=np.int32)
    spot_colours_full[np.ix_(np.arange(n_spots), use_rounds, use_channels)] = spot_colours

    # save spot info to notebook
    nbp.local_yxz = local_yxz
    nbp.isolated = isolated
    nbp.tile = tile
    nbp.colours = spot_colours_full
    nbp.bg_colours = bg_colours

    log.debug("Get ref spots complete")

    return nbp
