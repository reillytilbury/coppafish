import numpy as np

from .. import find_spots as fs
from .. import log
from ..call_spots import base as call_spots_base
from ..setup import NotebookPage
from ..spot_colours import base as spot_colours_base


def get_reference_spots(
    nbp_basic: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_find_spots: NotebookPage,
    nbp_register: NotebookPage,
    nbp_stitch: NotebookPage,
) -> NotebookPage:
    """
    This takes each spot found on the reference round/channel and computes the corresponding intensity
    in each of the imaging rounds/channels.

    Args:
        nbp_file: `file_names` notebook page.
        nbp_basic: `basic_info` notebook page.
        nbp_find_spots: 'find_spots' notebook page.
        nbp_extract: `extract` notebook page.
        nbp_register: `register` notebook page.
        nbp_stitch: `stitch` notebook page.

    Returns:
        `NotebookPage[ref_spots]` - Page containing intensity of each reference spot on each imaging round/channel.
    """
    # We create a notebook page for ref_spots which stores information like local coords, isolated info, tile_no of each
    # spot and much more.
    nbp = NotebookPage("ref_spots")
    # The code is going to loop through all tiles, as we expect some anchor spots on each tile but r and c should stay
    # fixed as the value of the reference round and reference channel
    tile_origin = nbp_stitch.tile_origin
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
        if np.shape(t_local_yxz)[0] == 0:
            continue
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

    # Only save used rounds/channels initially
    n_use_rounds, n_use_channels, n_use_tiles = len(use_rounds), len(use_channels), len(use_tiles)
    spot_colours = np.zeros((0, n_use_rounds, n_use_channels), dtype=np.float32)
    local_yxz = np.zeros((0, 3), dtype=np.int16)
    isolated = np.zeros(0, dtype=bool)
    tile = np.zeros(0, dtype=np.int16)
    log.info("Reading in spot_colours for ref_round spots")
    for t in nbp_basic.use_tiles:
        in_tile = nd_local_tile == t
        if np.sum(in_tile) == 0:
            continue
        log.info(f"Tile {np.where(use_tiles==t)[0][0]+1}/{n_use_tiles}")
        colours = spot_colours_base.get_spot_colours(
            image=nbp_filter.images,
            flow=nbp_register.flow,
            affine_correction=nbp_register.icp_correction,
            tile=t,
            yxz_base=nd_local_yxz[in_tile],
            use_channels=use_channels,
        )
        valid = ~(np.isnan(colours).any(1).any(1))
        log.debug(f"Valid ref pixel colours: {valid.sum()} out of {valid.size} for tile {t}")
        spot_colours = np.append(spot_colours, colours[valid], axis=0)
        local_yxz = np.append(local_yxz, nd_local_yxz[in_tile][valid], axis=0)
        isolated = np.append(isolated, nd_isolated[in_tile][valid], axis=0)
        tile = np.append(tile, np.ones(valid.sum(), dtype=np.int16) * t)

    # save spot info to notebook
    nbp.local_yxz = local_yxz
    nbp.isolated = isolated
    nbp.tile = tile
    nbp.colours = spot_colours
    log.debug("Get ref spots complete")

    return nbp
