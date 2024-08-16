import os

import numpy as np
from tqdm import tqdm
import zarr

from .. import log, stitch as stitch_base
from ..setup import NotebookPage
from ..utils import tiles_io


def stitch(config: dict, nbp_basic: NotebookPage, nbp_file: NotebookPage, nbp_filter: NotebookPage) -> NotebookPage:
    """
    Run tile stitching. Tiles are shifted to better align using the DAPI images.

    Args:
        config: stitch config.
        nbp_basic: `basic_info` notebook page.
        nbp_file: `file_names` notebook page.
        nbp_filter: `filter` notebook page.

    Returns:
        new `stitch` notebook page.
    """
    log.debug("Stitch started")
    nbp = NotebookPage("stitch", {"stitch": config})

    # initialize the variables
    overlap = config["expected_overlap"]
    use_tiles, anchor_round, dapi_channel = list(nbp_basic.use_tiles), nbp_basic.anchor_round, nbp_basic.dapi_channel
    n_tiles_use, n_tiles = len(use_tiles), nbp_basic.n_tiles
    tilepos_yx = nbp_basic.tilepos_yx[use_tiles]

    # Build the arrays that we will use to compute the pairwise shift
    pairwise_shifts = np.zeros((n_tiles_use, n_tiles_use, 3))
    pairwise_shift_scores = np.zeros((n_tiles_use, n_tiles_use))

    # load the tiles
    tiles = []
    for t in tqdm(use_tiles, total=n_tiles_use, desc="Loading tiles"):
        tile = nbp_filter.images[t, anchor_round, dapi_channel]
        tiles.append(tile)
    tiles = np.array(tiles)

    # fill the pairwise shift and pairwise shift score matrices
    for i, j in tqdm(np.ndindex(n_tiles_use, n_tiles_use), total=n_tiles_use**2, desc="Computing shifts between tiles"):
        # if the tiles are not adjacent, skip
        if abs(tilepos_yx[i] - tilepos_yx[j]).sum() != 1:
            continue
        pairwise_shifts[i, j], pairwise_shift_scores[i, j] = stitch_base.compute_shift(
            t1=tiles[i], t2=tiles[j], t1_pos=tilepos_yx[i], t2_pos=tilepos_yx[j], overlap=overlap
        )

    # compute the nominal_origin_deviations using a minimisation of a quadratic loss function. 
    # Instead of recording the shift between adjacent tiles to yield an n_tiles_use x n_tiles_use x 3 array as in 
    # pairwise_shiftss, this is an n_tiles x 3 array of every tile's shift from its nominal origin
    nominal_origin_deviations = stitch_base.minimise_shift_loss(shift=pairwise_shifts, score=pairwise_shift_scores)

    # expand the pairwise shifts and pairwise shift scores from n_tiles_use x n_tiles_use x 3 to n_tiles x n_tiles x 3
    pairwise_shifts_full, pairwise_shift_scores_full, tile_origins_full = (
        np.zeros((n_tiles, n_tiles, 3)) * np.nan,
        np.zeros((n_tiles, n_tiles)) * np.nan,
        np.zeros((n_tiles, 3)) * np.nan,
    )
    im_size_y, im_size_x = tiles[0].shape[:-1]
    for i, t in enumerate(use_tiles):
        # fill the full shift and score matrices
        pairwise_shifts_full[t, use_tiles] = pairwise_shifts[i]
        pairwise_shift_scores_full[t, use_tiles] = pairwise_shift_scores[i]
        # fill the tile origins
        nominal_origin = np.array(
            [tilepos_yx[i][0] * im_size_y * (1 - overlap), tilepos_yx[i][1] * im_size_x * (1 - overlap), 0]
        )
        tile_origins_full[t] = nominal_origin + nominal_origin_deviations[i]

    # fuse the tiles and save the notebook page variables
    save_path = os.path.join(nbp_file.output_dir, "fused_dapi_image.zarr")
    _ = stitch_base.fuse_tiles(
        tiles=tiles,
        tile_origins=tile_origins_full[use_tiles],
        tilepos_yx=tilepos_yx,
        overlap=overlap,
        save_path=save_path,
    )
    nbp.dapi_image = zarr.open_array(save_path, mode="r")
    nbp.tile_origin = tile_origins_full
    nbp.shifts = pairwise_shifts_full
    nbp.scores = pairwise_shift_scores_full

    log.debug("Stitch finished")

    return nbp
