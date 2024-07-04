import os

import numpy as np
from tqdm import tqdm
import zarr

from .. import log, stitch as stitch_base
from ..setup import NotebookPage
from ..utils import tiles_io


def stitch(config: dict, nbp_basic: NotebookPage, nbp_file: NotebookPage) -> NotebookPage:
    """
    Run tile stitching. Tiles are shifted to better align using the DAPI images.

    Args:
        config: stitch config.
        nbp_basic: `basic_info` notebook page.
        nbp_file: `file_names` notebook page.
        nbp_extract: `extract` notebook page.

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

    # Build the tensors that we will use to compute the shifts
    shift = np.zeros((n_tiles_use, n_tiles_use, 3))
    score = np.zeros((n_tiles_use, n_tiles_use))

    # load the tiles
    tiles = []
    for t in tqdm(use_tiles, total=n_tiles_use, desc="Loading tiles"):
        tile = tiles_io.load_image(nbp_file=nbp_file, nbp_basic=nbp_basic, t=t, r=anchor_round, c=dapi_channel)[:]
        tiles.append(tile)
    tiles = np.array(tiles)

    # fill the shift and score matrices
    for i, j in tqdm(np.ndindex(n_tiles_use, n_tiles_use), total=n_tiles_use**2, desc="Computing shifts between tiles"):
        # if the tiles are not adjacent, skip
        if abs(tilepos_yx[i] - tilepos_yx[j]).sum() != 1:
            continue
        shift[i, j], score[i, j] = stitch_base.compute_shift(
            t1=tiles[i], t2=tiles[j], t1_pos=tilepos_yx[i], t2_pos=tilepos_yx[j], overlap=overlap
        )

    # compute the final shifts using a minimisation of a quadratic loss function
    shifts_final = stitch_base.minimise_shift_loss(shift=shift, score=score)

    # apply the shifts to the tiles
    shift_full, score_full, tile_origins_full = (
        np.zeros((n_tiles, n_tiles, 3)) * np.nan,
        np.zeros((n_tiles, n_tiles)) * np.nan,
        np.zeros((n_tiles, 3)) * np.nan,
    )
    im_size_y, im_size_x = tiles[0].shape[:-1]
    for i, t in enumerate(use_tiles):
        # fill the full shift and score matrices
        shift_full[t, use_tiles] = shift[i]
        score_full[t, use_tiles] = score[i]
        # fill the tile origins
        nominal_origin = np.array(
            [tilepos_yx[i][0] * im_size_y * (1 - overlap), tilepos_yx[i][1] * im_size_x * (1 - overlap), 0]
        )
        tile_origins_full[t] = nominal_origin + shifts_final[i]

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
    nbp.shifts = shift_full
    nbp.scores = score_full

    log.debug("Stitch finished")

    return nbp
