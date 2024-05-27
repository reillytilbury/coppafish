import numpy as np
import os
from tqdm import tqdm

from .. import stitch as stitch_base, log
from ..setup import NotebookPage
from ..utils import system, tiles_io


def stitch(nbp_basic: NotebookPage, nbp_file: NotebookPage, config_stitch: dict,
           file_type: str = '.zarr') -> NotebookPage:
    """

    Args:
        nbp_basic:
        nbp_file:
        config_stitch:
        file_type:

    Returns:

    """
    log.debug("Stitch started")
    nbp = NotebookPage("stitch")
    nbp.software_version = system.get_software_version()
    nbp.revision_hash = system.get_software_hash()

    # initialize the variables
    overlap = config_stitch["expected_overlap"]
    use_tiles, anchor_round, dapi_channel = nbp_basic.use_tiles, nbp_basic.anchor_round, nbp_basic.dapi_channel
    n_tiles_use, n_tiles = len(use_tiles), nbp_basic.n_tiles
    tilepos_yx = nbp_basic.tilepos_yx[use_tiles]

    # Build the tensors that we will use to compute the shifts
    shift = np.zeros((n_tiles_use, n_tiles_use, 3))
    score = np.zeros((n_tiles_use, n_tiles_use))

    # import the tile stack and the tile positions
    tiles = np.array([tiles_io.load_image(nbp_file=nbp_file, nbp_basic=nbp_basic, file_type=file_type,
                                          t=t, r=anchor_round, c=dapi_channel) for t in use_tiles])

    # fill the shift and score matrices
    for i, j in tqdm(np.ndindex(n_tiles_use, n_tiles_use), total=n_tiles_use**2, desc='Computing shifts between tiles'):
        # if the tiles are not adjacent, skip
        if abs(tilepos_yx[i] - tilepos_yx[j]).sum() != 1:
            continue
        shift[i, j], score[i, j] = stitch_base.compute_shift(t1=tiles[i], t2=tiles[j],
                                                             t1_pos=tilepos_yx[i], t2_pos=tilepos_yx[j],
                                                             overlap=overlap)

    # compute the final shifts using a minimisation of a quadratic loss function
    shifts_final = stitch_base.minimise_shift_loss(shift=shift, score=score)

    # apply the shifts to the tiles
    shift_full, score_full, tile_origins_full = (np.zeros((n_tiles, n_tiles, 3)) * np.nan,
                                                 np.zeros((n_tiles, n_tiles)) * np.nan,
                                                 np.zeros((n_tiles, 3)) * np.nan)
    im_size_y, im_size_x = tiles[0].shape[1:]
    for i, t in enumerate(use_tiles):
        # fill the full shift and score matrices
        shift_full[t, use_tiles] = shift[i]
        score_full[t, use_tiles] = score[i]
        # fill the tile origins
        nominal_origin = np.array([0, tilepos_yx[i][0] * im_size_y, tilepos_yx[i][1] * im_size_x])
        tile_origins_full[i] = nominal_origin + shifts_final[i]

    # fuse the tiles and save the notebook page variables
    _ = stitch_base.fuse_tiles(tiles=tiles, tile_origins=tile_origins_full[use_tiles], tilepos_yx=tilepos_yx,
                               overlap=overlap, save_path=os.path.join(nbp_file.output_dir, 'fused_dapi.npy'))
    nbp.tile_origins = tile_origins_full
    nbp.shifts = shift_full
    nbp.scores = score_full
    log.debug("Stitch finished")
    return nbp
