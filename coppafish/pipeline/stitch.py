import numpy as np
from tqdm import tqdm

from .. import stitch as stitch_base, log
from ..setup import NotebookPage
from ..utils import system, tiles_io


def stitch(nbp_basic: NotebookPage, nbp_file: NotebookPage, config_stitch: dict) -> NotebookPage:
    """

    Args:
        nbp_basic:
        nbp_file:
        config_stitch:

    Returns:

    """
    log.debug("Stitch started")
    nbp = NotebookPage("stitch")
    nbp.software_version = system.get_software_version()
    nbp.revision_hash = system.get_software_hash()

    # initialize the variables
    overlap = config_stitch["expected_overlap"]
    use_tiles, dapi_channel = nbp_basic.use_tiles, nbp_basic.dapi_channel
    tilepos_yx, n_tiles = nbp_basic.tilepos_yx, nbp_basic.n_tiles
    # Build the tensors that we will use to compute the shifts
    shift = np.zeros((n_tiles, n_tiles, 3))
    score = np.zeros((n_tiles, n_tiles))

    # import the tile stack and the tile positions
    tiles = [tiles_io.load_image(nbp_file, use_tiles, dapi_channel, t) for t in range(n_tiles) if t in use_tiles]

    # fill the shift and score matrices
    for i, j in tqdm(np.ndindex(n_tiles, n_tiles), total=n_tiles**2, desc='Computing shifts between tiles'):
        # if the tiles are not adjacent, skip
        if abs(tilepos_yx[i] - tilepos_yx[j]).sum() != 1:
            continue
        shift[i, j], score[i, j] = stitch_base.compute_shift(t1=tiles[i], t2=tiles[j],
                                                             t1_pos=tilepos_yx[i], t2_pos=tilepos_yx[j],
                                                             overlap=overlap)
    score = score ** 2
    # we need to build the n_tiles x n_tiles matrix A and the n_tiles x 2 matrix b that will be used to solve the linear
    # system of equations: Ax = b, where x is our final shift matrix (n_tiles x 2)
    A = np.zeros((n_tiles, n_tiles))
    b = np.zeros((n_tiles, 3))
    # fill the A matrix
    for i, j in np.ndindex(n_tiles, n_tiles):
        if i == j:
            A[i, j] = np.sum(score[i, :])
        else:
            A[i, j] = -score[i, j]
    # fill the b matrix
    for i in range(n_tiles):
        b[i] = np.sum(score[i, :, np.newaxis] * shift[i], axis=0)

    # solve the linear system of equations
    shifts_final = np.linalg.lstsq(A, b, rcond=None)[0]

    # apply the shifts to the tiles
    tile_origins = np.zeros((n_tiles, 3))
    im_size_y, im_size_x = tiles[0].shape[1:]
    for t in range(n_tiles):
        y, x = tilepos_yx[t]
        nominal_origin = np.array([0, y * im_size_y * (1 - overlap), x * im_size_x * (1 - overlap)])
        tile_origins[t] = nominal_origin + shifts_final[t]

    # fuse the tiles
    fused_image = stitch_base.fuse_tiles(tiles=tiles, tile_origins=tile_origins, tilepos_yx=tilepos_yx)
    np.save(r"C:\Users\reill\Desktop\local_datasets\daphne\fused_image.npy", fused_image)
    log.debug("Stitch finished")
    return nbp
