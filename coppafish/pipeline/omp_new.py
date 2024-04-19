import os
import numpy as np
import numpy_indexed
from scipy import sparse
from typing import Optional

from ..filter import base as filter_base
from ..setup.notebook import NotebookPage
from .. import utils, spot_colors, call_spots, omp, log


def call_spots_omp(
    config: dict,
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_call_spots: NotebookPage,
    tile_origin: np.ndarray,
    transform: np.ndarray,
    shape_tile: Optional[int],
) -> NotebookPage:
    """
    Run orthogonal matching pursuit (omp) on every pixel to determine a coefficient for each gene at each pixel.

    From the OMP coefficients, score every pixel using an expected spot shape. Detect spots using the image of spot
    scores and save all OMP spots with a large enough score.

    See `'omp'` section of `notebook_comments.json` file for description of the variables in the omp page.

    Args:
        config: Dictionary obtained from `'omp'` section of config file.
        nbp_file: `file_names` notebook page.
        nbp_basic: `basic_info` notebook page.
        nbp_extract: `extract` notebook page.
        nbp_filter: `filter` notebook page.
        nbp_call_spots: `call_spots` notebook page.
        tile_origin: `float [n_tiles x 3]`.
            `tile_origin[t,:]` is the bottom left yxz coordinate of tile `t`.
            yx coordinates in `yx_pixels` and z coordinate in `z_pixels`.
            This is saved in the `stitch` notebook page i.e. `nb.stitch.tile_origin`.
        transform: `float [n_tiles x n_rounds x n_channels x 4 x 3]`.
            `transform[t, r, c]` is the affine transform to get from tile `t`, `ref_round`, `ref_channel` to
            tile `t`, round `r`, channel `c`.
            This is saved in the register notebook page i.e. `nb.register.icp_correction`.
        shape_tile: Tile to use to compute the expected shape of a spot in the gene coefficient images.
            Should be the tile, for which the most spots where found in the `call_reference_spots` step.
            If `None`, will be set to the centre tile.

    Returns:
        `NotebookPage[omp]` - Page contains gene assignments and info for spots using omp.
    """
    nbp = NotebookPage("omp")
    nbp.software_version = utils.system.get_software_version()
    nbp.revision_hash = utils.system.get_software_hash()
    log.debug("OMP started")
