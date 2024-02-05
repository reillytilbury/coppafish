import numpy as np
from typing import Tuple

from ..setup import NotebookPage
from .. import utils, spot_colors


def get_pixel_colours(
    nbp_basic: NotebookPage,
    nbp_file: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_filter: NotebookPage,
    tile: int,
    z_chunk: int,
    z_chunk_size: int,
    transform: np.ndarray,
    colour_norm_factor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the normalised pixel colours and their pixel positions for one z chunk.

    Args:
        nbp_basic (NotebookPage): 'basic_info' notebook page.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_extract (NotebookPage): 'extract' notebook page.
        tile (int): tile index.
        z_chunk (int): z chunk index.
        z_chunk_size (int): z chunk size
        transform (`[n_tiles x n_rounds x n_channels x 4 x 3] ndarray[float]`): `transform[t, r, c]` is the affine
            transform to get from tile `t`, `ref_round`, `ref_channel` to tile `t`, round `r`, channel `c`.
        colour_norm_factor (`[n_rounds x n_channels] ndarray[float]`): Normalisation factors to divide colours by to
            equalise channel intensities.

    Returns:
        - (`[n_pixels x 3] ndarray[int16]`): `pixel_yxz_tz` is the y, x and z pixel positions of the pixel colours
            found.
        - (`[n_pixels x n_rounds x n_channels] ndarray[float32]`): `pixel_colours_tz` contains the colours for each
            pixel.
    """
    n_rounds, n_channels = colour_norm_factor.shape
    z_min, z_max = z_chunk * z_chunk_size, min((z_chunk + 1) * z_chunk_size, len(nbp_basic.use_z))
    pixel_yxz_tz = np.zeros((0, 3), dtype=np.int16)
    pixel_colours_tz = np.zeros((0, n_rounds, n_channels), dtype=np.float32)
    if nbp_basic.use_preseq:
        pixel_colours_t1, pixel_yxz_t1, _ = spot_colors.base.get_spot_colors(
            spot_colors.base.all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, np.arange(z_min, z_max + 1)),
            int(tile),
            transform,
            nbp_file,
            nbp_basic,
            nbp_extract,
            nbp_filter,
            return_in_bounds=True,
        )
    else:
        pixel_colours_t1, pixel_yxz_t1 = spot_colors.base.get_spot_colors(
            spot_colors.base.all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, np.arange(z_min, z_max + 1)),
            int(tile),
            transform,
            nbp_file,
            nbp_basic,
            nbp_extract,
            nbp_filter,
            return_in_bounds=True,
        )
    pixel_colours_t1 = pixel_colours_t1.astype(np.float32) / colour_norm_factor
    pixel_yxz_tz = np.append(pixel_yxz_tz, pixel_yxz_t1, axis=0)
    pixel_colours_tz = np.append(pixel_colours_tz, pixel_colours_t1, axis=0)

    return pixel_yxz_tz.astype(np.int16), pixel_colours_tz.astype(np.float32)


def get_initial_intensity_thresh(config: dict, nbp: NotebookPage) -> float:
    """
    Gets absolute intensity threshold from config file. OMP will only be run on
    pixels with absolute intensity greater than this.

    Args:
        config: `omp` section of config file.
        nbp: `call_spots` *NotebookPage*

    Returns:
        Either `config['initial_intensity_thresh']` or
            `nbp.abs_intensity_percentile[config['initial_intensity_thresh_percentile']]`.

    """
    initial_intensity_thresh = config["initial_intensity_thresh"]
    if initial_intensity_thresh is None:
        config["initial_intensity_thresh"] = utils.base.round_any(
            nbp.abs_intensity_percentile[config["initial_intensity_thresh_percentile"]],
            config["initial_intensity_precision"],
        )
    initial_intensity_thresh = float(
        np.clip(
            config["initial_intensity_thresh"],
            config["initial_intensity_thresh_min"],
            config["initial_intensity_thresh_max"],
        )
    )
    return initial_intensity_thresh
