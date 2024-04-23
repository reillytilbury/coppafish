import os
import scipy
from torch import maximum
from typing_extensions import assert_type
import numpy as np
import numpy_indexed
from typing import Optional, Tuple

from ..omp import coefs_new
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

    Returns:
        `NotebookPage[omp]` - Page contains gene assignments and info for spots using omp.
    """
    assert_type(config, dict)
    assert_type(nbp_file, NotebookPage)
    assert_type(nbp_basic, NotebookPage)
    assert_type(nbp_extract, NotebookPage)
    assert_type(nbp_filter, NotebookPage)
    assert_type(nbp_call_spots, NotebookPage)
    assert tile_origin.ndim == 2
    assert transform.shape[3:5] == (4, 3)
    assert transform.ndim == 5

    log.debug("OMP started")
    nbp = NotebookPage("omp")
    nbp.software_version = utils.system.get_software_version()
    nbp.revision_hash = utils.system.get_software_hash()

    n_genes = nbp_call_spots.bled_codes_ge.shape[0]
    n_rounds_use = len(nbp_basic.use_rounds)
    n_channels_use = len(nbp_basic.use_channels)
    spot_shape_size_z = config["shape_max_size"][2]
    tile_shape: Tuple[int] = nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)
    bled_codes_ge = nbp_call_spots.bled_codes_ge[range(n_genes), np.ix_(nbp_basic.use_rounds, nbp_basic.use_channels)]
    bled_codes_ge = bled_codes_ge.astype(np.float32)
    assert (~np.isnan(bled_codes_ge)).all(), "bled codes GE cannot contain nan values"
    assert np.allclose(np.linalg.norm(bled_codes_ge, axis=(1, 2)), 1), "bled codes GE must be L2 normalised"
    colour_norm_factor = np.array(nbp_call_spots.colo_norm_factor, dtype=np.float32)
    colour_norm_factor = colour_norm_factor[
        np.ix_(range(colour_norm_factor.shape[0]), nbp_basic.use_rounds, nbp_basic.use_channels)
    ]

    for t in nbp_basic.use_tiles:
        # Step 1: Load every registered sequencing round/channel image into memory
        log.info(f"Tile {t}")
        yxz_all_pixels = np.array(np.ones(tile_shape, dtype=bool).nonzero(), dtype=np.int16).T
        # Load the colour image in batches so that we do not run out of RAM since the output is int32 and we want to
        # convert it down to float16
        maximum_batch_size = 100_000_000
        n_batches = np.ceil(yxz_all_pixels.shape[0] / maximum_batch_size)
        colour_image = np.zeros((yxz_all_pixels.shape[0], n_rounds_use, n_channels_use), dtype=np.float16)
        for i in range(n_batches):
            index_min, index_max = i * maximum_batch_size, min([yxz_all_pixels.shape[0], (i + 1) * maximum_batch_size])
            colour_image[index_min, index_max] = spot_colors.get_spot_colors(
                yxz_all_pixels[index_min, index_max],
                t,
                transform,
                nbp_filter.bg_scale,
                nbp_extract.file_type,
                nbp_file,
                nbp_basic,
            )[0].astype(np.float16)
        # Divide every colour by the colour normalisation factors to equalise intensities.
        colour_image /= colour_norm_factor[[t]].astype(np.float16)
        assert colour_image.shape == (yxz_all_pixels.shape[0], n_rounds_use, n_channels_use)
        colour_image = colour_image.reshape(tile_shape + (n_rounds_use, n_channels_use))

        while True:
            # Step 2: Compute OMP coefficients for spot_shape_size_z * 3 z planes (zeros when out of bounds)
            z_min: int = -spot_shape_size_z  # Inclusive
            z_max: int = z_min + 3 * spot_shape_size_z  # Exclusive
            compute_on_z_planes = [z for z in range(z_min, z_max) if z >= 0 and z <= np.max(nbp_basic.use_z)]
            compute_colours_image = (
                colour_image[:, :, compute_on_z_planes].astype(np.float32).reshape((-1, n_rounds_use, n_channels_use))
            )
            # Fit and subtract the "background genes" off every spot colour.
            log.debug("Fitting background")
            compute_colours_image, bg_coefficients, bg_codes = call_spots.fit_background(compute_colours_image)
            log.debug("Fitting background complete")
            compute_colours_image.reshape(tile_shape + (n_rounds_use, n_channels_use))
            bg_coefficients.reshape(tile_shape + (n_channels_use,))
            coefficient_image = scipy.sparse.csr_matrix(
                np.zeros((nbp_basic.tile_sz, nbp_basic.tile_sz, z_max - z_min), dtype=np.float32)
            )
            log.debug(f"Comuting OMP coefficients for z={compute_on_z_planes}")
            coefficient_image[:, :, compute_on_z_planes] = coefs_new.compute_omp_coefficients(
                compute_colours_image,
                bled_codes_ge,
                maximum_iterations=config["max_genes"],
                background_coefficients=bg_coefficients,
                background_codes=bg_codes,
                dot_product_threshold=config["dp_thresh"],
                dot_product_norm_shift=0.0,
                weight_coefficient_fit=config["weight_coef_fit"],
                alpha=config["alpha"],
                beta=config["beta"],
            )
            log.debug("Computing OMP coefficients complete")
            del bled_codes_ge, compute_colours_image, bg_coefficients, bg_codes

            detect_z_min: int = (z_max - z_min) // 3 + z_min
            detect_z_max: int = detect_z_min + spot_shape_size_z
            # If this is the first OMP z chunk and first tile, compute the OMP spot shape using the results
            for g in range(n_genes):
                # Step 3: Detect spots on the middle spot_shape_size_z z planes
                pass

                # Step 4: Score the detections using the coefficients.

            # Step 5: Repeat steps 2 to 4 after shifting z planes up by spot_shape_size_z, stopping once beyond the z stack
