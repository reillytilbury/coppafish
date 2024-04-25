import tqdm
import scipy
import math as maths
from typing_extensions import assert_type
import numpy as np
from typing import Tuple

from ..omp import coefs_new, spots_new, scores
from ..setup.notebook import NotebookPage
from .. import utils, spot_colors, call_spots, find_spots, log


def run_omp(
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

    log.info("OMP started")
    nbp = NotebookPage("omp")
    nbp.software_version = utils.system.get_software_version()
    nbp.revision_hash = utils.system.get_software_hash()

    n_genes = nbp_call_spots.bled_codes_ge.shape[0]
    n_rounds_use = len(nbp_basic.use_rounds)
    n_channels_use = len(nbp_basic.use_channels)
    spot_shape_size_z = config["spot_shape"][2]
    tile_shape: Tuple[int] = nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)
    bled_codes_ge = nbp_call_spots.bled_codes_ge[np.ix_(range(n_genes), nbp_basic.use_rounds, nbp_basic.use_channels)]
    assert (~np.isnan(bled_codes_ge)).all(), "bled codes GE cannot contain nan values"
    assert np.allclose(np.linalg.norm(bled_codes_ge, axis=(1, 2)), 1), "bled codes GE must be L2 normalised"
    bled_codes_ge = bled_codes_ge.astype(np.float16)
    colour_norm_factor = np.array(nbp_call_spots.color_norm_factor, dtype=np.float16)
    colour_norm_factor = colour_norm_factor[
        np.ix_(range(colour_norm_factor.shape[0]), nbp_basic.use_rounds, nbp_basic.use_channels)
    ]
    first_computation = True

    # Results are appended to these arrays
    spots_local_yxz = np.zeros((0, 3), dtype=np.int16)
    spots_tile = np.zeros(0, dtype=np.int16)
    spots_gene_no = np.zeros(0, dtype=np.int16)
    spots_score = np.zeros(0, dtype=np.int16)

    for t in nbp_basic.use_tiles:
        # STEP 1: Load every registered sequencing round/channel image into memory
        log.info(f"Tile {t}")
        yxz_all_pixels = np.array(np.ones(tile_shape, dtype=bool).nonzero(), dtype=np.int16).T
        # Load the colour image in batches so that we do not run out of RAM since the output is int32 and we convert it
        # down to float16
        maximum_batch_size = 100_000_000
        n_batches = maths.ceil(yxz_all_pixels.shape[0] / maximum_batch_size)
        colour_image = np.zeros((yxz_all_pixels.shape[0], n_rounds_use, n_channels_use), dtype=np.float16)
        for i in range(n_batches):
            index_min, index_max = i * maximum_batch_size, min([yxz_all_pixels.shape[0], (i + 1) * maximum_batch_size])
            colour_image[index_min:index_max], _, _, _ = spot_colors.get_spot_colors(
                yxz_all_pixels[index_min:index_max],
                t,
                transform,
                nbp_filter.bg_scale,
                nbp_extract.file_type,
                nbp_file,
                nbp_basic,
                output_dtype=np.float16,
            )
        # Set any "invalid" (i.e. out of bounds) colours to zero.
        colour_image[colour_image <= -nbp_basic.tile_pixel_value_shift] = 0.0
        # Divide every colour by the colour normalisation factors to equalise intensities.
        colour_image /= colour_norm_factor[[t]].astype(np.float16)
        assert colour_image.shape == (yxz_all_pixels.shape[0], n_rounds_use, n_channels_use)
        colour_image = colour_image.reshape(tile_shape + (n_rounds_use, n_channels_use))

        z_min: int = -spot_shape_size_z  # Inclusive
        z_max: int = z_min + 3 * spot_shape_size_z  # Exclusive
        subset_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, z_max - z_min)
        # Minimum and maximum z planes to detect OMP spots on relative to the subset image
        detect_z_min = spot_shape_size_z  # Inclusive
        detect_z_max = 2 * spot_shape_size_z  # Exclusive

        def get_z_detect_bounds(z_min: int, z_max: int) -> Tuple[int, int]:
            """Minimum and maximum z planes to detect OMP spots on relative to the entire tile image"""
            detect_z_min = (z_max - z_min) // 3 + z_min
            return detect_z_min, detect_z_min + spot_shape_size_z

        while True:
            # STEP 2: Compute OMP coefficients for spot_shape_size_z * 3 z planes (zeros when out of bounds)
            # z planes that can have OMP coefficients computed for. z planes are relative to the tile image.
            compute_on_z_planes = [z for z in range(z_min, z_max) if z >= 0 and z <= np.max(nbp_basic.use_z)]
            compute_on_z_planes_subset = [
                i for (i, z) in enumerate(range(z_min, z_max)) if z >= 0 and z <= np.max(nbp_basic.use_z)
            ]
            log.info(f"{compute_on_z_planes=}")
            compute_image_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, len(compute_on_z_planes))
            compute_colours_image = colour_image[:, :, compute_on_z_planes].reshape((-1, n_rounds_use, n_channels_use))
            # Fit and subtract the "background genes" off every spot colour.
            log.debug("Fitting background")
            compute_colours_image, bg_coefficients, bg_codes = call_spots.fit_background(compute_colours_image)
            compute_colours_image = compute_colours_image.reshape(compute_image_shape + (n_rounds_use, n_channels_use))
            bg_coefficients = bg_coefficients.reshape(compute_image_shape + (n_channels_use,))
            log.debug("Fitting background complete")
            coefficient_image = scipy.sparse.lil_matrix((np.prod(subset_shape), n_genes), dtype=np.float32)
            # Populate coefficient_image with the coefficients that can be computed in the subset (all others remain zeros).
            in_compute_on_z_planes = np.zeros(subset_shape, dtype=bool)
            in_compute_on_z_planes[:, :, compute_on_z_planes_subset] = True
            in_compute_on_z_planes = in_compute_on_z_planes.reshape(-1)
            coefficient_image[in_compute_on_z_planes] = coefs_new.compute_omp_coefficients(
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
            log.info("Computing OMP coefficients complete")
            del compute_colours_image, bg_coefficients, bg_codes

            # STEP 2.5: On the first OMP z-chunk/tile, compute the OMP spot shape using the found coefficients.
            if first_computation:
                log.info("Computing spot shape")
                n_isolated_spots = 0
                mean_spot = np.zeros(tuple(config["spot_shape"]), dtype=np.float16)
                for g in tqdm.trange(n_genes, desc="Computing spot shape", unit="gene"):
                    g_coefficient_image = coefficient_image[:, g].toarray().reshape(subset_shape)
                    shape_isolation_distance_z = config["shape_isolation_distance_z"]
                    if shape_isolation_distance_z is None:
                        shape_isolation_distance_z = maths.ceil(
                            config["shape_isolation_distance_yx"] * nbp_basic.pixel_size_xy / nbp_basic.pixel_size_z
                        )
                    isolated_spots_yxz, _ = find_spots.detect_spots(
                        image=g_coefficient_image,
                        intensity_thresh=config["shape_coefficient_threshold"],
                        radius_xy=config["shape_isolation_distance_yx"],
                        radius_z=shape_isolation_distance_z,
                    )
                    # Only keep spot detections in the central z region specified
                    isolated_spots_yxz = isolated_spots_yxz[
                        np.logical_and(
                            isolated_spots_yxz[:, 2] >= detect_z_min, isolated_spots_yxz[:, 2] < detect_z_max
                        )
                    ]
                    n_g_isolated_spots = isolated_spots_yxz.shape[0]
                    n_isolated_spots += n_g_isolated_spots
                    if n_g_isolated_spots == 0:
                        log.debug(f"No isolated spots found for gene {g}")
                        continue
                    g_mean_spot = spots_new.compute_mean_spot_from(
                        image=g_coefficient_image,
                        spot_positions_yxz=isolated_spots_yxz,
                        spot_shape=tuple(config["spot_shape"]),
                    )
                    del g_coefficient_image
                    mean_spot += g_mean_spot * n_g_isolated_spots
                if n_isolated_spots == 0:
                    raise ValueError(
                        f"OMP Failed to find any isolated spots. Consider reducing shape_isolation_distance_yx or ",
                        "shape_coefficient_threshold in the omp config and re-running",
                    )
                mean_spot /= n_isolated_spots
                spot = np.zeros_like(mean_spot, dtype=np.int16)
                spot[mean_spot >= config["shape_coefficient_threshold"]] = 1

                nbp.spot_tile = t
                nbp.mean_spot = mean_spot
                nbp.spot = spot
                log.debug(f"Spot and mean spot computed using {n_isolated_spots}")
                log.info("Computing spot shape complete")

            for g in tqdm.trange(n_genes, desc="Detecting and scoring spots", unit="gene"):
                # STEP 3: Detect spots on the central spot_shape_size_z z planes
                g_coefficient_image = coefficient_image[:, g].toarray().reshape(subset_shape + (1,))
                g_spots_yxz, _ = find_spots.detect_spots(
                    image=g_coefficient_image[..., 0],
                    intensity_thresh=config["coefficient_threshold"],
                    radius_xy=config["radius_xy"],
                    radius_z=config["radius_z"],
                )
                # Only keep spot detections in the central z region specified
                g_spots_yxz = g_spots_yxz[
                    np.logical_and(g_spots_yxz[:, 2] >= detect_z_min, g_spots_yxz[:, 2] < detect_z_max)
                ]
                # Convert z positions in the subset image to positions in the entire tile
                g_spots_local_yxz = g_spots_yxz.copy()
                g_spots_local_yxz[:, 2] += z_min
                valid_positiions = np.logical_and(
                    g_spots_local_yxz[:, 2] >= 0, g_spots_local_yxz[:, 2] <= np.max(nbp_basic.use_z)
                )
                g_spots_yxz = g_spots_yxz[valid_positiions]
                g_spots_local_yxz = g_spots_local_yxz[valid_positiions]

                # STEP 4: Score the detections using the coefficients.
                g_spots_score = scores.score_coefficient_image(
                    coefs_image=g_coefficient_image,
                    spot=spot,
                    mean_spot=mean_spot,
                    high_coef_bias=config["high_coef_bias"],
                )[..., 0]
                g_spots_score = g_spots_score[tuple(g_spots_yxz.T)]

                # Remove bad scoring spots (i.e. false gene reads)
                keep_scores = g_spots_score >= config["score_threshold"]
                g_spots_local_yxz = g_spots_local_yxz[keep_scores]
                g_spots_yxz = g_spots_yxz[keep_scores]
                g_spots_score = g_spots_score[keep_scores]

                n_g_spots = g_spots_local_yxz.shape[0]
                if n_g_spots == 0:
                    continue
                g_spots_score = scores.omp_scores_float_to_int(g_spots_score)
                g_spots_tile = np.ones(n_g_spots, dtype=np.int16) * t
                g_spots_gene_no = np.ones(n_g_spots, dtype=np.int16) * g

                spots_local_yxz = np.append(spots_local_yxz, g_spots_local_yxz, axis=0)
                spots_score = np.append(spots_score, g_spots_score, axis=0)
                spots_tile = np.append(spots_tile, g_spots_tile, axis=0)
                spots_gene_no = np.append(spots_gene_no, g_spots_gene_no, axis=0)

                del g_spots_yxz, g_spots_local_yxz, g_spots_score, g_spots_tile, g_spots_gene_no
                del g_coefficient_image

            # STEP 5: Repeat steps 2 to 4 after shifting z planes up by spot_shape_size_z, stopping once beyond the z stack
            first_computation = False
            z_min += spot_shape_size_z
            z_max += spot_shape_size_z
            if get_z_detect_bounds(z_min, z_max)[0] > np.max(nbp_basic.use_z):
                break

    nbp.local_yxz = spots_local_yxz
    nbp.scores = spots_score
    nbp.tile = spots_tile
    nbp.gene_no = spots_gene_no
    log.info("OMP complete")
    return nbp
