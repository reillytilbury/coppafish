import tqdm
import math as maths
import numpy as np
import torch
from typing_extensions import assert_type
from typing import Tuple

from ..omp import base, coefs_torch, spots_torch, scores_torch
from ..call_spots import background_pytorch
from ..find_spots import detect_torch
from .. import utils, log
from ..setup.notebook import NotebookPage


def run_omp(
    config: dict,
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_register: NotebookPage,
    nbp_register_debug: NotebookPage,
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
        nbp_register: `register` notebook page.
        nbp_register_debug: `register_debug` notebook page.
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
    assert_type(nbp_register, NotebookPage)
    assert_type(nbp_register_debug, NotebookPage)
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
    spot_shape_size_xy: int = config["spot_shape"][0]
    spot_shape_size_z: int = config["spot_shape"][2]
    spot_radius_xy: int = maths.ceil(spot_shape_size_xy / 2)
    spot_radius_z: int = maths.ceil(spot_shape_size_z / 2)
    tile_shape: Tuple[int] = nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)
    colour_norm_factor = np.array(nbp_call_spots.color_norm_factor, dtype=np.float32)
    colour_norm_factor = colour_norm_factor[
        np.ix_(range(colour_norm_factor.shape[0]), nbp_basic.use_rounds, nbp_basic.use_channels)
    ]
    colour_norm_factor = torch.asarray(colour_norm_factor).float()
    first_computation = True

    subset_z_size: int = len(nbp_basic.use_z) + 2 * spot_radius_z
    subset_size_xy = config["subset_size_xy"]

    assert subset_z_size > spot_radius_z * 2
    if subset_size_xy <= spot_radius_xy * 2:
        raise ValueError(
            "The subset size is too small for the given spot size. Reduce spot_shape in x and y directions or increase"
            + " the subset_size_xy to facilitate a large spot in the config"
        )

    subset_shape = (subset_size_xy, subset_size_xy, subset_z_size)
    # Find the bottom-left position of every subset to break the entire tile up into.
    subset_origin_new = [-spot_radius_xy, -spot_radius_xy, -spot_radius_z]
    subset_origins_yxz = []
    while True:
        if subset_origin_new[0] >= nbp_basic.tile_sz:
            break
        subset_origins_yxz.append(subset_origin_new.copy())
        subset_origin_new[1] += subset_size_xy - 2 * spot_radius_xy
        if subset_origin_new[1] >= nbp_basic.tile_sz:
            subset_origin_new[1] = 0
            subset_origin_new[0] += subset_size_xy - 2 * spot_radius_xy

    log.info(f"Running {len(subset_origins_yxz)} subsets for each tile")

    # Results are appended to these arrays
    spots_local_yxz = torch.zeros((0, 3), dtype=torch.int16)
    spots_tile = torch.zeros(0, dtype=torch.int16)
    spots_gene_no = torch.zeros(0, dtype=torch.int16)
    spots_score = torch.zeros(0, dtype=torch.int16)

    for t in nbp_basic.use_tiles:
        # STEP 1: Load every registered sequencing round/channel image into memory
        colour_image = base.load_spot_colours(nbp_basic, nbp_file, nbp_extract, nbp_register, nbp_register_debug, t)
        assert colour_image.shape == tile_shape + (n_rounds_use, n_channels_use)

        for i, subset_yxz in enumerate(subset_origins_yxz):
            # STEP 2: Compute OMP coefficients on a subset of the tile which is a mini tile with the same number of z
            # planes.
            log.debug(f"Subset {i}, Subset origin {subset_yxz}")

            def subset_positions_to_tile_positions(positions_yxz: torch.Tensor) -> torch.Tensor:
                return positions_yxz.detach().detach().clone() + torch.asarray(subset_yxz)[np.newaxis]

            def get_valid_subset_positions(positions_yxz: torch.Tensor) -> torch.Tensor:
                valid = (
                    (positions_yxz[:, 0] >= spot_radius_xy)
                    * (positions_yxz[:, 1] >= spot_radius_xy)
                    * (positions_yxz[:, 2] >= spot_radius_z)
                    * (positions_yxz[:, 0] < (subset_shape[0] - spot_radius_xy))
                    * (positions_yxz[:, 1] < (subset_shape[1] - spot_radius_xy))
                    * (positions_yxz[:, 2] < (subset_shape[2] - spot_radius_z))
                )
                tile_positions_yxz = subset_positions_to_tile_positions(positions_yxz)
                valid *= (
                    (tile_positions_yxz[:, 0] >= 0)
                    * (tile_positions_yxz[:, 1] >= 0)
                    * (tile_positions_yxz[:, 2] >= 0)
                    * (tile_positions_yxz[:, 0] < tile_shape[0])
                    * (tile_positions_yxz[:, 1] < tile_shape[1])
                    * (tile_positions_yxz[:, 2] < tile_shape[2])
                )
                return valid

            subset_colours = torch.zeros(subset_shape + (n_rounds_use, n_channels_use), dtype=torch.float32)
            # Gather all subset colours that exist, the rest remain zeros.
            index_min_y = max([subset_yxz[0], 0])
            index_max_y = min([subset_yxz[0] + subset_shape[0], tile_shape[0]])
            index_min_x = max([subset_yxz[1], 0])
            index_max_x = min([subset_yxz[1] + subset_shape[1], tile_shape[1]])
            index_min_z = max([subset_yxz[2], 0])
            index_max_z = min([subset_yxz[2] + subset_shape[2], tile_shape[2]])
            subset_colours[
                index_min_y - subset_yxz[0] : index_max_y - subset_yxz[0],
                index_min_x - subset_yxz[1] : index_max_x - subset_yxz[1],
                index_min_z - subset_yxz[2] : index_max_z - subset_yxz[2],
            ] = torch.asarray(
                colour_image[
                    index_min_y:index_max_y,
                    index_min_x:index_max_x,
                    index_min_z:index_max_z,
                ].astype(np.float32)
            )
            subset_colours = torch.reshape(subset_colours, (-1, n_rounds_use, n_channels_use))
            # Place the zero in the correct position.
            subset_colours -= nbp_basic.tile_pixel_value_shift
            # Set any out of bounds colours to zero.
            subset_colours[subset_colours <= -nbp_basic.tile_pixel_value_shift] = 0.0
            # Divide all colours by the colour normalisation factors.
            subset_colours /= colour_norm_factor[[t]]
            # Fit and subtract the "background genes" off every spot colour.
            log.debug("Fitting background")
            subset_colours, bg_coefficients, bg_codes = background_pytorch.fit_background(subset_colours)
            log.debug("Fitting background complete")
            bled_codes_ge = nbp_call_spots.bled_codes_ge
            bled_codes_ge = bled_codes_ge[np.ix_(range(n_genes), nbp_basic.use_rounds, nbp_basic.use_channels)]
            assert (~np.isnan(bled_codes_ge)).all(), "bled codes GE cannot contain nan values"
            assert np.allclose(np.linalg.norm(bled_codes_ge, axis=(1, 2)), 1), "bled codes GE must be L2 normalised"
            bled_codes_ge = torch.asarray(bled_codes_ge.astype(np.float32))
            # Populate coefficient_image with the coefficients that can be computed in the subset.
            log.debug("Computing OMP coefficients started")
            subset_colours = subset_colours.reshape((-1, n_rounds_use * n_channels_use))
            bled_codes_ge = bled_codes_ge.reshape((n_genes, n_rounds_use * n_channels_use))
            bg_codes = bg_codes.reshape((n_channels_use, n_rounds_use * n_channels_use))
            coefficient_image = coefs_torch.compute_omp_coefficients(
                subset_colours,
                bled_codes_ge,
                maximum_iterations=config["max_genes"],
                background_coefficients=bg_coefficients,
                background_codes=bg_codes,
                dot_product_threshold=config["dp_thresh"],
                dot_product_norm_shift=0.0,
                weight_coefficient_fit=config["weight_coef_fit"],
                alpha=config["alpha"],
                beta=config["beta"],
                force_cpu=config["force_cpu"],
            )
            log.debug("Computing OMP coefficients complete")
            del subset_colours, bg_coefficients, bg_codes, bled_codes_ge

            # STEP 2.5: On the first OMP z-chunk/tile, compute the OMP spot shape using the found coefficients.
            if first_computation:
                log.debug("Computing spot shape")
                isolated_spots_yxz = torch.zeros((0, 3), dtype=torch.int16)
                isolated_gene_numbers = torch.zeros(0, dtype=torch.int16)
                for g in tqdm.trange(n_genes, desc="Computing spot shape", unit="gene"):
                    if isolated_spots_yxz.size(0) >= config["spot_shape_max_spots"]:
                        isolated_spots_yxz = isolated_spots_yxz[: config["spot_shape_max_spots"]]
                        isolated_gene_numbers = isolated_gene_numbers[: config["spot_shape_max_spots"]]
                        continue
                    g_coefficient_image = torch.asarray(coefficient_image[:, g].toarray().reshape(subset_shape)).float()
                    shape_isolation_distance_z = config["shape_isolation_distance_z"]
                    if shape_isolation_distance_z is None:
                        shape_isolation_distance_z = maths.ceil(
                            config["shape_isolation_distance_yx"] * nbp_basic.pixel_size_xy / nbp_basic.pixel_size_z
                        )
                    isolated_spots_yxz_g, _ = detect_torch.detect_spots(
                        image=g_coefficient_image,
                        intensity_thresh=config["shape_coefficient_threshold"],
                        radius_xy=config["shape_isolation_distance_yx"],
                        radius_z=shape_isolation_distance_z,
                        force_cpu=config["force_cpu"],
                    )
                    valid_positions = get_valid_subset_positions(isolated_spots_yxz_g)
                    isolated_spots_yxz_g = isolated_spots_yxz_g[valid_positions]
                    n_g_isolated_spots = isolated_spots_yxz_g.shape[0]
                    isolated_spots_yxz = torch.cat((isolated_spots_yxz, isolated_spots_yxz_g), dim=0)
                    isolated_gene_numbers = torch.cat(
                        (isolated_gene_numbers, torch.ones(n_g_isolated_spots).to(torch.int16) * g), dim=0
                    )
                    del g_coefficient_image, isolated_spots_yxz_g, n_g_isolated_spots
                mean_spots = torch.zeros((n_genes,) + tuple(config["spot_shape"]), dtype=torch.float32)
                weights = torch.zeros(n_genes, dtype=torch.float32)
                for g in range(n_genes):
                    if (isolated_gene_numbers == g).sum() == 0:
                        continue
                    g_coefficient_image = torch.asarray(coefficient_image[:, g].toarray().reshape(subset_shape)).float()
                    g_mean_spot = spots_torch.compute_mean_spot_from(
                        g_coefficient_image,
                        isolated_spots_yxz[isolated_gene_numbers == g],
                        config["spot_shape"],
                        config["force_cpu"],
                    )
                    mean_spots[g] = g_mean_spot
                    weights[g] = (isolated_gene_numbers == g).sum()
                if weights.sum() == 0:
                    raise ValueError(
                        f"OMP Failed to find any isolated spots. Make sure that regstration is working as expected. "
                        + "If so, Consider reducing shape_isolation_distance_yx or shape_coefficient_threshold in the "
                        + "omp config then re-running.",
                    )
                mean_spot = torch.mean(mean_spots * weights[:, np.newaxis, np.newaxis, np.newaxis], dim=0).float()
                mean_spot *= n_genes / weights.sum()
                log.info(f"OMP spot and mean spot computed using {weights.sum().item()} detected spots")
                del mean_spots, weights

                spot = torch.zeros_like(mean_spot, dtype=torch.int16)
                spot[mean_spot >= config["shape_sign_thresh"]] = 1

                nbp.spot_tile = t
                nbp.mean_spot = np.array(mean_spot)
                nbp.spot = np.array(spot)
                log.debug("Computing spot shape complete")

            for g in tqdm.trange(n_genes, desc="Detecting and scoring spots", unit="gene"):
                # STEP 3: Detect spots on the subset except at the x and y edges.
                g_coefficient_image = torch.asarray(coefficient_image[:, g].toarray()).reshape(subset_shape + (1,))
                log.debug(f"Detecting spots for gene {g}")
                g_spots_yxz, _ = detect_torch.detect_spots(
                    image=g_coefficient_image[..., 0],
                    intensity_thresh=config["coefficient_threshold"],
                    radius_xy=config["radius_xy"],
                    radius_z=config["radius_z"],
                )
                log.debug(f"Detecting spots for gene {g} complete")
                # Convert spot positions in the subset image to positions on the tile.
                g_spots_local_yxz = subset_positions_to_tile_positions(g_spots_yxz)
                valid_positions = get_valid_subset_positions(g_spots_yxz)
                g_spots_yxz = g_spots_yxz[valid_positions]
                g_spots_local_yxz = g_spots_local_yxz[valid_positions]

                # STEP 4: Score the detections using the coefficients.
                log.debug(f"Scoring gene {g} image")
                g_spots_score = scores_torch.score_coefficient_image(
                    coefficient_image=g_coefficient_image,
                    spot=spot,
                    mean_spot=mean_spot,
                    high_coefficient_bias=config["high_coef_bias"],
                    force_cpu=config["force_cpu"],
                )[..., 0]
                log.debug(f"Scoring gene {g} image complete")
                g_spots_score = g_spots_score[tuple(g_spots_yxz.T)]

                # Remove bad scoring spots (i.e. false gene reads)
                keep_scores = g_spots_score >= config["score_threshold"]
                g_spots_local_yxz = g_spots_local_yxz[keep_scores]
                g_spots_yxz = g_spots_yxz[keep_scores]
                g_spots_score = g_spots_score[keep_scores]

                n_g_spots = g_spots_local_yxz.shape[0]
                if n_g_spots == 0:
                    continue
                g_spots_score = scores_torch.omp_scores_float_to_int(g_spots_score)
                g_spots_tile = torch.ones(n_g_spots, dtype=torch.int16) * t
                g_spots_gene_no = torch.ones(n_g_spots, dtype=torch.int16) * g

                spots_local_yxz = torch.cat((spots_local_yxz, g_spots_local_yxz), dim=0)
                spots_score = torch.cat((spots_score, g_spots_score), dim=0)
                spots_tile = torch.cat((spots_tile, g_spots_tile), dim=0)
                spots_gene_no = torch.cat((spots_gene_no, g_spots_gene_no), dim=0)

                del g_spots_yxz, g_spots_local_yxz, g_spots_score, g_spots_tile, g_spots_gene_no
                del g_coefficient_image

            # STEP 5: Repeat steps 2 to 4 on every mini-tile subset.
            first_computation = False
        if (spots_tile == t).sum() == 0:
            raise ValueError(
                f"No OMP spots were found on tile {t}. Please check that registration and call spots is working. "
                + "If so, consider adjusting OMP config parameters."
            )

    if spots_score.size == 0:
        raise ValueError(
            "OMP failed to find any spots. Please check that registration and call spots is working. "
            + "If so, consider adjusting OMP config parameters."
        )

    nbp.local_yxz = np.array(spots_local_yxz)
    nbp.scores = np.array(spots_score)
    nbp.tile = np.array(spots_tile)
    nbp.gene_no = np.array(spots_gene_no)
    log.info("OMP complete")
    return nbp
