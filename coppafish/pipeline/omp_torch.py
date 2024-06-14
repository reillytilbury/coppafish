import os
import math as maths
from typing import List, Tuple

import tqdm
import torch
import numpy as np
from typing_extensions import assert_type

from .. import log
from ..call_spots import background_pytorch, qual_check_torch
from ..find_spots import detect_torch
from ..omp import base, coefs_torch, scores_torch, spots_torch
from ..setup import NotebookPage


def run_omp(
    config: dict,
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_register: NotebookPage,
    nbp_register_debug: NotebookPage,
    nbp_call_spots: NotebookPage,
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

    log.info("OMP started")
    log.debug(f"{torch.cuda.is_available()=}")
    log.debug(f"{config['force_cpu']=}")

    nbp = NotebookPage("omp")

    # We want exact, reproducible results.
    torch.backends.cudnn.deterministic = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    n_genes = nbp_call_spots.bled_codes.shape[0]
    n_rounds_use = len(nbp_basic.use_rounds)
    n_channels_use = len(nbp_basic.use_channels)
    spot_shape_size_xy: int = config["spot_shape"][0]
    spot_radius_xy: int = maths.ceil(spot_shape_size_xy / 2)
    tile_shape: Tuple[int] = nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)
    colour_norm_factor = np.array(nbp_call_spots.colour_norm_factor, dtype=np.float32)
    colour_norm_factor = torch.asarray(colour_norm_factor).float()
    first_computation = True

    subset_z_size: int = max(nbp_basic.use_z)
    subset_size_xy: int = config["subset_size_xy"]

    if subset_size_xy <= spot_radius_xy * 2:
        raise ValueError(
            "The subset size is too small for the given spot size. Reduce spot_shape in x and y directions or increase"
            + " the subset_size_xy to facilitate a large spot in the config"
        )

    subset_shape = (subset_size_xy, subset_size_xy, subset_z_size)
    # Find the bottom-left position of every subset to break the entire tile up into.
    subset_origin_start = (-spot_radius_xy, -spot_radius_xy, 0)
    subset_origin_new = list(subset_origin_start)
    subset_origins_yxz: List[Tuple[int]] = []
    subset_step_yx = subset_size_xy - 2 * spot_radius_xy
    while True:
        if subset_origin_new[0] >= nbp_basic.tile_sz:
            break
        subset_origins_yxz.append(tuple(subset_origin_new.copy()))

        subset_origin_new[1] += subset_step_yx
        if subset_origin_new[1] >= nbp_basic.tile_sz:
            subset_origin_new[1] = subset_origin_start[1]
            subset_origin_new[0] += subset_step_yx

    log.debug(f"Subset shape: {subset_shape}")
    log.debug(f"Running {len(subset_origins_yxz)} subsets for each tile")
    # Set the first subset to somewhere in the middle of the tile for better mean spot computation.
    subset_origins_yxz[0], subset_origins_yxz[len(subset_origins_yxz) // 2] = (
        subset_origins_yxz[len(subset_origins_yxz) // 2],
        subset_origins_yxz[0],
    )

    # Results are appended to these arrays
    spots_local_yxz = torch.zeros((0, 3), dtype=torch.int16)
    spots_tile = torch.zeros(0, dtype=torch.int16)
    spots_gene_no = torch.zeros(0, dtype=torch.int16)
    spots_score = torch.zeros(0, dtype=torch.float16)
    spots_colours = torch.zeros((0, n_rounds_use, n_channels_use), dtype=torch.int32)

    for t in nbp_basic.use_tiles:
        # STEP 1: Load every registered sequencing round/channel image into memory
        log.debug(f"Loading tile {t} colours")
        colour_image = base.load_spot_colours(nbp_basic, nbp_file, nbp_extract, nbp_register, nbp_register_debug, t)
        log.debug(f"Loading tile {t} colours complete")

        description = f"Computing OMP on tile {t} using the "
        description += "gpu" if (not config["force_cpu"] and torch.cuda.is_available()) else "cpu"
        for subset_yxz in tqdm.tqdm(subset_origins_yxz, desc=description, unit="subset"):
            # STEP 2: Compute OMP coefficients on a subset: a mini tile with the same number of z planes.

            def subset_to_tile_positions(positions_yxz: torch.Tensor) -> torch.Tensor:
                return positions_yxz.detach().clone() + torch.asarray(subset_yxz)[np.newaxis]

            def get_valid_subset_positions(positions_yxz: torch.Tensor) -> torch.Tensor:
                assert (positions_yxz >= 0).all()
                assert (positions_yxz[:, [0, 1]] < subset_shape[0]).all()

                valid = (
                    (positions_yxz[:, 0] >= spot_radius_xy)
                    * (positions_yxz[:, 1] >= spot_radius_xy)
                    * (positions_yxz[:, 2] >= 0)
                    * (positions_yxz[:, 0] < (subset_shape[0] - spot_radius_xy))
                    * (positions_yxz[:, 1] < (subset_shape[1] - spot_radius_xy))
                    * (positions_yxz[:, 2] < (subset_shape[2]))
                )
                tile_positions_yxz = subset_to_tile_positions(positions_yxz)
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
            subset_colours *= colour_norm_factor[[t]]
            subset_colours, bg_coefficients, bg_codes = background_pytorch.fit_background(subset_colours)
            subset_intensities = qual_check_torch.get_spot_intensity(subset_colours)
            pixel_intensity_threshold = torch.quantile(subset_intensities, q=config["pixel_max_percentile"] / 100)
            do_not_compute_on = subset_intensities < pixel_intensity_threshold
            del subset_intensities, pixel_intensity_threshold
            bled_codes = nbp_call_spots.bled_codes
            assert (~np.isnan(bled_codes)).all(), "bled codes GE cannot contain nan values"
            assert np.allclose(np.linalg.norm(bled_codes, axis=(1, 2)), 1), "bled codes GE must be L2 normalised"
            bled_codes = torch.asarray(bled_codes.astype(np.float32))
            subset_colours = subset_colours.reshape((-1, n_rounds_use * n_channels_use))
            bled_codes = bled_codes.reshape((n_genes, n_rounds_use * n_channels_use))
            bg_codes = bg_codes.reshape((n_channels_use, n_rounds_use * n_channels_use))
            coefficient_image = coefs_torch.compute_omp_coefficients(
                subset_colours,
                bled_codes,
                maximum_iterations=config["max_genes"],
                background_coefficients=bg_coefficients,
                background_codes=bg_codes,
                dot_product_threshold=config["dp_thresh"],
                dot_product_norm_shift=0.0,
                weight_coefficient_fit=config["weight_coef_fit"],
                alpha=config["alpha"],
                beta=config["beta"],
                do_not_compute_on=do_not_compute_on,
                force_cpu=config["force_cpu"],
            )
            del subset_colours, bg_coefficients, bg_codes, bled_codes, do_not_compute_on

            # STEP 2.5: On the first OMP subset/tile, compute the OMP spot shape using the found coefficients.
            if first_computation:
                log.debug("Computing spot and mean spot")
                isolated_spots_yxz = torch.zeros((0, 3), dtype=torch.int16)
                isolated_gene_numbers = torch.zeros(0, dtype=torch.int16)
                shape_isolation_distance_z = config["shape_isolation_distance_z"]
                if shape_isolation_distance_z is None:
                    shape_isolation_distance_z = maths.ceil(
                        config["shape_isolation_distance_yx"] * nbp_basic.pixel_size_xy / nbp_basic.pixel_size_z
                    )
                for g in range(n_genes):
                    if isolated_spots_yxz.size(0) >= config["spot_shape_max_spots"]:
                        isolated_spots_yxz = isolated_spots_yxz[: config["spot_shape_max_spots"]]
                        isolated_gene_numbers = isolated_gene_numbers[: config["spot_shape_max_spots"]]
                        continue
                    g_coefficient_image = torch.asarray(coefficient_image[:, [g]].toarray()).reshape(subset_shape)
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
                    g_coefficient_image = torch.asarray(coefficient_image[:, [g]].toarray().reshape(subset_shape))
                    g_mean_spot = spots_torch.compute_mean_spot_from(
                        g_coefficient_image,
                        isolated_spots_yxz[isolated_gene_numbers == g],
                        config["spot_shape"],
                        config["force_cpu"],
                    )
                    del g_coefficient_image
                    mean_spots[g] = g_mean_spot
                    weights[g] = (isolated_gene_numbers == g).sum()
                    del g_mean_spot
                del isolated_gene_numbers, isolated_spots_yxz
                if weights.sum() == 0:
                    raise ValueError(
                        f"OMP Failed to find any isolated spots. Make sure that registration is working as expected. "
                        + "If so, Consider reducing shape_isolation_distance_yx or shape_coefficient_threshold in the "
                        + "omp config then re-running.",
                    )
                mean_spot = torch.mean(mean_spots * weights[:, np.newaxis, np.newaxis, np.newaxis], dim=0).float()
                mean_spot = torch.clip(mean_spot * n_genes / weights.sum(), -1, 1).float()
                log.debug(f"OMP spot and mean spot computed using {int(weights.sum().item())} detected spots")
                del mean_spots, weights

                spot = torch.zeros_like(mean_spot, dtype=torch.int16)
                spot[mean_spot >= config["shape_sign_thresh"]] = 1

                edge_counts = spots_torch.count_edge_ones(spot)
                if edge_counts > 0:
                    log.warn(
                        f"The spot contains {edge_counts} ones on the x/y edges. You may need to increase spot_shape in"
                        + " the OMP config to avoid cropping the spot. Check _omp.pdf to see the spot image."
                    )

                n_positives = (spot == 1).sum()
                message = f"Computed spot contains only {n_positives} strongly positive values."
                if n_positives < 5:
                    message += f" You may need to reduce shape_sign_thresh in OMP config"
                    if n_positives == 0:
                        raise ValueError(message)
                    log.warn(message)
                else:
                    log.debug(message)

                nbp.spot_tile = t
                nbp.mean_spot = np.array(mean_spot)
                nbp.spot = np.array(spot)
                log.debug("Computing spot and mean spot complete")

            for g in range(n_genes):
                # STEP 3: Detect spots on the subset except at the x and y edges.
                g_coefficient_image = torch.asarray(coefficient_image[:, [g]].toarray().reshape(subset_shape))
                g_spots_yxz, _ = detect_torch.detect_spots(
                    image=g_coefficient_image,
                    intensity_thresh=config["coefficient_threshold"],
                    radius_xy=config["radius_xy"],
                    radius_z=config["radius_z"],
                    force_cpu=config["force_cpu"],
                )
                # Convert spot positions in the subset image to positions on the tile.
                valid_positions = get_valid_subset_positions(g_spots_yxz)
                g_spots_local_yxz = subset_to_tile_positions(g_spots_yxz)
                if valid_positions.sum() == 0:
                    continue

                g_spots_yxz = g_spots_yxz[valid_positions]
                g_spots_local_yxz = g_spots_local_yxz[valid_positions]

                # STEP 4: Score the detections using the coefficients.
                g_spots_score = scores_torch.score_coefficient_image(
                    coefficient_image=g_coefficient_image,
                    points=g_spots_yxz,
                    spot=spot,
                    mean_spot=mean_spot,
                    high_coefficient_bias=config["high_coef_bias"],
                    force_cpu=config["force_cpu"],
                )

                # Remove bad scoring spots (i.e. false gene reads)
                keep_scores = g_spots_score >= config["score_threshold"]
                g_spots_local_yxz = g_spots_local_yxz[keep_scores].to(dtype=torch.int16)
                g_spots_yxz = g_spots_yxz[keep_scores]
                g_spots_score = g_spots_score[keep_scores].to(dtype=torch.float16)
                n_g_spots = g_spots_local_yxz.shape[0]
                if n_g_spots == 0:
                    continue

                g_spots_tile = torch.ones(n_g_spots, dtype=torch.int16) * t
                g_spots_gene_no = torch.ones(n_g_spots, dtype=torch.int16) * g

                spots_local_yxz = torch.cat((spots_local_yxz, g_spots_local_yxz), dim=0)
                spots_score = torch.cat((spots_score, g_spots_score), dim=0)
                spots_tile = torch.cat((spots_tile, g_spots_tile), dim=0)
                spots_gene_no = torch.cat((spots_gene_no, g_spots_gene_no), dim=0)

                del g_spots_yxz, g_spots_local_yxz, g_spots_score, g_spots_tile, g_spots_gene_no
                del g_coefficient_image

            # STEP 5: Repeat steps 2 to 4 on every subset.
            first_computation = False

        t_spots = spots_tile == t
        if t_spots.sum() == 0:
            raise ValueError(
                f"No OMP spots found on tile {t}. Please check that registration and call spots is working. "
                + "If so, consider adjusting OMP config parameters."
            )
        # For each detected spot, save the image intensity at its location, without background fitting.
        t_local_yxzs = tuple(spots_local_yxz[t_spots].int().T)
        t_spots_colours = torch.asarray(colour_image[t_local_yxzs].astype(np.int32) - nbp_basic.tile_pixel_value_shift)
        spots_colours = torch.cat((spots_colours, t_spots_colours), dim=0)

        del colour_image, t_spots, t_local_yxzs, t_spots_colours

    nbp.local_yxz = np.array(spots_local_yxz)
    nbp.scores = np.array(spots_score)
    nbp.tile = np.array(spots_tile)
    nbp.gene_no = np.array(spots_gene_no)
    nbp.colours = np.array(spots_colours)
    log.info("OMP complete")
    return nbp
