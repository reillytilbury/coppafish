import itertools
from typing import Tuple
import numpy as np

from .. import log
from ..call_spots import dot_product_score, gene_prob_score, bayes_mean, compute_bleed_matrix
from ..setup import NotebookPage

try:
    import importlib_resources
except ModuleNotFoundError:
    import importlib.resources as importlib_resources


def call_reference_spots(
    config: dict, nbp_ref_spots: NotebookPage, nbp_file: NotebookPage, nbp_basic: NotebookPage
) -> Tuple[NotebookPage, NotebookPage]:
    """
    Function to do gene assignments to reference spots. In doing so we compute some important parameters for the
    downstream analysis.

    Args:
        config: dict
            The configuration dictionary for the call spots page. Should contain the following keys:
            - gene_prob_threshold: float
                The threshold for the gene probability score.
            - target_values: list (length n_channels_use)
                If d_max[c] is the dye which channel c is brightest in, then target_values[c] is the value that we will
                try to scale all instances of d_max[c] to in this channel.
            - concentration_param_parallel: float
                The concentration parameter for the parallel direction of the prior.
            - concentration_param_perpendicular: float
                The concentration parameter for the perpendicular direction of the prior.
        nbp_ref_spots: NotebookPage
            The reference spots notebook page. This will be altered in the process.
        nbp_file: NotebookPage
            The file names notebook page.
        nbp_basic: NotebookPage
            The basic info notebook page.

    Returns:
        nbp: NotebookPage
            The call spots notebook page.
        nbp_ref_spots: NotebookPage
            The reference spots notebook page.
    """
    log.debug("Call spots started")
    nbp = NotebookPage("call_spots", {"call_spots": config})

    # assign config values that have not been provided
    if config["target_values"] is None:
        if len(nbp_basic.use_channels) == 7:
            config["target_values"] = [1, 1, 0.9, 0.7, 0.8, 1, 1]
        elif len(nbp_basic.use_channels) == 9:
            config["target_values"] = [1, 0.8, 0.2, 0.9, 0.6, 0.8, 0.3, 0.7, 1]
        else:
            raise ValueError("The target values should be provided in the config.")
    if config["d_max"] is None:
        if len(nbp_basic.use_channels) == 7:
            config["d_max"] = [0, 1, 3, 2, 4, 5, 6]
        elif len(nbp_basic.use_channels) == 9:
            config["d_max"] = [0, 1, 1, 3, 2, 4, 5, 5, 6]
        else:
            raise ValueError("The d_max values should be provided in the config.")

    gene_names, gene_codes = np.genfromtxt(nbp_file.code_book, dtype=(str, str)).transpose()
    gene_codes = np.array([[int(i) for i in gene_codes[j]] for j in range(len(gene_codes))])
    if config["kappa"] is None:
        n_genes = len(gene_names)
        config["kappa"] = 2 if n_genes <= 100 else 3

    # check shapes
    assert (
        len(config["target_values"]) == len(config["d_max"]) == len(nbp_basic.use_channels)
    ), "The target values, d_max and use_channels should have the same length."

    # load in frequently used variables
    spot_colours = nbp_ref_spots.colours.astype(float)
    spot_tile = nbp_ref_spots.tile
    n_tiles, n_rounds, n_channels_use = nbp_basic.n_tiles, nbp_basic.n_rounds, len(nbp_basic.use_channels)
    n_dyes, n_spots, n_genes = len(nbp_basic.dye_names), len(spot_colours), len(gene_names)
    use_tiles, use_channels = (nbp_basic.use_tiles, nbp_basic.use_channels)

    if nbp_file.initial_bleed_matrix is not None:
        raw_bleed_matrix = np.load(nbp_file.initial_bleed_matrix)
    else:
        raw_bleed_path = importlib_resources.files("coppafish.setup").joinpath("dye_info_raw.npy")
        raw_bleed_matrix = np.load(raw_bleed_path)[:, use_channels].astype(float)
    raw_bleed_matrix = raw_bleed_matrix / np.linalg.norm(raw_bleed_matrix, axis=1)[:, None]

    # 1. Normalise spot colours and remove background as constant offset across different rounds of the same channel
    colour_norm_factor_initial = np.zeros((n_tiles, n_rounds, n_channels_use))
    for t in use_tiles:
        colour_norm_factor_initial[t] = 1 / (np.percentile(spot_colours[spot_tile == t], 95, axis=0))
        spot_colours[spot_tile == t] *= colour_norm_factor_initial[t]
    # remove background as constant offset across different rounds of the same channel
    spot_colours -= np.percentile(spot_colours, 25, axis=1)[:, None, :]

    # 2. Compute gene probabilities for each spot
    bled_codes = raw_bleed_matrix[gene_codes]
    gene_prob_initial = gene_prob_score(spot_colours, bled_codes, kappa=config["kappa"])

    # 3. Use spots with score above threshold to work out global dye codes
    prob_mode_initial, prob_score_initial = np.argmax(gene_prob_initial, axis=1), np.max(gene_prob_initial, axis=1)
    prob_threshold = min(config["gene_prob_threshold"], np.percentile(prob_score_initial, 90))
    good = prob_score_initial > prob_threshold
    bleed_matrix_initial = compute_bleed_matrix(spot_colours[good], prob_mode_initial[good], gene_codes, n_dyes)

    # 4. Compute the free_bled_codes
    free_bled_codes_tile_indep = np.zeros((n_genes, n_rounds, n_channels_use))
    free_bled_codes = np.zeros((n_genes, n_tiles, n_rounds, n_channels_use))

    for g in range(n_genes):
        good_g = (prob_mode_initial == g) & good
        for r in range(n_rounds):
            free_bled_codes_tile_indep[g, r] = bayes_mean(
                spot_colours=spot_colours[good_g, r],
                prior_colours=bleed_matrix_initial[gene_codes[g, r]],
                conc_param_parallel=config["concentration_parameter_parallel"],
                conc_param_perp=config["concentration_parameter_perpendicular"],
            )
            for t in use_tiles:
                good_gt = (prob_mode_initial == g) & (spot_tile == t) & good
                free_bled_codes[g, t, r] = bayes_mean(
                    spot_colours=spot_colours[good_gt, r],
                    prior_colours=bleed_matrix_initial[gene_codes[g, r]],
                    conc_param_parallel=config["concentration_parameter_parallel"],
                    conc_param_perp=config["concentration_parameter_perpendicular"],
                )
    # normalise the free bled codes
    free_bled_codes_tile_indep /= np.linalg.norm(free_bled_codes_tile_indep, axis=(1, 2))[:, None, None]
    free_bled_codes[:, use_tiles] /= np.linalg.norm(free_bled_codes[:, use_tiles], axis=(2, 3))[:, :, None, None]

    # 5. compute the scale factor V_rc maximising the similarity between the tile independent codes and the target
    # values. Then rename the product V_rc * free_bled_codes to bled_codes
    rc_scale = np.ones((n_rounds, n_channels_use))
    for r, c in np.ndindex(n_rounds, n_channels_use):
        rc_genes = np.where(gene_codes[:, r] == config["d_max"][c])[0]
        n_spots_per_gene = np.array(
            [np.sum((prob_mode_initial == g) & (prob_score_initial > prob_threshold)) for g in rc_genes]
        )
        if np.sum(n_spots_per_gene) == 0:
            continue
        rc_scale[r, c] = np.sum(
            np.sqrt(n_spots_per_gene) * free_bled_codes_tile_indep[rc_genes, r, c] * config["target_values"][c]
        ) / np.sum(np.sqrt(n_spots_per_gene) * free_bled_codes_tile_indep[rc_genes, r, c] ** 2)
    bled_codes = free_bled_codes_tile_indep * rc_scale[None, :, :]
    # normalise the constrained bled codes
    bled_codes /= np.linalg.norm(bled_codes, axis=(1, 2))[:, None, None]

    # 6. compute the scale factor Q_trc maximising the similarity between the tile independent codes and the constrained
    # bled codes
    tile_scale = np.ones((n_tiles, n_rounds, n_channels_use))
    for t, r, c in itertools.product(use_tiles, range(n_rounds), range(n_channels_use)):
        relevant_genes = np.where(gene_codes[:, r] == config["target_values"][c])[0]
        n_spots_per_gene = np.array(
            [
                np.sum((prob_mode_initial == g) & (prob_score_initial > prob_threshold) & (spot_tile == t))
                for g in relevant_genes
            ]
        )
        if np.sum(n_spots_per_gene) == 0:
            continue
        tile_scale[t, r, c] = np.sum(
            np.sqrt(n_spots_per_gene) * bled_codes[relevant_genes, r, c] * free_bled_codes[relevant_genes, t, r, c]
        ) / np.sum(np.sqrt(n_spots_per_gene) * free_bled_codes[relevant_genes, t, r, c] ** 2)

    # 7. update the normalised spots and the bleed matrix, then do a second round of gene assignments with the new bled
    # codes
    spot_colours = spot_colours * tile_scale[spot_tile, :, :]  # update the spot colours
    gene_prob = gene_prob_score(spot_colours=spot_colours, bled_codes=bled_codes, kappa=config["kappa"])  # update probs
    prob_mode, prob_score = np.argmax(gene_prob, axis=1), np.max(gene_prob, axis=1)
    gene_dot_products = dot_product_score(
        spot_colours=spot_colours.reshape((n_spots, n_rounds * n_channels_use)),
        bled_codes=bled_codes.reshape((n_genes, n_rounds * n_channels_use)),
    )[-1]
    dp_mode, dp_score = np.argmax(gene_dot_products, axis=1), np.max(gene_dot_products, axis=1)
    # update bleed matrix
    good = prob_score > prob_threshold
    bleed_matrix = compute_bleed_matrix(spot_colours[good], prob_mode[good], gene_codes, n_dyes)

    # 8. save the results
    nbp.intensity = np.median(np.max(spot_colours, axis=-1), axis=-1)
    nbp.dot_product_gene_no, nbp.dot_product_gene_score = dp_mode.astype(np.int16), dp_score
    nbp.gene_probabilities_initial = gene_prob_initial
    nbp.gene_probabilities = gene_prob
    nbp.gene_names, nbp.gene_codes = gene_names, gene_codes
    nbp.initial_scale, nbp.rc_scale, nbp.tile_scale = colour_norm_factor_initial, rc_scale, tile_scale
    nbp.colour_norm_factor = colour_norm_factor_initial * tile_scale
    nbp.free_bled_codes, nbp.free_bled_codes_tile_independent = free_bled_codes, free_bled_codes_tile_indep
    nbp.bled_codes = bled_codes
    nbp.bleed_matrix_raw, nbp.bleed_matrix_initial, nbp.bleed_matrix = (
        raw_bleed_matrix,
        bleed_matrix_initial,
        bleed_matrix,
    )

    return nbp
