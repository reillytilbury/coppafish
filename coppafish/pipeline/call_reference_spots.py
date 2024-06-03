# from itertools import product
# from typing import Tuple
#
# import numpy as np
# from scipy.sparse.linalg import svds
# from scipy.sparse.linalg import svds
# from tqdm import tqdm
#
# from .. import call_spots
# from .. import spot_colors
# from .. import utils
# from .. import log
# from ..filter import base as filter_base
# from ..filter import base as filter_base
# from ..setup import NotebookPage
# from ..setup.notebook import NotebookPage
#
#
# def call_reference_spots(
#     config: dict,
#     nbp_file: NotebookPage,
#     nbp_basic: NotebookPage,
#     nbp_ref_spots: NotebookPage,
#     nbp_extract: NotebookPage,
#     nbp_register: NotebookPage,
#     transform: np.ndarray,
# ) -> Tuple[NotebookPage, NotebookPage]:
#     """
#     This produces the bleed matrix and expected code for each gene as well as producing a gene assignment based on a
#     simple dot product for spots found on the reference round.
#
#     Returns the `call_spots` notebook page and adds the following variables to the `ref_spots` page:
#     `gene_no`, `score`, `score_diff`, `intensity`.
#
#     See `'call_spots'` and `'ref_spots'` sections of `notebook_comments.json` file
#     for description of the variables in each page.
#
#     Args:
#         config: Dictionary obtained from `'call_spots'` section of config file.
#         nbp_file: `file_names` notebook page.
#         nbp_basic: `basic_info` notebook page.
#         nbp_ref_spots: `ref_spots` notebook page containing all variables produced in `pipeline/reference_spots.py` i.e.
#             `local_yxz`, `isolated`, `tile`, `colours`.
#         nbp_extract: `extract` notebook page.
#         nbp_register: `register` notebook page.
#         transform: float [n_tiles x n_rounds x n_channels x 4 x 3] affine transform for each tile, round and channel
#
#     Returns:
#         `NotebookPage[call_spots]` - Page contains bleed matrix and expected code for each gene.
#         `NotebookPage[ref_spots]` - Page contains gene assignments and info for spots found on reference round.
#             Parameters added are: intensity, score, gene_no, score_diff
#     """
#     nbp = NotebookPage("call_spots")
#     log.debug("Call ref spots started")
#
#     # 0. Initialise frequently used variables
#     # Load gene names and codes
#     gene_names, gene_codes = np.genfromtxt(nbp_file.code_book, dtype=(str, str)).transpose()
#     gene_codes = np.array([[int(i) for i in gene_codes[j]] for j in range(len(gene_codes))])
#     n_genes = len(gene_names)
#     # Load bleed matrix info
#     if nbp_file.initial_bleed_matrix is None:
#         expected_dye_names = ("ATTO425", "AF488", "DY520XL", "AF532", "AF594", "AF647", "AF750")
#         assert nbp_basic.dye_names == expected_dye_names, (
#             f"To use the default bleed matrix, dye names must be given in the order {expected_dye_names}, but got "
#             + f"{nbp_basic.dye_names}."
#         )
#         # default_bleed_matrix_filepath = importlib_resources.files('coppafish.setup').joinpath('default_bleed.npy')
#         # initial_bleed_matrix = np.load(default_bleed_matrix_filepath).copy()
#         dye_info = {
#             "ATTO425": np.array(
#                 [
#                     394,
#                     7264,
#                     499,
#                     132,
#                     53625,
#                     46572,
#                     4675,
#                     488,
#                     850,
#                     51750,
#                     2817,
#                     226,
#                     100,
#                     22559,
#                     124,
#                     124,
#                     100,
#                     100,
#                     260,
#                     169,
#                     100,
#                     100,
#                     114,
#                     134,
#                     100,
#                     100,
#                     99,
#                     103,
#                 ]
#             ),
#             "AF488": np.array(
#                 [
#                     104,
#                     370,
#                     162,
#                     114,
#                     62454,
#                     809,
#                     2081,
#                     254,
#                     102,
#                     45360,
#                     8053,
#                     368,
#                     100,
#                     40051,
#                     3422,
#                     309,
#                     100,
#                     132,
#                     120,
#                     120,
#                     100,
#                     100,
#                     100,
#                     130,
#                     99,
#                     100,
#                     99,
#                     103,
#                 ]
#             ),
#             "DY520XL": np.array(
#                 [
#                     103,
#                     114,
#                     191,
#                     513,
#                     55456,
#                     109,
#                     907,
#                     5440,
#                     99,
#                     117,
#                     2440,
#                     8675,
#                     100,
#                     25424,
#                     5573,
#                     42901,
#                     100,
#                     100,
#                     10458,
#                     50094,
#                     100,
#                     100,
#                     324,
#                     4089,
#                     100,
#                     100,
#                     100,
#                     102,
#                 ]
#             ),
#             "AF532": np.array(
#                 [
#                     106,
#                     157,
#                     313,
#                     123,
#                     55021,
#                     142,
#                     1897,
#                     304,
#                     101,
#                     1466,
#                     7980,
#                     487,
#                     100,
#                     31753,
#                     49791,
#                     4511,
#                     100,
#                     849,
#                     38668,
#                     1919,
#                     100,
#                     100,
#                     100,
#                     131,
#                     100,
#                     100,
#                     99,
#                     102,
#                 ]
#             ),
#             "AF594": np.array(
#                 [
#                     104,
#                     113,
#                     1168,
#                     585,
#                     65378,
#                     104,
#                     569,
#                     509,
#                     102,
#                     119,
#                     854,
#                     378,
#                     100,
#                     42236,
#                     5799,
#                     3963,
#                     100,
#                     100,
#                     36766,
#                     14856,
#                     100,
#                     100,
#                     3519,
#                     3081,
#                     100,
#                     100,
#                     100,
#                     103,
#                 ]
#             ),
#             "AF647": np.array(
#                 [
#                     481,
#                     314,
#                     124,
#                     344,
#                     50254,
#                     125,
#                     126,
#                     374,
#                     98,
#                     202,
#                     152,
#                     449,
#                     100,
#                     26103,
#                     402,
#                     5277,
#                     100,
#                     101,
#                     1155,
#                     27251,
#                     100,
#                     100,
#                     442,
#                     65457,
#                     100,
#                     100,
#                     100,
#                     118,
#                 ]
#             ),
#             "AF750": np.array(
#                 [
#                     106,
#                     114,
#                     107,
#                     127,
#                     65531,
#                     108,
#                     124,
#                     193,
#                     104,
#                     142,
#                     142,
#                     153,
#                     100,
#                     55738,
#                     183,
#                     168,
#                     100,
#                     99,
#                     366,
#                     245,
#                     100,
#                     100,
#                     101,
#                     882,
#                     100,
#                     100,
#                     99,
#                     2219,
#                 ]
#             ),
#         }
#         # initial_bleed_matrix is n_channels x n_dyes
#         initial_bleed_matrix = np.zeros((len(nbp_basic.use_channels), len(nbp_basic.dye_names)))
#         # Populate initial_bleed_matrix with dye info for all channels in use
#         for i, dye in enumerate(nbp_basic.dye_names):
#             initial_bleed_matrix[:, i] = dye_info[dye][list(nbp_basic.use_channels)]
#     if nbp_file.initial_bleed_matrix is not None:
#         # Use an initial bleed matrix given by the user
#         initial_bleed_matrix = np.load(nbp_file.initial_bleed_matrix)
#     expected_shape = (len(nbp_basic.use_channels), len(nbp_basic.dye_names))
#     assert initial_bleed_matrix.shape == expected_shape, (
#         f"Initial bleed matrix at {nbp_file.initial_bleed_matrix} has shape {initial_bleed_matrix.shape}, "
#         + f"expected {expected_shape}."
#     )
#
#     # Load spot colours and background colours
#     bleed_matrix = initial_bleed_matrix / np.linalg.norm(initial_bleed_matrix, axis=0)
#     colours = nbp_ref_spots.colours[:, :, nbp_basic.use_channels].astype(float)
#     bg_colours = nbp_ref_spots.bg_colours.astype(float)
#     spot_tile = nbp_ref_spots.tile
#     n_spots, n_rounds, n_channels_use = colours.shape
#     n_dyes = initial_bleed_matrix.shape[1]
#     n_tiles, use_channels = nbp_basic.n_tiles, nbp_basic.use_channels
#     colour_norm_factor = np.ones((n_tiles, n_rounds, n_channels_use))
#     gene_efficiency = np.ones((n_genes, n_rounds))
#     pseudo_bleed_matrix = np.zeros((n_tiles, n_rounds, initial_bleed_matrix.shape[0], initial_bleed_matrix.shape[1]))
#     bad_trc = [tuple(trc) for trc in nbp_basic.bad_trc]
#
#     # Part 1: Estimate norm_factor[t, r, c] for each tile t, round r and channel c + remove background
#     for t in tqdm(nbp_basic.use_tiles, desc="Estimating norm_factors for each tile"):
#         tile_colours = colours[spot_tile == t]
#         if nbp_basic.use_preseq:
#             tile_bg_colours = bg_colours[spot_tile == t]
#         else:
#             tile_bg_colours = np.percentile(tile_colours, 25, axis=1)
#             tile_bg_colours = np.repeat(tile_bg_colours[:, np.newaxis, :], n_rounds, axis=1)
#         tile_bg_strength = np.sum(np.abs(tile_bg_colours), axis=(1, 2))
#         if tile_bg_strength.size == 0 or np.allclose(tile_bg_strength, tile_bg_strength[0]):
#             log.warn(
#                 f"Failed to compute colour norm factor for {t=} because there was a lack of spots or uniform "
#                 f"background strength. Norm factor will be set to 1."
#             )
#             continue
#         weak_bg = tile_bg_strength < np.percentile(tile_bg_strength, 50)
#         if np.all(np.logical_not(weak_bg)):
#             continue
#         tile_colours = tile_colours[weak_bg]
#         # normalise pixel colours by round and channel on this tile (+ a small constant to avoid division by zero)
#         colour_norm_factor[t] = np.percentile(abs(tile_colours), 95, axis=0) + 1
#         colours[spot_tile == t] /= colour_norm_factor[t]
#     # Remove background
#     bg_codes = np.zeros((n_spots, n_rounds, n_channels_use))
#     bg = np.percentile(colours, 25, axis=1)
#     for t, r, c in product(nbp_basic.use_tiles, range(n_rounds), range(n_channels_use)):
#         if (t, r, c) in bad_trc:
#             continue
#         bg_codes[:, r, c] = bg[:, c]
#     colours -= bg_codes
#
#     # Part 2: Estimate gene assignment g[s] for each spot s
#     # This is done by calculating prob(g|s) = (prob(s|g) * prob(g) / prob(s)) under a bayes model where
#     # prob(s|g) ~ Fisher Von Mises distribution
#     bled_codes = call_spots.get_bled_codes(
#         gene_codes=gene_codes, bleed_matrix=bleed_matrix, gene_efficiency=gene_efficiency
#     )
#     # loop through tiles and append gene probs. This is done so probs can be calculated with respect to all
#     # possible rounds within a tile.
#     gene_prob = np.zeros((n_spots, n_genes))
#     gene_no = np.zeros(n_spots, dtype=int)
#     gene_prob_score = np.zeros(n_spots)
#     bad_tr = [(t, r) for t, r, _ in bad_trc]
#     for t in tqdm(nbp_basic.use_tiles, desc="Estimating gene probabilities"):
#         tile_t_spots = spot_tile == t
#         bad_r = [r for r in range(n_rounds) if (t, r) in bad_tr]
#         good_r = [r for r in range(n_rounds) if r not in bad_r]
#         tile_colours = colours[tile_t_spots][:, good_r, :]
#         bled_codes_tile = bled_codes[:, good_r, :]
#         gene_prob[tile_t_spots] = call_spots.gene_prob_score(spot_colours=tile_colours, bled_codes=bled_codes_tile)
#         gene_no[tile_t_spots] = np.argmax(gene_prob[tile_t_spots], axis=1)
#         gene_prob_score[tile_t_spots] = np.max(gene_prob[tile_t_spots], axis=1)
#
#     # Part 3: Update our colour norm factor and bleed matrix. To do this, we will sample dyes from each tile and round
#     # - generating a new un-normalised bleed matrix for each tile and round. We will then use least squares to find the
#     # best colour scaling factors omega = (w_1, ..., w_7) for each tile and round such that
#     # omega_i * initial_bleed_matrix[i] ~ bleed_matrix[t, r, i] for all i. We can then assimilate these scaling factors
#     # into our colour norm factor.
#     gene_prob_bleed_thresh = min(np.percentile(gene_prob_score, 80), 0.8)
#     bg_percentile = 50
#     bg_strength = np.linalg.norm(bg_codes, axis=(1, 2))
#     # first, estimate bleed matrix
#     bad_t = [t for t, _, _ in nbp_basic.bad_trc]
#     good_t = [t for t in nbp_basic.use_tiles if t not in bad_t]
#     for d in tqdm(range(n_dyes), desc="Estimating bleed matrix"):
#         for r in range(n_rounds):
#             my_genes = [g for g in range(n_genes) if gene_codes[g, r] == d]
#             keep = (
#                 (gene_prob_score > gene_prob_bleed_thresh)
#                 * (bg_strength < np.percentile(bg_strength, bg_percentile))
#                 * np.isin(spot_tile, good_t)
#                 * np.isin(gene_no, my_genes)
#             )
#             colours_d = colours[keep, r, :]
#             is_positive = np.sum(colours_d, axis=1) > 0
#             colours_d = colours_d[is_positive]
#             if len(colours_d) == 0:
#                 continue
#             # Now we have colours_d, we can estimate the bleed matrix for this dye
#             _, _, v = svds(colours_d, k=1)
#             v = v[0]
#             v *= np.sign(v[np.argmax(np.abs(v))])  # Make sure the largest element is positive
#             bleed_matrix[:, d] = v
#
#     # now get pseudo bleed matrix for each tile and round
#     for t, r in product(nbp_basic.use_tiles, range(n_rounds)):
#         for d in range(n_dyes):
#             my_genes = [g for g in range(n_genes) if gene_codes[g, r] == d]
#             # Skip if no spots or if this tile and round are bad
#             if [t, r] in bad_tr:
#                 pseudo_bleed_matrix[t, r, :, d] = bleed_matrix[:, d]
#                 continue
#             keep = (
#                 (spot_tile == t)
#                 * (gene_prob_score > gene_prob_bleed_thresh)
#                 * (bg_strength < np.percentile(bg_strength, bg_percentile))
#                 * np.isin(gene_no, my_genes)
#             )
#             colours_trd = colours[keep, r, :]
#             log.info(
#                 "Tile " + str(t) + " Round " + str(r) + " Dye" + str(d) + " has " + str(len(colours_trd)) + " spots."
#             )
#             if len(colours_trd) == 0:
#                 pseudo_bleed_matrix[t, r, :, d] = bleed_matrix[:, d]
#             else:
#                 pseudo_bleed_matrix[t, r, :, d] = np.mean(colours_trd, axis=0)
#
#     # We'll use these to update our colour norm factor
#     colour_norm_factor_update = np.ones_like(colour_norm_factor)
#     for t, r, c in product(nbp_basic.use_tiles, range(n_rounds), range(n_channels_use)):
#         colour_norm_factor_update[t, r, c] = np.dot(pseudo_bleed_matrix[t, r, c], bleed_matrix[c]) / np.dot(
#             bleed_matrix[c], bleed_matrix[c]
#         )
#     # Update colours and colour_norm_factor
#     for t in nbp_basic.use_tiles:
#         colours[spot_tile == t] /= colour_norm_factor_update[t]
#     colour_norm_factor *= colour_norm_factor_update
#
#     # Part 4: Estimate gene_efficiency[g, r] for each gene g and round r
#
#     ge_min_spots = 10
#     gene_prob_ge_thresh = max(np.percentile(gene_prob_score, 75), 0.75)
#     use_ge = np.zeros(n_spots, dtype=bool)
#     for g in tqdm(range(n_genes), desc="Estimating gene efficiencies"):
#         keep = (gene_no == g) * (gene_prob_score > gene_prob_ge_thresh) * np.isin(spot_tile, good_t)
#         gene_g_colours = colours[keep]
#         # Skip gene if not enough spots.
#         if len(gene_g_colours) < ge_min_spots:
#             continue
#         for r in range(n_rounds):
#             expected_dye_colour = bleed_matrix[:, gene_codes[g, r]]
#             gene_efficiency[g, r] = np.dot(np.mean(gene_g_colours[:, r], axis=0), expected_dye_colour)
#         use_ge += keep
#     # Recalculate bled_codes with updated gene_efficiency
#     bled_codes = call_spots.get_bled_codes(
#         gene_codes=gene_codes, bleed_matrix=bleed_matrix, gene_efficiency=gene_efficiency
#     )
#
#     # 3.3 Update gene coefficients
#     n_spots = colours.shape[0]
#     n_genes = bled_codes.shape[0]
#     gene_no, gene_score, gene_score_second = call_spots.dot_product_score(
#         spot_colours=colours.reshape((n_spots, -1)), bled_codes=bled_codes.reshape((n_genes, -1))
#     )[:3]
#
#     # save overwritable variables in nbp_ref_spots
#     # delete all variables in ref_spots set to None so can add them later.
#     for var in ["gene_no", "scores", "score_diff", "intensity", "background_strength", "gene_probs"]:
#         if hasattr(nbp_ref_spots, var):
#             nbp_ref_spots.__delattr__(var)
#     nbp_ref_spots.gene_no = gene_no.astype(np.int16)
#     nbp_ref_spots.scores = gene_score
#     nbp_ref_spots.score_diff = gene_score - gene_score_second
#     nbp_ref_spots.intensity = np.median(np.max(colours, axis=2), axis=1).astype(np.float32)
#     nbp_ref_spots.background_strength = bg_codes
#     nbp_ref_spots.gene_probs = gene_prob
#
#     # Save variables in nbp
#     nbp.use_ge = np.asarray(use_ge)
#     nbp.gene_names = gene_names
#     nbp.gene_codes = gene_codes
#     # Now expand variables to have n_channels channels instead of n_channels_use channels. For some variables, we
#     # also need to swap axes as the expand channels function assumes the last axis is the channel axis.
#     nbp.color_norm_factor = utils.base.expand_channels(colour_norm_factor, use_channels, nbp_basic.n_channels)
#     nbp.initial_bleed_matrix = utils.base.expand_channels(initial_bleed_matrix.T, use_channels, nbp_basic.n_channels).T
#     nbp.bleed_matrix = utils.base.expand_channels(bleed_matrix.T, use_channels, nbp_basic.n_channels).T
#     nbp.bled_codes_ge = utils.base.expand_channels(bled_codes, use_channels, nbp_basic.n_channels)
#     nbp.bled_codes = utils.base.expand_channels(
#         call_spots.get_bled_codes(
#             gene_codes=gene_codes, bleed_matrix=bleed_matrix, gene_efficiency=np.ones((n_genes, n_rounds))
#         ),
#         use_channels,
#         nbp_basic.n_channels,
#     )
#     nbp.gene_efficiency = gene_efficiency
#
#     # Extract abs intensity percentile
#     central_tile = filter_base.central_tile(nbp_basic.tilepos_yx, list(nbp_basic.use_tiles))
#     if nbp_basic.is_3d:
#         mid_z = int(nbp_basic.use_z[0] + (nbp_basic.use_z[-1] - nbp_basic.use_z[0]) // 2 - min(nbp_basic.use_z))
#     else:
#         mid_z = None
#     pixel_colours = spot_colors.get_spot_colors(
#         yxz_base=spot_colors.all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, mid_z),
#         t=central_tile,
#         transform=transform,
#         bg_scale=nbp_register.bg_scale,
#         file_type=nbp_extract.file_type,
#         nbp_file=nbp_file,
#         nbp_basic=nbp_basic,
#         nbp_register=nbp_register,
#         return_in_bounds=True,
#     )[0]
#     pixel_intensity = call_spots.get_spot_intensity(np.abs(pixel_colours) / colour_norm_factor[central_tile])
#     nbp.abs_intensity_percentile = np.percentile(pixel_intensity, np.arange(1, 101))
#     log.debug("Call ref spots complete")
#
#     return nbp, nbp_ref_spots

import itertools
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse.linalg import svds
from coppafish.call_spots.dot_product import gene_prob_score, dot_product_score

target_values = [1, 1, 0.9, 0.7, 0.8, 1, 1]
d_max = [0, 1, 3, 2, 4, 5, 6]

conc_param_parallel = .1 # it takes this many spots to scale the bled code for any round
conc_param_perp = 40 # it takes this many spots to change the bled code in a round


def bayes_mean(spot_colours: np.ndarray, prior_colours: np.ndarray, conc_param_parallel: float,
               conc_param_perp: float) -> np.ndarray:
    """
    This function computes the posterior mean of the spot colours under a prior distribution with mean prior_colours
    and covariance matrix given by a diagonal matrix with diagonal entry conc_param_parallel for the direction parallel
    to prior_colours and conc_param_perp for the direction orthogonal to prior_colours.

    Args:
        spot_colours: np.ndarray [n_spots x n_channels]
            The spot colours for each spot.
        prior_colours: np.ndarray [n_channels]
            The prior mean colours.
        conc_param_parallel: np.ndarray [n_channels]
            The concentration parameter for the direction parallel to prior_colours.
        conc_param_perp: np.ndarray [n_channels]
            The concentration parameter for the direction orthogonal to prior_colours.
    """
    n_spots, data_sum = len(spot_colours), np.sum(spot_colours, axis=0)

    prior_direction = prior_colours / np.linalg.norm(prior_colours) # normalized prior direction
    sum_parallel = (data_sum @ prior_direction) * prior_direction # projection of data sum along prior direction
    sum_perp = data_sum - sum_parallel # projection of data sum orthogonal to mean direction

    # now compute the weighted sum of the posterior mean for parallel and perpendicular directions
    posterior_parallel = (sum_parallel + conc_param_parallel * prior_direction) / (n_spots + conc_param_parallel)
    posterior_perp = sum_perp / (n_spots + conc_param_perp)
    return posterior_parallel + posterior_perp


def compute_bleed_matrix(spot_colours: np.ndarray, gene_no: np.ndarray, gene_codes: np.ndarray,
                         n_dyes: int) -> np.ndarray:
    """
    Function to compute the bleed matrix from the spot colours and the gene assignments.
    Args:
        spot_colours: np.ndarray [n_spots x n_rounds x n_channels]
            The spot colours for each spot in each round and channel.
        gene_no: np.ndarray [n_spots]
            The gene assignment for each spot.
        gene_codes: np.ndarray [n_genes x n_rounds]
            The gene codes for each gene in each round.
        n_dyes: int
            The number of dyes.

    Returns:
        bleed_matrix: np.ndarray [n_dyes x n_channels]
            The bleed matrix.
    """
    assert len(spot_colours) == len(gene_no), "Spot colours and gene_no must have the same length."
    n_spots, n_rounds, n_channels = spot_colours.shape
    bleed_matrix = np.zeros((n_dyes, n_channels))

    # loop over all dyes, find the spots which are meant to be dye d in round r, and compute the SVD
    for d in range(n_dyes):
        dye_d_colours = []
        for r in range(n_rounds):
            relevant_genes = np.where(gene_codes[:, r] == d)[0]
            relevant_gene_mask = np.isin(gene_no, relevant_genes)
            dye_d_colours.append(spot_colours[relevant_gene_mask, r, :])
        # now we have all the good colours for dye d, compute the SVD
        dye_d_colours = np.concatenate(dye_d_colours, axis=0)
        u, s, v = svds(dye_d_colours, k=1)
        v = v[0]
        # make sure largest entry in v is positive
        v *= np.sign(v[np.argmax(np.abs(v))])
        bleed_matrix[d] = v

    return bleed_matrix


def view_all_free_bled_codes(free_bled_codes: np.ndarray, n_spots: np.ndarray, gene_names: np.ndarray) -> None:
    """
    View all the tile independent free bled codes for each gene.
    Args:
        free_bled_codes: np.ndarray [n_genes x n_rounds x n_channels]
            The free bled codes for each gene in each round and channel.
        n_spots: np.ndarray [n_genes]
            The number of spots for each gene.
        gene_names: np.ndarray [n_genes]
            The names of the genes.
    """
    n_cols = 12
    n_rows = int(np.ceil(len(gene_names) / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    for i, gene_name in enumerate(gene_names):
        r, c = i // n_cols, i % n_cols
        ax[r, c].imshow(free_bled_codes[i].T, cmap='viridis')
        ax[r, c].set_title(f'{gene_name} ({n_spots[i]} spots)', fontsize=8)
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        if r == n_rows - 1:
            ax[r, c].set_xlabel('Round')
        if c == 0:
            ax[r, c].set_ylabel('Channel')

    for i in range(len(gene_names), n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        ax[r, c].axis('off')

    # add a big title
    fig.suptitle('Tile independent free bled codes', fontsize=20)
    plt.show()


def call_spots(spot_colours: np.ndarray, spot_tile: np.ndarray,
               gene_codes: np.ndarray, gene_names: np.ndarray) -> np.ndarray:
    """
    Function to do gene assignments to reference spots. In doing so we compute some important parameters for the
    downstream analysis.
    Args:
        spot_colours: np.ndarray [n_spots x n_rounds x n_channels]
            The spot colours for each spot in each round and channel.
        spot_tile: np.ndarray [n_spots]
            The tile number for each spot.
        gene_codes: np.ndarray [n_genes x n_rounds]
            The gene codes for each gene in each round.
        gene_names: np.ndarray [n_genes]
    """
    # convert spot colours to float
    spot_colours = spot_colours.astype(float)

    n_tiles, n_rounds, n_channels, n_dyes, n_spots, n_genes = 8, 7, 7, 7, len(spot_tile), len(gene_names)
    use_tiles, use_rounds, use_channels = [4, 5], np.arange(7), [5, 9, 14, 15, 18, 23, 27]
    raw_bleed_dir = r"C:\Users\reill\PycharmProjects\coppafish\coppafish\setup\dye_info_raw.npy"
    raw_bleed_matrix = np.load(raw_bleed_dir)[:, use_channels].astype(float)
    raw_bleed_matrix = raw_bleed_matrix / np.linalg.norm(raw_bleed_matrix, axis=1)[:, None]

    # 1. Normalise spot colours and remove background as constant offset across different rounds of the same channel
    colour_norm_factor_initial = np.zeros((8, 7, 7))
    for t in use_tiles:
        colour_norm_factor_initial[t] = 1 / (np.percentile(spot_colours[spot_tile == t], 95, axis=0))
        spot_colours[spot_tile == t] *= colour_norm_factor_initial[t]
    # remove background as constant offset across different rounds of the same channel
    spot_colours -= np.percentile(spot_colours, 25, axis=1)[:, None, :]

    # 2. Compute gene probabilities for each spot
    bled_codes = raw_bleed_matrix[gene_codes]
    gene_prob = gene_prob_score(spot_colours, bled_codes)

    # 3. Use spots with score above threshold to work out global dye codes
    prob_mode_initial, prob_score_initial = np.argmax(gene_prob, axis=1), np.max(gene_prob, axis=1)
    prob_threshold = 0.9
    good = prob_score_initial > prob_threshold
    bleed_matrix_initial = compute_bleed_matrix(spot_colours[good], prob_mode_initial[good], gene_codes, n_dyes)

    # 4. Compute the free_bled_codes
    free_bled_codes = np.zeros((n_genes, n_tiles, n_rounds, n_channels))
    free_bled_codes_tile_indep = np.zeros((n_genes, n_rounds, n_channels))
    for g in range(n_genes):
        for r in range(n_rounds):
            good_g = (prob_mode_initial == g) & good
            free_bled_codes_tile_indep[g, r] = bayes_mean(spot_colours=spot_colours[good_g, r],
                                                          prior_colours=bleed_matrix_initial[gene_codes[g, r]],
                                                          conc_param_parallel=conc_param_parallel,
                                                          conc_param_perp=conc_param_perp)
            for t in use_tiles:
                good_gt = (prob_mode_initial == g) & (spot_tile == t) & good
                free_bled_codes[g, t, r] = bayes_mean(spot_colours=spot_colours[good_gt, r],
                                                      prior_colours=bleed_matrix_initial[gene_codes[g, r]],
                                                      conc_param_parallel=conc_param_parallel,
                                                      conc_param_perp=conc_param_perp)
    # normalise the free bled codes
    free_bled_codes_tile_indep /= np.linalg.norm(free_bled_codes_tile_indep, axis=(1, 2))[:, None, None]
    free_bled_codes[use_tiles] /= np.linalg.norm(free_bled_codes[use_tiles], axis=(2, 3))[:, :, None, None]

    # 5. compute the scale factor V_rc maximising the similarity between the tile independent codes and the target
    # values. Then rename the product V_rc * free_bled_codes to target_bled_codes
    target_scale = np.zeros((n_rounds, n_channels))
    for r in range(n_rounds):
        for i, c in enumerate(use_channels):
            rc_genes = np.where(gene_codes[:, r] == d_max[i])[0]
            rc_gene_mask = np.isin(prob_mode_initial, rc_genes)
            good = rc_gene_mask & (prob_score_initial > prob_threshold)
            target_scale[r, i] = (np.sum(spot_colours[good, r, i] * target_values[i]) /
                                  np.sum(spot_colours[good, r, i] ** 2))
    target_bled_codes = free_bled_codes_tile_indep * target_scale[None, :, :]
    # normalise the target bled codes
    target_bled_codes /= np.linalg.norm(target_bled_codes, axis=(1, 2))[:, None, None]

    # 6. compute the scale factor Q_trc maximising the similarity between the tile independent codes and the target
    # bled codes
    scale_factor_update = np.ones((n_tiles, n_rounds, n_channels))
    for t, r, c in itertools.product(use_tiles, range(n_rounds), range(n_channels)):
        n_tg = np.array([np.sum((spot_tile == t) & (prob_mode_initial == g) & (prob_score_initial > prob_threshold))
                         for g in range(n_genes)])
        scale_factor_update[t, r, c] = (np.sum(np.sqrt(n_tg) * target_bled_codes[:, r, c], axis=0) /
                                        np.sum(np.sqrt(n_tg) * free_bled_codes[:, t, r, c], axis=0))

    # 7. update the normalised spots and the bleed matrix, then do a second round of gene assignments with the free
    # bled codes
    spot_colours = spot_colours * scale_factor_update[spot_tile, :, :] # update the spot colours
    gene_prob = gene_prob_score(spot_colours=spot_colours, bled_codes=target_bled_codes) # update probs
    prob_mode, prob_score = np.argmax(gene_prob, axis=1), np.max(gene_prob, axis=1)
    # update bleed matrix
    good = prob_score > prob_threshold
    bleed_matrix = compute_bleed_matrix(spot_colours[good], prob_mode[good], gene_codes, n_dyes)

    return None


colours = np.load(r"C:\Users\reill\Desktop\local_datasets\dante\dante_0_11_0\call_spots_variables\colours.npy")
tile = np.load(r"C:\Users\reill\Desktop\local_datasets\dante\dante_0_11_0\call_spots_variables\spot_tile.npy")
gene_codes = np.load(r"C:\Users\reill\Desktop\local_datasets\dante\dante_0_11_0\call_spots_variables\gene_codes.npy")
gene_names = np.load(r"C:\Users\reill\Desktop\local_datasets\dante\dante_0_11_0\call_spots_variables\gene_names.npy")
call_spots(colours, tile, gene_codes, gene_names)
