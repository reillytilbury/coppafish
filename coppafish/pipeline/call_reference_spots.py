import numpy as np
from scipy.sparse.linalg import svds
import warnings
try:
    import importlib_resources
except ModuleNotFoundError:
    import importlib.resources as importlib_resources
from typing import Tuple

from ..setup.notebook import NotebookPage, Notebook
from .. import call_spots
from .. import spot_colors
from .. import utils
from coppafish import extract
from ..call_spots import get_spot_intensity
from tqdm import tqdm
from itertools import product
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib
import mplcursors
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def call_reference_spots(config: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage,
                         nbp_ref_spots: NotebookPage, nbp_extract: NotebookPage, transform: np.ndarray,
                         overwrite_ref_spots: bool = False) -> Tuple[NotebookPage, NotebookPage]:
    """
    This produces the bleed matrix and expected code for each gene as well as producing a gene assignment based on a
    simple dot product for spots found on the reference round.

    Returns the `call_spots` notebook page and adds the following variables to the `ref_spots` page:
    `gene_no`, `score`, `score_diff`, `intensity`.

    See `'call_spots'` and `'ref_spots'` sections of `notebook_comments.json` file
    for description of the variables in each page.

    Args:
        config: Dictionary obtained from `'call_spots'` section of config file.
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        nbp_ref_spots: `ref_spots` notebook page containing all variables produced in `pipeline/reference_spots.py` i.e.
            `local_yxz`, `isolated`, `tile`, `colors`.
            `gene_no`, `score`, `score_diff`, `intensity` should all be `None` to add them here, unless
            `overwrite_ref_spots == True`.
        transform: float [n_tiles x n_rounds x n_channels x 4 x 3] affine transform for each tile, round and channel
        overwrite_ref_spots: If `True`, the variables:
            * `gene_no`
            * `score`
            * `score_diff`
            * `intensity`

            in `nbp_ref_spots` will be overwritten if they exist. If this is `False`, they will only be overwritten
            if they are all set to `None`, otherwise an error will occur.

    Returns:
        `NotebookPage[call_spots]` - Page contains bleed matrix and expected code for each gene.
        `NotebookPage[ref_spots]` - Page contains gene assignments and info for spots found on reference round.
            Parameters added are: intensity, score, gene_no, score_diff
    """
    if overwrite_ref_spots:
        warnings.warn("\noverwrite_ref_spots = True so will overwrite:\ngene_no, gene_score, score_diff, intensity,"
                      "\nbackground_strength in nbp_ref_spots.")
    else:
        # Raise error if data in nbp_ref_spots already exists that will be overwritten in this function.
        error_message = ""
        for var in ['gene_no', 'gene_score', 'score_diff', 'intensity', 'background_strength', 'gene_probs',
                    'dye_strengths']:
            if hasattr(nbp_ref_spots, var) and nbp_ref_spots.__getattribute__(var) is not None:
                error_message += f"\nnbp_ref_spots.{var} is not None but this function will overwrite {var}." \
                                 f"\nRun with overwrite_ref_spots = True to get past this error."
        if len(error_message) > 0:
            raise ValueError(error_message)

    nbp_ref_spots.finalized = False  # So we can add and delete ref_spots page variables
    # delete all variables in ref_spots set to None so can add them later.
    for var in ['gene_no', 'score', 'score_diff', 'intensity', 'background_strength',  'gene_probs', 'dye_strengths',
                'gene_probs_initial', 'gene_probs_mid']:
        if hasattr(nbp_ref_spots, var):
            nbp_ref_spots.__delattr__(var)
    nbp = NotebookPage("call_spots")

    # 0. Initialise frequently used variables
    # Load gene names and codes
    gene_names, gene_codes = np.genfromtxt(nbp_file.code_book, dtype=(str, str)).transpose()
    gene_codes = np.array([[int(i) for i in gene_codes[j]] for j in range(len(gene_codes))])
    n_genes = len(gene_names)
    # Load bleed matrix info
    if nbp_file.initial_bleed_matrix is None:
        expected_dye_names = ['ATTO425', 'AF488', 'DY520XL', 'AF532', 'AF594', 'AF647', 'AF750']
        assert nbp_basic.dye_names == expected_dye_names, \
            f'To use the default bleed matrix, dye names must be given in the order {expected_dye_names}, but got ' \
                + f'{nbp_basic.dye_names}.'
        # default_bleed_matrix_filepath = importlib_resources.files('coppafish.setup').joinpath('default_bleed.npy')
        # initial_bleed_matrix = np.load(default_bleed_matrix_filepath).copy()
        dye_info = \
            {'ATTO425': np.array([394, 7264, 499, 132, 53625, 46572, 4675, 488, 850,
                                  51750, 2817, 226, 100, 22559, 124, 124, 100, 100,
                                  260, 169, 100, 100, 114, 134, 100, 100, 99,
                                  103]),
             'AF488': np.array([104, 370, 162, 114, 62454, 809, 2081, 254, 102,
                                45360, 8053, 368, 100, 40051, 3422, 309, 100, 132,
                                120, 120, 100, 100, 100, 130, 99, 100, 99,
                                103]),
             'DY520XL': np.array([103, 114, 191, 513, 55456, 109, 907, 5440, 99,
                                  117, 2440, 8675, 100, 25424, 5573, 42901, 100, 100,
                                  10458, 50094, 100, 100, 324, 4089, 100, 100, 100,
                                  102]),
             'AF532': np.array([106, 157, 313, 123, 55021, 142, 1897, 304, 101,
                                1466, 7980, 487, 100, 31753, 49791, 4511, 100, 849,
                                38668, 1919, 100, 100, 100, 131, 100, 100, 99,
                                102]),
             'AF594': np.array([104, 113, 1168, 585, 65378, 104, 569, 509, 102,
                                119, 854, 378, 100, 42236, 5799, 3963, 100, 100,
                                36766, 14856, 100, 100, 3519, 3081, 100, 100, 100,
                                103]),
             'AF647': np.array([481, 314, 124, 344, 50254, 125, 126, 374, 98,
                                202, 152, 449, 100, 26103, 402, 5277, 100, 101,
                                1155, 27251, 100, 100, 442, 65457, 100, 100, 100,
                                118]),
             'AF750': np.array([106, 114, 107, 127, 65531, 108, 124, 193, 104,
                                142, 142, 153, 100, 55738, 183, 168, 100, 99,
                                366, 245, 100, 100, 101, 882, 100, 100, 99,
                                2219])}
        # initial_bleed_matrix is n_channels x n_dyes
        initial_bleed_matrix = np.zeros((len(nbp_basic.use_channels), len(nbp_basic.dye_names)))
        # Populate initial_bleed_matrix with dye info for all channels in use
        for i, dye in enumerate(nbp_basic.dye_names):
            initial_bleed_matrix[:, i] = dye_info[dye][nbp_basic.use_channels]
    if nbp_file.initial_bleed_matrix is not None:
        # Use an initial bleed matrix given by the user
        initial_bleed_matrix = np.load(nbp_file.initial_bleed_matrix)
    expected_shape = (len(nbp_basic.use_channels), len(nbp_basic.dye_names))
    assert initial_bleed_matrix.shape == expected_shape, \
        f'Initial bleed matrix at {nbp_file.initial_bleed_matrix} has shape {initial_bleed_matrix.shape}, ' \
            + f'expected {expected_shape}.'

    # Load spot colours and background colours
    n_tiles, n_rounds, n_channels_use = nbp_basic.n_tiles, nbp_basic.n_rounds, len(nbp_basic.use_channels)
    colours = nbp_ref_spots.colours[:, :, nbp_basic.use_channels].astype(float)
    bg_colours = nbp_ref_spots.bg_colours.astype(float)
    spot_tile = nbp_ref_spots.tile.astype(int)
    n_spots, n_dyes, use_channels = colours.shape[0], len(nbp_basic.dye_names), nbp_basic.use_channels
    target_channel_strength = config['target_channel_strength']
    use_tiles = nbp_basic.use_tiles

    # initialise bleed matrices
    initial_bleed_matrix = initial_bleed_matrix / np.linalg.norm(initial_bleed_matrix, axis=0)
    bled_codes = call_spots.get_bled_codes(gene_codes=gene_codes, bleed_matrix=initial_bleed_matrix)

    # Initialise the colour norm factor
    colour_norm_factor_initial = np.ones((n_tiles, n_rounds, n_channels_use))

    # Part 1: Estimate norm_factor[t, r, c] for each tile t, round r and channel c + remove background
    for t in tqdm(nbp_basic.use_tiles, desc='Estimating norm_factors for each tile'):
        tile_colours = colours[spot_tile == t]
        tile_bg_colours = bg_colours[spot_tile == t]
        tile_bg_strength = np.sum(np.abs(tile_bg_colours), axis=(1, 2))
        weak_bg = tile_bg_strength < np.percentile(tile_bg_strength, 50)
        if (np.all(np.logical_not(weak_bg))):
            continue
        tile_colours = tile_colours[weak_bg]
        # normalise pixel colours by round and channel on this tile
        colour_norm_factor_initial[t] = 1 / np.percentile(abs(tile_colours), 95, axis=0)
        colours[spot_tile == t] *= colour_norm_factor_initial[t]
    # Remove background
    bg_codes = np.zeros((n_spots, n_rounds, n_channels_use))
    bg = np.percentile(colours, 25, axis=1)
    for r, c in product(range(n_rounds), range(n_channels_use)):
        bg_codes[:, r, c] = bg[:, c]
    colours -= bg_codes
    # Define an inlier mask to remove outliers
    rc_max = np.max(colours.reshape(n_spots, -1), axis=1)
    rc_min = np.min(colours.reshape(n_spots, -1), axis=1)
    inlier_mask = (np.max(colours.reshape(n_spots, -1), axis=1) < np.percentile(rc_max, 99)) * \
                  (np.min(colours.reshape(n_spots, -1), axis=1) > np.percentile(rc_min, 1))

    # Begin the iterative process of estimating bleed matrix, colour norm factor and free bled codes
    n_iter = config['n_iter']
    gene_prob_score = np.zeros((n_iter, n_spots))
    gene_no = np.zeros((n_iter, n_spots), dtype=int)
    target_matching_scale = np.zeros((n_iter, n_rounds, n_channels_use))
    scale = np.ones((n_iter, n_tiles, n_rounds, n_channels_use))
    bleed_matrix = np.zeros((n_iter, n_channels_use, n_dyes))
    free_bled_codes_tile_indep = np.zeros((n_genes, n_rounds, n_channels_use))
    colours_scaled = colours.copy()

    # Begin iterations!
    for iter in tqdm(range(n_iter), desc='Annealing Iterations'):
        # 1. Scale the spots by the previous iteration's scale
        if iter > 0:
            colours_scaled *= scale[iter - 1, spot_tile]

        # 2. Gene assignments using FVM
        gene_prob = call_spots.gene_prob_score(spot_colours=colours_scaled, bled_codes=bled_codes)
        gene_no_temp = np.argmax(gene_prob, axis=1)
        gene_prob_score_temp = np.max(gene_prob, axis=1)
        # Save gene_no and gene_prob_score, so we can see how much they change between iterations
        gene_no[iter] = gene_no_temp
        gene_prob_score[iter] = gene_prob_score_temp

        if iter == 0:
            gene_probs_initial = gene_prob
        else:
            frac_mismatch = np.sum(gene_no[iter] != gene_no[iter - 1]) / n_spots
            print('Fraction of spots assigned to different genes: {:.3f}'.format(frac_mismatch))

        # 3. Update bleed matrix
        bleed_matrix_prob_thresh = config['bleed_matrix_prob_thresh']
        high_prob = gene_prob_score[iter] > bleed_matrix_prob_thresh
        low_bg = np.linalg.norm(bg_codes, axis=(1, 2)) < np.percentile(np.linalg.norm(bg_codes, axis=(1, 2)), 50)
        for d in range(n_dyes):
            dye_d_spots = np.zeros((0, n_channels_use))
            for r in range(n_rounds):
                dye_d_round_r_genes = np.where(gene_codes[:, r] == d)[0]
                is_relevant_gene = np.isin(gene_no[iter], dye_d_round_r_genes)
                dye_d_spots = np.concatenate((dye_d_spots, colours_scaled[is_relevant_gene * high_prob * low_bg *
                                                                          inlier_mask, r]))
            is_positive = np.sum(dye_d_spots, axis=1) > 0
            dye_d_spots = dye_d_spots[is_positive]
            if len(dye_d_spots) == 0:
                continue
            u, s, v = svds(dye_d_spots, k=1)
            v = v[0]
            v *= np.sign(v[np.argmax(np.abs(v))])  # Make sure the largest element is positive
            bleed_matrix[iter, :, d] = v

        # as part of this step, update the bled_codes (not the free_bled_codes)
        for g, r in product(range(n_genes), range(n_rounds)):
            bled_codes[g, r] = bleed_matrix[iter, :, gene_codes[g, r]] / np.sqrt(n_rounds)

        # We need to have a way of changing between dyes and channels, so we define d_max and d_2_max
        d_max = np.zeros(n_channels_use, dtype=int)
        d_2nd_max = np.zeros(n_channels_use, dtype=int)
        for c in range(n_channels_use):
            d_max[c] = np.argmax(bleed_matrix[iter, c])
            d_2nd_max[c] = np.argsort(bleed_matrix[iter, c])[-2]

        # 4. Estimate free_bled_codes
        # conc_param_round = .1  # it takes this many spots to scale the bled code for any round
        # conc_param_other = 40  # it takes this many spots to change the bled code in a round
        conc_param_round = config['conc_param_round']
        conc_param_other = config['conc_param_other']
        free_bled_prob_thresh = config['free_bled_prob_thresh']
        for g in range(n_genes):
            keep = (gene_prob_score[iter] > free_bled_prob_thresh) * (gene_no[iter] == g) * inlier_mask
            colours_g = colours_scaled[keep]
            if np.sum(keep) <= 1:
                continue
            for r in range(n_rounds):
                dye_colour = bleed_matrix[iter, :, gene_codes[g, r]] / np.sqrt(n_rounds)
                free_bled_codes_tile_indep[g, r] = bayes_mean(data=colours_g[:, r], prior_mean=dye_colour,
                                                              mean_dir_conc=conc_param_round,
                                                              other_conc=conc_param_other)
            free_bled_codes_tile_indep[g] /= np.linalg.norm(free_bled_codes_tile_indep[g])

        free_bled_codes = np.zeros((n_tiles, n_genes, n_rounds, n_channels_use))
        for t, g in product(use_tiles, range(n_genes)):
            keep = (spot_tile == t) * (gene_prob_score[iter] > free_bled_prob_thresh) * (gene_no[iter] == g) * \
                   inlier_mask
            colours_tg = colours_scaled[keep]
            if np.sum(keep) <= 1:
                for r in range(n_rounds):
                    dye_colour = bleed_matrix[iter, :, gene_codes[g, r]] / np.sqrt(n_rounds)
                    free_bled_codes[t, g, r] = dye_colour
                continue
            for r in range(n_rounds):
                dye_colour = bleed_matrix[iter, :, gene_codes[g, r]] / np.sqrt(n_rounds)
                free_bled_codes[t, g, r] = bayes_mean(data=colours_tg[:, r], prior_mean=dye_colour,
                                                      mean_dir_conc=conc_param_round, other_conc=conc_param_other)
            free_bled_codes[t, g] /= np.linalg.norm(free_bled_codes[t, g])
        n_reads = np.zeros((n_tiles, n_genes))
        for t, g in product(use_tiles, range(n_genes)):
            n_reads[t, g] = np.sum((spot_tile == t) * (gene_no == g) * (gene_prob_score > free_bled_prob_thresh) *
                                   inlier_mask)

        # 5. Estimate scale.
        # This needs to be broken down into the computation of an auxiliary scale (target_matching_scale)
        # and the computation of the final scale (scale)
        for r, c in product(range(n_rounds), range(n_channels_use)):
            relevant_dye = d_max[c]
            target_value = target_channel_strength[c] / np.sqrt(n_rounds)
            relevant_genes = np.where(gene_codes[:, r] == relevant_dye)[0]
            F = target_value * np.ones(len(relevant_genes))
            G = free_bled_codes_tile_indep[relevant_genes, r, c]
            n = np.sqrt(np.sum(n_reads[:, relevant_genes], axis=0))
            target_matching_scale[iter, r, c] = np.sum(F * G * n) / np.sum(G * G * n)

        for t, r, c in product(use_tiles, range(n_rounds), range(n_channels_use)):
            dye_c = d_max[c]
            dye_2nd_c = d_2nd_max[c]
            # we want to know how much more to weight contributions from dye_c than dye_2nd_c. We will do this by
            # looking at the ratio of their expected strengths.
            dye_2_strength = bleed_matrix[iter, c, dye_2nd_c] / bleed_matrix[iter, c, dye_c]
            relevant_genes_dye_c = np.where(gene_codes[:, r] == dye_c)[0]
            v_rc = target_matching_scale[iter, r, c]
            F = free_bled_codes_tile_indep[relevant_genes_dye_c, r, c] * v_rc
            G = free_bled_codes[t, relevant_genes_dye_c, r, c]
            n = np.sqrt(n_reads[t, relevant_genes_dye_c])
            relevant_genes_dye_2nd_c = np.where(gene_codes[:, r] == dye_2nd_c)[0]
            F = np.concatenate((F, free_bled_codes_tile_indep[relevant_genes_dye_2nd_c, r, c] * v_rc))
            G = np.concatenate((G, free_bled_codes[t, relevant_genes_dye_2nd_c, r, c]))
            n = np.concatenate((n, dye_2_strength * np.sqrt(n_reads[t, relevant_genes_dye_2nd_c])))
            if np.sum(n) == 0:
                continue
            A = np.sum(n * F * G) / np.sum(n * G * G)
            scale[iter, t, r, c] = A

    gene_probs_mid = gene_prob
    colour_norm_factor_correction = np.product(scale, axis=0)
    auxiliary_correction = np.product(target_matching_scale, axis=0)
    colour_norm_factor = colour_norm_factor_initial * colour_norm_factor_correction
    # Get the dot product score for each spot and the final gene probabilities
    gene_probs_final = np.zeros((n_spots, n_genes))
    dp_gene_no = np.zeros(n_spots, dtype=int)
    dp_gene_score = np.zeros(n_spots)
    dp_gene_score_second = np.zeros(n_spots)
    for t in use_tiles:
        t_mask = spot_tile == t
        n_spots_t = np.sum(t_mask)
        dp_gene_no[t_mask], dp_gene_score[t_mask], dp_gene_score_second[t_mask] \
            = call_spots.dot_product_score(spot_colours=colours_scaled[t_mask].reshape((n_spots_t, -1)),
                                           bled_codes=free_bled_codes[t].reshape((n_genes, -1)))[:3]
        gene_probs_final[t_mask] = call_spots.gene_prob_score(spot_colours=colours_scaled[t_mask],
                                                              bled_codes=free_bled_codes[t])

    # save overwritable variables in nbp_ref_spots
    nbp_ref_spots.gene_no = dp_gene_no
    nbp_ref_spots.score = dp_gene_score
    nbp_ref_spots.score_diff = dp_gene_score - dp_gene_score_second
    nbp_ref_spots.intensity = np.median(np.max(colours_scaled, axis=2), axis=1).astype(np.float32)
    nbp_ref_spots.background_strength = bg_codes
    nbp_ref_spots.gene_probs = gene_probs_final
    nbp_ref_spots.gene_probs_initial = gene_probs_initial
    nbp_ref_spots.gene_probs_mid = gene_probs_mid

    # Save variables in nbp
    nbp.gene_names = gene_names
    nbp.gene_codes = gene_codes
    # Now expand variables to have n_channels channels instead of n_channels_use channels. For some variables, we
    # also need to swap axes as the expand channels function assumes the last axis is the channel axis.
    nbp.colour_norm_factor = utils.base.expand_channels(colour_norm_factor, use_channels, nbp_basic.n_channels)
    nbp.initial_bleed_matrix = utils.base.expand_channels(initial_bleed_matrix.T, use_channels, nbp_basic.n_channels).T
    nbp.bleed_matrix = utils.base.expand_channels(bleed_matrix[-1].T, use_channels, nbp_basic.n_channels).T
    nbp.initial_bled_codes = utils.base.expand_channels(call_spots.get_bled_codes(gene_codes=gene_codes,
                                                                                  bleed_matrix=initial_bleed_matrix),
                                                        use_channels, nbp_basic.n_channels)
    nbp.bled_codes = utils.base.expand_channels(call_spots.get_bled_codes(gene_codes=gene_codes,
                                                                          bleed_matrix=bleed_matrix[-1]),
                                                 use_channels, nbp_basic.n_channels)
    nbp.free_bled_codes = utils.base.expand_channels(free_bled_codes, use_channels, nbp_basic.n_channels)
    nbp.free_bled_codes_tile_indep = utils.base.expand_channels(free_bled_codes_tile_indep, use_channels,
                                                                nbp_basic.n_channels)
    nbp.target_matching_scale = utils.base.expand_channels(auxiliary_correction, use_channels, nbp_basic.n_channels)

    # Extract abs intensity percentile
    central_tile = extract.scale.central_tile(nbp_basic.tilepos_yx, nbp_basic.use_tiles)
    if nbp_basic.is_3d:
        mid_z = int(nbp_basic.use_z[0] + (nbp_basic.use_z[-1] - nbp_basic.use_z[0]) // 2)
    else:
        mid_z = None
    pixel_colors = spot_colors.get_spot_colors(spot_colors.all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, mid_z),
                                               central_tile, transform, nbp_file, nbp_basic, nbp_extract,
                                               return_in_bounds=True)[0]
    pixel_intensity = get_spot_intensity(np.abs(pixel_colors * scale[-1, central_tile]))
    nbp.abs_intensity_percentile = np.percentile(pixel_intensity, np.arange(1, 101))

    return nbp, nbp_ref_spots


def bayes_mean(data, prior_mean, mean_dir_conc, other_conc):
    ''' Bayesian mean estimation with specific prior concentration matrix
    prior mean vector is input prior_mean
    prior concentration along direction of prior mean is mean_dir_conc
    prior concentration along other dimensions is other_conc'''

    my_sum = data.sum(0)
    n = data.shape[0]

    prior_mean_norm = prior_mean / np.linalg.norm(prior_mean)  # normalized prior mean
    sum_proj = (my_sum @ prior_mean_norm) * prior_mean_norm  # projection of data sum along mean direction
    sum_other = my_sum - sum_proj  # projection of data sum orthogonal to mean direction

    posterior_proj = (sum_proj + prior_mean * mean_dir_conc) / (n + mean_dir_conc)
    posterior_other = (sum_other) / (n + other_conc)
    # print(my_sum)
    # print(sum_proj)
    # print(sum_other)
    return posterior_proj + posterior_other


# Below are some functions for plotting the results of the call_spots pipeline


def view_bleed_calculation(nb: Notebook):
    """
    View the bleed matrix calculation for each dye and channel
    Args:
        nb: Notebook for current experiment. Must contain the call_spots page.
    """
    colours_raw = nb.ref_spots.colors[:, :, nb.basic.use_channels].astype(float)
    spot_tile = nb.ref_spots.tile.astype(int)
    colours_scaled = colours_raw * nb.call_spots.color_norm_factor[spot_tile]
    dye_names, gene_codes, gene_no = nb.call_spots.dye_names, nb.call_spots.gene_codes, nb.ref_spots.gene_no
    n_dyes, n_rounds, n_channels_use = len(dye_names), nb.basic_info.n_rounds, len(nb.basic_info.use_channels)
    bleed_matrix = np.zeros((n_channels_use, n_dyes))

    # Begin the calculation and plotting
    fig, ax = plt.subplots(2, n_dyes + 1, figsize=(20, 10))
    for d in range(n_dyes):
        dye_d_spots = np.zeros((0, n_channels_use))
        for r in range(n_rounds):
            dye_d_round_r_genes = np.where(gene_codes[:, r] == d)[0]
            is_relevant_gene = np.isin(gene_no, dye_d_round_r_genes)
            dye_d_spots = np.concatenate((dye_d_spots, colours_scaled[is_relevant_gene, r]))
        is_positive = np.sum(dye_d_spots, axis=1) > 0
        dye_d_spots = dye_d_spots[is_positive]
        order = np.argsort(np.sum(dye_d_spots, axis=1))[::-1]
        dye_d_spots = dye_d_spots[order]
        ax[0, d].imshow(dye_d_spots, aspect='auto', interpolation='none')
        ax[0, d].set_title(dye_names[d] + ' spots')
        ax[0, d].set_ylabel('spots')
        ax[0, d].set_xlabel('channel')
        if len(dye_d_spots) == 0:
            continue
        u, s, v = svds(dye_d_spots, k=1)
        v = v[0]
        v *= np.sign(v[np.argmax(np.abs(v))])  # Make sure the largest element is positive
        bleed_matrix[iter, :, d] = v
        ax[1, d].imshow(v[None, :], aspect='auto', interpolation='none')
        ax[1, d].set_title(dye_names[d] + ' bleed. Eigenvalue = {:.3f}'.format(s))
        ax[1, d].set_xlabel('channel')
    # Now we want to plot the bleed matrix
    ax[0, -1].imshow(bleed_matrix, aspect='auto', interpolation='none')
    ax[0, -1].set_title('Bleed matrix')
    ax[0, -1].set_xlabel('dye')
    ax[0, -1].set_ylabel('channel')
    ax[0, -1].set_xticks(range(n_dyes))
    ax[0, -1].set_xticklabels(dye_names, rotation=90, fontsize=6)

    # turn off blank axes
    ax[1, -1].axis('off')
    plt.show()


def plot_cnf(nb: Notebook):
    """
    Plot the colour norm factor for each tile, round and channel
    Args:
        nb: Notebook for current experiment. Must contain the call_spots page.
    """
    n_tiles, n_rounds, use_channels = (len(nb.basic_info.use_tiles), len(nb.basic_info.use_rounds),
                                       nb.basic_info.use_channels)
    colour_norm_factor = nb.call_spots.colour_norm_factor
    # Now we are going to plot the colour norm factor for each tile, round and channel
    norm_factor_reshaped = colour_norm_factor.swapaxes(1, 2).reshape(n_tiles, -1)
    plt.figure(figsize=(10, 10))
    plt.imshow(norm_factor_reshaped.T, aspect='auto')
    # Add a white dashed line every 7 rows to show the different channels
    for i in range(1, 7):
        plt.axhline(i * n_rounds - 0.5, color='white', linestyle='--')
    plt.colorbar()
    plt.xlabel('Tile')
    plt.ylabel('Round and Channel')
    # add round channel names for every row, so row 0 will read R0C{channel[0]}, etc.
    round_channel_names = [f'R{r}C{c}' for c, r in product(use_channels, range(n_rounds))]
    plt.yticks(np.arange(len(round_channel_names)), round_channel_names)
    plt.title('Colour Norm Factor for each tile round and channel')
    plt.show()


class ViewGeneTransitions:
    def __init__(self, nb: Notebook, pre: str = 'gene_probs_initial'):
        """
        View the gene transitions between the original and new gene assignments.
        Note that the order of the gene probability assignments is gene_probs_initial -> gene_probs_mid -> gene_probs
        and the option to use anchor gene assignments allows the user to compare the anchor method and any probability
        method, but it makes the most sense to compare anchor and gene_probs.
        Args:
            nb: Notebook for current experiment. Must contain the call_spots page.
            pre: The gene assignment to use for the original gene assignment. Must be one of
                ['gene_probs_initial', 'gene_probs_mid', 'anchor']
            The post gene assignment will always be gene_probs
        """
        self.nb = nb
        self.pre = pre
        self.post = 'gene_probs'
        # First, add the data to the object (this will be the gene_no_original, gene_no_new and transition_matrix)
        self.update_data()
        # Now plot the data
        self.plot_data()

    def check_args(self):
        """
        Check that the arguments are valid
        """
        assert self.pre in ['gene_probs_initial', 'gene_probs_mid', 'gene_probs', 'anchor'], \
            f'pre must be one of ["gene_probs_initial", "gene_probs_mid", "gene_probs", "anchor"], not {self.pre}'

    def update_data(self):
        """
        Update the parameters: gene_no_original, gene_no_new and transition_matrix
        """
        self.check_args()
        if 'prob' in self.pre:
            gene_probs_original = self.nb.ref_spots.__getattribute__(self.pre)
            self.gene_no_original = np.argmax(gene_probs_original, axis=1)
            self.score_original = np.max(gene_probs_original, axis=1)
        else:
            self.gene_no_original = self.nb.ref_spots.gene_no
            self.score_original = self.nb.ref_spots.score
        # update the new gene assignments
        gene_probs_new = self.nb.ref_spots.gene_probs
        self.gene_no_new = np.argmax(gene_probs_new, axis=1)
        self.score_new = np.max(gene_probs_new, axis=1)
        # first record the gene transitions
        n_genes = len(self.nb.call_spots.gene_names)
        transition_matrix = np.zeros((n_genes, n_genes))
        for g1, g2 in product(range(n_genes), range(n_genes)):
            if g1 != g2:
                transition_matrix[g1, g2] = np.sum((self.gene_no_original == g1) * (self.gene_no_new == g2))
            else:
                transition_matrix[g1, g2] = np.nan
        self.transition_matrix = transition_matrix

    def plot_data(self):
        """
        Plot the gene transitions and the number of reads in the original and new gene assignments
        Args:
            self: the ViewGeneTransitions object
        """
        if not hasattr(self, 'fig'):
            # make sure the axes and their labels do not go below y = 0.2
            self.fig, self.ax = plt.subplots(1, 2, figsize=(20, 10))
        else:
            self.ax[0].clear()
            self.ax[1].clear()

        ax = self.ax
        n_genes = len(self.nb.call_spots.gene_names)
        gene_names = self.nb.call_spots.gene_names
        # now plot the transition matrix
        ax[0].imshow(self.transition_matrix, aspect='auto', interpolation='none')
        ax[0].set_xticks(range(n_genes))
        ax[0].set_yticks(range(n_genes))
        # make each tick label a gene name and its index
        ax[0].set_xticklabels([f'{i}: {gene_names[i]}' for i in range(n_genes)], rotation=90, fontsize=6)
        ax[0].set_yticklabels([f'{i}: {gene_names[i]}' for i in range(n_genes)], fontsize=6)
        ax[0].set_xlabel('New gene')
        ax[0].set_ylabel('Original gene')
        ax[0].set_title('Gene transitions')
        # now plot the number of reads initially versus finally
        num_reads_original = np.zeros(n_genes)
        num_reads_new = np.zeros(n_genes)
        for g in range(n_genes):
            num_reads_original[g] = np.sum(self.gene_no_original == g)
            num_reads_new[g] = np.sum(self.gene_no_new == g)
        ax[1].plot(num_reads_original, label=self.pre)
        ax[1].plot(num_reads_new, label=self.post)
        ax[1].set_xticks(range(n_genes))
        ax[1].set_xticklabels(gene_names, rotation=90, fontsize=6)
        ax[1].set_xlabel('Gene')
        ax[1].set_ylabel('Number of reads')
        ax[1].legend()
        ax[1].set_title('Assignment of reads to genes')
        # Add check buttons to change the gene assignments
        self.add_buttons()
        ratio_change = np.sum(self.gene_no_original != self.gene_no_new) / len(self.gene_no_original)
        plt.suptitle(f'Gene transitions from {self.pre} to {self.post}. The percentage of reads that changed gene '
                     f'assignment is {ratio_change * 100:.2f}%')
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.show()

    def add_buttons(self):
        """
        Make each pixel clickable. Clicking on a pixel will show the gene transitions for that gene pair.
        """
        # Add check buttons to change the pre gene assignment
        if hasattr(self, 'button_ax'):
            self.button_ax.clear()
        else:
            self.button_ax = plt.axes([0.55, 0.8125, 0.1, 0.1])
        self.check = CheckButtons(self.button_ax, ['gene_probs_initial', 'gene_probs_mid', 'anchor'],
                                  [self.pre == 'gene_probs_initial', self.pre == 'gene_probs_mid',
                                   self.pre == 'anchor'], frame_props={'color': ['white', 'white', 'white']})
        self.check.on_clicked(self.change_pre)
        # make the transition matrix clickable. Add a white rectangle around any pixel that we hover over
        mplcursors.cursor(self.ax[0], hover=True).connect('add', self.add_rectangle)
        # Now let's make it so that clicking on a pixel will show the gene transitions for that gene pair
        mplcursors.cursor(self.ax[0]).connect('add', self.view_specific_transitions)

    def change_pre(self, label):
        """
        Change the pre gene assignment
        Args:
            label: the label of the check button that was clicked
        """
        if label == 'gene_probs_initial':
            self.pre = 'gene_probs_initial'
        elif label == 'gene_probs_mid':
            self.pre = 'gene_probs_mid'
        elif label == 'anchor':
            self.pre = 'anchor'
        self.update_data()
        self.plot_data()

    def add_rectangle(self, selection):
        """
        Add a white rectangle around any pixel that we hover over
        Args:
            event: the event that triggered this function
        """
        # first remove any existing rectangles
        for rectangle in self.ax[0].patches:
            rectangle.remove()
        # now add the new rectangle
        x, y = np.rint(selection.target).astype(int)
        max_index = len(self.nb.call_spots.gene_names)
        if 0 <= x < max_index and 0 <= y < max_index:
            self.ax[0].add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor='white',
                                           linewidth=2))

    def view_specific_transitions(self, selection):
        """
        View the gene transitions between the original and new gene assignments for a specific gene pair
        Args:
            event: the event that triggered this function
        """
        x, y = np.rint(selection.target).astype(int)
        max_index = len(self.nb.call_spots.gene_names)
        if 0 <= x < max_index and 0 <= y < max_index and x != y:
            g1, g2 = y, x
            num_mismatch = np.sum((self.gene_no_original == g1) * (self.gene_no_new == g2))
            if num_mismatch == 0:
                print(f'No reads changed from {self.pre} gene {g1} to {self.post} gene {g2}')
                return
        else:
            print('Invalid Choice')
            return
        print('Printing gene transitions from', self.pre, 'gene', g1, 'to', self.post, 'gene', g2)
        colours = self.nb.ref_spots.colours[:, :, self.nb.basic_info.use_channels].astype(float)
        colour_nom_factor = self.nb.call_spots.colour_norm_factor[:, :, self.nb.basic_info.use_channels]
        spot_tile = self.nb.ref_spots.tile.astype(int)
        colours_scaled = colours * colour_nom_factor[spot_tile]
        n_spots, n_rounds, n_channels_use = colours_scaled.shape
        gene_names = self.nb.call_spots.gene_names
        use_channels = self.nb.basic_info.use_channels
        print('Gene 1 is ' + gene_names[g1] + ' and gene 2 is ' + gene_names[g2])
        if not hasattr(self, 'fig2'):
            self.fig2, self.ax2 = plt.subplots(3, 2, figsize=(20, 10))
        else:
            del self.fig2, self.ax2
            self.fig2, self.ax2 = plt.subplots(3, 2, figsize=(20, 10))
        fig, ax = self.fig2, self.ax2
        transition_spots = np.where((self.gene_no_original == g1) * (self.gene_no_new == g2))[0]
        new_score = self.score_new[transition_spots] * len(transition_spots)
        order = np.argsort(new_score)

        if self.pre == 'gene_probs_mid':
            transition_colours = colours_scaled[transition_spots][order].reshape((len(transition_spots),
                                                                           n_rounds * n_channels_use))
            original_gene_code = self.nb.call_spots.bled_codes[g1][:, use_channels].ravel()[None, :]
            new_gene_code = self.nb.call_spots.bled_codes[g2][:, use_channels].ravel()[None, :]
        else:
            transition_colours = colours[transition_spots][order].reshape((len(transition_spots),
                                                                           n_rounds * n_channels_use))
            original_gene_code = self.nb.call_spots.initial_bled_codes[g1][:, use_channels].ravel()[None, :]
            new_gene_code = self.nb.call_spots.initial_bled_codes[g2][:, use_channels].ravel()[None, :]
        round_dp_score_original = np.zeros((len(transition_spots), n_rounds))
        round_dp_score_new = np.zeros((len(transition_spots), n_rounds))
        for r in range(n_rounds):
            colours_r = transition_colours[:, r*n_channels_use:(r+1)*n_channels_use]
            colours_r /= np.linalg.norm(colours_r, axis=1)[:, None]
            original_gene_code_r = original_gene_code[:, r*n_channels_use:(r+1)*n_channels_use]
            original_gene_code_r /= np.linalg.norm(original_gene_code_r, axis=1)[:, None]
            new_gene_code_r = new_gene_code[:, r*n_channels_use:(r+1)*n_channels_use]
            new_gene_code_r /= np.linalg.norm(new_gene_code_r, axis=1)[:, None]
            round_dp_score_original[:, r] = np.sum(colours_r * original_gene_code_r, axis=1)
            round_dp_score_new[:, r] = np.sum(colours_r * new_gene_code_r, axis=1)

        mean_dp_original = np.mean(round_dp_score_original, axis=0)
        mean_dp_new = np.mean(round_dp_score_new, axis=0)
        original_rounds_win = mean_dp_original > mean_dp_new
        new_rounds_win = mean_dp_new > mean_dp_original
        print('Original rounds win:', original_rounds_win)

        print('Number of transitions:', len(transition_spots))

        # add vertical lines every n_channels_use to separate rounds
        ax[0, 0].imshow(original_gene_code, aspect='auto', interpolation='none')
        for i in range(1, n_rounds):
            ax[0, 0].axvline(i * n_channels_use - 0.5, color='white', linestyle='--')
            # add a green box around the winning round
        for i in range(n_rounds):
            if original_rounds_win[i]:
                ax[0, 0].add_patch(Rectangle((i * n_channels_use - 0.5, -0.5), n_channels_use, 1, fill=False,
                                             edgecolor='red', linewidth=6))
        if self.pre == 'gene_probs_mid':
            ax[0, 0].set_title('Bled Code for ' + gene_names[g1])
        else:
            ax[0, 0].set_title('Unscaled ' + gene_names[g1])
        ax[0, 0].set_xticks(np.arange(n_channels_use // 2, n_channels_use * n_rounds, n_channels_use))
        ax[0, 0].set_xticklabels(range(n_rounds))
        ax[0, 0].set_yticks([])

        ax[1, 0].imshow(transition_colours, aspect='auto', interpolation='none', vmin=0,
                        vmax=np.percentile(transition_colours, 99))
        for i in range(1, n_rounds):
            ax[1, 0].axvline(i * n_channels_use - 0.5, color='white', linestyle='--')
        ax[1, 0].set_xticks(np.arange(n_channels_use // 2, n_channels_use * n_rounds, n_channels_use))
        ax[1, 0].set_xticklabels(range(n_rounds))
        ax[1, 0].set_yticks([])

        ax[2, 0].imshow(new_gene_code, aspect='auto', interpolation='none')
        if self.pre == 'gene_probs_mid':
            ax[2, 0].set_title('Bled Code for ' + gene_names[g2])
        else:
            ax[2, 0].set_title('Unscaled ' + gene_names[g2])
        for i in range(1, n_rounds):
            ax[2, 0].axvline(i * n_channels_use - 0.5, color='white', linestyle='--')
        for i in range(n_rounds):
            if new_rounds_win[i]:
                ax[2, 0].add_patch(Rectangle((i * n_channels_use - 0.5, -0.5), n_channels_use, 1, fill=False,
                                             edgecolor='red', linewidth=6))

        ax[2, 0].set_xticks(np.arange(n_channels_use // 2, n_channels_use * n_rounds, n_channels_use))
        ax[2, 0].set_xticklabels(range(n_rounds))
        ax[2, 0].set_yticks([])

        # Now we will plot the scaled colours
        transition_colours = colours_scaled[transition_spots][order].reshape((len(transition_spots),
                                                                              n_rounds * n_channels_use)).astype(float)
        original_gene_code = self.nb.call_spots.free_bled_codes_tile_indep[g1][:, use_channels].ravel()[None, :]
        new_gene_code = self.nb.call_spots.free_bled_codes_tile_indep[g2][:, use_channels].ravel()[None, :]
        round_dp_score_original = np.zeros((len(transition_spots), n_rounds))
        round_dp_score_new = np.zeros((len(transition_spots), n_rounds))
        for r in range(n_rounds):
            colours_r = transition_colours[:, r * n_channels_use:(r + 1) * n_channels_use]
            colours_r /= np.linalg.norm(colours_r, axis=1)[:, None]
            original_gene_code_r = original_gene_code[:, r * n_channels_use:(r + 1) * n_channels_use]
            original_gene_code_r /= np.linalg.norm(original_gene_code_r, axis=1)[:, None]
            new_gene_code_r = new_gene_code[:, r * n_channels_use:(r + 1) * n_channels_use]
            new_gene_code_r /= np.linalg.norm(new_gene_code_r, axis=1)[:, None]
            round_dp_score_original[:, r] = np.sum(colours_r * original_gene_code_r, axis=1)
            round_dp_score_new[:, r] = np.sum(colours_r * new_gene_code_r, axis=1)
        mean_dp_original = np.mean(round_dp_score_original, axis=0)
        mean_dp_new = np.mean(round_dp_score_new, axis=0)
        original_rounds_win = mean_dp_original > mean_dp_new
        new_rounds_win = mean_dp_new > mean_dp_original
        print('Original rounds win:', original_rounds_win)

        # add vertical lines every n_channels_use to separate rounds
        ax[0, 1].imshow(original_gene_code, aspect='auto', interpolation='none')
        for i in range(1, n_rounds):
            ax[0, 1].axvline(i * n_channels_use - 0.5, color='white', linestyle='--')
        for i in range(n_rounds):
            if original_rounds_win[i]:
                ax[0, 1].add_patch(Rectangle((i * n_channels_use - 0.5, -0.5), n_channels_use, 1, fill=False,
                                             edgecolor='red', linewidth=6))
        if self.pre == 'gene_probs_mid':
            ax[0, 1].set_title('Free Bled Code for ' + gene_names[g1])
        else:
            ax[0, 1].set_title('Scaled ' + gene_names[g1])
        ax[0, 1].set_xticks(np.arange(n_channels_use // 2, n_channels_use * n_rounds, n_channels_use))
        ax[0, 1].set_xticklabels(range(n_rounds))
        ax[0, 1].set_yticks([])

        ax[1, 1].imshow(transition_colours, aspect='auto', interpolation='none', vmin=0,
                        vmax=np.percentile(transition_colours, 99))
        for i in range(1, n_rounds):
            ax[1, 1].axvline(i * n_channels_use - 0.5, color='white', linestyle='--')
        ax[1, 1].set_xticks(np.arange(n_channels_use // 2, n_channels_use * n_rounds, n_channels_use))
        ax[1, 1].set_xticklabels(range(n_rounds))
        ax[1, 1].set_yticks([])

        ax[2, 1].imshow(new_gene_code, aspect='auto', interpolation='none')
        if self.pre == 'gene_probs_mid':
            ax[2, 1].set_title('Free Bled Code for ' + gene_names[g2])
        else:
            ax[2, 1].set_title('Scaled ' + gene_names[g2])
        for i in range(1, n_rounds):
            ax[2, 1].axvline(i * n_channels_use - 0.5, color='white', linestyle='--')
        for i in range(n_rounds):
            if new_rounds_win[i]:
                ax[2, 1].add_patch(Rectangle((i * n_channels_use - 0.5, -0.5), n_channels_use, 1, fill=False,
                                                edgecolor='red', linewidth=6))
        ax[2, 1].set_xticks(np.arange(n_channels_use // 2, n_channels_use * n_rounds, n_channels_use))
        ax[2, 1].set_xticklabels(range(n_rounds))
        ax[2, 1].set_yticks([])

        plt.suptitle('Gene transitions from gene ' + gene_names[g1] + ' to gene ' + gene_names[g2])
        fig.canvas.draw_idle()
        plt.show()


def view_all_genes(free_bled_codes_tile_indep, gene_names, scale_correction):
    """
    View all genes in a grid of plots, with the tile independent free bled codes in the even columns and the odd columns
    showing the scaled tile independent free bled codes. The scale here is the target matching scale which scales the
    tile independent free bled codes to match the target channel strength as well as possible.
    Args:
        free_bled_codes_tile_indep: n_genes x n_rounds x n_channels array of tile independent free bled codes
        gene_names: n_genes array of gene names
        scale_correction: n_rounds x n_channels array of scale corrections
    """
    n_genes, n_rounds, n_channels = free_bled_codes_tile_indep.shape
    height = int(np.ceil(np.sqrt(n_genes)))
    width = 2 * height
    fig, ax = plt.subplots(height, width, figsize=(20, 10))
    scaled_free_bled_codes_tile_indep = free_bled_codes_tile_indep * scale_correction[None, :, :]
    existing = np.max(free_bled_codes_tile_indep, axis=(1, 2)) > 0
    scaled_free_bled_codes_tile_indep[existing] /= np.linalg.norm(scaled_free_bled_codes_tile_indep[existing],
                                                                  axis=(1, 2))[:, None, None]
    # we will plot the raw codes in the even columns and the scaled codes in the odd columns
    for g in range(n_genes):
        row = g // height
        col = (g % height) * 2
        ax[row, col].imshow(free_bled_codes_tile_indep[g].T, aspect='auto', interpolation='none', vmin=0,
                            vmax=max(np.max(free_bled_codes_tile_indep[g]),
                                     np.max(free_bled_codes_tile_indep[g] * scale_correction)))
        if row == height - 1:
            ax[row, col].set_xlabel('Round')
        if col == 0:
            ax[row, col].set_ylabel('Channel')
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        ax[row, col].set_title(gene_names[g] + ' raw', fontsize=6)

    for g in range(n_genes):
        row = g // height
        col = 2 * (g % height) + 1
        ax[row, col].imshow(scaled_free_bled_codes_tile_indep[g].T, aspect='auto', interpolation='none', vmin=0,
                            vmax=max(np.max(free_bled_codes_tile_indep[g]),
                                     np.max(free_bled_codes_tile_indep[g] * scale_correction)))
        if row == height - 1:
            ax[row, col].set_xlabel('Round')
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        ax[row, col].set_title(gene_names[g] + ' scaled', fontsize=6)

    # loop through the rest of the axes and turn them off
    for g in range(n_genes, height ** 2):
        row = g // height
        col = 2 * (g % height)
        ax[row, col].axis('off')
        ax[row, col + 1].axis('off')

    # for final row and column, plot the scale correction
    row = height - 1
    col = width - 1
    ax[row, col].imshow(scale_correction.T, aspect='auto', interpolation='none')
    ax[row, col].set_xlabel('Round')
    ax[row, col].set_ylabel('Channel')
    ax[row, col].set_xticks([])
    ax[row, col].set_yticks([])
    ax[row, col].set_title('Scale correction', fontsize=6)

    fig.suptitle('Tile independent free bled codes for all genes, scaled and unscaled')
    plt.show()


def view_all_gene_round_strengths(free_bled_codes_tile_indep, gene_names, scale_correction: np.ndarray):
    """
    View the round strengths for all genes in a grid of plots.
    Args:
        free_bled_codes_tile_indep: n_genes x n_rounds x n_channels array of tile independent free bled codes
        gene_names: n_genes array of gene names
        scale_correction: n_rounds x n_channels total auxiliary scale correction
        save: bool, optional. Whether to save the figure or not
    """
    n_genes, n_rounds, n_channels = free_bled_codes_tile_indep.shape
    square_side = int(np.ceil(np.sqrt(n_genes)))
    fig, ax = plt.subplots(square_side, square_side, figsize=(20, 10))
    scaled_free_bled_codes_tile_indep = free_bled_codes_tile_indep * scale_correction[None, :, :]
    existing = np.max(free_bled_codes_tile_indep, axis=(1, 2)) > 0
    scaled_free_bled_codes_tile_indep[existing] /= np.linalg.norm(scaled_free_bled_codes_tile_indep[existing],
                                                                  axis=(1, 2))[:, None, None]
    free_bled_round_strength = np.sum(free_bled_codes_tile_indep, axis=2)
    free_bled_round_strength_scaled = np.sum(scaled_free_bled_codes_tile_indep, axis=2)
    max_strength = max(np.max(free_bled_round_strength), np.max(free_bled_round_strength_scaled))
    for g in range(n_genes):
        row = g // square_side
        col = g % square_side
        ax[row, col].plot(free_bled_round_strength[g])
        ax[row, col].plot(free_bled_round_strength_scaled[g])
        if row == square_side - 1:
            ax[row, col].set_xlabel('Round')
        if col == 0:
            ax[row, col].set_ylabel('Intensity')
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        ax[row, col].set_ylim(0, max_strength)
        ax[row, col].set_title(gene_names[g])
    for g in range(n_genes, square_side ** 2):
        row = g // square_side
        col = g % square_side
        ax[row, col].axis('off')
    cv = np.mean(np.std(free_bled_round_strength, axis=1)) / np.mean(free_bled_round_strength)
    scaled_cv= np.mean(np.std(free_bled_round_strength_scaled, axis=1)) / np.mean(free_bled_round_strength_scaled)
    fig.suptitle('Tile independent free bled codes for all genes. Blue is unscaled, orange is scaled \n'
                 'Raw Coef. of Variation = {:.3f}, Scaled Coef. of Variation = {:.3f}'.format(cv, scaled_cv))
    plt.show()


# Below are some functions for plotting the iterative process of call spots scaling, pertaining to the bleed matrix,
# colour norm factor and free bled codes and gene scores
def plot_scale_iters(colour_norm_factor: np.ndarray, tile: int = None):
    """
    Plot the colour norm factor for each round and channel for tile t as a function of iterations.
    If t is None, then plot the average over all tiles.
    Args:
        colour_norm_factor: (n_iter, n_tiles, n_rounds, n_channels) colour norm factor
        tile: int, optional. The tile to plot
    """
    n_iter, n_tiles, n_rounds, n_channels = colour_norm_factor.shape
    fig, ax = plt.subplots(n_rounds, n_channels, figsize=(20, 10))
    y_max = np.max(colour_norm_factor)
    for r, c in product(range(n_rounds), range(n_channels)):
        if tile is None:
            ax[r, c].plot(np.arange(n_iter), colour_norm_factor[:, :, r, c].mean(axis=1))
        else:
            ax[r, c].plot(np.arange(n_iter), colour_norm_factor[:, tile, r, c])
        # Add dotted line at 1 in red
        ax[r, c].axhline(1, linestyle='--', color='red')
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_ylim(0, y_max)
        if r == 0:
            ax[r, c].set_title('Channel {}'.format(c))
        if c == 0:
            ax[r, c].set_ylabel('Round {}'.format(r))

    plt.suptitle('Colour norm factor for each round and channel')
    plt.show()


def plot_bleed_iters(bleed_matrix: np.ndarray):
    """
    Plot the bleed matrix for each channel and dye for tile t as a function of iterations.
    If t is None, then plot the average over all tiles.
    Args:
        bleed_matrix: (n_iter, n_channels, n_dyes) bleed matrix
    """
    n_iter, n_channels, n_dyes = bleed_matrix.shape
    fig, ax = plt.subplots(2, n_iter, figsize=(20, 10))
    vmin, vmax = np.min(bleed_matrix), np.max(bleed_matrix)
    bleed_diff = np.diff(bleed_matrix, axis=0)
    vmin_diff, vmax_diff = np.min(bleed_diff), np.max(bleed_diff)
    for i in range(n_iter):
        ax[0, i].imshow(bleed_matrix[i], aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        if i > 0:
            ax[1, i].imshow(bleed_diff[i-1], aspect='auto', interpolation='none', vmin=vmin_diff, vmax=vmax_diff)
            ax[1, i].set_xticks([])
            ax[1, i].set_yticks([])
        else:
            # turn axis off for first iteration
            ax[1, i].axis('off')
        ax[1, i].set_xlabel('Dye')
        ax[1, i].set_title('Diff')
        ax[0, i].set_title('Bleed')
    ax[0, 0].set_ylabel('channel')
    ax[1, 0].set_ylabel('Channel')
    plt.suptitle('Bleed matrix for each iteration')
    # add a colorbar for the first row
    cax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    fig.colorbar(ax[0, 0].imshow(bleed_matrix[0], aspect='auto', interpolation='none', vmin=vmin, vmax=vmax), cax=cax)
    # add a colorbar for the second row
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    fig.colorbar(ax[1, 1].imshow(bleed_diff[0], aspect='auto', interpolation='none', vmin=vmin_diff, vmax=vmax_diff),
                 cax=cax)
    plt.show()


def plot_aux_scale_iters(aux_scale: np.ndarray):
    """
    Plot the auxiliary scale for each round and channel as a function of iterations.
    Args:
        aux_scale: np.ndarray of shape (n_iter, n_rounds, n_channels)
    """
    n_iter, n_rounds, n_channels = aux_scale.shape
    fig, ax = plt.subplots(n_rounds, n_channels, figsize=(20, 10))
    y_max = np.max(aux_scale)
    for r, c in product(range(n_rounds), range(n_channels)):
        ax[r, c].plot(np.arange(n_iter), aux_scale[:, r, c])
        # Add dotted line at 1 in red
        ax[r, c].axhline(1, linestyle='--', color='red')
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_ylim(0, y_max)
        if r == 0:
            ax[r, c].set_title('Channel {}'.format(c))
        if c == 0:
            ax[r, c].set_ylabel('Round {}'.format(r))

    plt.suptitle('Auxiliary scale for each round and channel')
    plt.show()


def plot_bg_subtraction(colours: np.ndarray, bg_colours: np.ndarray, spot_tile: np.ndarray, bg_scale: np.ndarray,
                        use_channels: np.ndarray):
    """
    Plot 2 raster plots of colours before and after subtracting background.
    Args:
        colours: n_spots x n_rounds x n_channels array of spot colours (bg removed)
        bg_colours: n_spots x n_rounds x n_channels array of background colours
        spot_tile: n_spots array of tile numbers
        bg_scale: n_tiles x n_rounds x n_channels array of background scale
        use_channels: n_channels array of channels in use
    """
    n_spots, n_rounds, n_channels = colours.shape
    colours_post = colours.copy()
    colours_pre = colours_post + bg_colours * bg_scale[spot_tile, :, :]
    rc_max = np.max(colours_pre.reshape(n_spots, -1), axis=1)
    rc_min = np.min(colours_pre.reshape(n_spots, -1), axis=1)
    inlier_mask = (np.max(colours_pre.reshape(n_spots, -1), axis=1) < np.percentile(rc_max, 99)) * \
                  (np.min(colours_pre.reshape(n_spots, -1), axis=1) > np.percentile(rc_min, 1))
    colours_pre, colours_post = colours_pre[inlier_mask], colours_post[inlier_mask]
    bg_colours = bg_colours[inlier_mask]
    # reorder colours by strongest background
    bg_strength = np.sum(np.abs(bg_colours), axis=(1, 2))
    order = np.argsort(bg_strength)[::-1]
    colours_pre, colours_post = colours_pre[order], colours_post[order]
    # Need to reshape colours_pre and colours_post to be n_spots x (n_rounds x n_channels)
    colours_pre = colours_pre.swapaxes(1, 2)
    colours_post = colours_post.swapaxes(1, 2)
    colours_pre = colours_pre.reshape((colours_pre.shape[0], -1))
    colours_post = colours_post.reshape((colours_post.shape[0], -1))
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # Plot colours_pre
    ax[0].imshow(colours_pre, aspect='auto', interpolation='none', vmin=np.percentile(colours_pre, 1),
                 vmax=np.percentile(colours_pre, 99))
    # plot vertical dotted lines to separate channels
    for c in range(1, n_channels):
        ax[0].axvline(c * n_rounds - 0.5, color='white', linestyle='--')
    x_ticks = []
    for c,r in product(range(n_channels), range(n_rounds)):
        x_ticks.append('C{}R{}'.format(use_channels[c], r))
    ax[0].set_xticks(np.arange(n_channels * n_rounds))
    ax[0].set_xticklabels(x_ticks, rotation=90)
    # plot_colours_post
    ax[1].imshow(colours_post, aspect='auto', interpolation='none', vmin=np.percentile(colours_post, 1),
                 vmax=np.percentile(colours_post, 99))
    # plot vertical dotted lines to separate channels
    for c in range(1, n_channels):
        ax[1].axvline(c * n_rounds - 0.5, color='white', linestyle='--')
    ax[1].set_xticks(np.arange(n_channels * n_rounds))
    ax[1].set_xticklabels(x_ticks, rotation=90)
    # set titles
    ax[0].set_title('Before background subtraction')
    ax[1].set_title('After background subtraction')
    plt.suptitle('Raster plots of colours before and after background subtraction')
    plt.show()


def plot_initial_normalisation(colours: np.ndarray, spot_tile: np.ndarray, initial_norm_factor: np.ndarray,
                               use_channels: np.ndarray):
    """
    Plot 3 raster plots of colours before normalisation, after normalisation and after normalisation and bg removal.
    Args:
        colours: n_spots x n_rounds x n_channels array of spot colours
        spot_tile: n_spots array of tile numbers
        initial_norm_factor: n_tiles x n_rounds x n_channels array of initial normalisation factor
        use_channels: n_channels array of channels in use
    """
    n_spots, n_rounds, n_channels = colours.shape
    colours_raw = colours.copy()
    colours = colours_raw / initial_norm_factor[spot_tile, :, :]
    bg_colours = np.percentile(colours, 25, axis=1)
    bg_codes = np.repeat(bg_colours[:, None, :], n_rounds, axis=1)
    colours_no_bg = colours - bg_codes
    # reorder colours by strongest background
    bg_strength = np.sum(np.abs(bg_colours), axis=1)
    order = np.argsort(bg_strength)[::-1]
    colours_raw, colours, colours_no_bg = colours_raw[order], colours[order], colours_no_bg[order]
    vmin_raw, vmax_raw = np.percentile(colours_raw, 1), np.percentile(colours_raw, 99)
    vmin, vmax = np.percentile(colours, 1), np.percentile(colours, 99)
    vmin_no_bg, vmax_no_bg = np.percentile(colours_no_bg, 1), np.percentile(colours_no_bg, 99)
    # Need to reshape colours_raw, colours and colours_no_bg to be n_spots x (n_rounds x n_channels)
    colours_raw = colours_raw.swapaxes(1, 2)
    colours = colours.swapaxes(1, 2)
    colours_no_bg = colours_no_bg.swapaxes(1, 2)
    colours_raw = colours_raw.reshape((colours_raw.shape[0], -1))
    colours = colours.reshape((colours.shape[0], -1))
    colours_no_bg = colours_no_bg.reshape((colours_no_bg.shape[0], -1))
    # plot
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    x_ticks = []
    for c, r in product(range(n_channels), range(n_rounds)):
        x_ticks.append('C{}R{}'.format(use_channels[c], r))
    colour_list = [colours_raw, colours, colours_no_bg]
    vmin_list = [vmin_raw, vmin, vmin_no_bg]
    vmax_list = [vmax_raw, vmax, vmax_no_bg]
    title_list = ['Raw', 'Scaled', 'Scaled and bg removed']
    for i in range(3):
        ax[0, i].imshow(colour_list[i], aspect='auto', interpolation='none', vmin=vmin_list[i], vmax=vmax_list[i])
        ax[1, i].plot(np.percentile(colour_list[i], 90, axis=0), label='90', linestyle='--')
        ax[1, i].plot(np.percentile(colour_list[i], 95, axis=0), label='95')
        ax[1, i].plot(np.percentile(colour_list[i], 99, axis=0), label='99', linestyle='--')
        ax[1, i].legend(loc='upper right')
        ax[1, i].set_title(title_list[i] + ' percentiles')
        for j in range(2):
            # plot vertical dotted lines to separate channels
            for c in range(1, n_channels):
                ax[j, i].axvline(c * n_rounds - 0.5, color='white', linestyle='--')
            ax[j, i].set_xticks(np.arange(n_channels * n_rounds))
            ax[j, i].set_xticklabels(x_ticks, rotation=90, fontsize=4)
            ax[j, i].set_yticks([])
            ax[j, i].set_title(title_list[i])

    plt.suptitle('Raster plots of colours before and after initial normalisation')
    plt.show()


def plot_final_normalisation(colours: np.ndarray, spot_tile: np.ndarray, gene_prob: np.ndarray,
                             final_norm_factor: np.ndarray, use_channels: np.ndarray, gene_names: np.ndarray,
                             gene: int = None, prob_thresh: float = 0.9, save: bool = False):
    """
    Plot raster plots of colours before and after final normalisation for a given gene. If gene is None, plot all
    colours.
    Args:
        colours: n_spots x n_rounds x n_channels array of spot colours
        spot_tile: n_spots array of tile numbers
        gene_prob: n_spots x n_genes array of gene probabilities
        final_norm_factor: n_tiles x n_rounds x n_channels array of final normalisation factor. This is the pointwise
        division colour_norm_factor / initial_norm_factor
        use_channels: n_channels array of channels in use
        gene_names: n_genes array of gene names
        gene: int, optional. The gene to plot
        prob_thresh: float, optional. The probability threshold to use for assigning genes
        save: bool, optional. Whether to save the figure
    """
    n_spots, n_rounds, n_channels = colours.shape
    gene_no = np.argmax(gene_prob, axis=1)
    gene_score = np.max(gene_prob, axis=1)
    if gene is not None:
        keep = (gene_score > prob_thresh) * (gene_no == gene)
        colours = colours[keep]
        spot_tile = spot_tile[keep]
        order = np.argsort(np.sum(colours, axis=(1,2)))[::-1]
    else:
        # reorder colours by strongest gene no, then strongest gene score
        order = np.lexsort((gene_score, gene_no))[::-1]
    colours = colours[order]
    spot_tile = spot_tile[order]
    colours_norm = colours * final_norm_factor[spot_tile, :, :]
    vmin, vmax = np.percentile(colours, 1), np.percentile(colours, 99)
    vmin_norm, vmax_norm = np.percentile(colours_norm, 1), np.percentile(colours_norm, 99)
    # Need to reshape colours and colours_norm to be n_spots x (n_rounds x n_channels)
    colours = colours.reshape((colours.shape[0], -1))
    colours_norm = colours_norm.reshape((colours_norm.shape[0], -1))
    # plot
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    x_ticks = []
    for r, c in product(range(n_rounds), range(n_channels)):
        x_ticks.append('R{}C{}'.format(r, use_channels[c]))
    colour_list = [colours, colours_norm]
    vmin_list = [vmin, vmin_norm]
    vmax_list = [vmax, vmax_norm]
    title_list = ['Raw', 'Scaled']
    for i in range(2):
        ax[0, i].imshow(colour_list[i], aspect='auto', interpolation='none', vmin=vmin_list[i], vmax=vmax_list[i])
        ax[0, i].set_title(title_list[i])
    for i in range(2):
        ax[1, i].plot(np.mean(colour_list[i], axis=0), label='mean')
        ax[1, i].set_title(title_list[i] + ' mean')
    for i, j in product(range(2), range(2)):
        # plot vertical dotted lines to separate channels
        for c in range(1, n_channels):
            ax[i, j].axvline(c * n_rounds - 0.5, color='white', linestyle='--')
        ax[i, j].set_xticks(np.arange(n_channels * n_rounds))
        ax[i, j].set_xticklabels(x_ticks, rotation=90, fontsize=5)
    if gene is not None:
        plt.suptitle('Raster plots of colours before and after final normalisation for gene {}'.format(gene_names[gene]))
    else:
        plt.suptitle('Raster plots of colours before and after final normalisation')
    if save:
        plt.savefig('/home/reilly/Desktop/final_normalisation.png')
        plt.close(fig)
    else:
        plt.show()