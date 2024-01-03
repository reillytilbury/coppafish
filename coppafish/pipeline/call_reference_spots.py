import numpy as np
from scipy.sparse.linalg import svds
import warnings
try:
    import importlib_resources
except ModuleNotFoundError:
    import importlib.resources as importlib_resources
from typing import Tuple

from ..setup.notebook import NotebookPage
from .. import call_spots
from .. import spot_colors
from .. import utils
from coppafish import extract
from ..call_spots import get_spot_intensity
from tqdm import tqdm
from itertools import product
# import matplotlib
import matplotlib.pyplot as plt


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
    for var in ['gene_no', 'score', 'score_diff', 'intensity', 'background_strength',  'gene_probs', 'dye_strengths']:
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
    colours = nbp_ref_spots.colors[:, :, nbp_basic.use_channels].astype(float)
    bg_colours = nbp_ref_spots.bg_colours.astype(float)
    spot_tile = nbp_ref_spots.tile.astype(int)
    n_spots, n_dyes, use_channels = colours.shape[0], len(nbp_basic.dye_names), nbp_basic.use_channels
    target_channel_strength = [1, 1, 0.9, 0.8, 0.8, 1, 1]
    use_tiles = nbp_basic.use_tiles

    # initialise bleed matrices
    initial_bleed_matrix = initial_bleed_matrix / np.linalg.norm(initial_bleed_matrix, axis=0)
    bled_codes = call_spots.get_bled_codes(gene_codes=gene_codes, bleed_matrix=initial_bleed_matrix,
                                           gene_efficiency=np.ones((n_genes, n_rounds)))

    # Initialise the 2 variables we are most interested in estimating: colour_norm_factor and gene_efficiency
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
        colour_norm_factor_initial[t] = np.percentile(abs(tile_colours), 95, axis=0)
        colours[spot_tile == t] /= colour_norm_factor_initial[t]
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
    n_iter = 10
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
            colours_scaled = colours * scale[iter - 1, spot_tile]

        # 2. Gene assignments using FVM
        gene_prob = call_spots.gene_prob_score(spot_colours=colours_scaled, bled_codes=bled_codes)
        gene_no_temp = np.argmax(gene_prob, axis=1)
        gene_prob_score_temp = np.max(gene_prob, axis=1)
        # Save gene_no and gene_prob_score, so we can see how much they change between iterations
        gene_no[iter] = gene_no_temp
        gene_prob_score[iter] = gene_prob_score_temp

        if iter > 0:
            frac_mismatch = np.sum(gene_no[iter] != gene_no[iter - 1]) / n_spots
            print('Fraction of spots assigned to different genes: {:.3f}'.format(frac_mismatch))

        # 3. Update bleed matrix
        bleed_matrix_prob_thresh = 0.9
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
        bled_codes = call_spots.get_bled_codes(gene_codes=gene_codes, bleed_matrix=bleed_matrix[iter],
                                               gene_efficiency=np.ones((n_genes, n_rounds)))

        # We need to have a way of changing between dyes and channels, so we define d_max and d_2_max
        d_max = np.zeros(n_channels_use, dtype=int)
        d_2nd_max = np.zeros(n_channels_use, dtype=int)
        for c in range(n_channels_use):
            d_max[c] = np.argmax(bleed_matrix[iter, c])
            d_2nd_max[c] = np.argsort(bleed_matrix[iter, c])[-2]

        # 4. Estimate free_bled_codes
        conc_param_round = .1  # it takes this many spots to scale the bled code for any round
        conc_param_other = 40  # it takes this many spots to change the bled code in a round
        free_bled_prob_thresh = 0.9
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

        free_bled_codes = np.repeat(free_bled_codes_tile_indep[None, :, :, :], n_tiles, axis=0)
        for t, g in product(use_tiles, range(n_genes)):
            keep = (spot_tile == t) * (gene_prob_score[iter] > free_bled_prob_thresh) * (gene_no[iter] == g) * \
                   inlier_mask
            colours_tg = colours_scaled[keep]
            if np.sum(keep) <= 1:
                continue
            for r in range(n_rounds):
                dye_colour = bleed_matrix[iter, :, gene_codes[g, r]] / np.sqrt(n_rounds)
                free_bled_codes[t, g, r] = bayes_mean(data=colours_tg[:, r], prior_mean=dye_colour,
                                                      mean_dir_conc=conc_param_round, other_conc=conc_param_other)
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
            n = np.sum(n_reads[:, relevant_genes], axis=0)
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
            n = n_reads[t, relevant_genes_dye_c]
            relevant_genes_dye_2nd_c = np.where(gene_codes[:, r] == dye_2nd_c)[0]
            F = np.concatenate((F, free_bled_codes_tile_indep[relevant_genes_dye_2nd_c, r, c] * v_rc))
            G = np.concatenate((G, free_bled_codes[t, relevant_genes_dye_2nd_c, r, c]))

            n = np.concatenate((n, dye_2_strength * n_reads[t, relevant_genes_dye_2nd_c]))
            if np.sum(n) == 0:
                continue
            A = np.sum(n * F * G) / np.sum(n * G * G)
            scale[iter, t, r, c] = A

    # Calculate bled codes with this gene efficiency
    # dp_gene_no, dp_gene_score, dp_gene_score_second \
    #     = call_spots.dot_product_score(spot_colours=colours.reshape((n_spots, -1)),
    #                                    bled_codes=bled_codes.reshape((n_genes, -1)))[:3]

    dp_gene_no = np.zeros(n_spots, dtype=int)
    dp_gene_score = np.zeros(n_spots)
    dp_gene_score_second = np.zeros(n_spots)
    for t in use_tiles:
        t_mask = spot_tile == t
        n_spots_t = np.sum(t_mask)
        dp_gene_no[t_mask], dp_gene_score[t_mask], dp_gene_score_second[t_mask] \
            = call_spots.dot_product_score(spot_colours=colours_scaled[t_mask].reshape((n_spots_t, -1)),
                                           bled_codes=free_bled_codes[t].reshape((n_genes, -1)))[:3]

    # save overwritable variables in nbp_ref_spots
    nbp_ref_spots.gene_no = dp_gene_no
    nbp_ref_spots.score = dp_gene_score
    nbp_ref_spots.score_diff = dp_gene_score - dp_gene_score_second
    nbp_ref_spots.intensity = np.median(np.max(colours, axis=2), axis=1).astype(np.float32)
    nbp_ref_spots.background_strength = bg_codes
    nbp_ref_spots.gene_probs = gene_prob
    nbp_ref_spots.finalized = True

    # Save variables in nbp
    nbp.gene_names = gene_names
    nbp.gene_codes = gene_codes
    # Now expand variables to have n_channels channels instead of n_channels_use channels. For some variables, we
    # also need to swap axes as the expand channels function assumes the last axis is the channel axis.
    nbp.color_norm_factor = utils.base.expand_channels(scale, use_channels, nbp_basic.n_channels)
    nbp.initial_bleed_matrix = utils.base.expand_channels(initial_bleed_matrix.T, use_channels, nbp_basic.n_channels).T
    nbp.bleed_matrix = utils.base.expand_channels(bleed_matrix[-1].T, use_channels, nbp_basic.n_channels).T
    nbp.bled_codes_ge = utils.base.expand_channels(bled_codes, use_channels, nbp_basic.n_channels)
    nbp.bled_codes = utils.base.expand_channels(bled_codes, use_channels, nbp_basic.n_channels)
    nbp.gene_efficiency = np.ones((n_genes, n_rounds))

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
    nbp.use_ge = False

    return nbp, nbp_ref_spots


def plot_svd(gene_name, tile, colours_tg, u, v):
    order = np.argsort(np.sum(colours_tg, axis=1))[::-1]
    mean = np.mean(colours_tg, axis=0)
    mean = mean / np.linalg.norm(mean)

    fig = plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(colours_tg[order], aspect='auto', interpolation='none')
    plt.title('Raw colours')
    plt.xticks([])
    plt.ylabel('spots')
    plt.colorbar()
    plt.subplot(2, 3, 2)
    plt.imshow((u * v[None, :])[order], aspect='auto', interpolation='none')
    plt.title('SVD Fit')
    plt.xticks([])
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.imshow((u * mean[None, :])[order], aspect='auto', interpolation='none')
    plt.xticks([])
    plt.title('Mean Fit')
    plt.colorbar()
    plt.subplot(2, 3, 4)
    plt.xticks([])
    plt.xlabel('Round channel')
    plt.ylabel('spots')
    plt.imshow((u * mean[None, :] - u * v[None, :])[order], aspect='auto', interpolation='none')
    plt.title('svd - mean')
    plt.colorbar()
    plt.subplot(2, 3, 5)
    plt.plot(mean, label='mean')
    plt.plot(v, label='svd')
    plt.ylim(0, 1)
    plt.xticks([])
    plt.xlabel('Round channel')
    plt.title('svd and mean for each round channel')
    plt.legend()
    plt.suptitle('Spots and predictions for gene ' + gene_name + ' on tile ' + str(tile))
    plt.savefig('/home/reilly/Desktop/svd_plots_new/' + gene_name + '_' + str(tile) + '.png')
    plt.close(fig)


def plot_free_comparison(F, b, A, gene_names, save=False):
    predicted = np.zeros_like(F)
    for t in range(F.shape[0]):
        for g in range(F.shape[1]):
            predicted[t, g] = A[t] * b[g]
    score = np.zeros((F.shape[0], F.shape[1])) * np.nan
    for t in range(F.shape[0]):
        for g in range(F.shape[1]):
            if np.max(F[t, g]) == 0:
                continue
            score[t, g] = np.sum(F[t, g] * predicted[t, g]) / np.linalg.norm(F[t, g]) / np.linalg.norm(predicted[t, g])
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(score.T, aspect='auto', interpolation='none')
    plt.xlabel('Tile')
    plt.ylabel('Gene')
    plt.xticks(range(F.shape[0]), range(F.shape[0]))
    plt.yticks(range(F.shape[1]), gene_names, fontsize=4)
    plt.colorbar()
    plt.title('Similarity score between predicted and actual free bled codes for each gene and tile')
    plt.show()
    if save:
        plt.savefig('/home/reilly/Desktop/free_comparison.png')
        plt.close(fig)


def plot_cnf(colour_norm_factor, use_channels, n_tiles, n_rounds):
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


def view_A_fit(F, G, n, t, r, c, save=False):
    A = np.sum(n * F * G) / np.sum(n * G * G)
    F_bar = weighted_mean(F, n)
    G_bar = weighted_mean(G, n)
    weighted_cov = weighted_mean((F - F_bar) * (G - G_bar), n)
    weighted_var_F = weighted_mean((F - F_bar) ** 2, n)
    weighted_var_G = weighted_mean((G - G_bar) ** 2, n)
    weighted_correlation = weighted_cov / np.sqrt(weighted_var_F * weighted_var_G)

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(G, F, label='data. Weighted correlation = {:.3f}'.format(weighted_correlation), c='red',
                s=np.clip(n, 0, 100))
    plt.plot(G, A * G, label='fit', c='blue')
    plt.legend()
    plt.xlabel('Tile independent free bled code')
    plt.ylabel('Tile dependent free bled code')
    plt.title('Fit for tile {}, round {}, channel {}'.format(t, r, c))
    if save:
        plt.savefig('/home/reilly/Desktop/A_fits/' + str(t) + '_' + str(r) + '_' + str(c) + '.png')
        plt.close(fig)
    else:
        plt.show()


def view_all_genes(bled_codes_tile_indep, gene_names, scale_correction):
    n_genes, n_rounds, n_channels = bled_codes_tile_indep.shape
    square_side = int(np.ceil(np.sqrt(n_genes)))
    fig, ax = plt.subplots(square_side, 2 * square_side, figsize=(20, 10))
    # we will plot the raw codes in the even columns and the scaled codes in the odd columns
    for g in range(n_genes):
        row = g // square_side
        col = (g % square_side) * 2
        ax[row, col].imshow(bled_codes_tile_indep[g].T, aspect='auto', interpolation='none', vmin=0,
                            vmax=max(np.max(bled_codes_tile_indep[g]),
                                     np.max(bled_codes_tile_indep[g] / scale_correction)))
        if row == square_side - 1:
            ax[row, col].set_xlabel('Round')
        if col == 0:
            ax[row, col].set_ylabel('Intensity')
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        ax[row, col].set_title(gene_names[g] + 'raw', fontsize=6)

    for g in range(n_genes):
        row = g // square_side
        col = 2 * (g % square_side) + 1
        ax[row, col].imshow(bled_codes_tile_indep[g].T / scale_correction.T, aspect='auto', interpolation='none', vmin=0,
                            vmax=max(np.max(bled_codes_tile_indep[g]),
                                     np.max(bled_codes_tile_indep[g] / scale_correction)))
        if row == square_side - 1:
            ax[row, col].set_xlabel('Round')
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        ax[row, col].set_title(gene_names[g] + 'scaled', fontsize=6)

    # loop through the rest of the axes and turn them off
    for g in range(n_genes, square_side ** 2):
        row = g // square_side
        col = 2 * (g % square_side)
        ax[row, col].axis('off')
        ax[row, col + 1].axis('off')

    fig.suptitle('Tile independent free bled codes for all genes, scaled and unscaled')
    plt.show()


def view_all_gene_round_strengths(bled_codes_tile_indep, gene_names, scale_correction: np.ndarray, save=False):
    n_genes, n_rounds, n_channels = bled_codes_tile_indep.shape
    square_side = int(np.ceil(np.sqrt(n_genes)))
    fig, ax = plt.subplots(square_side, square_side, figsize=(20, 10))
    for g in range(n_genes):
        row = g // square_side
        col = g % square_side
        ax[row, col].plot(np.sum(bled_codes_tile_indep[g], axis=1))
        ax[row, col].plot(np.sum(bled_codes_tile_indep[g] / scale_correction, axis=1))
        if row == square_side - 1:
            ax[row, col].set_xlabel('Round')
        if col == 0:
            ax[row, col].set_ylabel('Intensity')
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        ax[row, col].set_title(gene_names[g])
    for g in range(n_genes, square_side ** 2):
        row = g // square_side
        col = g % square_side
        ax[row, col].axis('off')
    raw_round_strength = np.sum(bled_codes_tile_indep, axis=2)
    scaled_round_strength = np.sum(bled_codes_tile_indep / scale_correction, axis=2)
    raw_std = np.std(raw_round_strength)
    scaled_std = np.std(scaled_round_strength)
    fig.suptitle('Tile independent free bled codes for all genes. Blue is unscaled, orange is scaled \n'
                 'Raw Average Standard Deviation = {:.3f}, Scaled Average Standard Deviation = {:.3f}'.format(raw_std,scaled_std))
    if save:
        plt.savefig('/home/reilly/Desktop/tile_indep_free_bled_codes.png')
        plt.close(fig)
    else:
        plt.show()


def weighted_mean(x, w):
    return np.sum(x * w) / np.sum(w)


def compare_dyes(free_bled_codes_tile_indep, gene_codes, dye, gene_names):
    # We will compare the dye reads from each gene to each other
    n_genes, n_rounds, n_channels = free_bled_codes_tile_indep.shape
    relevant_genes = np.unique(np.where(gene_codes == dye)[0])
    gene_codes = gene_codes.tolist()
    dye_d_index = [gene_codes[g].index(dye) for g in relevant_genes]
    dye_corr = np.zeros((len(relevant_genes), len(relevant_genes)))
    for g1, g2 in product(range(len(relevant_genes)), range(len(relevant_genes))):
        if g1 == g2:
            dye_corr[g1, g2] = 1
        g1_dye = free_bled_codes_tile_indep[relevant_genes[g1], dye_d_index[g1]]
        g2_dye = free_bled_codes_tile_indep[relevant_genes[g2], dye_d_index[g2]]
        dye_corr[g1, g2] = np.sum(g1_dye * g2_dye) / np.linalg.norm(g1_dye) / np.linalg.norm(g2_dye)

    # Now we will plot the correlation matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(dye_corr, aspect='auto', interpolation='none')
    plt.xticks(range(len(relevant_genes)), [gene_names[g] for g in relevant_genes], rotation=90, fontsize=4)
    plt.yticks(range(len(relevant_genes)), [gene_names[g] for g in relevant_genes], fontsize=4)
    plt.colorbar()
    plt.title('Correlation between dye reads for dye {} for all genes containing this dye'.format(dye))
    plt.show()


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


def call_ref_spots_test(colours: np.ndarray, gene_prob_initial: np.ndarray, spot_tile: np.ndarray, bg_codes:np.ndarray,
                        gene_names: np.ndarray, gene_codes: np.ndarray):
    """
    This produces the bleed matrix and expected code for each gene as well as producing a gene assignment based on a
    simple dot product for spots found on the reference round.

    Returns the `call_spots` notebook page and adds the following variables to the `ref_spots` page:
    `gene_no`, `score`, `score_diff`, `intensity`.

    See `'call_spots'` and `'ref_spots'` sections of `notebook_comments.json` file
    for description of the variables in each page.

    Args:
        colours: n_spots x n_rounds x n_channels_use array of spot intensities
        gene_prob_initial: n_spots x n_genes array of gene probabilities
        spot_tile: n_spots array of spot tile numbers
        bg_codes: n_spots x n_rounds x n_channels_use array of background codes
        gene_names: n_genes array of gene names
        gene_codes: n_genes x n_rounds array of gene codes
    """

    # 0. Initialise frequently used variables
    n_genes = len(gene_names)
    use_channels = [5, 9, 14, 15, 18, 23, 27]
    use_tiles = [0, 1, 2, 3, 4, 5, 6, 7]
    target_channel_strength = [1, 1, 0.9, 0.8, 0.8, 1, 1]
    n_tiles, n_rounds, n_channels_use = 8, 7, 7
    n_spots, n_dyes = colours.shape[0], 7

    # Load bleed matrix info
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
    dye_names = list(dye_info.keys())
    # initial_bleed_matrix is n_channels x n_dyes
    initial_bleed_matrix = np.zeros((len(use_channels), len(dye_names)))
    # Populate initial_bleed_matrix with dye info for all channels in use
    for i, dye in enumerate(dye_names):
        initial_bleed_matrix[:, i] = dye_info[dye][use_channels]

    # initialise bleed matrices
    initial_bleed_matrix = initial_bleed_matrix / np.linalg.norm(initial_bleed_matrix, axis=0)
    bled_codes = call_spots.get_bled_codes(gene_codes=gene_codes, bleed_matrix=initial_bleed_matrix,
                                           gene_efficiency=np.ones((n_genes, n_rounds)))

    # Define an inlier mask to remove outliers
    rc_max = np.max(colours.reshape(n_spots, -1), axis=1)
    rc_min = np.min(colours.reshape(n_spots, -1), axis=1)
    inlier_mask = (np.max(colours.reshape(n_spots, -1), axis=1) < np.percentile(rc_max, 99)) * \
                  (np.min(colours.reshape(n_spots, -1), axis=1) > np.percentile(rc_min, 1))

    # Begin the iterative process of estimating bleed matrix, colour norm factor and free bled codes
    n_iter = 10
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
            colours_scaled = colours * scale[iter - 1]

        # 2. Gene assignments using FVM
        gene_prob = call_spots.gene_prob_score(spot_colours=colours_scaled, bled_codes=bled_codes)
        gene_no_temp = np.argmax(gene_prob, axis=1)
        gene_prob_score_temp = np.max(gene_prob, axis=1)
        # Save gene_no and gene_prob_score, so we can see how much they change between iterations
        gene_no[iter] = gene_no_temp
        gene_prob_score[iter] = gene_prob_score_temp

        if iter > 0:
            frac_mismatch = np.sum(gene_no[iter] != gene_no[iter - 1]) / n_spots
            print('Fraction of spots assigned to different genes: {:.3f}'.format(frac_mismatch))

        # 3. Update bleed matrix
        bleed_matrix_prob_thresh = 0.9
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
            v *= np.sign(v[np.argmax(np.abs(v))])   # Make sure the largest element is positive
            bleed_matrix[iter, :, d] = v

        # as part of this step, update the bled_codes (not the free_bled_codes)
        bled_codes = call_spots.get_bled_codes(gene_codes=gene_codes, bleed_matrix=bleed_matrix[iter],
                                               gene_efficiency=np.ones((n_genes, n_rounds)))

        # We need to have a way of changing between dyes and channels, so we define d_max and d_2_max
        d_max = np.zeros(n_channels_use, dtype=int)
        d_2nd_max = np.zeros(n_channels_use, dtype=int)
        for c in range(n_channels_use):
            d_max[c] = np.argmax(bleed_matrix[iter, c])
            d_2nd_max[c] = np.argsort(bleed_matrix[iter, c])[-2]

        # 4. Estimate free_bled_codes
        conc_param_round = .1  # it takes this many spots to scale the bled code for any round
        conc_param_other = 40  # it takes this many spots to change the bled code in a round
        free_bled_prob_thresh = 0.9
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

        free_bled_codes = np.repeat(free_bled_codes_tile_indep[None, :, :, :], n_tiles, axis=0)
        for t, g in product(use_tiles, range(n_genes)):
            keep = (spot_tile == t) * (gene_prob_score[iter] > free_bled_prob_thresh) * (gene_no[iter] == g) * \
                   inlier_mask
            colours_tg = colours_scaled[keep]
            if np.sum(keep) <= 1:
                continue
            for r in range(n_rounds):
                dye_colour = bleed_matrix[iter, :, gene_codes[g, r]] / np.sqrt(n_rounds)
                free_bled_codes[t, g, r] = bayes_mean(data=colours_tg[:, r], prior_mean=dye_colour,
                                                      mean_dir_conc=conc_param_round, other_conc=conc_param_other)
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
            n = np.sum(n_reads[:, relevant_genes], axis=0)
            target_matching_scale[r, c] = np.sum(F * G * n) / np.sum(G * G * n)

        for t, r, c in product(use_tiles, range(n_rounds), range(n_channels_use)):
            dye_c = d_max[c]
            dye_2nd_c = d_2nd_max[c]
            # we want to know how much more to weight contributions from dye_c than dye_2nd_c. We will do this by
            # looking at the ratio of their expected strengths.
            dye_2_strength = bleed_matrix[iter, c, dye_2nd_c] / bleed_matrix[iter, c, dye_c]
            relevant_genes_dye_c = np.where(gene_codes[:, r] == dye_c)[0]
            v_rc = target_matching_scale[r, c]
            F = free_bled_codes_tile_indep[relevant_genes_dye_c, r, c] * v_rc
            G = free_bled_codes[t, relevant_genes_dye_c, r, c]
            n = n_reads[t, relevant_genes_dye_c]
            relevant_genes_dye_2nd_c = np.where(gene_codes[:, r] == dye_2nd_c)[0]
            F = np.concatenate((F, free_bled_codes_tile_indep[t, relevant_genes_dye_2nd_c, r, c] * v_rc))
            G = np.concatenate((G, free_bled_codes[relevant_genes_dye_2nd_c, r, c]))

            n = np.concatenate((n, dye_2_strength * n_reads[t, relevant_genes_dye_2nd_c]))
            if np.sum(n) == 0:
                continue
            A = np.sum(n * F * G) / np.sum(n * G * G)
            scale[iter, t, r, c] = A


