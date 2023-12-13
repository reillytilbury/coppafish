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
from ..extract import scale
from ..call_spots import get_spot_intensity
from tqdm import tqdm
from itertools import product


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

    # initialise various bleed matrices
    free_bleed_matrix = np.zeros((n_tiles, n_rounds, initial_bleed_matrix.shape[0], initial_bleed_matrix.shape[1]))
    initial_bleed_matrix = initial_bleed_matrix / np.linalg.norm(initial_bleed_matrix, axis=0)
    bleed_matrix = np.zeros_like(initial_bleed_matrix)

    # Initialise the 2 variables we are most interested in estimating: colour_norm_factor and gene_efficiency
    colour_norm_factor = np.ones((n_tiles, n_rounds, n_channels_use))
    gene_efficiency = np.ones((n_genes, n_rounds))

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
        colour_norm_factor[t] = np.percentile(abs(tile_colours), 95, axis=0)
        colours[spot_tile == t] /= colour_norm_factor[t]
    # Remove background
    bg_codes = np.zeros((n_spots, n_rounds, n_channels_use))
    bg = np.percentile(colours, 25, axis=1)
    for r, c in product(range(n_rounds), range(n_channels_use)):
        bg_codes[:, r, c] = bg[:, c]
    colours -= bg_codes

    # Part 2: Re estimate bleed matrix. To do this, assign spots to genes and we will then use this classification to
    # estimate the bleed matrix.
    initial_bled_codes = call_spots.get_bled_codes(gene_codes=gene_codes, bleed_matrix=initial_bleed_matrix,
                                                   gene_efficiency=np.ones((n_genes, n_rounds)))
    gene_prob = call_spots.gene_prob_score(spot_colours=colours, bled_codes=initial_bled_codes)
    gene_no = np.argmax(gene_prob, axis=1)
    gene_prob_score = np.max(gene_prob, axis=1)

    # Now we can update our bleed matrix using these gene assignments.
    # Do this by looking at high probability, low background spots, look at the preimage of these genes to get the dye
    # in each round, then combine all such dyes across all rounds and take the SVD to get the dye colour. This is how
    # we estimate each column of the bleed matrix.
    high_prob = gene_prob_score > 0.9
    low_bg = np.linalg.norm(bg_codes, axis=(1, 2)) < np.percentile(np.linalg.norm(bg_codes, axis=(1, 2)), 50)
    for d in range(n_dyes):
        dye_d_spots = np.zeros((0, n_channels_use))
        for r in range(n_rounds):
            dye_d_round_r_genes = np.where(gene_codes[:, r] == d)[0]
            is_relevant_gene = np.isin(gene_no, dye_d_round_r_genes)
            dye_d_spots = np.concatenate((dye_d_spots, colours[is_relevant_gene * high_prob * low_bg, r, :]))
        dye_d_spots_norm = np.linalg.norm(dye_d_spots, axis=1)
        dye_d_spots = dye_d_spots[(dye_d_spots_norm < np.percentile(dye_d_spots_norm, 99)) *
                                  (dye_d_spots_norm > np.percentile(dye_d_spots_norm, 1))]
        if len(dye_d_spots) == 0:
            continue
        u, s, v = svds(dye_d_spots, k=1)
        v = v[0]
        v *= np.sign(v[np.argmax(np.abs(v))])   # Make sure the largest element is positive
        bleed_matrix[:, d] = v

    # part 3: estimate the free bled codes for each gene and tile
    free_bled_codes = np.zeros((n_tiles, n_genes, n_rounds, n_channels_use))
    free_g_prob_thresh = min(0.6, np.percentile(gene_prob_score, 50))
    for t, g in product(nbp_basic.use_tiles, range(n_genes)):
        keep = (spot_tile == t) * (gene_prob_score > free_g_prob_thresh) * (gene_no == g)
        colours_tg = colours[keep].reshape((np.sum(keep), -1))
        if np.sum(keep) <= 1:
            continue
        u, s, v = svds(colours_tg, k=1)
        v = v[0]
        v *= np.sign(v[np.argmax(np.abs(v))])  # Make sure the largest element is positive
        free_bled_codes[t, g] = v.reshape((n_rounds, n_channels_use))

    # part 4: estimate gene efficiency and colour norm factor
    d_max = np.zeros(n_channels_use, dtype=int)
    for c in range(n_channels_use):
        d_max[c] = np.argmax(bleed_matrix[c])

    # We will loop through each round and channel and perform an SVD on the free bled code matrix (which is n_tiles x
    # n_genes_rc, where n_genes_rc is the number of genes that have dye d_max[c] in round r). These singular vectors
    # will be the gene efficiency (n_genes_rc x 1) and the colour norm factor (n_tiles x 1) for round r and channel c.
    ge_prob_thresh = min(0.6, np.percentile(gene_prob_score, 50))
    for r, c in product(range(n_rounds), range(n_channels_use)):
        c_dye = d_max[c]
        relevant_genes = np.where(gene_codes[:, r] == c_dye)[0]
        n_genes_rc = len(relevant_genes)
        colour_matrix = free_bled_codes[:, relevant_genes, r, c_dye]
        # we would like to perform an svd on colour_matrix, but certain tile gene pairs are missing. We will use the
        # following trick to get around this. We will create a matrix n_reads of size n_tiles x n_genes_rc, with
        # n_reads[t, g] = the number of spots of gene relevant_genes[g] on tile t. We will then find the best rank 1
        # approximation to n_reads, multiply this by colour_matrix and then perform an SVD on the result.
        n_reads = np.zeros((n_tiles, n_genes_rc))
        for t, g in product(nbp_basic.use_tiles, range(n_genes_rc)):
            n_reads[t, g] = np.sum((spot_tile == t) * (gene_no == relevant_genes[g]) *
                                   (gene_prob_score > ge_prob_thresh))
        u_n, s, v_n = svds(n_reads, k=1)
        weight_matrix = u_n * v_n
        colour_matrix = weight_matrix * colour_matrix
        u, s, v = svds(colour_matrix, k=1)
        gene_efficiency[relevant_genes, r] = (u / u_n)[:, 0]
        colour_norm_factor[:, r, c] = (v / v_n)[0]
        # The next line ensures that the sum of the colour norm factors for each tile is equal to the number of dyes
        tile_scale_rc = n_tiles / np.sum(colour_norm_factor[:, r, c])
        colour_norm_factor[:, r, c] *= tile_scale_rc
        gene_efficiency[relevant_genes, r] *= s / (tile_scale_rc * bleed_matrix[c, c_dye])

    # Divide the spots through by the colour norm factor
    for t in nbp_basic.use_tiles:
        colours[spot_tile == t] /= colour_norm_factor[t]

    # Calculate bled codes with this gene efficiency
    bled_codes = call_spots.get_bled_codes(gene_codes=gene_codes, bleed_matrix=bleed_matrix,
                                           gene_efficiency=gene_efficiency)
    n_spots = colours.shape[0]
    n_genes = bled_codes.shape[0]
    gene_no, gene_score, gene_score_second \
        = call_spots.dot_product_score(spot_colours=colours.reshape((n_spots, -1)),
                                       bled_codes=bled_codes.reshape((n_genes, -1)))[:3]

    # save overwritable variables in nbp_ref_spots
    nbp_ref_spots.gene_no = gene_no
    nbp_ref_spots.score = gene_score
    nbp_ref_spots.score_diff = gene_score - gene_score_second
    nbp_ref_spots.intensity = np.median(np.max(colours, axis=2), axis=1).astype(np.float32)
    nbp_ref_spots.background_strength = bg_codes
    nbp_ref_spots.gene_probs = gene_prob
    # nbp_ref_spots.dye_strengths = dye_strength
    nbp_ref_spots.finalized = True

    # Save variables in nbp
    nbp.gene_names = gene_names
    nbp.gene_codes = gene_codes
    # Now expand variables to have n_channels channels instead of n_channels_use channels. For some variables, we
    # also need to swap axes as the expand channels function assumes the last axis is the channel axis.
    nbp.color_norm_factor = utils.base.expand_channels(colour_norm_factor, use_channels, nbp_basic.n_channels)
    nbp.initial_bleed_matrix = utils.base.expand_channels(initial_bleed_matrix.T, use_channels, nbp_basic.n_channels).T
    nbp.bleed_matrix = utils.base.expand_channels(bleed_matrix.T, use_channels, nbp_basic.n_channels).T
    nbp.bled_codes_ge = utils.base.expand_channels(bled_codes, use_channels, nbp_basic.n_channels)
    nbp.bled_codes = utils.base.expand_channels(call_spots.get_bled_codes(gene_codes=gene_codes, 
                                                                          bleed_matrix=bleed_matrix, 
                                                                          gene_efficiency=np.ones((n_genes, n_rounds))),
                                                use_channels, nbp_basic.n_channels)
    nbp.gene_efficiency = gene_efficiency

    # Extract abs intensity percentile
    central_tile = scale.central_tile(nbp_basic.tilepos_yx, nbp_basic.use_tiles)
    if nbp_basic.is_3d:
        mid_z = int(nbp_basic.use_z[0] + (nbp_basic.use_z[-1] - nbp_basic.use_z[0]) // 2)
    else:
        mid_z = None
    pixel_colors = spot_colors.get_spot_colors(spot_colors.all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, mid_z),
                                               central_tile, transform, nbp_file, nbp_basic, nbp_extract,
                                               return_in_bounds=True)[0]
    pixel_intensity = get_spot_intensity(np.abs(pixel_colors) / colour_norm_factor[central_tile])
    nbp.abs_intensity_percentile = np.percentile(pixel_intensity, np.arange(1, 101))

    return nbp, nbp_ref_spots

