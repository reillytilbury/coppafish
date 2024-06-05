import itertools
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse.linalg import svds
from .. import log
from ..setup import NotebookPage
from ..call_spots import dot_product_score, gene_prob_score


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


def view_free_and_target_bled_codes(free_bled_codes_tile_indep: np.ndarray,
                                    target_bled_codes: np.ndarray,
                                    target_scale: np.ndarray,
                                    gene_names: np.ndarray,
                                    n_spots: np.ndarray) -> None:
    """
    Function to plot the free and target bleed codes for each gene.
    Args:
        free_bled_codes_tile_indep: np.ndarray [n_genes x n_rounds x n_channels]
            The free bled codes.
        target_bled_codes: np.ndarray [n_genes x n_rounds x n_channels]
            The target bled codes.
        target_scale: np.ndarray [n_rounds x n_channels]
            The scale factor for each round and channel.
        gene_names: np.ndarray [n_genes]
            The gene names.
        n_spots: np.ndarray [n_genes]
            The number of spots for each gene.
    """
    n_columns = 9
    n_genes, n_rounds, n_channels = free_bled_codes_tile_indep.shape
    n_rows = n_genes // n_columns + 1

    fig, ax = plt.subplots(n_rows, n_columns, figsize=(25, 15))
    codes = np.zeros((n_genes, n_rounds, 2 * n_channels + 1)) * np.nan
    # fill in the codes
    for g in range(n_genes):
        for r in range(n_rounds):
            codes[g, r, :n_channels] = free_bled_codes_tile_indep[g, r]
            codes[g, r, -n_channels:] = target_bled_codes[g, r]
    # fill in the image grid
    for g in range(n_genes):
        row, col = g // n_columns, g % n_columns
        ax[row, col].imshow(codes[g], cmap='viridis')
        ax[row, col].set_title(f"{gene_names[g]} ({n_spots[g]})", fontsize=8)
        ax[row, col].axis('off')
    for g in range(n_genes, n_rows * n_columns - 1):
        row, col = g // n_columns, g % n_columns
        ax[row, col].axis('off')

    # add the target scale
    ax[-1, -1].imshow(target_scale.T, cmap='viridis')
    ax[-1, -1].set_title("Target scale", fontsize=8)
    ax[-1, -1].set_xlabel("Round", fontsize=8)
    ax[-1, -1].set_ylabel("Channel", fontsize=8)
    ax[-1, -1].set_xticks([])
    ax[-1, -1].set_yticks([])

    # add title
    plt.suptitle("Free (left) and target (right) bleed codes")
    plt.show()


def view_tile_bled_codes(free_bled_codes: np.ndarray, free_bled_codes_tile_indep: np.ndarray,
                         gene_names: np.ndarray, use_tiles: np.ndarray, gene: int) -> None:
    """
    Function to plot the free bled codes for each tile for a given gene.
    Args:
        free_bled_codes: np.ndarray [n_genes x n_tiles x n_rounds x n_channels]
            The free bled codes.
        free_bled_codes_tile_indep: np.ndarray [n_genes x n_rounds x n_channels]
            The tile independent free bled codes.
        gene_names: np.ndarray [n_genes]
            The gene names.
        use_tiles: np.ndarray [n_tiles]
            The tiles to use.
        gene: int
            The gene to plot.
    """
    n_columns = 4
    _, _, n_rounds, n_channels = free_bled_codes.shape
    n_tiles = len(use_tiles)
    n_rows = int(np.ceil(n_tiles / n_columns)) + 1

    fig, ax = plt.subplots(n_rows, n_columns, figsize=(25, 15))
    for i, t in enumerate(use_tiles):
        row, col = i // n_columns, i % n_columns
        ax[row, col].imshow(free_bled_codes[gene, t].T, cmap='viridis')
        ax[row, col].set_title(f"Tile {t}")
        ax[row, col].set_xlabel("Round")
        ax[row, col].set_ylabel("Channel")
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
    for i in range(n_tiles, n_rows * n_columns - 1):
        row, col = i // n_columns, i % n_columns
        ax[row, col].axis('off')

    # add the tile independent bled code
    ax[-1, -1].imshow(free_bled_codes_tile_indep[gene].T, cmap='viridis')
    ax[-1, -1].set_title("Tile independent")
    ax[-1, -1].set_xlabel("Round")
    ax[-1, -1].set_ylabel("Channel")
    ax[-1, -1].set_xticks([])
    ax[-1, -1].set_yticks([])

    plt.suptitle(f"Free bled codes for gene {gene_names[gene]}")
    plt.show()


def view_target_scale_regression(target_scale: np.ndarray, gene_codes:np.ndarray, d_max: np.ndarray,
                                 target_values: np.ndarray, free_bled_codes_tile_indep: np.ndarray,
                                 n_spots: np.ndarray) -> None:
    """
    Plotter to show the regression of the target scale factor for each round and channel.
    Args:
        target_scale: np.ndarray [n_rounds x n_channels]
            The target scale factor.
        gene_codes: np.ndarray [n_genes x n_rounds]
            gene_codes[g, r] is the expected dye for gene g in round r.
        d_max: np.ndarray [n_channels]
            d_max[c] is the dye with the highest expression in channel c.
        target_values: np.ndarray [n_dyes]
            target_values[d] is the target value for dye d in its brightest channel.
        free_bled_codes_tile_indep: np.ndarray [n_genes x n_rounds x n_channels]
            The tile independent free bled codes.
        n_spots: np.ndarray [n_genes]
            The number of spots for each gene.

    """
    n_genes, n_rounds, n_channels = free_bled_codes_tile_indep.shape
    fig, ax = plt.subplots(n_rounds, n_channels, figsize=(25, 15))
    for r, c in np.ndindex(n_rounds, n_channels):
        relevant_genes = np.where(gene_codes[:, r] == d_max[c])[0]
        n_spots_rc = n_spots[relevant_genes]
        x = free_bled_codes_tile_indep[relevant_genes, r, c]
        y = [target_values[d_max[c]]] * len(relevant_genes)
        sizes = np.sqrt(n_spots_rc)
        ax[r, c].scatter(x, y, s=sizes)
        ax[r, c].plot(x, target_scale[r, c] * x, color='red')
        ax[r, c].set_title(f"Round {r}, Channel {c}", fontsize=8)
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_xlim(0, free_bled_codes_tile_indep.max())
        if r == n_rounds - 1:
            ax[r, c].set_xlabel("free rc")
        if c == 0:
            ax[r, c].set_ylabel("target rc")
    plt.suptitle("Each scale $V_{rc}$ is chosen so that the loss function "
                 "$L(V_{rc}) = \sum_{g \in G_{rc}} \sqrt{N_g}(E_{grc}V_{rc} - T_{d_{max}(c)})^2$ is minimised, \n"
                 "where $G_{rc}$ is the set of genes with gene code $d_{max}(c)$ in round r and $E_g$ is the "
                 "tile-independent bled code for gene g.")
    plt.show()


def view_homogeneous_scale_regression(homogeneous_scale: np.ndarray, gene_codes: np.ndarray, d_max: np.ndarray,
                                      target_bled_codes: np.ndarray, free_bled_codes: np.ndarray,
                                      n_spots: np.ndarray, t: int) -> None:
    """
    Plotter to show the regression of the homogeneous scale factor for tile t for all rounds and channels.

    Args:
        homogeneous_scale: np.ndarray [n_tiles x n_rounds x n_channels]
            The homogeneous scale factor.
        gene_codes: np.ndarray [n_genes x n_rounds]
            gene_codes[g, r] is the expected dye for gene g in round r.
        d_max: np.ndarray [n_channels]
            d_max[c] is the dye with the highest expression in channel c.
        target_bled_codes: np.ndarray [n_genes x n_rounds x n_channels]
            The target bled codes. target_bled_codes[g, r, c] = free_bled_codes_tile_indep[g, r, c] * target_scale[r, c]
        free_bled_codes: np.ndarray [n_genes x n_tiles x n_rounds x n_channels]
            The free bled codes.
        n_spots: np.ndarray [n_genes]
            The number of spots for each gene.

    """
    n_tiles, n_rounds, n_channels = homogeneous_scale.shape
    fig, ax = plt.subplots(n_rounds, n_channels, figsize=(25, 15))
    for r, c in np.ndindex(n_rounds, n_channels):
        relevant_genes = np.where(gene_codes[:, r] == d_max[c])[0]
        n_spots_rc = n_spots[relevant_genes]
        x = free_bled_codes[relevant_genes, t, r, c]
        y = target_bled_codes[relevant_genes, r, c]
        sizes = np.sqrt(n_spots_rc)
        ax[r, c].scatter(x, y, s=sizes)
        ax[r, c].plot(x, homogeneous_scale[t, r, c] * x, color='red')
        ax[r, c].set_title(f"r {r}, c {c}", fontsize=8)
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_xlim(0, free_bled_codes[:, t].max())
        ax[r, c].set_ylim(0, target_bled_codes.max())
        if r == n_rounds - 1:
            ax[r, c].set_xlabel("free rc")
        if c == 0:
            ax[r, c].set_ylabel("target rc")
    plt.suptitle("Each homogeneous scale $Q_{trc}$ is chosen so that the loss function "
                 "$L(Q_{trc}) = \sum_{g \in G_{rc}} \sqrt{N_g}(D_{gtrc}Q_{trc} - E_{grc}V_{rc})^2$ is minimised, \n"
                 "where $G_{rc}$ is the set of genes with gene code $d_{max}(c)$ in round r, $E_g$ is the "
                 "tile-independent bled code for gene g, $D_g$ is the tile-dependent bled code for gene g, "
                 "and $V_{rc}$ is the preliminary target scale factor.")

    plt.show()


def call_spots(nbp_ref_spots: NotebookPage, nbp_basic: NotebookPage, nbp_file: NotebookPage,
               nbp_extract: NotebookPage) -> NotebookPage:
    """
    Function to do gene assignments to reference spots. In doing so we compute some important parameters for the
    downstream analysis.

    Args:
        nbp_ref_spots: NotebookPage
            The reference spots notebook page. This will be altered in the process.
        nbp_basic: NotebookPage
            The basic info notebook page.
        nbp_file: NotebookPage
            The file names notebook page.
        nbp_extract: NotebookPage
            The extract notebook page.

    """
    log.debug("Call spots started")
    nbp = NotebookPage("call_spots")
    nbp_ref_spots.finalized = False

    # convert spot colours to float
    spot_colours = nbp_ref_spots.colours.astype(float)

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
    for r, c in np.ndindex(n_rounds, n_channels):
        rc_genes = np.where(gene_codes[:, r] == d_max[c])[0]
        n_spots = np.array([np.sum((prob_mode_initial == g) & (prob_score_initial > prob_threshold)) for g in rc_genes])
        target_scale[r, c] = np.sum(
            np.sqrt(n_spots) * free_bled_codes_tile_indep[rc_genes, r, c] * target_values[d_max[c]]) / np.sum(
            np.sqrt(n_spots) * free_bled_codes_tile_indep[rc_genes, r, c] ** 2)
    target_bled_codes = free_bled_codes_tile_indep * target_scale[None, :, :]
    # normalise the target bled codes
    target_bled_codes /= np.linalg.norm(target_bled_codes, axis=(1, 2))[:, None, None]

    # 6. compute the scale factor Q_trc maximising the similarity between the tile independent codes and the target
    # bled codes
    homogeneous_scale = np.ones((n_tiles, n_rounds, n_channels))
    for t, r, c in itertools.product(use_tiles, range(n_rounds), range(n_channels)):
        relevant_genes = np.where(gene_codes[:, r] == d_max[c])[0]
        n_spots = np.array([np.sum((prob_mode_initial == g) & (prob_score_initial > prob_threshold) & (spot_tile == t))
                            for g in relevant_genes])
        homogeneous_scale[t, r, c] = (
                np.sum(np.sqrt(n_spots) * target_bled_codes[relevant_genes, r, c] * free_bled_codes[
                    relevant_genes, t, r, c]) /
                np.sum(np.sqrt(n_spots) * free_bled_codes[relevant_genes, t, r, c] ** 2))

    # 7. update the normalised spots and the bleed matrix, then do a second round of gene assignments with the free
    # bled codes
    spot_colours = spot_colours * homogeneous_scale[spot_tile, :, :] # update the spot colours
    gene_prob = gene_prob_score(spot_colours=spot_colours, bled_codes=target_bled_codes) # update probs
    prob_mode, prob_score = np.argmax(gene_prob, axis=1), np.max(gene_prob, axis=1)
    gene_dot_products = dot_product_score(spot_colours=spot_colours, bled_codes=target_bled_codes)[-1]
    dp_mode, dp_score = np.argmax(gene_dot_products, axis=1), np.max(gene_dot_products, axis=1)
    # update bleed matrix
    good = prob_score > prob_threshold
    bleed_matrix = compute_bleed_matrix(spot_colours[good], prob_mode[good], gene_codes, n_dyes)

    return None

#
# colours = np.load(r"C:\Users\reill\Desktop\local_datasets\dante\dante_0_11_0\call_spots_variables\colours.npy")
# tile = np.load(r"C:\Users\reill\Desktop\local_datasets\dante\dante_0_11_0\call_spots_variables\spot_tile.npy")
# gene_codes = np.load(r"C:\Users\reill\Desktop\local_datasets\dante\dante_0_11_0\call_spots_variables\gene_codes.npy")
# gene_names = np.load(r"C:\Users\reill\Desktop\local_datasets\dante\dante_0_11_0\call_spots_variables\gene_names.npy")
# call_spots(colours, tile, gene_codes, gene_names)
