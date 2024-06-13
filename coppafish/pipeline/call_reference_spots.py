import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.sparse.linalg import svds
from .. import log
from ..setup import NotebookPage
from ..call_spots import dot_product_score, gene_prob_score


def bayes_mean(spot_colours: np.ndarray, prior_colours: np.ndarray, conc_param_parallel: float,
               conc_param_perp: float) -> np.ndarray:
    """
    This function computes the posterior mean of the spot colours under a prior distribution with mean prior_colours
    and covariance matrix given by a diagonal matrix with diagonal entry conc_param_parallel for the direction parallel
    to prior_colours and conc_param_perp for the direction orthogonal to prior_colours.

    Args:
        spot_colours: np.ndarray [n_spots x n_channels_use]
            The spot colours for each spot.
        prior_colours: np.ndarray [n_channels_use]
            The prior mean colours.
        conc_param_parallel: np.ndarray [n_channels_use]
            The concentration parameter for the direction parallel to prior_colours.
        conc_param_perp: np.ndarray [n_channels_use]
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
        spot_colours: np.ndarray [n_spots x n_rounds x n_channels_use]
            The spot colours for each spot in each round and channel.
        gene_no: np.ndarray [n_spots]
            The gene assignment for each spot.
        gene_codes: np.ndarray [n_genes x n_rounds]
            The gene codes for each gene in each round.
        n_dyes: int
            The number of dyes.

    Returns:
        bleed_matrix: np.ndarray [n_dyes x n_channels_use]
            The bleed matrix.
    """
    assert len(spot_colours) == len(gene_no), "Spot colours and gene_no must have the same length."
    n_spots, n_rounds, n_channels_use = spot_colours.shape
    bleed_matrix = np.zeros((n_dyes, n_channels_use))

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


def view_free_and_constrained_bled_codes(free_bled_codes_tile_indep: np.ndarray,
                                    bled_codes: np.ndarray,
                                    target_scale: np.ndarray,
                                    gene_names: np.ndarray,
                                    n_spots: np.ndarray) -> None:
    """
    Function to plot the free and target bleed codes for each gene.
    Args:
        free_bled_codes_tile_indep: np.ndarray [n_genes x n_rounds x n_channels_use]
            The free bled codes.
        bled_codes: np.ndarray [n_genes x n_rounds x n_channels_use]
            The target bled codes.
        target_scale: np.ndarray [n_rounds x n_channels_use]
            The scale factor for each round and channel.
        gene_names: np.ndarray [n_genes]
            The gene names.
        n_spots: np.ndarray [n_genes]
            The number of spots for each gene.
    """
    n_columns = 9
    n_genes, n_rounds, n_channels_use = free_bled_codes_tile_indep.shape
    n_rows = n_genes // n_columns + 1

    fig, ax = plt.subplots(n_rows, n_columns)
    codes = np.zeros((n_genes, n_channels_use, 2 * n_rounds + 1)) * np.nan
    # fill in the codes
    for g in range(n_genes):
        codes[g, :, :n_rounds] = free_bled_codes_tile_indep[g].T
        codes[g, :, -n_rounds:] = bled_codes[g].T
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
        free_bled_codes: np.ndarray [n_genes x n_tiles x n_rounds x n_channels_use]
            The free bled codes.
        free_bled_codes_tile_indep: np.ndarray [n_genes x n_rounds x n_channels_use]
            The tile independent free bled codes.
        gene_names: np.ndarray [n_genes]
            The gene names.
        use_tiles: np.ndarray [n_tiles]
            The tiles to use.
        gene: int
            The gene to plot.
    """
    n_columns = 4
    _, _, n_rounds, n_channels_use = free_bled_codes.shape
    n_tiles = len(use_tiles)
    n_rows = int(np.ceil(n_tiles / n_columns)) + 1

    fig, ax = plt.subplots(n_rows, n_columns)
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
                                 n_spots: np.ndarray, use_channels: tuple) -> None:
    """
    Plotter to show the regression of the target scale factor for each round and channel.
    Args:
        target_scale: np.ndarray [n_rounds x n_channels_use]
            The target scale factor.
        gene_codes: np.ndarray [n_genes x n_rounds]
            gene_codes[g, r] is the expected dye for gene g in round r.
        d_max: np.ndarray [n_channels_use]
            d_max[c] is the dye with the highest expression in channel c.
        target_values: np.ndarray [n_dyes]
            target_values[d] is the target value for dye d in its brightest channel.
        free_bled_codes_tile_indep: np.ndarray [n_genes x n_rounds x n_channels_use]
            The tile independent free bled codes.
        n_spots: np.ndarray [n_genes]
            The number of spots for each gene.
        use_channels: np.ndarray [n_channels_use]

    """
    n_genes, n_rounds, n_channels_use = free_bled_codes_tile_indep.shape
    fig, ax = plt.subplots(n_channels_use, 1)
    for c in range(n_channels_use):
        x, y, y_scaled, s = [], [], [], []
        for r in range(n_rounds):
            # get data
            relevant_genes = np.where(gene_codes[:, r] == d_max[c])[0]
            n_spots_rc = n_spots[relevant_genes]
            new_y_vals = free_bled_codes_tile_indep[relevant_genes, r, c]
            new_y_scaled_vals = free_bled_codes_tile_indep[relevant_genes, r, c] * target_scale[r, c]
            # append data
            y.append(new_y_vals)
            y_scaled.append(new_y_scaled_vals)
            s.append(np.sqrt(n_spots_rc))
            x.append(np.repeat(2 * r, len(relevant_genes)))
        # convert data to numpy arrays
        x, y, y_scaled, s = np.concatenate(x), np.concatenate(y), np.concatenate(y_scaled), np.concatenate(s)
        jitter = np.random.normal(0, 0.1, len(x))
        ax[c].scatter(x + jitter, y, s=s, c='cyan', alpha=0.5)
        ax[c].scatter(x + 1 + jitter, y_scaled, s=s, c='red', alpha=0.5)
        # add a horizontal line for the target value
        ax[c].axhline(target_values[d_max[c]], color='white', linestyle='--')
        ax[c].set_xticks([])
        max_val = max(np.max(free_bled_codes_tile_indep), np.max(free_bled_codes_tile_indep * target_scale))
        ax[c].set_yticks(np.round([0, target_values[d_max[c]], max_val], 2))
        ax[c].set_xlim(-1, 2 * n_rounds)
        ax[c].set_ylim(0, max_val)
        ax[c].set_ylabel(f"Channel {use_channels[c]}")
        # add text to the right hand side of each row
        ax[c].text(2 * n_rounds + 0.25, 1, f"mean scale = {np.mean(target_scale[:, c]) :.2f}")
        if c == n_channels_use - 1:
            ax[c].set_xlabel("Round")
            ax[c].set_xticks(np.arange(0, 2 * n_rounds, 2), labels=np.arange(n_rounds))

    plt.suptitle("Each scale $V_{rc}$ is chosen so that the loss function "
                 "$L(V_{rc}) = \sum_{g \in G_{rc}} \sqrt{N_g}(E_{grc}V_{rc} - T_{d_{max}(c)})^2$ is minimised, \n"
                 "where $G_{rc}$ is the set of genes with gene code $d_{max}(c)$ in round r and $E_g$ is the "
                 "tile-independent bled code for gene g. Cyan points are raw values, red points are scaled values.")
    # add another figure to show the target scale
    fig2, ax2 = plt.subplots(1, 1)
    ax2.imshow(target_scale, cmap='viridis')
    ax2.set_title("Target scale")
    ax2.set_ylabel("Round")
    ax2.set_xlabel("Channel")
    ax2.set_yticks(np.arange(n_rounds))
    ax2.set_xticks(np.arange(n_channels_use), labels=use_channels)
    # add colorbar
    cbar = plt.colorbar(ax2.imshow(target_scale, cmap='viridis'), ax=ax2)
    cbar.set_label("Scale factor")
    plt.show()


def view_homogeneous_scale_regression(homogeneous_scale: np.ndarray, gene_codes: np.ndarray, d_max: np.ndarray,
                                      bled_codes: np.ndarray, free_bled_codes: np.ndarray,
                                      n_spots: np.ndarray, t: int, use_channels: tuple) -> None:
    """
    Plotter to show the regression of the homogeneous scale factor for tile t for all rounds and channels.

    Args:
        homogeneous_scale: np.ndarray [n_tiles x n_rounds x n_channels_use]
            The homogeneous scale factor.
        gene_codes: np.ndarray [n_genes x n_rounds]
            gene_codes[g, r] is the expected dye for gene g in round r.
        d_max: np.ndarray [n_channels_use]
            d_max[c] is the dye with the highest expression in channel c.
        bled_codes: np.ndarray [n_genes x n_rounds x n_channels_use]
            The target bled codes. bled_codes[g, r, c] = free_bled_codes_tile_indep[g, r, c] * target_scale[r, c]
        free_bled_codes: np.ndarray [n_genes x n_tiles x n_rounds x n_channels_use]
            The free bled codes.
        n_spots: np.ndarray [n_genes]
            The number of spots for each gene.
        t: int
            The tile to plot.
        use_channels: tuple
            The channels to use.

    """
    use_channels = list(use_channels)
    n_tiles, n_rounds, n_channels_use = homogeneous_scale.shape
    fig, ax = plt.subplots(n_rounds, n_channels_use)
    for r, c in np.ndindex(n_rounds, n_channels_use):
        relevant_genes = np.where(gene_codes[:, r] == d_max[c])[0]
        n_spots_rc = n_spots[relevant_genes]
        x = free_bled_codes[relevant_genes, t, r, c]
        y = bled_codes[relevant_genes, r, c]
        sizes = np.sqrt(n_spots_rc)
        ax[r, c].scatter(x, y, s=sizes)
        ax[r, c].plot(x, homogeneous_scale[t, r, c] * x, color='red')
        ax[r, c].set_title(f"r {r}, c {use_channels[c]}", fontsize=8)
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_xlim(0, free_bled_codes[:, t].max())
        ax[r, c].set_ylim(0, bled_codes.max())
        if r == n_rounds - 1:
            ax[r, c].set_xlabel("$D_{gtrc}$")
        if c == 0:
            ax[r, c].set_ylabel("$K_{grc}$")
    plt.suptitle("Each homogeneous scale $Q_{trc}$ is chosen so that the loss function "
                 "$L(Q_{trc}) = \sum_{g \in G_{rc}} \sqrt{N_g}(D_{gtrc}Q_{trc} - K_{grc})^2$ is minimised, \n"
                 "where $G_{rc}$ is the set of genes with gene code $d_{max}(c)$ in round r, $K_{grc} = E_{grc}V_{rc}$ "
                 "is the target bled code for gene g, $D_g$ is the tile-dependent bled code for gene g, "
                 "and $V_{rc}$ is the preliminary target scale factor.")

    plt.show()


def view_homogeneous_scale_factors(homogeneous_scale: np.ndarray, target_scale: np.ndarray,
                                   use_tiles: tuple, use_rounds: tuple, use_channels: tuple) -> None:
    """
    Function to plot the homogeneous scale factors for each tile, round and channel.
    Args:
        homogeneous_scale: np.ndarray [n_tiles x n_rounds x n_channels_use]
            The homogeneous scale factors.
        target_scale: np.ndarray [n_rounds x n_channels_use]
            The target scale factors.
        use_tiles: tuple
            The tiles to use.
        use_rounds: tuple
            The rounds to use.
        use_channels:  tuple
            The channels to use.
    """
    use_tiles, use_rounds, use_channels = list(use_tiles), list(use_rounds), list(use_channels)
    homogeneous_scale = homogeneous_scale[use_tiles]
    relative_scale = homogeneous_scale / target_scale[None, :, :]
    n_tiles, n_rounds, n_channels_use = homogeneous_scale.shape
    homogeneous_scale = homogeneous_scale.reshape((n_tiles * n_rounds, n_channels_use))
    relative_scale = relative_scale.reshape((n_tiles * n_rounds, n_channels_use))
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(target_scale, cmap='viridis')
    ax[0].set_xticks(np.arange(n_channels_use), labels=use_channels)
    ax[0].set_yticks(np.arange(n_rounds), labels=use_rounds)
    ax[0].set_title("Target Scale Factors")
    ax[0].set_xlabel("Channel")
    ax[0].set_ylabel("Round")
    plt.colorbar(ax[0].imshow(target_scale, cmap='viridis'), ax=ax[0])
    for i, scale in enumerate([homogeneous_scale, relative_scale]):
        i += 1
        ax[i].imshow(scale, cmap='viridis')
        ax[i].set_xticks(np.arange(n_channels_use), labels=use_channels)
        ax[i].set_yticks(np.arange(n_tiles * n_rounds), labels=[f"T{t}, R{r}" for t, r in np.ndindex(n_tiles, n_rounds)],
                         fontsize=8)
        ax[i].set_title("Homogeneous Scale Factors" if i == 1 else "Homogeneous Scale Factors \n / Target Scale Factors")
        ax[i].set_xlabel("Channel")
        ax[i].set_ylabel("Tile x Round")
        # add horizontal red lines at each tile
        for t in range(1, n_tiles):
            ax[i].axhline(t * n_rounds - 0.5, color='red', linestyle='--')
        # add colorbar
        plt.colorbar(ax[i].imshow(scale, cmap='viridis'), ax=ax[i])
    plt.show()


def call_reference_spots(config: dict,
                         nbp_ref_spots: NotebookPage,
                         nbp_file: NotebookPage,
                         nbp_basic: NotebookPage) -> [NotebookPage, NotebookPage]:
    """
    Function to do gene assignments to reference spots. In doing so we compute some important parameters for the
    downstream analysis.

    Args:
        config: dict
            The configuration dictionary for the call spots page. Should contain the following keys:
            - gene_prob_threshold: float
                The threshold for the gene probability score.
            - target_values: list (length n_dyes)
                The target values for each dye.
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
    nbp = NotebookPage("call_spots")

    # load in frequently used variables
    spot_colours = nbp_ref_spots.colours.astype(float)
    spot_tile = nbp_ref_spots.tile

    gene_names, gene_codes = np.genfromtxt(nbp_file.code_book, dtype=(str, str)).transpose()
    gene_codes = np.array([[int(i) for i in gene_codes[j]] for j in range(len(gene_codes))])
    n_tiles, n_rounds, n_channels_use = nbp_basic.n_tiles, nbp_basic.n_rounds, len(nbp_basic.use_channels)
    n_dyes, n_spots, n_genes = len(nbp_basic.dye_names), len(spot_colours), len(gene_names)
    use_tiles, use_rounds, use_channels = (list(nbp_basic.use_tiles), list(nbp_basic.use_rounds),
                                           list(nbp_basic.use_channels))

    if nbp_file.initial_bleed_matrix is not None:
        raw_bleed_matrix = np.load(nbp_file.initial_bleed_matrix)
    else:
        setup_dir = os.path.join(os.getcwd().split('coppafish')[0], 'coppafish', 'coppafish', 'setup')
        raw_bleed_dir = os.path.join(setup_dir, 'dye_info_raw.npy')
        raw_bleed_matrix = np.load(raw_bleed_dir)[:, use_channels].astype(float)
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
    gene_prob = gene_prob_score(spot_colours, bled_codes)

    # 3. Use spots with score above threshold to work out global dye codes
    prob_mode_initial, prob_score_initial = np.argmax(gene_prob, axis=1), np.max(gene_prob, axis=1)
    prob_threshold = min(config['gene_prob_threshold'], np.percentile(prob_score_initial, 90))
    good = prob_score_initial > prob_threshold
    bleed_matrix_initial = compute_bleed_matrix(spot_colours[good], prob_mode_initial[good], gene_codes, n_dyes)
    d_max = np.argmax(bleed_matrix_initial, axis=0)

    # 4. Compute the free_bled_codes
    free_bled_codes = np.zeros((n_genes, n_tiles, n_rounds, n_channels_use))
    free_bled_codes_tile_indep = np.zeros((n_genes, n_rounds, n_channels_use))
    for g in range(n_genes):
        for r in range(n_rounds):
            good_g = (prob_mode_initial == g) & good
            free_bled_codes_tile_indep[g, r] = bayes_mean(spot_colours=spot_colours[good_g, r],
                                                          prior_colours=bleed_matrix_initial[gene_codes[g, r]],
                                                          conc_param_parallel=config['concentration_parameter_parallel'],
                                                          conc_param_perp=config['concentration_parameter_perpendicular'])
            for t in use_tiles:
                good_gt = (prob_mode_initial == g) & (spot_tile == t) & good
                free_bled_codes[g, t, r] = bayes_mean(spot_colours=spot_colours[good_gt, r],
                                                      prior_colours=bleed_matrix_initial[gene_codes[g, r]],
                                                      conc_param_parallel=config['concentration_parameter_parallel'],
                                                      conc_param_perp=config['concentration_parameter_perpendicular'])
    # normalise the free bled codes
    free_bled_codes_tile_indep /= np.linalg.norm(free_bled_codes_tile_indep, axis=(1, 2))[:, None, None]
    free_bled_codes[:, use_tiles] /= np.linalg.norm(free_bled_codes[:, use_tiles], axis=(2, 3))[:, :, None, None]

    # 5. compute the scale factor V_rc maximising the similarity between the tile independent codes and the target
    # values. Then rename the product V_rc * free_bled_codes to bled_codes
    target_scale = np.zeros((n_rounds, n_channels_use))
    for r, c in np.ndindex(n_rounds, n_channels_use):
        rc_genes = np.where(gene_codes[:, r] == d_max[c])[0]
        n_spots_per_gene = np.array([np.sum((prob_mode_initial == g) & (prob_score_initial > prob_threshold)) for g in rc_genes])
        target_scale[r, c] = (np.sum(
            np.sqrt(n_spots_per_gene) * free_bled_codes_tile_indep[rc_genes, r, c] * config['target_values'][d_max[c]])/
                              np.sum(
            np.sqrt(n_spots_per_gene) * free_bled_codes_tile_indep[rc_genes, r, c] ** 2))
    bled_codes = free_bled_codes_tile_indep * target_scale[None, :, :]
    # normalise the target bled codes
    bled_codes /= np.linalg.norm(bled_codes, axis=(1, 2))[:, None, None]

    # 6. compute the scale factor Q_trc maximising the similarity between the tile independent codes and the target
    # bled codes
    homogeneous_scale = np.ones((n_tiles, n_rounds, n_channels_use))
    for t, r, c in itertools.product(use_tiles, range(n_rounds), range(n_channels_use)):
        relevant_genes = np.where(gene_codes[:, r] == d_max[c])[0]
        n_spots_per_gene = np.array([np.sum((prob_mode_initial == g) &
                                            (prob_score_initial > prob_threshold) &
                                            (spot_tile == t))
                                     for g in relevant_genes])
        homogeneous_scale[t, r, c] = (
                np.sum(np.sqrt(n_spots_per_gene) * bled_codes[relevant_genes, r, c] * free_bled_codes[
                    relevant_genes, t, r, c]) /
                np.sum(np.sqrt(n_spots_per_gene) * free_bled_codes[relevant_genes, t, r, c] ** 2))

    # 7. update the normalised spots and the bleed matrix, then do a second round of gene assignments with the free
    # bled codes
    spot_colours = spot_colours * homogeneous_scale[spot_tile, :, :] # update the spot colours
    gene_prob = gene_prob_score(spot_colours=spot_colours, bled_codes=bled_codes) # update probs
    prob_mode, prob_score = np.argmax(gene_prob, axis=1), np.max(gene_prob, axis=1)
    gene_dot_products = dot_product_score(spot_colours=spot_colours.reshape((n_spots, n_rounds * n_channels_use)),
                                          bled_codes=bled_codes.reshape((n_genes, n_rounds * n_channels_use)))[-1]
    dp_mode, dp_score = np.argmax(gene_dot_products, axis=1), np.max(gene_dot_products, axis=1)
    # update bleed matrix
    good = prob_score > prob_threshold
    bleed_matrix = compute_bleed_matrix(spot_colours[good], prob_mode[good], gene_codes, n_dyes)

    # add all information to the reference spots notebook page
    nbp_ref_spots.intensity = np.median(np.max(spot_colours, axis=-1), axis=-1)
    nbp_ref_spots.dot_product_gene_no, nbp_ref_spots.dot_product_gene_score = dp_mode.astype(np.int16), dp_score
    nbp_ref_spots.probability_gene_no, nbp_ref_spots.probability_gene_score = prob_mode.astype(np.int16), prob_score
    nbp_ref_spots.probability_gene_no_initial, nbp_ref_spots.probability_gene_score_initial = (
        prob_mode_initial.astype(np.int16), prob_score_initial)

    # add all information to the call spots notebook page
    nbp.gene_names, nbp.gene_codes = gene_names, gene_codes
    nbp.target_scale, nbp.homogeneous_scale = target_scale, homogeneous_scale
    nbp.colour_norm_factor = colour_norm_factor_initial * target_scale[None, :, :] * homogeneous_scale
    nbp.free_bled_codes, nbp.free_bled_codes_tile_independent = free_bled_codes, free_bled_codes_tile_indep
    nbp.bled_codes = bled_codes
    nbp.bleed_matrix_raw, nbp.bleed_matrix_initial, nbp.bleed_matrix = (raw_bleed_matrix, bleed_matrix_initial,
                                                                        bleed_matrix)

    return nbp, nbp_ref_spots

