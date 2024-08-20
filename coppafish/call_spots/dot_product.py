from typing import Tuple
import numpy as np


def dot_product_score(
    spot_colours: np.ndarray, bled_codes: np.ndarray, variance: np.ndarray = None, norm_shift: float = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple dot product score assigning each spot to the gene with the highest score.

    Args:
        spot_colours (`[n_spots x (n_rounds * n_channels_use)] ndarray[float]`): spot colours.
        bled_codes (`[n_genes x (n_rounds * n_channels_use)] ndarray[float]`): normalised bled codes.
        variance (`[n_spots x (n_rounds * n_channels_use)] ndarray[float]`, optional): array of variances. Default:
            all ones.
        norm_shift (float, optional): added to the norm of each spot colour to avoid boosting weak spots too much.
            Default: 0.

    Returns:
        - gene_no: np.ndarray of gene numbers [n_spots]
        - gene_score: np.ndarray of gene scores [n_spots]
        - gene_score_second: np.ndarray of second-best gene scores [n_spots]
        - `[n_spots x n_genes] ndarray[float]`: `score` such that `score[d, c]` gives dot product between
            `spot_colours` vector `d` with `bled_codes` vector `c`.
    """
    n_genes = bled_codes.shape[0]
    # If no variance is provided, we assume all spots are equally reliable
    variance = np.ones_like(spot_colours) if variance is None else variance
    # Normalise spot colours
    spot_colours = spot_colours / (np.linalg.norm(spot_colours, axis=1)[:, None] + norm_shift)

    # z-score spot colours and bled codes by dividing by the standard deviation
    # we want these to both have shape [n_spots, n_genes, n_rounds * n_channels_use]
    spot_colours_z_scored = np.repeat((spot_colours / np.sqrt(variance))[:, None, :], n_genes, axis=1)
    bled_codes_z_scored = bled_codes[None, :, :] / np.sqrt(variance)[:, None, :]

    # Now we can obtain the dot product score for each spot and each gene
    all_score = np.sum(spot_colours_z_scored * bled_codes_z_scored, axis=2) / np.sum(bled_codes_z_scored ** 2, axis=2)
    gene_no = np.argmax(all_score, axis=1)
    all_score_sorted = np.sort(all_score, axis=1)
    gene_score = all_score_sorted[:, -1]
    gene_score_second = all_score_sorted[:, -2]

    return gene_no, gene_score, gene_score_second, all_score


def gene_prob_score(spot_colours: np.ndarray, bled_codes: np.ndarray, kappa: float = 2) -> np.ndarray:
    """
    Probability model says that for each spot in a particular round, the normalised fluorescence vector follows a
    Von-Mises Fisher distribution with mean equal to the normalised fluorescence for each dye and concentration
    parameter kappa. Then invert this to get prob(dye | fluorescence) and multiply across rounds to get
    prob(gene | spot_colours).

    Args:
        spot_colours (`(n_spots x n_rounds x n_channels_use) ndarray`): spot colours.
        bled_codes (`(n_genes x n_rounds x n_channels_use) ndarray`): normalised bled codes.
        kappa (float, optional), scaling factor for dot product score. Default: 2.

    Returns:
        (`(n_spots x n_genes) ndarray[float]`): gene probabilities.
    """
    n_genes = bled_codes.shape[0]
    n_spots, n_rounds, n_channels_use = spot_colours.shape
    # First, normalise spot_colours so that for each spot s and round r, norm(spot_colours[s, r, :]) = 1
    spot_colours = spot_colours / np.linalg.norm(spot_colours, axis=2)[:, :, None]
    # Do the same for bled_codes
    bled_codes = bled_codes / np.linalg.norm(bled_codes, axis=2)[:, :, None]
    # Flip the sign of a single spot and round if spot_colours[s, r, c] < 0 for the greatest magnitude channel.
    spot_colours_reshaped = spot_colours.copy().reshape((-1, n_channels_use))
    negatives = np.take_along_axis(spot_colours_reshaped, np.argmax(spot_colours_reshaped, axis=1)[:, None], 1) < 0
    spot_colours[negatives.reshape((n_spots, n_rounds))] *= -1
    # At this point, reshape spot_colours to be [n_spots, n_rounds * n_channels_use] and bled_codes to be
    # [n_genes, n_rounds * n_channels_use]
    spot_colours = spot_colours.reshape((n_spots, -1))
    bled_codes = bled_codes.reshape((n_genes, -1))
    # Now we can compute the dot products of each spot with each gene, producing a matrix of shape [n_spots, n_genes]
    dot_product = spot_colours @ bled_codes.T
    probability = np.exp(kappa * dot_product)
    # Now normalise so that each row sums to 1
    probability = np.nan_to_num(probability / np.sum(probability, axis=1)[:, None])

    return probability
