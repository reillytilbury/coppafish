from typing import Tuple

import torch


def dot_product_score(
    spot_colours: torch.Tensor, bled_codes: torch.Tensor, variance: torch.Tensor = None, norm_shift: float = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted dot product score between spot colours and bled codes. The weighting is determined by the variance of the
    spot colours, which is a function of the genes previously fit to the spot, and the coefficients of each of these
    genes.

    The dot product is calculated as the maximum likelihood estimate under a noise model which will be described in
    the documentation.

    Args:
        spot_colours (`[n_spots x (n_rounds * n_channels_use)] ndarray[float]`): spot colours.
        bled_codes (`[n_genes x (n_rounds * n_channels_use)] ndarray[float]`): normalised bled codes.
        variance (`[n_spots x (n_rounds * n_channels_use)] ndarray[float]`, optional): array of sigma_rc^2.
            Default: all ones.
        norm_shift (float, optional): added to the norm of each spot colour to avoid boosting weak spots too much.
            Default: 0.

    Returns:
        - gene_no: tensor of gene numbers [n_spots]
        - gene_score: tensor of gene scores [n_spots]
        - gene_score_second: torch.Tensor of second-best gene scores [n_spots]
        - `(n_spots x n_genes) tensor[float]`: `score` such that `score[d, c]` gives dot product between
            `spot_colours` vector `d` with `bled_codes` vector `c`.
    """
    # If no variance is provided, we assume all spots are equally reliable
    variance = torch.ones_like(spot_colours) if variance is None else variance
    concentration = torch.sqrt(torch.reciprocal(variance))
    del variance

    # Normalise spot colours
    spot_colours_norm = spot_colours / (torch.linalg.norm(spot_colours, dim=1)[:, None] + norm_shift)

    # Now we can obtain the dot product score for each spot and each gene (this is done in a vectorised way to avoid
    # storing intermediate results which would be very large)
    # score[s, g] = sum_i spot_colours_weighted[s, i] * bled_codes_weighted[g, i] / sum_i bled_codes_weighted[g, i] ** 2
    # where spot_colours_weighted[s, i] = spot_colours[s, i] / sqrt(variance[s, i]), and similarly
    # bled_codes_weighted[g, i] = bled_codes[g, i] / sqrt(variance[s, i])
    all_score = torch.sum(spot_colours_norm[:, None, :] * bled_codes[None, :, :] * concentration[:, None, :] ** 2,
                          dim=2) / torch.sum(bled_codes[None, :, :] ** 2 * concentration[:, None, :] ** 2, dim=2)
    gene_no = torch.argmax(all_score, dim=1)
    all_score_sorted = torch.sort(all_score, dim=1)[0]
    gene_score = all_score_sorted[:, -1]
    gene_score_second = all_score_sorted[:, -2]

    return gene_no, gene_score, gene_score_second, all_score
