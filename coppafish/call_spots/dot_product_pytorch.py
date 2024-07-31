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
    # Normalise spot colours
    spot_colours_norm = spot_colours / (torch.linalg.norm(spot_colours, dim=1)[:, None] + norm_shift)

    # z-score spot colours and bled codes by dividing by the standard deviation
    spot_colours_z_scored = spot_colours_norm / torch.sqrt(variance) # [n_spots, n_r * n_c]
    # repeat spot colours for each gene
    spot_colours_z_scored = spot_colours_z_scored[:, None, :].repeat(1, bled_codes.shape[0], 1) # [n_spots, n_genes, n_r * n_c]
    bled_codes_z_scored = bled_codes[None, :, :] / torch.sqrt(variance)[:, None, :] # [n_spots, n_genes, n_r * n_c]

    # Now we can obtain the dot product score for each spot and each gene
    all_score = (torch.sum(spot_colours_z_scored * bled_codes_z_scored, dim=2) /
                 torch.sum(bled_codes_z_scored ** 2, dim=2))
    gene_no = torch.argmax(all_score, dim=1)
    all_score_sorted = torch.sort(all_score, dim=1)[0]
    gene_score = all_score_sorted[:, -1]
    gene_score_second = all_score_sorted[:, -2]

    return gene_no, gene_score, gene_score_second, all_score
