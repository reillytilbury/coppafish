import torch
from typing import Tuple


def dot_product_score(
    spot_colours: torch.Tensor, bled_codes: torch.Tensor, weight_squared: torch.Tensor = None, norm_shift: float = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simple dot product score assigning each spot to the gene with the highest score.

    Args:
        spot_colours (`[n_spots x (n_rounds * n_channels_use)] ndarray[float]`): spot colours.
        bled_codes (`[n_genes x (n_rounds * n_channels_use)] ndarray[float]`): normalised bled codes.
        weight_squared (`[n_spots x (n_rounds * n_channels_use)] ndarray[float]`, optional): array of weights. Default:
            all ones.
        norm_shift (float, optional): added to the norm of each spot colour to avoid boosting weak spots too much.
            Default: 0.

    Returns:
        - gene_no: tensor of gene numbers [n_spots]
        - gene_score: tensor of gene scores [n_spots]
        - gene_score_second: torch.Tensor of second-best gene scores [n_spots]
        - `(n_spots x n_genes) tensor[float]`: `score` such that `score[d, c]` gives dot product between
            `spot_colours` vector `d` with `bled_codes` vector `c`.
    """
    n_spots, n_rounds_channels_use = spot_colours.shape
    # If no weighting is given, use equal weighting
    if weight_squared is None:
        weight_squared = torch.ones((n_spots, n_rounds_channels_use))

    weight_squared = weight_squared / torch.sum(weight_squared, dim=1)[:, None]
    spot_colours = spot_colours / (torch.linalg.norm(spot_colours, dim=1)[:, None] + norm_shift)
    spot_colours = n_rounds_channels_use * spot_colours * weight_squared

    # Now we can obtain the dot product score for each spot and each gene
    all_score = spot_colours @ bled_codes.T
    gene_no = torch.argmax(all_score, dim=1)
    all_score_sorted = torch.sort(all_score, dim=1)[0]
    gene_score = all_score_sorted[:, -1]
    gene_score_second = all_score_sorted[:, -2]

    return gene_no, gene_score, gene_score_second, all_score
