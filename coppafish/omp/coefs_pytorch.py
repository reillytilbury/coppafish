import torch
from typing import Tuple

from .. import call_spots
from .. import utils
from .. import spot_colors
from ..call_spots import dot_product_pytorch as dot_product
from ..setup import NotebookPage


def fit_coefs(
    bled_codes: torch.Tensor, pixel_colors: torch.Tensor, genes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This finds the least squared solution for how the `n_genes` `bled_codes` can best explain each `pixel_color`. Can
    also find weighted least squared solution if `weight` provided.

    Args:
        bled_codes (`((n_rounds * n_channels) x n_genes) tensor[float]`): flattened then transposed bled codes which
            usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors (`((n_rounds * n_channels) x n_pixels) tensor[float]` if `n_genes==1` otherwise
            `(n_rounds * n_channels) tensor[float]`): flattened then transposed pixel colors which usually has the
            shape `[n_pixels x n_rounds x n_channels]`.
        genes: `(n_pixels x n_genes_add) tensor[int]`: indices of codes in bled_codes to find coefficients for which
            best explain each pixel_color.

    Returns:
        - `(n_pixels x (n_rounds * n_channels)) tensor[float]`: residual pixel_colors after removing bled_codes with
            coefficients specified by coef.
        - (`(n_pixels x n_genes_add) tensor[float]` if n_genes == 1, `n_genes tensor[float]` if n_pixels == 1):
            coefficient found through least squares fitting for each gene.
    """
    n_pixels = pixel_colors.shape[1]
    residuals = torch.zeros((n_pixels, pixel_colors.shape[0]))

    # The arguments given are of shapes (n_pixels, (n_rounds * n_channels), n_genes_add) and
    # (n_pixels, (n_rounds * n_channels), 1). Pytorch then knows to batch over pixels
    # Coefs is shape (n_pixels, n_genes_add)
    coefs = torch.linalg.lstsq(
        bled_codes[:, genes].transpose(0, 1),
        pixel_colors.T[..., None],
        rcond=None,
        driver="gelss",
    )[0][:, :, 0]
    for p in range(n_pixels):
        residuals[p] = pixel_colors[:, p] - bled_codes[:, genes[p]] @ coefs[p]

    return residuals.type(torch.float32), coefs.type(torch.float32)


def fit_coefs_weight(
    bled_codes: torch.Tensor, pixel_colors: torch.Tensor, genes: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This finds the weighted least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes[s]`
    can best explain `pixel_colors[:, s]` for each pixel s. The `weight` indicates which rounds/channels should
    have more influence when finding the coefficients of each gene.

    Args:
        bled_codes (`((n_rounds * n_channels) x n_genes) tensor[float]`): flattened then transposed bled codes which
            usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors (`((n_rounds * n_channels) x n_pixels) tensor[float]`): flattened then transposed `pixel_colors`
            which usually has the shape `(n_pixels x n_rounds x n_channels)`.
        genes: `(n_pixels x n_genes_add) tensor[int]`: indices of codes in bled_codes to find coefficients for which
            best explain each pixel_color.
        weight: (`(n_pixels x (n_rounds * n_channels)) tensor[float]`: `weight[s, i]` is the weight to be applied to
            round_channel `i` when computing coefficient of each `bled_code` for pixel `s`.

    Returns:
        - residual (`(n_pixels x (n_rounds * n_channels)] tensor[float32]`): residual pixel_colors after removing
            bled_codes with coefficients specified by coefs.
        - coefs - (`[n_pixels x n_genes_add] tensor[float32]`): coefficients found through least squares fitting for
            each gene.
    """
    n_pixels, n_genes_add = genes.shape
    n_rounds_channels = bled_codes.shape[0]

    residuals = torch.zeros((n_pixels, n_rounds_channels), dtype=torch.float32)
    coefs = torch.zeros((n_pixels, n_genes_add), dtype=torch.float32)
    # (n_pixels, n_rounds_channels, n_genes_add)
    bled_codes_weighted = bled_codes[:, genes].swapaxes(0, 1) * weight[..., None]
    # (n_pixels, n_rounds_channels)
    pixel_colors_weighted = pixel_colors.T * weight
    coefs = torch.linalg.lstsq(bled_codes_weighted, pixel_colors_weighted, rcond=-1, driver="gelss")[0]
    for p in range(n_pixels):
        residuals[p] = pixel_colors_weighted[p] - torch.matmul(bled_codes_weighted[p], coefs[p])
    residuals = residuals / weight

    return residuals.type(torch.float32), coefs.type(torch.float32)


def get_best_gene_base(
    residual_pixel_colours: torch.Tensor,
    all_bled_codes: torch.Tensor,
    norm_shift: float,
    score_thresh: float,
    inverse_var: torch.Tensor,
    ignore_genes: torch.Tensor,
) -> Tuple[int, bool]:
    """
    Computes the `dot_product_score` between `residual_pixel_color` and each code in `all_bled_codes`. If `best_score`
    is less than `score_thresh` or if the corresponding `best_gene` is in `ignore_genes`, then `pass_score_thresh` will
    be False.

    Args:
        residual_pixel_colours (`(n_pixels x (n_rounds * n_channels)) ndarray[float]`): residual pixel colors from
            previous iteration of omp.
        all_bled_codes (`[n_genes x (n_rounds * n_channels)] ndarray[float]`): `bled_codes` such that `spot_color` of a
            gene `g` in round `r` is expected to be a constant multiple of `bled_codes[g, r]`. Includes codes of genes
            and background.
        norm_shift (float): shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh (float): `dot_product_score` of the best gene for a pixel must exceed this for that gene to be
            added in the current iteration.
        inverse_var (`(n_pixels x (n_rounds * n_channels)) ndarray[float]`): inverse of variance in each round/channel
            for each pixel based on genes fit on previous iteration. Used as `weight_squared` when computing
            `dot_product_score`.
        ignore_genes (`(n_genes_ignore) or (n_pixels x n_genes_ignore) ndarray[int]`): if `best_gene` is one of these,
            `pass_score_thresh` will be `False`. If no pixel axis, then the same genes are ignored for each pixel
            (useful for the first iteration of OMP edge case).

    Returns:
        - best_genes (n_pixels ndarray[int]): The best gene to add next for each pixel.
        - pass_score_threshes (n_pixels ndarray[bool]): `True` if `best_score > score_thresh` and `best_gene` not in
            `ignore_genes`.
    """
    assert residual_pixel_colours.ndim == 2, "`residual_pixel_colors` must be two dimensional"
    assert all_bled_codes.ndim == 2, "`all_bled_codes` must be two dimensional"
    assert inverse_var.ndim == 2, "`inverse_var` must be two dimensional"
    assert ignore_genes.ndim == 1 or ignore_genes.ndim == 2, "`ignore_genes` must be one or two dimensional"
    n_pixels = residual_pixel_colours.shape[0]
    if ignore_genes.ndim == 2:
        assert ignore_genes.shape[0] == n_pixels, "`ignore_genes` must have n_pixels in first axis if two dimensional"

    # Calculate score including background genes as if best gene is background, then stop iteration. all_scores has
    # shape (n_pixels, n_genes)
    all_scores = dot_product.dot_product_score(residual_pixel_colours, all_bled_codes, inverse_var, norm_shift)[3]
    # best_genes has shape (n_pixels, )
    best_genes = torch.argmax(torch.abs(all_scores), dim=1)
    # Take the best gene score for each pixel.
    best_scores = all_scores[range(n_pixels), best_genes]
    # If best_gene is in ignore_genes, set score below score_thresh, i.e. set the score to zero.
    if ignore_genes.ndim == 1:
        best_scores *= torch.isin(best_genes, ignore_genes, invert=True)
    else:
        # TODO: Vectorise this
        for p in range(n_pixels):
            best_scores[p] *= torch.isin(best_genes[p], ignore_genes[p], invert=True)
    pass_score_threshes = torch.abs(best_scores) > score_thresh
    return best_genes, pass_score_threshes
