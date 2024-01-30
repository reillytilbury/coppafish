import torch
from typing import Tuple

from .. import call_spots
from .. import utils
from .. import spot_colors
from ..call_spots import dot_product
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


# def fit_coefs_weight(
#     bled_codes: torch.Tensor, pixel_colors: torch.Tensor, genes: torch.Tensor, weight: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     This finds the weighted least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes[s]`
#     can best explain `pixel_colors[:, s]` for each pixel s. The `weight` indicates which rounds/channels should
#     have more influence when finding the coefficients of each gene.

#     Args:
#         bled_codes: `float [(n_rounds * n_channels) x n_genes]`.
#             Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
#         pixel_colors: `float [(n_rounds * n_channels) x n_pixels]`.
#             Flattened then transposed `pixel_colors` which usually has the shape `[n_pixels x n_rounds x n_channels]`.
#         genes: `int [n_pixels x n_genes_add]`.
#             Indices of codes in bled_codes to find coefficients for which best explain each pixel_color.
#         weight: `float [n_pixels x (n_rounds * n_channels)]`.
#             `weight[s, i]` is the weight to be applied to round_channel `i` when computing coefficient of each
#             `bled_code` for pixel `s`.

#     Returns:
#         - residual - `float32 [n_pixels x (n_rounds * n_channels)]`.
#             Residual pixel_colors after removing bled_codes with coefficients specified by coefs.
#         - coefs - `float32 [n_pixels x n_genes_add]`.
#             Coefficients found through least squares fitting for each gene.
#     """
#     n_pixels, n_genes_add = genes.shape
#     n_rounds_channels = bled_codes.shape[0]

#     residuals = torch.zeros((n_pixels, n_rounds_channels), dtype=torch.float32)
#     coefs = torch.zeros((n_pixels, n_genes_add), dtype=torch.float32)

#     for p in range(n_pixels):
#         pixel_colour = pixel_colors[:,p]
#         gene = genes[p]
#         w = weight[p]
#         coefs[p] = torch.linalg.lstsq(bled_codes[:, gene] * w[:, None], pixel_colour * w, rcond=-1)[0]
#         residuals[p] = pixel_colour * w - torch.matmul(bled_codes[:, gene] * w[:, None], coefs[p])
#     residuals = residuals / weight

#     return residuals.type(torch.float32), coefs.type(torch.float32)
