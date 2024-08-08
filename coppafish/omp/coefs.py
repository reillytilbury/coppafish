from typing import Tuple

import numpy as np
import scipy
import torch


NO_GENE_ASSIGNMENT: int = -32_768


def compute_omp(
    pixel_colours: np.ndarray[np.float16],
    bled_codes: torch.Tensor,
    background_codes: torch.Tensor,
    maximum_iterations: int,
    dot_product_threshold: float,
    normalisation_shift: float,
    pixel_subset_count: int,
) -> scipy.sparse.lil_matrix:
    """
    Compute OMP coefficients/gene selections from all pixel colours.

    Args:
        - pixel_colours (`(n_pixels x n_rounds_use x n_channels_use) ndarray[float16]`): pixel intensity in each
            sequencing round and channel.
        - bled_codes (`(n_genes x n_rounds_use x n_channels_use) tensor[float]`): every gene bled code.
        - background_codes (`(n_channels_use x n_rounds_use x n_channels_use) tensor[float]`): the background bled
            codes. These are simply uniform brightness in one channel for all rounds. background_codes[0] is the first
            code, background_codes[1] is the second code, etc.
        - maximum_iterations (int): the maximum number of gene assignments allowed for one pixel.
        - dot_product_threshold (float): a gene must have a dot product score above this value on the residual spot
            colour to be assigned the gene. If more than one gene is above this threshold, the top score is used.
        - normalisation_shift (float): during OMP each gene assignment iteration, the residual spot colour is
            normalised by dividing by its L2 norm + normalisation_shift. At the end of the computation, the final
            coefficients are divided by the L2 norm of the pixel colour + normalisation_shift.
        - pixel_subset_count (int): the maximum number of pixels to compute OMP on at a time. Used when memory limited.

    Returns:
        (`(n_pixels x n_genes) scipy.sparse.lil_matrix[float32]`) coefficients: each gene coefficient for every pixel.
            Most values will be zero, so kept in a sparse matrix.
    """
    pass


def get_next_gene_assignment(
    residual_colours: torch.Tensor,
    all_bled_codes: torch.Tensor,
    fail_gene_indices: torch.Tensor,
    dot_product_threshold: float,
    maximum_pass_count: int,
) -> torch.Tensor:
    """
    Get the next best gene assignment for each residual colour. Each gene is scored to each pixel using a dot product
    scoring. A pixel fails gene assignment if one or more of the conditions is met:

    - The top gene dot product score is below the dot_product_threshold.
    - The next best gene is in the fail_gene_indices list.
    - There are more than maximum_pass_count genes scoring above the dot_product_threshold.

    The reason for each of these conditions is:

    - Cut out dim background and bad gene reads.
    - Do not doubly assign a gene and avoid assigning background genes.
    - Cut out ambiguous colours.

    respectively.

    Args:
        - residual_colours (`(n_pixels x (n_rounds_use * n_channels_use)) tensor[float32]`): residual pixel colour.
        - all_bled_codes (`(n_genes_all x (n_rounds_use * n_channels_use)) tensor[float32]`): gene bled codes and
            background genes appended.
        - fail_gene_indices (`(n_pixels x n_genes_fail) tensor[int32]`): if the next gene assignment for a pixel is
            included on the list of gene indices, consider gene assignment a fail.
        - dot_product_threshold (float): a gene can only be assigned if the dot product score is above this threshold.
        - maximum_pass_count (int): if a pixel has more than maximum_pass_count dot product scores above the
            dot_product_threshold, then gene assignment has failed.

    Returns:
        (`(n_pixels) tensor[int32]`) next_best_genes: the next best gene assignment for each pixel. A value of -32_768
            is placed for pixels that failed to find a next best gene.
    """
    assert type(residual_colours) is torch.Tensor
    assert type(all_bled_codes) is torch.Tensor
    assert type(fail_gene_indices) is torch.Tensor
    assert type(dot_product_threshold) is float
    assert type(maximum_pass_count) is int
    assert residual_colours.ndim == 2
    assert all_bled_codes.ndim == 2
    assert fail_gene_indices.ndim == 2
    assert residual_colours.shape[0] > 0, "Require at least one pixel"
    assert residual_colours.shape[1] > 0, "Require at least one round/channel"
    assert residual_colours.shape[1] == all_bled_codes.shape[1]
    assert all_bled_codes.shape[0] > 0, "Require at least one bled code"
    assert fail_gene_indices.shape[0] == residual_colours.shape[0]
    assert (fail_gene_indices >= 0).all() and (fail_gene_indices < all_bled_codes.shape[0]).all()
    assert dot_product_threshold >= 0
    assert maximum_pass_count > 0

    # Matrix multiply (n_pixels x 1 x n_rounds_channel_use) normalised residual colours with
    # (1 x n_rounds_channels_use x n_genes_all) all bled codes
    # Gets (n_pixels x 1 x n_genes_all) all scores
    all_gene_scores = residual_colours[:, np.newaxis] @ all_bled_codes.T[np.newaxis]
    all_gene_scores = all_gene_scores[:, 0]
    _, next_best_genes = torch.max(all_gene_scores, dim=1)
    next_best_genes = next_best_genes.int()

    genes_passed = all_gene_scores > dot_product_threshold

    # A pixel only passes if the highest scoring gene is above the dot product threshold.
    pixels_passed = genes_passed.detach().clone().any(1)

    # Best gene in the fail_gene_indices causes a failed assignment.
    in_fail_gene_indices = (fail_gene_indices == next_best_genes[:, np.newaxis]).any(1)
    pixels_passed = pixels_passed & (~in_fail_gene_indices)

    # Too many high scoring genes on a single pixel causes a failed assignment.
    pixels_passed = pixels_passed & (genes_passed.sum(1) <= maximum_pass_count)

    next_best_genes[~pixels_passed] = NO_GENE_ASSIGNMENT

    return next_best_genes


def get_next_gene_coefficients(
    pixel_colours: torch.Tensor,
    bled_codes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gene coefficients for each given pixel colour by least squares with the gene bled codes.

    Args:
        - pixel_colours (`(n_pixels x n_rounds_channels_use x 1) tensor[float32]`): each pixels colour.
        - bled_codes (`(n_pixels x n_rounds_channels_use x n_genes_added) tensor[float32]`): the bled code for each
            added gene for each pixel.

    Returns:
        - (`(n_pixels x n_genes_added)`) coefficients: the computed coefficients for each gene.
        - (`(n_pixels x n_rounds_channels_use)`) residuals: the residual colour after subtracting off the assigned gene
            bled code weighted with their coefficients.
    """
    assert type(pixel_colours) is torch.Tensor
    assert type(bled_codes) is torch.Tensor
    assert pixel_colours.ndim == 3
    assert bled_codes.ndim == 3
    assert pixel_colours.shape[0] == bled_codes.shape[0]
    assert pixel_colours.shape[1] == bled_codes.shape[1]
    assert pixel_colours.shape[2] == 1
    assert bled_codes.shape[0] > 0, "Require at least one pixel to run on"
    assert bled_codes.shape[1] > 0, "Require at least one round and channel"
    assert bled_codes.shape[2] > 0, "Require at least one gene assigned"

    # Compute least squares for coefficients.
    # First parameter A has shape (n_pixels x n_rounds_channels_use x n_genes_added)
    # Second parameter B has shape (n_pixels x n_rounds_channels_use x 1)
    # So, the resulting coefficients has shape (n_pixels x n_genes_added x 1)
    # The least squares is minimising || A @ coefficients - B || ^ 2
    coefficients = torch.linalg.lstsq(bled_codes, pixel_colours, rcond=-1, driver="gels")[0]
    # Squeeze shape to (n_pixels x n_genes_added).
    coefficients = coefficients[..., 0]

    # From the new coefficients, find the spot colour residual.
    pixel_residuals = pixel_colours[..., 0] - (coefficients[:, np.newaxis] * bled_codes).sum(2)

    return coefficients, pixel_residuals
