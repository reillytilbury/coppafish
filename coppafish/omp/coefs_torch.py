import math as maths
from typing import Optional, Tuple

import numpy as np
import scipy
import torch

from .. import utils
from ..call_spots import dot_product_pytorch


NO_GENE_SELECTION = -32768


def non_linear_function_coefficients(coefficients: torch.Tensor, a: float) -> torch.Tensor:
    """
    Non-linear function element-wise coefficients using the following function:

    x / (x + a) for x > 0.
    0 for x <= 0.

    Args:
        - coefficients (`tensor`): OMP coefficient image. Can be any shape.
        - a (float): a constant.

    Returns:
        (`(im_y x im_x x im_z) tensor`) function_coefficients: functioned coefficients.
    """
    assert type(coefficients) is torch.Tensor
    assert type(a) is float

    result = coefficients.detach().clone()
    positives = coefficients > 0
    result[~positives] = 0
    result[positives] /= result[positives] + a
    return result


def compute_omp_coefficients(
    pixel_colours: torch.Tensor,
    bled_codes: torch.Tensor,
    maximum_iterations: int,
    background_coefficients: torch.Tensor,
    background_codes: torch.Tensor,
    dot_product_threshold: float,
    dot_product_norm_shift: float,
    weight_coefficient_fit: bool,
    alpha: float,
    beta: float,
    do_not_compute_on: Optional[torch.Tensor] = None,
    force_cpu: bool = True,
) -> torch.Tensor:
    """
    Find OMP coefficients on all pixels.

    Args:
        pixel_colours (`(n_pixels x (n_rounds_use * n_channels_use)) tensor`): pixel colours with all
            processing done already, like background gene fitting and pre-sequence subtraction.
        bled_codes (`(n_genes x n_rounds_use * n_channels_use) tensor`): gene bled codes.
        maximum_iterations (int): the maximum number of unique genes that can be assigned to each pixel.
        background_coefficients (`(n_pixels x n_channels_use) tensor`): background coefficients for each
            pixel.
        background_codes (`(n_channels_use x (n_rounds_use * n_channels_use)) tensor`): each background gene code.
        dot_product_threshold (float): any dot product below this threshold is a failed gene assignment and the
            iterations stop for that pixel.
        dot_product_norm_shift (float): a shift applied during the dot product calculations to limit the boost of weak
            pixel intensities.
        weight_coefficient_fit (bool): apply Josh's OMP weighting, used to try and reduce the boost of large residuals
            after subtracting a gene assignment.
        alpha (float): OMP weighting parameter. Applied if weight_coefficient_fit is true.
        beta (float): OMP weighting parameter. Applied if weight_coefficient_fit is true.
        do_not_compute_on (`(n_pixels) tensor[bool]`): if true on a pixel, the pixel is not computed on, every
            coefficient remains zero. Default: compute on every pixel.
        force_cpu (bool): force the computation to run on a CPU, even if a GPU is available.

    Returns:
        - (`(n_pixels x n_genes) tensor[float32]`) pixel_coefficients: OMP coefficients for every pixel.
    """
    assert type(pixel_colours) is torch.Tensor
    assert type(bled_codes) is torch.Tensor
    assert type(background_coefficients) is torch.Tensor
    assert type(background_codes) is torch.Tensor
    assert type(weight_coefficient_fit) is bool
    assert do_not_compute_on is None or type(do_not_compute_on) is torch.Tensor
    assert pixel_colours.dim() == 2
    assert bled_codes.dim() == 2
    assert maximum_iterations >= 1
    assert background_coefficients.ndim == 2
    assert background_codes.ndim == 2
    assert dot_product_threshold >= 0
    assert dot_product_norm_shift >= 0
    assert alpha >= 0
    assert beta >= 0
    if do_not_compute_on is None:
        do_not_compute_on = torch.zeros(pixel_colours.shape[0], dtype=bool)
    assert do_not_compute_on.dim() == 1
    assert do_not_compute_on.shape[0] == pixel_colours.shape[0]

    cpu = torch.device("cpu")
    run_on = cpu
    if not force_cpu and torch.cuda.is_available():
        run_on = torch.device("cuda")

    n_genes = bled_codes.shape[0]
    n_pixels = pixel_colours.shape[0]
    n_channels_use = background_codes.shape[0]

    all_bled_codes = torch.concatenate((bled_codes, background_codes), dim=0)
    # Background genes are placed at the end of the other gene bled codes
    background_genes = torch.arange(n_genes, n_genes + n_channels_use)
    # Background variance will not change between iterations
    background_variance = (
        torch.square(background_coefficients) @ torch.square(all_bled_codes[background_genes]) * alpha + beta**2
    )

    genes_added = torch.full((n_pixels, 0), fill_value=NO_GENE_SELECTION, dtype=torch.int16)
    residual_pixel_colours = pixel_colours.detach().clone()

    # Move all variables used in computation to the selected device.
    residual_pixel_colours = residual_pixel_colours.to(device=run_on)
    do_not_compute_on = do_not_compute_on.to(device=run_on)
    bled_codes = bled_codes.to(device=run_on)
    all_bled_codes = all_bled_codes.to(device=run_on)
    genes_added_coefficients = torch.zeros_like(genes_added).float().to(device=run_on)
    genes_added = genes_added.to(device=run_on)
    background_genes = background_genes.to(device=run_on)
    background_variance = background_variance.to(device=run_on)

    # Run on every non-zero pixel colour.
    iterate_on_pixels = torch.logical_not(torch.isclose(pixel_colours, torch.asarray(0).float()).all(dim=1)).to(run_on)
    iterate_on_pixels = torch.logical_and(iterate_on_pixels, do_not_compute_on.logical_not_())
    # Start with a lil_matrix when populating results as this is faster than the csr matrix.
    coefficient_image = torch.zeros((n_pixels, n_genes), dtype=torch.float32)

    for i in range(maximum_iterations):
        best_genes, pass_threshold, inverse_variance = get_next_best_gene(
            iterate_on_pixels,
            residual_pixel_colours,
            all_bled_codes.T,
            genes_added_coefficients,
            genes_added,
            dot_product_norm_shift,
            dot_product_threshold,
            alpha,
            background_genes,
            background_variance,
        )

        # Update what pixels to continue iterating on
        iterate_on_pixels = torch.logical_and(iterate_on_pixels, pass_threshold).to(device=run_on)
        if iterate_on_pixels.sum() == 0:
            break
        genes_added = torch.cat((genes_added, best_genes[:, np.newaxis]), dim=1).to(device=run_on)

        # Update coefficients for pixels with new a gene assignment and keep the residual pixel colour.
        genes_added_coefficients, residual_pixel_colours = weight_selected_genes(
            iterate_on_pixels,
            bled_codes.T,
            pixel_colours.detach().clone().to(device=run_on),
            genes_added,
            weight=torch.sqrt(inverse_variance) if weight_coefficient_fit else None,
        )

        selected_pixels = torch.nonzero(genes_added[:, i] != NO_GENE_SELECTION, as_tuple=True)[0].cpu()
        for j in range(i + 1):
            selected_genes = genes_added[:, j][selected_pixels].cpu().int()
            coefficient_image[selected_pixels, selected_genes] = genes_added_coefficients[selected_pixels, j].cpu()

    return coefficient_image


def get_next_best_gene(
    consider_pixels: torch.Tensor,
    residual_pixel_colours: torch.Tensor,
    all_bled_codes: torch.Tensor,
    coefficients: torch.Tensor,
    genes_added: torch.Tensor,
    norm_shift: float,
    score_threshold: float,
    alpha: float,
    background_genes: torch.Tensor,
    background_variance: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the next "best gene" to add to each pixel based on their dot product scores with the bled code. If the next
    best gene is a background gene, already added to the pixel, or a score below the score threshold, then it has
    failed the passing and the pixel is not iterated through OMP any more.

    Args:
        consider_pixels (`(n_pixels) tensor`): true for a pixel to compute on.
        residual_pixel_colours (`(n_pixels x (n_rounds * n_channels)) tensor`): residual pixel colours, left over from
            previous OMP iteration.
        all_bled_codes (`((n_rounds * n_channels) x n_genes) tensor`): bled codes for each gene in the dataset. Each
            pixel should be made up of a superposition of the different bled codes.
        coefficients (`(n_pixels x n_genes_added) tensor`): OMP coefficients (weights) computed from the
            previous iteration of OMP.
        genes_added (`(n_pixels x n_genes_added) tensor`): the genes that have already been assigned to the
            pixels.
        norm_shift (float): shift to apply to normalisation of spot colours to limit the boost of weak spots.
        score_threshold (float): a gene assignment with a dot product score below this value is not assigned to the
            pixel and OMP iterations will stop for the pixel.
        alpha (float): an OMP weighting parameter.
        background_genes (`(n_channels) tensor`): indices of bled codes that correspond to background genes. If a
            pixel is assigned a background gene, then OMP iterations stop.
        background_variance(`(n_pixels x (n_rounds * n_channels)) tensor`): background genes contribute to
            variance, this is given here for each pixel.

    Returns:
        - (`(n_pixels) tensor`) best_gene: the best gene to add for each pixel. A pixel is given a value of
            np.iinfo(np.int16).min if consider_pixels is false for the given pixel.
        - (`(n_pixels) tensor`) pass_threshold: true if the next gene given passes the thresholds. This is
            false if consider_pixels is false for the given pixel.
        - (`(n_pixels x (n_rounds * n_channels)) tensor`) inverse_variance: the reciprocal of the variance
            for each round/channel based on the genes fit to the pixel.
    """
    assert type(consider_pixels) is torch.Tensor
    assert type(residual_pixel_colours) is torch.Tensor
    assert type(all_bled_codes) is torch.Tensor
    assert type(coefficients) is torch.Tensor
    assert type(genes_added) is torch.Tensor
    assert type(background_genes) is torch.Tensor
    assert type(background_variance) is torch.Tensor
    assert consider_pixels.dim() == 1
    assert residual_pixel_colours.dim() == 2
    assert all_bled_codes.dim() == 2
    assert coefficients.dim() == 2
    assert genes_added.dim() == 2
    assert coefficients.dim() == genes_added.dim()
    assert score_threshold >= 0
    assert alpha >= 0
    assert background_genes.dim() == 1
    assert background_variance.dim() == 2

    n_pixels, n_rounds_channels = residual_pixel_colours.shape

    # Ensure bled_codes are normalised for each gene
    all_bled_codes /= torch.linalg.norm(all_bled_codes, dim=0, keepdim=True)

    # See Josh's OMP documentation for details about this exact equation
    inverse_variances = torch.zeros((n_pixels, n_rounds_channels), dtype=torch.float32, device=coefficients.device)
    inverse_variances[consider_pixels] = torch.reciprocal(
        torch.matmul(
            (coefficients[consider_pixels] ** 2)[:, None],
            all_bled_codes.T[genes_added[consider_pixels].int()] ** 2 * alpha,
        )[:, 0]
        + background_variance[consider_pixels]
    )
    # Do not assign any pixels to background genes or to already added genes. If this happens, then gene assignment
    # failed.
    ignore_genes = background_genes[np.newaxis].repeat_interleave(n_pixels, dim=0)
    ignore_genes = torch.cat((ignore_genes, genes_added), dim=1)
    n_genes_ignore = ignore_genes.shape[1]
    # Pick the best scoring gene for each pixel.
    all_gene_scores = dot_product_pytorch.dot_product_score(
        residual_pixel_colours, all_bled_codes.T, inverse_variances, norm_shift
    )[3]
    best_genes = torch.full((n_pixels,), fill_value=NO_GENE_SELECTION, device=coefficients.device, dtype=torch.int16)
    best_genes[consider_pixels] = torch.argmax(torch.abs(all_gene_scores[consider_pixels]), dim=1).type(torch.int16)
    best_scores = torch.full((n_pixels,), fill_value=np.nan, device=coefficients.device, dtype=torch.float32)
    best_scores[consider_pixels] = all_gene_scores[consider_pixels, best_genes[consider_pixels].int()]
    consider_pixels[consider_pixels.clone()] *= torch.logical_not(
        torch.any(
            (
                best_genes[consider_pixels.clone()][:, None].repeat_interleave(n_genes_ignore, dim=1)
                == ignore_genes[consider_pixels.clone()]
            ),
            dim=1,
        ),
    )
    # The score is considered zero if the assigned gene is in ignore_genes
    best_scores[~consider_pixels] = 0
    best_genes[~consider_pixels] = NO_GENE_SELECTION
    genes_passing_score = torch.zeros((n_pixels,), device=coefficients.device, dtype=torch.bool)
    genes_passing_score[consider_pixels] = torch.abs(best_scores[consider_pixels]) > score_threshold
    del best_scores, all_gene_scores

    return (best_genes, genes_passing_score, inverse_variances)


def weight_selected_genes(
    consider_pixels: torch.Tensor,
    bled_codes: torch.Tensor,
    pixel_colours: torch.Tensor,
    genes: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds how to best weight the given genes to describe the pixel colours. Done for every given pixel individually.

    Args:
        consider_pixels (`(n_pixels) tensor`): true for pixels to compute on.
        bled_codes (`((n_rounds * n_channels) x n_genes) tensor`): bled code for every gene in every sequencing
            round/channel.
        pixel_colours (`(n_pixels x (n_rounds * n_channels)) tensor`): pixel colour for every sequencing
            round/channel.
        genes (`(n_pixels x n_genes_added) tensor`): the indices of the genes selected for each image pixel.
        weight (`(n_pixels x (n_rounds * n_channels)) tensor`, optional): the weight is applied to every
            round/channel when computing coefficients. Default: no weighting.

    Returns:
        - (`(n_pixels x n_genes_added) tensor`) coefficients: OMP coefficients computed through
            least squares. Set to 0 for any pixel that is not computed on.
        - (`(n_pixels x (n_rounds * n_channels)) tensor`) residuals: pixel colours left after removing
            bled codes with computed coefficients. Remains pixel colour for any pixel that is not computed on.
    """
    # This function used to be called fit_coefs and fit_coefs_weight
    assert type(consider_pixels) is torch.Tensor
    assert type(bled_codes) is torch.Tensor
    assert type(pixel_colours) is torch.Tensor
    assert type(genes) is torch.Tensor
    assert weight is None or type(weight) is torch.Tensor
    assert consider_pixels.dim() == 1
    assert bled_codes.dim() == 2
    assert pixel_colours.dim() == 2
    assert genes.dim() == 2
    if weight is None:
        weight = torch.ones_like(pixel_colours).float()
    assert weight.dim() == 2
    assert pixel_colours.shape == weight.shape

    n_pixels, _ = pixel_colours.shape
    n_genes_added = genes.shape[1]

    # Flatten and weight the bled codes, becomes shape n_pixels x n_rounds_channels x n_genes_added.
    bled_codes_weighted = (
        bled_codes[:, genes[consider_pixels].int()].swapaxes(0, 1) * weight[consider_pixels, :, np.newaxis]
    )

    coefficients = torch.zeros((n_pixels, n_genes_added), device=consider_pixels.device, dtype=torch.float32)
    coefficients[:] = 0

    coefficients[consider_pixels] = torch.linalg.lstsq(
        bled_codes[:, genes[consider_pixels].int()].swapaxes(0, 1) * weight[consider_pixels, :, np.newaxis],
        pixel_colours[consider_pixels] * weight[consider_pixels],
        rcond=-1,
    )[0]

    residuals = pixel_colours
    residuals[consider_pixels] = (
        pixel_colours[consider_pixels]
        - torch.matmul(bled_codes_weighted, coefficients[consider_pixels, :, np.newaxis])[..., 0]
    )
    residuals[consider_pixels] /= weight[consider_pixels]

    return coefficients.float(), residuals.float()
