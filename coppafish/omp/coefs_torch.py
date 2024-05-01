import tqdm
import torch
import scipy
import numpy as np
from typing_extensions import assert_type
from typing import Tuple, Optional, List

from .. import log
from ..call_spots import dot_product_pytorch


NO_GENE_SELECTION = -32768


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
    force_cpu: bool = False,
) -> scipy.sparse.lil_matrix:
    """
    Find OMP coefficients on all pixels.

    Args:
        pixel_colours (`(im_y x im_x x im_z x n_rounds_use x n_channels_use) ndarray`): pixel colours with all
            processing done already, like background gene fitting and pre-sequence subtraction.
        bled_codes (`(n_genes x n_rounds_use x n_channels_use) ndarray`): gene bled codes.
        maximum_iterations (int): the maximum number of unique genes that can be assigned to each pixel.
        background_coefficients (`(im_y x im_x x im_z x n_channels_use) ndarray`): background coefficients for each
            pixel.
        background_codes (`(n_channels_use x n_rounds_use x n_channels_use) ndarray`): each background gene code.
        dot_product_threshold (float): any dot product below this threshold is a failed gene assignment and the
            iterations stop for that pixel.
        dot_product_norm_shift (float): a shift applied during the dot product calculations to limit the boost of weak
            pixel intensities.
        weight_coefficient_fit (bool): apply Josh's OMP weighting, used to try and reduce the boost of large residuals
            after subtracting a gene assignment.
        alpha (float): OMP weighting parameter. Applied if weight_coefficient_fit is true.
        beta (float): OMP weighting parameter. Applied if weight_coefficient_fit is true.
        force_cpu (bool): force the computation to run on a CPU, even if a GPU is available.

    Returns:
        - (`((im_y * im_x * im_z) x n_genes) sparse lil_matrix`) pixel_coefficients: OMP
            coefficients for every pixel. Since most coefficients are zero, the results are stored as a sparse matrix.
            Flattening the image dimensions is done using numpy's reshape method for consistency.
    """
    device = torch.device("cpu")
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    assert_type(pixel_colours, torch.Tensor)
    assert_type(bled_codes, torch.Tensor)
    assert_type(background_coefficients, torch.Tensor)
    assert_type(background_codes, torch.Tensor)

    assert pixel_colours.ndim == 5
    assert bled_codes.ndim == 3
    assert maximum_iterations >= 1
    assert background_coefficients.ndim == 4
    assert background_codes.ndim == 3
    assert dot_product_threshold >= 0
    assert dot_product_norm_shift >= 0
    assert_type(weight_coefficient_fit, bool)
    assert alpha >= 0
    assert beta >= 0

    n_genes, n_rounds_use, n_channels_use = bled_codes.shape
    image_shape = pixel_colours.shape[:3]

    # Convert all n_rounds_use x n_channels_use shapes to n_rounds_use * n_channels_use
    pixel_colours = torch.reshape(pixel_colours, image_shape + (n_rounds_use * n_channels_use,))
    bled_codes = torch.reshape(bled_codes, (n_genes, -1))
    background_codes = torch.reshape(background_codes, (n_channels_use, -1))

    all_bled_codes = torch.concatenate((bled_codes, background_codes), dim=0)
    # Background genes are placed at the end of the other gene bled codes
    background_genes = torch.arange(n_genes, n_genes + n_channels_use)
    # Background variance will not change between iterations
    background_variance = (
        torch.square(background_coefficients) @ torch.square(all_bled_codes[background_genes]) * alpha + beta**2
    )

    iterate_on_pixels = torch.ones(image_shape, dtype=bool)
    verbose = iterate_on_pixels.sum() > 1_000
    pixels_iterated: List[int] = []
    genes_added = torch.full(image_shape + (0,), fill_value=NO_GENE_SELECTION, dtype=torch.int16)
    genes_added_coefficients = torch.zeros_like(genes_added).float()
    coefficient_image = scipy.sparse.lil_matrix(np.zeros((np.prod(image_shape), n_genes), dtype=np.float32))

    # Move all variables used in computation to the selected device.
    iterate_on_pixels = iterate_on_pixels.to(device=device)
    pixel_colours = pixel_colours.to(device=device)
    all_bled_codes = all_bled_codes.to(device=device)
    genes_added_coefficients = genes_added_coefficients.to(device=device)
    genes_added = genes_added.to(device=device)
    background_genes = background_genes.to(device=device)
    background_variance = background_variance.to(device=device)

    for i in tqdm.trange(maximum_iterations, desc="Computing OMP coefficients", unit="iteration", disable=not verbose):
        pixels_iterated.append(iterate_on_pixels.sum())
        best_genes, pass_threshold, inverse_variance = get_next_best_gene(
            iterate_on_pixels,
            pixel_colours,
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
        torch.logical_and(iterate_on_pixels, pass_threshold, out=iterate_on_pixels)
        genes_added = torch.cat((genes_added, best_genes[:, :, :, np.newaxis]), dim=3)

        # Update coefficients for pixels with new a gene assignment and keep the residual pixel colour
        genes_added_coefficients, pixel_colours = weight_selected_genes(
            iterate_on_pixels,
            bled_codes.T,
            pixel_colours,
            genes_added,
            weight=torch.sqrt(inverse_variance) if weight_coefficient_fit else None,
        )

        # FIXME: This is unoptimised.
        # Populate sparse matrix with the updated coefficient results
        genes_added_flattened = torch.reshape(genes_added, (-1, i + 1))
        genes_added_coefficients_flattened = torch.reshape(genes_added_coefficients, (-1, i + 1))
        log.debug("For loop over pixels started")
        for p in torch.where(genes_added_flattened[:, i] != NO_GENE_SELECTION)[0]:
            p_gene = genes_added_flattened[p, i]
            coefficient_image[p, p_gene.int()] = genes_added_coefficients_flattened[p, i].numpy()
        log.debug("For loop over pixels complete")

    iterate_on_pixels = iterate_on_pixels.cpu()
    pixel_colours = pixel_colours.cpu()
    all_bled_codes = all_bled_codes.cpu()
    genes_added_coefficients = genes_added_coefficients.cpu()
    genes_added = genes_added.cpu()
    background_genes = background_variance.cpu()

    if verbose:
        log.info(f"Pixels iterated on: {pixels_iterated}")

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
        consider_pixels (`(im_y x im_x x im_z) ndarray`): true for a pixel to compute on.
        residual_pixel_colours (`(im_y x im_x x im_z x (n_rounds * n_channels)) ndarray`): residual pixel colours, left
            over from any previous OMP iteration.
        all_bled_codes (`((n_rounds * n_channels) x n_genes) ndarray`): bled codes for each gene in the dataset. Each
            pixel should be made up of a superposition of the different bled codes.
        coefficients (`(im_y x im_x x im_z x n_genes_added) ndarray`): OMP coefficients (weights) computed from the
            previous iteration of OMP.
        genes_added (`(im_y x im_x x im_z x n_genes_added) ndarray`): the genes that have already been assigned to the
            pixels.
        norm_shift (float): shift to apply to normalisation of spot colours to limit the boost of weak spots.
        score_threshold (float): a gene assignment with a dot product score below this value is not assigned to the
            pixel and OMP iterations will stop for the pixel.
        alpha (float): an OMP weighting parameter.
        background_genes (`(n_channels) ndarray`): indices of bled codes that correspond to background genes. If a
            pixel is assigned a background gene, then OMP iterations stop.
        background_variance(`(im_y x im_x x im_z x (n_rounds * n_channels)) ndarray`): background genes contribute to
            variance, this is given here for each pixel.

    Returns:
        - (`(im_y x im_x x im_z) ndarray`) best_gene: the best gene to add for each pixel. A pixel is given a value of
            np.iinfo(np.int16).min if consider_pixels is false for the given pixel.
        - (`(im_y x im_x x im_z) ndarray`) pass_threshold: true if the next gene given passes the thresholds. This is
            false if consider_pixels is false for the given pixel.
        - (`(im_y x im_x x im_z x (n_rounds * n_channels)) ndarray`) inverse_variance: the reciprocal of the variance
            for each round/channel based on the genes fit to the pixel.
    """
    assert_type(consider_pixels, torch.Tensor)
    assert_type(residual_pixel_colours, torch.Tensor)
    assert_type(all_bled_codes, torch.Tensor)
    assert_type(coefficients, torch.Tensor)
    assert_type(genes_added, torch.Tensor)
    assert_type(background_genes, torch.Tensor)
    assert_type(background_variance, torch.Tensor)

    assert consider_pixels.dim() == 3
    assert residual_pixel_colours.dim() == 4
    assert all_bled_codes.dim() == 2
    assert coefficients.dim() == 4
    assert genes_added.dim() == 4
    assert coefficients.dim() == genes_added.dim()
    assert score_threshold >= 0
    assert alpha >= 0
    assert background_genes.dim() == 1
    assert background_variance.dim() == 4

    image_shape = tuple(residual_pixel_colours.shape[:3])
    n_pixels = residual_pixel_colours.shape[0] * residual_pixel_colours.shape[1] * residual_pixel_colours.shape[2]
    n_rounds_channels = residual_pixel_colours.shape[3]
    n_genes_added = coefficients.shape[3]
    n_genes = all_bled_codes.shape[1]

    # Flatten all shapes of type im_y x im_x x im_z into n_pixels
    consider_pixels_flattened = torch.reshape(consider_pixels, (n_pixels,))
    residual_pixel_colours_flattened = torch.reshape(residual_pixel_colours, (n_pixels, n_rounds_channels))
    coefficients_flattened = torch.reshape(coefficients, (n_pixels, n_genes_added))
    genes_added_flattened = torch.reshape(genes_added, (n_pixels, n_genes_added))
    background_variance_flattened = torch.reshape(background_variance, (n_pixels, n_rounds_channels))
    # Ensure bled_codes are normalised for each gene
    all_bled_codes /= torch.linalg.norm(all_bled_codes, dim=0, keepdim=True)

    # See Josh's OMP documentation for details about this exact equation
    inverse_variances_flattened = torch.zeros((n_pixels, n_rounds_channels)).float()
    inverse_variances_flattened[consider_pixels_flattened] = torch.reciprocal(
        torch.matmul(
            (coefficients_flattened[consider_pixels_flattened] ** 2)[:, None],
            all_bled_codes.T[genes_added_flattened[consider_pixels_flattened].int()] ** 2 * alpha,
        )[:, 0]
        + background_variance_flattened[consider_pixels_flattened]
    )
    # Do not assign any pixels to background genes or to already added genes. If this happens, then gene assignment
    # failed.
    ignore_genes = background_genes[np.newaxis].repeat_interleave(n_pixels, dim=0)
    ignore_genes = torch.cat((ignore_genes, genes_added_flattened), dim=1)
    n_genes_ignore = ignore_genes.shape[1]
    # Pick the best scoring one for each pixel
    all_gene_scores = torch.full((n_pixels, n_genes), fill_value=np.nan, dtype=torch.float32)
    all_gene_scores[consider_pixels_flattened] = dot_product_pytorch.dot_product_score(
        residual_pixel_colours_flattened[consider_pixels_flattened],
        all_bled_codes.T,
        inverse_variances_flattened[consider_pixels_flattened],
        norm_shift,
    )[3]
    best_genes = torch.full((n_pixels,), fill_value=NO_GENE_SELECTION, dtype=torch.int16)
    best_genes[consider_pixels_flattened] = torch.argmax(
        torch.abs(all_gene_scores[consider_pixels_flattened]), dim=1
    ).type(torch.int16)
    best_scores = torch.full((n_pixels,), fill_value=np.nan).float()
    best_scores[consider_pixels_flattened] = all_gene_scores[
        consider_pixels_flattened, best_genes[consider_pixels_flattened].int()
    ]
    consider_pixels_flattened[consider_pixels_flattened.clone()] *= torch.logical_not(
        torch.any(
            (
                best_genes[consider_pixels_flattened.clone()][:, None].repeat_interleave(n_genes_ignore, dim=1)
                == ignore_genes[consider_pixels_flattened.clone()]
            ),
            dim=1,
        ),
    )
    # The score is considered zero if the assigned gene is in ignore_genes
    best_scores[~consider_pixels_flattened] = 0
    best_genes[~consider_pixels_flattened] = NO_GENE_SELECTION
    genes_passing_score = torch.zeros((n_pixels,)).bool()
    genes_passing_score[consider_pixels_flattened] = torch.abs(best_scores[consider_pixels_flattened]) > score_threshold
    del best_scores, all_gene_scores

    best_genes = torch.reshape(best_genes, image_shape)
    genes_passing_score = torch.reshape(genes_passing_score, image_shape)
    inverse_variances_flattened = torch.reshape(inverse_variances_flattened, image_shape + (n_rounds_channels,))

    return (best_genes, genes_passing_score, inverse_variances_flattened)


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
        consider_pixels (`(im_y x im_x x im_z) tensor`): true for pixels to compute on.
        bled_codes (`((n_rounds * n_channels) x n_genes) tensor`): bled code for every gene in every sequencing
            round/channel.
        pixel_colours (`(im_y x im_x x im_z x (n_rounds * n_channels)) tensor`): pixel colour for every sequencing
            round/channel.
        genes (`(im_y x im_x x im_z x n_genes_added) tensor`): the indices of the genes selected for each image pixel.
        weight (`(im_y x im_x x im_z x (n_rounds * n_channels)) tensor`, optional): the weight is applied to every
            round/channel when computing coefficients. Default: no weighting.

    Returns:
        - (`(im_y x im_x x im_z x n_genes_added) tensor`) coefficients: OMP coefficients computed through
            least squares. Set to 0 for any pixel that is not computed on.
        - (`(im_y x im_x x im_z x (n_rounds * n_channels)) tensor`) residuals: pixel colours left after removing
            bled codes with computed coefficients. Remains pixel colour for any pixel that is not computed on.
    """
    # This function used to be called fit_coefs and fit_coefs_weight
    assert_type(consider_pixels, torch.Tensor)
    assert_type(bled_codes, torch.Tensor)
    assert_type(pixel_colours, torch.Tensor)
    assert_type(genes, torch.Tensor)
    assert_type(weight, torch.Tensor)

    assert bled_codes.dim() == 2
    assert pixel_colours.dim() == 4
    assert genes.dim() == 4
    if weight is None:
        weight = torch.ones_like(pixel_colours).float()
    assert weight.dim() == 4
    assert pixel_colours.shape == weight.shape

    n_pixels = pixel_colours.shape[0] * pixel_colours.shape[1] * pixel_colours.shape[2]
    n_rounds_channels = pixel_colours.shape[3]
    n_genes_added = genes.shape[3]
    image_shape = tuple(pixel_colours.shape[:3])

    genes_flattened = torch.reshape(genes, (np.prod(image_shape).item(), n_genes_added))
    # Flatten to be n_pixels x (n_rounds * n_channels).
    weight_flattened = torch.reshape(weight, (np.prod(image_shape).item(), n_rounds_channels))
    # Flatten to be n_pixels x (n_rounds * n_channels).
    pixel_colours_flattened = torch.reshape(pixel_colours, (np.prod(image_shape).item(), n_rounds_channels))
    pixel_colours_flattened = pixel_colours_flattened * weight_flattened
    consider_pixels_flattened = torch.reshape(consider_pixels, (np.prod(image_shape).item(),))
    # Flatten and weight the bled codes, becomes shape n_pixels x n_rounds_channels x n_genes_added.
    bled_codes_weighted = (
        bled_codes[:, genes_flattened[consider_pixels_flattened].int()].swapaxes(0, 1)
        * weight_flattened[consider_pixels_flattened, :, np.newaxis]
    )

    coefficients = torch.zeros((n_pixels, n_genes_added), dtype=torch.float32)
    coefficients[:] = 0

    coefficients[consider_pixels_flattened] = torch.linalg.lstsq(
        bled_codes[:, genes_flattened[consider_pixels_flattened].int()].swapaxes(0, 1)
        * weight_flattened[consider_pixels_flattened, :, np.newaxis],
        pixel_colours_flattened[consider_pixels_flattened] * weight_flattened[consider_pixels_flattened],
        rcond=-1,
    )[0]

    residuals = pixel_colours_flattened
    residuals[consider_pixels_flattened] = (
        pixel_colours_flattened[consider_pixels_flattened]
        - torch.matmul(bled_codes_weighted, coefficients[consider_pixels_flattened, :, np.newaxis])[..., 0]
    )
    residuals[consider_pixels_flattened] /= weight_flattened[consider_pixels_flattened]

    coefficients = torch.reshape(coefficients, image_shape + (n_genes_added,))
    residuals = torch.reshape(residuals, image_shape + (n_rounds_channels,))

    return coefficients.float(), residuals.float()
