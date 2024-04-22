import tqdm
import scipy
import numpy as np
import numpy.typing as npt
from typing_extensions import assert_type
from typing import Tuple, Optional, List

from .. import log, call_spots


NO_GENE_SELECTION = -32768


def compute_omp_coefficients(
    pixel_colours: npt.NDArray[np.float32],
    bled_codes: npt.NDArray[np.float32],
    maximum_iterations: int,
    background_coefficients: npt.NDArray[np.float32],
    background_codes: npt.NDArray[np.float32],
    dot_product_threshold: float,
    dot_product_norm_shift: float,
    weight_coefficient_fit: bool,
    alpha: float,
    beta: float,
):
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

    Returns:
        - (`((im_y * im_x * im_z) x n_genes) sparse csr_matrix`) pixel_coefficients: OMP
            coefficients for every pixel. Since most coefficients are zero, the results are stored as a sparse matrix.
            Flattening the image dimensions is done using numpy's reshape method for consistency.
    """
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
    pixel_colours = pixel_colours.reshape(image_shape + (n_rounds_use * n_channels_use,))
    bled_codes = bled_codes.reshape((n_genes, -1))
    background_codes = background_codes.reshape((n_channels_use, -1))

    all_bled_codes = np.concatenate((bled_codes, background_codes))
    # Background genes are placed at the end of the other gene bled codes
    background_genes = np.arange(n_genes, n_genes + n_channels_use)
    # Background variance will not change between iterations
    background_variance = (
        np.square(background_coefficients) @ np.square(all_bled_codes[background_genes]) * alpha + beta**2
    )

    iterate_on_pixels = np.ones(image_shape, dtype=bool)
    verbose = iterate_on_pixels.sum() > 1_000
    pixels_iterated: List[int] = []
    genes_added = np.full(image_shape + (0,), fill_value=NO_GENE_SELECTION, dtype=np.int16)
    genes_added_coefficients = np.zeros_like(genes_added, dtype=np.float32)
    coefficient_image = scipy.sparse.csr_matrix(np.zeros((np.prod(image_shape), n_genes), dtype=np.float32))

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
        iterate_on_pixels = np.logical_and(iterate_on_pixels, pass_threshold)
        genes_added = np.append(genes_added, best_genes[:, :, :, np.newaxis], axis=3)

        # Update coefficients for pixels with new a gene assignment and keep the residual pixel colour
        genes_added_coefficients, pixel_colours = weight_selected_genes(
            iterate_on_pixels,
            bled_codes.T,
            pixel_colours,
            genes_added,
            weight=np.sqrt(inverse_variance) if weight_coefficient_fit else None,
        )

        # Populate sparse matrix with the updated coefficient results
        # FIXME: This is a horribly unoptimised piece of code.
        n_pixels = coefficient_image.shape[0]
        print(f"{coefficient_image.shape=}")
        print(f"{genes_added_coefficients.shape=}")
        print(f"{genes_added.shape=}")
        print(f"{genes_added.max()=}")
        print(f"{genes_added.min()=}")
        for p in range(n_pixels):
            coefficient_image[p, genes_added.reshape(-1, i + 1)[p]] = genes_added_coefficients.reshape(-1, i + 1)[p, i]

    log.info(f"Pixels iterated on: {pixels_iterated}")
    return coefficient_image


def get_next_best_gene(
    consider_pixels: npt.NDArray[np.bool_],
    residual_pixel_colours: npt.NDArray[np.float32],
    all_bled_codes: npt.NDArray[np.float32],
    coefficients: npt.NDArray[np.float32],
    genes_added: npt.NDArray[np.int16],
    norm_shift: float,
    score_threshold: float,
    alpha: float,
    background_genes: npt.NDArray[np.int16],
    background_variance: npt.NDArray[np.float32],
) -> Tuple[npt.NDArray[np.int16], npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
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
    assert consider_pixels.ndim == 3
    assert residual_pixel_colours.ndim == 4
    assert all_bled_codes.ndim == 2
    assert coefficients.ndim == 4
    assert genes_added.ndim == 4
    assert coefficients.ndim == genes_added.ndim
    assert score_threshold >= 0
    assert alpha >= 0
    assert background_genes.ndim == 1
    assert background_variance.ndim == 4

    image_shape = residual_pixel_colours.shape[:3]
    n_pixels = residual_pixel_colours.shape[0] * residual_pixel_colours.shape[1] * residual_pixel_colours.shape[2]
    n_rounds_channels = residual_pixel_colours.shape[3]
    n_genes_added = coefficients.shape[3]
    n_genes = all_bled_codes.shape[1]

    # Flatten all shapes of type im_y x im_x x im_z into n_pixels
    consider_pixels_flattened = consider_pixels.reshape((n_pixels))
    residual_pixel_colours_flattened = residual_pixel_colours.reshape((n_pixels, n_rounds_channels))
    coefficients_flattened = coefficients.reshape((n_pixels, n_genes_added))
    genes_added_flattened = genes_added.reshape((n_pixels, n_genes_added))
    background_variance_flattened = background_variance.reshape((n_pixels, n_rounds_channels))
    # Ensure bled_codes are normalised for each gene
    all_bled_codes /= np.linalg.norm(all_bled_codes, axis=0, keepdims=True)

    # See Josh's OMP documentation for details about this exact equation
    inverse_variances_flattened = np.reciprocal(
        np.matmul((coefficients_flattened**2)[:, None], all_bled_codes.T[genes_added_flattened] ** 2 * alpha)[:, 0]
        + background_variance_flattened
    )
    # Do not assign any pixels to background genes or to already added genes. If this happens, then gene assignment
    # failed.
    ignore_genes = background_genes[np.newaxis].repeat(n_pixels, axis=0)
    ignore_genes = np.append(ignore_genes, genes_added_flattened, axis=1)
    n_genes_ignore = ignore_genes.shape[1]
    # Pick the best scoring one for each pixel
    all_gene_scores = np.full((n_pixels, n_genes), fill_value=np.nan, dtype=np.float32)
    all_gene_scores[consider_pixels_flattened] = call_spots.dot_product_score(
        residual_pixel_colours_flattened[consider_pixels_flattened],
        all_bled_codes.T,
        inverse_variances_flattened[consider_pixels_flattened],
        norm_shift,
    )[3]
    best_genes = np.full(n_pixels, fill_value=NO_GENE_SELECTION, dtype=np.int16)
    best_genes[consider_pixels_flattened] = np.argmax(np.abs(all_gene_scores[consider_pixels_flattened]), axis=1)
    best_scores = np.full(n_pixels, fill_value=np.nan, dtype=np.float32)
    best_scores[consider_pixels_flattened] = all_gene_scores[
        consider_pixels_flattened, best_genes[consider_pixels_flattened]
    ]
    consider_pixels_flattened[consider_pixels_flattened] *= np.logical_not(
        (
            best_genes[consider_pixels_flattened][:, None].repeat(n_genes_ignore, axis=1)
            == ignore_genes[consider_pixels_flattened]
        ).any(1)
    )
    # The score is considered zero if the assigned gene is in ignore_genes
    best_scores[~consider_pixels_flattened] = 0
    best_genes[~consider_pixels_flattened] = NO_GENE_SELECTION
    genes_passing_score = np.zeros(n_pixels, dtype=bool)
    genes_passing_score[consider_pixels_flattened] = np.abs(best_scores[consider_pixels_flattened]) > score_threshold
    del best_scores, all_gene_scores

    return (
        best_genes.reshape(image_shape),
        genes_passing_score.reshape(image_shape),
        inverse_variances_flattened.reshape(image_shape + (n_rounds_channels,)),
    )


def weight_selected_genes(
    consider_pixels: npt.NDArray[np.bool_],
    bled_codes: npt.NDArray[np.float32],
    pixel_colours: npt.NDArray[np.float32],
    genes: npt.NDArray[np.int16],
    weight: Optional[npt.NDArray[np.float32]] = None,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Finds how to best weight the given genes to describe the pixel colours. Done for every given pixel individually.

    Args:
        consider_pixels (`(im_y x im_x x im_z) ndarray`): true for pixels to compute on.
        bled_codes (`((n_rounds * n_channels) x n_genes) ndarray`): bled code for every gene in every sequencing
            round/channel.
        pixel_colours (`(im_y x im_x x im_z x (n_rounds * n_channels)) ndarray`): pixel colour for every sequencing
            round/channel.
        genes (`(im_y x im_x x im_z x n_genes_added) ndarray`): the indices of the genes selected for each image pixel.
        weight (`(im_y x im_x x im_z x (n_rounds * n_channels)) ndarray`, optional): the weight is applied to every
            round/channel when computing coefficients. Default: no weighting.

    Returns:
        - (`(im_y x im_x x im_z x n_genes_added) ndarray[float32]`) coefficients: OMP coefficients computed through
            least squares. Set to 0 for any pixel that is not computed on.
        - (`(im_y x im_x x im_z x (n_rounds * n_channels)) ndarray[float32]`) residuals: pixel colours left after removing
            bled codes with computed coefficients. Remains pixel colour for any pixel that is not computed on.
    """
    # This function used to be called fit_coefs and fit_coefs_weight
    assert bled_codes.ndim == 2
    assert pixel_colours.ndim == 4
    assert genes.ndim == 4
    if weight is None:
        weight = np.ones_like(pixel_colours)
    assert weight.ndim == 4
    assert pixel_colours.shape == weight.shape

    n_pixels = pixel_colours.shape[0] * pixel_colours.shape[1] * pixel_colours.shape[2]
    n_rounds_channels = pixel_colours.shape[3]
    n_genes_added = genes.shape[3]
    image_shape = pixel_colours.shape[:3]

    genes_flattened = genes.reshape((np.prod(image_shape), n_genes_added))
    # Flatten to be n_pixels x (n_rounds * n_channels).
    weight_flattened = weight.reshape((-1, n_rounds_channels))
    # Flatten to be n_pixels x (n_rounds * n_channels).
    pixel_colours_flattened = pixel_colours.reshape((-1, n_rounds_channels))
    pixel_colours_flattened = pixel_colours_flattened * weight_flattened
    consider_pixels_flattened = consider_pixels.reshape(-1)
    # Flatten and weight the bled codes, becomes shape n_pixels x n_rounds_channels x n_genes_added.
    bled_codes_weighted = (
        bled_codes[:, genes_flattened[consider_pixels_flattened]].swapaxes(0, 1)
        * weight_flattened[consider_pixels_flattened, :, np.newaxis]
    )

    coefficients = np.zeros((n_pixels, n_genes_added), dtype=np.float32)
    residuals = np.zeros((n_pixels, n_rounds_channels), dtype=np.float32)
    residuals = pixel_colours_flattened
    coefficients[:] = 0
    for p in np.where(consider_pixels_flattened)[0]:
        pixel_colour = pixel_colours_flattened[p]
        gene = genes[p]
        w = weight_flattened[p]
        coefficients[p] = np.linalg.lstsq(bled_codes[:, gene] * w[:, np.newaxis], pixel_colour * w, rcond=-1)[0]
    residuals[consider_pixels_flattened] = (
        pixel_colours_flattened[consider_pixels_flattened]
        - np.matmul(bled_codes_weighted, coefficients[consider_pixels_flattened, :, np.newaxis])[..., 0]
    )
    residuals[consider_pixels_flattened] /= weight_flattened[consider_pixels_flattened]
    coefficients = coefficients.reshape(image_shape + (n_genes_added,))
    residuals = residuals.reshape(image_shape + (n_rounds_channels,))

    return coefficients.astype(np.float32), residuals.astype(np.float32)
