import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional


def get_next_best_gene(
    consider_pixels: npt.NDArray[np.bool_],
    residual_pixel_colours: npt.NDArray[np.float32],
    all_bled_codes: npt.NDArray[np.float32],
    coefficients: npt.NDArray[np.float32],
    genes_added: npt.NDArray[np.int16],
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
        all_bled_codes (`(n_genes x (n_rounds * n_channels)) ndarray`): bled codes for each gene in the dataset. Each
            pixel should be made up of a superposition of the different bled codes.
        coefficients (`(im_y x im_x x im_z x n_genes_added) ndarray`): OMP coefficients (weights) computed from the
            previous iteration of OMP.
        genes_added (`(im_y x im_x x im_z x n_genes_added) ndarray`): the genes that have already been assigned to the
            pixels.
        score_threshold (float): a gene assignment with a dot product score below this value is not assigned to the
            pixel and OMP iterations will stop for the pixel.
        alpha (float): an OMP weighting parameter.
        background_genes (`(n_channels) ndarray`): indices of bled codes that correspond to background genes. If a
            pixel is assigned a background gene, then OMP iterations stop.
        background_variance(`(im_y x im_x x im_z x (n_rounds * n_channels)) ndarray`): background genes contribute to
            variance, this is given here for each pixel.

    Returns:
        - (`(im_y x im_x x im_z) ndarray`) best_gene: the best gene to add for each pixel. A pixel is given a np.nan
            value if consider_pixels is false for the given pixel.
        - (`(im_y x im_x x im_z) ndarray`) pass_threshold: true if the next gene given passes the thresholds. This is
            false if consider_pixels is false for the given pixel.
        - (`(im_y x im_x x im_z x (n_rounds * n_channels)) ndarray`) inverse_variance: the reciprocal of the variance
            for each round/channel based on the genes fit to the pixel.
    """
    pass


def weight_selected_genes(
    consider_pixels: npt.NDArray[np.bool_],
    bled_codes: npt.NDArray[np.float32],
    pixel_colours: npt.NDArray[np.float32],
    genes: npt.NDArray[np.int_],
    weight: Optional[npt.NDArray[np.float32]] = None,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Finds how to best weight the given genes to describe the seen pixel colour. Done for every given pixel individually.

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
            least squares. np.nan for any pixel that is not computed on.
        - (`(im_y x im_x x im_z x (n_rounds * n_channels)) ndarray[float32]`) residuals: pixel colours left after removing
            bled codes with computed coefficients. np.nan for any pixel that is not computed on.
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

    genes_flattened = genes.reshape((-1, n_rounds_channels))
    # Flatten to be n_pixels x (n_rounds * n_channels).
    weight_flattened = weight.reshape((-1, n_rounds_channels))
    # Flatten and weight the bled codes, becomes shape n_pixels x n_rounds_channels x n_genes_added.
    bled_codes_weighted = bled_codes[:, genes_flattened].swapaxes(0, 1) * weight_flattened[..., np.newaxis]
    # Flatten to be n_pixels x (n_rounds * n_channels).
    pixel_colours_flattened = pixel_colours.reshape((-1, n_rounds_channels))
    pixel_colours_flattened = pixel_colours_flattened * weight_flattened
    consider_pixels_flattened = consider_pixels.reshape(-1)

    coefficients = np.zeros((n_pixels, n_genes_added), dtype=np.float32)
    residuals = np.zeros((n_pixels, n_rounds_channels), dtype=np.float32)
    coefficients[:] = np.nan
    residuals[:] = np.nan
    for p in np.where(consider_pixels_flattened)[0]:
        pixel_colour = pixel_colours_flattened[p]
        gene = genes_flattened[p]
        w = weight_flattened[p]
        coefficients[p] = np.linalg.lstsq(bled_codes[:, gene] * w[:, np.newaxis], pixel_colour * w, rcond=-1)[0]
    residuals[consider_pixels_flattened] = (
        pixel_colours_flattened[consider_pixels_flattened]
        - np.matmul(
            bled_codes_weighted[consider_pixels_flattened], coefficients[consider_pixels_flattened, :, np.newaxis]
        )[..., 0]
    )
    residuals[consider_pixels_flattened] /= weight_flattened[consider_pixels_flattened]
    coefficients = coefficients.reshape(image_shape + (n_genes_added,))
    residuals = residuals.reshape(image_shape + (n_rounds_channels,))

    return coefficients.astype(np.float32), residuals.astype(np.float32)
