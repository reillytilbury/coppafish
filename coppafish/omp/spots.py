import tqdm
import numpy as np
from scipy.sparse import csr_matrix
import numpy_indexed
from typing import Union, List, Tuple, Optional

from .scores import score_coefficient_image, omp_scores_float_to_int
from .. import utils
from .. import log
from ..utils.spot_images import get_average_spot_image, get_spot_images
from ..find_spots import detect_spots, get_isolated_points


def count_spot_neighbours(
    image: np.ndarray, spot_yxz: np.ndarray, kernel: np.ndarray
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Counts the number of positive (and negative) pixels in a neighbourhood about each spot.
    If `filter` contains only 1 and 0, then number of positive pixels returned near each spot.
    If `filter` contains only -1 and 0, then number of negative pixels returned near each spot.
    If `filter` contains -1, 0 and 1, then number of positive and negative pixels returned near each spot.

    Args:
        image: `float [n_y x n_x (x n_z)]`.
            image spots were found on.
        spot_yxz: `int [n_spots x image.ndim]`.
            yx or yxz location of spots found.
        kernel: `int [filter_sz_y x filter_sz_x (x filter_sz_z)]`.
            Number of positive (and negative) pixels counted in this neighbourhood about each spot in image.
            Only contains values 0 and 1 (and -1).

    Returns:
        - n_pos_neighbours - `int [n_spots]` (Only if `filter` contains 1).
            Number of positive pixels around each spot in neighbourhood given by `pos_filter`.
        - n_neg_neighbours - `int [n_spots]` (Only if `filter` contains -1).
            Number of negative pixels around each spot in neighbourhood given by `neg_filter`.
    """
    # Correct for 2d cases where an empty dimension has been used for some variables.
    if all([image.ndim == spot_yxz.shape[1] - 1, np.max(np.abs(spot_yxz[:, -1])) == 0]):
        # Image 2D but spots 3D
        spot_yxz = spot_yxz[:, : image.ndim]
    if all([image.ndim == spot_yxz.shape[1] + 1, image.shape[-1] == 1]):
        # Image 3D but spots 2D
        image = np.mean(image, axis=image.ndim - 1)  # average over last dimension just means removing it.
    if all([image.ndim == kernel.ndim - 1, kernel.shape[-1] == 1]):
        # Image 2D but pos_filter 3D
        kernel = np.mean(kernel, axis=kernel.ndim - 1)

    # Check kernel contains right values.
    kernel_vals = np.unique(kernel)
    if not np.isin(kernel_vals, [-1, 0, 1]).all():
        log.error(ValueError("filter contains values other than -1, 0 or 1."))

    # Check that all spot positions are in the image bounds
    max_yxz = np.asarray(image.shape) - 1
    spot_yxz_out_of_bounds = spot_yxz[np.logical_or((spot_yxz < 0).any(1), (spot_yxz > max_yxz[None]).any(1))]
    if spot_yxz_out_of_bounds.size > 0:
        log.error(utils.errors.OutOfBoundsError("spot_yxz", spot_yxz_out_of_bounds[0], [0] * image.ndim, max_yxz))

    n_spots = spot_yxz.shape[0]
    zero_counts = np.asarray([0] * n_spots, dtype=int)
    if np.isin([-1, 1], kernel_vals).all():
        # Return positive and negative counts
        n_pos = utils.morphology.imfilter_coords(image > 0, kernel > 0, spot_yxz)
        n_neg = utils.morphology.imfilter_coords(image < 0, kernel < 0, spot_yxz)
        return n_pos, n_neg
    elif np.isin(-1, kernel_vals):
        # Return negative counts
        return zero_counts, utils.morphology.imfilter_coords(image < 0, kernel < 0, spot_yxz).astype(int)
    elif np.isin(1, kernel_vals):
        # Return positive counts
        return utils.morphology.imfilter_coords(image > 0, kernel > 0, spot_yxz).astype(int), zero_counts
    else:
        log.error(ValueError("filter contains only 0."))


def cropped_coef_image(
    pixel_yxz: np.ndarray, pixel_coefs: Union[csr_matrix, np.array]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Make cropped coef_image which is smallest possible image such that all non-zero pixel_coefs included.

    Args:
        pixel_yxz: `int [n_pixels x 3]`
            `pixel_yxz[s, :2]` are the local yx coordinates in `yx_pixels` for pixel `s`.
            `pixel_yxz[s, 2]` is the local z coordinate in `z_pixels` for pixel `s`.
        pixel_coefs: `float [n_pixels x 1]`.
            `pixel_coefs[s]` is the weighting of pixel `s` for a given gene found by the omp algorithm.
             Most are zero hence sparse form used.

    Returns:
        - coef_image - `float [im_size_y x im_size_x x im_size_z]`
            cropped omp coefficient.
            Will be `None` if there are no non-zero coefficients.
        - coord_shift - `int [3]`.
            yxz shift subtracted from pixel_yxz to build coef_image.
            Will be `None` if there are no non-zero coefficients.
    """
    if isinstance(pixel_coefs, csr_matrix):
        nz_ind = pixel_coefs.nonzero()[0]
        nz_pixel_coefs = pixel_coefs[nz_ind].toarray().flatten()
    else:
        nz_ind = pixel_coefs != 0
        nz_pixel_coefs = pixel_coefs[nz_ind]
    if nz_pixel_coefs.size == 0:
        # If no non-zero coefficients, return nothing
        return None, None
    else:
        nz_pixel_yxz = pixel_yxz[nz_ind, :]

        # shift nz_pixel_yxz so min is 0 in each axis so smaller image can be formed.
        coord_shift = nz_pixel_yxz.min(axis=0)
        nz_pixel_yxz = nz_pixel_yxz - coord_shift
        n_y, n_x, n_z = nz_pixel_yxz.max(axis=0) + 1

        # coef_image at pixels other than nz_pixel_yxz is set to 0.
        if n_z == 1:
            coef_image = np.zeros((n_y, n_x), dtype=np.float32)
        else:
            coef_image = np.zeros((n_y, n_x, n_z), dtype=np.float32)
        coef_image[tuple([nz_pixel_yxz[:, j] for j in range(coef_image.ndim)])] = nz_pixel_coefs
        return coef_image, coord_shift


def spot_neighbourhood(
    pixel_coefs: Union[csr_matrix, np.array],
    pixel_yxz: np.ndarray,
    spot_yxz: np.ndarray,
    spot_gene_no: np.ndarray,
    max_size: Union[np.ndarray, List],
    pos_neighbour_thresh: int,
    isolation_dist: float,
    z_scale: float,
    mean_sign_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the expected sign the coefficient should have in the neighbourhood about a spot.

    Args:
        pixel_coefs: `float [n_pixels x n_genes]`.
            `pixel_coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm.
             Most are zero hence sparse form used.
        pixel_yxz: `int [n_pixels x 3]`.
            `pixel_yxz[s, :2]` are the local yx coordinates in `yx_pixels` for pixel `s`.
            `pixel_yxz[s, 2]` is the local z coordinate in `z_pixels` for pixel `s`.
        spot_yxz: `int [n_spots x 3]`.
            `spot_yxz[s, :2]` are the local yx coordinates in `yx_pixels` for spot `s`.
            `spot_yxz[s, 2]` is the local z coordinate in `z_pixels` for spot `s`.
        spot_gene_no: `int [n_spots]`.
            `spot_gene_no[s]` is the gene that this spot is assigned to.
        max_size: `int [3]`.
            max YXZ size of spot shape returned. Zeros at extremities will be cropped in `av_spot_image`.
        pos_neighbour_thresh: For spot to be used to find av_spot_image, it must have this many pixels
            around it on the same z-plane that have a positive coefficient.
            If 3D, also, require 1 positive pixel on each neighbouring plane (i.e. 2 is added to this value).
            Typical = 9.
        isolation_dist: Spots are isolated if nearest neighbour (across all genes) is further away than this.
            Only isolated spots are used to find av_spot_image.
        z_scale: Scale factor to multiply z coordinates to put them in units of yx pixels.
            I.e. `z_scale = pixel_size_z / pixel_size_yx` where both are measured in microns.
            typically, `z_scale > 1` because `z_pixels` are larger than the `yx_pixels`.
        mean_sign_thresh: If the mean absolute coefficient sign is less than this in a region near a spot,
            we set the expected coefficient in av_spot_image to be 0.

    Returns:
        - av_spot_image - `int8 [av_shape_y x av_shape_x x av_shape_z]`
            Expected sign of omp coefficient in neighbourhood centered on spot.
        - spot_indices_used - `int [n_spots_used]`.
            indices of spots in `spot_yxzg` used to make av_spot_image.
        - av_spot_image_float - `float [max_size[0] x max_size[1] x max_size[2]]`
            Mean of omp coefficient sign in neighbourhood centered on spot.
    """
    # TODO: Maybe provide pixel_coef_sign instead of pixel_coef as less memory or use csr_matrix.
    n_pixels, n_genes = pixel_coefs.shape
    if not utils.errors.check_shape(pixel_yxz, [n_pixels, 3]):
        log.error(utils.errors.ShapeError("pixel_yxz", pixel_yxz.shape, (n_pixels, 3)))
    n_spots = spot_gene_no.shape[0]
    if not utils.errors.check_shape(spot_yxz, [n_spots, 3]):
        log.error(utils.errors.ShapeError("spot_yxz", spot_yxz.shape, (n_spots, 3)))

    n_z = pixel_yxz.max(axis=0)[2] + 1

    pos_filter_shape_yx = np.ceil(np.sqrt(pos_neighbour_thresh)).astype(int)
    if pos_filter_shape_yx % 2 == 0:
        # Shape must be odd
        pos_filter_shape_yx = pos_filter_shape_yx + 1
    if n_z <= 2:
        pos_filter_shape_z = 1
    else:
        pos_filter_shape_z = 3
    pos_filter = np.zeros((pos_filter_shape_yx, pos_filter_shape_yx, pos_filter_shape_z), dtype=int)
    pos_filter[:, :, np.floor(pos_filter_shape_z / 2).astype(int)] = 1
    if pos_filter_shape_z == 3:
        mid_yx = np.floor(pos_filter_shape_yx / 2).astype(int)
        pos_filter[mid_yx, mid_yx, 0] = 1
        pos_filter[mid_yx, mid_yx, 2] = 1

    max_size = np.array(max_size)
    if n_z == 1:
        max_size[2] = 1
    max_size_odd_loc = np.where(np.array(max_size) % 2 == 0)[0]
    if max_size_odd_loc.size > 0:
        max_size[max_size_odd_loc] += 1  # ensure shape is odd

    # get spot image centred on each spot.
    # Big image shape which will be cropped later.
    spot_images = np.zeros((0, *max_size), dtype=int)
    spots_used = np.zeros(n_spots, dtype=bool)
    for g in range(n_genes):
        use = spot_gene_no == g
        if use.any():
            # Note size of image will be different for each gene.
            coef_sign_image, coord_shift = cropped_coef_image(pixel_yxz, pixel_coefs[:, g])
            if coef_sign_image is None:
                # Go to next gene if no non-zero coefficients for this gene
                continue
            coef_sign_image = np.sign(coef_sign_image).astype(int)
            g_spot_yxz = spot_yxz[use] - coord_shift

            # Only keep spots with all neighbourhood having positive coefficient.
            n_pos_neighb, _ = count_spot_neighbours(coef_sign_image, g_spot_yxz, pos_filter)
            g_use = n_pos_neighb == pos_filter.sum()
            use[np.where(use)[0][np.invert(g_use)]] = False
            if coef_sign_image.ndim == 2:
                coef_sign_image = coef_sign_image[:, :, np.newaxis]
            if use.any():
                # nan_to_num sets nan to zero i.e. if out of range of coef_sign_image, coef assumed zero.
                # This is what we want as have cropped coef_sign_image to exclude zero coefficients.
                spot_images = np.append(
                    spot_images,
                    np.nan_to_num(get_spot_images(coef_sign_image, g_spot_yxz[g_use], max_size)).astype(int),
                    axis=0,
                )
                spots_used[use] = True

    if not spots_used.any():
        log.error(ValueError("No spots found to make average spot image from."))
    # Compute average spot image from all isolated spots
    isolated = get_isolated_points(spot_yxz[spots_used] * [1, 1, z_scale], isolation_dist)
    # get_average below ignores the nan values.
    av_spot_image = get_average_spot_image(spot_images[isolated].astype(float), "mean", "annulus_3d")
    av_spot_image_float = av_spot_image.copy()
    spot_indices_used = np.where(spots_used)[0][isolated]

    # Where mean sign is low, set to 0.
    av_spot_image[np.abs(av_spot_image) < mean_sign_thresh] = 0
    av_spot_image = np.sign(av_spot_image).astype(np.int8)

    if np.sum(av_spot_image == 1) == 0:
        log.warn(
            f"In av_spot_image, no pixels have a value of 1.\n"
            f"Maybe mean_sign_thresh = {mean_sign_thresh} is too high."
        )
    if np.sum(av_spot_image == 0) == 0:
        log.warn(
            f"In av_spot_image, no pixels have a value of 0.\n"
            f"Maybe mean_sign_thresh = {mean_sign_thresh} is too low."
        )

    return av_spot_image, spot_indices_used, av_spot_image_float


def get_spots(
    pixel_coefs: Union[csr_matrix, np.ndarray],
    pixel_yxz: np.ndarray,
    radius_xy: int,
    radius_z: Optional[int],
    high_coef_bias: float,
    spot_score_thresh: float,
    coef_thresh: float = 0,
    spot_shape: Optional[np.ndarray] = None,
    spot_shape_float: Optional[np.ndarray] = None,
    spot_yxzg: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Finds all local maxima in `coef_image` of each gene with coefficient exceeding `coef_thresh`
    and returns corresponding `yxz` position and `gene_no`.
    If provide `spot_shape`, also counts number of positive and negative pixels in neighbourhood of each spot.

    Args:
        pixel_coefs: `float [n_pixels x n_genes]`.
            `pixel_coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm.
             Most are zero hence sparse form used.
        pixel_yxz: `int [n_pixels x 3]`.
            `pixel_yxz[s, :2]` are the local yx coordinates in `yx_pixels` for pixel `s`.
            `pixel_yxz[s, 2]` is the local z coordinate in `z_pixels` for pixel `s`.
        radius_xy: Radius of dilation structuring element in xy plane (approximately spot radius).
        radius_z: Radius of dilation structuring element in z direction (approximately spot radius).
            If `None`, 2D filter is used.
        high_coef_bias (float): how much emphasis to put on large coefficients in the OMP spot scoring.
        coef_thresh: Local maxima in `coef_image` exceeding this value are considered spots.
        spot_shape: `int [shape_size_y x shape_size_x x shape_size_z]` or `None`.
            Indicates expected sign of coefficients in neighbourhood of spot.
            1 means expected positive coefficient.
            -1 means expected negative coefficient.
            0 means unsure of expected sign so ignore.
        spot_shape_float (`(shape_size_y x shape_size_x x shape_size_z) ndarray[float]`, optional): the mean spot
            shape, which spot_shape was found using. Default: not given.
        spot_score_thresh (float): Only spots with a score greater than this value will be kept if `spot_shape` is
            given.
        spot_yxzg: `float [n_spots x 4]`.
            Can provide location and gene identity of spots if already computed.
            Where spots are local maxima above `coef_thresh` in `pixel_coefs` image for each gene.
            If None, spots are determined from `pixel_coefs`.
            `spot_yxzg[s, :2]` are the local yx coordinates in `yx_pixels` for spot `s`.
            `spot_yxzg[s, 2]` is the local z coordinate in `z_pixels` for spot `s`.
            `spot_yxzg[s, 3]` is the gene number of spot `s`.

    Returns:
        - spot_yxz - `int [n_spots x 3]`
            `spot_yxz[s, :2]` are the local yx coordinates in `yx_pixels` for spot `s`.
            `spot_yxz[s, 2]` is the local z coordinate in `z_pixels` for spot `s`.
        - spot_gene_no - `int [n_spots]`.
            `spot_gene_no[s]` is the gene that spot s is assigned to.
        - `(n_spots) ndarray[int]` (Only given if `spot_shape` is not None).
            gene score for each spot exceeding the thresholds. The scores are between 0 and 1, so they are multiplied
            by `np.iinfo(int).max` and rounded to the nearest integer.
    """
    n_pixels, n_genes = pixel_coefs.shape
    if not utils.errors.check_shape(pixel_yxz, [n_pixels, 3]):
        log.error(utils.errors.ShapeError("pixel_yxz", pixel_yxz.shape, (n_pixels, 3)))

    if spot_shape is None or spot_shape_float is None:
        spot_info = np.zeros((0, 4), dtype=int)
    else:
        if np.sum(spot_shape == 1) == 0:
            log.error(
                ValueError(
                    f"spot_shape contains no pixels with a value of 1 which indicates the "
                    f"neighbourhood about a spot where we expect a positive coefficient."
                )
            )
        spot_info = np.zeros((0, 5), dtype=int)

    if spot_yxzg is not None:
        # check pixel coefficient is positive for random subset of 500 spots.
        rng = np.random.RandomState(0)
        spots_to_check = rng.choice(range(spot_yxzg.shape[0]), np.clip(500, 0, spot_yxzg.shape[0]), replace=False)
        pixel_index = numpy_indexed.indices(pixel_yxz, spot_yxzg[spots_to_check, :3].astype(pixel_yxz.dtype))
        spot_coefs_check = pixel_coefs[pixel_index, spot_yxzg[spots_to_check, 3]]
        if spot_coefs_check.min() <= coef_thresh:
            bad_spot = spots_to_check[spot_coefs_check.argmin()]
            log.error(
                ValueError(
                    f"spot_yxzg provided but gene {spot_yxzg[bad_spot, 3]} coefficient for spot {bad_spot}\n"
                    f"at yxz = {spot_yxzg[bad_spot, :3]} is {spot_coefs_check.min()} \n"
                    f"whereas it should be more than coef_thresh = {coef_thresh} as it is listed as a spot."
                )
            )
        del spots_to_check, pixel_index
    for g in tqdm.trange(
        n_genes, desc=f"Finding{' and scoring' * (spot_shape is not None)} OMP spots for all genes", unit="gene"
    ):
        log.debug(f"Finding and scoring spots {g=} started")
        # shift nzg_pixel_yxz so min is 0 in each axis so smaller image can be formed.
        # Note size of image will be different for each gene.
        coef_image, coord_shift = cropped_coef_image(pixel_yxz, pixel_coefs[:, g])
        if coef_image is None:
            # If no non-zero coefficients, go to next gene
            continue
        image_dims = coef_image.ndim
        if spot_yxzg is None:
            spot_yxz = detect_spots(coef_image, coef_thresh, radius_xy, radius_z, False)[0]
        else:
            # spot_yxz match pixel_yxz so if crop pixel_yxz need to crop spot_yxz too.
            spot_yxz = spot_yxzg[spot_yxzg[:, 3] == g, :image_dims] - coord_shift[:image_dims]
        if spot_yxz.shape[0] == 0:
            continue
        if spot_shape is None or spot_shape_float is None:
            keep = np.ones(spot_yxz.shape[0], dtype=bool)
            spot_info_g = np.zeros((keep.sum(), 4), dtype=int)
        else:
            # Score every detected OMP spot.
            coef_image_scores = score_coefficient_image(
                coef_image[..., np.newaxis], spot_shape, spot_shape_float, high_coef_bias
            )
            spot_scores = coef_image_scores[..., 0][tuple(spot_yxz.T)].copy()
            del coef_image_scores
            keep = spot_scores > spot_score_thresh
            log.debug(f"For gene {g}, keeping {keep.sum()} out of {keep.size} spots")
            if keep.sum() == 0:
                log.warn(f"Out of {keep.size} spots, gene {g} had no kept spots due to scores <{spot_score_thresh}")
                continue
            spot_scores = omp_scores_float_to_int(spot_scores)
            spot_info_g = np.zeros((keep.sum(), 5), dtype=int)
            spot_info_g[:, 4] = spot_scores[keep]
            del spot_scores
        del coef_image
        spot_info_g[:, :image_dims] = spot_yxz[keep]
        del spot_yxz
        spot_info_g[:, :3] = spot_info_g[:, :image_dims] + coord_shift[np.newaxis, :image_dims]  # shift spot_yxz back
        del coord_shift
        spot_info_g[:, 3] = g
        spot_info = np.append(spot_info, spot_info_g, axis=0)
        del spot_info_g, keep
        log.debug(f"Finding and scoring spots {g=} complete")

    if spot_shape is None:
        return spot_info[:, :3], spot_info[:, 3]
    else:
        return spot_info[:, :3], spot_info[:, 3], spot_info[:, 4]
