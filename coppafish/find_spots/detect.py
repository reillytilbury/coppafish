import numpy as np
from typing import Optional, Tuple

from .. import utils
from .. import logging


def detect_spots(
    image: np.ndarray,
    intensity_thresh: float,
    radius_xy: Optional[int],
    radius_z: Optional[int] = None,
    remove_duplicates: bool = False,
    se: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds local maxima in image exceeding `intensity_thresh`.
    This is achieved through a dilation being run on the whole image.
    Should use for a large se.

    Args:
        image: `float [n_y x n_x x n_z]`.
            `image` to find spots on.
        intensity_thresh: Spots are local maxima in image with `pixel_value > intensity_thresh`.
        radius_xy: Radius of dilation structuring element in xy plane (approximately spot radius).
        radius_z: Radius of dilation structuring element in z direction (approximately spot radius).
            Must be more than 1 to be 3D.
            If `None`, 2D filter is used.
        remove_duplicates: Whether to only keep one pixel if two or more pixels are local maxima and have
            same intensity. Only works with integer image.
        se: `int [se_sz_y x se_sz_x x se_sz_z]`.
            Can give structuring element manually rather than using a cuboid element.
            Must only contain zeros and ones.

    Returns:
        - `peak_yxz` - `int [n_peaks x image.ndim]`.
            yx or yxz location of spots found.
        - `peak_intensity` - `float [n_peaks]`.
            Pixel value of spots found.
    """
    from .. import get_local_maxima

    if se is None:
        # Default is a cuboid se of all ones as is quicker than disk and very similar results.
        if radius_z is not None:
            se = np.ones((2 * radius_xy - 1, 2 * radius_xy - 1, 2 * radius_z - 1), dtype=int)
            pad_size_z = radius_z - 1
        else:
            se = np.ones((2 * radius_xy - 1, 2 * radius_xy - 1), dtype=int)
            pad_size_z = 0
        pad_size_y = radius_xy - 1
        pad_size_x = radius_xy - 1
    else:
        se = utils.morphology.ensure_odd_kernel(se)
        pad_size_y = int((se.shape[0] - 1) / 2)
        pad_size_x = int((se.shape[1] - 1) / 2)
        if se.ndim == 3:
            pad_size_z = int((se.shape[2] - 1) / 2)
        else:
            pad_size_z = 0
    if image.ndim == 2 and se.ndim == 3:
        mid_z = int(np.floor((se.shape[2] - 1) / 2))
        logging.warn(
            f"2D image provided but 3D filter asked for.\n" f"Using the middle plane ({mid_z}) of this filter."
        )
        se = se[:, :, mid_z]

    # set central pixel to 0
    se[np.ix_(*[(np.floor((se.shape[i] - 1) / 2).astype(int),) for i in range(se.ndim)])] = 0
    se_shifts = np.array(utils.morphology.filter.get_shifts_from_kernel(se))

    consider_yxz = np.where(image > intensity_thresh)
    n_consider = consider_yxz[0].shape[0]
    if remove_duplicates:
        # perturb image by small amount so two neighbouring pixels that did have the same value now differ slightly.
        # hence when find maxima, will only get one of the pixels not both.
        rng = np.random.default_rng(0)  # So shift is always the same.
        # rand_shift must be larger than small to detect a single spot.
        rand_im_shift = rng.uniform(low=2e-6, high=0.2, size=n_consider).astype(np.float32)
        image = image.astype(np.float32)
        image[consider_yxz] = image[consider_yxz] + rand_im_shift

    consider_intensity = image[consider_yxz]
    consider_yxz = np.array(consider_yxz)
    logging.debug(f"{consider_yxz.shape=}")
    if consider_yxz.max() <= np.iinfo(np.int32).max:
        consider_yxz = consider_yxz.astype(np.int32)
    # Sometimes consider_yxz can have too many spots in it to be run all at once through get_local_maxima without
    # running out of memory, so it is separated into smaller batches and then recombined after.
    max_batch_size = np.floor(5_000_000 * utils.system.get_available_memory() / 64.5).astype(int)
    logging.debug(f"{max_batch_size=}")
    paddings = np.array([pad_size_y, pad_size_x, pad_size_z])[: image.ndim]
    keep = np.zeros(n_consider, dtype=bool)
    final_i = np.ceil(n_consider / max_batch_size) - 1
    for i in range(np.ceil(n_consider / max_batch_size).astype(int)):
        index_start = i * max_batch_size
        if i == final_i:
            index_end = n_consider
        else:
            index_end = (i + 1) * max_batch_size
        consider_yxz_batch = consider_yxz[:, index_start:index_end]
        consider_intensity_batch = consider_intensity[index_start:index_end]

        keep[index_start:index_end] = get_local_maxima(
            image, se_shifts, paddings, consider_yxz_batch, consider_intensity_batch
        )
    if remove_duplicates:
        peak_intensity = np.round(consider_intensity[keep]).astype(int)
    else:
        peak_intensity = consider_intensity[keep]
    peak_yxz = consider_yxz.transpose()[keep]
    return peak_yxz.astype(int), peak_intensity


def get_local_maxima(
    image: np.ndarray,
    se_shifts: np.ndarray,
    pad_sizes: np.ndarray,
    consider_yxz: np.ndarray,
    consider_intensity: np.ndarray,
) -> np.ndarray:
    """
    Finds the local maxima from a given set of pixels to consider.

    Args:
        image (`(n_y x n_x x n_z) ndarray[float]`): `image` to find spots on.
        se_shifts (`(image.ndim x n_shifts) ndarray[int]`): y, x, z shifts which indicate neighbourhood about each spot
            where a local maxima search is carried out.
        pad_sizes ((image.ndim) ndarray[int]): `pad_sizes[i]` zeroes are added to the image on either end along the
            `i`th axis. `i=0,1,2` represents y, x and z respectively. This is done so that a spot near the image edge
            is not wrapped around when a local region is looked at.
        consider_yxz (`(image.ndim x n_consider) ndarray[int]`): all yxz coordinates where value in image is greater
            than an intensity threshold.
        consider_intensity (`[n_consider] ndarray[float]`): value of image at coordinates given by `consider_yxz`.

    Returns:
        `[n_consider] ndarray[bool]`: whether each point in `consider_yxz` is a local maxima or not.
    """
    n_consider = consider_yxz.shape[1]
    n_shifts = se_shifts.shape[1]

    image = np.pad(image, [(p, p) for p in pad_sizes], mode="constant", constant_values=0)
    # Local pixel positions of spots must change after padding is added
    consider_yxz_se_shifted = consider_yxz + pad_sizes[:, np.newaxis]
    # (image.ndim, n_consider, n_shifts) shape
    consider_yxz_se_shifted = np.repeat(consider_yxz_se_shifted[..., np.newaxis], se_shifts.shape[1], axis=2)
    consider_yxz_se_shifted += se_shifts[None].transpose((1, 0, 2))
    # image.ndim items in tuple of `(n_consider * n_shifts) ndarray[int]`
    consider_yxz_se_shifted = tuple(consider_yxz_se_shifted.reshape((image.ndim, -1)))
    consider_intensity = np.repeat(consider_intensity[:, np.newaxis], n_shifts, axis=1)
    keep = (image[consider_yxz_se_shifted].reshape((n_consider, n_shifts)) <= consider_intensity).all(1)

    return keep
