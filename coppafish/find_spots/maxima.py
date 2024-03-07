import numpy as np


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
