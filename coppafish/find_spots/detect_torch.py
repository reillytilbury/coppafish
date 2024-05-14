import torch
import numpy as np
from typing_extensions import assert_type
from typing import Optional, Tuple

from .. import utils
from .. import log


def detect_spots(
    image: torch.Tensor,
    intensity_thresh: float,
    radius_xy: Optional[int],
    radius_z: Optional[int] = None,
    remove_duplicates: bool = False,
    se: Optional[torch.Tensor] = None,
    force_cpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        force_cpu (bool): use only the CPU for computation in pytorch.

    Returns:
        - `peak_yxz` - `int [n_peaks x image.ndim]`.
            yx or yxz location of spots found.
        - `peak_intensity` - `float [n_peaks]`.
            Pixel value of spots found.
    """
    assert_type(image, torch.Tensor)

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
        log.warn(f"2D image provided but 3D filter asked for.\n" f"Using the middle plane ({mid_z}) of this filter.")
        se = se[:, :, mid_z]

    # set central pixel to 0
    se[np.ix_(*[(np.floor((se.shape[i] - 1) / 2).astype(int),) for i in range(se.ndim)])] = 0
    se_shifts = torch.asarray(np.array(utils.morphology.filter.get_shifts_from_kernel(se)))

    consider_yxz = (image > intensity_thresh).nonzero(as_tuple=True)
    n_consider = consider_yxz[0].shape[0]
    if remove_duplicates:
        image = image.to(dtype=torch.float32)
        # perturb image by small amount so two neighbouring pixels that did have the same value now differ slightly.
        # hence when find maxima, will only get one of the pixels not both.
        rng = np.random.default_rng(0)  # So shift is always the same.
        # rand_shift must be larger than small to detect a single spot.
        rand_im_shift = torch.asarray(rng.uniform(low=2e-6, high=0.2, size=n_consider), dtype=image.dtype)
        image[consider_yxz] = image[consider_yxz] + rand_im_shift

    consider_intensity = image[consider_yxz]
    consider_yxz = torch.vstack(consider_yxz)
    if (consider_yxz <= np.iinfo(np.int32).max).all():
        consider_yxz = consider_yxz.to(torch.int32)

    cpu = torch.device("cpu")
    run_on = cpu
    if torch.cuda.is_available() and not force_cpu:
        run_on = torch.device("cuda")

    image = image.to(run_on)
    consider_intensity = consider_intensity.to(run_on)
    consider_yxz = consider_yxz.to(run_on)
    se_shifts = se_shifts.to(run_on)

    n_consider = consider_yxz.shape[1]
    n_shifts = se_shifts.shape[1]
    paddings = (pad_size_z, pad_size_z, pad_size_x, pad_size_x, pad_size_y, pad_size_y)

    image = torch.nn.functional.pad(image, paddings, mode="constant", value=0)
    # Local pixel positions of spots must change after padding is added
    consider_yxz_se_shifted = (
        consider_yxz + torch.asarray([pad_size_y, pad_size_x, pad_size_z], device=run_on)[:, np.newaxis]
    )
    # (image.ndim, n_consider, n_shifts) shape
    consider_yxz_se_shifted = torch.repeat_interleave(
        consider_yxz_se_shifted[..., np.newaxis], se_shifts.shape[1], dim=2
    )
    consider_yxz_se_shifted += se_shifts[None].movedim((0, 1, 2), (1, 0, 2))
    # image.ndim items in tuple of `(n_consider * n_shifts) ndarray[int]`
    consider_yxz_se_shifted = tuple(consider_yxz_se_shifted.reshape((image.ndim, -1)))
    keep = (image[consider_yxz_se_shifted].reshape((n_consider, n_shifts)) <= consider_intensity[:, np.newaxis]).all(1)

    if remove_duplicates:
        peak_intensity = torch.round(consider_intensity[keep]).to(int)
    else:
        peak_intensity = consider_intensity[keep]
    peak_yxz = consider_yxz.T[keep]

    return peak_yxz.to(dtype=torch.int32, device=cpu), peak_intensity.to(device=cpu)
