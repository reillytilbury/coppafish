import math as maths
from typing import Optional, Tuple

import numpy as np
import torch

from .. import utils
from .. import log


def detect_spots(
    image: torch.Tensor,
    intensity_thresh: float,
    radius_xy: int,
    radius_z: int,
    remove_duplicates: bool = False,
    se: Optional[torch.Tensor] = None,
    force_cpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds local maxima in image exceeding `intensity_thresh`. This is achieved through a dilation being run on the
    whole image.

    Args:
        - image (`(n_y x n_x x n_z) tensor[Any]`): `image` to find spots on.
        - intensity_thresh (float-like): spots are local maxima in image with `pixel_value > intensity_thresh`.
        - radius_xy (int): radius of dilation structuring element in xy plane (approximately spot radius). The total
            length in y/x will be 2 * radius_xy - 1.
        - radius_z (int): radius of dilation structuring element in z direction (approximately spot radius). The total
            length in z will be 2 * radius_z - 1.
        - remove_duplicates: whether to only keep one pixel if two or more pixels are local maxima and have same
            intensity. Only works with integer image.
        - se: `int [se_sz_y x se_sz_x x se_sz_z]`.
            can give structuring element manually rather than using a cuboid element.
            Must only contain zeros and ones.
        - force_cpu (bool): use only the CPU for computation in pytorch.

    Returns:
        - `peak_yxz` - `int [n_peaks x image.ndim]`.
            yx or yxz location of spots found.
        - `peak_intensity` - `float [n_peaks]`.
            Pixel value of spots found.
    """
    assert type(image) is torch.Tensor
    assert image.dim() == 3
    assert type(float(intensity_thresh)) is float
    assert type(radius_xy) is int
    assert type(radius_z) is int
    assert type(remove_duplicates) is bool
    assert se is None or type(se) is torch.Tensor
    assert type(force_cpu) is bool

    se = np.ones((2 * radius_xy - 1, 2 * radius_xy - 1, 2 * radius_z - 1), dtype=int)
    pad_size_y = radius_xy - 1
    pad_size_x = radius_xy - 1
    pad_size_z = radius_z - 1
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

    run_on = torch.device("cpu")
    if torch.cuda.is_available() and not force_cpu:
        run_on = torch.device("cuda")

    n_consider = consider_yxz.shape[1]
    n_shifts = se_shifts.shape[1]
    paddings = (pad_size_z, pad_size_z, pad_size_x, pad_size_x, pad_size_y, pad_size_y)
    image = torch.nn.functional.pad(image, paddings, mode="constant", value=0)

    keep = torch.zeros((n_consider,), dtype=bool, device=run_on)
    image = image.to(run_on)
    se_shifts = se_shifts.to(run_on)

    # If there are too many potential spots in consider_yxz, then we must run through a subset at a time.
    n_consider_max = maths.floor(utils.system.get_available_memory(run_on) * 4.1e6 / n_shifts)
    for i in range(maths.ceil(n_consider / n_consider_max)):
        index_min, index_max = i * n_consider_max, (i + 1) * n_consider_max
        index_max = min(index_max, n_consider)
        n_consider_batch = index_max - index_min
        consider_yxz_batch = consider_yxz[:, index_min:index_max].to(run_on)
        consider_intensity_batch = consider_intensity[index_min:index_max].to(run_on)
        # Local pixel positions of spots must change after padding is added
        consider_yxz_se_shifted = (
            consider_yxz_batch + torch.asarray([pad_size_y, pad_size_x, pad_size_z], device=run_on)[:, np.newaxis]
        )
        del consider_yxz_batch
        # (image.ndim, n_consider, n_shifts) shape
        consider_yxz_se_shifted = torch.repeat_interleave(consider_yxz_se_shifted[..., np.newaxis], n_shifts, dim=2)
        consider_yxz_se_shifted += se_shifts[None].movedim((0, 1, 2), (1, 0, 2))
        # image.ndim items in tuple of `(n_consider * n_shifts) ndarray[int]`
        consider_yxz_se_shifted = tuple(consider_yxz_se_shifted.reshape((image.ndim, -1)))
        keep_batch = (
            image[consider_yxz_se_shifted].reshape((n_consider_batch, n_shifts))
            <= consider_intensity_batch[:, np.newaxis]
        ).all(1)
        keep[index_min:index_max] = keep_batch
        del consider_yxz_se_shifted, consider_intensity_batch, keep_batch

    keep = keep.cpu()
    image = image.cpu()

    peak_intensity = consider_intensity[keep].to(image.dtype)
    peak_yxz = consider_yxz.T[keep]

    return peak_yxz.to(dtype=torch.int32), peak_intensity
