import torch
import numpy as np


def get_local_maxima(
    image: np.ndarray,
    se_shifts: np.ndarray,
    pad_sizes: np.ndarray,
    consider_yxz: np.ndarray,
    consider_intensity: np.ndarray,
    force_cpu: bool = True,
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
        force_cpu (bool): force the use of CPU when computing maxima.

    Returns:
        `[n_consider] ndarray[bool]`: whether each point in `consider_yxz` is a local maxima or not.
    """
    cpu_device = torch.device("cpu")
    run_on_device = cpu_device
    if not force_cpu and torch.cuda.is_available():
        run_on_device = torch.device("cuda")

    n_consider = consider_yxz.shape[1]
    n_shifts = se_shifts.shape[1]

    image = np.pad(image, [(p, p) for p in pad_sizes], mode="constant", constant_values=0)
    # Local pixel positions of spots must change after padding is added
    consider_yxz_se_shifted = consider_yxz + pad_sizes[:, np.newaxis]
    image = torch.asarray(image, device=run_on_device)
    consider_yxz_se_shifted = torch.asarray(consider_yxz_se_shifted, device=run_on_device)
    se_shifts = torch.asarray(se_shifts, device=run_on_device)
    consider_intensity = torch.asarray(consider_intensity, device=run_on_device)
    # (image.ndim, n_consider, n_shifts) shape
    consider_yxz_se_shifted = consider_yxz_se_shifted[..., np.newaxis].repeat_interleave(se_shifts.shape[1], 2)
    consider_yxz_se_shifted += se_shifts[np.newaxis].swapaxes(0, 1)
    # image.ndim items in tuple of `(n_consider * n_shifts) ndarray[int]`
    consider_yxz_se_shifted = tuple(consider_yxz_se_shifted.reshape((image.ndim, -1)))
    consider_intensity = (consider_intensity[:, np.newaxis]).repeat_interleave(n_shifts, axis=1)
    keep = torch.all(image[consider_yxz_se_shifted].reshape((n_consider, n_shifts)) <= consider_intensity, dim=1)

    return keep.to(device=cpu_device).numpy()
