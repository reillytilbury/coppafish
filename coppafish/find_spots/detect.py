from typing import Optional

import numpy as np
import scipy
import torch


def detect_spots(
    image: torch.Tensor,
    intensity_thresh: float,
    remove_duplicates: bool = False,
    radius_xy: Optional[int] = None,
    radius_z: Optional[int] = None,
) -> tuple[torch.Tensor]:
    """
    Spots are detected as local maxima on the given image above intensity_thresh.

    Args:
        - image (`(im_y x im_x x im_z) tensor[int or float]`): image to detect the local maxima.
        - intensity_thresh (float): local maxima are greater than intensity_thresh.
        - remove_duplicates (bool, optional): if two or more local maxima are close together, then only the greatest
            maxima value is detected. If they have identical intensities, one is chosen over the other. Default: false.
        - radius_xy (int, optional): two local maxima are considered close together if their distance along x and/or y
            is less than radius_xy. Default: not given.
        - radius_z (int, optional): two local maxima are considered close together if their distance along z is less
            than radius_xy. Default: not given.

    Returns:
        - `(n_spots x 3) tensor[int or float]` maxima_yxz: y, x, and z coordinate positions of local maxima.
        - `(n_spots) tensor[image.dtype]` maxima_intensity: maxima_intensity[i] is the image intensity at maxima_yxz[i].
    """
    assert type(image) is torch.Tensor
    assert type(intensity_thresh) is float
    assert type(remove_duplicates) is bool
    assert image.ndim == 3
    if remove_duplicates:
        assert type(radius_xy) is int, "radius_xy must be an int if remove_duplicates is true"
        assert type(radius_z) is int, "radius_z must be an int if remove_duplicates is true"
        assert radius_xy > 0
        assert radius_z > 0

    # (n_spots x 3) coordinate positions of the image local maxima.
    maxima_locations = (image > intensity_thresh).nonzero().int()
    maxima_intensities = image[tuple(maxima_locations.T)]
    if remove_duplicates:
        maxima_locations_norm = maxima_locations.numpy().astype(np.float32)
        maxima_locations_norm[:, 2] *= radius_xy / radius_z
        kdtree = scipy.spatial.KDTree(maxima_locations_norm)
        # Gives a list for each maxima that contains a list of indices that are nearby neighbours, including itself.
        pairs = kdtree.query_ball_tree(kdtree, r=radius_xy)
        # TODO: Can this be vectorised?
        keep_maxima = torch.zeros(maxima_locations.shape[0], dtype=bool)
        for i, i_pairs in enumerate(pairs):
            if len(i_pairs) == 1:
                keep_maxima[i] = True
                continue
            if keep_maxima[i_pairs].any():
                # A near neighbour has already been kept.
                continue
            if (maxima_intensities[i] >= maxima_intensities[i_pairs]).all():
                keep_maxima[i] = True
        maxima_locations = maxima_locations[keep_maxima]
        maxima_intensities = maxima_intensities[keep_maxima]

    return maxima_locations, maxima_intensities
