from typing import List

import numpy as np


def central_tile(tilepos_yx: np.ndarray, use_tiles: List[int]) -> int:
    """
    returns tile in use_tiles closest to centre.

    Args:
        tilepos_yx: ```int [n_tiles x 2]```.
            tiff tile positions (index ```0``` refers to ```[0,0]```).
        use_tiles: ```int [n_use_tiles]```.
            Tiles used in the experiment.

    Returns:
        tile in ```use_tiles``` closest to centre.
    """
    mean_yx = np.round(np.mean(tilepos_yx, 0))
    nearest_t = np.linalg.norm(tilepos_yx[use_tiles] - mean_yx, axis=1).argmin()
    return int(use_tiles[nearest_t])


def compute_auto_thresh(image: np.ndarray, auto_thresh_multiplier: float, z_plane: int) -> float:
    """
    Calculate auto_thresh value used as an intensity threshold on image to detect spots on. It is calculated as
    `median(abs(image at z_plane)) * auto_thresh_multiplier`.

    Args:
        image (`(im_y x im_x x im_z) ndarray`): image pixel intensities.
        auto_thresh_multiplier (float): multiplier to increase auto_thresh value by.
        z_plane (int): z plane to compute on.

    Returns:
        float: auto_thresh value.
    """
    assert image.ndim == 3, "image must be three-dimensional"

    auto_thresh = np.median(np.abs(image[:, :, z_plane])) * auto_thresh_multiplier
    return float(auto_thresh)
