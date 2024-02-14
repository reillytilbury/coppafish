import numpy as np
from typing import Optional, Tuple

from .. import logging


def get_filter_info(
    image: np.ndarray,
    auto_thresh_multiplier: float,
    hist_bin_edges: np.ndarray,
    max_pixel_value: int,
    scale: float,
    z_info: Optional[int] = None,
) -> Tuple[float, np.ndarray, int, float]:
    """
    Gets information from filtered scaled images useful for later in the pipeline.
    If 3D image, only z-plane used for `auto_thresh` and `hist_counts` calculation for speed and the that the
    exact value of these is not that important, just want a rough idea.

    Args:
        image: ```int [n_y x n_x (x n_z)]```
            Image of tile after filtering and scaling.
        auto_thresh_multiplier: ```auto_thresh``` is set to ```auto_thresh_multiplier * median(abs(image))```
            so that pixel values above this are likely spots. Typical = 10
        hist_bin_edges: ```float [len(nbp['hist_values']) + 1]```
            ```hist_values``` shifted by 0.5 to give bin edges not centres.
        max_pixel_value: Maximum pixel value that image can contain when saving as tiff file.
            If no shift was applied, this would be ```np.iinfo(np.uint16).max```.
        scale: Factor by which, ```image``` has been multiplied in order to fill out available values in tiff file.
        z_info: z-plane to get `auto_thresh` and `hist_counts` from.

    Returns:
        - ```auto_thresh``` - ```int``` Pixel values above ```auto_thresh``` in ```image``` are likely spots.
        - ```hist_counts``` - ```int [len(nbp['hist_values'])]```.
            ```hist_counts[i]``` is the number of pixels found in ```image``` with value equal to
            ```hist_values[i]```.
        - ```n_clip_pixels``` - ```int``` Number of pixels in ```image``` with value more than ```max_pixel_value```.
        - ```clip_scale``` - ```float``` Suggested scale factor to multiply un-scaled ```image``` by in order for
            ```n_clip_pixels``` to be 0.
    """
    if image.ndim == 3:
        if z_info is None:
            logging.error(ValueError("z_info not provided"))
        auto_thresh = np.median(np.abs(image[:, :, z_info])) * auto_thresh_multiplier
        hist_counts = np.histogram(image[:, :, z_info], hist_bin_edges)[0]
    else:
        auto_thresh = np.median(np.abs(image)) * auto_thresh_multiplier
        hist_counts = np.histogram(image, hist_bin_edges)[0]
    n_clip_pixels = np.sum(image > max_pixel_value)
    if n_clip_pixels > 0:
        # image has already been multiplied by scale hence inclusion of scale here
        # max_pixel_value / image.max() is less than 1 so recommended scaling becomes smaller than scale.
        clip_scale = scale * max_pixel_value / image.max()
    else:
        clip_scale = 0

    return np.round(auto_thresh).astype(int), hist_counts, n_clip_pixels, clip_scale
