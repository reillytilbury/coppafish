import os
import numpy as np
from typing import Tuple, List


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

    auto_thresh = np.median(np.abs(image[:, :, z_plane])).clip(min=1) * auto_thresh_multiplier
    return float(auto_thresh)


def get_scale_from_txt(txt_file: str) -> Tuple[float, float]:
    """
    This checks whether `scale` and `scale_anchor` values used for producing npy files in *tile_dir* match
    values used and saved to `txt_file` on previous run.

    Will raise error if they are different.

    Args:
        txt_file: `nb.file_names.scale`, path to text file where scale values are saved.
            File contains two values, `scale` first and `scale_anchor` second.
            Values will be 0 if not used or not yet computed.
        scale: Value of `scale` used for current run of extract method i.e. `config['extract']['scale']`.
        scale_anchor: Value of `scale_anchor` used for current run of extract method
            i.e. `config['extract']['scale_anchor']`.
        tol: Two scale values will be considered the same if they are closer than this.

    Returns:
        scale - scale found in text file, None otherwise.
        scale_anchor - anchor scale found in text file, None otherwise.
    """
    scale_saved = None
    scale_anchor_saved = None
    if os.path.isfile(txt_file):
        scale_saved, scale_anchor_saved = np.genfromtxt(txt_file)
    return scale_saved, scale_anchor_saved


def save_scale(txt_file: str, scale: float, scale_anchor: float):
    """
    This saves `scale` and `scale_anchor` to `txt_file`. If text file already exists, it is overwritten.

    Args:
        txt_file: `nb.file_names.scale`, path to text file where scale values are to be saved.
        scale: Value of `scale`.
        scale_anchor: Value of `scale_anchor`.
    """
    np.savetxt(txt_file, [scale, scale_anchor], header="scale followed by scale_anchor")
