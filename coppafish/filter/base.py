from typing import List, Union

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


def get_wiener_filter(psf: np.ndarray, image_shape: Union[np.ndarray, List[int]], constant: float) -> np.ndarray:
    """
    This tapers the psf so goes to 0 at edges and then computes wiener filter from it.

    Args:
        psf: `float [y_diameter x x_diameter x z_diameter]`.
            Average small image about a spot. Normalised so min is 0 and max is 1.
        image_shape: `int [n_im_y, n_im_x, n_im_z]`.
            Indicates the shape of the image to be convolved after padding.
        constant: Constant used in wiener filter.

    Returns:
        `complex128 [n_im_y x n_im_x x n_im_z]`. Wiener filter of same size as image.
    """
    # taper psf so smoothly goes to 0 at each edge.
    psf = (
        psf
        * np.hanning(psf.shape[0]).reshape(-1, 1, 1)
        * np.hanning(psf.shape[1]).reshape(1, -1, 1)
        * np.hanning(psf.shape[2]).reshape(1, 1, -1)
    )
    psf = psf_pad(psf, image_shape)
    psf_ft = np.fft.fftn(np.fft.ifftshift(psf))
    return np.conj(psf_ft) / np.real((psf_ft * np.conj(psf_ft) + constant))


def psf_pad(psf: np.ndarray, image_shape: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Pads psf with zeros so has same dimensions as image

    Args:
        psf: `float [y_shape x x_shape (x z_shape)]`.
            Point Spread Function with same shape as small image about each spot.
        image_shape: `int [psf.ndim]`.
            Number of pixels in `[y, x, (z)]` direction of padded image.

    Returns:
        `float [image_shape[0] x image_shape[1] (x image_shape[2])]`.
        Array same size as image with psf centered on middle pixel.
    """
    # must pad with ceil first so that ifftshift puts central pixel to (0,0,0).
    pre_pad = np.ceil((np.array(image_shape) - np.array(psf.shape)) / 2).astype(int)
    post_pad = np.floor((np.array(image_shape) - np.array(psf.shape)) / 2).astype(int)
    return np.pad(psf, [(pre_pad[i], post_pad[i]) for i in range(len(pre_pad))])
