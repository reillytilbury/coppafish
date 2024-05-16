from typing import List, Union

import numpy as np


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


def wiener_deconvolve(image: np.ndarray, im_pad_shape: List[int], filter: np.ndarray) -> np.ndarray:
    """
    This pads `image` so goes to median value of `image` at each edge. Then deconvolves using the given Wiener filter.

    Args:
        image: `int [n_im_y x n_im_x x n_im_z]`.
            Image to be deconvolved.
        im_pad_shape: `int [n_pad_y, n_pad_x, n_pad_z]`.
            How much to pad image in `[y, x, z]` directions.
        filter: `complex128 [n_im_y+2*n_pad_y, n_im_x+2*n_pad_x, n_im_z+2*n_pad_z]`.
            Wiener filter to use.

    Returns:
        `(n_im_y x n_im_x x n_im_z) ndarray[float]`: deconvolved image.
    """
    im_av = np.median(image[:, :, 0])
    image = np.pad(
        image,
        [(im_pad_shape[i], im_pad_shape[i]) for i in range(len(im_pad_shape))],
        "linear_ramp",
        end_values=[(im_av, im_av)] * 3,
    )
    im_deconvolved = np.real(np.fft.ifftn(np.fft.fftn(image) * filter))
    im_deconvolved = im_deconvolved[
        im_pad_shape[0] : -im_pad_shape[0], im_pad_shape[1] : -im_pad_shape[1], im_pad_shape[2] : -im_pad_shape[2]
    ]
    return im_deconvolved
