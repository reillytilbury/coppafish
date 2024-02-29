import torch
import numpy as np
from typing import List


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
    # Convert variables from numpy arrays to pytorch tensors
    image = torch.asarray(image, dtype=torch.float64)
    filter = torch.asarray(filter, dtype=torch.float64)
    im_deconvolved = torch.real(torch.fft.ifftn(torch.fft.fftn(image) * filter))
    im_deconvolved = im_deconvolved[
        im_pad_shape[0] : -im_pad_shape[0], im_pad_shape[1] : -im_pad_shape[1], im_pad_shape[2] : -im_pad_shape[2]
    ]
    # Convert result back to a numpy array
    return im_deconvolved.numpy()
