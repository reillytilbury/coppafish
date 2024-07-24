from typing import Tuple

import numpy as np
import torch


def wiener_deconvolve(
    image: np.ndarray, im_pad_shape: Tuple[int], filter: np.ndarray, force_cpu: bool = True
) -> np.ndarray:
    """
    This pads `image` so goes to median value of `image` at each edge. Then deconvolves using the given Wiener filter.

    Args:
        - `(n_im_y x n_im_x x n_im_z) ndarray[float]` image: image to be deconvolved.
        - (`tuple of three ints`) im_pad_shape: how much to pad the image in y, x, and z directions.
        - (`(n_im_y+2*n_pad_y, n_im_x+2*n_pad_x, n_im_z+2*n_pad_z) ndarray[complex128]`) filter: the Wiener filter to
            use.
        - (bool, optional) force_cpu: force the computation to run on the CPU only. Default: true.

    Returns:
        `(n_im_y x n_im_x x n_im_z) ndarray[float]`: deconvolved image.
    """
    assert type(image) is np.ndarray
    assert type(im_pad_shape) is tuple
    assert len(im_pad_shape) == 3
    assert type(filter) is np.ndarray
    assert type(force_cpu) is bool

    run_device = torch.device("cpu")
    if not force_cpu and torch.cuda.is_available():
        run_device = torch.device("cuda")

    im_av = np.median(image[:, :, 0])
    image = np.pad(
        image,
        [(im_pad_shape[i], im_pad_shape[i]) for i in range(len(im_pad_shape))],
        "linear_ramp",
        end_values=[(im_av, im_av)] * 3,
    )
    # Convert variables from numpy arrays to pytorch tensors
    image = torch.asarray(image, dtype=torch.complex64)
    filter = torch.asarray(filter, dtype=torch.complex64)
    image = image.to(run_device)
    filter = filter.to(run_device)
    im_deconvolved = torch.fft.fftn(image)
    im_deconvolved *= filter
    im_deconvolved = torch.fft.ifftn(im_deconvolved)
    im_deconvolved = torch.real(im_deconvolved)
    im_deconvolved = im_deconvolved.double()
    im_deconvolved = im_deconvolved[
        im_pad_shape[0] : -im_pad_shape[0],
        im_pad_shape[1] : -im_pad_shape[1],
        im_pad_shape[2] : -im_pad_shape[2],
    ]
    im_deconvolved = im_deconvolved.cpu()
    # Convert result back to a numpy array
    return im_deconvolved.numpy()
