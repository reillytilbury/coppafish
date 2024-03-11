import numbers
from typing import Union, Tuple
import numpy as np
import numpy.typing as npt
from scipy.ndimage import correlate, convolve
from scipy.signal import oaconvolve

from .base import ensure_odd_kernel
from ... import logging, utils


def imfilter(
    image: np.ndarray, kernel: np.ndarray, padding: Union[float, str] = 0, corr_or_conv: str = "corr", oa: bool = True
) -> np.ndarray:
    """
    Copy of MATLAB `imfilter` function with `'output_size'` equal to `'same'`.

    Args:
        image: `float [image_sz1 x image_sz2 x ... x image_szN]`.
            Image to be filtered.
        kernel: `float [kernel_sz1 x kernel_sz2 x ... x kernel_szN]`.
            Multidimensional filter.
        padding: One of the following, indicated which padding to be used.

            - numeric scalar - Input array values outside the bounds of the array are assigned the value `X`.
                When no padding option is specified, the default is `0`.
            - `‘symmetric’` - Input array values outside the bounds of the array are computed by
                mirror-reflecting the array across the array border.
            - `‘edge’`- Input array values outside the bounds of the array are assumed to equal
                the nearest array border value.
            - `'wrap'` - Input array values outside the bounds of the array are computed by implicitly
                assuming the input array is periodic.
        corr_or_conv:
            - `'corr'` - Performs multidimensional filtering using correlation.
            - `'conv'` - Performs multidimensional filtering using convolution.
        oa: Whether to use oaconvolve or scipy.ndimage.convolve.
            scipy.ndimage.convolve seems to be quicker for smoothing in extract step (3s vs 20s for 50 z-planes).

    Returns:
        `float [image_sz1 x image_sz2 x ... x image_szN]`.
            `image` after being filtered.
    """
    if oa:
        if corr_or_conv == "corr":
            kernel = np.flip(kernel)
        elif corr_or_conv != "conv":
            logging.error(
                ValueError(f"corr_or_conv should be either 'corr' or 'conv' but given value is {corr_or_conv}")
            )
        kernel = ensure_odd_kernel(kernel, "end")
        if kernel.ndim < image.ndim:
            kernel = np.expand_dims(kernel, axis=tuple(np.arange(kernel.ndim, image.ndim)))
        pad_size = [(int((ax_size - 1) / 2),) * 2 for ax_size in kernel.shape]
        if isinstance(padding, numbers.Number):
            return oaconvolve(np.pad(image, pad_size, "constant", constant_values=padding), kernel, "valid")
        else:
            return oaconvolve(np.pad(image, pad_size, padding), kernel, "valid")
    else:
        if padding == "symmetric":
            padding = "reflect"
        elif padding == "edge":
            padding = "nearest"
        # Old method, about 3x slower for filtering large 3d image with small 3d kernel
        if isinstance(padding, numbers.Number):
            pad_value = padding
            padding = "constant"
        else:
            pad_value = 0.0  # doesn't do anything for non-constant padding
        if corr_or_conv == "corr":
            kernel = ensure_odd_kernel(kernel, "start")
            return correlate(image, kernel, mode=padding, cval=pad_value)
        elif corr_or_conv == "conv":
            kernel = ensure_odd_kernel(kernel, "end")
            return convolve(image, kernel, mode=padding, cval=pad_value)
        else:
            logging.error(
                ValueError(f"corr_or_conv should be either 'corr' or 'conv' but given value is {corr_or_conv}")
            )


def manual_convolve(
    image: np.ndarray,
    y_kernel_shifts: np.ndarray,
    x_kernel_shifts: np.asarray,
    z_kernel_shifts: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """
    Finds result of convolution at specific locations indicated by `coords` with binary kernel.
    I.e. instead of convolving whole `image`, just find result at these `points`.

    Args:
        image: `int [image_szY x image_szX x image_szZ]`.
            Image to be filtered. Must be 3D.
        y_kernel_shifts: `int [n_nonzero_kernel]`
            Shifts indicating where kernel equals 1.
            I.e. if `kernel = np.ones((3,3))` then `y_shift = x_shift = z_shift = [-1, 0, 1]`.
        x_kernel_shifts: `int [n_nonzero_kernel]`
            Shifts indicating where kernel equals 1.
            I.e. if `kernel = np.ones((3,3))` then `y_shift = x_shift = z_shift = [-1, 0, 1]`.
        z_kernel_shifts: `int [n_nonzero_kernel]`
            Shifts indicating where kernel equals 1.
            I.e. if `kernel = np.ones((3,3))` then `y_shift = x_shift = z_shift = [-1, 0, 1]`.
        coords: `int [n_points x 3]`.
            yxz coordinates where result of filtering is desired.

    Returns:
        `int [n_points]`.
            Result of filtering of `image` at each point in `coords`.

    Notes:
        - Image needs to be padded before this function is called otherwise get an error when go out of bounds.
    """
    n_points = coords.shape[0]
    n_nonzero_kernel = y_kernel_shifts.size
    coords_shifted = np.repeat(coords[:, np.newaxis], n_nonzero_kernel, axis=1)
    coords_shifted[..., 0] += y_kernel_shifts
    coords_shifted[..., 1] += x_kernel_shifts
    coords_shifted[..., 2] += z_kernel_shifts
    return image[tuple(coords_shifted.reshape((-1, 3)).T)].reshape((n_points, n_nonzero_kernel)).sum(1)


def imfilter_coords(
    image: np.ndarray,
    kernel: np.ndarray,
    coords: np.ndarray,
    padding: Union[float, str] = 0,
    corr_or_conv: str = "corr",
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Copy of MATLAB `imfilter` function with `'output_size'` equal to `'same'`.
    Only finds result of filtering at specific locations but still filters entire image.

    !!! note
        image and image2 need to be np.int8 and kernel needs to be int otherwise will get cython error.

    Args:
        image: `int [image_szY x image_szX (x image_szZ)]`.
            Image to be filtered. Must be 2D or 3D.
        kernel: `int [kernel_szY x kernel_szX (x kernel_szZ)]`.
            Multidimensional filter, expected to be binary i.e. only contains 0 and/or 1.
        coords: `int [n_points x image.ndims]`.
            Coordinates where result of filtering is desired.
        padding: One of the following, indicated which padding to be used.

            - numeric scalar - Input array values outside the bounds of the array are assigned the value `X`.
                When no padding option is specified, the default is `0`.
            - `‘symmetric’` - Input array values outside the bounds of the array are computed by
                mirror-reflecting the array across the array border.
            - `‘edge’`- Input array values outside the bounds of the array are assumed to equal
                the nearest array border value.
            - `'wrap'` - Input array values outside the bounds of the array are computed by implicitly
                assuming the input array is periodic.
        corr_or_conv:
            - `'corr'` - Performs multidimensional filtering using correlation.
            - `'conv'` - Performs multidimensional filtering using convolution.

    Returns:
        `int [n_points]`.
            Result of filtering of `image` at each point in `coords`.
    """
    if corr_or_conv == "corr":
        kernel = np.flip(kernel)
    elif corr_or_conv != "conv":
        logging.error(ValueError(f"corr_or_conv should be either 'corr' or 'conv' but given value is {corr_or_conv}"))
    kernel = ensure_odd_kernel(kernel, "end")

    # Ensure shape of image and kernel correct
    if image.ndim != coords.shape[1]:
        logging.error(
            ValueError(f"Image has {image.ndim} dimensions but coords only have {coords.shape[1]} dimensions.")
        )
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
    elif image.ndim != 3:
        logging.error(ValueError(f"image must have 2 or 3 dimensions but given image has {image.ndim}."))
    if kernel.ndim == 2:
        kernel = np.expand_dims(kernel, 2)
    elif kernel.ndim != 3:
        logging.error(ValueError(f"kernel must have 2 or 3 dimensions but given image has {image.ndim}."))
    if kernel.max() > 1:
        logging.error(
            ValueError(f"kernel is expected to be binary, only containing 0 or 1 but kernel.max = {kernel.max()}")
        )

    if coords.shape[1] == 2:
        # set all z coordinates to 0 if 2D.
        coords = np.append(coords, np.zeros((coords.shape[0], 1), dtype=int), axis=1)
    if (coords.max(axis=0) >= np.array(image.shape)).any():
        logging.error(
            ValueError(f"Max yxz coordinates provided are {coords.max(axis=0)} but image has shape {image.shape}.")
        )

    pad_size = [(int((ax_size - 1) / 2),) * 2 for ax_size in kernel.shape]
    pad_coords = np.asarray(coords) + np.array([val[0] for val in pad_size])
    if isinstance(padding, numbers.Number):
        image_pad = np.pad(np.asarray(image), pad_size, "constant", constant_values=padding).astype(int)
    else:
        image_pad = np.pad(np.asarray(image), pad_size, padding).astype(int)
    y_shifts, x_shifts, z_shifts = get_shifts_from_kernel(np.asarray(np.flip(kernel)))
    # manual_convolve can be memory limited, hence the for loop
    n_points = pad_coords.shape[0]
    n_max_points = int(100_000 * utils.system.get_available_memory() / 107)
    n_batches = int(np.ceil(n_points / n_max_points))
    output = np.asarray([], dtype=image.dtype)
    for i in range(n_batches):
        index_start, index_end = i * n_max_points, min([(i + 1) * n_max_points, n_points])
        pad_coords_batch = pad_coords[index_start:index_end]
        output = np.append(output, manual_convolve(image_pad, y_shifts, x_shifts, z_shifts, pad_coords_batch))
    return output


def get_shifts_from_kernel(kernel: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Returns where kernel is positive as shifts in y, x and z.
    I.e. `kernel=np.ones((3,3,3))` would return `y_shifts = x_shifts = z_shifts = -1, 0, 1`.

    Args:
        kernel (`[kernel_szY x kernel_szX x kernel_szY] ndarray[int]`): the kernel.

    Returns:
        - `int [n_shifts]`.
            y_shifts.
        - `int [n_shifts]`.
            x_shifts.
        - `int [n_shifts]`.
            z_shifts.
    """
    shifts = list(np.where(kernel > 0))
    for i in range(kernel.ndim):
        shifts[i] = (shifts[i] - (kernel.shape[i] - 1) / 2).astype(int)
    return tuple(shifts)
