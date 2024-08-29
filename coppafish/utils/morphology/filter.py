from typing import Tuple

import numpy as np
import numpy.typing as npt


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
