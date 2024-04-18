import numpy as np
import numpy.typing as npt
from typing import Tuple


def compute_coefficients(
    bled_codes: npt.NDArray[np.float32], pixel_colors: npt.NDArray[np.float32], genes: npt.NDArray[np.int_]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Finds how to best weight the given genes to describe the seen pixel colour. Done for every given pixel.

    bled_codes (`((n_rounds * n_channels) x n_genes) ndarray`): bled code for every gene in every sequencing
        round/channel.
    pixel_colours (`((n_rounds * n_channels) x im_y x im_x x im_z) ndarray`): pixel colour for every sequencing
        round/channel.
    genes (`(im_y x im_x x im_z x n_genes_added) ndarray`): the indices of the genes selected for each image pixel.

    Returns:
        - (`(im_y x im_x x im_z x n_genes_added) ndarray[float32]`) coefs: OMP coefficients computed through least
            squares.
        - (`(im_y x im_x x im_z x (n_rounds * n_channels)) ndarray[float32]`) residual: pixel colours left after removing
            bled codes with computed coefficients.
    """
    pass
