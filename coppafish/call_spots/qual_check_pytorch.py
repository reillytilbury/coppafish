import torch
import numpy as np
from typing import Union

from .. import logging


def get_spot_intensity(spot_colors: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Finds the max intensity for each imaging round across all imaging channels for each spot.
    Then median of these max round intensities is returned.

    Args:
        spot_colors (`[n_spots x n_rounds x n_channels] ndarray[float]`: spot colors normalised to equalise intensities
            between channels (and rounds).

    Returns:
        `[n_spots] ndarray[float]`: index `s` is the intensity of spot `s`.

    Notes:
        - Logic is that we expect spots that are genes to have at least one large intensity value in each round
            so high spot intensity is more indicative of a gene.
        - This has to return a numpy and support a numpy input because it is used in optimised and non-optimised
            scripts. Very confusing!
    """
    if not isinstance(spot_colors, torch.Tensor):
        spot_colors = torch.asarray(spot_colors)
    if (spot_colors <= -15_000).sum() > 0:
        logging.warn(f"Found spot colors <= -15000")
    # Max over all channels, then median over all rounds
    return torch.median(torch.max(spot_colors, dim=2)[0], dim=1)[0].numpy()
