import torch

from .. import log


def get_spot_intensity(spot_colors: torch.Tensor) -> torch.Tensor:
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
    assert isinstance(spot_colors, torch.Tensor)
    assert spot_colors.dim() == 3

    if (spot_colors <= -15_000).sum() > 0:
        log.warn(f"Found spot colors <= -15000")
    # Max over all channels, then median over all rounds
    return spot_colors.abs().max(dim=2)[0].median(dim=1)[0]
