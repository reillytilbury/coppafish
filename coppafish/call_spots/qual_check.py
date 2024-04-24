from typing import Optional, Union, List
import numpy as np
import numpy.typing as npt

from ..omp.scores import omp_scores_int_to_float
from ..setup import NotebookPage, Notebook
from .. import log


def get_spot_intensity(spot_colors: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Finds the max intensity for each imaging round across all imaging channels for each spot.
    Then median of these max round intensities is returned.

    Args:
        spot_colors (`[n_spots x n_rounds x n_channels] ndarray[float]`: spot colors normalised to equalise intensities
            between channels (and rounds).

    Returns:
        `[n_spots] ndarray[float]`: index `s` is the intensity of spot `s`.

    Notes:
        Logic is that we expect spots that are genes to have at least one large intensity value in each round
        so high spot intensity is more indicative of a gene.
    """
    if (spot_colors <= -15_000).sum() > 0:
        log.warn(f"Found spot colors <= -15000")
    # Max over all channels, then median over all rounds
    return np.median(np.max(spot_colors, axis=2), axis=1)


def omp_spot_score(
    nbp: NotebookPage,
    spot_no: Optional[Union[int, List, np.ndarray]] = None,
) -> np.ndarray:
    """
    Score for omp gene assignment

    Args:
        nbp: OMP Notebook page
        spot_no: Which spots to get score for. If `None`, all scores will be found.

    Returns:
        Score for each spot in spot_no if given, otherwise all spot scores.
    """
    if spot_no is None:
        scores = nbp.scores
    else:
        scores = nbp.scores[spot_no]

    return omp_scores_int_to_float(scores)


def get_intensity_thresh(nb: Notebook) -> float:
    """
    Gets threshold for intensity from parameters in `config file` or Notebook.

    Args:
        nb: Notebook containing at least the `call_spots` page.

    Returns:
        intensity threshold
    """
    if nb.has_page("thresholds"):
        intensity_thresh = nb.thresholds.intensity
    else:
        config = nb.get_config()["omp"]
        intensity_thresh = nb.call_spots.abs_intensity_percentile[config["initial_intensity_thresh_percentile"]]
    return intensity_thresh


def quality_threshold(
    nb: Notebook, method: str = "omp", intensity_thresh: float = 0, score_thresh: float = 0
) -> np.ndarray:
    """
    Indicates which spots pass both the score and intensity quality thresholding.

    Args:
        nb: Notebook containing at least the `ref_spots` page.
        method: `'ref'` or `'omp'` or 'prob' indicating which spots to consider.
        intensity_thresh: Intensity threshold for spots included.
        score_thresh: Score threshold for spots included.

    Returns:
        `bool [n_spots]` indicating which spots pass quality thresholding.

    """
    if method.lower() != "omp" and method.lower() != "ref" and method.lower() != "anchor" and method.lower() != "prob":
        log.error(ValueError(f"method must be 'omp' or 'anchor' but {method} given."))
    method_omp = method.lower() == "omp"
    method_anchor = method.lower() == "anchor" or method.lower() == "ref"
    method_prob = method.lower() == "prob"
    # If thresholds are not given, get them from config file or notebook (preferably from notebook)
    if intensity_thresh == 0 and score_thresh == 0:
        intensity_thresh = get_intensity_thresh(nb)
        config = nb.get_config()["thresholds"]
        if method_omp:
            score_thresh = config["score_omp"]
        elif method_anchor:
            score_thresh = config["score_ref"]
        elif method_prob:
            score_thresh = config["score_prob"]
        # if thresholds page exists, use those values to override config file
        if nb.has_page("thresholds"):
            score_thresh = nb.thresholds.score_omp if method_omp else nb.thresholds.score_ref

    intensity = np.ones_like(nb.omp.gene_no, dtype=np.float32) if method_omp else nb.ref_spots.intensity
    if method_omp:
        score = omp_spot_score(nb.omp)
    elif method_anchor:
        score = nb.ref_spots.score
    elif method_prob:
        score = np.max(nb.ref_spots.gene_probs, axis=1)
    qual_ok = np.array([score > score_thresh, intensity > intensity_thresh]).all(axis=0)
    return qual_ok
