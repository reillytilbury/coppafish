from typing import Tuple

import numpy as np
import numpy.typing as npt

from ..setup import NotebookPage


def get_all_scores(
    nbp_basic: NotebookPage, nbp_omp: NotebookPage
) -> Tuple[npt.NDArray[np.float16], npt.NDArray[np.int16]]:
    """
    Get gene scores for every tile, concatenated together.

    Args:
        - nbp_basic (notebook page): `basic_info` notebook page.
        - nbp_omp (notebook page): `omp` notebook page.

    Returns:
        - (`(n_spots) ndarray[float16]`) all_scores: all gene scores.
        - (`(n_spots) ndarray[int16]`) all_tiles: the tile for each spot.
    """
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_omp) is NotebookPage

    all_scores = np.zeros(0, np.float16)
    all_tiles = np.zeros(0, np.int16)
    for t in nbp_basic.use_tiles:
        t_scores: np.ndarray = nbp_omp.results[f"tile_{t}/scores"][:]
        all_scores = np.append(all_scores, t_scores, 0)
        all_tiles = np.append(all_tiles, np.full(t_scores.size, t, np.int16))
    return all_scores, all_tiles


def get_all_gene_no(
    nbp_basic: NotebookPage, nbp_omp: NotebookPage
) -> Tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
    """
    Get gene numbers for every tile, concatenated together.

    Args:
        - nbp_basic (notebook page): `basic_info` notebook page.
        - nbp_omp (notebook page): `omp` notebook page.

    Returns:
        - (`(n_spots) ndarray[int16]`) all_gene_no: all gene numbers.
        - (`(n_spots) ndarray[int16]`) all_tiles: the tile for each spot.
    """
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_omp) is NotebookPage

    all_gene_no = np.zeros(0, np.int16)
    all_tiles = np.zeros(0, np.int16)
    for t in nbp_basic.use_tiles:
        t_gene_no: np.ndarray = nbp_omp.results[f"tile_{t}/gene_no"][:]
        all_gene_no = np.append(all_gene_no, t_gene_no, 0)
        all_tiles = np.append(all_tiles, np.full(t_gene_no.size, t, np.int16))
    return all_gene_no, all_tiles


def get_all_local_yxz(
    nbp_basic: NotebookPage, nbp_omp: NotebookPage
) -> Tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
    """
    Get spot local positions for every tile, concatenated together.

    Args:
        - nbp_basic (notebook page): `basic_info` notebook page.
        - nbp_omp (notebook page): `omp` notebook page.

    Returns:
        - (`(n_spots) ndarray[int16]`) all_local_yxz: all gene local positions.
        - (`(n_spots) ndarray[int16]`) all_tiles: the tile for each spot.
    """
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_omp) is NotebookPage

    all_local_yxz = np.zeros((0, 3), np.int16)
    all_tiles = np.zeros(0, np.int16)
    for t in nbp_basic.use_tiles:
        t_local_yxz: np.ndarray = nbp_omp.results[f"tile_{t}/local_yxz"][:]
        all_local_yxz = np.append(all_local_yxz, t_local_yxz, 0)
        all_tiles = np.append(all_tiles, np.full(t_local_yxz.shape[0], t, np.int16))
    return all_local_yxz, all_tiles


def get_all_colours(
    nbp_basic: NotebookPage, nbp_omp: NotebookPage
) -> Tuple[npt.NDArray[np.float16], npt.NDArray[np.int16]]:
    """
    Get spot local positions for every tile, concatenated together.

    Args:
        - nbp_basic (notebook page): `basic_info` notebook page.
        - nbp_omp (notebook page): `omp` notebook page.

    Returns:
        - (`(n_spots) ndarray[int16]`) all_colours: all spot colours.
        - (`(n_spots) ndarray[int16]`) all_tiles: the tile for each spot.
    """
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_omp) is NotebookPage

    all_colours = np.zeros((0, len(nbp_basic.use_rounds), len(nbp_basic.use_channels)), np.float16)
    all_tiles = np.zeros(0, np.int16)
    for t in nbp_basic.use_tiles:
        t_colours: np.ndarray = nbp_omp.results[f"tile_{t}/colours"][:]
        all_colours = np.append(all_colours, t_colours, 0)
        all_tiles = np.append(all_tiles, np.full(t_colours.size, t, np.int16))
    return all_colours, all_tiles


def global_to_local_index(
        nbp_basic: NotebookPage,
        nbp_omp: NotebookPage,
        spot_global_index: int
) -> int:
    """
    Get index of a spot on its local tile from its global index.

    Args:
        - nbp_basic (notebook page): `basic_info` notebook page.
        - nbp_omp (notebook page): `omp` notebook page.
        - spot_global_index (int): the global index of the spot. (an integer from 0 to n_spots_total)

    Returns:
        - (int) spot_local_index: the index of the spot on its local tile.
    """
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_omp) is NotebookPage

    # Get the number of spots in each tile
    n_spots_per_tile = [nbp_omp.results[f'tile_{t}'].scores.shape[0] for t in nbp_basic.use_tiles]
    # get the global indices where each tile starts (and a final one for the end)
    change_points = np.array([0] + list(np.cumsum(n_spots_per_tile)))
    # find the starting index of the tile that the spot is on
    spot_tile_start_index = change_points[np.where(change_points <= spot_global_index)[0][-1]]
    spot_local_index = spot_global_index - spot_tile_start_index
    return spot_local_index
