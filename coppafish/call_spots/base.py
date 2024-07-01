from scipy.spatial import KDTree
from scipy.sparse.linalg import svds
import numpy as np
from typing import List

from .. import log


def get_non_duplicate(
    tile_origin: np.ndarray, use_tiles: List, tile_centre: np.ndarray, spot_local_yxz: np.ndarray, spot_tile: np.ndarray
) -> np.ndarray:
    """
    Find duplicate spots as those detected on a tile which is not tile centre they are closest to.

    Args:
        tile_origin: `float [n_tiles x 3]`.
            `tile_origin[t,:]` is the bottom left yxz coordinate of tile `t`.
            yx coordinates in `yx_pixels` and z coordinate in `z_pixels`.
            This is saved in the `stitch` notebook page i.e. `nb.stitch.tile_origin`.
        use_tiles: ```int [n_use_tiles]```.
            Tiles used in the experiment.
        tile_centre: ```float [3]```
            ```tile_centre[:2]``` are yx coordinates in ```yx_pixels``` of the centre of the tile that spots in
            ```yxz``` were found on.
            ```tile_centre[2]``` is the z coordinate in ```z_pixels``` of the centre of the tile.
            E.g. for tile of ```yxz``` dimensions ```[2048, 2048, 51]```, ```tile_centre = [1023.5, 1023.5, 25]```
            Each entry in ```tile_centre``` must be an integer multiple of ```0.5```.
        spot_local_yxz: ```int [n_spots x 3]```.
            Coordinates of a spot s on tile spot_tile[s].
            ```yxz[s, :2]``` are the yx coordinates in ```yx_pixels``` for spot ```s```.
            ```yxz[s, 2]``` is the z coordinate in ```z_pixels``` for spot ```s```.
        spot_tile: ```int [n_spots]```.
            Tile each spot was found on.

    Returns:
        ```bool [n_spots]```.
            Whether spot_tile[s] is the tile that spot_global_yxz[s] is closest to.
    """
    tile_centres = tile_origin[use_tiles] + tile_centre
    # Do not_duplicate search in 2D as overlap is only 2D
    tree_tiles = KDTree(tile_centres[:, :2])
    if np.isnan(tile_origin[np.unique(spot_tile)]).any():
        nan_tiles = np.unique(spot_tile)[np.unique(np.where(np.isnan(tile_origin[np.unique(spot_tile)]))[0])]
        log.error(
            ValueError(
                f"tile_origin for tiles\n{nan_tiles}\ncontains nan values but some spot_tile "
                f"also contains these tiles. Maybe remove these from use_tiles to continue.\n"
                f"Also, consider coppafish.plot.n_spots_grid to check if these tiles have few spots."
            )
        )
    spot_global_yxz = spot_local_yxz + tile_origin[spot_tile]
    all_nearest_tile_ind = tree_tiles.query(spot_global_yxz[:, :2])[1]
    not_duplicate = np.asarray(use_tiles)[all_nearest_tile_ind.flatten()] == spot_tile
    return not_duplicate


def bayes_mean(
    spot_colours: np.ndarray, prior_colours: np.ndarray, conc_param_parallel: float, conc_param_perp: float
) -> np.ndarray:
    """
    This function computes the posterior mean of the spot colours under a prior distribution with mean prior_colours
    and covariance matrix given by a diagonal matrix with diagonal entry conc_param_parallel for the direction parallel
    to prior_colours and conc_param_perp for the direction orthogonal to prior_colours.

    Args:
        spot_colours: np.ndarray [n_spots x n_channels_use]
            The spot colours for each spot.
        prior_colours: np.ndarray [n_channels_use]
            The prior mean colours.
        conc_param_parallel: np.ndarray [n_channels_use]
            The concentration parameter for the direction parallel to prior_colours.
        conc_param_perp: np.ndarray [n_channels_use]
            The concentration parameter for the direction orthogonal to prior_colours.
    """
    n_spots, data_sum = len(spot_colours), np.sum(spot_colours, axis=0)
    # deal with the case where there are no spots
    if n_spots == 0:
        return prior_colours

    prior_direction = prior_colours / np.linalg.norm(prior_colours)  # normalized prior direction
    sum_parallel = (data_sum @ prior_direction) * prior_direction  # projection of data sum along prior direction
    sum_perp = data_sum - sum_parallel  # projection of data sum orthogonal to mean direction

    # now compute the weighted sum of the posterior mean for parallel and perpendicular directions
    posterior_parallel = (sum_parallel + conc_param_parallel * prior_direction) / (n_spots + conc_param_parallel)
    posterior_perp = sum_perp / (n_spots + conc_param_perp)
    return posterior_parallel + posterior_perp


def compute_bleed_matrix(
    spot_colours: np.ndarray, gene_no: np.ndarray, gene_codes: np.ndarray, n_dyes: int
) -> np.ndarray:
    """
    Function to compute the bleed matrix from the spot colours and the gene assignments.
    Args:
        spot_colours: np.ndarray [n_spots x n_rounds x n_channels_use]
            The spot colours for each spot in each round and channel.
        gene_no: np.ndarray [n_spots]
            The gene assignment for each spot.
        gene_codes: np.ndarray [n_genes x n_rounds]
            The gene codes for each gene in each round.
        n_dyes: int
            The number of dyes.

    Returns:
        bleed_matrix: np.ndarray [n_dyes x n_channels_use]
            The bleed matrix.
    """
    assert len(spot_colours) == len(gene_no), "Spot colours and gene_no must have the same length."
    n_spots, n_rounds, n_channels_use = spot_colours.shape
    bleed_matrix = np.zeros((n_dyes, n_channels_use))

    # loop over all dyes, find the spots which are meant to be dye d in round r, and compute the SVD
    for d in range(n_dyes):
        dye_d_colours = []
        for r in range(n_rounds):
            relevant_genes = np.where(gene_codes[:, r] == d)[0]
            relevant_gene_mask = np.isin(gene_no, relevant_genes)
            dye_d_colours.append(spot_colours[relevant_gene_mask, r, :])
        # now we have all the good colours for dye d, compute the SVD
        dye_d_colours = np.concatenate(dye_d_colours, axis=0)
        u, s, v = svds(dye_d_colours, k=1)
        v = v[0]
        # make sure largest entry in v is positive
        v *= np.sign(v[np.argmax(np.abs(v))])
        bleed_matrix[d] = v

    return bleed_matrix
