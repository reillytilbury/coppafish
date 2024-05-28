import numpy as np
import skimage
from ..register import preprocessing
from tqdm import tqdm


def compute_shift(t1: np.ndarray, t2: np.ndarray, t1_pos: np.ndarray, t2_pos: np.ndarray,
                  overlap: float) -> [np.ndarray, float]:
    """
    Compute the boundary shift between two tiles t1 and t2. The shift is computed by comparing the overlapping regions
    of the two tiles, and using a phase cross correlation algorithm to find any deviation from the expected overlap.
    Args:
        t1: (this is the tile that will be shifted) np.ndarray, [y_size, x_size, z_size] array of the first tile
        t2: (this is the reference tile) np.ndarray, [y_size, x_size, z_size] array of the second tile
        t1_pos: [y, x] position of tile 1 (integer indices)
        t2_pos: [y, x] position of tile 2 (integer indices)
        overlap: float, expected overlap between the two tiles

    Returns:
        shift: np.ndarray, shift in pixels between the two tiles
        score: float, square of the correlation coefficient between the reference tile and the shifted tile

    """

    # crop the tiles to the overlapping regions of the two tiles
    if (t2_pos[1] - t1_pos[1] == 1) and (t2_pos[0] - t1_pos[0] == 0): # t2 is to the right of t1
        t1 = t1[:, -int(overlap * t1.shape[2]):]
        t2 = t2[:, :int(overlap * t2.shape[2])]
    elif (t2_pos[1] - t1_pos[1] == -1) and (t2_pos[0] - t1_pos[0] == 0): # t2 is to the left of t1
        t1 = t1[:, :int(overlap * t1.shape[2])]
        t2 = t2[:, -int(overlap * t2.shape[2]):]
    elif (t2_pos[1] - t1_pos[1] == 0) and (t2_pos[0] - t1_pos[0] == 1): # t2 is below t1
        t1 = t1[-int(overlap * t1.shape[1]):, :]
        t2 = t2[:int(overlap * t2.shape[1]), :]
    elif (t2_pos[1] - t1_pos[1] == 0) and (t2_pos[0] - t1_pos[0] == -1): # t2 is above t1
        t1 = t1[:int(overlap * t1.shape[1]), :]
        t2 = t2[-int(overlap * t2.shape[1]):, :]
    else:
        raise ValueError('Tiles are not adjacent')  # this should never happen
    window = skimage.filters.window('hann', shape=(t1.shape[0], t1.shape[1]))
    # extend the window in z, but as we don't have many z-planes, fade first and last quarter of the planes
    z_planes = t1.shape[-1]
    n_z_fade = z_planes // 4
    window_z = np.concatenate((np.linspace(0, 1, n_z_fade),
                             np.ones(z_planes - 2 * n_z_fade),
                             np.linspace(1, 0, n_z_fade)))
    window = window[:, :, np.newaxis] * window_z[np.newaxis, np.newaxis, :]

    # compute the shift
    shift = skimage.registration.phase_cross_correlation(reference_image=t2 * window,
                                                         moving_image=t1 * window,
                                                         overlap_ratio=0.5,
                                                         disambiguate=True)[0]
    # compute the score
    t1_shifted = preprocessing.custom_shift(t1, shift.astype(int))
    score = np.corrcoef(t1_shifted.flatten(), t2.flatten())[0, 1]

    return shift, score ** 2


def minimise_shift_loss(shift: np.ndarray, score: np.ndarray) -> np.ndarray:
    """
    We have ~ 2 * n_tiles shifts that have been computed between tiles and only n_tiles shifts that we can apply to the
    tiles. We need to find the n_tiles shifts that minimise the quadratic loss function between the computed shifts
    and the shifts that we can apply to the tiles. Taking the derivative of the loss function and setting it to zero
    gives us a linear system of equations that we can solve to find the optimal shifts.

    Note: The loss function is defined as:

    L(w) = sum_i sum_j score[i, j] * (w[i] - w[j] - shift[i, j])^2,
    where w is the vector of shifts that we want to find, and shift is the matrix of computed shifts. This sum is over
    all neighbouring tiles (which is achieved by setting score = 0 for non-neighbouring tiles).

    Args:
        shift: np.ndarray, [n_tiles, n_tiles, 3] array of the shifts between the tiles
        score: np.ndarray, [n_tiles, n_tiles] array of the correlation scores between the tiles

    Returns:
        shifts_final: np.ndarray, [n_tiles, 3] array of the final shifts that will be applied to the tiles

    """
    n_tiles = shift.shape[0]
    # we need to build the n_tiles x n_tiles matrix A and the n_tiles x 3 matrix b that will be used to solve the linear
    # system of equations: Ax = b, where x is our final shift matrix (n_tiles x 3)
    A = np.zeros((n_tiles, n_tiles))
    b = np.zeros((n_tiles, 3))
    # fill the A matrix (do the maths on paper to understand this)
    for i, j in np.ndindex(n_tiles, n_tiles):
        if i == j:
            A[i, j] = np.sum(score[i, :])
        else:
            A[i, j] = -score[i, j]
    # fill the b matrix
    for i in range(n_tiles):
        b[i] = np.sum(score[i, :, np.newaxis] * shift[i], axis=0)

    # solve the linear system of equations
    shifts_final = np.linalg.lstsq(A, b, rcond=None)[0]

    return shifts_final


def fuse_tiles(tiles: np.ndarray, tile_origins: np.ndarray, tilepos_yx: np.ndarray,
               overlap: float, save_path: str) -> np.ndarray:
    """
    Fuse a stack of tiles into a single large image, using the tile_origins to determine the position of each tile in
    the large image. The function is not difficult conceptually, as it just involves applying the shifts to each tile
    and adding them to the image. Difficulty comes from 2 sources:
    1. Overflow: tiles that are shifted outside of the image need to be cropped
    2. Overlap: tiles that are shifted on top of each other need to be smoothly blended

    Note: Even though the tiles are in the form yxz, the large image is in the form zyx for compatibility with the
    main viewer
    Args:
        tiles: np.ndarray, [n_tiles, im_size_y, im_size_x, im_szie_z] array of the tiles to be fused
        tile_origins: np.ndarray, [n_tiles, 3] array of the y, x, z positions of each tile (need to be integers)
        tilepos_yx: np.ndarray, [n_tiles, 2] array of the y, x indices of each tile
        overlap: float, expected overlap between the tiles
        save_path: str, path to save the fused image

    Returns:
        fused_image: np.ndarray, [large_z, large_y, large_x] array of the fused image
    """
    # convert the tile origins to integers and declare frequently used variables
    tile_origins = tile_origins.astype(int)
    n_rows, n_cols = np.max(tilepos_yx, axis=0) + 1
    n_tiles = len(tiles)
    im_size, _, z_planes = tiles[0].shape
    large_im_shape = (z_planes, int(n_rows * im_size * (1 - overlap)), int(n_cols * im_size * (1 - overlap)))
    # create a list of cropped tiles (this cannot be an array as the tiles are of different sizes)
    cropped_tiles_list = []
    for t in tqdm(range(n_tiles), desc='Applying shifts to tiles', total=n_tiles):
        tile_t, tile_origins[t] = crop_image(image=tiles[t], image_origin=tile_origins[t],
                                             image_bound=large_im_shape)
        cropped_tiles_list.append(tile_t)
    # delete the original tiles to save memory
    del tiles
    # taper the edges of the tiles
    for t in tqdm(range(n_tiles), desc='Tapering tile edges', total=n_tiles):
        cropped_tiles_list[t] = taper_image(image=cropped_tiles_list[t], tile_start=tile_origins,
                                            tile_end=tile_origins + np.array([tile.shape for tile in cropped_tiles_list]),
                                            tilepos_yx=tilepos_yx, current_tile=t)
    # create the large image
    large_image = np.zeros(large_im_shape, dtype=np.uint16)
    for t in tqdm(range(n_tiles), desc='Adding tiles to the large image', total=n_tiles):
        y_start, x_start, z_start = tile_origins[t]
        y_end, x_end, z_end = tile_origins[t] + np.array(cropped_tiles_list[t].shape)
        large_image[z_start:z_end, y_start:y_end, x_start:x_end] += np.moveaxis(cropped_tiles_list[t],
                                                                                source=2, destination=0)

    # save the fused image
    np.save(save_path, large_image)

    return large_image


def crop_image(image: np.ndarray, image_origin: np.ndarray, image_bound: tuple) -> [np.ndarray, np.ndarray]:
    """
    Crop an image to the region defined by image_origin and image_bound.
    Args:
        image: (np.ndarray) [y_size, x_size, z_size] array of the image to be cropped
        image_origin: (np.ndarray) [3] array of the y, x, z position of the top left corner of the image to be cropped
        image_bound: (tuple) [3] maximum y, x, z position of the image to be cropped. If all of these are larger
                        than the image origin + the image size, the image is not cropped in this direction.

    Returns:
        im_cropped: (np.ndarray) [y_size_new, x_size_new, z_size_new] array of the cropped image
        bottom_left_corner: (np.ndarray) [3] array of the updated y, x, z position of the bottom left corner of the
        cropped image
    """
    y_size, x_size, z_size = image.shape
    y_start, x_start, z_start = image_origin
    y_end, x_end, z_end = y_start + y_size, x_start + x_size, z_start + z_size

    # correct for overflow (on the negative side)
    if y_start < 0:
        image = image[-y_start:, :, :]
        y_start = 0
    if x_start < 0:
        image = image[:, -x_start:, :]
        x_start = 0
    if z_start < 0:
        image = image[:, :, -z_start:]
        z_start = 0

    # correct for overflow (on the positive side)
    if y_end > image_bound[0]:
        y_end = image_bound[0]
        image = image[:y_end - y_start, :, :]
    if x_end > image_bound[1]:
        x_end = image_bound[1]
        image = image[:, :x_end - x_start, :]
    if z_end > image_bound[2]:
        z_end = image_bound[2]
        image = image[:, :, :z_end - z_start]

    # keep a copy of the new corner coords
    bottom_left_corner = np.array([y_start, x_start, z_start])
    return image, bottom_left_corner


def taper_image(image: np.ndarray, tile_start: np.ndarray, tile_end: np.ndarray, tilepos_yx: np.ndarray,
                current_tile: int) -> np.ndarray:
    """
    Taper the edges of the tiles to avoid visible seams in the final image. The tapering is done by applying a
    linear ramp to the edges of the tiles. The start point of this ramp is determined by the start point of the
    overlap region, and the end point is determined by the end point of the overlap region.
    Args:
        image: np.ndarray, [y_size, x_size, z_size] array of the image to be tapered
        tile_start: np.ndarray, [n_tiles, 3] array of the y, x, z position of the top left corner of the tile
        tile_end: np.ndarray, [n_tiles, 3] array of the y, x, z position of the bottom right corner of the tile
        tilepos_yx: np.ndarray, [n_tiles, 2] array of the y, x indices of each tile
        current_tile: int, index of the tile to be tapered

    Returns:
        tapered_image: np.ndarray, [y_size, x_size, z_size] array of the tapered image
    """
    # convert the image to float32 for multiplication. We will convert it back to uint16 at the end
    image = image.astype(np.float32)
    tilepos_current = tilepos_yx[current_tile]
    n_tiles = len(tilepos_yx)

    # find the adjacent tiles
    adjacent_tiles = []
    for t in range(n_tiles):
        if abs(tilepos_current - tilepos_yx[t]).sum() == 1:
            adjacent_tiles.append(t)

    # if there are no adjacent tiles, we skip the tapering
    if len(adjacent_tiles) == 0:
        print(f'Tile {current_tile} has no adjacent tiles. Skipping tapering.')
        return image.astype(np.uint16)

    # taper the edges where we are adjacent to another tile
    for t in adjacent_tiles:
        if tilepos_current[1] == tilepos_yx[t][1] + 1:
            # current tile is to the right of t
            x_end = tile_end[t, 1] - tile_start[current_tile, 1]
            image[:, :x_end] *= np.linspace(0, 1, x_end)[None, :, None]
        elif tilepos_current[1] == tilepos_yx[t][1] - 1:
            # current tile is to the left of t
            x_start = tile_start[t, 1] - tile_end[current_tile, 1]
            image[:, x_start:] *= np.linspace(1, 0, - x_start)[None, :, None]
        elif tilepos_current[0] == tilepos_yx[t][0] + 1:
            # current tile is below t
            y_end = tile_end[t, 0] - tile_start[current_tile, 0]
            image[:y_end, :] *= np.linspace(0, 1, y_end)[:, None, None]
        elif tilepos_current[0] == tilepos_yx[t][0] - 1:
            # current tile is above t
            y_start = tile_start[t, 0] - tile_end[current_tile, 0]
            image[y_start:, :] *= np.linspace(1, 0, - y_start)[:, None, None]

    return image.astype(np.uint16)
