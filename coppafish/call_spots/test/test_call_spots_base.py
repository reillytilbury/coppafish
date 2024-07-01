import numpy as np

from coppafish.call_spots import base


def test_get_non_duplicate():
    # Place two tiles
    tile_origin = np.array([[0,0,0],[100,0,0]])
    # Use both tiles
    use_tiles = [0,1]
    tile_centre = np.array([50,0,0], float)
    spot_local_yxz = np.array([[80,0,0], [110,0,0]], int)
    # Say each spot was found on the first tile
    spot_tile = np.array([0,0], int)
    output = base.get_non_duplicate(tile_origin, use_tiles, tile_centre, spot_local_yxz, spot_tile)
    assert spot_local_yxz.shape[0] == output.size, 'Expect output to have the same number of spots'
    assert output[0], 'First spot should be found closest to the first tile'
    assert not output[1], 'Second spot should be found closest to the other tile'

