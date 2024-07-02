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


def test_bayes_mean():
    seed = 0
    np.random.seed(seed)
    # get 100 random spots with 5 colours
    n_spots, n_channels = 100, 5
    expected_colour = np.array([1, 0, 0, 0, 0])
    spot_colours = np.tile(expected_colour, (n_spots, 1)).astype(float)
    # add noise to the colours
    spot_colours += np.random.normal(0, 0.1, spot_colours.shape)
    # rescale column 0
    spot_colours[:, 0] *= 10
    bayes_mean = base.bayes_mean(spot_colours, expected_colour, 0.1, 50)
    # check shape
    assert bayes_mean.shape == (n_channels,), 'Expect output to have the same number of channels'
    # check that the first entry has average approximately 10
    assert np.isclose(bayes_mean[0], 10, atol=1), 'Expect first column to have average 10'
    # check that the other entries are approximately 0
    assert np.all(np.isclose(bayes_mean[1:], 0, atol=0.1)), 'Expect other columns to have average 0'


def test_compute_bleed_matrix():
    seed = 0
    np.random.seed(seed)
    # 3 rounds, 4 channels, 2 genes, 3 dyes, 100 spots
    n_rounds, n_channels, n_genes, n_dyes, n_spots = 3, 4, 2, 3, 100
    # gene codes
    gene_codes = np.random.randint(0, n_dyes, (n_genes, n_rounds))
    # expected dye colours
    expected_bleed = np.eye(n_dyes, n_channels) + np.random.normal(0, 0.1, (n_dyes, n_channels))
    expected_bled_codes = expected_bleed[gene_codes]
    # gene assignments
    gene_no = np.random.randint(0, n_genes, n_spots)
    # spot colours
    spot_colours = (expected_bled_codes[gene_no]
                    + np.random.normal(0, 0.1, (n_spots, n_rounds, n_channels)))
    bleed_matrix = base.compute_bleed_matrix(spot_colours=spot_colours, gene_no=gene_no,
                                             gene_codes=gene_codes, n_dyes=n_dyes)
    # check shape
    assert bleed_matrix.shape == (n_dyes, n_channels), 'Expect output to have the same shape as the bleed matrix'
    # check that the bleed matrix is close to the expected bleed matrix
    # need to normalise each row of expected_bleed first
    expected_bleed = expected_bleed / np.sum(expected_bleed, axis=1)[:, None]
    assert np.all(np.isclose(bleed_matrix, expected_bleed, atol=0.1)), \
        'Expect bleed matrix to be close to the expected bleed matrix'
