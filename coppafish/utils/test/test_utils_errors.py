import numpy as np

from coppafish.utils import errors
from coppafish import NotebookPage


def test_check_shape():
    shape = (5, 10, 4, 2)
    wrong_shape_1 = (4, 10, 4, 2)
    wrong_shape_2 = (5, 10, 1, 2)
    wrong_shape_3 = (5, 10, 4)
    wrong_shape_4 = (5, 10, 4, 2, 20)
    array = np.empty(shape)
    assert errors.check_shape(array, shape)
    assert not errors.check_shape(array, wrong_shape_1)
    assert not errors.check_shape(array, wrong_shape_2)
    assert not errors.check_shape(array, wrong_shape_3)
    assert not errors.check_shape(array, wrong_shape_4)
    assert errors.check_shape(array, list(shape))
    assert not errors.check_shape(array, list(wrong_shape_1))
    assert not errors.check_shape(array, list(wrong_shape_2))
    assert not errors.check_shape(array, list(wrong_shape_3))
    assert not errors.check_shape(array, list(wrong_shape_4))
    assert errors.check_shape(array, np.array(shape))
    assert not errors.check_shape(array, np.array(wrong_shape_1))
    assert not errors.check_shape(array, np.array(wrong_shape_2))
    assert not errors.check_shape(array, np.array(wrong_shape_3))
    assert not errors.check_shape(array, np.array(wrong_shape_4))


def test_check_color_nan():
    n_codes = 2
    n_rounds = 6
    n_channels = 8
    shape = (n_codes, n_rounds, n_channels)
    nbp_basic = NotebookPage("basic_info")
    for name, value in {
        "tile_pixel_value_shift": 10,
        "use_rounds": (0, 1, 2, 4),
        "use_channels": (1, 6, 7),
        "n_rounds": n_rounds,
        "n_channels": n_channels,
    }.items():
        nbp_basic.__setattr__(name, value)
    rng = np.random.RandomState(38)
    # Place the correct invalid value expected by the function in the unused rounds/channels
    array = np.full(shape, fill_value=np.nan, dtype=float)
    for s in range(n_codes):
        for r in range(n_rounds):
            for c in range(n_channels):
                if not r in nbp_basic.use_rounds:
                    continue
                if not c in nbp_basic.use_channels:
                    continue
                # Set co,r,c to a non invalid value
                array[s, r, c] = rng.rand()
    errors.check_color_nan(array, nbp_basic)
    del array
    array = np.full(shape, fill_value=-nbp_basic.tile_pixel_value_shift, dtype=int)
    for s in range(n_codes):
        for r in range(n_rounds):
            for c in range(n_channels):
                if not r in nbp_basic.use_rounds:
                    continue
                if not c in nbp_basic.use_channels:
                    continue
                # Set co,r,c to a non invalid value
                array[s, r, c] = rng.randint(0, 100, dtype=int)
    errors.check_color_nan(array, nbp_basic)


def test_compare_spots() -> None:
    spot_positions_0 = np.zeros((0, 3), np.float32)
    spot_gene_indices_0 = np.zeros(0, np.int16)
    spot_positions_1 = np.zeros((0, 3), np.float32)
    spot_gene_indices_1 = np.zeros(0, np.int16)
    distance_threshold = 0.1
    TPs, WPs, FPs, FNs = errors.compare_spots(
        spot_positions_0, spot_gene_indices_0, spot_positions_1, spot_gene_indices_1, distance_threshold
    )
    assert TPs == WPs == FPs == FNs == 0
    spot_positions_0 = np.zeros((2, 3), np.float32)
    spot_positions_0[0] = [3.75, 0, 0]
    spot_gene_indices_0 = np.zeros(2, np.int16)
    spot_positions_1 = np.zeros((1, 3), np.float32)
    spot_gene_indices_1 = np.zeros(1, np.int16)
    distance_threshold = 3.7
    TPs, WPs, FPs, FNs = errors.compare_spots(
        spot_positions_0, spot_gene_indices_0, spot_positions_1, spot_gene_indices_1, distance_threshold
    )
    assert TPs == 1
    assert WPs == 0
    assert FPs == 1
    assert FNs == 0
    spot_gene_indices_0[1] = 1
    TPs, WPs, FPs, FNs = errors.compare_spots(
        spot_positions_0, spot_gene_indices_0, spot_positions_1, spot_gene_indices_1, distance_threshold
    )
    assert TPs == 0
    assert WPs == 1
    assert FPs == 1
    assert FNs == 0
    # False negative spot example.
    spot_positions_0 = np.zeros((1, 3), np.float32)
    spot_gene_indices_0 = np.zeros(1, np.int16)
    spot_positions_1 = np.zeros((2, 3), np.float32)
    spot_positions_1[1] = [0.1, 0.5, 10]
    spot_gene_indices_1 = np.zeros(2, np.int16)
    distance_threshold = 7.1
    TPs, WPs, FPs, FNs = errors.compare_spots(
        spot_positions_0, spot_gene_indices_0, spot_positions_1, spot_gene_indices_1, distance_threshold
    )
    assert TPs == 1
    assert WPs == 0
    assert FPs == 0
    assert FNs == 1
    # Too many matching spots example.
    spot_positions_0 = np.zeros((5, 3), np.float32)
    spot_gene_indices_0 = np.zeros(5, np.int16)
    spot_positions_1 = np.zeros((2, 3), np.float32)
    spot_gene_indices_1 = np.zeros(2, np.int16)
    distance_threshold = 1.0
    TPs, WPs, FPs, FNs = errors.compare_spots(
        spot_positions_0, spot_gene_indices_0, spot_positions_1, spot_gene_indices_1, distance_threshold
    )
    assert TPs == 2
    assert WPs == 0
    assert FPs == 3
    assert FNs == 0
