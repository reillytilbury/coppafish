import numpy as np

from coppafish import NotebookPage


def test_NotebookPage__combine_variables():
    rng = np.random.RandomState(0)
    nbp_0 = NotebookPage("")
    var_0, var_1 = 3, 3
    tile_indices = [[0], [1]]
    combined_var = nbp_0._combine_variables(["eq"], "test_eq", var_0, var_1, tile_indices)
    assert combined_var == var_0
    assert combined_var == var_1
    var_0, var_1 = 1.5, 2.25
    tile_indices = [[0], [1]]
    combined_var = nbp_0._combine_variables(["add"], "test_add", var_0, var_1, tile_indices)
    assert np.isclose(combined_var, var_0 + var_1)
    var_0, var_1 = 1.5, 1.5
    tile_indices = [[0], [1]]
    combined_var = nbp_0._combine_variables(["close"], "test_close_0", var_0, var_1, tile_indices)
    assert np.isclose(combined_var, var_0)
    var_0 = rng.rand(5, 6, 7)
    var_1 = var_0.copy()
    tile_indices = [[0], [1]]
    combined_var = nbp_0._combine_variables(["close"], "test_close_1", var_0, var_1, tile_indices)
    assert np.allclose(combined_var, var_0)
    var_0 = rng.rand(5, 6, 7)
    var_1 = 5.5
    tile_indices = [[0], [1]]
    combined_var = nbp_0._combine_variables(["ignore"], "test_ignore", var_0, var_1, tile_indices)
    assert np.allclose(combined_var, var_0)
    var_0 = rng.rand(5, 6, 7)
    var_1 = rng.rand(5, 4, 7)
    tile_indices = [[0], [1]]
    combined_var = nbp_0._combine_variables(["append", "1"], "test_close_1", var_0, var_1, tile_indices)
    assert combined_var.shape == (5, 10, 7), "Expected shape (5, 10, 7) after appending variables by tile indices"
    assert np.allclose(combined_var[:, :6, :], var_0)
    assert np.allclose(combined_var[:, 6:, :], var_1)
    var_0 = rng.rand(5, 6, 7)
    var_1 = rng.rand(5, 6, 7)
    tile_indices = [[0], [1]]
    combined_var = nbp_0._combine_variables(["append", "2"], "test_close_1", var_0, var_1, tile_indices)
    assert combined_var.shape == (5, 6, 14), "Expected shape (5, 6, 14) after appending variables by tile indices"
    assert np.allclose(combined_var[:, :, :7], var_0)
    assert np.allclose(combined_var[:, :, 7:], var_1)
    var_0 = rng.rand(5, 6, 7)
    var_1 = rng.rand(5, 4, 7)
    tile_indices = [[0], [1]]
    combined_var = nbp_0._combine_variables(["tile", "1"], "test_tile_0", var_0, var_1, tile_indices)
    assert combined_var.shape == (5, 2, 7), "Expected shape (5, 2, 7) after appending variables by tile indices"
    assert np.allclose(combined_var[:, 0], var_0[:, 0])
    assert np.allclose(combined_var[:, 1], var_1[:, 1])
    var_0 = rng.rand(5, 6, 2)
    var_1 = rng.rand(5, 6, 4)
    tile_indices = [[0, 1], [2, 3]]
    combined_var = nbp_0._combine_variables(["tile", "2"], "test_tile_1", var_0, var_1, tile_indices)
    assert combined_var.shape == (5, 6, 4), "Expected shape (5, 6, 4) after appending variables by tile indices"
    assert np.allclose(combined_var[..., 0], var_0[..., 0])
    assert np.allclose(combined_var[..., 1], var_0[..., 1])
    assert np.allclose(combined_var[..., 2], var_1[..., 2])
    assert np.allclose(combined_var[..., 3], var_1[..., 3])
    var_0 = rng.rand(2, 5, 6)
    var_1 = rng.rand(4, 5, 6)
    tile_indices = [[0, 1], [2, 3]]
    combined_var = nbp_0._combine_variables(["tile", "0"], "test_tile_2", var_0, var_1, tile_indices)
    assert combined_var.shape == (4, 5, 6), "Expected shape (4, 5, 6) after appending variables by tile indices"
    assert np.allclose(combined_var[0], var_0[0])
    assert np.allclose(combined_var[1], var_0[1])
    assert np.allclose(combined_var[2], var_1[2])
    assert np.allclose(combined_var[3], var_1[3])
