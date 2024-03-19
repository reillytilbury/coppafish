import numpy as np
import statistics
from skimage import data
from skimage.filters import gaussian
from coppafish.register import base as reg_base
from coppafish.register import preprocessing as reg_pre


def test_find_shift_array():
    # TODO: Make sure we have access to skimage data
    # Set up random data (10, 256, 256)
    rng = np.random.RandomState(12)
    brain = rng.rand(10, 256, 256)
    brain_shifted = reg_pre.custom_shift(brain, np.array([0, 10, 10]))
    # Test the function
    z_box, y_box, x_box = 10, 128, 128
    brain_split, pos = reg_pre.split_3d_image(brain, 1, 2, 2, z_box, y_box, x_box)
    brain_shifted_split, _ = reg_pre.split_3d_image(brain_shifted, 1, 2, 2, z_box, y_box, x_box)
    shift_array, shift_corr = reg_base.find_shift_array(brain_split, brain_shifted_split, pos, r_threshold=0.5)
    # Test that the shape is correct
    assert shift_array.shape == (2 * 2, 3)
    # Test that the values are correct
    assert np.allclose(shift_array, np.array([[0, 10, 10], [0, 10, 10], [0, 10, 10], [0, 10, 10]]))
    assert np.allclose(shift_corr, np.array([1, 1, 1, 1]))


def test_find_z_tower_shifts():
    # set up data (60, 128, 128)
    rng = np.random.RandomState(31)
    cell = rng.rand(60, 128, 128)
    cell_shifted = reg_pre.custom_shift(cell, np.array([3, 0, 0]))
    # Test the function
    z_box, y_box, x_box = 10, 64, 64
    cell_split, pos = reg_pre.split_3d_image(cell, 6, 4, 4, z_box, y_box, x_box)
    cell_shift_split, _ = reg_pre.split_3d_image(cell_shifted, 6, 4, 4, z_box, y_box, x_box)
    pos = pos.reshape(6, 4, 4, 3)
    pos = pos[:, 0, 0]
    cell_tower = cell_split[:, 0, 0]
    cell_shift_tower = cell_shift_split[:, 0, 0]
    shift_array, shift_corr = reg_base.find_z_tower_shifts(cell_tower, cell_shift_tower, pos, pearson_r_threshold=0.5)
    # Test that the shape is correct
    assert shift_array.shape == (6, 3)
    # Test that the values are correct. They should be 6 copies of [3, 0, 0]
    for i in range(6):
        assert np.allclose(shift_array[i], np.array([3, 0, 0]))
    assert np.allclose(shift_corr, np.array([1, 1, 1, 1, 1, 1]))


def test_find_zyx_shift():
    rng = np.random.RandomState(52)
    kidney = np.sum(rng.rand(16, 512, 512, 3), axis=-1)[:, :128, :128]
    kidney_shifted = reg_pre.custom_shift(kidney, np.array([3, 15, 20]))
    # Test the function
    shift, shift_corr = reg_base.find_zyx_shift(kidney, kidney_shifted, pearson_r_threshold=0.5)
    # Test that the shape is correct
    assert shift.shape == (3,)
    # Test that the values are correct
    assert np.allclose(shift, np.array([3, 15, 20]))
    assert np.allclose(shift_corr, 1)


def test_ols_regression():
    # set up data
    rng = np.random.RandomState(0)
    pos = rng.rand(10, 3)
    shift1 = 5 * pos - pos
    # Test the function
    transform = reg_base.ols_regression(shift1, pos)
    # Test that the shape is correct
    assert transform.shape == (3, 4)
    # Test that the values are correct
    assert np.allclose(transform, np.array([[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0]]))


def test_huber_regression():
    rng = np.random.RandomState(0)
    pos = rng.rand(10, 3)
    shift1 = 5 * pos - pos
    # Test the function
    transform = reg_base.huber_regression(shift1, pos, False)
    # Test that the shape is correct
    assert transform.shape == (3, 4)
    # Test that the values are correct
    assert np.allclose(transform, np.array([[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0]]), atol=2e-6)


def test_brightness_scale():
    rng = np.random.RandomState(0)
    nx = 40
    ny = 40
    seq = rng.randint(2**8, dtype=np.int32, size=(ny, nx))
    preseq = reg_pre.custom_shift(seq, [1, 2]) * 4
    scale, sub_image_seq, sub_image_preseq = reg_base.brightness_scale(preseq, seq, intensity_percentile=0.5)
    assert isinstance(scale, float), "Expected scale to be type float"
    assert isinstance(sub_image_seq, np.ndarray), "Expected sub_image_seq to be type ndarray"
    assert isinstance(sub_image_preseq, np.ndarray), "Expected sub_image_preseq to be type ndarray"
    assert np.isclose(scale, 0.25, atol=1e-2)


def test_upsample_yx():
    # set up data
    im = np.eye(2, 2)[:, :, None]
    # upsample
    im_up = reg_base.upsample_yx(im, factor=2, order=0)
    # check that the shape is correct
    assert im_up.shape == (4, 4, 1)
    # check that the values are correct
    assert np.sum(im_up) == 8
    assert np.allclose(im_up[:, :, 0], np.array([[1, 1, 0, 0],
                                                 [1, 1, 0, 0],
                                                 [0, 0, 1, 1],
                                                 [0, 0, 1, 1]]))


def test_interpolate_flow():
    # set up data. we will have nz = ny = nx = 10
    flow = np.ones((3, 10, 10, 10))
    flow[0] = 1
    flow[1] = 2
    flow[2] = 3
    corr = np.ones((10, 10, 10))
    # interpolate
    flow_interp = reg_base.interpolate_flow(flow, correlation=corr,
                                            upsample_factor_yx=2)
    # check that the shape is correct
    assert flow_interp.shape == (3, 20, 20, 10)
    # check that the values are correct
    assert np.allclose(flow_interp[0], 2)
    assert np.allclose(flow_interp[1], 4)
    assert np.allclose(flow_interp[2], 3)


def test_flow_correlation():
    # set up data
    flow = np.ones((3, 10, 10, 10))
    flow[0] = 0
    flow[1] = 1
    flow[2] = 0
    # set up base and target
    base = np.ones((10, 10, 10))
    target = np.ones((10, 10, 10))
    base[5, 5, 5] = 2
    target[5, 4, 5] = 2
    # add tiny random noise to the target
    rng = np.random.RandomState(0)
    target += rng.rand(10, 10, 10) * 1e-6
    # correlate
    win_size = np.array([2, 2, 1])
    _, flow_corr = reg_base.flow_correlation(base=base, target=target, flow=flow,
                                             win_size=win_size, upsample_factor_yx=2)
    # check that the shape is correct
    assert flow_corr.shape == (20, 20, 10)
    # check that the values are correct
    indices = np.ix_(range(5, 15), range(5, 15), range(10))
    assert np.all(flow_corr[indices] >= 1)


def test_optical_flow_single():
    # set up data
    base = np.sum(data.astronaut(), axis=2)[::2, ::2]
    base = gaussian(base, sigma=2, preserve_range=True)
    base = np.repeat(base[:, :, None], 10, axis=2)
    base = base.astype(np.float32)
    for i in range(10):
        base[:, :, i] *= i / 10
    target = reg_pre.custom_shift(base, np.array([3, 2, 0]))
    # calculate the flow
    flow = reg_base.optical_flow_single(base, target, upsample_factor_yx=1)
    # check that the shape is correct
    ny, nx, nz = base.shape
    assert flow.shape == (3, ny, nx, nz)
    # check that the values are correct
    flow = np.round(flow)

    centre_idx = np.ix_(range(ny//4, 3*ny//4), range(nx//4, 3*nx//4), range(nz//4, 3*nz//4))
    correct_y = flow[0][centre_idx] == -3
    correct_x = flow[1][centre_idx] == -2
    correct_z = flow[2][centre_idx] == -0
    ny, nx, nz = ny // 2, nx // 2, nz // 2
    assert np.sum(correct_y)/(ny*nx*nz) > 0.9
    assert np.sum(correct_x)/(ny*nx*nz) > 0.9
    assert np.sum(correct_z)/(ny*nx*nz) > 0.9


