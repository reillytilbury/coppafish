import numpy as np
from skimage.filters import gaussian
from skimage import data

from coppafish.register import base as reg_base
from coppafish.register import preprocessing as reg_pre


def test_upsample_yx():
    # set up data
    im = np.eye(2, 2)[:, :, None]
    # upsample
    im_up = reg_base.upsample_yx(im, factor=2, order=0)
    # check that the shape is correct
    assert im_up.shape == (4, 4, 1)
    # check that the values are correct
    assert np.sum(im_up) == 8
    assert np.allclose(im_up[:, :, 0], np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]))


def test_interpolate_flow():
    # set up data. We will have nz = ny = nx = 10
    flow = np.ones((3, 10, 10, 10))
    flow[0] = 1
    flow[1] = 2
    flow[2] = 3
    corr = np.ones((10, 10, 10))
    # interpolate
    flow_interp = reg_base.interpolate_flow(flow, correlation=corr, upsample_factor_yx=2, tile=0, round=0)
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
    _, flow_corr = reg_base.flow_correlation(
        base=base, target=target, flow=flow, upsample_factor_yx=2, tile=0, round=0
    )
    # check that the shape is correct
    assert flow_corr.shape == (20, 20, 10)
    # check that the values are correct
    indices = np.ix_(range(5, 15), range(5, 15), range(10))
    assert np.all(flow_corr[indices] >= 1)


def test_optical_flow_single():
    # set up data
    base = data.cells3d()[20:40, 1]
    base = np.swapaxes(base, 0, 2)
    base = gaussian(base, sigma=1, preserve_range=True)
    base = base.astype(np.float32)
    target = reg_pre.custom_shift(base, np.array([3, 2, 0]))
    # calculate the flow
    flow = reg_base.optical_flow_single(base, target, upsample_factor_yx=1, tile=0, round=0, chunks_yx=5, overlap=0.25)
    # check that the shape is correct
    ny, nx, nz = base.shape
    assert flow.shape == (3, ny, nx, nz)

    # check that the values are correct (check near the centre so we don't have to worry about edge effects)
    r = 0.1
    flow = np.round(flow)
    centre_idx = np.ix_(range(int( r * ny), int((1 - r) * ny)),
                        range(int( r * nx), int((1 - r) * nx)),
                        range(int( r * nz), int((1 - r) * nz)))
    correct_y = flow[0][centre_idx] == -3
    correct_x = flow[1][centre_idx] == -2
    correct_z = flow[2][centre_idx] == 0
    ny, nx, nz = ny // 2, nx // 2, nz // 2
    assert np.sum(correct_y) / (ny * nx * nz) > 0.9
    assert np.sum(correct_x) / (ny * nx * nz) > 0.9
    assert np.sum(correct_z) / (ny * nx * nz) > 0.9
