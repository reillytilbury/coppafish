import numpy as np
from skimage import data

from coppafish.register import preprocessing as reg_pre


def test_split_3d_image():
    # Set up data (256, 256, 10)
    brain = data.brain()
    # swap axes to match the expected shape
    brain = np.swapaxes(brain, 0, 2)
    # Test the function
    brain_split, pos = reg_pre.split_image(brain, n_subvols_yx=5, overlap=0.25)
    # Test that the shape is correct
    expected_size_yx = 64
    assert brain_split.shape == (25, expected_size_yx, expected_size_yx, 10)
    # Test that the values are correct
    assert np.allclose(brain_split[0], brain[:expected_size_yx, :expected_size_yx])
    assert np.allclose(brain_split[-1], brain[-expected_size_yx:, -expected_size_yx:])

    # test on some images of the same size that we use in the registration
    # Set up data (576, 576, 10)
    rng = np.random.RandomState(23)
    im = rng.rand(576, 576, 10)
    # Test the function
    im_split, pos = reg_pre.split_image(im, n_subvols_yx=4, overlap=1/3)
    # Test that the shape is correct
    expected_size_yx = 192
    assert im_split.shape == (16, expected_size_yx, expected_size_yx, 10)
    # Test that the values are correct
    assert np.allclose(im_split[0], im[:expected_size_yx, :expected_size_yx])
    assert np.allclose(im_split[-1], im[-expected_size_yx:, -expected_size_yx:])


def test_custom_shift():
    # set up data
    im = np.sum(data.astronaut(), axis=2)
    shift = np.array([10, 20]).astype(int)
    im_new = reg_pre.custom_shift(im, shift)
    # check that the shape is correct
    assert im_new.shape == im.shape
    # check that the values are correct
    assert np.allclose(im_new[10:, 20:], im[:-10, :-20])
    assert np.allclose(im_new[:10, :20], 0)


def test_merge_subvols():
    # Set up data (256, 256, 10)
    brain = data.brain().astype(float)
    # swap axes to match the expected shape
    brain = np.swapaxes(brain, 0, 2)
    # Test the function
    brain_split, pos = reg_pre.split_image(brain, n_subvols_yx=5, overlap=0.25)
    brain_merged = reg_pre.merge_subvols(im_split=brain_split, positions=pos, output_shape=brain.shape, overlap=0.25)
    # Test that the shape is correct
    assert brain_merged.shape == brain.shape
    # Test that the values are correct
    assert np.allclose(brain_merged, brain)

    # test on some images of the same size that we use in the registration
    rng = np.random.RandomState(61)
    # Set up data (576, 576, 10)
    im = rng.rand(576, 576, 10)
    # Test the function
    im_split, pos = reg_pre.split_image(im, n_subvols_yx=4, overlap=1/3)
    im_merged = reg_pre.merge_subvols(im_split=im_split, positions=pos, output_shape=im.shape, overlap=1/3)
    # Test that the shape is correct
    assert im_merged.shape == im.shape
    # Test that the values are correct
    assert np.allclose(im_merged, im)