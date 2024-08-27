import numpy as np
import scipy
import skimage
import torch

from coppafish.spot_colours.base import get_spot_colours


def test_get_spot_colours():
    """
    Function to test the get_spot_colours function from the spot_colours.base module.
    """
    # create some artificial data with 2 rounds, 3 channels and a 10 x 10 x 5 image
    rng = np.random.RandomState(0)
    tile_shape = 100, 100, 10
    n_rounds, n_channels = 2, 3
    images_aligned = rng.rand(n_rounds, n_channels, *tile_shape)
    # set values below 0.5 to 0, and above 0.5 to 1
    images_aligned = (images_aligned > 0.5).astype(np.float32)
    # smooth each round and channel independently
    for r in range(n_rounds):
        for c in range(n_channels):
            images_aligned[r, c] = skimage.filters.gaussian(images_aligned[r, c], sigma=5)

    # now we would like to move these images by applying the inverse transform of the one we want to apply
    # to the spot colours and then check if we can recover the original images
    affine = np.zeros((n_channels, 4, 3))
    affine[:, :3, :3] = np.eye(3)
    # repeat the affine transforms for each round
    affine = np.repeat(affine[None], n_rounds, axis=0)
    # initialise flow to be 0
    flow = np.zeros((n_rounds, 3, *tile_shape))

    # 1. check that the images are the same as the spot colours (no affine or flow applied)
    # get coords
    coords = np.array(np.meshgrid(*[np.arange(s) for s in tile_shape], indexing="ij"))
    yxz_base = coords.reshape(3, -1).T
    spot_colours = get_spot_colours(
        image=images_aligned[None], flow=flow[None], affine_correction=affine[None], yxz_base=yxz_base, tile=0
    )
    # reshape spot colours from n_spots x n_rounds x n_channels to n_y x n_x x n_z x n_rounds x n_channels
    spot_colours = spot_colours.reshape(*tile_shape, n_rounds, n_channels)
    # reorder spot colours array to n_rounds x n_channels x n_y x n_x x n_z
    spot_colours = np.transpose(spot_colours, (3, 4, 0, 1, 2))
    mid_z = tile_shape[2] // 2

    # # plot scatter plot of true vs predicted values for mid_z slice
    # import matplotlib.pyplot as plt
    # plt.scatter(x=images_aligned[:, :, :, :, mid_z].flatten(), y=spot_colours[:, :, :, :, mid_z].flatten())
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    # plt.title('True vs Predicted Values')
    # plt.show()
    #
    # open napari viewer to check the images
    # import napari
    # v = napari.Viewer()
    # v.add_image(images_aligned, name='Aligned Images', colormap='red', blending='additive',
    #             contrast_limits=np.nanpercentile(images_aligned, [1, 99]))
    # v.add_image(spot_colours, name='Spot Colours', colormap='green', blending='additive',
    #             contrast_limits=np.nanpercentile(spot_colours, [1, 99]))
    # v.add_image(images_aligned - spot_colours, name='Difference',
    #             contrast_limits=np.nanpercentile(images_aligned - spot_colours, [1, 99]))
    # napari.run()

    # check that the spot colours are the same as the original images
    assert np.allclose(images_aligned, spot_colours, atol=1e-6)

    # 2. check that the images are the same as the spot colours (affine applied + flow applied)
    # set these affine transforms to be shifts in y and x by 1, 2, 3 and scales in y and x by 0.9, 0.8
    # these are the transforms we need to apply to go from anchor to target, so we need to apply the inverse to the
    # images
    affine[:, :, 3, 0] = [1, 2, 3]
    affine[:, :, 3, 1] = [1, 2, 3]
    affine[:, :, 0, 0] = 0.9
    affine[:, :, 1, 1] = 0.8

    # set flow to be +1 shift in z for round 0 and +2 shift in z for round 1
    flow[0, 2] = 1
    flow[1, 2] = 2

    # define the inverse warp
    warp_inv = np.array([coords - flow[r] for r in range(n_rounds)])

    # the transform we will apply to align is A(F(x)), so to disalign apply A^(-1)(F^(-1)(x))
    images_disaligned = np.zeros_like(images_aligned)
    for r in range(n_rounds):
        for c in range(n_channels):
            # invert the affine transform
            affine_rc = np.vstack([affine[r, c].T, [0, 0, 0, 1]])
            affine_rc = np.linalg.inv(affine_rc)
            images_disaligned[r, c] = scipy.ndimage.affine_transform(
                images_aligned[r, c], affine_rc, order=1, cval=np.nan
            )
            images_disaligned[r, c] = skimage.transform.warp(images_disaligned[r, c], warp_inv[r], order=1, cval=np.nan)

    # now we want to get the spot colours from the disaligned images
    spot_colours = get_spot_colours(
        image=images_disaligned[None], flow=flow[None], affine_correction=affine[None], yxz_base=yxz_base, tile=0
    )
    # reshape spot colours from n_spots x n_rounds x n_channels to n_y x n_x x n_z x n_rounds x n_channels
    spot_colours = spot_colours.reshape(*tile_shape, n_rounds, n_channels)
    # reorder spot colours array to n_rounds x n_channels x n_y x n_x x n_z
    spot_colours = np.transpose(spot_colours, (3, 4, 0, 1, 2))

    # plot scatter plot of true vs predicted values for mid_z slice
    # import matplotlib.pyplot as plt
    # plt.scatter(x=images_aligned[:, :, :, :, mid_z].flatten(), y=spot_colours[:, :, :, :, mid_z].flatten())
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    # plt.title('True vs Predicted Values')
    # plt.show()

    # open napari viewer to check the images
    # import napari
    # v = napari.Viewer()
    # v.add_image(images_aligned, name='Aligned Images', colormap='red', blending='additive',
    #             contrast_limits=np.nanpercentile(images_aligned, [1, 99]))
    # v.add_image(images_disaligned, name='Disaligned Images', visible=False,
    #             contrast_limits=np.nanpercentile(images_disaligned, [1, 99]))
    # v.add_image(spot_colours, name='Spot Colours', colormap='green', blending='additive',
    #             contrast_limits=np.nanpercentile(spot_colours, [1, 99]))
    # napari.run()

    # check that the spot colours are the same as the original images
    assert np.nanmax(np.abs(images_aligned - spot_colours)[:, :, :, :, mid_z]) < 0.01


def test_grid_sample():
    """
    Simple test of the grid sampling function that makes our heads hurt less.
    """
    # set up data
    brain = skimage.data.brain()
    # convert this from zyx to yxz to match our data
    brain = np.moveaxis(brain, 0, -1)
    im_sz = np.array(brain.shape)

    # set some points to sample
    rng = np.random.RandomState(0)
    random_points = rng.randint(0, im_sz - 1, (100, 3))
    # convert to torch tensors
    brain = torch.tensor(brain).float()
    random_points = torch.tensor(random_points).float()
    true_vals = brain[random_points[:, 0].long(), random_points[:, 1].long(), random_points[:, 2].long()].numpy()

    # sample the points using grid_sample

    # input will be of size [N, M, D, H, W] with
    # N = number of images in the batch (1 in this case)
    # M = number of channels in the image (1 in this case)
    # D = depth of the image (the y dimension, 256)
    # H = height of the image (the x dimension, 256)
    # W = width of the image (the z dimension, 10)

    # grid will be of size [N, D', H', W', 3] with
    # N = number of images in the batch (1 in this case)
    # D' = Depth of output grid (we use this as number of points to sample, 100 in this case)
    # H' = Height of output grid (1 in this case)
    # W' = width of the output grid (1 in this case)
    # 3 = 3D coordinates of the points to sample (z, x, y)

    # grid values should be between -1 and 1, so we need to scale the random points to be between -1 and 1
    random_points = 2 * random_points / (im_sz - 1) - 1
    random_points = random_points[:, [2, 1, 0]]  # convert from yxz to zxy
    random_points = random_points.float()  # convert to float
    predicted_vals = torch.nn.functional.grid_sample(
        input=brain[None, None, :, :, :], grid=random_points[None, :, None, None, :], mode="nearest"
    ).squeeze()
    # reshape the predicted values to be the same shape as the true values and turn into numpy
    predicted_vals = predicted_vals.numpy()

    # check that the predicted values are the same as the true values
    # import matplotlib.pyplot as plt
    # plt.scatter(x=true_vals, y=predicted_vals)
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    # plt.title('True vs Predicted Values')
    # plt.show()

    assert np.allclose(true_vals, predicted_vals, atol=1e-6)
