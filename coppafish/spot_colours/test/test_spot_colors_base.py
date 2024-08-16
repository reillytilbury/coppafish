import numpy as np
import skimage
import scipy
from coppafish.spot_colours.base import get_spot_colours_new, get_spot_colours


def test_get_spot_colours_new() -> None:
    rng = np.random.RandomState(0)
    tile_shape = 2, 3, 4
    image = rng.rand(*tile_shape).astype(np.float64)
    all_images = image[None, None, None]
    use_channels = [0]
    dapi_channel = 2
    tile = 0
    round = 0
    channels = (0,)
    yxz = None
    registration_type = "flow_and_icp"
    dtype = np.float64
    # Test the optical flow is working using a positive x shift of one pixel on the image.
    flow = np.zeros((1, 1, 3) + tile_shape, np.float16)
    flow[:, :, 1] = -1
    # Add one flow shift of -1.5 in the z direction.
    flow[:, :, 1, 1, 2, 3] = 0
    flow[:, :, 2, 1, 2, 3] = 1.5
    icp_correction = np.eye(4, 3)[np.newaxis, np.newaxis, np.newaxis]
    channel_correction = np.eye(4, 3)[np.newaxis, np.newaxis]
    result = get_spot_colours_new(
        all_images=all_images,
        flow=flow,
        icp_correction=icp_correction,
        channel_correction=channel_correction,
        use_channels=use_channels,
        dapi_channel=dapi_channel,
        tile=tile,
        round=round,
        channels=channels,
        yxz=yxz,
        registration_type=registration_type,
        dtype=dtype,
    )
    assert type(result) is np.ndarray
    assert result.shape == (1, np.prod(tile_shape))
    assert result.dtype == dtype
    result = result.reshape(tile_shape)
    assert np.allclose(result[:, :-1], image[:, 1:])
    assert np.isnan(result[:-1, -1, :-1]).all()
    assert np.isnan(result[:, -1]).sum() == 7
    assert np.isclose(result[1, 2, 3], (image[1, 2, 1] + image[1, 2, 2]) / 2)

    # Test the gathering of a subset of pixels.
    yxz = np.zeros((4, 3), np.int32)
    yxz[0] = [0, 0, 2]
    yxz[1] = [0, 0, 2]
    yxz[2] = [1, 0, 2]
    yxz[3] = [1, 0, 2]
    result = get_spot_colours_new(
        all_images=all_images,
        flow=flow,
        icp_correction=icp_correction,
        channel_correction=channel_correction,
        use_channels=use_channels,
        dapi_channel=dapi_channel,
        tile=tile,
        round=round,
        channels=channels,
        yxz=yxz,
        registration_type=registration_type,
        dtype=dtype,
    )[0]
    assert np.allclose(result[0], image[0, 1, 2])
    assert np.allclose(result[1], image[0, 1, 2])
    assert np.allclose(result[2], image[1, 1, 2])
    assert np.allclose(result[3], image[1, 1, 2])

    # Test the affine transform with a y and x transpose.
    tile_shape = 3, 3, 4
    image = rng.rand(*tile_shape).astype(np.float64)
    all_images = image[None, None, None]
    yxz = None
    flow = np.zeros((1, 1, 3) + tile_shape, np.float16)
    icp_correction = np.zeros((4, 3))
    icp_correction[0, 1] = 1
    icp_correction[1, 0] = 1
    icp_correction[2, 2] = 1
    icp_correction = icp_correction[np.newaxis, np.newaxis, np.newaxis]
    dtype = np.float32
    result = get_spot_colours_new(
        all_images=all_images,
        flow=flow,
        icp_correction=icp_correction,
        channel_correction=channel_correction,
        use_channels=use_channels,
        dapi_channel=dapi_channel,
        tile=tile,
        round=round,
        channels=channels,
        yxz=yxz,
        registration_type=registration_type,
        dtype=dtype,
    )[0]
    assert result.dtype == dtype
    result = result.reshape(tile_shape)
    assert np.allclose(result, image.swapaxes(0, 1))

    # Affine and optical flow shift together.
    flow = np.zeros((1, 1, 3) + tile_shape, np.float16)
    flow[:, :, 0] = -1
    result = get_spot_colours_new(
        all_images=all_images,
        flow=flow,
        icp_correction=icp_correction,
        channel_correction=channel_correction,
        use_channels=use_channels,
        dapi_channel=dapi_channel,
        tile=tile,
        round=round,
        channels=channels,
        yxz=yxz,
        registration_type=registration_type,
        dtype=dtype,
    )[0]
    assert result.dtype == dtype
    result = result.reshape(tile_shape)
    assert np.allclose(result[:, :-1], image.swapaxes(0, 1)[:, 1:])
    assert np.isnan(result[:, -1]).all()


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
    # set these affine transforms to be shifts in y and x by 1, 2, 3 and scales in y and x by 0.9, 0.8
    # these are the transforms we need to apply to go from anchor to target, so we need to apply the inverse to the
    # images
    # affine[:, 3, 0] = [1, 2, 3]
    # affine[:, 3, 1] = [1, 2, 3]
    # affine[:, 0, 0] = 0.9
    # affine[:, 1, 1] = 0.8
    # repeat the affine transforms for each round
    affine = np.repeat(affine[None], n_rounds, axis=0)

    # define flow shifts to be 1 shift in z for round 0 and 2 shifts in z for round 1
    flow = np.zeros((n_rounds, 3, *tile_shape))
    # flow[0, 2] = 1
    # flow[1, 2] = 2

    # get coords and define warps (coords + flow) for each round
    coords = np.array(np.meshgrid(*[np.arange(s) for s in tile_shape], indexing='ij'))
    warp = np.array([coords + flow[r] for r in range(n_rounds)])

    # the transform we will apply to align is A(F(x)), so to disalign apply A^(-1)(F^(-1)(x))
    images_disaligned = np.zeros_like(images_aligned)
    for r in range(n_rounds):
        for c in range(n_channels):
            # scipy ndimage affine automatically applies the inverse transform
            images_disaligned[r, c] = scipy.ndimage.affine_transform(images_aligned[r, c], affine[r, c].T, order=0,
                                                                     cval=np.nan)
            # skimage transform warp will also apply the inverse of the flow
            images_disaligned[r, c] = skimage.transform.warp(images_disaligned[r, c], warp[r], order=0, cval=np.nan)

    # now we want to get the spot colours from the disaligned images
    yxz_base = coords.reshape(3, -1).T
    spot_colours = get_spot_colours(image=images_disaligned[None], flow=flow[None], icp_correction=affine[None],
                                    yxz_base=yxz_base).numpy()
    # check the spot_colours against the image
    spot_colours_true = np.zeros_like(spot_colours)
    for r in range(n_rounds):
        for c in range(n_channels):
            spot_colours_true[:, r, c] = images_aligned[r, c].reshape(-1)
    # reshape spot colours from n_spots x n_rounds x n_channels to n_y x n_x x n_z x n_rounds x n_channels
    spot_colours = spot_colours.reshape(*tile_shape, n_rounds, n_channels)
    # reorder spot colours tensor to n_rounds x n_channels x n_y x n_x x n_z
    spot_colours = spot_colours.permute(3, 4, 0, 1, 2)
