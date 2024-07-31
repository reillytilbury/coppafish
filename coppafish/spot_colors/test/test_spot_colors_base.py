import numpy as np

from coppafish.spot_colors.base import get_spot_colours_new


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
