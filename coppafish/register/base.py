import time
from typing import Optional, Tuple

import joblib
import nd2
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
import skimage
import torch
from tqdm import tqdm
import zarr

from .. import log, utils
from ..register import preprocessing


def optical_flow_register(
    target: np.ndarray,
    base: np.ndarray,
    tile: int,
    round: int,
    raw_loc: str,
    corr_loc: str,
    smooth_loc: str,
    sample_factor_yx: int = 4,
    chunks_yx: int = 4,
    overlap: float = 1 / 3,
    window_radius: int = 5,
    smooth_sigma: list = [10, 10, 2],
    clip_val: np.ndarray = np.array([40, 40, 15]),
    n_cores: Optional[int] = None,
):
    """
    Function to carry out optical flow registration on a single tile and round.

    Optical flow is computed using an iterative Lucas-Kanade method with 10 iterations and a window radius specified
    by the user.

    Flow is interpolated based on the correlation between the base and target images. The correlation is computed
    as a dot product between the base and target images within a small window of each pixel.

    Args:
        target: np.ndarray size [n_y, n_x, n_z] of the target image (this will be the round image)
        base: np.ndarray size [n_y, n_x, n_z] of the base image (this will be the anchor image)
        tile: int specifying the tile index
        round: int specifying the round index
        raw_loc: str specifying the location to save/ load the optical flow
        corr_loc: str specifying the location to save/ load the correlation
        smooth_loc: str specifying the location to save/ load the smoothed flow
        sample_factor_yx: int specifying how much to downsample the images in y and x before computing the optical flow
        chunks_yx: int specifying the number of subvolumes to split the downsampled images into in y and x
        overlap: float specifying the overlap between subvolumes
        window_radius: int specifying the window radius for the optical flow algorithm and correlation calculation
        (Note that this is the radius on the downsampled image, so a radius of 5 with a downsample factor of 4 will
        correspond to a radius of 20 on the original image)
        smooth_sigma: float specifying the standard deviation of the Gaussian filter to be used for smoothing the flow
        clip_val: np.ndarray size [3] of the clip value for the optical flow in y, x and z
        n_cores (int, optional): maximum cpu cores to use in parallel when computing optical flow. Default: all found
            cpu cores.

    Returns:
        Tuple containing path to optical flow, correlation, and smoothed flow zarr arrays respectively.
    """
    # convert the images to float32
    target = target.astype(np.float32)
    base = base.astype(np.float32)
    # down-sample the images
    target = skimage.measure.block_reduce(target, (sample_factor_yx, sample_factor_yx, 1), np.mean)
    base = skimage.measure.block_reduce(base, (sample_factor_yx, sample_factor_yx, 1), np.mean)
    # smooth the images slightly
    target = gaussian_filter(target, 1)
    base = gaussian_filter(base, 1)
    # update clip_val based on down-sampling
    clip_val = np.array(clip_val, dtype=np.float32)
    clip_val[:2] = clip_val[:2] / sample_factor_yx

    # compute the optical flow for each round/channel
    flow = optical_flow_single(
        base=base,
        target=target,
        tile=tile,
        round=round,
        window_radius=window_radius,
        clip_val=clip_val,
        chunks_yx=chunks_yx,
        overlap=overlap,
        loc=raw_loc,
        n_cores=n_cores,
    )
    # compute the correlation between the base and target images within a small window of each pixel
    correlation, _ = flow_correlation(
        base=base,
        target=target,
        tile=tile,
        round=round,
        flow=flow,
        loc=corr_loc,
    )
    # smooth the flow
    interpolate_flow(
        flow,
        correlation,
        tile=tile,
        round=round,
        sigma=smooth_sigma,
        window_radius=window_radius,
        loc=smooth_loc,
    )


def optical_flow_single(
    base: np.ndarray,
    target: np.ndarray,
    tile: int,
    round: int,
    window_radius: int = 5,
    clip_val: np.ndarray = np.array([10, 10, 15]),
    upsample_factor_yx: int = 4,
    chunks_yx: int = 4,
    overlap: float = 1 / 3,
    n_cores: Optional[int] = None,
    loc: str = "",
) -> np.ndarray:
    """
    Function to carry out optical flow registration on 2 3D images.
    Args:
        base: np.ndarray size [n_y, n_x, n_z] of the base image
        target: np.ndarray size [n_y, n_x, n_z] of the target image
        tile: int specifying the tile index
        round: int specifying the round index
        window_radius: int specifying the window radius for the optical flow algorithm
        clip_val: np.ndarray size [3] of the clip value for the optical flow in y, x and z
        upsample_factor_yx: int specifying how much to upsample the optical flow in y and x
        chunks_yx: int specifying the number of subvolumes to split the images into in y and x
        overlap: float specifying the overlap between subvolumes
        n_cores (int, optional): int specifying the number of cores to use for parallel processing. Default: CPU core
            count available.
        loc: str specifying the location to save/ load the optical flow
    Returns:
        flow: np.ndarray size [3, n_y, n_x, n_z] of the optical flow
    """
    t_start = time.time()
    # start by ensuring images are float32
    base = base.astype(np.float32)
    target = target.astype(np.float32)

    # First, correct for a shift in the images
    base_windowed, target_windowed = preprocessing.window_image(base), preprocessing.window_image(target)
    shift = skimage.registration.phase_cross_correlation(
        reference_image=target_windowed, moving_image=base_windowed, upsample_factor=10
    )[0]
    # round this shift and make it an integer
    shift = np.round(shift).astype(int)
    # adjust the base image by this shift
    base = preprocessing.custom_shift(base, shift)

    # split the images into subvolumes and run optical flow on each subvol in parallel
    target_sub, pos = preprocessing.split_image(im=target, n_subvols_yx=chunks_yx, overlap=overlap)
    base_sub, _ = preprocessing.split_image(im=base, n_subvols_yx=chunks_yx)
    # Normalise each subvolume to have mean 1 (this is just so that the images are in a similar range in each subvol)
    target_sub = target_sub / np.mean(target_sub, axis=(1, 2, 3))[:, None, None, None]
    base_sub = base_sub / np.mean(base_sub, axis=(1, 2, 3))[:, None, None, None]

    # compute the optical flow (in parallel)
    if n_cores is None:
        n_cores = utils.system.get_core_count()
    log.info(f"Computing optical flow using {n_cores} cores")
    flow_sub = joblib.Parallel(n_jobs=n_cores)(
        joblib.delayed(skimage.registration.optical_flow_ilk)(
            target_sub[n], base_sub[n], radius=window_radius, prefilter=True
        )
        for n in range(pos.shape[0])
    )
    flow_sub = np.array(flow_sub)  # convert list to numpy array. Shape: (n_subvols, 3, n_y, n_x, n_z)

    # Now that we have the optical flow for each subvolume, we need to merge them back together
    flow = np.array(
        [
            preprocessing.merge_subvols(
                im_split=flow_sub[:, i], positions=pos, output_shape=target.shape, overlap=overlap
            )
            for i in range(3)
        ]
    )
    # clip the flow
    flow = np.array([np.clip(flow[i], -clip_val[i], clip_val[i]) for i in range(3)])
    # change the sign of the flow
    flow = -flow
    # add back the shift
    flow = np.array([flow[i] + shift[i] for i in range(3)])
    # upsample the flow
    upsample_factor = (upsample_factor_yx, upsample_factor_yx, 1)
    flow_up = np.array(
        [
            upsample_yx(flow[i], upsample_factor_yx, order=1) * upsample_factor[i]
            for i in tqdm(range(3), desc="Upsampling raw flow")
        ],
        dtype=np.float16,
    )
    # save the flow
    if loc:
        # save in yxz format
        zarray = zarr.open_array(loc, mode="r+")
        zarray[tile, round] = flow_up
    t_end = time.time()
    log.info("Optical flow computation took " + str(t_end - t_start) + " seconds")

    return flow


def flow_correlation(
    base: np.ndarray,
    target: np.ndarray,
    tile: int,
    round: int,
    flow: np.ndarray,
    upsample_factor_yx: int = 4,
    loc: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the correlation between the base and target images. The correlation is just the dot product of the base and
    target images at each pixel.

    Args:
        base: n_y x n_x x n_z array of the base image
        target: n_y x n_x x n_z array of the target image
        flow: 3 x n_y x n_x x n_z array of flow in y, x and z
        upsample_factor_yx: int specifying the upsample factor in y and x
        loc: str specifying the location to save/ load the correlation

    Returns:
        correlation: n_y x n_x x n_z array of correlation coefficients
        upsampled_correlation: n_up_y x n_up_x x n_up_z array of upsampled correlation coefficients. Returns none if
            correlation is already saved at `loc`.
    """
    t_start = time.time()
    ny, nx, nz = target.shape
    # apply the flow to the base image and compute the correlation between th shifted base and the target image
    coords = np.array(np.meshgrid(range(ny), range(nx), range(nz), indexing="ij"), dtype=np.float32)
    base_warped = skimage.transform.warp(base, coords - flow, order=0, mode="constant", cval=0)
    del coords, base, flow

    # compute the correlation
    correlation = base_warped * target
    # normalise to be between 0 and 1
    correlation_min, correlation_max = np.nanpercentile(correlation, [25, 99])
    correlation = (correlation - correlation_min) / (correlation_max - correlation_min)
    correlation = np.clip(correlation, 0, 1)
    # up-sample the correlation
    correlation_upsampled = upsample_yx(correlation, upsample_factor_yx, order=1).astype(np.float16)
    # save the correlation
    if loc:
        # save in yxz format
        zarray = zarr.open_array(loc, mode="r+")
        zarray[tile, round] = correlation_upsampled
    t_end = time.time()
    log.info("Computing correlation took " + str(t_end - t_start) + " seconds")
    return correlation, correlation_upsampled


def interpolate_flow(
    flow: np.ndarray,
    correlation: np.ndarray,
    tile: int,
    round: int,
    sigma: list = [10, 10, 5],
    window_radius: int = 8,
    upsample_factor_yx: int = 4,
    loc: str = "",
):
    """
    Interpolate the flow based on the correlation between the base and target images.

    The interpolation is done between by smoothing the correlation weighted flow and dividing by the smoothed
    correlation. This replaces the flow in regions of low correlation with a smoothed version of the flow in regions of
    high correlation.

    This smoothing is done in 3D with sigma much smaller in z than xy to avoid getting rid of variable shifts in z.

    Args:
        flow: 3 x n_y x n_x x n_z array of flow in y, x and z
        correlation: n_y x n_x x n_z array of correlation coefficients
        tile: int specifying the tile index
        round: int specifying the round index
        sigma: standard deviation of the Gaussian filter to be used for smoothing the flow
        window_radius: int specifying the window radius for the optical flow algorithm
        upsample_factor_yx: int specifying the upsample factor in y and x
        loc: str specifying the location to save/ load the interpolated flow
    """
    time_start = time.time()
    # smooth the correlation
    correlation_smooth = gaussian_filter(correlation, sigma, truncate=6)
    flow_smooth = np.zeros_like(flow)

    # smooth the correlation weighted flow in yx
    for i in range(3):
        flow_smooth[i] = gaussian_filter(flow[i] * correlation, sigma, truncate=6) / correlation_smooth

    # When dist(z, n_z) <= window_radius, the shift in z is very biased towards 0.
    # We correct for this by predicting the z-shift from each coordinate in y, x, and z
    # (excluding the top 2 * window_radius z-planes) using linear regression.
    # we then replace the z_flow with the linearly predicted z_flow

    # grab samples for linear regression
    flow_z_crop = flow_smooth[2][::10, ::10, : -2 * window_radius]
    # grab coords of sample points
    coords = np.array(np.meshgrid(10 * np.arange(flow_z_crop.shape[0]),
                                  10 * np.arange(flow_z_crop.shape[1]),
                                  np.arange(flow_z_crop.shape[2]),
                                  indexing="ij"), dtype=np.float32)

    # reshape coords and flow_z_crop and pad coords with 1s for linear regression
    coords = coords.reshape(3, -1).T
    coords_pad = np.pad(coords, [(0, 0), (0, 1)], constant_values=1)
    flow_z_crop = flow_z_crop.reshape(-1)
    # compute linear regression
    coefficients = np.linalg.lstsq(a=coords_pad, b=flow_z_crop, rcond=None)[0]

    # now we can use this to predict the flow in z
    coords = np.array(np.meshgrid(range(flow_smooth.shape[1]),
                                  range(flow_smooth.shape[2]),
                                  range(flow_smooth.shape[3]),
                                  indexing="ij"), dtype=np.float32)
    coords = coords.reshape(3, -1).T
    coords_pad = np.pad(coords, [(0, 0), (0, 1)], constant_values=1)
    flow_smooth[2] = (coords_pad @ coefficients).reshape(flow_smooth.shape[1],
                                                         flow_smooth.shape[2],
                                                         flow_smooth.shape[3])
    del coords, coords_pad, flow_z_crop, coefficients

    # remove nan values
    flow_smooth = np.nan_to_num(flow_smooth)
    upsample_factor = (upsample_factor_yx, upsample_factor_yx, 1)
    # upsample the flow before saving
    flow_smooth = np.array(
        [
            upsample_yx(flow_smooth[i], upsample_factor_yx, order=1) * upsample_factor[i]
            for i in tqdm(range(3), desc="Upsampling smooth flow")
        ],
        dtype=np.float16,
    )
    # save the flow
    if loc:
        # save in yxz format
        zarray = zarr.open_array(loc, mode="r+")
        zarray[tile, round] = flow_smooth
    time_end = time.time()
    log.info("Interpolating flow took " + str(time_end - time_start) + " seconds")
    return flow_smooth


def upsample_yx(im: np.ndarray, factor: float = 4, order: int = 1) -> np.ndarray:
    """
    Function to upsample an 3D image in yx.
    Args:
        im: np.ndarray size [n_y, n_x, n_z] of the image to upsample
        factor: float specifying the factor to upsample by in y and x
        order: int specifying the order of the interpolation
    Returns:
        im_up: np.ndarray size [n_y * factor, n_x * factor, n_z] of the upsampled image
    """
    # upsampling is quicker when we do it in 2D
    im_up = np.zeros((im.shape[0] * factor, im.shape[1] * factor, im.shape[2]), dtype=im.dtype)
    for z in range(im.shape[2]):
        im_up[:, :, z] = scipy.ndimage.zoom(im[:, :, z], factor, order=order)
    return im_up


def channel_registration(
    fluorescent_bead_path: str = None, anchor_cam_idx: int = 3, n_cams: int = 4, bead_radii: list = [10, 11, 12]
) -> np.ndarray:
    """
    Function to carry out channel registration using fluorescent beads. This function assumes that the fluorescent
    bead images are 2D images and that there is one image for each camera.

    Fluorescent beads are assumed to be circles and their centres are detected using the Hough transform.
    The centres of the detected circles are then used as point clouds to compute the affine transform between each
    camera and the round_reg_channel_cam via ICP.

    Args:
        fluorescent_bead_path: Path to fluorescent beads directory containing the fluorescent bead images.
        If none then we assume that the channels are registered to each other and just set channel_transforms to
        identity matrices
        anchor_cam_idx: int index of the camera to use as the anchor camera
        n_cams: int number of cameras
        bead_radii: list of possible bead radii

    Returns:
        transform: n_cams x 3 x 4 array of affine transforms taking anchor camera to each other camera
    """
    transform = np.zeros((n_cams, 4, 3))
    # First check if the fluorescent beads path exists. If not, we assume that the channels are registered to each
    # other and just set channel_transforms to identity matrices
    if fluorescent_bead_path is None:
        # Set registration_data['channel_registration']['channel_transform'][c] = np.eye(3) for all channels c
        for c in range(n_cams):
            transform[c] = np.eye(4, 3)
        log.warn("Fluorescent beads directory does not exist. Assuming that all channels are registered to each other.")
        return transform

    # open the fluorescent bead images as nd2 files
    with nd2.ND2File(fluorescent_bead_path) as fbim:
        fluorescent_beads = fbim.asarray()

    # if the fluorescent bead images are 4D, take the middle z-plane
    if fluorescent_beads.ndim == 4:
        nz = fluorescent_beads.shape[0]
        fluorescent_beads = fluorescent_beads[int(nz / 2)]
    # if fluorescent bead images are for all channels, just take one from each camera
    if fluorescent_beads.shape[0] == 28:
        fluorescent_beads = fluorescent_beads[[0, 9, 18, 23]]

    # Typically, the image histogram will be bimodal, with the second peak corresponding to the fluorescent beads.
    # The isodata thresholding method is used to separate the beads from the background.
    # We will set the pixels above the isodata threshold to a high value.
    for i in range(n_cams):
        bead_pixels = fluorescent_beads[i] > skimage.filters.threshold_isodata(fluorescent_beads[i])
        fluorescent_beads[i][bead_pixels] = skimage.filters.threshold_isodata(fluorescent_beads[i])

    # correct for shifts with phase cross correlation
    target_cams = [i for i in range(n_cams) if i != anchor_cam_idx]
    initial_transform = np.repeat(np.eye(4, 3)[None, :, :], n_cams, axis=0)
    for i in target_cams:
        shift_yx = skimage.registration.phase_cross_correlation(
            reference_image=fluorescent_beads[i], moving_image=fluorescent_beads[anchor_cam_idx]
        )[0]
        initial_transform[i][-1, :-1] = shift_yx

    # Now we'll turn each image into a point cloud
    bead_point_clouds = []
    for i in range(n_cams):
        edges = skimage.feature.canny(fluorescent_beads[i], sigma=3, low_threshold=10, high_threshold=50)
        hough_res = skimage.transform.hough_circle(edges, bead_radii)
        accums, cx, cy, radii = skimage.transform.hough_circle_peaks(
            hough_res, bead_radii, min_xdistance=10, min_ydistance=10
        )
        cy, cx = cy.astype(int), cx.astype(int)

        # Remove points that are below the noise threshold
        values = fluorescent_beads[i][cy, cx]
        cy_rand, cx_rand = (
            np.random.randint(0, fluorescent_beads[i].shape[0] - 1, 100),
            np.random.randint(0, fluorescent_beads[i].shape[1] - 1, 100),
        )
        noise = np.mean(fluorescent_beads[i][cy_rand, cx_rand])
        keep = values > noise
        cy, cx = cy[keep], cx[keep]
        # Add the points to the point cloud
        bead_point_clouds.append(np.vstack((cy, cx)).T)

    # Now convert the point clouds from yx to yxz. This is because our ICP algorithm assumes that the point clouds
    # are in 3D space
    bead_point_clouds = [
        np.hstack((bead_point_clouds[i], np.zeros((bead_point_clouds[i].shape[0], 1)))) for i in range(n_cams)
    ]

    # Set up ICP
    transform = np.zeros((n_cams, 4, 3))
    mse = np.zeros((n_cams, 50))
    # target_cams = [i for i in range(n_cams) if i != anchor_cam_idx]
    with tqdm(total=len(target_cams)) as pbar:
        for i in target_cams:
            pbar.set_description("Running ICP for camera " + str(i))
            # Set initial transform to identity (shift already accounted for)
            # Run ICP
            transform[i], _, mse[i], converged = icp(
                yxz_base=bead_point_clouds[anchor_cam_idx],
                yxz_target=bead_point_clouds[i],
                start_transform=initial_transform[i],
                n_iters=50,
                dist_thresh_yx=5,
                dist_thresh_z=5,
            )
            if not converged:
                transform[i] = np.eye(4, 3)
                log.error(Warning("ICP did not converge for camera " + str(i) + ". Replacing with identity."))
            pbar.update(1)

    # Need to add in z coord info as not accounted for by registration due to all coords being equal
    transform[:, 2, 2] = 1
    transform[anchor_cam_idx] = np.eye(4, 3)

    return transform


# Function which runs a single iteration of the icp algorithm
def get_transform(
    yxz_base: np.ndarray,
    yxz_target: np.ndarray,
    transform_old: np.ndarray,
    dist_thresh_yx: float,
    dist_thresh_z: float,
    robust=False,
):
    """
    This finds the affine transform that transforms ```yxz_base``` such that the distances between the neighbours
    with ```yxz_target``` are minimised.

    Args:
        yxz_base: ```float [n_base_spots x 3]```.
            Coordinates of spots you want to transform.
        transform_old: ```float [4 x 3]```.
            Affine transform found for previous iteration of PCR algorithm.
        yxz_target: ```float [n_target_spots x 3]```.
            Coordinates of spots in image that you want to transform ```yxz_base``` to.
        dist_thresh_yx: distance threshold in y and x directions. If neighbours closer than this, they are used to
            compute the new transform.
        dist_thresh_z:
        robust: Boolean option to make regression robust. Selecting true will result in the algorithm maximising
            correntropy as opposed to minimising mse.

    Returns:
        - ```transform``` - ```float [4 x 3]```.
            Updated affine transform.
        - ```neighbour``` - ```int [n_base_spots]```.
            ```neighbour[i]``` is index of coordinate in ```yxz_target``` to which transformation of
            ```yxz_base[i]``` is closest.
        - ```n_matches``` - ```int```.
            Number of neighbours which have distance less than ```dist_thresh```.
        - ```error``` - ```float```.
            Average distance between ```neighbours``` below ```dist_thresh```.
    """
    # Scale the points down in x, y and z so that we can use a dist thresh of 1
    scale_factor = np.array([dist_thresh_yx, dist_thresh_yx, dist_thresh_z])[None, :]
    # Step 1 computes matching, since yxz_target is a subset of yxz_base, we will loop through yxz_target and find
    # their nearest neighbours within yxz_transform, which is the initial transform applied to yxz_base
    yxz_base_pad = np.pad(yxz_base, [(0, 0), (0, 1)], constant_values=1)
    yxz_transform = yxz_base_pad @ transform_old
    # scale down
    yxz_base, yxz_target, yxz_transform = (
        yxz_base / scale_factor,
        yxz_target / scale_factor,
        yxz_transform / scale_factor,
    )
    yxz_transform_tree = scipy.spatial.KDTree(yxz_transform)
    # the next query works the following way. For each point in yxz_target, we look for the closest neighbour in the
    # anchor, which we have now applied the initial transform to. If this is below dist_thresh, we append its distance
    # to distances and append the index of this neighbour to neighbour
    distances, neighbour = yxz_transform_tree.query(yxz_target, distance_upper_bound=1)
    # scale the points back up
    yxz_base, yxz_target = yxz_base * scale_factor, yxz_target * scale_factor
    yxz_base_pad = np.pad(yxz_base, [(0, 0), (0, 1)], constant_values=1)
    neighbour = neighbour.flatten()
    distances = distances.flatten()
    use = distances < 1
    distances[~use] = 1
    n_matches = np.sum(use)
    error = np.sqrt(np.mean(distances**2))
    base_pad_use = yxz_base_pad[neighbour[use], :]
    target_use = yxz_target[use, :]

    if not robust:
        transform = np.linalg.lstsq(base_pad_use, target_use, rcond=None)[0]
    else:
        sigma = dist_thresh_yx / 2
        target_pad_use = np.pad(target_use, [(0, 0), (0, 1)], constant_values=1)
        D = np.diag(np.exp(-0.5 * (np.linalg.norm(base_pad_use @ transform_old - target_use, axis=1) / sigma) ** 2))
        transform = (target_pad_use.T @ D @ base_pad_use @ np.linalg.inv(base_pad_use.T @ D @ base_pad_use))[:3, :4].T

    return transform, neighbour, n_matches, error


# Simple ICP implementation, calls above until no change
def icp(yxz_base, yxz_target, dist_thresh_yx, dist_thresh_z, start_transform, n_iters, robust=False):
    """
    Applies n_iters rounds of the above least squares regression
    Args:
        yxz_base: ```float [n_base_spots x 3]```.
            Coordinates of spots you want to transform.
        yxz_target: ```float [n_target_spots x 3]```.
            Coordinates of spots in image that you want to transform ```yxz_base``` to.
        start_transform: initial transform
        dist_thresh_yx: If neighbours closer than this, they are used to compute the new transform.
            Typical: ```8```.
        dist_thresh_z: If neighbours closer than this, they are used to compute the new transform.
            Typical: ```2```.
        n_iters: max number of times to compute regression
        robust: whether to compute robust icp
    Returns:
        - ```transform``` - ```float [4 x 3]```.
            Updated affine transform.
        - ```n_matches``` - ```int```.
            Number of neighbours which have distance less than ```dist_thresh```.
        - ```error``` - ```float```.
            Average distance between ```neighbours``` below ```dist_thresh```.
        - converged - bool
            True if completed in less than n_iters and false o/w
    """
    # initialise transform
    transform = start_transform
    n_matches = np.zeros(n_iters)
    error = np.zeros(n_iters)
    prev_neighbour = np.ones(yxz_target.shape[0]) * yxz_base.shape[0]

    # Update transform. We want this to have max n_iters iterations. We will end sooner if all neighbours do not change
    # in 2 successive iterations. Define the variables for iteration 0 before we start the loop
    transform, neighbour, n_matches[0], error[0] = get_transform(
        yxz_base, yxz_target, transform, dist_thresh_yx, dist_thresh_z, robust
    )
    i = 0
    while i + 1 < n_iters and not all(prev_neighbour == neighbour):
        # update i and prev_neighbour
        prev_neighbour, i = neighbour, i + 1
        transform, neighbour, n_matches[i], error[i] = get_transform(
            yxz_base, yxz_target, transform, dist_thresh_yx, dist_thresh_z, robust
        )
    # now fill in any variables that were not completed due to early exit
    n_matches[i:] = n_matches[i] * np.ones(n_iters - i)
    error[i:] = error[i] * np.ones(n_iters - i)
    converged = i < n_iters

    return transform, n_matches, error, converged
