import os
import time
from typing import Optional, Tuple

import joblib
import nd2
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
import skimage
from sklearn.linear_model import HuberRegressor
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
    window_radius: int = 5,
    smooth_sigma: list = [20, 20, 2],
    clip_val: np.ndarray = np.array([40, 40, 15]),
    n_cores: Optional[int] = None,
):
    """
    Function to carry out optical flow registration on a single tile and round.

    Optical flow is computed using an iterative Lucas-Kanade method with 10 iterations and a window radius specified
    by the user.

    Bad regions are removed by thresholding the correlation coefficient of the optical flow and interpolating the flow
    in these regions. The interpolation is done by smoothing the optical flow and dividing by a smoothed indicator of
    the flow.

    Args:
        target: np.ndarray size [n_y, n_x, n_z] of the target image (this will be the round image)
        base: np.ndarray size [n_y, n_x, n_z] of the base image (this will be the anchor image)
        sample_factor_yx: int specifying how much to downsample the images in y and x before computing the optical flow
        window_radius: int specifying the window radius for the optical flow algorithm and correlation calculation
        (Note that this is the radius on the downsampled image, so a radius of 5 with a downsample factor of 4 will
        correspond to a radius of 20 on the original image)
        smooth_threshold: float specifying the threshold for the correlation coefficient to be considered for smoothing
        smooth_sigma: float specifying the standard deviation of the Gaussian filter to be used for smoothing the flow
        clip_val: np.ndarray size [3] of the clip value for the optical flow in y, x and z
        output_dir: str specifying the directory to save the optical flow information (flow, corr and smooth)
        file_name: str specifying the file name to save the optical flow information (flow, corr and smooth)
        n_cores (int, optional): maximum cpu cores to use in parallel when computing optical flow. Default: all found
            cpu cores.

    Returns:
        Tuple containing path to optical flow, correlation, and smoothed flow zarr arrays respectively.
    """
    # convert the images to float32
    target = target.astype(np.float32)
    base = base.astype(np.float32)
    # down-sample the images
    target = target[::sample_factor_yx, ::sample_factor_yx]
    base = base[::sample_factor_yx, ::sample_factor_yx]
    # match historams of the images
    # target = skimage.exposure.match_histograms(target, base)
    target = target / np.median(target)
    base = base / np.median(base)
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
        chunks_yx=4,
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
        win_size=np.array([6, 6, 2]),
        loc=corr_loc,
    )
    # smooth the flow
    interpolate_flow(
        flow,
        correlation,
        tile=tile,
        round=round,
        sigma=smooth_sigma,
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
    n_cores: Optional[int] = None,
    loc: str = "",
) -> np.ndarray:
    """
    Function to carry out optical flow registration on 2 3D images.
    Args:
        base: np.ndarray size [n_y, n_x, n_z] of the base image
        target: np.ndarray size [n_y, n_x, n_z] of the target image
        window_radius: int specifying the window radius for the optical flow algorithm
        clip_val: np.ndarray size [3] of the clip value for the optical flow in y, x and z
        upsample_factor_yx: int specifying how much to upsample the optical flow in y and x
        chunks_yx: int specifying the number of subvolumes to split the images into in y and x
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
    # First, correct for a yx shift in the images
    # mid_z = int(target.shape[2] / 2)
    window_yx = skimage.filters.window("hann", target.shape)
    shift = skimage.registration.phase_cross_correlation(
        reference_image=target * window_yx, moving_image=base * window_yx, upsample_factor=10
    )[0]
    base = preprocessing.custom_shift(base, shift.astype(int))
    ny, _, nz = target.shape
    yx_sub = int((ny / chunks_yx) * 1.25)
    while (ny - yx_sub) % (chunks_yx - 1) != 0 or yx_sub % 2 != 0:
        yx_sub += 1
    target_sub, pos = preprocessing.split_3d_image(
        image=target,
        z_subvolumes=1,
        y_subvolumes=chunks_yx,
        x_subvolumes=chunks_yx,
        z_box=nz,
        y_box=yx_sub,
        x_box=yx_sub,
    )
    base_sub, _ = preprocessing.split_3d_image(
        image=base, z_subvolumes=1, y_subvolumes=chunks_yx, x_subvolumes=chunks_yx, z_box=nz, y_box=yx_sub, x_box=yx_sub
    )
    # this next line assumes that the images are split into 1 subvolume in z (which we always do)
    target_sub, base_sub = target_sub[0], base_sub[0]
    # flatten dims 0 and 1
    target_sub = target_sub.reshape((chunks_yx**2, nz, yx_sub, yx_sub))
    base_sub = base_sub.reshape((chunks_yx**2, nz, yx_sub, yx_sub))
    # divide each subvolume by its mean
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
    flow_sub = np.array(flow_sub)
    # swap so axis 0 is z,y,x and axis 1 is the subvolume
    flow_sub = flow_sub.swapaxes(0, 1)
    # merge the subvolumes
    flow = np.array([preprocessing.merge_subvols(pos, flow_sub[i]) for i in range(3)])
    # now convert flow back to yxz
    flow = preprocessing.flow_zyx_to_yxz(flow)
    # clip the flow
    flow = np.array([np.clip(flow[i], -clip_val[i], clip_val[i]) for i in range(3)])
    # add back the shift
    flow = np.array([flow[i] - shift[i] for i in range(3)])
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
    win_size: np.ndarray,
    upsample_factor_yx: int = 4,
    loc: str = "",
) -> np.ndarray:
    """
    Compute the correlation between the base and target images within a small window of each pixel.
    This is done in a vectorized manner by reshaping the images into windows and then computing the correlation
    coefficient between the base and target images within each window. For this reason the window size must be a factor
    of the image size in each dimension.

    Args:
        base: n_y x n_x x n_z array of the base image
        target: n_y x n_x x n_z array of the target image
        flow: 3 x n_y x n_x x n_z array of flow in y, x and z
        win_size: 3 element array of the window size in y, x and z
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
    base_warped = skimage.transform.warp(base, coords + flow, order=0, mode="constant", cval=0)
    del coords, base, flow
    # divide base_warped and target by their max.
    base_warped = base_warped / np.max(base_warped)
    target = target / np.max(target)
    for i in range(3):
        while target.shape[i] % win_size[i] != 0:
            win_size[i] += 1
    # compute number of windows in each dimension
    n_win = np.array(target.shape) // win_size
    # reshape the images into windows
    base_warped = base_warped.reshape(n_win[0], win_size[0], n_win[1], win_size[1], n_win[2], win_size[2])
    target = target.reshape(n_win[0], win_size[0], n_win[1], win_size[1], n_win[2], win_size[2])
    # move the window dimensions to the front
    base_warped = np.moveaxis(base_warped, [0, 2, 4], [0, 1, 2])
    target = np.moveaxis(target, [0, 2, 4], [0, 1, 2])
    # Now reshape so the window dimensions and pixel dimensions are flattened
    base_warped = base_warped.reshape(np.prod(n_win), np.prod(win_size))
    target = target.reshape(np.prod(n_win), np.prod(win_size))
    # compute the correlation
    correlation = np.sum(base_warped * target, axis=1)
    # reshape the correlation back to the window dimensions
    correlation = correlation.reshape(n_win)
    # upsample the correlation to the original image size just by repeating the correlation values
    correlation = np.repeat(
        np.repeat(np.repeat(correlation, win_size[2], axis=2), win_size[1], axis=1), win_size[0], axis=0
    )
    # upsample
    correlation_up = upsample_yx(correlation, upsample_factor_yx, order=0).astype(np.float16)
    # save the correlation
    if loc:
        # save in yxz format
        zarray = zarr.open_array(loc, mode="r+")
        zarray[tile, round] = correlation_up
    t_end = time.time()
    log.info("Computing correlation took " + str(t_end - t_start) + " seconds")
    return correlation, correlation_up


def interpolate_flow(
    flow: np.ndarray,
    correlation: np.ndarray,
    tile: int,
    round: int,
    sigma: list = [20, 20, 2],
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
        upsample_factor_yx: int specifying the upsample factor in y and x
        loc: str specifying the location to save/ load the interpolated flow
    """
    time_start = time.time()
    # normalise the correlation so that the mean of each z-plane is 1. This is done to use the best shifts on each z
    # plane, otherwise the middle z-planes dominate the smoothed flow
    correlation = correlation / np.mean(correlation, axis=(0, 1))[None, None, :]
    correlation_smooth = gaussian_filter(correlation, sigma, truncate=6)
    for i in range(3):
        flow[i] = gaussian_filter(flow[i] * correlation, sigma, truncate=6)
    # divide the flow by the smoothed indicator
    flow = np.array([flow[i] / correlation_smooth for i in range(3)])
    # remove nan values
    flow = np.nan_to_num(flow)
    upsample_factor = (upsample_factor_yx, upsample_factor_yx, 1)
    # upsample the flow before saving
    flow = np.array(
        [
            upsample_yx(flow[i], upsample_factor_yx, order=1) * upsample_factor[i]
            for i in tqdm(range(3), desc="Upsampling smooth flow")
        ],
        dtype=np.float16,
    )
    # save the flow
    if loc:
        # save in yxz format
        zarray = zarr.open_array(loc, mode="r+")
        zarray[tile, round] = flow
    time_end = time.time()
    log.info("Interpolating flow took " + str(time_end - time_start) + " seconds")
    return flow


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

    Fluorescent beads are assumed to be circles and are detected using the Hough transform. The centre of the detected
    circles are then used to compute the affine transform between each camera and the round_reg_channel_cam via ICP.

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
    transform = np.zeros((n_cams, 3, 4))
    # First check if the fluorescent beads path exists. If not, we assume that the channels are registered to each
    # other and just set channel_transforms to identity matrices
    if fluorescent_bead_path is None:
        # Set registration_data['channel_registration']['channel_transform'][c] = np.eye(3) for all channels c
        for c in range(n_cams):
            transform[c] = np.eye(3, 4)
        log.warn("Fluorescent beads directory does not exist. Assuming that all channels are registered to each other.")
        return transform

    # open the fluorescent bead images as nd2 files
    with nd2.ND2File(fluorescent_bead_path) as fbim:
        fluorescent_beads = fbim.asarray()
    # if fluorescent bead images are for all channels, just take one from each camera

    # TODO better deal with 3D
    if fluorescent_beads.ndim == 4:
        nz = fluorescent_beads.shape[0]
        fluorescent_beads = fluorescent_beads[int(nz / 2)]

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
        shift_yx = skimage.registration.phase_cross_correlation(reference_image=fluorescent_beads[i],
                                                                moving_image=fluorescent_beads[anchor_cam_idx])[0]
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
        values = fluorescent_beads[i][cy, cx]
        cy_rand, cx_rand = (
            np.random.randint(0, fluorescent_beads[i].shape[0] - 1, 100),
            np.random.randint(0, fluorescent_beads[i].shape[1] - 1, 100),
        )
        noise = np.mean(fluorescent_beads[i][cy_rand, cx_rand])
        keep = values > noise
        cy, cx = cy[keep], cx[keep]
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

    # Convert transforms from yxz to zyx
    transform_zyx = np.zeros((4, 3, 4))
    for i in range(n_cams):
        transform_zyx[i] = preprocessing.yxz_to_zyx_affine(transform[i])

    return transform_zyx


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

