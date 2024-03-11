import nd2
import scipy
import numpy as np
import skimage
import pickle
from tqdm import tqdm
from sklearn.linear_model import HuberRegressor
from typing import Optional, Tuple

from .. import logging
from . import preprocessing


def find_shift_array(subvol_base, subvol_target, position, r_threshold):
    """
    This function takes in 2 3d images which have already been split up into 3d subvolumes. We then find the shift from
    each base subvolume to its corresponding target subvolume.
    NOTE: This function does allow matching to non-corresponding subvolumes in z.
    NOTE: This function performs flattening of the output arrays, and this is done according to np.reshape.
    Args:
        subvol_base: Base subvolume array (n_z_subvolumes, n_y_subvolumes, n_x_subvolumes, z_box, y_box, x_box)
        subvol_target: Target subvolume array (n_z_subvolumes, n_y_subvolumes, n_x_subvolumes, z_box, y_box, x_box)
        position: Position of centre of subvolumes in base array (n_z_subvolumes, n_y_subvolumes, n_x_subvolumes, 3)
        r_threshold: measure of shift quality. If the correlation between corrected base subvol and fixed target
        subvol is beneath this, we store shift as [nan, nan, nan]
    Returns:
        shift: 2D array, with first dimension referring to subvolume index and final dim referring to shift
        (n_z_subvolumes * n_y_subvolumes * n_x_subvolumes, 3)
        shift_corr: 2D array, with first dimension referring to subvolume index and final dim referring to
        shift_corr coef (n_z_subvolumes * n_y_subvolumes * n_x_subvolumes, 1)
    """
    if subvol_base.shape != subvol_target.shape:
        logging.error(ValueError("Subvolume arrays have different shapes"))
    z_subvolumes, y_subvolumes, x_subvolumes = subvol_base.shape[0], subvol_base.shape[1], subvol_base.shape[2]
    shift = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, 3))
    shift_corr = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes))
    position = np.reshape(position, (z_subvolumes, y_subvolumes, x_subvolumes, 3))

    for y, x in tqdm(np.ndindex(y_subvolumes, x_subvolumes), desc="Computing subvolume shifts",
                     total=y_subvolumes * x_subvolumes):
            shift[:, y, x], shift_corr[:, y, x] = find_z_tower_shifts(
                subvol_base=subvol_base[:, y, x],
                subvol_target=subvol_target[:, y, x],
                position=position[:, y, x].copy(),
                pearson_r_threshold=r_threshold,
            )

    return np.reshape(shift, (shift.shape[0] * shift.shape[1] * shift.shape[2], 3)), np.reshape(
        shift_corr, shift.shape[0] * shift.shape[1] * shift.shape[2]
    )


def find_z_tower_shifts(subvol_base, subvol_target, position, pearson_r_threshold, z_neighbours=1):
    """
    This function takes in 2 split up 3d images along one tower of subvolumes in z and computes shifts for each of these
    to its nearest neighbour subvol
    Args:
        subvol_base: (n_z_subvols, z_box, y_box, x_box) Base subvolume array (this is a z-tower of n_z_subvols base
        subvolumes)
        subvol_target: (n_z_subvols, z_box, y_box, x_box) Target subvolume array (this is a z-tower of n_z_subvols
        target subvolumes)
        position: (n_z_subvols, 3) Position of centre of subvolumes (z, y, x)
        pearson_r_threshold: (float) threshold of correlation used in degenerate cases
        z_neighbours: (int) number of neighbouring sub-volumes to merge with the current sub-volume to compute the shift
    Returns:
        shift: (n_z_subvols, 3) shift of each subvolume in z-tower
        shift_corr: (n_z_subvols, 1) correlation coefficient of each subvolume in z-tower
    """
    position = position.astype(int)
    # for the purposes of this section, we'll take position to be the bottom left corner of the subvolume
    position = position - np.array([subvol_base.shape[1], subvol_base.shape[2], subvol_base.shape[3]]) // 2
    z_subvolumes = subvol_base.shape[0]
    z_box = subvol_base.shape[1]
    shift = np.zeros((z_subvolumes, 3))
    shift_corr = np.zeros(z_subvolumes)
    for z in range(z_subvolumes):
        z_start, z_end = int(max(0, z - z_neighbours)), int(min(z_subvolumes, z + z_neighbours + 1))
        merged_subvol_target = preprocessing.merge_subvols(
            position=np.copy(position[z_start:z_end]), subvol=subvol_target[z_start:z_end]
        )
        merged_subvol_base = np.zeros_like(merged_subvol_target)
        merged_subvol_base_windowed = np.zeros_like(merged_subvol_target)
        merged_subvol_min_z = position[z_start, 0]
        current_box_min_z = position[z, 0]
        merged_subvol_start_z = current_box_min_z - merged_subvol_min_z
        merged_subvol_base[merged_subvol_start_z : merged_subvol_start_z + z_box] = subvol_base[z]
        # window the images
        merged_subvol_target_windowed = preprocessing.window_image(merged_subvol_target)
        merged_subvol_base_windowed[merged_subvol_start_z : merged_subvol_start_z + z_box] = preprocessing.window_image(
            subvol_base[z]
        )
        # Now we have the merged subvolumes, we can compute the shift
        shift[z], _, _ = skimage.registration.phase_cross_correlation(
            reference_image=merged_subvol_target_windowed,
            moving_image=merged_subvol_base_windowed,
            upsample_factor=10,
            disambiguate=True,
            overlap_ratio=0.5,
        )
        # compute pearson correlation coefficient
        shift_base = preprocessing.custom_shift(merged_subvol_base, shift[z].astype(int))
        mask = (shift_base != 0) * (merged_subvol_target != 0)
        if np.sum(mask) == 0:
            shift_corr[z] = 0
        else:
            shift_corr[z] = np.corrcoef(shift_base[mask], merged_subvol_target[mask])[0, 1]
        if shift_corr[z] < pearson_r_threshold:
            shift[z] = np.array([np.nan, np.nan, np.nan])

    return shift, shift_corr


def find_zyx_shift(subvol_base, subvol_target, pearson_r_threshold=0.9):
    """
    This function takes in 2 3d images and finds the optimal shift from one to the other. We use a phase cross
    correlation method to find the shift.
    Args:
        subvol_base: Base subvolume array (this will contain a lot of zeroes) (n_z_pixels, n_y_pixels, n_x_pixels)
        subvol_target: Target subvolume array (this will be a merging of subvolumes with neighbouring subvolumes)
        (nz_pixels2, n_y_pixels2, n_x_pixels2) size 2 >= size 1
        pearson_r_threshold: Threshold used to accept a shift as valid (float)

    Returns:
        shift: zyx shift (3,)
        shift_corr: correlation coefficient of shift (float)
    """
    if subvol_base.shape != subvol_target.shape:
        logging.error(ValueError("Subvolume arrays have different shapes"))
    shift, _, _ = skimage.registration.phase_cross_correlation(
        reference_image=subvol_target, moving_image=subvol_base, upsample_factor=10
    )
    alt_shift = np.copy(shift)
    # now anti alias the shift in z. To do this, consider that the other possible aliased z shift is the either one
    # subvolume above or below the current shift. (In theory, we could also consider the subvolume 2 above or below,
    # but this is unlikely to be the case in practice as we are already merging subvolumes)
    if shift[0] > 0:
        alt_shift[0] = shift[0] - subvol_base.shape[0]
    else:
        alt_shift[0] = shift[0] + subvol_base.shape[0]

    # Now we need to compute the correlation coefficient of the shift and the anti aliased shift
    shift_base = preprocessing.custom_shift(subvol_base, shift.astype(int))
    alt_shift_base = preprocessing.custom_shift(subvol_base, alt_shift.astype(int))
    # Now compute the correlation coefficients. First create a mask of the nonzero values
    mask = shift_base != 0
    shift_corr = np.corrcoef(shift_base[mask], subvol_target[mask])[0, 1]
    if np.isnan(shift_corr):
        shift_corr = 0.0
    mask = alt_shift_base != 0
    alt_shift_corr = np.corrcoef(alt_shift_base[mask], subvol_target[mask])[0, 1]
    if np.isnan(alt_shift_corr):
        alt_shift_corr = 0.0
    mask = subvol_base != 0
    base_corr = np.corrcoef(subvol_base[mask], subvol_target[mask])[0, 1]
    if np.isnan(base_corr):
        base_corr = 0.0

    # Now return the shift with the highest correlation coefficient
    if alt_shift_corr > shift_corr:
        shift = alt_shift
        shift_corr = alt_shift_corr
    if base_corr > shift_corr:
        shift = np.array([0, 0, 0])
        shift_corr = base_corr
    # Now check if the correlation coefficient is above the threshold. If not, set the shift to nan
    if shift_corr < pearson_r_threshold:
        shift = np.array([np.nan, np.nan, np.nan])
        shift_corr = np.nanmax([shift_corr, alt_shift_corr, base_corr])

    return shift, shift_corr


# ols regression to find transform from shifts
def ols_regression(shift, position):
    """
    Args:
        shift: (z_sv x y_sv x x_sv) x 3 array which of shifts in zyx format
        position: (z_sv x y_sv x x_sv) x 3 array which of positions in zyx format
    Returns:
        transform: 3 x 4 affine transform in yxz format with final col being shift
    """

    # We are going to get rid of the shifts where any of the values are nan for regression
    position = position[~np.isnan(shift[:, 0])]
    shift = shift[~np.isnan(shift[:, 0])]

    new_position = position + shift
    position = np.vstack((position.T, np.ones(shift.shape[0]))).T

    # Now compute the regression
    transform, _, _, _ = np.linalg.lstsq(position, new_position, rcond=None)

    # Unsure what taking transpose means for off diagonals here
    return transform.T


def huber_regression(shift, position, predict_shift=True):
    """
    Function to predict shift as a function of position using robust huber regressor. If we do not have >= 3 z-coords
    in position, the z-coords of the affine transform will be estimated as no scaling, and a shift of mean(shift).
    Args:
        shift: n_tiles x 3 ndarray of zyx shifts
        position: n_tiles x 2 ndarray of yx tile coords or n_tiles x 3 ndarray of zyx tile coords
        predict_shift: If True, predict shift as a function of position. If False, predict position as a function of
        position. Default is True. Difference is that if false, we add 1 to each diagonal of the transform matrix.
    Returns:
        transform: 3 x 3 matrix where each row predicts shift of z y z as a function of y index, x index and the final
        row is the offset at 0,0
        or 3
    """
    # We are going to get rid of the shifts where any of the values are nan for regression
    position = position[~np.isnan(shift[:, 0])]
    shift = shift[~np.isnan(shift[:, 0])]
    # Check if we have any shifts to predict
    if len(shift) == 0 and predict_shift:
        transform = np.zeros((3, 4))
        return transform
    elif len(shift) == 0 and not predict_shift:
        transform = np.eye(3, 4)
        return transform
    # Do robust regression
    # Check we have at least 3 z-coords in position
    if len(set(position[:, 0])) <= 2:
        z_coef = np.array([0, 0, 0])
        z_shift = np.mean(shift[:, 0])
        logging.warn(
            "Fewer than 3 z-coords in position. Setting z-coords of transform to no scaling and shift of mean(shift)"
        )
    else:
        huber_z = HuberRegressor(epsilon=2, max_iter=400, tol=1e-4).fit(X=position, y=shift[:, 0])
        z_coef = huber_z.coef_
        z_shift = huber_z.intercept_
    huber_y = HuberRegressor(epsilon=2, max_iter=400, tol=1e-4).fit(X=position, y=shift[:, 1])
    huber_x = HuberRegressor(epsilon=2, max_iter=400, tol=1e-4).fit(X=position, y=shift[:, 2])
    transform = np.vstack(
        (
            np.append(z_coef, z_shift),
            np.append(huber_y.coef_, huber_y.intercept_),
            np.append(huber_x.coef_, huber_x.intercept_),
        )
    )
    if not predict_shift:
        transform += np.eye(3, 4)

    return transform


# Bridge function for all functions in round registration
def round_registration(anchor_image: np.ndarray, round_image: list, t: int, round_ind: list,
                       config: dict, registration_data: dict, save_path: str):
    """
    Function to carry out sub-volume round registration on a single tile.

    Args:
        anchor_image: np.ndarray size [n_z, n_y, n_x] of the anchor image
        round_image: list of length n_rounds, where round_image[r] is  np.ndarray size [n_z, n_y, n_x] of round r
        t: int index of the tile
        round_ind: list (length n_rounds) of the round indices
        config: dict of configuration parameters for registration
        registration_data: dict of registration data (to be updated)
        save_path: str path to save the updated registration data
    """
    # Initialize variables from config
    z_subvols, y_subvols, x_subvols = config["subvols"]
    z_box, y_box, x_box = config["box_size"]
    r_thresh = config["pearson_r_thresh"]

    if config["sobel"]:
        anchor_image = skimage.filters.sobel(anchor_image)
        round_image = [skimage.filters.sobel(r) for r in round_image]

    # Now compute round shifts for this tile and the affine transform for each round
    for r in tqdm(range(len(round_image)), desc="Computing round shifts", total=len(round_image)):
        # Set progress bar title
        # next we split image into overlapping cuboids
        subvol_base, position = preprocessing.split_3d_image(
            image=anchor_image,
            z_subvolumes=z_subvols,
            y_subvolumes=y_subvols,
            x_subvolumes=x_subvols,
            z_box=z_box,
            y_box=y_box,
            x_box=x_box,
        )
        subvol_target, _ = preprocessing.split_3d_image(
            image=round_image[r],
            z_subvolumes=z_subvols,
            y_subvolumes=y_subvols,
            x_subvolumes=x_subvols,
            z_box=z_box,
            y_box=y_box,
            x_box=x_box,
        )
        # remove the round_image from memory
        round_image[r] = None

        # Find the subvolume shifts
        shift, corr = find_shift_array(subvol_base, subvol_target, position=position.copy(), r_threshold=r_thresh)
        transform = huber_regression(shift, position, predict_shift=False)
        # Append these arrays to the round_shift, round_shift_corr, round_transform and position storage
        # note: r is not the round index, but the index of the round in the round_image list so need to use round_ind
        registration_data["round_registration"]["shift"][t, round_ind[r]] = shift
        registration_data["round_registration"]["shift_corr"][t, round_ind[r]] = corr
        if registration_data["round_registration"]["position"].max() == 0:
            registration_data["round_registration"]["position"] = position
        registration_data["round_registration"]["transform_raw"][t, round_ind[r]] = transform

        # Save the registration data
        with open(save_path, "wb") as f:
            pickle.dump(registration_data, f)


def channel_registration(
    fluorescent_bead_path: str = None, anchor_cam_idx: int = 2, n_cams: int = 4, bead_radii: list = [10, 11, 12]
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
        logging.warn(
            "Fluorescent beads directory does not exist. Assuming that all channels are registered to each other."
        )
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

    # Now we'll turn each image into a point cloud
    bead_point_clouds = []
    for i in range(n_cams):
        edges = skimage.feature.canny(fluorescent_beads[i], sigma=3, low_threshold=10, high_threshold=50)
        hough_res = skimage.transform.hough_circle(edges, bead_radii)
        accums, cx, cy, radii = skimage.transform.hough_circle_peaks(
            hough_res, bead_radii, min_xdistance=10, min_ydistance=10
        )
        bead_point_clouds.append(np.vstack((cy, cx)).T)

    # Now convert the point clouds from yx to yxz. This is because our ICP algorithm assumes that the point clouds
    # are in 3D space
    bead_point_clouds = [
        np.hstack((bead_point_clouds[i], np.zeros((bead_point_clouds[i].shape[0], 1)))) for i in range(n_cams)
    ]

    # Set up ICP
    initial_transform = np.zeros((n_cams, 4, 3))
    transform = np.zeros((n_cams, 4, 3))
    mse = np.zeros((n_cams, 50))
    target_cams = [i for i in range(n_cams) if i != anchor_cam_idx]
    with tqdm(total=len(target_cams)) as pbar:
        for i in target_cams:
            pbar.set_description("Running ICP for camera " + str(i))
            # Set initial transform to identity (shift already accounted for)
            initial_transform[i, :3, :3] = np.eye(3)
            # Run ICP
            transform[i], _, mse[i], converged = icp(
                yxz_base=bead_point_clouds[anchor_cam_idx],
                yxz_target=bead_point_clouds[i],
                start_transform=initial_transform[i],
                n_iters=50,
                dist_thresh=5,
            )
            if not converged:
                transform[i] = np.eye(4, 3)
                logging.error(Warning("ICP did not converge for camera " + str(i) + ". Replacing with identity."))
            pbar.update(1)

    # Need to add in z coord info as not accounted for by registration due to all coords being equal
    transform[:, 2, 2] = 1
    transform[anchor_cam_idx] = np.eye(4, 3)

    # Convert transforms from yxz to zyx
    transform_zyx = np.zeros((4, 3, 4))
    for i in range(4):
        transform_zyx[i] = preprocessing.yxz_to_zyx_affine(transform[i])

    return transform_zyx


# Function to remove outlier shifts
def regularise_shift(shift, tile_origin, residual_threshold) -> np.ndarray:
    """
    Function to remove outlier shifts from a collection of shifts whose start location can be determined by position.
    Uses robust linear regression to compute a prediction of what the shifts should look like and if any shift differs
    from this by more than some threshold it is declared to be an outlier.
    Args:
        shift: [n_tiles_use x n_rounds_use x 3]
        tile_origin: yxz positions of the tiles [n_tiles_use x 3]
        residual_threshold: This is a threshold above which we will consider a point to be an outlier

    Returns:
        shift_regularised: [n_tiles x n_rounds x 4 x 3]
    """
    # rearrange columns so that tile origins are in zyx like the shifts are, then initialise commonly used variables
    tile_origin = np.roll(tile_origin, 1, axis=1)
    tile_origin_padded = np.pad(tile_origin, ((0, 0), (0, 1)), mode="constant", constant_values=1)
    n_tiles_use, n_rounds_use = shift.shape[0], shift.shape[1]
    shift_regularised = np.zeros((n_tiles_use, n_rounds_use, 3))
    shift_norm = np.linalg.norm(shift, axis=2)

    # Loop through rounds and perform a linear regression on the shifts for each round
    for r in range(n_rounds_use):
        lb, ub = np.percentile(shift_norm[:, r], [10, 90])
        valid = (shift_norm[:, r] > lb) * (shift_norm[:, r] < ub)
        if np.sum(valid) < 3:
            continue
        # Carry out regression
        big_transform = ols_regression(shift[valid, r], tile_origin[valid])
        shift_regularised[:, r] = tile_origin_padded @ big_transform.T - tile_origin
        # Compute residuals
        residuals = np.linalg.norm(shift[:, r] - shift_regularised[:, r], axis=1)
        # Identify outliers
        outliers = residuals > residual_threshold
        # Replace outliers with predicted shifts
        shift[outliers, r] = shift_regularised[outliers, r]

    return shift


# Function to remove outlier round scales
def regularise_round_scaling(scale: np.ndarray):
    """
    Function to remove outliers in the scale parameters for round transforms. Experience shows these should be close
    to 1 for x and y regardless of round and should be increasing or decreasing for z dependent on whether anchor
    round came before or after.

    Args:
        scale: 3 x n_tiles_use x n_rounds_use ndarray of z y x scales for each tile and round

    Returns:
        scale_regularised:  n_tiles_use x n_rounds_use x 3 ndarray of z y x regularised scales for each tile and round
    """
    # Define num tiles and separate the z scale and yx scales for different analysis
    n_tiles = scale.shape[1]
    z_scale = scale[0]
    yx_scale = scale[1:]

    # Regularise z_scale. Expect z_scale to vary by round but not by tile.
    # We classify outliers in z as:
    # a.) those that lie outside 1 iqr of the median for z_scales of that round
    # b.) those that increase when the majority decrease or those that decrease when majority increase
    # First carry out removal of outliers of type (a)
    median_z_scale = np.repeat(np.median(z_scale, axis=0)[np.newaxis, :], n_tiles, axis=0)
    iqr_z_scale = np.repeat(scipy.stats.iqr(z_scale, axis=0)[np.newaxis, :], n_tiles, axis=0)
    outlier_z = np.abs(z_scale - median_z_scale) > iqr_z_scale
    z_scale[outlier_z] = median_z_scale[outlier_z]

    # Now carry out removal of outliers of type (b)
    delta_z_scale = np.diff(z_scale, axis=1)
    dominant_sign = np.sign(np.median(delta_z_scale))
    outlier_z = np.hstack((np.sign(delta_z_scale) != dominant_sign, np.zeros((n_tiles, 1), dtype=bool)))
    z_scale[outlier_z] = median_z_scale[outlier_z]

    # Regularise yx_scale: No need to account for variation in tile or round
    outlier_yx = np.abs(yx_scale - 1) > 0.01
    yx_scale[outlier_yx] = 1

    return scale


# Bridge function for all regularisation
def regularise_transforms(
    registration_data: dict, tile_origin: np.ndarray, residual_threshold: float, use_tiles: list, use_rounds: list
):
    """
    Function to regularise affine transforms by comparing them to affine transforms from other tiles.
    As the channel transforms do not depend on tile, they do not need to be regularised.
    Args:
        registration_data: dictionary of registration data
        tile_origin: n_tiles x 3 tile origin in zyx
        residual_threshold: threshold for classifying outlier shifts
        use_tiles: list of tiles in use
        use_rounds: list of rounds in use

    Returns:
        registration_data: dictionary of registration data with regularised transforms
    """
    # Extract transforms
    round_transform = np.copy(registration_data["round_registration"]["transform_raw"])

    # Code becomes easier when we disregard tiles, rounds, channels not in use
    n_tiles, n_rounds = round_transform.shape[0], round_transform.shape[1]
    tile_origin = tile_origin[use_tiles]
    round_transform = round_transform[use_tiles][:, use_rounds]

    # Regularise round transforms
    round_transform[:, :, :, 3] = regularise_shift(
        shift=round_transform[:, :, :, 3], tile_origin=tile_origin, residual_threshold=residual_threshold
    )
    round_scale = np.array([round_transform[:, :, 0, 0], round_transform[:, :, 1, 1], round_transform[:, :, 2, 2]])
    round_scale_regularised = regularise_round_scaling(round_scale)
    round_transform = preprocessing.replace_scale(transform=round_transform, scale=round_scale_regularised)
    round_transform = preprocessing.populate_full(
        sublist_1=use_tiles,
        list_1=np.arange(n_tiles),
        sublist_2=use_rounds,
        list_2=np.arange(n_rounds),
        array=round_transform,
    )

    registration_data["round_registration"]["transform"] = round_transform

    return registration_data


# Function which runs a single iteration of the icp algorithm
def get_transform(
    yxz_base: np.ndarray, yxz_target: np.ndarray, transform_old: np.ndarray, dist_thresh: float, robust=False
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
        dist_thresh: If neighbours closer than this, they are used to compute the new transform.
            Typical: ```5```.
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
    # Step 1 computes matching, since yxz_target is a subset of yxz_base, we will loop through yxz_target and find
    # their nearest neighbours within yxz_transform, which is the initial transform applied to yxz_base
    yxz_base_pad = np.pad(yxz_base, [(0, 0), (0, 1)], constant_values=1)
    yxz_transform = yxz_base_pad @ transform_old
    yxz_transform_tree = scipy.spatial.KDTree(yxz_transform)
    # the next query works the following way. For each point in yxz_target, we look for the closest neighbour in the
    # anchor, which we have now applied the initial transform to. If this is below dist_thresh, we append its distance
    # to distances and append the index of this neighbour to neighbour
    distances, neighbour = yxz_transform_tree.query(yxz_target, distance_upper_bound=dist_thresh)
    neighbour = neighbour.flatten()
    distances = distances.flatten()
    use = distances < dist_thresh
    n_matches = np.sum(use)
    error = np.sqrt(np.mean(distances[use] ** 2))
    base_pad_use = yxz_base_pad[neighbour[use], :]
    target_use = yxz_target[use, :]

    if not robust:
        transform = np.linalg.lstsq(base_pad_use, target_use, rcond=None)[0]
    else:
        sigma = dist_thresh / 2
        target_pad_use = np.pad(target_use, [(0, 0), (0, 1)], constant_values=1)
        D = np.diag(np.exp(-0.5 * (np.linalg.norm(base_pad_use @ transform_old - target_use, axis=1) / sigma) ** 2))
        transform = (target_pad_use.T @ D @ base_pad_use @ np.linalg.inv(base_pad_use.T @ D @ base_pad_use))[:3, :4].T

    return transform, neighbour, n_matches, error


# Simple ICP implementation, calls above until no change
def icp(yxz_base, yxz_target, dist_thresh, start_transform, n_iters, robust=False):
    """
    Applies n_iters rounds of the above least squares regression
    Args:
        yxz_base: ```float [n_base_spots x 3]```.
            Coordinates of spots you want to transform.
        yxz_target: ```float [n_target_spots x 3]```.
            Coordinates of spots in image that you want to transform ```yxz_base``` to.
        start_transform: initial transform
        dist_thresh: If neighbours closer than this, they are used to compute the new transform.
            Typical: ```3```.
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
    transform, neighbour, n_matches[0], error[0] = get_transform(yxz_base, yxz_target, transform, dist_thresh, robust)
    i = 0
    while i + 1 < n_iters and not all(prev_neighbour == neighbour):
        # update i and prev_neighbour
        prev_neighbour, i = neighbour, i + 1
        transform, neighbour, n_matches[i], error[i] = get_transform(
            yxz_base, yxz_target, transform, dist_thresh, robust
        )
    # now fill in any variables that were not completed due to early exit
    n_matches[i:] = n_matches[i] * np.ones(n_iters - i)
    error[i:] = error[i] * np.ones(n_iters - i)
    converged = i < n_iters

    return transform, n_matches, error, converged


def brightness_scale(
    preseq: np.ndarray,
    seq: np.ndarray,
    intensity_percentile: float,
    sub_image_size: Optional[int] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Function to find scale factor m and such that m * preseq ~ seq. First, the preseq and seq images are
    aligned using phase cross correlation. Then apply regression on the pixels of `preseq' which are above the
    `intensity_percentile` percentile. This is because we want to use the brightest pixels for regression as these
    are the pixels which are likely to be background in the preseq image and foreground in the seq image.

    Args:
        preseq: (n_y x n_x) ndarray of presequence image.
        seq: (n_y x n_x) ndarray of sequence image.
        intensity_percentile (float): brightness percentile such that all pixels with brightness > this percentile (in
            the preseq im only) are used for regression.
        sub_image_size (int, optional): size of sub-images in x and y to use for regression. This is because the images
            are not perfectly registered and so we need to find the best registered sub-image to use for regression.
            Default: `floor(n_x / 4.095)`.

    Returns:
        - scale: float scale factor `m`.
        - sub_image_seq: (sub_image_size, sub_image_size) ndarray of aligned sequence image used for regression.
        - sub_image_preseq: (sub_image_size, sub_image_size) ndarray of aligned presequence image used for regression.

    Notes:
        - This function isn't really part of registration but needs registration to be run before it can be used and
            here seems like a good place to put it.
    """
    assert preseq.ndim == seq.ndim == 2, "Sequence and presequence image must be 2 dimensional"
    assert preseq.shape == seq.shape, "Presequence and sequence images must have the same shape"
    tile_size = seq.shape[-1]
    if sub_image_size is None:
        sub_image_size = int(tile_size // 4.095)
    n_sub_images = max(1, int(tile_size / sub_image_size))
    sub_image_shifts, sub_image_shift_score = (
        np.zeros((n_sub_images, n_sub_images, 2)),
        np.zeros((n_sub_images, n_sub_images)),
    )
    window = skimage.filters.window("hann", (sub_image_size, sub_image_size))
    for i, j in np.ndindex((n_sub_images, n_sub_images)):
        y_range = np.arange(i * sub_image_size, (i + 1) * sub_image_size)
        x_range = np.arange(j * sub_image_size, (j + 1) * sub_image_size)
        yx_range = np.ix_(y_range, x_range)
        sub_image_preseq, sub_image_seq = preseq[yx_range], seq[yx_range]
        sub_image_seq_windowed = sub_image_seq * window
        sub_image_preseq_windowed = sub_image_preseq * window
        # phase cross correlation struggles with all zeros so we will skip this sub-image if it is all zeros
        if min(np.max(sub_image_seq_windowed), np.max(sub_image_preseq_windowed)) == 0:
            sub_image_shifts[i, j] = np.array([0, 0])
            sub_image_shift_score[i, j] = 0
            continue
        sub_image_shifts[i, j] = skimage.registration.phase_cross_correlation(
            sub_image_preseq_windowed, sub_image_seq_windowed, overlap_ratio=0.75, disambiguate=True
        )[0]
        # Now calculate the correlation coefficient
        sub_image_seq_shifted = preprocessing.custom_shift(sub_image_seq, sub_image_shifts[i, j].astype(int))
        sub_image_shift_score[i, j] = np.corrcoef(sub_image_preseq.ravel(), sub_image_seq_shifted.ravel())[0, 1]
    # Now find the best sub-image
    max_score = np.max(sub_image_shift_score)
    i_max, j_max = np.argwhere(sub_image_shift_score == max_score)[0]
    shift_max_score = sub_image_shifts[i_max, j_max]
    y_range = np.arange(i_max * sub_image_size, (i_max + 1) * sub_image_size)
    x_range = np.arange(j_max * sub_image_size, (j_max + 1) * sub_image_size)
    yx_range = np.ix_(y_range, x_range)
    sub_image_preseq, sub_image_seq = preseq[yx_range], seq[yx_range]
    sub_image_seq = preprocessing.custom_shift(sub_image_seq, shift_max_score.astype(int))

    # Now find the top intensity_percentile pixels from the preseq image and use these for regression
    mask = sub_image_preseq > np.percentile(sub_image_preseq, intensity_percentile)

    sub_image_preseq_flat = sub_image_preseq[mask].ravel()
    sub_image_seq_flat = sub_image_seq[mask].ravel()
    # Least squares to find im = m * im_pre best fit coefficients
    m = np.linalg.lstsq(sub_image_preseq_flat[:, None], sub_image_seq_flat, rcond=None)[0]

    return m.item(), sub_image_seq, sub_image_preseq


def get_single_affine_transform(
    spot_yxz_base: np.ndarray,
    spot_yxz_transform: np.ndarray,
    z_scale_base: float,
    z_scale_transform: float,
    start_transform: np.ndarray,
    neighb_dist_thresh: float,
    tile_centre: np.ndarray,
    n_iter: int = 100,
    reg_constant_scale: Optional[float] = None,
    reg_constant_shift: Optional[float] = None,
    reg_transform: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, float, bool]:
    """
    Finds the affine transform taking `spot_yxz_base` to `spot_yxz_transform`.

    Args:
        spot_yxz_base: Point cloud want to find the shift from.
            spot_yxz_base[:, 2] is the z coordinate in units of z-pixels.
        spot_yxz_transform: Point cloud want to find the shift to.
            spot_yxz_transform[:, 2] is the z coordinate in units of z-pixels.
        z_scale_base: Scaling to put z coordinates in same units as yx coordinates for spot_yxz_base.
        z_scale_transform: Scaling to put z coordinates in same units as yx coordinates for spot_yxz_base.
        start_transform: `float [4 x 3]`.
            Start affine transform for iterative closest point.
            Typically, `start_transform[:3, :]` is identity matrix and
            `start_transform[3]` is approx yxz shift (z shift in units of xy pixels).
        neighb_dist_thresh: Distance between 2 points must be less than this to be constituted a match.
        tile_centre: int [3].
            yxz coordinates of centre of image where spot_yxz found on.
        n_iter: Max number of iterations to perform of ICP.
        reg_constant_scale: Constant used for scaling and rotation when doing regularized least squares.
            `None` means no regularized least squares performed.
            Typical = `5e8`.
        reg_constant_shift: Constant used for shift when doing regularized least squares.
            `None` means no regularized least squares performed.
            Typical = `500`
        reg_transform: `float [4 x 3]`.
            Transform to regularize to when doing regularized least squares.
            `None` means no regularized least squares performed.

    Returns:
        - `transform` - `float [4 x 3]`.
            `transform` is the final affine transform found.
        - `n_matches` - Number of matches found for each transform.
        - `error` - Average distance between neighbours below `neighb_dist_thresh`.
        - `is_converged` - `False` if max iterations reached before transform converged.
    """
    # ? Is this function needed or is it copying similar knowledge used in other functions? Is used in a ICP plot only
    spot_yxz_base = (spot_yxz_base - tile_centre) * [1, 1, z_scale_base]
    spot_yxz_transform = (spot_yxz_transform - tile_centre) * [1, 1, z_scale_transform]
    tree_transform = scipy.spatial.KDTree(spot_yxz_transform)
    neighbour = np.zeros(spot_yxz_base.shape[0], dtype=int)
    transform = start_transform.copy()
    for _ in range(n_iter):
        neighbour_last = neighbour.copy()
        transform, neighbour, n_matches, error = get_transform(
            spot_yxz_base,
            transform,
            spot_yxz_transform,
            neighb_dist_thresh,
            tree_transform,
            reg_constant_scale,
            reg_constant_shift,
            reg_transform,
        )

        is_converged = np.abs(neighbour - neighbour_last).max() == 0
        if is_converged:
            break

    return transform, n_matches, error, is_converged
