import itertools
import os
from typing import Tuple

import numpy as np
from tqdm import tqdm
import zarr

from .. import find_spots, log
from ..register import preprocessing
from ..register import base as register_base
from ..setup import NotebookPage


def register(
    nbp_basic: NotebookPage,
    nbp_file: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_find_spots: NotebookPage,
    config: dict,
) -> Tuple[NotebookPage, NotebookPage]:
    """
    Registration pipeline. Returns register Notebook Page.
    Finds affine transforms by using linear regression to find the best matrix (in the least squares sense) taking a
    bunch of points in one image to corresponding points in another. These shifts are found with a phase cross
    correlation algorithm.

    To get greater precision with this algorithm, we update these transforms with an iterative closest point algorithm.

    Args:
        nbp_basic (NotebookPage): Basic Info notebook page.
        nbp_file (NotebookPage): File Names notebook page.
        nbp_filter (NotebookPage): `filter` notebook page.
        nbp_find_spots (NotebookPage): Find Spots notebook page.
        config: Register part of the config dictionary.

    Returns:
        - nbp (NotebookPage): register notebook page.
        - nbp_debug (NotebookPage): register_debug notebook page.
    """

    # Break algorithm up into initialisation and then 3 parts.

    # Part 0: Initialise variables and load in data from previous runs of the software
    # Part 1: Generate subvolumes, use these in a regression to obtain an initial estimate for affine transform
    # Part 2: Compare the transforms across tiles to remove outlier transforms
    # Part 3: Correct the initial transform guesses with an ICP algorithm

    # Part 0: Initialisation
    # Initialise frequently used variables
    log.debug("Register started")
    nbp = NotebookPage("register", {"register": config})
    nbp_debug = NotebookPage("register_debug", {"register": config})
    use_tiles, use_rounds, use_channels = (
        list(nbp_basic.use_tiles),
        list(nbp_basic.use_rounds),
        list(nbp_basic.use_channels),
    )
    n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels
    # Initialise variable for ICP step
    neighb_dist_thresh_yx = config["neighb_dist_thresh_yx"]
    neighb_dist_thresh_z = config["neighb_dist_thresh_z"]
    if neighb_dist_thresh_z is None:
        neighb_dist_thresh_z = int(np.ceil(neighb_dist_thresh_yx * nbp_basic.pixel_size_xy / nbp_basic.pixel_size_z))

    # Load in registration data from previous runs of the software
    registration_data = preprocessing.load_reg_data(nbp_file, nbp_basic)

    # Part 1: Channel registration
    if registration_data["channel_registration"]["transform"].max() == 0:
        log.info("Running channel registration")
        if nbp_basic.channel_camera.size == 0:
            cameras = [0] * n_channels
        else:
            cameras = list(set(nbp_basic.channel_camera.tolist()))

        cameras.sort()
        anchor_cam_idx = cameras.index(nbp_basic.channel_camera[nbp_basic.anchor_channel])
        cam_transform = register_base.channel_registration(
            fluorescent_bead_path=nbp_file.fluorescent_bead_path,
            anchor_cam_idx=anchor_cam_idx,
            n_cams=len(cameras),
            bead_radii=config["bead_radii"],
        )
        # Now loop through all channels and set the channel transform to its cam transform
        for c in use_channels:
            cam_idx = cameras.index(nbp_basic.channel_camera[c])
            registration_data["channel_registration"]["transform"][c] = cam_transform[cam_idx]

    # Part 2: Round registration
    use_rounds = list(nbp_basic.use_rounds)
    corr_loc = os.path.join(nbp_file.output_dir, "corr.zarr")
    raw_loc = os.path.join(nbp_file.output_dir, "raw.zarr")
    smooth_loc = os.path.join(nbp_file.output_dir, "smooth.zarr")
    raw_smooth_shape = (
        max(nbp_basic.use_tiles) + 1,
        max(use_rounds) + 1,
        3,
        nbp_basic.tile_sz,
        nbp_basic.tile_sz,
        len(nbp_basic.use_z),
    )
    raw_smooth_chunks = (1, 1, None, 250, 250, 4)
    zarr.open_array(
        store=corr_loc,
        mode="w",
        shape=raw_smooth_shape[:2] + raw_smooth_shape[3:],
        dtype=np.float16,
        chunks=raw_smooth_chunks[:1] + raw_smooth_chunks[3:],
        zarr_version=2,
    )
    zarr.open_array(
        store=raw_loc,
        mode="w",
        shape=raw_smooth_shape,
        dtype=np.float16,
        chunks=raw_smooth_chunks,
        zarr_version=2,
    )
    zarr.open_array(
        store=smooth_loc,
        shape=raw_smooth_shape,
        mode="w",
        dtype=np.float16,
        chunks=raw_smooth_chunks,
        zarr_version=2,
    )
    for t in tqdm(use_tiles, desc="Optical Flow on uncompleted tiles", total=len(use_tiles)):
        # Load in the anchor image and the round images. Note that here anchor means anchor round, not necessarily
        # anchor channel
        anchor_image = nbp_filter.images[t, nbp_basic.anchor_round, nbp_basic.dapi_channel]
        for r in tqdm(use_rounds, desc="Round", total=len(use_rounds)):
            round_image = nbp_filter.images[t, r, nbp_basic.dapi_channel]
            # Now run the registration algorithm on this tile and round
            register_base.optical_flow_register(
                target=round_image,
                base=anchor_image,
                tile=t,
                round=r,
                raw_loc=raw_loc,
                corr_loc=corr_loc,
                smooth_loc=smooth_loc,
                sample_factor_yx=config["sample_factor_yx"],
                window_radius=config["window_radius"],
                smooth_sigma=config["smooth_sigma"],
                clip_val=config["flow_clip"],
                n_cores=config["flow_cores"],
            )
            nbp.flow_raw = zarr.open_array(raw_loc, "r")
            nbp.correlation = zarr.open_array(corr_loc, "r")
            nbp.flow = zarr.open_array(smooth_loc, "r")
    del anchor_image, round_image

    # Part 3: ICP
    log.info("Running Iterative Closest Point (ICP)")
    # Initialise variables for ICP step
    ny, nx, nz = nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)
    use_rounds, c_ref = nbp_basic.use_rounds, nbp_basic.anchor_channel
    icp_correction = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    round_correction = np.zeros((n_tiles, n_rounds, 4, 3))
    channel_correction = np.zeros((n_tiles, n_channels, 4, 3))
    # Initialise variables for ICP step
    # 1. round icp stats
    n_matches_round = np.zeros((n_tiles, n_rounds, config["icp_max_iter"]), dtype=np.int64)
    mse_round = np.zeros((n_tiles, n_rounds, config["icp_max_iter"]))
    converged_round = np.zeros((n_tiles, n_rounds), dtype=bool)
    # 2. channel icp stats
    n_matches_channel = np.zeros((n_tiles, n_channels, config["icp_max_iter"]), dtype=np.int64)
    mse_channel = np.zeros((n_tiles, n_channels, config["icp_max_iter"]))
    converged_channel = np.zeros((n_tiles, n_channels), dtype=bool)
    for t in tqdm(use_tiles, desc="ICP on all tiles", total=len(use_tiles)):
        # compute an affine correction to the round transforms. This is done by finding the best affine map that
        # takes the anchor round (post application of optical flow) to the other rounds.
        for r in use_rounds:
            # check if there are enough spots to run ICP
            if nbp_find_spots.spot_no[t, r, c_ref] < config["icp_min_spots"]:
                log.info(f"Tile {t}, round {r}, channel {c_ref} has too few spots to run ICP.")
                round_correction[t, r][:3, :3] = np.eye(3)
                continue
            # load in reference spots
            ref_spots_tr_ref = find_spots.spot_yxz(
                nbp_find_spots.spot_yxz, t, nbp_basic.anchor_round, nbp_basic.anchor_channel, nbp_find_spots.spot_no
            )
            # load in optical flow
            flow_tr = nbp.flow[t, r]
            # apply the flow to the reference spots to put anchor spots in the target frame
            ref_spots_tr_ref = preprocessing.apply_flow(flow=flow_tr, points=ref_spots_tr_ref, round_to_int=False)
            # load in target spots
            ref_spots_tr = find_spots.spot_yxz(nbp_find_spots.spot_yxz, t, r, c_ref, nbp_find_spots.spot_no)
            round_correction[t, r], n_matches_round[t, r], mse_round[t, r], converged_round[t, r] = register_base.icp(
                yxz_base=ref_spots_tr_ref,
                yxz_target=ref_spots_tr,
                dist_thresh_yx=neighb_dist_thresh_yx,
                dist_thresh_z=neighb_dist_thresh_z,
                start_transform=np.eye(4, 3),
                n_iters=config["icp_max_iter"],
                robust=False,
            )
            log.info(f"Tile: {t}, Round: {r}, Converged: {converged_round[t, r]}")
        # compute an affine correction to the channel transforms. This is done by finding the best affine map that
        # takes the anchor channel (post application of optical flow and round correction) to the other channels.
        for c in use_channels:
            im_spots_tc = np.zeros((0, 3))
            for r in use_rounds:
                im_spots_trc = find_spots.spot_yxz(nbp_find_spots.spot_yxz, t, r, c, nbp_find_spots.spot_no)
                # pad the spots with 1s to make them n_points x 4
                im_spots_trc = np.pad(im_spots_trc, ((0, 0), (0, 1)), constant_values=1)
                # put the spots from round r frame into the anchor frame. this is done in 2 steps:
                # 1. apply the inverse of the round correction to the spots
                round_correction_matrix = np.linalg.inv(
                    np.hstack((round_correction[t, r], np.array([0, 0, 0, 1])[:, None]))
                )[:, :3]
                im_spots_trc = np.round(im_spots_trc @ round_correction_matrix).astype(int)
                # remove spots that are out of bounds
                oob = (
                    (im_spots_trc[:, 0] < 0)
                    | (im_spots_trc[:, 0] >= ny)
                    | (im_spots_trc[:, 1] < 0)
                    | (im_spots_trc[:, 1] >= nx)
                    | (im_spots_trc[:, 2] < 0)
                    | (im_spots_trc[:, 2] >= nz)
                )
                im_spots_trc = im_spots_trc[~oob]
                # 2. apply the inverse of the flow to the spots
                flow_tr = nbp.flow[t, r]
                im_spots_trc = preprocessing.apply_flow(flow=-flow_tr, points=im_spots_trc, round_to_int=False)
                im_spots_tc = np.vstack((im_spots_tc, im_spots_trc))
            # check if there are enough spots to run ICP
            if im_spots_tc.shape[0] < config["icp_min_spots"]:
                log.info(f"Tile {t}, channel {c} has too few spots to run ICP.")
                channel_correction[t, c][:3, :3] = np.eye(3)
                continue
            # run ICP
            channel_correction[t, c], n_matches_channel[t, c], mse_channel[t, c], converged_channel[t, c] = (
                register_base.icp(
                    yxz_base=ref_spots_tr_ref,
                    yxz_target=im_spots_tc,
                    dist_thresh_yx=neighb_dist_thresh_yx,
                    dist_thresh_z=neighb_dist_thresh_yx,
                    start_transform=registration_data["channel_registration"]["transform"][c],
                    n_iters=config["icp_max_iter"],
                    robust=False,
                )
            )
            log.info(f"Tile: {t}, Channel: {c}, Converged: {converged_channel[t, c]}")

        # combine these corrections into the icp_correction
        use_rounds = list(nbp_basic.use_rounds)
        for t, r, c in itertools.product(use_tiles, use_rounds, use_channels):
            round_correction_matrix = np.hstack((round_correction[t, r], np.array([0, 0, 0, 1])[:, None]))
            channel_correction_matrix = np.hstack((channel_correction[t, c], np.array([0, 0, 0, 1])[:, None]))
            icp_correction[t, r, c] = (channel_correction_matrix @ round_correction_matrix)[:, :3]

        registration_data["icp"] = {
            "icp_correction": icp_correction,
            "round_correction": round_correction,
            "channel_correction": channel_correction,
            "n_matches_round": n_matches_round,
            "mse_round": mse_round,
            "converged_round": converged_round,
            "n_matches_channel": n_matches_channel,
            "mse_channel": mse_channel,
            "converged_channel": converged_channel,
        }

    nbp.icp_correction = registration_data["icp"]["icp_correction"]

    nbp_debug.channel_transform_initial = registration_data["channel_registration"]["transform"]
    nbp_debug.round_correction = registration_data["icp"]["round_correction"]
    nbp_debug.channel_correction = registration_data["icp"]["channel_correction"]
    nbp_debug.n_matches_round = registration_data["icp"]["n_matches_round"]
    nbp_debug.mse_round = registration_data["icp"]["mse_round"]
    nbp_debug.converged_round = registration_data["icp"]["converged_round"]
    nbp_debug.n_matches_channel = registration_data["icp"]["n_matches_channel"]
    nbp_debug.mse_channel = registration_data["icp"]["mse_channel"]
    nbp_debug.converged_channel = registration_data["icp"]["converged_channel"]

    # Save a registered image subsets for debugging/plotting purposes.
    preprocessing.generate_reg_images(nbp_basic, nbp_file, nbp_filter, nbp, nbp_debug)

    log.debug("Register complete")
    return nbp, nbp_debug
