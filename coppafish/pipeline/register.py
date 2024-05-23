import itertools
import os
import pickle
from typing import Tuple

import numpy as np
import scipy
from tqdm import tqdm
import zarr

from .. import find_spots, log
from ..register import preprocessing
from ..register import base as register_base
from ..setup import NotebookPage
from ..utils import tiles_io


def register(
    nbp_basic: NotebookPage,
    nbp_file: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_find_spots: NotebookPage,
    config: dict,
    pre_seq_blur_radius: float,
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
        nbp_extract: (NotebookPage): Extract notebook page.
        nbp_filter (NotebookPage): filter notebook page.
        nbp_find_spots (NotebookPage): Find Spots notebook page.
        config: Register part of the config dictionary.
        pre_seq_blur_radius: Radius of gaussian blur to apply to pre-seq round images.

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
    nbp = NotebookPage("register")
    nbp_debug = NotebookPage("register_debug")
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
        if not nbp_basic.channel_camera:
            cameras = [0] * n_channels
        else:
            cameras = list(set(nbp_basic.channel_camera))

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

            registration_data["channel_registration"]["transform"][c] = preprocessing.zyx_to_yxz_affine(
                A=cam_transform[cam_idx]
            )

    # Part 2: Round registration
    for t in tqdm(use_tiles, desc="Optical Flow on uncompleted tiles", total=len(use_tiles)):
        # Load in the anchor image and the round images. Note that here anchor means anchor round, not necessarily
        # anchor channel
        anchor_image = tiles_io.load_image(
            nbp_file,
            nbp_basic,
            nbp_extract.file_type,
            t=t,
            r=nbp_basic.anchor_round,
            c=nbp_basic.dapi_channel,
        )
        use_rounds = nbp_basic.use_rounds + [nbp_basic.pre_seq_round] * nbp_basic.use_preseq
        for r in tqdm(use_rounds, desc="Round", total=len(use_rounds)):
            round_image = tiles_io.load_image(
                nbp_file,
                nbp_basic,
                nbp_extract.file_type,
                t=t,
                r=r,
                c=nbp_basic.dapi_channel,
                suffix="_raw" if r == nbp_basic.pre_seq_round else "",
            )
            # Now run the registration algorithm on this tile and round
            register_base.optical_flow_register(
                target=round_image,
                base=anchor_image,
                sample_factor_yx=config["sample_factor_yx"],
                window_radius=config["window_radius"],
                smooth_sigma=config["smooth_sigma"],
                smooth_threshold=config["smooth_thresh"],
                clip_val=config["flow_clip"],
                output_dir=os.path.join(nbp_file.output_dir, "flow"),
                file_name=f"t{t}_r{r}.npy",
                n_cores=config["flow_cores"],
            )
        # Save the data to file
        with open(os.path.join(nbp_file.output_dir, "registration_data.pkl"), "wb") as f:
            pickle.dump(registration_data, f)
    del anchor_image, round_image

    # Part 3: ICP
    log.info("Running Iterative Closest Point (ICP)")
    # Initialise variables for ICP step
    ny, nx, nz = nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)
    use_rounds = nbp_basic.use_rounds
    n_rounds += nbp_basic.use_preseq + nbp_basic.use_anchor
    c_ref = nbp_basic.anchor_channel
    icp_correction = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    round_correction = np.zeros((n_tiles, n_rounds, 4, 3))
    channel_correction = np.zeros((n_tiles, n_channels, 4, 3))
    # Initialise variables for ICP step
    # 1. round icp stats
    n_matches_round = np.zeros((n_tiles, n_rounds, config["icp_max_iter"]))
    mse_round = np.zeros((n_tiles, n_rounds, config["icp_max_iter"]))
    converged_round = np.zeros((n_tiles, n_rounds), dtype=bool)
    # 2. channel icp stats
    n_matches_channel = np.zeros((n_tiles, n_channels, config["icp_max_iter"]))
    mse_channel = np.zeros((n_tiles, n_channels, config["icp_max_iter"]))
    converged_channel = np.zeros((n_tiles, n_channels), dtype=bool)
    for t in tqdm(use_tiles, desc="ICP on all tiles", total=len(use_tiles)):
        ref_spots_tr_ref = find_spots.spot_yxz(
            nbp_find_spots.spot_yxz, t, nbp_basic.anchor_round, nbp_basic.anchor_channel, nbp_find_spots.spot_no
        )
        # compute an affine correction to the round transforms. This is done by finding the best affine map that
        # takes the anchor round (post application of optical flow) to the other rounds.
        for r in use_rounds:
            # check if there are enough spots to run ICP
            if nbp_find_spots.spot_no[t, r, c_ref] < config["icp_min_spots"]:
                log.info(f"Tile {t}, round {r}, channel {c_ref} has too few spots to run ICP.")
                round_correction[t, r][:3, :3] = np.eye(3)
                continue
            # load in flow
            flow_loc = os.path.join(nbp_file.output_dir, "flow", "smooth", f"t{t}_r{r}.npy")
            flow_tr = zarr.load(flow_loc)[:]
            # load in spots
            ref_spots_tr = find_spots.spot_yxz(nbp_find_spots.spot_yxz, t, r, c_ref, nbp_find_spots.spot_no)
            # put ref_spots from round r frame into the anchor frame. This is done by applying the inverse of the
            # flow to the ref_spots
            ref_spots_tr = preprocessing.apply_flow(flow=-flow_tr, points=ref_spots_tr, round_to_int=False)
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
        # for preseq round, compute the correction between a high background channel in r_pre and the same channel in
        # round r. Then we can compose this correction with the correction for round r.
        if nbp_basic.use_preseq:
            c_bg = 15 if 15 in use_channels else 2
            r_pre, r_mid = nbp_basic.pre_seq_round, 3
            rounds_preseq_reg = [r_pre, r_mid]
            spots_preseq_req = []
            for r in rounds_preseq_reg:
                spots_preseq_req.append(
                    find_spots.spot_yxz(nbp_find_spots.spot_yxz, t, r, c_bg, nbp_find_spots.spot_no)
                )
                # apply flow to put these in anchor frame of ref
                # load in flow
                flow_loc = os.path.join(nbp_file.output_dir, "flow", "smooth", f"t{t}_r{r}.npy")
                flow_tr = zarr.load(flow_loc)[:]
                spots_preseq_req[-1] = preprocessing.apply_flow(
                    flow=-flow_tr, points=spots_preseq_req[-1], round_to_int=False
                )
            # Now compute the bg correction
            r_pre_correction = register_base.icp(
                yxz_base=spots_preseq_req[1],
                yxz_target=spots_preseq_req[0],
                dist_thresh_yx=neighb_dist_thresh_yx,
                dist_thresh_z=neighb_dist_thresh_z,
                start_transform=np.eye(4, 3),
                n_iters=config["icp_max_iter"],
                robust=False,
            )[0]
            # r7 -> r_pre = r7 -> r3 -> r_pre
            r_pre_correction = np.hstack((r_pre_correction, np.array([0, 0, 0, 1])[:, None]))
            r_mid_correction = np.hstack((round_correction[t, r_mid], np.array([0, 0, 0, 1])[:, None]))
            round_correction[t, r_pre] = (r_pre_correction @ r_mid_correction)[:, :3]

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
                # filter out spots that are out of bounds
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
                flow_loc = os.path.join(nbp_file.output_dir, "flow", "smooth", f"t{t}_r{r}.npy")
                flow_tr = zarr.load(flow_loc)[:]
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
        use_rounds = nbp_basic.use_rounds + [nbp_basic.pre_seq_round] * nbp_basic.use_preseq
        for t, r, c in itertools.product(use_tiles, use_rounds, use_channels):
            round_correction_matrix = np.hstack((round_correction[t, r], np.array([0, 0, 0, 1])[:, None]))
            channel_correction_matrix = np.hstack((channel_correction[t, c], np.array([0, 0, 0, 1])[:, None]))
            icp_correction[t, r, c] = (round_correction_matrix @ channel_correction_matrix)[:, :3]

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

    # Now add the registration data to the notebook pages (nbp and nbp_debug)
    nbp.flow_dir = os.path.join(nbp_file.output_dir, "flow")
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

    # first, let us blur the pre-seq round images
    if nbp_basic.use_preseq and registration_data["blur"] is False:
        if pre_seq_blur_radius is None:
            pre_seq_blur_radius = 3
        for t, c in tqdm(
            itertools.product(use_tiles, use_channels),
            desc="Blurring pre-seq images",
            total=len(use_tiles) * len(use_channels),
        ):
            image_preseq = tiles_io.load_image(
                nbp_file,
                nbp_basic,
                nbp_extract.file_type,
                t=t,
                r=nbp_basic.pre_seq_round,
                c=c,
                suffix="_raw",
                apply_shift=True,
            )
            image_preseq = scipy.ndimage.gaussian_filter(image_preseq, pre_seq_blur_radius)
            tiles_io.save_image(
                nbp_file, nbp_basic, nbp_extract.file_type, image_preseq, t=t, r=nbp_basic.pre_seq_round, c=c
            )
        for t in use_tiles:
            image_preseq = tiles_io.load_image(
                nbp_file,
                nbp_basic,
                nbp_extract.file_type,
                t=t,
                r=nbp_basic.pre_seq_round,
                c=nbp_basic.dapi_channel,
                suffix="_raw",
            )
            tiles_io.save_image(
                nbp_file,
                nbp_basic,
                nbp_extract.file_type,
                image_preseq,
                t=t,
                r=nbp_basic.pre_seq_round,
                c=nbp_basic.dapi_channel,
            )

        registration_data["blur"] = True
    # Save registration data externally
    with open(os.path.join(nbp_file.output_dir, "registration_data.pkl"), "wb") as f:
        pickle.dump(registration_data, f)

    # Load in the middle z-planes of each tile and compute the scale factors to be used when removing background
    # fluorescence
    log.debug("Compute background scale factors started")
    if nbp_basic.use_preseq:
        bg_scale = np.zeros((n_tiles, n_rounds, n_channels))
        r_pre = nbp_basic.pre_seq_round
        use_rounds = nbp_basic.use_rounds
        mid_z = len(nbp_basic.use_z) // 2
        tile_centre = np.array([nbp_basic.tile_sz // 2, nbp_basic.tile_sz // 2, mid_z])
        yx_rad, z_rad = min(nbp_basic.tile_sz // 2 - 1, 250), min(len(nbp_basic.use_z) // 2 - 1, 5)
        yxz = [
            np.arange(tile_centre[0] - yx_rad, tile_centre[0] + yx_rad),
            np.arange(tile_centre[1] - yx_rad, tile_centre[1] + yx_rad),
            np.arange(mid_z - z_rad, mid_z + z_rad),
        ]
        flow_ind = np.ix_(
            np.arange(3),
            np.arange(tile_centre[0] - yx_rad, tile_centre[0] + yx_rad),
            np.arange(tile_centre[1] - yx_rad, tile_centre[1] + yx_rad),
            np.arange(mid_z - z_rad, mid_z + z_rad),
        )
        new_origin = np.array([yxz[0][0], yxz[1][0], yxz[2][0]])
        for t, c in tqdm(
            itertools.product(use_tiles, use_channels),
            desc="Computing background scale factors",
            total=len(use_tiles) * len(use_channels),
        ):
            flow_t_pre_dir = os.path.join(nbp.flow_dir, "smooth", f"t{t}_r{r_pre}.npy")
            affine_correction = preprocessing.adjust_affine(icp_correction[t, r_pre, c].copy(), new_origin)
            im_pre = tiles_io.load_image(
                nbp_file,
                nbp_basic,
                nbp_extract.file_type,
                t=t,
                r=r_pre,
                c=c,
                suffix="_raw",
                apply_shift=True,
                yxz=yxz,
            )
            im_pre = preprocessing.transform_im(im_pre, affine_correction, flow_t_pre_dir, flow_ind)
            im_pre = im_pre[:, :, z_rad - 1 : z_rad + 1]
            bright = im_pre > np.percentile(im_pre, 99)
            im_pre = im_pre[bright]
            for r in use_rounds:
                flow_t_r_dir = os.path.join(nbp.flow_dir, "smooth", f"t{t}_r{r}.npy")
                affine_correction = preprocessing.adjust_affine(icp_correction[t, r, c].copy(), new_origin)
                im_r = tiles_io.load_image(
                    nbp_file,
                    nbp_basic,
                    nbp_extract.file_type,
                    t=t,
                    r=r,
                    c=c,
                    apply_shift=True,
                    yxz=yxz,
                )
                im_r = preprocessing.transform_im(im_r, affine_correction, flow_t_r_dir, flow_ind)
                im_r = im_r[:, :, z_rad - 1 : z_rad + 1]
                im_r = im_r[bright]
                bg_scale[t, r, c] = np.median(im_r) / np.median(im_pre)

        # Now add the bg_scale to the nbp_filter page. To do this we need to delete the bg_scale attribute.
        nbp.bg_scale = bg_scale
        log.debug("Compute background scale factors complete")

    log.debug("Register complete")
    return nbp, nbp_debug
