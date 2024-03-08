import os
import scipy
import pickle
import itertools
import numpy as np
from tqdm import tqdm
from typing import Tuple

from ..setup import NotebookPage
from .. import find_spots, logging
from ..register import preprocessing
from ..register import base as register_base
from ..utils import system, tiles_io


def register(
    nbp_basic: NotebookPage,
    nbp_file: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_find_spots: NotebookPage,
    config: dict,
    tile_origin: np.ndarray,
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
        tile_origin: n_tiles x 3 ndarray of tile origins.
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
    logging.debug("Register started")
    nbp, nbp_debug = NotebookPage("register"), NotebookPage("register_debug")
    nbp.software_version = system.get_software_version()
    nbp.revision_hash = system.get_software_hash()
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels
    round_registration_channel = config["round_registration_channel"]
    if round_registration_channel is None:
        round_registration_channel = nbp_basic.anchor_channel
    # Initialise variables for ICP step
    icp_dist_thresh_yx = config["icp_dist_thresh_yx"]
    if nbp_basic.is_3d:
        icp_dist_thresh_z = config["icp_dist_thresh_z"]
        if icp_dist_thresh_z > icp_dist_thresh_yx:
            logging.warn(f"neighb_dist_thresh_z is set larger than neighb_dist_thresh_yx in the register config")

    # Load in registration data from previous runs of the software
    registration_data = preprocessing.load_reg_data(nbp_file, nbp_basic, config)
    uncompleted_tiles = np.setdiff1d(use_tiles, registration_data["round_registration"]["tiles_completed"])

    # Part 1: Initial affine transform
    # Start with channel registration
    logging.debug("Compute channel transforms started")
    pbar = tqdm(total=len(uncompleted_tiles))
    pbar.set_description(f"Running initial channel registration")
    if registration_data["channel_registration"]["transform"].max() == 0:
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
            registration_data["channel_registration"]["transform"][c] = cam_transform[cam_idx]
    logging.debug("Compute channel transforms complete")

    # round registration
    logging.debug("Compute round transforms started")
    with tqdm(total=len(uncompleted_tiles)) as pbar:
        pbar.set_description(f"Running initial round registration on all tiles")
        for t in uncompleted_tiles:
            # Load in the anchor image and the round images. Note that here anchor means anchor round, not necessarily
            # anchor channel
            use_dapi = round_registration_channel == nbp_basic.dapi_channel
            anchor_image = preprocessing.yxz_to_zyx(
                tiles_io.load_image(
                    nbp_file,
                    nbp_basic,
                    nbp_extract.file_type,
                    t=t,
                    r=nbp_basic.anchor_round,
                    c=round_registration_channel,
                    apply_shift=True,
                )
            )
            use_rounds = nbp_basic.use_rounds + [nbp_basic.pre_seq_round] * nbp_basic.use_preseq
            # split the rounds into two chunks, as we can't fit all of them into memory at once
            round_chunks = [use_rounds[: len(use_rounds) // 2], use_rounds[len(use_rounds) // 2 :]]
            for round_chunk in round_chunks:
                round_image = []
                for r in round_chunk:
                    if not (use_dapi and r == nbp_basic.anchor_round):
                        round_image.append(
                            preprocessing.yxz_to_zyx(
                                tiles_io.load_image(
                                    nbp_file,
                                    nbp_basic,
                                    nbp_extract.file_type,
                                    t=t,
                                    r=r,
                                    c=round_registration_channel,
                                    suffix="_raw" if r == nbp_basic.pre_seq_round else "",
                                )
                            )
                        )
                    else:
                        round_image.append(
                            preprocessing.yxz_to_zyx(
                                tiles_io.load_image(
                                    nbp_file,
                                    nbp_basic,
                                    nbp_extract.file_type,
                                    t=t,
                                    r=r,
                                    c=round_registration_channel,
                                    suffix="_raw" if r == nbp_basic.pre_seq_round else "",
                                ),
                            )
                        )
                round_reg_data = register_base.round_registration(
                    anchor_image=anchor_image, round_image=round_image, config=config
                )
                # Now save the data
                registration_data["round_registration"]["transform_raw"][t, round_chunk] = round_reg_data["transform"]
                registration_data["round_registration"]["shift"][t, round_chunk] = round_reg_data["shift"]
                registration_data["round_registration"]["shift_corr"][t, round_chunk] = round_reg_data["shift_corr"]
                registration_data["round_registration"]["position"][t, round_chunk] = round_reg_data["position"]
            # Now append anchor info and tile number to the registration data, then save to file
            registration_data["round_registration"]["tiles_completed"].append(t)
            # Save the data to file
            with open(os.path.join(nbp_file.output_dir, "registration_data.pkl"), "wb") as f:
                pickle.dump(registration_data, f)
            pbar.update(1)
    logging.debug("Compute round transforms complete")

    # Part 2: Regularisation
    logging.info("Regularising transforms")
    registration_data = register_base.regularise_transforms(
        registration_data=registration_data,
        tile_origin=np.roll(tile_origin, 1, axis=1),
        residual_threshold=config["residual_thresh"],
        use_tiles=nbp_basic.use_tiles,
        use_rounds=nbp_basic.use_rounds + [nbp_basic.pre_seq_round] * nbp_basic.use_preseq,
    )

    # Now combine all of these into single sub-vol transform array via composition
    logging.info("Combining sub-volume transforms")
    for t, r, c in itertools.product(
        use_tiles, nbp_basic.use_rounds + [nbp_basic.pre_seq_round] * nbp_basic.use_preseq, use_channels
    ):
        registration_data["initial_transform"][t, r, c] = preprocessing.zyx_to_yxz_affine(
            preprocessing.compose_affine(
                registration_data["channel_registration"]["transform"][c],
                registration_data["round_registration"]["transform"][t, r],
            )
        )
    # Now save registration data externally
    with open(os.path.join(nbp_file.output_dir, "registration_data.pkl"), "wb") as f:
        pickle.dump(registration_data, f)

    # Part 3: ICP
    logging.info("Running Iterative Closest Point (ICP)")
    if "icp" not in registration_data.keys():
        # Initialise variables for ICP step
        icp_transform = np.zeros((n_tiles, n_rounds + nbp_basic.use_anchor + nbp_basic.use_preseq, n_channels, 4, 3))
        n_matches = np.zeros(
            (n_tiles, n_rounds + nbp_basic.use_anchor + nbp_basic.use_preseq, n_channels, config["icp_max_iter"])
        )
        mse = np.zeros(
            (n_tiles, n_rounds + nbp_basic.use_anchor + nbp_basic.use_preseq, n_channels, config["icp_max_iter"])
        )
        converged = np.zeros((n_tiles, n_rounds + nbp_basic.use_anchor + nbp_basic.use_preseq, n_channels), dtype=bool)
        # Create a progress bar for the ICP step
        with tqdm(total=len(use_tiles) * len(use_rounds) * len(use_channels)) as pbar:
            pbar.set_description(f"Running ICP")
            for t in use_tiles:
                ref_spots_t = find_spots.spot_yxz(
                    nbp_find_spots.spot_yxz, t, nbp_basic.anchor_round, nbp_basic.anchor_channel, nbp_find_spots.spot_no
                )
                for r, c in itertools.product(
                    nbp_basic.use_rounds + [nbp_basic.pre_seq_round] * nbp_basic.use_preseq, use_channels
                ):
                    pbar.set_postfix({"Tile": t, "Round": r, "Channel": c})
                    # Only do ICP on non-degenerate tiles with more than ~ 100 spots, otherwise just use the
                    # starting transform
                    if nbp_find_spots.spot_no[t, r, c] < config["icp_min_spots"] and r in use_rounds:
                        logging.warn(
                            f"Tile {t}, round {r}, channel {c} has too few spots to run ICP. Using initial transform"
                            f" instead."
                        )
                        icp_transform[t, r, c] = registration_data["initial_transform"][t, r, c]
                        continue
                    imaging_spots_trc = find_spots.spot_yxz(nbp_find_spots.spot_yxz, t, r, c, nbp_find_spots.spot_no)
                    icp_transform[t, r, c], n_matches[t, r, c], mse[t, r, c], converged[t, r, c] = register_base.icp(
                        yxz_base=ref_spots_t,
                        yxz_target=imaging_spots_trc,
                        dist_thresh_yx=icp_dist_thresh_yx,
                        dist_thresh_z=icp_dist_thresh_z,
                        start_transform=registration_data["initial_transform"][t, r, c],
                        n_iters=config["icp_max_iter"],
                        robust=False,
                    )
                    pbar.update(1)
        # Save ICP data
        registration_data["icp"] = {
            "transform": icp_transform,
            "n_matches": n_matches,
            "mse": mse,
            "converged": converged,
        }
        # Save registration data externally
        with open(os.path.join(nbp_file.output_dir, "registration_data.pkl"), "wb") as f:
            pickle.dump(registration_data, f)

    # Add round statistics to debugging page.
    nbp_debug.position = registration_data["round_registration"]["position"]
    nbp_debug.round_shift = registration_data["round_registration"]["shift"]
    nbp_debug.round_shift_corr = registration_data["round_registration"]["shift_corr"]
    nbp_debug.round_transform_raw = registration_data["round_registration"]["transform_raw"]

    # Now add the channel registration statistics
    nbp_debug.channel_transform = registration_data["channel_registration"]["transform"]

    # Now add the ICP statistics
    nbp_debug.mse = registration_data["icp"]["mse"]
    nbp_debug.n_matches = registration_data["icp"]["n_matches"]
    nbp_debug.converged = registration_data["icp"]["converged"]

    # Now add relevant information to the nbp object
    nbp.round_transform = registration_data["round_registration"]["transform"]
    nbp.channel_transform = registration_data["channel_registration"]["transform"]
    nbp.initial_transform = registration_data["initial_transform"]
    # combine icp transform, channel transform and initial transform to get final transform
    transform = np.zeros((n_tiles, n_rounds + nbp_basic.use_anchor + nbp_basic.use_preseq, n_channels, 4, 3))
    use_rounds_new = nbp_basic.use_rounds + [nbp_basic.pre_seq_round] * nbp_basic.use_preseq
    transform[:, use_rounds_new] = registration_data["icp"]["transform"][:, use_rounds_new]
    nbp.transform = transform

    # first, let us blur the pre-seq round images
    logging.info("Blurring presequence round images")
    if nbp_basic.use_preseq:
        if pre_seq_blur_radius is None:
            pre_seq_blur_radius = 3
        has_dapi = nbp_basic.dapi_channel is not None
        for t, c in itertools.product(use_tiles, use_channels + has_dapi * [nbp_basic.dapi_channel]):
            image_preseq = tiles_io.load_image(
                nbp_file, nbp_basic, nbp_extract.file_type, t=t, r=nbp_basic.pre_seq_round, c=c, suffix="_raw"
            )
            image_preseq = scipy.ndimage.gaussian_filter(image_preseq, pre_seq_blur_radius)
            tiles_io.save_image(
                nbp_file, nbp_basic, nbp_extract.file_type, image_preseq, t=t, r=nbp_basic.pre_seq_round, c=c
            )

    # Load in the middle z-planes of each tile and compute the scale factors to be used when removing background
    # fluorescence
    logging.debug("Compute background scale factors started")
    if nbp_basic.use_preseq:
        bg_scale = np.zeros((n_tiles, n_rounds, n_channels))
        r_pre = nbp_basic.pre_seq_round
        use_rounds = nbp_basic.use_rounds
        mid_z = nbp_basic.use_z[len(nbp_basic.use_z) // 2]
        for t, c in tqdm(
            itertools.product(use_tiles, use_channels),
            desc="Computing background scale factors",
            total=len(use_tiles) * len(use_channels),
        ):
            # load in pre-seq round
            transform_pre = preprocessing.invert_affine(preprocessing.yxz_to_zyx_affine(transform[t, r_pre, c]))
            z_scale_pre = transform_pre[0, 0]
            z_shift_pre = transform_pre[0, 3]
            mid_z_pre = int(np.clip((mid_z - z_shift_pre) / z_scale_pre, 0, len(nbp_basic.use_z) - 1))
            yxz = [None, None, mid_z_pre]
            image_preseq = tiles_io.load_image(nbp_file, nbp_basic, nbp_extract.file_type, t=t, r=r_pre, c=c, yxz=yxz)
            image_preseq = image_preseq.astype(np.float32)
            # we have to load in inverse transform to use scipy.ndimage.affine_transform
            inv_transform_pre_yx = preprocessing.yxz_to_zyx_affine(transform[t, r_pre, c])[1:, 1:]
            image_preseq = scipy.ndimage.affine_transform(image_preseq, inv_transform_pre_yx)
            for r in use_rounds:
                transform_seq = preprocessing.invert_affine(preprocessing.yxz_to_zyx_affine(transform[t, r, c]))
                z_scale_seq = transform_seq[0, 0]
                z_shift_seq = transform_seq[0, 3]
                mid_z_seq = int(np.clip((mid_z - z_shift_seq) / z_scale_seq, 0, len(nbp_basic.use_z) - 1))
                yxz = [None, None, mid_z_seq]
                image_seq = tiles_io.load_image(nbp_file, nbp_basic, nbp_extract.file_type, t=t, r=r, c=c, yxz=yxz)
                image_seq = image_seq.astype(np.float32)
                # we have to load in inverse transform to use scipy.ndimage.affine_transform
                inv_transform_seq_yx = preprocessing.yxz_to_zyx_affine(transform[t, r, c])[1:, 1:]
                image_seq = scipy.ndimage.affine_transform(image_seq, inv_transform_seq_yx)
                # Now compute the scale factor
                bg_scale[t, r, c] = register_base.brightness_scale(image_preseq, image_seq, 99)[0]
                if bg_scale[t, r, c] < 0:
                    logging.warn(f"Background scale for {t=}, {r=}, {c=} is negative")
        # Now add the bg_scale to the nbp_filter page. To do this we need to delete the bg_scale attribute.
        nbp_filter.finalized = False
        del nbp_filter.bg_scale
        logging.debug("Compute background scale factors complete")
        nbp_filter.bg_scale = bg_scale
        nbp_filter.finalized = True

    logging.debug("Register complete")
    return nbp, nbp_debug
