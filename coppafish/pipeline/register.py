import os
import scipy
import pickle
import skimage
import itertools
import numpy as np
from tqdm import tqdm
from typing import Tuple
from ..setup import NotebookPage
from .. import find_spots
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
    nbp, nbp_debug = NotebookPage("register"), NotebookPage("register_debug")
    nbp.software_version = system.get_software_version()
    nbp.revision_hash = system.get_git_revision_hash()
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels
    # Initialise variable for ICP step
    neighb_dist_thresh = config["neighb_dist_thresh"]

    # Load in registration data from previous runs of the software
    registration_data = preprocessing.load_reg_data(nbp_file, nbp_basic)

    # Part 1: Channel registration
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
            registration_data["channel_registration"]["transform"][c] = (
                preprocessing.zyx_to_yxz_affine(A=cam_transform[cam_idx]))

    # Part 2: Round registration
    for t in tqdm(use_tiles, desc='Optical Flow on uncompleted tiles', total=len(use_tiles)):
        # Load in the anchor image and the round images. Note that here anchor means anchor round, not necessarily
        # anchor channel
        anchor_image = preprocessing.yxz_to_zyx(
            tiles_io.load_image(
                nbp_file,
                nbp_basic,
                nbp_extract.file_type,
                t=t,
                r=nbp_basic.anchor_round,
                c=nbp_basic.dapi_channel,
            )
        )
        use_rounds = nbp_basic.use_rounds + [nbp_basic.pre_seq_round] * nbp_basic.use_preseq
        upsample_factor = (1, config["downsample_factor"], config["downsample_factor"])
        for r in tqdm(use_rounds, desc='Round', total=len(use_rounds)):
            round_image = preprocessing.yxz_to_zyx(
                tiles_io.load_image(
                    nbp_file,
                    nbp_basic,
                    nbp_extract.file_type,
                    t=t,
                    r=r,
                    c=nbp_basic.dapi_channel,
                    suffix="_raw" if r == nbp_basic.pre_seq_round else "",
                )
            )
            # Now run the registration algorithm on this tile and round
            register_base.optical_flow_register(target=round_image, base=anchor_image,
                                                upsample_factor=upsample_factor,
                                                window_radius=config["window_radius"],
                                                smooth_sigma=config["smooth_sigma"],
                                                smooth_threshold=config["smooth_thresh"],
                                                clip_val=config["flow_clip"],
                                                output_dir=os.path.join(nbp_file.output_dir, "flow"),
                                                file_name=f"t{t}_r{r}.npy",
                                                )
        # Save the data to file
        with open(os.path.join(nbp_file.output_dir, "registration_data.pkl"), "wb") as f:
            pickle.dump(registration_data, f)
    del anchor_image, round_image

    # Part 3: ICP
    if "icp" not in registration_data.keys():
        # Initialise variables for ICP step
        n_rounds += nbp_basic.use_preseq + nbp_basic.use_anchor
        icp_correction = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
        n_matches = np.zeros((n_tiles, n_rounds, n_channels, config["icp_max_iter"]))
        mse = np.zeros((n_tiles, n_rounds, n_channels, config["icp_max_iter"]))
        converged = np.zeros((n_tiles, n_rounds, n_channels), dtype=bool)
        use_rounds = nbp_basic.use_rounds + [nbp_basic.pre_seq_round] * nbp_basic.use_preseq
        for t in tqdm(use_tiles, desc='ICP on all tiles', total=len(use_tiles)):
            ref_spots_t_ambient = find_spots.spot_yxz(
                nbp_find_spots.spot_yxz, t, nbp_basic.anchor_round, nbp_basic.anchor_channel, nbp_find_spots.spot_no
            )
            for r in use_rounds:
                flow_loc = os.path.join(nbp_file.output_dir, "flow", "smooth", f"t{t}_r{r}.npy")
                # convert flow to yxz
                flow_tr = preprocessing.flow_zyx_to_yxz(np.load(flow_loc, mmap_mode='r'))
                # put ref_spots from anchor frame into the round r frame
                ref_spots_t = preprocessing.apply_flow(flow=flow_tr, points=ref_spots_t_ambient)
                for c in use_channels:
                    # Only do ICP on non-degenerate tiles with more than ~ 100 spots
                    if nbp_find_spots.spot_no[t, r, c] < config["icp_min_spots"]:
                        print(
                            f"Tile {t}, round {r}, channel {c} has too few spots to run ICP. Using initial transform"
                            f" instead."
                        )
                        icp_correction[t, r, c] = registration_data["channel_registration"]["transform"][c]
                        continue
                    imaging_spots_trc = find_spots.spot_yxz(nbp_find_spots.spot_yxz, t, r, c, nbp_find_spots.spot_no)
                    icp_correction[t, r, c], n_matches[t, r, c], mse[t, r, c], converged[t, r, c] = register_base.icp(
                        yxz_base=ref_spots_t,
                        yxz_target=imaging_spots_trc,
                        dist_thresh=neighb_dist_thresh,
                        start_transform=registration_data["channel_registration"]["transform"][c],
                        n_iters=config["icp_max_iter"],
                        robust=False,
                    )
                    print(t, r, c, converged[t, r, c])
        # Save ICP data
        registration_data["icp"] = {
            "transform": icp_correction,
            "n_matches": n_matches,
            "mse": mse,
            "converged": converged,
        }
        # Save registration data externally
        with open(os.path.join(nbp_file.output_dir, "registration_data.pkl"), "wb") as f:
            pickle.dump(registration_data, f)
        del ref_spots_t_ambient, ref_spots_t, imaging_spots_trc
    # Now add the registration data to the notebook pages (nbp and nbp_debug)
    nbp.flow_dir = os.path.join(nbp_file.output_dir, "flow")
    nbp.icp_correction = registration_data["icp"]["transform"]

    nbp_debug.channel_transform_initial = registration_data["channel_registration"]["transform"]
    nbp_debug.n_matches = registration_data["icp"]["n_matches"]
    nbp_debug.mse = registration_data["icp"]["mse"]
    nbp_debug.converged = registration_data["icp"]["converged"]

    # first, let us blur the pre-seq round images
    if nbp_basic.use_preseq:
        if pre_seq_blur_radius is None:
            pre_seq_blur_radius = 3
        for t, c in tqdm(itertools.product(use_tiles, use_channels), total=len(use_tiles) * len(use_channels),
                         desc="Blurring pre-seq round images"):
            image_preseq = tiles_io.load_image(
                nbp_file, nbp_basic, nbp_extract.file_type, t=t, r=nbp_basic.pre_seq_round, c=c, suffix="_raw")
            image_preseq = scipy.ndimage.gaussian_filter(image_preseq, pre_seq_blur_radius)
            image_preseq = image_preseq.astype(np.int32)
            image_preseq = image_preseq - nbp_basic.tile_pixel_value_shift
            tiles_io.save_image(
                nbp_file, nbp_basic, nbp_extract.file_type, image_preseq, t=t, r=nbp_basic.pre_seq_round, c=c)

    # Load in the middle z-planes of each tile and compute the scale factors to be used when removing background
    # fluorescence
    if nbp_basic.use_preseq:
        bg_scale = np.zeros((n_tiles, n_rounds, n_channels))
        r_pre = nbp_basic.pre_seq_round
        use_rounds = nbp_basic.use_rounds
        mid_z = len(nbp_basic.use_z) // 2
        for t, c in tqdm(
            itertools.product(use_tiles, use_channels),
            desc="Computing background scale factors",
            total=len(use_tiles) * len(use_channels),
        ):
            # load in the flow for the pre-seq round
            flow_pre = np.load(os.path.join(nbp.flow_dir, "smooth", f"t{t}_r{r_pre}.npy")).astype(np.float32)
            grid = np.array(np.meshgrid(range(flow_pre.shape[1]), range(flow_pre.shape[2]),
                                        range(flow_pre.shape[3]), indexing='ij'))
            warp_pre = grid + flow_pre
            image_preseq = preprocessing.yxz_to_zyx(
                tiles_io.load_image(nbp_file, nbp_basic, nbp_extract.file_type, t=t, r=r_pre, c=c))
            # warp the pre-seq round image
            image_preseq = skimage.transform.warp(image_preseq, warp_pre, order=0)[mid_z]
            del flow_pre, warp_pre
            for r in use_rounds:
                # load in the flow for the round
                flow_r = np.load(os.path.join(nbp.flow_dir, "smooth", f"t{t}_r{r}.npy"))
                warp_r = grid + flow_r
                image_seq = preprocessing.yxz_to_zyx(
                    tiles_io.load_image(nbp_file, nbp_basic, nbp_extract.file_type, t=t, r=r, c=c))
                # warp the sequence round image
                image_seq = skimage.transform.warp(image_seq, warp_r, order=0)[mid_z]
                del flow_r, warp_r
                # Now compute the scale factor
                bg_scale[t, r, c] = register_base.brightness_scale(image_preseq, image_seq, 99)[0]
        # Now add the bg_scale to the nbp_filter page. To do this we need to delete the bg_scale attribute.
        nbp_filter.finalized = False
        del nbp_filter.bg_scale
        nbp_filter.bg_scale = bg_scale
        nbp_filter.finalized = True

    return nbp, nbp_debug
