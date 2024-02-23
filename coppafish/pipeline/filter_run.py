import os
import time
import numpy as np
from tqdm import tqdm
import numpy.typing as npt
from typing import Optional, Tuple

from .. import utils, extract, logging
from ..utils import tiles_io, indexing
from ..filter import deconvolution
from ..filter import base as filter_base
from ..setup.notebook import NotebookPage


def run_filter(
    config: dict,
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_scale: NotebookPage,
    nbp_extract: NotebookPage,
) -> Tuple[NotebookPage, NotebookPage, Optional[npt.NDArray[np.uint16]]]:
    """
    Read in extracted raw images, filter them, then re-save in a different location.

    Args:
        config (dict): dictionary obtained from 'filter' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.
        nbp_scale (NotebookPage): 'scale' notebook page.
        nbp_extract (NotebookPage): 'extract' notebook page.
        image_t_raw (`(n_rounds x n_channels x nz x ny x nx) ndarray[uint16]`, optional): extracted image for single
            tile. Can only be used for a single tile notebooks. Default: not given.

    Returns:
        - NotebookPage: 'filter' notebook page.
        - NotebookPage: 'filter_debug' notebook page.
        - `(n_rounds x n_channels x nz x ny x nx) ndarray[uint16]` or None: if `nbp_basic.use_tiles` is a single tile,
            returns all saved tile images. Otherwise, returns None.

    Notes:
        - See `'filter'` and `'filter_debug'` sections of `notebook_comments.json` file for description of variables.
    """
    if not nbp_basic.is_3d:
        NotImplementedError(f"2d coppafish is not stable, very sorry! :9")

    nbp = NotebookPage("filter")
    nbp_debug = NotebookPage("filter_debug")
    nbp.software_version = utils.system.get_software_version()
    nbp.revision_hash = utils.system.get_git_revision_hash()

    logging.debug("Filter started")
    start_time = time.time()
    if not os.path.isdir(nbp_file.tile_dir):
        os.mkdir(nbp_file.tile_dir)
    file_type = nbp_extract.file_type

    auto_thresh_path = os.path.join(nbp_file.tile_dir, "auto_thresh.npz")
    if os.path.isfile(auto_thresh_path):
        auto_thresh = np.load(auto_thresh_path)["arr_0"]
    else:
        auto_thresh = np.full(
            (nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds, nbp_basic.n_channels),
            fill_value=-1,
            dtype=int,
        )
    hist_counts_values_path = os.path.join(nbp_file.tile_dir, "hist_counts_values.npz")
    hist_values = np.arange(np.iinfo(np.uint16).max - np.iinfo(np.uint16).min + 1)
    hist_counts = np.zeros(
        (hist_values.size, nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds, nbp_basic.n_channels),
        dtype=int,
    )
    if os.path.isfile(hist_counts_values_path):
        results = np.load(hist_counts_values_path)
        hist_counts, hist_values = results["arr_0"], results["arr_1"]
    hist_counts_values_exists = ~(hist_counts == 0).all(0)

    # initialise debugging info as 'debug' page
    nbp_debug.n_clip_pixels = np.zeros_like(auto_thresh, dtype=int)
    nbp_debug.clip_extract_scale = np.zeros_like(auto_thresh)

    if config["n_clip_error"] is None:
        # default is 1% of pixels on single z-plane
        config["n_clip_error"] = int(nbp_basic.tile_sz * nbp_basic.tile_sz / 100)

    nbp_debug.z_info = int(np.floor(nbp_basic.nz / 2))  # central z-plane to get info from.
    hist_bin_edges = np.concatenate((hist_values - 0.5, hist_values[-1:] + 0.5))
    nbp_debug.r_dapi = config["r_dapi"]
    filter_kernel = utils.morphology.hanning_diff(nbp_scale.r1, nbp_scale.r2)
    if nbp_debug.r_dapi is not None:
        filter_kernel_dapi = utils.strel.disk(nbp_debug.r_dapi)
    else:
        filter_kernel_dapi = None

    if nbp_scale.r_smooth is not None:
        # smooth_kernel = utils.strel.fspecial(*tuple(nbp_scale.r_smooth]))
        smooth_kernel = np.ones(tuple(np.array(nbp_scale.r_smooth, dtype=int) * 2 - 1))
        smooth_kernel = smooth_kernel / np.sum(smooth_kernel)
    if config["deconvolve"]:
        if not os.path.isfile(nbp_file.psf):
            (
                spot_images,
                config["psf_intensity_thresh"],
                psf_tiles_used,
            ) = deconvolution.get_psf_spots(
                nbp_file,
                nbp_basic,
                nbp_extract,
                nbp_basic.anchor_round,
                nbp_basic.use_tiles,
                nbp_basic.anchor_channel,
                nbp_basic.use_z,
                config["psf_detect_radius_xy"],
                config["psf_detect_radius_z"],
                config["psf_min_spots"],
                config["psf_intensity_thresh"],
                config["auto_thresh_multiplier"],
                config["psf_isolation_dist"],
                config["psf_shape"],
                config["psf_max_spots"],
            )
            psf = deconvolution.get_psf(spot_images, config["psf_annulus_width"])
            np.save(nbp_file.psf, np.moveaxis(psf, 2, 0))  # save with z as first axis
        else:
            # Know psf only computed for 3D pipeline hence know ndim=3
            psf = np.moveaxis(np.load(nbp_file.psf), 0, 2)  # Put z to last index
            psf_tiles_used = None
        # normalise psf so min is 0 and max is 1.
        psf = psf - psf.min()
        psf = psf / psf.max()
        pad_im_shape = (
            np.array([nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)])
            + np.array(config["wiener_pad_shape"]) * 2
        )
        wiener_filter = deconvolution.get_wiener_filter(psf, pad_im_shape, config["wiener_constant"])
        nbp_debug.psf = psf
        if config["psf_intensity_thresh"] is not None:
            config["psf_intensity_thresh"] = int(config["psf_intensity_thresh"])
        nbp_debug.psf_intensity_thresh = config["psf_intensity_thresh"]
        nbp_debug.psf_tiles_used = psf_tiles_used
    else:
        nbp_debug.psf = None
        nbp_debug.psf_intensity_thresh = None
        nbp_debug.psf_tiles_used = None

    indices = indexing.create(
        nbp_basic,
        include_anchor_round=True,
        include_anchor_channel=True,
        include_preseq_round=True,
        include_dapi_seq=True,
        include_dapi_anchor=True,
        include_dapi_preseq=True,
    )
    with tqdm(total=len(indices), desc=f"Filtering extracted {nbp_extract.file_type} files") as pbar:
        for t, r, c in indices:
            if r == nbp_basic.anchor_round:
                scale = nbp_scale.scale_anchor
            else:
                scale = nbp_scale.scale
            if c == nbp_basic.dapi_channel:
                max_pixel_value = np.iinfo(np.uint16).max
            else:
                max_pixel_value = np.iinfo(np.uint16).max - nbp_basic.tile_pixel_value_shift

            if r != nbp_basic.pre_seq_round:
                file_path = nbp_file.tile[t][r][c]
                filtered_image_exists = tiles_io.image_exists(file_path, file_type)
                file_path_raw = nbp_file.tile_unfiltered[t][r][c]
                raw_image_exists = tiles_io.image_exists(file_path_raw, file_type)
            if r == nbp_basic.pre_seq_round:
                file_path = nbp_file.tile[t][r][c]
                file_path = file_path[: file_path.index(file_type)] + "_raw" + file_type
                filtered_image_exists = tiles_io.image_exists(file_path, file_type)
                file_path_raw = nbp_file.tile_unfiltered[t][r][c]
                file_path_raw = file_path_raw[: file_path_raw.index(file_type)] + "_raw" + file_type
                raw_image_exists = tiles_io.image_exists(file_path_raw, file_type)
            assert raw_image_exists, f"Raw, extracted file at\n\t{file_path_raw}\nnot found"
            pbar.set_postfix(
                {
                    "round": r,
                    "tile": t,
                    "channel": c,
                    "exists": str(filtered_image_exists).lower(),
                }
            )
            if filtered_image_exists and hist_counts_values_exists[t, r, c] and c == nbp_basic.dapi_channel:
                pbar.update()
                continue
            if filtered_image_exists and hist_counts_values_exists[t, r, c] and auto_thresh[t, r, c] != -1:
                # We already have everything we need for this tile, round, channel image.
                pbar.update()
                continue

            # Get t, r, c image from raw files
            im_raw = tiles_io._load_image(file_path_raw, file_type)
            # zyx -> yxz
            im_raw = im_raw.transpose((1, 2, 0))
            im_filtered, bad_columns = extract.strip_hack(im_raw)  # check for faulty columns
            assert bad_columns.size == 0, f"Bad y column(s) were found during image filtering: {t=}, {r=}, {c=}"
            del im_raw
            if config["deconvolve"]:
                # Deconvolves dapi images too
                im_filtered = deconvolution.wiener_deconvolve(im_filtered, config["wiener_pad_shape"], wiener_filter)
            if c == nbp_basic.dapi_channel:
                if filter_kernel_dapi is not None:
                    im_filtered = utils.morphology.top_hat(im_filtered, filter_kernel_dapi)
            elif c != nbp_basic.dapi_channel:
                # im converted to float in convolve_2d so no point changing dtype beforehand.
                im_filtered = utils.morphology.convolve_2d(im_filtered, filter_kernel) * scale
                if nbp_scale.r_smooth is not None:
                    # oa convolve uses lots of memory and much slower here.
                    im_filtered = utils.morphology.imfilter(im_filtered, smooth_kernel, oa=False)
                # get_info is quicker on int32 so do this conversion first.
                im_filtered = np.rint(im_filtered, np.zeros_like(im_filtered, dtype=np.int32), casting="unsafe")
                # only use image unaffected by strip_hack to get information from tile
                (
                    auto_thresh[t, r, c],
                    _,
                    nbp_debug.n_clip_pixels[t, r, c],
                    nbp_debug.clip_extract_scale[t, r, c],
                ) = filter_base.get_filter_info(
                    im_filtered,
                    config["auto_thresh_multiplier"],
                    hist_bin_edges,
                    max_pixel_value,
                    scale,
                    nbp_debug.z_info,
                )
                np.savez(auto_thresh_path, auto_thresh)
                # Deal with pixels outside uint16 range when saving
                if nbp_debug.n_clip_pixels[t, r, c] > config["n_clip_warn"]:
                    logging.warn(
                        f"\nTile {t}, round {r}, channel {c} has "
                        f"{nbp_debug.n_clip_pixels[t, r, c]} pixels\n"
                        f"that will be clipped when converting to uint16."
                    )
                if nbp_debug.n_clip_pixels[t, r, c] > config["n_clip_error"]:
                    logging.error(
                        ValueError(f"{t=}, {r=}, {c=} filter image clipped {nbp_debug.n_clip_pixels[t, r, c]} pixels")
                    )
            # Delay gaussian blurring of preseq until after reg to give it a better chance
            saved_im = tiles_io.save_image(
                nbp_file,
                nbp_basic,
                file_type,
                im_filtered,
                t,
                r,
                c,
                suffix="_raw" if r == nbp_basic.pre_seq_round else "",
                num_rotations=config["num_rotations"],
            )
            # zyx -> yxz
            saved_im = saved_im.transpose((1, 2, 0))
            del im_filtered
            hist_counts[:, t, r, c] = np.histogram(saved_im, hist_values.size)[0]
            np.savez_compressed(hist_counts_values_path, hist_counts, hist_values)
            del saved_im
            pbar.update()
    nbp.auto_thresh = auto_thresh
    # Add a variable for bg_scale (actually computed in register)
    nbp.bg_scale = None
    end_time = time.time()
    nbp_debug.time_taken = end_time - start_time
    logging.debug("Filter complete")
    return nbp, nbp_debug
