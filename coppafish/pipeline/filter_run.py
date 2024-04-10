import os
import time
import numpy as np
from tqdm import tqdm
import numpy.typing as npt
from typing import Optional, Tuple

from .. import utils, extract, logging, filter
from ..utils import tiles_io, indexing
from ..filter import deconvolution
from ..filter import base as filter_base
from ..setup.notebook import NotebookPage


def run_filter(
    config: dict,
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_extract: NotebookPage,
) -> Tuple[NotebookPage, NotebookPage, Optional[npt.NDArray[np.uint16]]]:
    """
    Read in extracted raw images, filter them, then re-save in a different location.

    Args:
        config (dict): dictionary obtained from 'filter' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.
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
    nbp.revision_hash = utils.system.get_software_hash()

    logging.debug("Filter started")
    start_time = time.time()
    if not os.path.isdir(nbp_file.tile_dir):
        os.mkdir(nbp_file.tile_dir)
    file_type = nbp_extract.file_type

    INVALID_AUTO_THRESH = -1
    nbp_debug.invalid_auto_thresh = INVALID_AUTO_THRESH
    auto_thresh_path = os.path.join(nbp_file.tile_dir, "auto_thresh.npz")
    if os.path.isfile(auto_thresh_path):
        auto_thresh = np.load(auto_thresh_path)["arr_0"]
    else:
        auto_thresh = np.full(
            (nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds, nbp_basic.n_channels),
            fill_value=INVALID_AUTO_THRESH,
            dtype=int,
        )
    hist_counts_values_path = os.path.join(nbp_file.tile_dir, "hist_counts_values.npz")
    hist_values = np.arange(tiles_io.get_pixel_max() - tiles_io.get_pixel_min() + 1)
    hist_counts = np.zeros(
        (hist_values.size, nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds, nbp_basic.n_channels),
        dtype=int,
    )
    if os.path.isfile(hist_counts_values_path):
        results = np.load(hist_counts_values_path)
        hist_counts, hist_values = results["arr_0"], results["arr_1"]
    hist_counts_values_exists = ~(hist_counts == 0).all(0)

    nbp_debug.z_info = int(np.floor(nbp_basic.nz / 2))  # central z-plane to get info from.
    nbp_debug.r_dapi = config["r_dapi"]
    if config["r1"] is None:
        config["r1"] = extract.base.get_pixel_length(config["r1_auto_microns"], nbp_basic.pixel_size_xy)
    if config["r2"] is None:
        config["r2"] = config["r1"] * 2
    filter_kernel = utils.morphology.hanning_diff(config["r1"], config["r2"])

    if nbp_debug.r_dapi is not None:
        filter_kernel_dapi = utils.strel.disk(nbp_debug.r_dapi)
    else:
        filter_kernel_dapi = None

    if config["r_smooth"] is not None:
        smooth_kernel = np.ones(tuple(np.array(config["r_smooth"], dtype=int) * 2 - 1))
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
    compute_scale = True
    if os.path.isfile(nbp_file.scale):
        scale = float(filter_base.get_scale_from_txt(nbp_file.scale)[0])
        logging.info(f"Using image scale {scale} found at {nbp_file.scale}")
        compute_scale = False

    indices = indexing.create(
        nbp_basic,
        include_anchor_round=True,
        include_anchor_channel=True,
        include_preseq_round=True,
        include_dapi_seq=True,
        include_dapi_anchor=True,
        include_dapi_preseq=True,
        include_bad_trc=False,
    )
    with tqdm(total=len(indices), desc=f"Filtering extracted {nbp_extract.file_type} files") as pbar:
        for t, r, c in indices:
            if c == nbp_basic.dapi_channel:
                min_pixel_value = tiles_io.get_pixel_min()
                max_pixel_value = tiles_io.get_pixel_max()
            else:
                min_pixel_value = tiles_io.get_pixel_min() - nbp_basic.tile_pixel_value_shift
                max_pixel_value = tiles_io.get_pixel_max() - nbp_basic.tile_pixel_value_shift

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
            # assert raw_image_exists, f"Raw, extracted file at\n\t{file_path_raw}\nnot found"
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
            if (
                filtered_image_exists
                and hist_counts_values_exists[t, r, c]
                and auto_thresh[t, r, c] != INVALID_AUTO_THRESH
                and compute_scale == False
            ):
                # We already have everything we need for this tile, round, channel image.
                pbar.update()
                continue

            assert raw_image_exists, f"Raw, extracted file at\n\t{file_path_raw}\nnot found"
            # Get t, r, c image from raw files
            im_raw = tiles_io._load_image(file_path_raw, file_type)
            # zyx -> yxz
            im_raw = im_raw.transpose((1, 2, 0))
            im_filtered, bad_columns = extract.strip_hack(im_raw)  # check for faulty columns
            assert bad_columns.size == 0, f"Bad y column(s) were found during {t=}, {r=}, {c=} image filtering"
            del im_raw
            # Move to floating point before doing any filtering
            im_filtered = im_filtered.astype(np.float64)
            if config["deconvolve"]:
                # Deconvolves dapi images too
                im_filtered = filter.wiener_deconvolve(im_filtered, config["wiener_pad_shape"], wiener_filter)
            if c == nbp_basic.dapi_channel:
                if filter_kernel_dapi is not None:
                    im_filtered = utils.morphology.top_hat(im_filtered, filter_kernel_dapi)
                # DAPI images are shifted so all negative pixels are now positive so they can be saved without clipping
                im_filtered -= im_filtered.min()
            elif c != nbp_basic.dapi_channel:
                if config["difference_of_hanning"]:
                    im_filtered = utils.morphology.convolve_2d(im_filtered, filter_kernel)
                if config["r_smooth"] is not None:
                    # oa convolve uses lots of memory and much slower here.
                    im_filtered = utils.morphology.imfilter(im_filtered, smooth_kernel, oa=False)
                if (im_filtered > np.iinfo(np.int32).max).sum() > 0:
                    logging.warn(f"Converting to int32 has cut off pixels for {t=}, {r=}, {c=} filtered image")
                if compute_scale:
                    compute_scale = False
                    # Images cannot scale too much as to make negative pixels below the invalid pixel value of -15,000
                    scale = np.abs(min_pixel_value) / np.abs(im_filtered.min())
                    scale = min([scale, max_pixel_value / im_filtered.max()])
                    # A margin for max/min pixel variability between images. Scale can never be below 1.
                    scale = max([config["scale_multiplier"] * float(scale), 1])
                    logging.debug(f"{scale=} computed from {t=}, {r=}, {c=}")
                    # Save scale in case need to re-run without the notebook
                    filter_base.save_scale(nbp_file.scale, scale, scale)
                # Scale non DAPI images up by scale (or anchor_scale) factor after all filtering
                im_filtered = im_filtered.astype(np.float64) * scale
                im_filtered = np.rint(im_filtered, np.zeros_like(im_filtered, dtype=np.int32), casting="unsafe")
                auto_thresh[t, r, c] = filter_base.compute_auto_thresh(
                    im_filtered, config["auto_thresh_multiplier"], nbp_debug.z_info
                )
                np.savez(auto_thresh_path, auto_thresh)
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
                percent_clip_warn=config["percent_clip_warn"],
                percent_clip_error=config["percent_clip_error"],
            )
            # zyx -> yxz
            saved_im = saved_im.transpose((1, 2, 0))
            del im_filtered
            hist_counts[:, t, r, c] = np.histogram(
                saved_im, hist_values.size, range=(tiles_io.get_pixel_min(), tiles_io.get_pixel_max())
            )[0]
            np.savez_compressed(hist_counts_values_path, hist_counts, hist_values)
            del saved_im
            pbar.update()
        for t, r, c in nbp_basic.bad_trc:
            # in case of bad trc, save a blank image
            im_filtered = np.zeros((nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)), dtype=np.int32)
            saved_im = tiles_io.save_image(
                nbp_file,
                nbp_basic,
                file_type,
                im_filtered,
                t,
                r,
                c,
                suffix="_raw",
                num_rotations=config["num_rotations"],
                percent_clip_warn=config["percent_clip_warn"],
                percent_clip_error=config["percent_clip_error"],
            )
            del im_filtered, saved_im

    nbp.auto_thresh = auto_thresh
    nbp.image_scale = scale
    # Add a variable for bg_scale (actually computed in register)
    nbp.bg_scale = None
    end_time = time.time()
    nbp_debug.time_taken = end_time - start_time
    logging.debug("Filter complete")
    return nbp, nbp_debug
