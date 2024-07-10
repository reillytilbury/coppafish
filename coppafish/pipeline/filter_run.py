import os
import time
from typing import Tuple

import numpy as np
from tqdm import tqdm
import zarr

from .. import extract, filter, log, utils
from ..filter import deconvolution
from ..filter import base as filter_base
from ..setup import NotebookPage
from ..utils import indexing, tiles_io


def run_filter(config: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage) -> Tuple[NotebookPage, NotebookPage]:
    """
    Read in extracted raw images, filter them, then re-save in a different location.

    Args:
        config (dict): dictionary obtained from 'filter' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.

    Returns:
        - NotebookPage: 'filter' notebook page.
        - NotebookPage: 'filter_debug' notebook page.

    Notes:
        - See `'filter'` and `'filter_debug'` sections of `notebook_page.py` file for description of variables.
    """
    if not nbp_basic.is_3d:
        NotImplementedError(f"2d coppafish is not stable, very sorry! :9")

    nbp = NotebookPage("filter", {"filter": config})
    nbp_debug = NotebookPage("filter_debug", {"filter": config})

    log.debug("Filter started")
    start_time = time.time()

    indices = indexing.create(
        nbp_basic,
        include_anchor_round=True,
        include_anchor_channel=True,
        include_dapi_seq=True,
        include_dapi_anchor=True,
        include_bad_trc=False,
    )

    max_ind = np.array(indices).max(0).tolist()
    shape = (max_ind[0] + 1, max_ind[1] + 1, max_ind[2] + 1, nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
    chunks = (1, 1, 1, 1_100, 1_100, 1)
    images = zarr.open_array(
        os.path.join(nbp_file.output_dir, "filter_images.zarr"),
        "w",
        shape=shape,
        chunks=chunks,
        zarr_version=2,
        dtype=np.float16,
    )

    INVALID_AUTO_THRESH = -1
    nbp_debug.invalid_auto_thresh = INVALID_AUTO_THRESH
    auto_thresh = np.full(
        (nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds, nbp_basic.n_channels),
        fill_value=INVALID_AUTO_THRESH,
        dtype=int,
    )

    nbp_debug.z_info = int(np.floor(nbp_basic.nz / 2))  # central z-plane to get info from.
    nbp_debug.r_dapi = config["r_dapi"]

    if nbp_debug.r_dapi is not None:
        filter_kernel_dapi = utils.strel.disk(nbp_debug.r_dapi)
    else:
        filter_kernel_dapi = None

    if config["deconvolve"]:
        if not os.path.isfile(nbp_file.psf):
            raise FileNotFoundError(f"Could not find the PSF at location {nbp_file.psf}")
        else:
            psf = np.moveaxis(np.load(nbp_file.psf)["arr_0"], 0, 2)  # Put z to last index
        # normalise psf so min is 0 and max is 1.
        psf = psf - psf.min()
        psf = psf / psf.max()
        pad_im_shape = (
            np.array([nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)])
            + np.array(config["wiener_pad_shape"]) * 2
        )
        wiener_filter = deconvolution.get_wiener_filter(psf, pad_im_shape, config["wiener_constant"])
        nbp_debug.psf = psf
    else:
        nbp_debug.psf = None

    with tqdm(total=len(indices), desc=f"Filtering extract images") as pbar:
        for t, r, c in indices:
            if (t, r, c) in nbp_basic.bad_trc:
                # In case of bad trc, save a blank image
                im_filtered = np.zeros((nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)), dtype=np.float16)
                images[t, r, c] = im_filtered
                del im_filtered
                pbar.update()
                continue
            file_path_raw = nbp_file.tile_unfiltered[t][r][c]
            raw_image_exists = tiles_io.image_exists(file_path_raw)
            pbar.set_postfix({"round": r, "tile": t, "channel": c})
            assert raw_image_exists, f"Raw, extracted file at\n\t{file_path_raw}\nnot found"
            # Get t, r, c image from raw files
            im_raw = tiles_io._load_image(file_path_raw)[:]
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
                if (im_filtered > np.iinfo(np.int32).max).sum() > 0:
                    log.warn(f"Converting to int32 has cut off pixels for {t=}, {r=}, {c=} filtered image")
                im_filtered = im_filtered.astype(np.float64)
                im_filtered = np.rint(im_filtered, np.zeros_like(im_filtered, dtype=np.int32), casting="unsafe")
                auto_thresh[t, r, c] = filter_base.compute_auto_thresh(
                    im_filtered, config["auto_thresh_multiplier"], nbp_debug.z_info
                )
            im_filtered = im_filtered.astype(np.float16)
            images[t, r, c] = im_filtered
            del im_filtered
            pbar.update()

    nbp.images = images
    nbp.auto_thresh = auto_thresh
    end_time = time.time()
    nbp_debug.time_taken = end_time - start_time
    log.debug("Filter complete")
    return nbp, nbp_debug
