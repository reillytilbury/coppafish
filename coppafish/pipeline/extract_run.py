import os
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from .. import log, utils
from ..setup import NotebookPage
from ..utils import indexing, tiles_io


def run_extract(
    config: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage
) -> Tuple[NotebookPage, NotebookPage, Optional[np.ndarray]]:
    """
    This reads in images from the raw `nd2` files, filters them and then saves them as `config[extract][file_type]`
    files in the tile directory. Also gets `hist_values` and `hist_counts` required for normalisation between channels.

    Args:
        config (dict): dictionary obtained from 'extract' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.

    Returns:
        - `NotebookPage[extract]`: page containing `auto_thresh` for use in turning images to point clouds and
            `hist_values`, `hist_counts` required for normalisation between channels.

    Notes:
        - See `'extract'` sections of `notebook_comments.json` file for description of the variables in each page.
    """
    # initialise notebook pages
    if not nbp_basic.is_3d:
        # config["deconvolve"] = False  # only deconvolve if 3d pipeline
        log.error(
            NotImplementedError(f"coppafish 2d is not in a stable state, please contact a dev to add this. Sorry! ;(")
        )

    nbp = NotebookPage("extract")
    nbp.file_type = config["file_type"]

    log.debug("Extraction started")

    if nbp_basic.use_preseq:
        pre_seq_round = nbp_basic.pre_seq_round
    else:
        pre_seq_round = None

    hist_counts_values_path = os.path.join(nbp_file.tile_unfiltered_dir, "hist_counts_values.npz")
    hist_values = np.arange(tiles_io.get_pixel_max() - tiles_io.get_pixel_min() + 1)
    hist_counts = np.zeros(
        (hist_values.size, nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds, nbp_basic.n_channels),
        dtype=int,
    )
    if os.path.isfile(hist_counts_values_path):
        results = np.load(hist_counts_values_path)
        hist_counts, hist_values = results["arr_0"], results["arr_1"]
    hist_counts_values_exists = ~(hist_counts == 0).all(0)

    if not os.path.isdir(nbp_file.tile_unfiltered_dir):
        os.mkdir(nbp_file.tile_unfiltered_dir)

    indices = indexing.create(
        nbp_basic,
        include_anchor_round=True,
        include_anchor_channel=True,
        include_preseq_round=True,
        include_dapi_seq=True,
        include_dapi_anchor=True,
        include_dapi_preseq=True,
    )
    indices_t = indexing.unique(indices, axis=0)
    indices_r = indexing.unique(indices, axis=1)
    # first_rounds = indexing.unique(indices, 1)
    with tqdm(
        total=len(indices_t) * len(indices_r),
        desc=f"Extracting raw {nbp_file.raw_extension} files to {config['file_type']}",
    ) as pbar:
        for t, _, _ in indices_t:
            for _, r, _ in indices_r:
                pbar.set_postfix({"tile": t, "round": r})
                channels = list(indexing.find_channels_for(indices, tile=t, round=r))

                if r != pre_seq_round:
                    file_paths = [nbp_file.tile_unfiltered[t][r][c] for c in channels]
                    files_exist = [tiles_io.image_exists(file_path, config["file_type"]) for file_path in file_paths]
                else:
                    file_paths = [nbp_file.tile_unfiltered[t][r][c] for c in channels]
                    for i, file_path in enumerate(file_paths):
                        file_paths[i] = file_path[: file_path.index(config["file_type"])] + "_raw" + config["file_type"]
                    files_exist = [tiles_io.image_exists(file_path, config["file_type"]) for file_path in file_paths]

                if hist_counts_values_exists[t, r, channels].all() and np.all(files_exist):
                    pbar.update()
                    continue

                channel_images: tuple[np.ndarray] = utils.raw.load_image(
                    nbp_file, nbp_basic, t=t, c=channels, r=r, use_z=list(nbp_basic.use_z)
                )
                for im, c, file_path, file_exists in zip(channel_images, channels, file_paths, files_exist):
                    if file_exists:
                        im = tiles_io._load_image(file_path, config["file_type"])
                    else:
                        im = im.astype(np.uint16, casting="safe")
                        # yxz -> zyx
                        im = im.transpose((2, 0, 1))
                        im = np.rot90(im, k=config["num_rotations"], axes=(1, 2))
                        if (im.mean((1, 2)) < config["z_plane_mean_warning"]).any():
                            log.warn(
                                f"Raw image {t=}, {r=}, {c=} has dim z plane(s). You may wish to remove the affected image by"
                                + f" setting `bad_trc = ({t}, {r}, {c}), (...` in the basic_info config and re-run the pipeline"
                                + " with an empty output directory."
                            )
                        tiles_io._save_image(im, file_path, config["file_type"])
                    # Compute the counts of each possible uint16 pixel value for the image.
                    hist_counts[:, t, r, c] = np.histogram(
                        im, hist_values.size, range=(tiles_io.get_pixel_min(), tiles_io.get_pixel_max())
                    )[0]
                    np.savez_compressed(hist_counts_values_path, hist_counts, hist_values)
                    del im
                pbar.update()
    log.debug("Extraction complete")
    return nbp
