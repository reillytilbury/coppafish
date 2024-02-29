import os
import pickle
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional

from ..setup.notebook import NotebookPage
from .. import utils, logging
from ..utils import tiles_io, indexing


def run_extract(
    config: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage, nbp_scale: NotebookPage
) -> Tuple[NotebookPage, NotebookPage, Optional[np.ndarray]]:
    """
    This reads in images from the raw `nd2` files, filters them and then saves them as `config[extract][file_type]`
    files in the tile directory. Also gets `auto_thresh` for use in turning images to point clouds and `hist_values`,
    `hist_counts` required for normalisation between channels.

    Args:
        config (dict): dictionary obtained from 'extract' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.
        nbp_scale (NotebookPage): 'scale' notebook page.

    Returns:
        - `NotebookPage[extract]`: page containing `auto_thresh` for use in turning images to point clouds and
            `hist_values`, `hist_counts` required for normalisation between channels.

    Notes:
        - See `'extract'` sections of `notebook_comments.json` file for description of the variables in each page.
    """
    # initialise notebook pages
    if not nbp_basic.is_3d:
        # config["deconvolve"] = False  # only deconvolve if 3d pipeline
        logging.error(
            NotImplementedError(f"coppafish 2d is not in a stable state, please contact a dev to add this. Sorry! ;(")
        )

    nbp = NotebookPage("extract")
    nbp.software_version = utils.system.get_software_version()
    nbp.revision_hash = utils.system.get_software_hash()
    nbp.file_type = config["file_type"]
    nbp.continuous_dapi = config["continuous_dapi"]

    logging.debug("Extraction started")

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
    first_rounds = indexing.unique(indices, 1)
    loaded_dasks = [utils.raw.load_dask(nbp_file, nbp_basic, r=r) for _, r, _ in first_rounds]
    with tqdm(
        total=len(indices), desc=f"Extracting raw {nbp_file.raw_extension} files to {config['file_type']}"
    ) as pbar:
        for t, r, c in indices:
            round_dask_array, metadata = loaded_dasks[r]
            if (t, r, c) in first_rounds:
                # Save raw metadata for each round
                metadata_path = os.path.join(
                    nbp_file.tile_unfiltered_dir, f"{nbp_file.raw_extension}_metadata_r{r}.pkl"
                )
                if not os.path.isfile(metadata_path) and metadata is not None:
                    # Dump all metadata to pickle file
                    with open(metadata_path, "wb") as file:
                        pickle.dump(metadata, file)
            # Load and save each image, if it does not already exist
            if r != pre_seq_round:
                file_path = nbp_file.tile_unfiltered[t][r][c]
                file_exists = tiles_io.image_exists(file_path, config["file_type"])
            else:
                file_path = nbp_file.tile_unfiltered[t][r][c]
                file_path = file_path[: file_path.index(config["file_type"])] + "_raw" + config["file_type"]
                file_exists = tiles_io.image_exists(file_path, config["file_type"])
            pbar.set_postfix({"round": r, "tile": t, "channel": c, "exists": str(file_exists)})
            if hist_counts_values_exists[t, r, c] and file_exists:
                # This image and its pixel histogram values already exist
                pbar.update()
                continue

            if file_exists:
                im = tiles_io._load_image(file_path, config["file_type"])
            if not file_exists:
                im = utils.raw.load_image(nbp_file, nbp_basic, t, c, round_dask_array, r, nbp_basic.use_z)
                im = im.astype(np.uint16, casting="safe")
                # yxz -> zyx
                im = im.transpose((2, 0, 1))
                if ((im.max((1, 2)) - im.min((1, 2))) == 0).any():
                    logging.warn(f"Raw image {t=}, {r=}, {c=} has a single valued plane!")
                tiles_io._save_image(im, file_path, config["file_type"])
            # Compute the counts of each possible uint16 pixel value for the image.
            hist_counts[:, t, r, c] = np.histogram(
                im, hist_values.size, range=(tiles_io.get_pixel_min(), tiles_io.get_pixel_max())
            )[0]
            np.savez_compressed(hist_counts_values_path, hist_counts, hist_values)
            del im
            pbar.update()
            del round_dask_array
    logging.debug("Extraction complete")
    return nbp
