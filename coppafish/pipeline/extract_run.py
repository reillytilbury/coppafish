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
    This reads in images from the raw `nd2` files, filters them and then saves them as zarr array files in the tile
    directory.

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
        log.error(
            NotImplementedError(f"coppafish 2d is not in a stable state, please contact a dev to add this. Sorry! ;(")
        )

    nbp = NotebookPage("extract", {"extract": config})
    nbp.num_rotations = config["num_rotations"]

    log.debug("Extraction started")

    if not os.path.isdir(nbp_file.tile_unfiltered_dir):
        os.mkdir(nbp_file.tile_unfiltered_dir)

    indices = indexing.create(
        nbp_basic,
        include_anchor_round=True,
        include_anchor_channel=True,
        include_dapi_seq=True,
        include_dapi_anchor=True,
    )
    indices_t = indexing.unique(indices, axis=0)
    indices_r = indexing.unique(indices, axis=1)
    with tqdm(
        total=len(indices_t) * len(indices_r),
        desc=f"Extracting raw {nbp_file.raw_extension} files",
    ) as pbar:
        for t, _, _ in indices_t:
            for _, r, _ in indices_r:
                pbar.set_postfix({"tile": t, "round": r})

                channels = list(indexing.find_channels_for(indices, tile=t, round=r))
                file_paths = [nbp_file.tile_unfiltered[t][r][c] for c in channels]
                files_exist = [tiles_io.image_exists(file_path) for file_path in file_paths]

                if all(files_exist):
                    pbar.update()
                    continue

                channel_images = utils.raw.load_image(nbp_file, nbp_basic, t=t, c=channels, r=r, use_z=nbp_basic.use_z)
                for im, c, file_path, file_exists in zip(channel_images, channels, file_paths, files_exist):
                    if file_exists:
                        continue
                    im = im.astype(np.uint16, casting="safe")
                    im = np.rot90(im, k=config["num_rotations"], axes=(0, 1))
                    z_plane_means = im.mean((0, 1))
                    if (z_plane_means < config["z_plane_mean_warning"]).any():
                        log.warn(
                            f"Raw image {t=}, {r=}, {c=} has dim z plane(s) at "
                            + f"{np.where(z_plane_means < config['z_plane_mean_warning'])[0].tolist()}. You may "
                            + f"wish to remove the affected image by setting `bad_trc = ({t}, {r}, {c}), (...` in "
                            + f"the basic_info config then re-run the pipeline with an empty output directory."
                        )
                    tiles_io._save_image(im, file_path)
                    del im
                pbar.update()
    log.debug("Extraction complete")
    return nbp
