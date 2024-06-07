from concurrent.futures import ProcessPoolExecutor
import math as maths

import numpy as np
import numpy.typing as npt
import scipy
import torch
import tqdm
from typing_extensions import assert_type

from .. import log, utils
from ..register import preprocessing
from ..setup import NotebookPage


def load_spot_colours(
    nbp_basic: NotebookPage,
    nbp_file: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_register: NotebookPage,
    nbp_register_debug: NotebookPage,
    tile: int,
    dtype: np.dtype = np.float16,
) -> npt.NDArray:
    """
    Load the full registered image for every sequencing round/channel for the given tile. No post-processing is
    applied, including background subtraction and colour normalisation. The images are correctly centred around the
    zero mark, so dtype must support negative numbers.

    Args:
        - nbp_basic (NotebookPage): `basic_info` notebook page.
        - nbp_file (NotebookPage): `file_names` notebook page.
        - nbp_extract (NotebookPage): `extract` notebook page.
        - nbp_register (NotebookPage): `register` notebook page.
        - nbp_register_debug (NotebookPage): `register_debug` notebook page.
        - tile (int): tile index.
        - dtype (numpy dtype): datatype to return images. Default: float16.

    Returns:
        (`(im_y x im_x x im_z x n_rounds_use x n_channels_use) ndarray`) spot_colours: tile loaded spot colours.
    """
    assert_type(nbp_basic, NotebookPage)
    assert_type(nbp_file, NotebookPage)
    assert_type(nbp_extract, NotebookPage)
    assert_type(nbp_register, NotebookPage)
    assert_type(nbp_register_debug, NotebookPage)
    assert_type(tile, int)
    assert_type(dtype, np.dtype)

    image_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
    n_channels_use = len(nbp_basic.use_channels)
    colours = np.zeros(image_shape + (len(nbp_basic.use_rounds), n_channels_use), dtype=dtype)

    n_rounds_batch = max(
        1, maths.floor(4.4e7 * utils.system.get_available_memory() / (n_channels_use * np.prod(image_shape)))
    )
    log.debug(f"{n_rounds_batch=}")
    final_round = nbp_basic.use_rounds[-1]
    half_pixel_0, half_pixel_1, half_pixel_2 = [1 / image_shape[i] for i in range(3)]
    image_batch = torch.zeros((0, len(nbp_basic.use_channels)) + image_shape, dtype=torch.float32)
    grid_0, grid_1, grid_2 = torch.meshgrid(
        torch.linspace(half_pixel_0 - 1, 1 - half_pixel_0, image_shape[0]),
        torch.linspace(half_pixel_1 - 1, 1 - half_pixel_1, image_shape[1]),
        torch.linspace(half_pixel_2 - 1, 1 - half_pixel_2, image_shape[2]),
        indexing="ij",
    )
    base_grid = torch.cat((grid_2[:, :, :, None], grid_1[:, :, :, None], grid_0[:, :, :, None]), dim=3)[None]
    grids = torch.zeros((0,) + image_shape + (3,), dtype=torch.float32)
    for i, r in enumerate(tqdm.tqdm(nbp_basic.use_rounds, desc=f"Loading tile {tile} colours", unit="round")):
        suffix = "_raw" if r == nbp_basic.pre_seq_round else ""
        image_r = tuple()
        # While each image is being affine transformed, we are disk loading the next image at the same time.
        with ProcessPoolExecutor(max_workers=len(nbp_basic.use_channels)) as executor:
            futures = []
            for c in nbp_basic.use_channels:
                im_c = utils.tiles_io.load_image(
                    nbp_file, nbp_basic, nbp_extract.file_type, tile, r, c, yxz=None, suffix=suffix
                ).astype(np.float32)
                new_origin = np.zeros(3, dtype=int)
                affine = nbp_register.icp_correction[tile, r, c].copy()
                affine = preprocessing.adjust_affine(affine=affine, new_origin=new_origin)
                futures.append(
                    executor.submit(scipy.ndimage.affine_transform, im_c, affine, order=1, mode="constant", cval=0)
                )
            for future in futures:
                image_r += (future.result(timeout=300)[np.newaxis].copy(),)
            del futures
        image_r = torch.asarray(np.concatenate(image_r, axis=0, dtype=np.float32))
        image_batch = torch.cat((image_batch, image_r[np.newaxis]), dim=0)
        del image_r
        # Flow_field_r[0] are y shifts, flow_field_r[2] are z shifts.
        flow_field_tr = nbp_register.flow[tile, r].astype(np.float32)
        flow_field_tr[[0, 1, 2]] = flow_field_tr[[2, 1, 0]]
        # Flow's shape changes (3, im_y, im_x, im_z) -> (1, im_y, im_x, im_z, 3).
        flow_field_tr = torch.asarray(flow_field_tr.transpose((1, 2, 3, 0)))[np.newaxis]
        # A one in the flow field represents a shift of one pixel on the grid.
        flow_field_tr[..., 0] *= half_pixel_2 * 2
        flow_field_tr[..., 1] *= half_pixel_1 * 2
        flow_field_tr[..., 2] *= half_pixel_0 * 2
        flow_field_tr = base_grid.detach().clone() + flow_field_tr
        grids = torch.cat((grids, flow_field_tr), dim=0)
        del flow_field_tr
        if (i + 1) % n_rounds_batch == 0 or r == final_round:
            i_min, i_max = max(i + 1 - grids.shape[0], 0), i + 1
            registered_image_batch = torch.nn.functional.grid_sample(
                image_batch, grids, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            registered_image_batch = registered_image_batch.numpy().astype(dtype).transpose((2, 3, 4, 0, 1))
            colours[:, :, :, i_min:i_max, :] = registered_image_batch

            image_batch = image_batch[[]]
            grids = grids[[]]

    return colours
