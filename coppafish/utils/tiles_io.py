import enum
import numbers
import os
from typing import Any, List, Optional, Tuple, Union

from numcodecs import Blosc, blosc
import numpy as np
import numpy.typing as npt
import numpy_indexed
import zarr

from .. import log, utils
from ..setup import NotebookPage


EXTRACT_IMAGE_DTYPE = np.uint16
FILTER_IMAGE_DTYPE = np.float16


class OptimisedFor(enum.Enum):
    FULL_READ_AND_WRITE = enum.auto()
    Z_PLANE_READ = enum.auto()


def add_suffix_to_path(file_path: str, suffix: str) -> str:
    """
    Add the suffix string to the given file path by placing the suffix before the last '.' character. For example
    "hi.exe" with suffix "_there" will become "hi_there.exe".

    Args:
        - file_path (str): the file path, must contain at least one '.' character.
        - suffix (str): the suffix to add.
    """
    assert type(file_path) is str
    assert "." in file_path
    assert type(suffix) is str

    file_type_start = len(file_path) - file_path[::-1].index(".") - 1
    file_path = file_path[:file_type_start] + suffix + file_path[file_type_start:]
    return file_path


def get_compressor_and_chunks(
    optimised_for: OptimisedFor, image_shape: Tuple[int], image_z_index: Optional[int] = None
) -> Tuple[Any, Tuple[int]]:
    # By benchmarking every single blosc algorithm type (except snappy since this is unsupported by zarr),
    # chunk size and compression level, it was found that zstd, chunk size of 1x2x2 and compression level 3 was
    # fast at the sum of full image reading + writing times while still compressing the files to ~70-80%.
    # Benchmarking done by Paul Shuker (paul.shuker@outlook.com), January 2024.
    if image_z_index is None:
        image_z_index = np.argmin(image_shape).item()
    if optimised_for == OptimisedFor.FULL_READ_AND_WRITE:
        compressor = Blosc(cname="lz4", clevel=2, shuffle=Blosc.BITSHUFFLE)
        chunk_size_z = 1
        chunk_size_yx = 288
    elif optimised_for == OptimisedFor.Z_PLANE_READ:
        compressor = Blosc(cname="lz4", clevel=4, shuffle=Blosc.SHUFFLE)
        chunk_size_z = 1
        chunk_size_yx = 576
    else:
        raise ValueError(f"Unknown OptimisedFor value of {optimised_for}")
    if len(image_shape) >= 3:
        chunks = tuple()
        for i in range(len(image_shape)):
            if image_shape[i] < 10:
                chunks += (1,)
            elif i == image_z_index:
                chunks += (chunk_size_z,)
            else:
                chunks += (chunk_size_yx,)
    elif len(image_shape) == 2:
        chunks = (chunk_size_yx, chunk_size_yx)
    else:
        raise ValueError(f"Got image_shape with {len(image_shape)} dimensions: {image_shape}")
    return compressor, chunks


def image_exists(file_path: str) -> bool:
    """
    Checks if an image exists at the given path location.

    Args:
        file_path (str): tile path.

    Returns:
        bool: tile existence.
    """
    # Require a non-empty zarr directory
    return os.path.isdir(file_path) and len(os.listdir(file_path)) > 0


def _save_image(
    image: npt.NDArray[Union[np.float16, np.uint16]],
    file_path: str,
    optimised_for: OptimisedFor = None,
) -> None:
    """
    Save extract image in `file_path` location. No manipulation or logic is applied here, just purely saving the image
    as it was given.

    Args:
        image (`(nz x ny x nx) ndarray[uint16]`): extract image to save.
        file_path (str): file path.
        optimised_for (literal[int]): what speed to optimise compression for. Affects the blosc compressor. Default:
            optimised for full image reading + writing time.
    """
    if image.dtype != FILTER_IMAGE_DTYPE and image.dtype != EXTRACT_IMAGE_DTYPE:
        raise ValueError(f"Expected image dtype {FILTER_IMAGE_DTYPE}, got {image.dtype}")
    if optimised_for is None:
        optimised_for = OptimisedFor.FULL_READ_AND_WRITE

    blosc.use_threads = True
    blosc.set_nthreads(utils.system.get_core_count())
    compressor, chunks = get_compressor_and_chunks(optimised_for, image.shape)
    zarray = zarr.open(
        store=file_path,
        shape=image.shape,
        mode="w",
        zarr_version=2,
        chunks=chunks,
        dtype=image.dtype,
        compressor=compressor,
    )
    zarray[:] = image


def _load_image(file_path: str) -> zarr.Array:
    """
    Read in extract zarr array from file_path location.

    Args:
        file_path (str): image location.

    Returns `(im_y x im_x x im_z) zarray[uint16]`: loaded extract image.
    """
    return zarr.open(file_path, mode="r")


def get_npy_tile_ind(
    tile_ind_nd2: Union[int, List[int]], tile_pos_yx_nd2: np.ndarray, tile_pos_yx_npy: np.ndarray
) -> Union[int, List[int]]:
    """
    Gets index of tile in npy file from tile index of nd2 file.

    Args:
        tile_ind_nd2: Index of tile in nd2 file
        tile_pos_yx_nd2: ```int [n_tiles x 2]```.
            ```[i,:]``` contains YX position of tile with nd2 index ```i```.
            Index 0 refers to ```YX = [0, 0]```.
            Index 1 refers to ```YX = [0, 1] if MaxX > 0```.
        tile_pos_yx_npy: ```int [n_tiles x 2]```.
            ```[i,:]``` contains YX position of tile with npy index ```i```.
            Index 0 refers to ```YX = [MaxY, MaxX]```.
            Index 1 refers to ```YX = [MaxY, MaxX - 1] if MaxX > 0```.

    Returns:
        Corresponding indices in npy file.
    """
    if isinstance(tile_ind_nd2, numbers.Number):
        tile_ind_nd2 = [tile_ind_nd2]
    npy_index = numpy_indexed.indices(tile_pos_yx_npy, tile_pos_yx_nd2[tile_ind_nd2]).tolist()
    if len(npy_index) == 1:
        return npy_index[0]
    else:
        return npy_index
