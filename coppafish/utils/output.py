import itertools
import os
from pathlib import PurePath

from ..setup import Notebook

from . import base
from .. import log


def set_notebook_output_dir(notebook_path: str, new_output_dir: str) -> None:
    """
    Changes the notebook variables to use the given `output_dir`, then re-saves the notebook.

    Args:
        notebook_path (str): path to notebook.
        new_dir (str): new output directory.
    """
    assert os.path.isdir(notebook_path), f"Notebook at {notebook_path} not found"
    assert os.path.isdir(new_output_dir), f"{new_output_dir} output directory not found"

    nb = Notebook(notebook_path)

    old_name = PurePath(nb.file_names.psf).name
    if PurePath(nb.file_names.output_dir) in PurePath(nb.file_names.psf).parents:
        del nb.file_names.psf
        nb.file_names.psf = os.path.join(new_output_dir, old_name)

    # Set the copied notebook variables to the right output directory.
    del nb.file_names.output_dir
    nb.file_names.output_dir = new_output_dir

    nb.resave()


def set_notebook_tile_dir(notebook_path: str, new_tile_dir: str) -> None:
    """
    Changes the notebook variables to use the given `tile_dir`, then re-saves the notebook.

    Args:
        notebook_path (str): path to notebook. Can be relative.
        new_tile_dir (str): new tile directory.
    """
    assert os.path.isdir(notebook_path), f"Notebook at {notebook_path} not found"
    if not os.path.isdir(new_tile_dir):
        log.warn(f"New tile directory {new_tile_dir} does not exist. Continuing anyway.")

    new_tile_dir = os.path.normpath(new_tile_dir)

    nb = Notebook(notebook_path)

    del nb.file_names.tile_unfiltered_dir
    nb.file_names.tile_unfiltered_dir = os.path.join(new_tile_dir, "extract")
    old_tile_unfiltered = nb.file_names.tile_unfiltered
    new_tile_unfiltered = base.deep_convert(old_tile_unfiltered, list)
    for i, j, k in itertools.product(
        range(len(old_tile_unfiltered)), range(len(old_tile_unfiltered[0])), range(len(old_tile_unfiltered[0][0]))
    ):
        old_tile_unfiltered_ijk = os.path.normpath(old_tile_unfiltered[i][j][k])
        new_tile_unfiltered[i][j][k] = os.path.join(
            nb.file_names.tile_unfiltered_dir, PurePath(old_tile_unfiltered_ijk).name
        )
    del nb.file_names.tile_unfiltered
    nb.file_names.tile_unfiltered = base.deep_convert(new_tile_unfiltered, tuple)

    nb.resave()
