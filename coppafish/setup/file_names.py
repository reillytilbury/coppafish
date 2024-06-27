import os

from .. import log
from .. import setup
from .. import utils
from ..setup import Notebook
from ..setup import NotebookPage
from .tile_details import get_tile_file_names

try:
    import importlib_resources
except ModuleNotFoundError:
    import importlib.resources as importlib_resources


def get_file_names(nb: Notebook):
    """
    Function to set add `file_names` page to notebook. It requires notebook to be able to access a
    config file containing a `file_names` section and also the notebook to contain a `basic_info` page.

    !!! note
        This will be called every time the notebook is loaded to deal will case when `file_names` section of
        config file changed.

    Args:
        nb: *Notebook* containing at least the `basic_info` page.
    """
    config = setup.config.get_config(nb.config_path)["file_names"]
    nbp = NotebookPage("file_names")
    # Copy some variables that are in config to page.
    nbp.input_dir = config["input_dir"]
    nbp.output_dir = config["output_dir"]
    nbp.tile_dir = os.path.join(config["tile_dir"], "filter")
    nbp.tile_unfiltered_dir = os.path.join(config["tile_dir"], "extract")
    nbp.fluorescent_bead_path = config["fluorescent_bead_path"]

    # remove file extension from round and anchor file names if it is present
    if config["raw_extension"] == "jobs":

        if bool(config["pre_seq"]):
            all_files = os.listdir(config["input_dir"])
            all_files.sort()  # Sort files by ascending number
            n_tiles = int(len(all_files) / 7 / 9)
            config["pre_seq"] = [r.replace(".nd2", "") for r in all_files[: n_tiles * 7]]
            config["round"] = tuple(
                [
                    [f.replace(".nd2", "") for f in all_files[n_tiles * r * 7 : n_tiles * (r + 1) * 7]]
                    for r in range(1, 8)
                ]
            )
            # TODO replace range(7) by the by the number of rounds?
            config["anchor"] = tuple([r.replace(".nd2", "") for r in all_files[n_tiles * 8 * 7 :]])

        else:
            all_files = os.listdir(config["input_dir"])
            all_files.sort()  # Sort files by ascending number
            n_tiles = int(len(all_files) / 7 / 8)
            # FIXME: r is not defined within the scope of the square brackets, this will probably cause a runtime error
            config["round"] = tuple(
                [f.replace(".nd2", "") for f in all_files[n_tiles * r * 7 : n_tiles * (r + 1) * 7] for r in range(7)]
            )
            # TODO replace range(7) by the by the number of rounds?
            config["anchor"] = tuple([r.replace(".nd2", "") for r in all_files[n_tiles * 7 * 7 :]])

    else:
        if config["round"] is None:
            if config["anchor"] is None:
                log.error(ValueError(f"Neither imaging rounds nor anchor_round provided"))
            config["round"] = tuple()  # Sometimes the case where just want to run the anchor round.
        config["round"] = tuple([r.replace(config["raw_extension"], "") for r in config["round"]])

        if config["anchor"] is not None:
            config["anchor"] = config["anchor"].replace(config["raw_extension"], "")

    nbp.round = config["round"]
    nbp.anchor = config["anchor"]
    nbp.pre_seq = config["pre_seq"]
    nbp.raw_extension = config["raw_extension"]
    nbp.raw_metadata = config["raw_metadata"]
    nbp.initial_bleed_matrix = config["initial_bleed_matrix"]

    if nbp.initial_bleed_matrix is not None:
        assert os.path.isfile(
            nbp.initial_bleed_matrix
        ), f"Initial bleed matrix located at {nbp.initial_bleed_matrix} does not exist"

    if config["dye_camera_laser"] is None:
        # Default information is project
        config["dye_camera_laser"] = str(
            importlib_resources.files("coppafish.setup").joinpath("dye_camera_laser_raw_intensity.csv")
        )
    nbp.dye_camera_laser = config["dye_camera_laser"]

    if config["code_book"] is not None:
        config["code_book"] = config["code_book"].replace(".txt", "")
        nbp.code_book = config["code_book"] + ".txt"
    else:
        # If the user has not put their code_book in, default to the one included in this project
        config["code_book"] = os.path.join(os.getcwd(), "coppafish/setup/code_book_73g.txt")

    # where to save scale and scale_anchor values used in extract step.
    config["scale"] = config["scale"].replace(".txt", "")
    nbp.scale = os.path.join(config["tile_dir"], config["scale"] + ".txt")

    if config["psf"] is None:
        config["psf"] = str(importlib_resources.files("coppafish.setup").joinpath("default_psf.npz"))
    nbp.psf = config["psf"]

    # Add files so save plotting information for pciseq
    config["pciseq"] = tuple([val.replace(".csv", "") for val in config["pciseq"]])
    nbp.pciseq = tuple([os.path.join(config["output_dir"], val + ".csv") for val in config["pciseq"]])

    if config["anchor"] is not None:
        round_files = config["round"] + (config["anchor"],)
    else:
        round_files = config["round"]

    if config["pre_seq"] is not None:
        round_files = round_files + (config["pre_seq"],)

    if config["raw_extension"] == "jobs":
        if nb.basic_info.is_3d:
            round_files = config["round"] + [config["anchor"]] + [config["pre_seq"]]
            tile_names, tile_names_unfiltered = get_tile_file_names(
                nbp.tile_dir,
                nbp.tile_unfiltered_dir,
                round_files,
                nb.basic_info.n_tiles,
                nb.get_config()["extract"]["file_type"],
                nb.basic_info.n_channels,
                jobs=True,
            )
        else:
            log.error(ValueError("JOBs file format is only compatible with 3D"))
    else:
        if nb.basic_info.is_3d:
            tile_names, tile_names_unfiltered = get_tile_file_names(
                nbp.tile_dir,
                nbp.tile_unfiltered_dir,
                round_files,
                nb.basic_info.n_tiles,
                setup.config.get_config(nb.config_path)["extract"]["file_type"],
                nb.basic_info.n_channels,
            )
        else:
            tile_names, tile_names_unfiltered = get_tile_file_names(
                nbp.tile_dir,
                nbp.tile_unfiltered_dir,
                round_files,
                nb.basic_info.n_tiles,
                setup.config.get_config(nb.config_path)["extract"]["file_type"],
            )

    nbp.tile = utils.base.deep_convert(tile_names.tolist())
    nbp.tile_unfiltered = utils.base.deep_convert(tile_names_unfiltered.tolist())
    return nbp
