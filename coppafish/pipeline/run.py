import os
import sys
import numpy as np
from scipy import sparse

from .. import utils
from ..setup import Notebook
from ..find_spots import check_spots
from ..call_spots import base as call_spots_base
from ..pdf.base import BuildPDF
from .. import log
from . import basic_info
from . import extract_run
from . import filter_run
from . import find_spots
from . import register
from . import stitch
from . import get_reference_spots
from . import call_reference_spots
from . import omp


def run_pipeline(
    config_file: str,
    overwrite_ref_spots: bool = False,
) -> Notebook:
    """
    Bridge function to run every step of the pipeline.

    Args:
        config_file: Path to config file.
        overwrite_ref_spots: Only used if *Notebook* contains *ref_spots* but not *call_spots* page.
            If `True`, the variables:

            * `gene_no`
            * `score`
            * `score_diff`
            * `intensity`

            in `nb.ref_spots` will be overwritten if they exist. If this is `False`, they will only be overwritten
            if they are all set to `None`, otherwise an error will occur.

    Returns:
        Notebook: notebook containing all information gathered during the pipeline.
    """
    nb = initialize_nb(config_file)
    log.error_catch(run_tile_indep_pipeline, nb)
    log.error_catch(run_stitch, nb)
    log.error_catch(run_reference_spots, nb, overwrite_ref_spots)
    log.error_catch(BuildPDF, nb)
    log.error_catch(run_omp, nb)
    log.error_catch(BuildPDF, nb, auto_open=True)
    return nb


def run_tile_indep_pipeline(nb: Notebook) -> None:
    """
    Run tile-independent pipeline processes.

    Args:
        nb (Notebook): notebook containing 'basic_info' and 'file_names' pages.
        run_tile_by_tile (bool, optional): run each tile on a separate notebook through 'find_spots' and 'register',
            then merge them together. Default: true if PC has >110GB of available memory. False otherwise.
    """
    run_extract(nb)
    BuildPDF(nb)
    run_filter(nb)
    BuildPDF(nb)
    run_find_spots(nb)
    run_register(nb)
    BuildPDF(nb)
    check_spots.check_n_spots(nb)


def initialize_nb(config_file: str) -> Notebook:
    """
    Creates a `Notebook` and adds `basic_info` page before saving.
    `file_names` page will be added automatically as soon as `basic_info` page is added.
    If `Notebook` already exists and contains these pages, it will just be returned.

    Args:
        config_file: Path to config file.

    Returns:
        `Notebook` containing `file_names` and `basic_info` pages.
    """
    nb = Notebook(config_file=config_file)

    config = nb.get_config()
    config_file = config["file_names"]

    log.base.set_log_config(
        config["basic_info"]["minimum_print_severity"],
        os.path.join(config_file["output_dir"], config_file["log_name"]),
    )
    log.info(
        f" COPPAFISH v{utils.system.get_software_version()} ".center(utils.system.current_terminal_size_xy(-33)[0], "=")
    )

    if not nb.has_page("basic_info"):
        nbp_basic = basic_info.set_basic_info_new(config)
        nb += nbp_basic
    else:
        log.warn(utils.warnings.NotebookPageWarning("basic_info"))
    if utils.system.get_software_version() not in nb.get_all_variable_instances(nb._SOFTWARE_VERSION):
        log.warn(
            f"You are running on v{utils.system.get_software_version()}, but the notebook contains "
            + f"data from versions {nb.get_all_variable_instances(nb._SOFTWARE_VERSION)}.",
        )
        log.warn("Are you sure you want to continue? (automatically continuing in 60s)")
        user_input = utils.system.input_timeout("type y or n: ", timeout_result="y")
        if user_input.strip().lower() != "y":
            log.info("Exiting...")
            sys.exit()
    online_version = utils.system.get_remote_software_version()
    if online_version != utils.system.get_software_version():
        log.warn(
            f"You are running v{utils.system.get_software_version()}. The latest online version is v{online_version}"
        )
    return nb


def run_extract(nb: Notebook) -> None:
    """
    This runs the `extract_and_filter` step of the pipeline to produce the tiff files in the tile directory.

    `extract` and pages are added to the `Notebook` before saving.

    If `Notebook` already contains these pages, it will just be returned.

    Args:
        nb: `Notebook` containing `file_names`, `basic_info` and `scale` pages.

    Returns:
        `(n_rounds x n_channels x nz x ny x nx) ndarray[uint16]` or None: all extracted images if running on a single
            tile, otherwise None.
    """
    if not nb.has_page("extract"):
        config = nb.get_config()
        nbp = extract_run.run_extract(config["extract"], nb.file_names, nb.basic_info)
        nb += nbp
    else:
        log.warn(utils.warnings.NotebookPageWarning("extract"))


def run_filter(nb: Notebook) -> None:
    """
    Run `filter` step of the pipeline to produce filtered images in the tile directory.

    Args:
        nb (Notebook): `Notebook` containing `file_names`, `basic_info`, `scale` and `extract` pages.
    """
    if not nb.has_page("filter"):
        config = nb.get_config()
        nbp, nbp_debug = filter_run.run_filter(config["filter"], nb.file_names, nb.basic_info, nb.extract)
        nb += nbp
        nb += nbp_debug
    else:
        log.warn(utils.warnings.NotebookPageWarning("filter"))


def run_find_spots(nb: Notebook) -> Notebook:
    """
    This runs the `find_spots` step of the pipeline to produce point cloud from each tiff file in the tile directory.

    `find_spots` page added to the `Notebook` before saving if image_t is not given.

    If `Notebook` already contains this page, it will just be returned.

    Args:
        nb: `Notebook` containing `extract` page.

    Returns:
        NoteBook containing 'find_spots' page.
    """
    if not nb.has_page("find_spots"):
        config = nb.get_config()
        nbp = find_spots.find_spots(
            config["find_spots"],
            nb.file_names,
            nb.basic_info,
            nb.extract,
            nb.filter,
            nb.filter.auto_thresh,
        )
        nb += nbp
    else:
        log.warn(utils.warnings.NotebookPageWarning("find_spots"))
    return nb


def run_stitch(nb: Notebook) -> None:
    """
    This runs the `stitch` step of the pipeline to produce origin of each tile
    such that a global coordinate system can be built. Also saves stitched DAPI and reference channel images.

    `stitch` page added to the `Notebook` before saving.

    If `Notebook` already contains this page, it will just be returned.
    If stitched images already exist, they won't be created again.

    Args:
        nb: `Notebook` containing `find_spots` page.
    """
    config = nb.get_config()
    if not nb.has_page("stitch"):
        nbp_debug = stitch.stitch(config["stitch"], nb.basic_info, nb.find_spots.spot_yxz, nb.find_spots.spot_no)
        nb += nbp_debug
    else:
        log.warn(utils.warnings.NotebookPageWarning("stitch"))
    # Two conditions below:
    # 1. Check if there is a big dapi_image
    # 2. Check if there is NOT a file in the path directory for the dapi image
    # if nb.file_names.big_dapi_image is not None and not os.path.isfile(nb.file_names.big_dapi_image):
    #     # save stitched dapi
    #     # Will load in from nd2 file if nb.filter_debug.r_dapi is None i.e. if no DAPI filtering performed.
    #     utils.tiles_io.save_stitched(
    #         nb.file_names.big_dapi_image,
    #         nb.file_names,
    #         nb.basic_info,
    #         nb.extract,
    #         nb.stitch.tile_origin,
    #         nb.basic_info.anchor_round,
    #         nb.basic_info.dapi_channel,
    #         nb.filter_debug.r_dapi is None,
    #         config["stitch"]["save_image_zero_thresh"],
    #         config["filter"]["num_rotations"],
    #     )
    #
    # if nb.file_names.big_anchor_image is not None and not os.path.isfile(nb.file_names.big_anchor_image):
    #     # save stitched reference round/channel
    #     utils.tiles_io.save_stitched(
    #         nb.file_names.big_anchor_image,
    #         nb.file_names,
    #         nb.basic_info,
    #         nb.extract,
    #         nb.stitch.tile_origin,
    #         nb.basic_info.anchor_round,
    #         nb.basic_info.anchor_channel,
    #         False,
    #         config["stitch"]["save_image_zero_thresh"],
    #         config["filter"]["num_rotations"],
    #     )


def run_register(nb: Notebook) -> None:
    """
    This runs the `register_initial` step of the pipeline to find shift between ref round/channel to each imaging round
    for each tile. It then runs the `register` step of the pipeline which uses this as a starting point to get
    the affine transforms to go from the ref round/channel to each imaging round/channel for every tile.

    `register_initial`, `register` and `register_debug` pages are added to the `Notebook` before saving.

    If `Notebook` already contains these pages, it will just be returned.

    Args:
        nb: `Notebook` containing `extract` page.
    """
    config = nb.get_config()
    # if not all(nb.has_page(["register", "register_debug"])):
    if not nb.has_page("register"):
        nbp, nbp_debug = register.register(
            nb.basic_info,
            nb.file_names,
            nb.extract,
            nb.filter,
            nb.find_spots,
            config["register"],
            pre_seq_blur_radius=None,
        )
        nb += nbp
        nb += nbp_debug
        register.preprocessing.generate_reg_images(nb)
    else:
        log.warn(utils.warnings.NotebookPageWarning("register"))
        log.warn(utils.warnings.NotebookPageWarning("register_debug"))


def run_reference_spots(nb: Notebook, overwrite_ref_spots: bool = False) -> None:
    """
    This runs the `reference_spots` step of the pipeline to get the intensity of each spot on the reference
    round/channel in each imaging round/channel. The `call_spots` step of the pipeline is then run to produce the
    `bleed_matrix`, `bled_code` for each gene and the gene assignments of the spots on the reference round.

    `ref_spots` and `call_spots` pages are added to the Notebook before saving.

    If `Notebook` already contains these pages, it will just be returned.

    Args:
        nb: `Notebook` containing `stitch` and `register` pages.
        overwrite_ref_spots: Only used if *Notebook* contains *ref_spots* but not *call_spots* page.
            If `True`, the variables:

            * `gene_no`
            * `score`
            * `score_diff`
            * `intensity`

            in `nb.ref_spots` will be overwritten if they exist. If this is `False`, they will only be overwritten
            if they are all set to `None`, otherwise an error will occur.
    """
    if not nb.has_page("ref_spots"):
        nbp = get_reference_spots.get_reference_spots(
            nb.file_names,
            nb.basic_info,
            nb.find_spots,
            nb.extract,
            nb.filter,
            nb.stitch.tile_origin,
            nb.register.icp_correction,
        )
        nb += nbp  # save to Notebook with gene_no, score, score_diff, intensity = None.
        # These will be added in call_reference_spots
    else:
        log.warn(utils.warnings.NotebookPageWarning("ref_spots"))
    if not nb.has_page("call_spots"):
        config = nb.get_config()
        nbp, nbp_ref_spots = call_reference_spots.call_reference_spots(
            config["call_spots"],
            nb.file_names,
            nb.basic_info,
            nb.ref_spots,
            nb.extract,
            nb.filter,
            transform=nb.register.icp_correction,
            overwrite_ref_spots=overwrite_ref_spots,
        )
        nb += nbp
    else:
        log.warn(utils.warnings.NotebookPageWarning("call_spots"))


def run_omp(nb: Notebook) -> None:
    """
    This runs the orthogonal matching pursuit section of the pipeline as an alternate method to determine location of
    spots and their gene identity.
    It achieves this by fitting multiple gene bled codes to each pixel to find a coefficient for every gene at
    every pixel. Spots are then local maxima in these gene coefficient images.

    `omp` page is added to the Notebook before saving.

    Args:
        nb: `Notebook` containing `call_spots` page.
    """
    if not nb.has_page("omp"):
        config = nb.get_config()
        # Use tile with most spots on to find spot shape in omp
        spots_tile = np.sum(nb.find_spots.spot_no, axis=(1, 2))
        tile_most_spots = nb.basic_info.use_tiles[np.argmax(spots_tile[nb.basic_info.use_tiles])]
        nbp = omp.run_omp(
            config["omp"],
            nb.file_names,
            nb.basic_info,
            nb.extract,
            nb.filter,
            nb.register,
            nb.register_debug,
            nb.call_spots,
            nb.stitch.tile_origin,
            nb.register.icp_correction,
        )
        nb += nbp

        # Update omp_info files after omp notebook page saved into notebook
        # Save only non-duplicates - important spot_coefs saved first for exception at start of call_spots_omp
        # which can deal with case where duplicates removed from spot_coefs but not spot_info.
        # After re-saving here, spot_coefs[s] should be the coefficients for gene at nb.omp.local_yxz[s]
        # i.e. indices should match up.
        # spot_info = np.load(nb.file_names.omp_spot_info)
        # not_duplicate = call_spots_base.get_non_duplicate(
        #     nb.stitch.tile_origin, nb.basic_info.use_tiles, nb.basic_info.tile_centre, spot_info[:, :3], spot_info[:, 5]
        # )
        # spot_coefs = sparse.load_npz(nb.file_names.omp_spot_coef)
        # sparse.save_npz(nb.file_names.omp_spot_coef, spot_coefs[not_duplicate])
        # np.save(nb.file_names.omp_spot_info, spot_info[not_duplicate])

        # only raise error after saving to notebook if spot_colors have nan in wrong places.
        # utils.errors.check_color_nan(nbp.colors, nb.basic_info)
    else:
        log.warn(utils.warnings.NotebookPageWarning("omp"))
