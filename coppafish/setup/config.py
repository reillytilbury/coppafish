# Load config files for the Python port of the coppafish pipeline.

# There are three main features of this file:
# 1. Load config from .ini files
# 2. Also load from a "default" .ini file
# 3. Perform validation of the files and the assigned values.

# Config will be available as the result of the function "get_config" in the
# form of a dictionary.  This is, by default, the "Config" variable defined in
# this file.  Since it is a dictionary, access elements of the configuration
# using the subscript operator, i.e. square brackets or item access.  E.g.,
# Config['section']['item'].

# Config files should be considered read-only.

# To add new configuration options, do the following:

# 1. Add it to the "_options" dictionary, defined below.  The name of the
#    configuration option should be the key, and the value should be the
#    "type".  (The types are denied below in the "_option_type_checkers" and
#    "_option_formatters" dictionaries.)
# 2. Add it, and a description of what it does, to "config.default.ini".
import configparser
import os
import re
from typing import Any, Dict

from .. import log

try:
    import importlib_resources
except ModuleNotFoundError:
    import importlib.resources as importlib_resources


# List of options and their type.  If you change this, update the
# config.default.ini file too.  Make sure the type is valid.
_options = {
    "basic_info": {
        "use_tiles": "maybe_tuple_int",
        "use_rounds": "maybe_tuple_int",
        "use_channels": "maybe_tuple_int",
        "use_z": "maybe_tuple_int",
        "use_dyes": "maybe_tuple_int",
        "use_anchor": "bool",
        "anchor_round": "maybe_int",
        "anchor_channel": "maybe_int",
        "dapi_channel": "maybe_int",
        "tile_pixel_value_shift": "int",
        "dye_names": "tuple_str",
        "is_3d": "bool",
        "ignore_first_z_plane": "bool",
        "minimum_print_severity": "int",
        "bad_trc": "maybe_tuple_tuple_int",
        # From here onwards these are not compulsory to enter and will be taken from the metadata
        # Only leaving them here to have backwards compatibility as Max thinks the user should influence these
        "channel_camera": "maybe_tuple_int",
        "channel_laser": "maybe_tuple_int",
        "ref_round": "maybe_int",
        "ref_channel": "maybe_int",
        "sender_email": "maybe_str",
        "sender_email_password": "maybe_str",
        "email_me": "maybe_str",
    },
    "file_names": {
        "notebook_name": "str",
        "input_dir": "str",  # all these directories used to be of type 'dir' but you may want to load the notebook
        "output_dir": "str",  # while not being connected to server where data is
        "tile_dir": "str",
        "round": "maybe_tuple_str",  #
        "anchor": "maybe_str",
        "raw_extension": "str",
        "raw_metadata": "maybe_str",
        "dye_camera_laser": "maybe_file",
        "code_book": "str",
        "scale": "str",
        "psf": "maybe_str",
        "pciseq": "tuple_str",
        "fluorescent_bead_path": "maybe_str",
        "initial_bleed_matrix": "maybe_str",
        "log_name": "str",
    },
    "extract": {
        "num_rotations": "int",
        "z_plane_mean_warning": "number",
    },
    "filter": {
        "r_dapi": "maybe_int",
        "r_dapi_auto_microns": "maybe_number",
        "auto_thresh_multiplier": "number",
        "deconvolve": "bool",
        "wiener_constant": "number",
        "wiener_pad_shape": "tuple_int",
    },
    "find_spots": {
        "radius_xy": "int",
        "radius_z": "int",
        "max_spots_2d": "int",
        "max_spots_3d": "int",
        "isolation_radius_inner": "number",
        "isolation_radius_xy": "number",
        "isolation_radius_z": "number",
        "isolation_thresh": "maybe_number",
        "auto_isolation_thresh_multiplier": "number",
        "n_spots_warn_fraction": "number",
        "n_spots_error_fraction": "number",
    },
    "stitch": {
        "expected_overlap": "number",
    },
    "register": {
        # this parameter is for channel registration
        "bead_radii": "maybe_tuple_number",
        # these parameters are for round registration
        "sample_factor_yx": "int",
        "window_radius": "int",
        "smooth_sigma": "tuple_number",
        "smooth_thresh": "number",
        "flow_cores": "maybe_int",
        "flow_clip": "maybe_tuple_number",
        # these parameters are for icp
        "neighb_dist_thresh_yx": "number",
        "neighb_dist_thresh_z": "maybe_number",
        "icp_min_spots": "int",
        "icp_max_iter": "int",
    },
    "call_spots": {
        "gene_prob_threshold": "number",
        "target_values": "tuple_number",
        "concentration_parameter_parallel": "number",
        "concentration_parameter_perpendicular": "number",
    },
    "omp": {
        "colour_normalise": "bool",
        "fit_background": "bool",
        "weight_coef_fit": "bool",
        "max_genes": "int",
        "dp_thresh": "number",
        "alpha": "number",
        "beta": "number",
        "subset_pixels": "int",
        "force_cpu": "bool",
        "radius_xy": "int",
        "radius_z": "int",
        "spot_shape": "tuple_int",
        "spot_shape_max_spots": "int",
        "shape_isolation_distance_yx": "int",
        "shape_isolation_distance_z": "maybe_int",
        "shape_coefficient_threshold": "number",
        "shape_sign_thresh": "number",
        "pixel_max_percentile": "number",
        "high_coef_bias": "number",
        "score_threshold": "number",
    },
    "thresholds": {
        "intensity": "maybe_number",
        "score_ref": "number",
        "score_omp": "number",
        "score_prob": "number",
        "score_omp_multiplier": "number",
    },
    "reg_to_anchor_info": {
        "full_anchor_y0": "maybe_number",
        "full_anchor_x0": "maybe_number",
        "partial_anchor_y0": "maybe_number",
        "partial_anchor_x0": "maybe_number",
        "side_length": "maybe_number",
    },
}

# If you want to add a new option type, first add a type checker, which will
# only allow valid values to be passed.  Then, add a formatter.  Since the
# config file is strings only, the formatter converts from a string to the
# desired type.  E.g. for the "integer" type, it should be available as an
# integer.
#
# Any new type checkers created should keep in mind that the input is a string,
# and so validation must be done in string form.
#
# "maybe" types come from the Haskell convention whereby it can either hold a
# value or be empty, where empty in this case is defined as an empty string.
# In practice, this means the option is optional.
_option_type_checkers = {
    "int": lambda x: re.match("-?[0-9]+", x) is not None,
    "number": lambda _: re.match("-?[0-9]+(\\.[0-9]+)?$", "-123") is not None,
    "str": lambda x: len(x) > 0,
    "bool": lambda x: re.match("True|true|False|false", x) is not None,
    "file": lambda x: os.path.isfile(x),
    "dir": lambda x: os.path.isdir(x),
    "tuple": lambda _: True,
    "tuple_int": lambda x: all([_option_type_checkers["int"](s.strip()) for s in x.split(",")]),
    "tuple_number": lambda x: all([_option_type_checkers["number"](s.strip()) for s in x.split(",")]),
    "tuple_str": lambda x: all([_option_type_checkers["str"](s.strip()) for s in x.split(",")]),
    "maybe_int": lambda x: x.strip() == "" or _option_type_checkers["int"](x),
    "maybe_number": lambda x: x.strip() == "" or _option_type_checkers["number"](x),
    "maybe_tuple_int": lambda x: x.strip() == "" or _option_type_checkers["tuple_int"](x),
    "maybe_tuple_number": lambda x: x.strip() == "" or _option_type_checkers["tuple_number"](x),
    "maybe_str": lambda x: x.strip() == "" or _option_type_checkers["str"](x),
    "maybe_tuple_str": lambda x: x.strip() == "" or _option_type_checkers["tuple_str"](x),
    "maybe_file": lambda x: x.strip() == "" or _option_type_checkers["file"](x),
    "maybe_tuple_tuple_int": lambda x: x.strip() == "" or all([_option_type_checkers["tuple_int"](y) for y in x]),
}
_option_formatters = {
    "int": lambda x: int(x),
    "number": lambda x: float(x),
    "str": lambda x: x,
    "bool": lambda x: True if "rue" in x else False,
    "file": lambda x: x,
    "dir": lambda x: x,
    "tuple": lambda x: tuple([s.strip() for s in x.split(",")]),
    "tuple_int": lambda x: tuple([_option_formatters["int"](s.strip()) for s in x.split(",")]),
    "tuple_number": lambda x: tuple([_option_formatters["number"](s.strip()) for s in x.split(",")]),
    "tuple_str": lambda x: tuple([_option_formatters["str"](s.strip()) for s in x.split(",")]),
    "maybe_int": lambda x: None if x == "" else _option_formatters["int"](x),
    "maybe_number": lambda x: None if x == "" else _option_formatters["number"](x),
    "maybe_tuple_int": lambda x: None if x == "" else _option_formatters["tuple_int"](x),
    "maybe_tuple_number": lambda x: None if x == "" else _option_formatters["tuple_number"](x),
    "maybe_str": lambda x: None if x == "" else _option_formatters["str"](x),
    "maybe_tuple_str": lambda x: None if x == "" else _option_formatters["tuple_str"](x),
    "maybe_file": lambda x: None if x == "" else _option_formatters["file"](x),
    "maybe_tuple_tuple_int": lambda x: (
        None if x == "" else tuple([tuple(_option_formatters["tuple_int"](y)) for y in x])
    ),
}


# Standard formatting for errors in the config file
class InvalidConfigError(Exception):
    """Exception for an invalid configuration item"""

    def __init__(self, section, name, val):
        if val is None:
            val = ""
        if name is None:
            if section in _options.keys():
                error = f"Error in config file: Section {section} must be included in config file"
            else:
                error = f"Error in config file: {section} is not a valid section"
        else:
            if name in _options[section].keys():
                error = (
                    f"Error in config file: {name} in section {section} must be a {_options[section][name]},"
                    f" but the current value {val!r} is not."
                )
            else:
                error = (
                    f"Error in config file: {name} in section {section} is not a valid configuration key,"
                    f" and should not exist in the config file. (It is currently set to value {val!r}.)"
                )
        super().__init__(error)


def get_config(ini_file) -> Dict[str, Any]:
    """Return the configuration as a dictionary"""
    if not os.path.isfile(ini_file):
        raise FileNotFoundError(f"Failed to find config file at {ini_file}")

    # Read the settings files, overwriting the default settings with any settings
    # in the user-editable settings file.  We use .ini files without sections, and
    # add the section (named "config") manually.
    _parser = configparser.ConfigParser()
    _parser.optionxform = str  # Make names case-sensitive
    ini_file_default = str(importlib_resources.files("coppafish.setup").joinpath("settings.default.ini"))
    with open(ini_file_default, "r") as f:
        _parser.read_string(f.read())
    # Try to autodetect whether the user has passed a config file or the full
    # text of the config file.  The easy way would be to use os.path.isfile to
    # check if it is a file, and if not, assume it is text.  However, this
    # could lead to confusing error messages.  Instead, we will use the
    # following procedure.  If the string contains only whitespace, assume it
    # is full text.  If it doesn't have a newline or an equal sign, assume it
    # is a path.
    if ini_file.strip() != "" and "=" not in ini_file and "\n" not in ini_file:
        with open(ini_file, "r") as f:
            _parser.read_string(f.read())
    else:
        _parser.read_string(ini_file)

    # Validate configuration.
    # First step: ensure two things...
    # 1. ensure all of the sections (defined in _options) included
    for section in _options.keys():
        if section not in _parser.keys():
            log.error(InvalidConfigError(section, None, None))
    # 2. ensure all of the options in each section (defined in
    # _options) have some value.
    for section in _options.keys():
        for name in _options[section].keys():
            if name not in _parser[section].keys():
                log.error(InvalidConfigError(section, name, None))
    # Second step of validation: ensure three things...
    ini_file_sections = list(_parser.keys())
    ini_file_sections.remove("DEFAULT")  # parser always contains this key.
    # 1. Ensure there are no extra sections in config file
    for section in ini_file_sections:
        if section not in _options.keys():
            log.error(InvalidConfigError(section, None, None))
    for section in _options.keys():
        for name, val in _parser[section].items():
            # 2. Ensure there are no extra options in config file, else remove them
            if name not in _options[section].keys():
                _parser[section].__delitem__(name)
                continue
            # 3. Ensure that all the option values pass type checking.
            if not _option_type_checkers[_options[section][name]](val):
                log.error(InvalidConfigError(section, name, val))

    # Now that we have validated, build the configuration dictionary
    out_dict = {section: {} for section in _options.keys()}
    for section in _options.keys():
        for name, val in _parser[section].items():
            out_dict[section][name] = _option_formatters[_options[section][name]](_parser[section][name])
    return out_dict
