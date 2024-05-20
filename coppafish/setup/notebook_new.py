import os
import time


class Notebook:
    _page_options = {
        "basic_info": [
            "*basic_info* page contains information that is used at all stages of the pipeline.",
        ],
        "file_names": [
            "*file_names* page contains all files that are used throughout the pipeline.",
        ],
        "extract": [
            "*extract* page contains information related to extraction of raw input files for use in coppafish."
        ],
        "filter": [
            "*filter* page contains information on image filtering applied to extracted images.",
        ],
        "filter_debug": [
            "*filter_debug* page contains additional information on filtering that is not used later in ",
            "the pipeline.",
        ],
        "find_spots": [
            "*find_spots* page contains information about spots found on every tiles, rounds and channels.",
        ],
        "stitch": [
            "*stitch* page contains information about how tiles were stitched together to produce global ",
            "coordinates.",
        ],
        "register": [
            "*register* page contains best found solutions to allign images.",
        ],
        "register_debug": [
            "*register_debug* page contains information on how the image allignments in the *register* ",
            "page were calculated.",
        ],
        "ref_spots": [
            "*ref_spots* page contains gene assignments and info for spots found on reference round.",
        ],
        "call_spots": [
            "*call_spots* page contains `bleed_matrix` and expected code for each gene.",
        ],
        "omp": [
            "*omp* page contains gene assigments and information for spots found through Orthogonal Matching Pursuit."
        ],
        "thresholds": [
            "*thresholds* page contains quality thresholds which affect which spots plotted and exported to ",
            "pciSeq.",
        ],
    }
