## Input data

Coppafish requires raw, `uint16` microscope images, metadata, and a configuration file. We currently only support raw 
data in ND2, JOBs, or numpy format. If your data is not already in one of these formats, we recommend configuring your 
data into numpy format (see below).

### Numpy

Each round is separated between directories. Label sequencing round directories `0`, `1`, etc. We recommend using 
[dask](https://docs.dask.org), this is installed in your coppafish environment by default. The code to save data in the 
right format would look something like

```python
import os
import dask.array

raw_path = "/path/to/raw/data"
dask_chunks = (1, n_total_channels, n_y, n_x, n_z)
for r in range(n_seq_rounds):
    save_path = os.path.join(raw_path, f"{r}")
    image_dask = dask.array.from_array(seq_image_tiles[r], chunks=dask_chunks)
    dask.array.to_npy_stack(save_path, image_dask)

# Anchor round
save_path = os.path.join(raw_path, "anchor")
image_dask = dask.array.from_array(anchor_image, chunks=dask_chunks)
dask.array.to_npy_stack(save_path, image_dask)

# Presequence round (optional)
save_path = os.path.join(raw_path, "presequence")
image_dask = dask.array.from_array(preseq_image, chunks=dask_chunks)
dask.array.to_npy_stack(save_path, image_dask)
```

where `n_...` variables represent counts (integers), `n_total_channels` can include other channels other than the 
sequencing channel (e.g. a DAPI channel and anchor channel). `seq_image_tiles` is a numpy array of shape 
`(n_seq_rounds, n_tiles, n_total_channels, n_y, n_x, n_z)`, while `anchor_image` and `preseq_image` are numpy arrays of 
shape `(n_tiles, n_total_channels, n_y, n_x, n_z)`. Note that `n_y` must be equal to `n_x`.


### Metadata

The metadata can be saved using python:

```python
import json

metadata = {
    "n_tiles": n_tiles,
    "n_rounds": n_rounds,
    "n_channels": n_total_channels,
    "tile_sz": n_y, # or n_x
    "pixel_size_xy": 0.26,
    "pixel_size_z": 0.9,
    "tile_centre": [n_y / 2, n_x / 2, n_z / 2],
    "tilepos_yx": tile_origins_yx,
    "tilepos_yx_nd2": list(reversed(tile_origins_yx)),
    "channel_camera": [1] * n_total_channels,
    "channel_laser": [1] * n_total_channels,
    "xy_pos": tile_xy_pos,
    "nz": n_z,
}
file_path = os.path.join(raw_path, "metadata.json")
with open(file_path, "w") as f:
    json.dump(metadata, f, indent=4)
```

### Code book

A code book is a `.txt` file that tells coppafish the expected gene codes for each gene. An example of a four 
gene code book is
```
gene_0 0123012
gene_1 1230123
gene_2 2301230
gene_3 3012301
```
the names (`gene_0`, `gene_1`, ...) can be changed. Do not assign any genes a constant gene code, e.g. `0000000`. For 
more details on how the codes can be generated, see coppafish's gene code generator `reed_solomon_codes` in 
[`coppafish/utils/base.py`](https://github.com/reillytilbury/coppafish/blob/alpha/coppafish/utils/base.py). Also, see 
[wikipedia](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction) for how gene code selection is 
optimised.

### Configuration

There are configuration variables used throughout the coppafish pipeline. Most of these have reasonable default values, 
but some must be set by the user and you may wish to tweak other values for better performance. Save the config file as 
something like `config.ini`. The config file should contain, at the minimum:
```
[file_names]
input_dir = path/to/input/data
output_dir = path/to/output/directory
tile_dir = path/to/tile/output
round = 0, 1, 2, 3, 4, 5, 6 ; Go up to the number of sequencing rounds used
anchor = anchor
pre_seq = presequence
raw_extension = .npy
raw_metadata = path/to/metadata.json

[basic_info]
is_3d = True
dye_names = dye_0, dye_1, dye_2, dye_3
use_rounds = 0, 1, 2, 3, 4, 5, 6
use_z = 0, 1, 2, 3, 4
use_tiles = 0, 1
anchor_round = 7
use_channels = 1, 2, 3, 4
anchor_channel = 1
dapi_channel = 0

[stitch]
expected_overlap = 0.15
```
where the `dapi_channel` is the index in the numpy arrays that the dapi channel is stored at. `use_channels` includes 
the `anchor_channel` in this case because the anchor channel can also be used as a sequencing channel in the sequencing 
rounds. `dye_names` does not have to be set explicitly if `n_seq_channels == n_dyes`. `expected_overlap` is the 
fraction of the tile in x (y) dimension that is overlapping between adjacent tiles, typically `0.1-0.15`. More details 
about every config variable can be found at 
<a href="https://github.com/reillytilbury/coppafish/blob/alpha/coppafish/setup/settings.default.ini" target="_blank">
`coppafish/setup/settings.default.ini`</a> in the source code. `use_z` contains all selected z planes, they should all 
be adjacent planes. It is recommended to use microscopic images where the middle z plane is roughly the brightest for 
best performance; this can be configured by changing the selected z planes in `use_z`. The z direction can be treated 
differently to the y and x directions because typically a z pixel corresponds to a larger, real distance.

## Running

Coppafish can be run with a config file. In the terminal
```console
python -m coppafish /path/to/config.ini
```

Or using a python script
```python
from coppafish import run_pipeline

run_pipeline("/path/to/config.ini")
```
