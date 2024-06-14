## Pipeline crash

If the coppafish pipeline is crashing, first read the error message. If there is a suggestion about how to fix the
issue in the config, try changing the config variable and run the pipeline again. If the suggestion does not make sense
to you, feel free to reach out to the developers for help or 
[create an issue](https://github.com/reillytilbury/coppafish/issues/new) on GitHub!

## Notebook will not open

A notebook file can be corrupted if a process is killed while the notebook is being re-saved. When this happens, an
error like:

``` bash
TypeError: byte indices must be integers or slices, not tuple
```

will occur when trying to load the notebook. To fix this issue, delete the corrupted notebook, rename the backup
notebook called `notebook_backup.npz` to the original notebook name and continue from there.

## Cannot open napari issues

If napari fails to open and you see an error such as

``` bash
WARNING: composeAndFlush: makeCurrent() failed
```

when trying to open the Viewer or RegistrationViewer, here are a few suggestions that might fix the issue:

* In the conda environment, run `#!bash conda install -c conda-forge libstdcxx-ng`
* In the conda environment, run `#!bash conda install -c conda-forge libffi`.

## Filter image clip error

An error can occur when a filtered image clips off too many pixels when trying to save. This happens because the filter
step will scale up every non-DAPI image by a common factor to improve precision. There are two options to deal with this
issue:

 * Reduce image clipping by lowering `scale_multiplier` below the default value found in the `filter` config (the
   default is found [here](https://github.com/reillytilbury/coppafish/raw/HEAD/coppafish/setup/settings.default.ini")).
   After this, delete the `filter` directory found in the tiles directory and the `scale.txt`. Then, restart the pipeline.
 * Follow a "I don't care" strategy by increasing `percent_clip_error` above the default to allow for more clipped
   pixels. You can then restart the pipeline without deleting any files. If you wish to ignore warnings too, increase
   `percent_clip_warn`.

## Memory crash at OMP

Try lowering `subset_size_xy` in the OMP config. This will cause OMP to compute on fewer pixels at time. It has a 
minimal effect on compute times, but can lower the RAM/VRAM usage. The default is found 
[here](https://github.com/reillytilbury/coppafish/raw/HEAD/coppafish/setup/settings.default.ini).
