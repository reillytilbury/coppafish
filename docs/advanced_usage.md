## Move the output directory

Once a coppafish pipeline is complete or partially complete, the output directory will contain various files. If you 
wish to move the output directory somewhere else while the notebook still exists, do not manually copy files. We 
recommend the following:

```python
from coppafish.utils import move_output_dir

move_output_dir("old/output_dir", "new/output_dir")
```

where the old and new output directories must both exist as well as the notebook inside of the old output directory. If 
the tile directory (given as `tile_dir` in the `file_names` config file) is kept inside the output directory, this will 
not be moved (see [](#move-the-tile-directory)). If the notebook does not exist, it is safe to copy the output 
directory manually.

## Move the tile directory

The extracted and filtered images can also be moved by doing:

```python
from coppafish.utils import move_tile_dir

move_tile_dir("old/tile_dir", "new/tile_dir", "path/to/notebook.npz")
```

Note that this will only apply the new tile directory to the given notebook. You can run the function multiple times to 
update every notebook clone. If no notebook is attached to the old tile directory, you can safely move the directory 
manually.

## Retrieve the Notebook config

A notebook stores a copy of the config file used to run it, this way the notebook becomes separate from the initial 
starting condition once it has run through the pipeline. To access a notebook's config file:

```python
from coppafish import Notebook

nb = Notebook("path/to/notebook.npz")
config = nb.get_config()
```

`config` is a dictionary of dictionaries. Each key is a section/page name, each item is a dictionary containing each 
config variable and its set value.

Access the absolute file path to the config file by:

```python
nb._config_file
```
