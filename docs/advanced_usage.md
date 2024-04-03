## Move the output directory

Once a coppafish pipeline is complete or partially complete, the output directory will contain various files. If you 
wish to move the output directory somewhere else while the notebook already exists, manually copy files to the new 
output directory (including the notebook). Then use the following command:

```python
from coppafish.utils import set_notebook_output_dir

set_notebook_output_dir("path/to/new/notebook.npz", "new/output_dir")
```

Note that this will not update the notebook when the tile directory is moved (see [here](#move-the-tile-directory)). If 
the notebook does not exist, it is safe to copy the output directory manually without running the command above.

## Move the tile directory

The extracted and filtered images can also be moved to a new tile directory. First, manually move the tile directory to 
a new location. Then, update the notebook to use the new tile directory by running the command:

```python
from coppafish.utils import set_notebook_tile_dir

set_notebook_tile_dir("path/to/notebook.npz", "path/to/new/tile_dir")
```

This will only apply the new tile directory to the given notebook. You can run the function multiple times to update 
every notebook you may have. If no notebook used the old tile directory, you can safely move the directory manually 
without running the command above.

## Retrieve the Notebook config

Notebooks will store a copy of the config file used, this way the notebook becomes separate from the initial starting 
condition once it has run through the pipeline. To access a notebook's config file:

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

## Remove notebook page

Each coppafish section is saved as a separate notebook page. To change the config variables and re-run the coppafish 
pipeline, you can delete the notebook and all output directory files and re-run again. But, if you only wished to 
re-run starting from an intermediate section, you can delete all subsequent sections and output files. For example, if 
you wished to re-run OMP after changing OMP config parameters, you can delete all output files marked `omp_*` then 
remove the OMP notebook page by:

```python
from coppafish import Notebook

nb = Notebook("path/to/notebook.npz")
del nb.omp
nb.save()
```

Now coppafish can be re-run and it will continue from OMP. This is particularly useful for many tile datasets. If you 
are unsure what must be re-run, then it is suggested to start from an empty output directory.
