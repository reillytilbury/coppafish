* Tile - A cuboid subset of the microscope image of size $n_z \times n_y \times n_x$ in z, y, and x, where $n_y = n_x$. 
Typically, $n_z\le58$. Usually, all adjacent tiles overlap by $10\%-15\%$ to give coppafish information on how to best 
align tiles (see [method](method.md) for details).

* Sequencing round - A series of images across all channels taken when the genes are fluorescing.

* Channel - A wavelength to capture the image around. We use multiple channels to distinguish every dye colour (almost 
always the number of channels is equal to the number of unique dyes). But, a dye can have "bleed through", i.e. causing 
significant brightness in more than one channel.

* Gene code - A sequence of dyes that are assigned to a gene type for each round. Each gene type has a unique gene 
code. For example, if the dyes are labelled `0, 1, 2` and there are 2 sequencing rounds, some example gene codes are 
`0, 1` (i.e. dye `0` in first round, dye `1` in second round), `1, 2`, `0, 2`. For more details on how the codes can be 
generated, see our own gene code generator `reed_solomon_codes` in 
[`coppafish/utils/base.py`](https://github.com/reillytilbury/coppafish/blob/alpha/coppafish/utils/base.py). Also, see 
[wikipedia](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction).

* DAPI - The dapi is an additional, optional channel that causes fluorescence on and around cells. It is used as an 
overlay in the Viewer for [diagnostics](diagnostics.md).

* Notebook - A write-once[^1] compressed file that stores all important outputs from coppafish. The notebook is used 
to plot many [diagnostics](diagnostics.md). Variables from the notebook can be directly read by:
```python
from coppafish import Notebook

nb = Notebook("path/to/notebook.npz")
```
For example, you can read the `use_tiles` variable from the `basic_info` section by
```python
print(nb.basic_info.use_tiles)
```

[^1]:
    There are some cases of a notebook being "rewritten", but these are done only by the developers. This includes 
    the combining of single tile notebooks into a multi-tile notebook.
