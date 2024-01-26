* Tile - A subset of the microscope image, split into cuboids of size $n_z \times n_y \times n_x$ in z, y, and x, where 
$n_y = n_x$. Typically, $n_z\sim50$.

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

* Notebook - 
