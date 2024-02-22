* Channel - A wavelength to capture the image around. We use multiple channels to distinguish every dye colour (almost 
always the number of channels is equal to the number of unique dyes). But, a dye can have "bleed through", i.e. causing 
significant brightness in more than one channel.

* DAPI - The dapi is an additional, optional channel that causes fluorescence on and around cells. It is used as an 
overlay in the Viewer for [diagnostics](diagnostics.md).

* Gene code - A sequence of dyes that are assigned to a gene type for each round. Each gene type has a unique gene 
code. For example, if the dyes are labelled `0, 1, 2` and there are 2 sequencing rounds, some example gene codes are 
`0, 1` (i.e. dye `0` in first round, dye `1` in second round), `1, 2`, `0, 2`.

* Notebook - A write-once[^1] compressed file that stores all important outputs from coppafish. The notebook is used 
to plot many [diagnostics](diagnostics.md). A notebook can be loaded by 
`from coppafish import Notebook; nb = Notebook("path/to/notebook.npz")`. Variables from the notebook can be directly 
read. For example, you can read the `use_tiles` variable from the `basic_info` section by 
`print(nb.basic_info.use_tiles)`. Each variable also has a description, which can be printed by 
`nb.basic_info.describe("use_tiles")`.

* OMP - Stands for Orthogonal Matching Pursuit. It is the final section of the coppafish pipeline. It is coppafish's 
most sophisticated algorithm for gene calling, and is used as a way of untangling genes that overlap on images by 
assuming that the pixel intensity is a linear combination of each gene intensity. We have no reason to believe that 
gene intensities would not combine linearly.

* Point cloud - A series of spatial pixel positions. Typically used to represent detected spot positions during find 
spots.

* PSF - Stands for Point Spread Function and is used during image filtering. It is used in the Wiener deconvolution 
which is applied to try and deblur images from noise which is caused by frequencies with a low signal-to-noise ratio. 
See <a href="https://en.wikipedia.org/wiki/Wiener_deconvolution" target="_blank">here</a> for more details.

* Sequencing round - A series of images across all channels taken when the genes are fluorescing.

* Spot - A fluorescing gene. A spot is spherical in shape and these are what need to be identified by coppafish.

* Tile - A cuboid subset of the microscope image of size $n_z \times n_y \times n_x$ in z, y, and x, where $n_y = n_x$. 
Typically, $n_z\le58$. Usually, all adjacent tiles overlap by $10\%-15\%$ to give coppafish information on how to best 
align tiles (see [method](method.md) for details).


[^1]:
    There are some cases of a notebook being "rewritten", but these are done only by the developers. This includes 
    the combining of single tile notebooks into a multi-tile notebook.
