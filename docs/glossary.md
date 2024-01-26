* Tile - A subset of the microscope image, split into cuboids of size $n_z \times n_y \times n_x$ in z, y, and x, where 
$n_y = n_x$. Typically, $n_z\sim50$.
* Sequencing round - A series of images across all channels taken when the genes are fluorescing.
* Channel - A wavelength to capture the image around. We use multiple channels to distinguish every dye colour (almost 
always the number of channels is equal to the number of unique dyes). But, a dye can have "bleed through", i.e. causing 
significant brightness in more than one channel.
* Gene code - 
