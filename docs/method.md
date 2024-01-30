The coppafish pipeline is separated into distinct sections. Some of these are for image pre-processing (scale, extract, 
filter), image alignment (register, stitch) and spot detection/gene calling (find spots, call spots, orthogonal 
matching pursuit). Below, each section is given in chronological order.

## Scale

Computes a scale factor for sequencing images and the anchor round. These numbers are typically between 1 and 10. All 
images are then scaled by a number during the [filter](#filter) stage. This is done so that the images take up more of 
the range when the filtered images are saved to disk and converted from floating point numbers to `uint16`, improving 
pixel value precision.

## Extract

Save all raw data again at the `tile_dir` in the `extract` config section. Coppafish does this for: 

* file compression support.
* raw data in a universal format that can then be used by multiple versions of our software.
* more optimised for data retrieval speed. The default file type is using 
[zarr](https://zarr.readthedocs.io/) arrays, but we also support saving as uncompressed numpy arrays by setting 
`file_type` to `.npy` in the extract config section.

Extract also saves metadata inside of the `tile_dir` directory if the raw files are ND2 format.

## Filter

All images are filtered to help us boost signal-to-noise ratio and exaggerate spots. The parts to this are:

* calculating a Point Spread Function (PSF) using “good” spot shapes which is used to apply a Wiener filtering on every 
image if `deconvolve` in the `filter` config settings is set to true (default is false).
* applying a smoothing kernel to every image by setting `r_smooth` in the `filter` config section. By default, each 
pixel is averaged with one other pixel along the z direction.
* a difference of Hannings 2D kernel is applied to every image that is not a DAPI. If it is a DAPI, instead apply a 2D 
top hat filter (which is just a 2D top hat kernel) if `r_dapi` is set to a number in the config.

## Find spots

Point clouds (a series of spot x, y, and z locations) are generated for each filtered image. These are found by 
detecting local maxima in image intensity around the rough spot size (specified by config variables `radius_xy` and 
`radius_z` in the `find_spots` section). If two local maxima are the same value and in the same spot region, then one 
is chosen at random. Warnings and errors are raised at this section if there are too few spots detected in a 
round/channel, these can be customised, see `find_spots` section in the 
<a href="https://github.com/reillytilbury/coppafish/blob/alpha/coppafish/setup/settings.default.ini" target="_blank">
config</a> default file.

## Register

## Stitch

## Call spots

## Orthogonal Matching Pursuit
