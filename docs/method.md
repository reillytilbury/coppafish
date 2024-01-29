## Scale

Computes a scale factor for sequencing images and the anchor round. These numbers are typically between 1 and 10. All 
images are then scaled by a number during the [filter](#filter) stage. This is done so that the images take up more of 
the range when the filtered images are saved to disk and converted from floating point numbers to `uint16`, improving 
pixel value precision.

## Extract

This saves all raw data again at the `tile_dir` in the `extract` config section. Coppafish does this to: 
* support file compression
* re-saving the raw data in a universal way that can then be used by multiple versions of our software 
* we re-save in a way that is more optimised for data retrieval speed. 
This section also saves metadata inside of the `tile_dir` directory if the raw files are ND2 format.

## Filter

All images are filtered to help us boost signal-to-noise ratio and exaggerate spots. The parts to this are:
* calculating a Point Spread Function (PSF) using “good” spot shapes which is used to apply a Wiener filtering on every 
image if `deconvolve` in the `filter` config settings is set to true (default is false).
* applying a smoothing kernel to every image by setting `r_smooth` in the `filter` config section. By default, each 
pixel is averaged with one other pixel along the z direction.
* a difference of Hannings 2D kernel is applied to every image that is not a DAPI. If it is a DAPI, instead apply a 2D 
top hat filter (which is just a 2D top hat kernel) if `r_dapi` is set to a number in the config.

## Find spots

## Register

## Stitch

## Call spots

## Orthogonal Matching Pursuit
