### Fixed

- Added an explicit `batch-fit run/submit --physical` mode that uses the selected OME-Zarr NGFF spatial scale for fitted centers, covariance, tile/content-box origins, partition split planes, and quality scoring. Index-space output remains the default, config-supplied `voxel_size` values still win, and incompatible volume merge-refinement is rejected during planning.
