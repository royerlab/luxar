#### Preserve OME-Zarr axis metadata through batch-fit conversion

`gsplat batch-fit` now records the selected array's axis names, units, and NGFF
coordinate scales in the manifest and merged `.gsplats.zarr`. `gsplat convert`
uses those descriptors while keeping ranges derived from the fitted splats, so a
recording converted from `time,z,y,x` produces a named time slider instead of
`dim3`, without mislabelling voxel-index coordinates as physical values.
