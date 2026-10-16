#### Fitted volumes open the way Fiji and napari show them

A z-stack converted with `luxar gsplat convert` used to open edge-on, with z drawn
as screen-x, and mirrored vertically compared with Fiji. A fit stores its center
columns in array order (z, y, x), and convert mapped them onto scene axes by
position. Convert now names the columns and maps them onto x, y, z scene
dimensions with `dim_order`, so the Cholesky factors are permuted along with the
centers. The scene opens facing the XY plane with y pointing down
(`viewer_config.camera = {up: (0, -1, 0), view_axis: "xy"}`). The flip is done by
the camera, not by a negative scale, so coordinates, units, picking and scene
bounds stay as fitted. `--view xy|xz|yz` picks a different face. `--dim-order`
names the columns (`t,z,y,x` hides `t` and draws x, y, z). On a z,y,x fit,
`--dim-order x,y,z` is the legacy positional mapping. An unnamed store with no
fitting provenance, such as geometric x, y, z data, keeps the positional mapping
too. A partition is grafted
unchanged and keeps its column order. The preset still finds its XY plane
because the viewer matches the preset letters against dimension names.

Fits now record their axis names. Every writer stamps root `dimension_metadata`
with the array-order names whenever a fit's `fitting/source_shape` has the same
rank as its centers, and an explicit descriptor always takes precedence.
`gsplat merge` writes the names again when all its inputs agree on them; with
`--as-dimension` it adds the new axis, named by `--dimension-name` (default `t`).
A merged timelapse of fits therefore no longer opens edge-on.
`CameraConfig` has a new `view_axis` field, defined in the format contract, which
the viewer uses for the opening frame and for Home/F. The URL parameter
`?viewAxis=` overrides it for one session; it has its own name because `?view=`
is taken by bookmarks. `load_volume(info=...)` now reports the voxel spacing of
OME-TIFF (`PhysicalSize*`) and calibrated ImageJ TIFF files as `axes` and
`voxel_size`. The size is normalized to micrometres (`voxel_unit: "um"`), and the
values and units as recorded are kept in `source_voxel_size` and
`source_voxel_units`.
