#### GSplat chunk bounds contain decoded centers on continuous barrier axes

GSplat spatial indexes now add the center encoder's per-axis round-trip slack to
`chunk_bounds`, including on top of the tight epsilon used for `slice_dims`.
This closes a silent under-fetch where a non-gridded time/channel coordinate
could move farther during uint16 decoding than its barrier bound was padded and
therefore make its own chunk unreachable. Exact gridded axes, LUT encodings and
centers escalated to float32 keep their previous bounds.
