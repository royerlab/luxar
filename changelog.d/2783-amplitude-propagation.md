#### Preserve source-matched gsplat amplitudes through rewrites

LOD, additive, partition, decimate, transform, re-encode, filter, slice, cull,
GSplat merge, plan-box fits, scene conversion, and batch merge now retain the
amplitude tier inferred from a fit's source dtype. The scene compiler exposes
the same gsplat-only choice without changing the encoding policy for point
radii or line widths.
