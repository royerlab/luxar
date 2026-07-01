# `luxar.gsplats.interop`

Adapters that connect Luxar's fitted Gaussian splats to external tools. The
external dependencies are imported **lazily**, so this subpackage imports fine
without any extra installed.

## tracksdata bridge

[`tracksdata`](https://github.com/royerlab/tracksdata) is the Royer-lab common
data structure for multi-object tracking (the basis for ultrack, trackedit and
inTRACKtive). This bridge turns each fitted Gaussian splat into a `tracksdata`
**node** carrying a binary segmentation mask, so cells fitted as splats can be
linked across time into lineages and proofread in that ecosystem.

Install the optional extra:

```bash
pip install 'luxar[tracksdata]'
```

### Key functions

| Function | Purpose |
|----------|---------|
| `splat_mask_and_bbox(center, cholesky_factor, frame_shape, n_sigma=2.0)` | Pure-NumPy: rasterize one splat's `n_sigma` support to a local boolean mask + bounding box. Computes the box analytically from the Cholesky factor (half-extent along axis `k` = `n_sigma·‖L[k]‖`), so it rasterizes only the local neighbourhood, not the whole frame. |
| `gsplats_to_tracksdata_graph(gsplats, frame_shape, t=0, ...)` | Add every splat of a `GSplatData` as a node (mask + bbox + amplitude + position) at timepoint `t` in a `tracksdata` graph. Requires the `tracksdata` extra. |

### Example

```python
from luxar.gsplats import GSplatData
from luxar.gsplats.interop import gsplats_to_tracksdata_graph

gsplats = GSplatData.load("cells.gsplats.zarr")        # 2D/3D fit of one frame
graph = gsplats_to_tracksdata_graph(gsplats, frame_shape=(256, 256), t=0)

# Build a time-lapse by adding more frames into the same graph, then link
# nodes into lineages with tracksdata's edge operators:
for t, fit in enumerate(per_timepoint_fits):
    gsplats_to_tracksdata_graph(fit, frame_shape, t=t, graph=graph)
```

### Notes

- Each splat node uses tracksdata's default attribute keys `t`, `mask`
  (`tracksdata.nodes.Mask`) and `bbox`, plus `amplitude` and per-axis position
  (`z`/`y`/`x` for the trailing dims).
- `splat_mask_and_bbox` is dependency-free and unit-tested; the graph builder is
  verified against `tracksdata` 0.1.0rc5.
- Originally prototyped by Jordão Bragantini (@JoOkuma) in PR #20.
