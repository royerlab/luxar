# `luxar.gsplats.interop`

Adapters that connect Luxar's fitted Gaussian splats to external tools and
file formats. The external dependencies are imported **lazily**, so this
subpackage imports fine without any extra installed.

## Classical (photogrammetric) splat import

`classical_splats.py` reads the four common classical 3D-Gaussian-Splatting
file dialects into a `GSplatData` — after which the entire Luxar toolchain
(LOD recipes, partition, filter, `gsplat convert`, scenes, viewer) applies
unchanged. Everything is NumPy + stdlib; **no new dependencies**.

| Dialect | Files | Notes |
|---------|-------|-------|
| `inria` | `point_cloud.ply` from INRIA-style trainers | Header-driven (any SH degree); opacity logit → sigmoid, log-scales → exp, quaternions normalized |
| `splat` | antimatter15 `.splat` | Flat 32-byte records; color/alpha already baked |
| `spz` | Niantic/Scaniverse `.spz` | Legacy gzip container v1–v3 (the format Scaniverse writes). v4 NGSP/zstd is detected and rejected with a clear error |
| `supersplat` | SuperSplat compressed `.ply` | Chunked bit-packed (11-10-11 positions/scales, smallest-three rotations, 8888 color); 12- and 18-property chunk layouts |

**Conversion semantics:**

- Covariance is rebuilt as `Σ = R·diag(scales²)·Rᵀ` and factorized to Luxar's
  packed lower-triangular Cholesky (with an eigenvalue-clamp guard for
  degenerate splats in real files).
- **Opacity → `amplitudes`** (both in [0, 1]).
- **SH color is reduced to the DC band**: `rgb = 0.5 + C₀·f_dc`, baked to
  per-splat SDR colors; view-dependent `f_rest` bands are dropped.
- **Orientation**: COLMAP-convention dialects (INRIA/.splat/SuperSplat store
  +Y down) get a 180°-about-X fix by default so scenes are upright in Y-up
  viewers; SPZ declares Y-up (RUB) data and is left untouched. Override with
  `rotate_x180=`/`flip=` (CLI: `--reorient/--no-reorient`, `--flip`).
- Columns stay in world **(x, y, z)** order — dimension inference labels them
  x/y/z. The applied orientation and source dialect are recorded in
  `stats["interop"]` so a future export can invert them.
- Imported scenes render best with `blending_mode="normal"` (correct once
  depth-sorted rendering lands; acceptable today).

### Usage

```bash
# CLI: sniffs the dialect from extension + header
luxar gsplat import garden.splat garden.gsplats.zarr
luxar gsplat import point_cloud.ply scene.gsplats.zarr --no-reorient
luxar gsplat import capture.spz capture.gsplats.zarr -e precision
```

```python
from luxar.gsplats.interop import import_gsplats

data = import_gsplats("garden.splat")          # → GSplatData
data.save("garden.gsplats.zarr")

# Or straight into a scene — add_gsplats_from_file sniffs classical formats:
scene.add_gsplats_from_file("garden", "garden.splat", blending_mode="normal")
```

Tests generate miniature files of every dialect on the fly
(`tests/_synthetic.py`) and assert reader parity plus covariance fidelity
through the full read → convert → save → load pipeline.

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
