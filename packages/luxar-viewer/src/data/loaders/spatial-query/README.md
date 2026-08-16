# Spatial Query

The chunk-bounds query path shared by the Points, Lines, and GSplats
spatial-index loaders: turn a view state into the minimal set of array ranges
that must be fetched, then load those ranges regardless of how the Python
`luxar.encoding` layer stored them.

## Overview

Each geometry loader probes its own `chunk_bounds` array (loader-private; the
names differ — `chunk_bounds` for points/gsplats, `vertex_chunk_bounds` +
`segment_chunk_bounds` for lines), then hands the resulting
[`ChunkSpatialIndex`](./spatial-query-builder.ts) to this folder for the
common work:

1. **Tolerance** — `tolerance-computer.ts` turns the view state into a
   per-dimension tolerance using geometry-aware semantics.
2. **Query** — `spatial-query-builder.ts` builds the query box, runs an AABB
   scan over the chunk bounds, converts matching chunks to ranges, and
   coalesces them.
3. **Fetch** — `range-loader.ts` loads each range from zarr, dispatching on
   the array's encoding (direct / quantized / LUT / broadcasted / array_ref /
   perchannel).

The query side runs on the **main thread** — an AABB scan is
O(numChunks × ndim) and finishes in microseconds, so worker dispatch would
only add structured-clone overhead. The fetch side offloads heavy decodes to
the worker pool (see `range-loader/`).

## File Structure

```
spatial-query/
├── spatial-query-builder.ts   # SpatialQueryBuilder + AABB scan / range helpers
├── tolerance-computer.ts      # Geometry-aware per-dimension tolerance
├── range-loader.ts            # Encoding-dispatch range loader (entry point)
└── range-loader/              # Per-encoding loader bodies (see its README)
    ├── detect-encoding.ts
    ├── direct.ts  quantized.ts  perchannel.ts  lut.ts  broadcasted.ts
    ├── array-ref.ts  ref-resolution.ts
    ├── encoding-types.ts  shared-instance.ts
    └── README.md
```

## SpatialQueryBuilder

`SpatialQueryBuilder` runs one chunk-bounds query end-to-end. The tolerance
source is selected at construction via a discriminated union — supply **either**
a `geometryType` (tolerance delegated to `computeTolerance`) **or** a
pre-computed `tolerance: number[]` (used by the points loader, which already
knows about `EffectiveRadiusConfig`). The points pre-computed path
(`calculateSpatialQueryTolerance` in `effective-radius-calculator.ts`) applies
the SAME shared `discreteDimTolerance` quarter-cell rule for discrete dims, so
both paths agree at runtime.

```typescript
import { SpatialQueryBuilder, type ChunkSpatialIndex } from './spatial-query-builder';

// geometry-aware path (gsplats / lines)
const ranges = await new SpatialQueryBuilder(index, viewState, {
  geometryType: 'gsplats',
  totalElements: attrs.n_splats,
  chunkSize: attrs.chunk_size,
  extendDims: attrs.extend_to_all,
  logModule: Modules.GSPLATS_SPATIAL_INDEX_LOADER,
}).execute();

// pre-computed tolerance path (points)
const tolerance = calculateSpatialQueryTolerance(viewState, config, ndim);
const ranges = await new SpatialQueryBuilder(index, viewState, {
  tolerance,
  totalElements: attrs.n_points,
  chunkSize: attrs.chunk_size,
}).execute();
```

`execute()` returns merged `LoadRange[]` for the visible elements, or a single
load-all range when an `extend_to_all` dimension is currently hidden (see
`shouldExtendVisibility`).

### Canonical index shape

All three geometry types produce indices of this shape — a flattened
`(numChunks, ndim, 2)` `[min, max]` array plus the chunk count and
dimensionality. For lines, only the **segment** side feeds the chunk query;
the vertex side is loaded via a separate sorted-indices path.

```typescript
interface ChunkSpatialIndex {
  chunkBounds: Float32Array; // (numChunks, ndim, 2) flattened row-major
  chunkCount: number;
  metadata: { ndim: number; chunk_size?: number };
}
```

### Standalone helpers

The builder is a thin shell over pure functions, all exported for callers that
don't need the full builder (e.g. `scene-loader`'s lines clipping path):

| Function                                          | Description                                                                                    |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `buildQueryPosition(viewState, ndim)`             | Pad/truncate `slicePosition` to `ndim`.                                                        |
| `executeSpatialQuery(params, logModule?)`         | AABB scan; returns matching chunk indices. Early-exits on the first non-overlapping dimension. |
| `chunkIndicesToRanges(indices, chunkSize, total)` | Map chunk indices to `LoadRange[]`, clipped at `total`.                                        |
| `mergeRanges(ranges)`                             | Coalesce overlapping/adjacent ranges (sorted by `start`).                                      |
| `shouldExtendVisibility(extendDims, viewState)`   | True iff an `extend_to_all` dim is currently hidden.                                           |
| `createLoadAllRange(total)`                       | Single range covering the whole dataset.                                                       |

## Tolerance computer

`computeTolerance(geometryType, displayDims, ndim, dimensions?, options?)`
returns a length-`ndim` tolerance array. Displayed dimensions always get an
infinite sentinel (`1e10`). Hidden **discrete** dims share ONE rule across all
three geometries, split by ROLE: the default **query** role
(`discreteDimTolerance`) is a quarter-cell `0.25 × step` (fallback `0.25`) —
deliberately `< 0.5 × step` so write-side chunk-bound padding plus tolerance
can never sum to a full step and bleed the neighbouring category (the barrier
over-fetch fix) — while the **membership** role
(`options.discreteRole: 'membership'`, used by the lines projection-clipping
path via `discreteDimMembershipTolerance`) is the half-cell `0.5 × step`
matching the points/gsplats projection visibility gates. Hidden **spatial/continuous** dims stay
geometry-specific:

| Geometry  | Hidden spatial dim                                                | Hidden discrete dim             |
| --------- | ----------------------------------------------------------------- | ------------------------------- |
| `points`  | `maxRadius` (so points whose radius intersects the slice load)    | `0.25 × step` (fallback `0.25`) |
| `lines`   | `0` (segment bounds already include line-width extent)            | `0.25 × step` (fallback `0.25`) |
| `gsplats` | `max(1e-3 × step, 2.75e-5)` — a float-safety epsilon, not a reach | `0.25 × step` (fallback `0.25`) |

The gsplats continuous value is deliberately near-zero
(`gsplatsContinuousDimTolerance`). Chunk bounds already carry the ellipsoidal
`truncation_radius · σ` expansion on every continuous dim
(`packages/luxar/src/luxar/io/_ordering/gsplats.py`, with
`coverage_sigma = truncation_radius`), and the hidden-dimension cutoff on the read
side is that same radius: the projection kernel
(`packages/luxar-viewer/src/wasm/typescript/gsplats-processing.ts`, function
`project_gsplats_nd_to_3d`, and its Rust twin) attenuates by
`(e^{-m²/2} − c)/(1 − c)` with `c = e^{-T²/2}`, clamped at 0 and therefore exactly
0 at `m = T = truncation_radius`. (The shader's `uTruncate`/`uTruncateSq` discard
is the DISPLAYED-dimension cutoff; it never sees a hidden dimension.) So a chunk
that misses a zero-tolerance query holds only splats the projection would
attenuate to nothing, and any wider reach is pure over-fetch. (This was
`step × 3.0`, documented as "3σ" although it was a multiple of the navigation STEP
and unrelated to the covariance.)

**Precondition.** `compute_chunk_bounds_gsplats` expands the dims NOT in its
`slice_dims` argument and gives the ones that ARE only a tight
`_BARRIER_BOUND_EPS` pad, so the claim above holds while the write side's barrier
set equals the set this module reads as `discrete`. It does for a scene that
declares its dimensions (`packages/luxar/src/luxar/io/_compiler/geometry_writers/gsplats.py` uses
`discrete and not display`). With `scene_dimensions` absent the writer falls back
to the value-based `detect_barrier_dims` in
`packages/luxar/src/luxar/io/_ordering/compound.py`, whose own
docstring calls a false positive a correctness bug — a really-spatial axis with
integer, low-cardinality values gets tight bounds while this side epsilon-only,
and a σ-extended splat near a chunk edge can be dropped. The disagreement predates
this rule, but the old `step × 3.0` reach (a bare `3.0` with no dimension
metadata) MASKED it by covering those tight bounds; a `1e-3 × step` epsilon does
not. The robust fix is plumbing rather than a wider reach: the writer already
publishes the set it used as the ordering `slice_dims` attr
(`packages/luxar/src/luxar/io/_compiler/gsplat_assembly.py`) and the
tolerance computer does not read it.

The epsilon itself covers two hazards, and has exactly one term for each.
(1) A continuous dim along which the splats have zero variance (a stacked axis
declared continuous) gets bounds that are float-EXACT at the axis value, because
the write side epsilon-pads discrete dims only; at a literal `0` tolerance
membership would be an exact float comparison. The dominant perturbation there is
that the bound is stored as **float32** (`chunk_bounds` is `dtype=np.float32`)
while the query is a float64 — ≈1.9e-7 at a coordinate of 5.3; the
`start + k × step` arithmetic drift in the query position is secondary (≈9e-16).
(2) The read side does not actually use a zero variance: `computeMarginalCholesky`
regularizes a degenerate pivot to `sqrt(CHOLESKY_EPSILON)` = 1e-5 for an all-zero
hidden block, so such a splat still renders out to
`truncation_radius × 1e-5` ≈ 2.75e-5 — which term 1 falls below for any
`step < 2.75e-2`.

So there are two regimes with one crossover, at `step = 2.75e-2`: the
`_BARRIER_BOUND_EPS` mirror `1e-3 × step` above it, the ABSOLUTE 2.75e-5 band
below. Below the crossover the epsilon spans many CELLS (≈27.5 at `step = 1e-6`)
and that is deliberate: the kernel's regularization floor is an absolute variance
backstop, so the rendered band does not shrink with the declared step, and a cap
at a fraction of a cell (tried and removed in review) would hide content the
renderer genuinely shows — at `step = 1e-6` a `0.25 × step` ceiling gives 2.5e-7
against a 2.75e-5 render band, narrower even than the old `step × 3` rule. The
quarter-cell rationale belongs to the DISCRETE arm, where over-reach bleeds a
neighbouring category; on a continuous axis it is a bandwidth question only (and
mesh already gets a full cell there). The property that holds at every step is the
one that matters: the epsilon is never smaller than the band a degenerate axis can
render in, so the query cannot miss renderable content. What that bounds is the
DISTANCE, not the fraction of a node fetched: if the dim's real σ is micro-scale
too (`step = σ = 1e-9`, a metre-declared axis with nanometre structure) the needed
window is `T · σ = 2.75e-9`, so the epsilon over-fetches by ~1e4 and can pull the
whole node — the one regime where this is worse than the old `step × 3`, and one
the σ-plumbing below fixes. The real fix is a
scale-aware regularization floor in the kernel, or plumbing the splats' actual σ
into the query — not a wider or narrower constant.

Three documented limits stay: coordinates far from the origin (an expansion below
half a float32 ULP rounds away, so with `T = 2.75` any
`σ_d ≲ (1.1–2.2)e-8 × |coord|` behaves like zero variance), a degenerate dim
alongside another hidden dim of very large σ (the regularizer's relative floor
`σ_max × 1e-6` renders `T × σ_max × 1e-6`, which exceeds the epsilon once
`σ_max > max(1e-3 × step / (T × 1e-6), 10)`, i.e. `≈363.6 × step` or the absolute
10 — the two branches agreeing exactly at the crossover, where the unrounded
coefficient gives 10), and a
node whose `truncation_radius` is well above the default (the band scales with it,
while this call site sees only a `DimensionInfo` and must use the default).

Points decide "spatial vs discrete" from `options.spatialExtendDims` (the
per-dimension flag array carried by `EffectiveRadiusConfig`) rather than the
`discrete` flag; lines and gsplats read `DimensionInfo.discrete` / `step`.
Called from `SpatialQueryBuilder`'s geometry-aware path and directly from
`scene-loader.ts` for lines projection clipping.

## RangeLoader

`RangeLoader` is the single entry point for range-based array loading shared by
all three spatial-index loaders. It detects the encoding from the array's
metadata and dispatches to the matching `load*` body under `range-loader/`,
writing the decoded result into a caller-supplied `Float32Array`.

```typescript
import { getSharedRangeLoader } from './range-loader/shared-instance';

const loader = getSharedRangeLoader();
const written = await loader.loadRangesResolvingRef(
  array,
  attrs,
  ranges,
  output,
  totalElements,
  elementsPerItem,
  zarrStore,
  logPrefix
);
```

- `loadRanges(...)` — dispatch on encoding (`broadcasted` / `quantized` /
  `lut` / `array_ref` / `perchannel` / `direct`, defaulting to `direct`).
- `loadRangesResolvingRef(...)` — same, but first resolves `array_ref`
  encodings (opens the target array via `resolveArrayRef`) and delegates
  against the resolved array. The standard entry point for loaders.
- `static detectEncoding(attrs)` — expose the encoding classifier.
- `getDecoder()` — the underlying `ArrayDecoder` for full-array operations.

Per-encoding semantics, the worker-offload threshold, the `array_ref`
resolution rules, and the process-wide shared singletons all live in
[`range-loader/README.md`](./range-loader/README.md).

## See Also

- [`range-loader/`](./range-loader/README.md) — per-encoding loader bodies and
  the shared `RangeLoader` / `ArrayRefRegistry` singletons.
- [`../README.md`](../README.md) — unified loader infrastructure overview.
- [`../chunk-bounds-loader.ts`](../chunk-bounds-loader.ts) — shared `chunk_bounds`
  zarr probe (the layer below this folder).
- [`../../array-decoder/`](../../array-decoder/) — `ArrayDecoder` /
  `ArrayRefRegistry`; the main-thread decode fallback and metadata helpers.
