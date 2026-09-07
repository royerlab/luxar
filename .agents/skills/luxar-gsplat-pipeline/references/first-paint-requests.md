# First-paint request cost for gsplat LOD recipes

This reference derives the eager-rung guidance in "Choosing a LOD recipe" and records
the measurements behind it. Paths are relative to `packages/luxar-viewer/src/` for
viewer code and `packages/luxar/src/luxar/` for Python code.

## Derivation

`parts × levels × rungs` counts the tree. First paint pays only the part fetched
eagerly:

```
first-pass rungs = eager parts × 1 level × min(3, rungs per level)
converged rungs  = eager parts × 1 level × rungs per level
requests         ≥ eager rungs × arrays per rung   (+ one per extra chunk)
```

A whole-object `levels` ladder is the exception to the `× 1 level`: it commonly
promotes on frame 1, so count two.

A `kind=lod` group loads exactly one level eagerly — index `default_level`, which
every gsplat recipe (`io/_compiler/gsplat_tree.py`) and scene graft
(`core/group/gsplats_pipeline/from_io.py`) resolves to 0, the coarsest. Every other
LOD-capable child is cheap-attached (placeholder plus loader thunk, no array fetch),
and a nested `kind=lod` or `kind=partition` child defers too unless it carries its own
`transform` or the viewer has no LOD registry
(`data/scene-loader/nodes/load-lod-group-node.ts::canDeferGroup`). A `kind=partition`
group has no selector and loads every part eagerly
(`data/scene-loader/nodes/load-partition-group-node.ts`).

The first pass is a floor, not a ceiling. A cold pass commits two stream rungs
(`data/loaders/progressive/streaming-policy.ts::shouldStopAfterLevel`) and prefetches
a third (`data/gsplats/gsplats-progressive-loader.ts::prefetchNextLOD`). Post-load
refinement, kicked from inside `loadScene`
(`data/scene-loader/lifecycle/load-scene.ts`), then drains the rest a rung per frame
and converges on the full eager-level ladder. Size a hard cap against converged
rungs; `min(3, …)` only describes the first visible pass. Refinement is suspended
while a playback frame budget is active. A slice absent from the SliceCache resets to
LOD 0, while revisiting a cached slice restores its stored full ladder or prefix and
does not re-pay it.

A rung is not one request. `chunk_bounds` is a real one-chunk data read when the leaf
has a spatial index (`data/loaders/chunk-bounds-loader.ts`), followed by `centers`,
`amplitudes`, `cholesky_factors_diag`, `cholesky_factors_offdiag`, and optionally
`colors`, at one HTTP request per zarr chunk with no coalescing. A typical 3D rung is
therefore 5–6 requests plus one per extra chunk, but re-count the arrays for the
actual format: v3.0 packs the Cholesky factors into one array, a colormapped fit has
no `colors`, and an nD rung whose slice query selects nothing costs only the
`chunk_bounds` read. Array metadata is free after the scene's consolidated index is
fetched.

## Selector behavior

Only a partition-anchored ladder reliably keeps its one eager level. The selector
runs on the first frame, and upgrades win without hysteresis
(`scene/lod-selector-math.ts`). A whole-object ladder anchors its finest level at
half-screen occupancy (`core/group/lod/group.py::WHOLE_OBJECT_FINEST_ANCHOR` = 0.5).
The top two rungs of a derived ladder with at least three levels are 1/4 and 1/2, so
typical opening occupancies — about 0.32 for a cube at 16:9 and 0.25 for a 100×80×60
blob — promote immediately to the second-finest level. A two-level ladder (`-L 1`)
holds its coarsest, as does genuinely elongated content such as a 100×1×1 object at
about 0.003 occupancy. A partition-bound ladder instead anchors at
`PARTITION_FINEST_AREA` = 1.0, meaning the tile alone fills the screen, so the opening
framing normally stays coarse.

This explains the recipe split:

- `overview` is a `kind=lod` over `[coarse_leaf, fine_partition]`; the fine branch
  defers behind a fills-screen selector. Crossing it loads every fine part in one
  activation, so it reduces opening-view requests rather than total session requests.
- `tiles` and `adaptive` fetch every part eagerly. Their cost buys frustum culling,
  and `adaptive` additionally lets nearby tiles refine independently of distant ones.
- `stream` and `levels` fetch one leaf or level initially, but a typical whole-object
  `levels` ladder promotes on frame 1.

## Sliced nodes: count the resident slice

Everything above counts rungs. On a node the viewer SLICES — any non-displayed
dimension — you also have to count how many elements land in the rung it actually
draws, because a ladder's rungs are sized against the WHOLE node while only one
hidden coordinate is on screen.

An explicit ABSOLUTE count is the trap. `-b stream:<c>` fixes a count `C` for the
whole node, so the resident slice receives part of `C`. The CLI's `--target-ms`
path avoids that mistake by surveying the store, multiplying its first chunk by
the observed slice count, and logging the multiplier. `--n-lods L` instead makes
rung 0 an aggregate `1/L` share of the whole node, so the average share is stable
as the slice count changes. It is not a per-slice guarantee: the global prefix
concentrates where the signal is, and sparse slices can receive much less.

Measured on the shipped corpus — the 5th-percentile rung-0 count per coordinate
against what the deployed viewer commits while the axis plays:

| node | levels | p05 rung 0 | observed playback | reads as |
|---|---|---|---|---|
| `drosophila_embryogenesis` | 14 | 7 | 20-51 of 166,443 | blank |
| `nexrad_supercell` (published store) | 7 | 4 | 187-440 of ~10,000 | structure gone |
| `zebrafish_timelapse/endoderm` | 4 | 330 | ~1,449 of ~11,159 | soft, usable |
| `celegans_tracking` | 4 | 1,069 | ~2,970 of 3,209 | fine |
| `neuromast_2ch/membranes` | 8 | 2,962 | 12,842-17,465 of 110,614 | soft, usable |

Under a playback frame budget a cold ladder's opening frame starts from the
first rung; since #2377, any further cache-resident rungs can join it. The
playback column above was measured under the old LOD-0-only policy. Rung 0
remains the floor every slice starts from, so a starved p05 still identifies a
starved opening frame wherever playback lands.

The `nexrad_supercell` row describes the **published** store and stays true until
the corpus is regenerated. Its *source* no longer authors that ladder: since #2485
the demo passes `additive_lod=dict(n_lods=4, slice_dims=[3], recompute=True)`,
whose rung 0 delivers a measured 2,744 splats per scan and carries the
5th-percentile scan (774 splats) whole. The row is kept rather than deleted
because the lesson — count per slice, not per node — is what it is here to teach.

**Rules that follow.** Prefer `--n-lods 3..4` on any node with a hidden
dimension, then inspect the per-slice histogram on a long or non-uniform axis
rather than trusting the aggregate share. From Python, the stronger option is
`additive_lod=dict(..., slice_dims=[3])` — raw pre-`dim_order` centre columns —
which interleaves the ordering round-robin across the hidden coordinates so every
rung carries an equal ABSOLUTE per-slice budget instead of a proportional share —
a guarantee rather than an average. Pair it with `recompute=True` on a stacked
dataset, or the spec is shadowed by the merged per-source ladder and never runs.
There is no CLI flag for it: `gsplat lod` has no scene to say which columns are
hidden. If you use `--target-ms`, read
the CLI's logged slice multiplier rather than assuming the number you typed is
what a viewer will see. Count stops as distinct OCCURRING combinations across
all hidden axes: not the product of per-axis cardinality
(`biodiversity_planetary_scale` is 126 populated of 140), and not the declared
`Dimension` range (`drosophila_embryogenesis` declares 500 timepoints and its
coarsest rung carries data at 499).

Four landed protections keep this from resting on memory: the CLI scales
`--target-ms` by the slice count; the demo policy floors sliced first rungs at an
aggregate share before stores are built; `additive_lod=dict(slice_dims=…)` makes
a gsplat ladder's per-slice budget absolute rather than proportional, and the demo
authoring gate requires it (with `recompute=True`) on any gsplats adder that
*authors* an `additive_lod=` in a hidden-dimension demo — 12 of the 13 such demos
pass none, so the gate does not reach them; and `hatch run check-demo-ladders`
fails a built store
whose sparsest slices fall below the floor. The gate measures the 5th percentile,
not the maximum — a ladder starves at its sparsest slice, and one busy coordinate
used to mask hundreds of starved ones.

## Measured example

One 1.58 GiB 4D timelapse built with `adaptive` had 44 parts × 4 levels × 4 rungs =
704 nodes. It therefore fetched 44 × 1 × 3 = 132 rungs on the first pass and 176 when
converged. Both compared stores carried five arrays per rung (`chunk_bounds`,
`centers`, `amplitudes`, two Cholesky arrays, no `colors`), placing the nominal
structure term at 660 first-pass and 880 converged requests. Measured first paint was
689 after `luxar optimise --profile archive` and 922 as built.

The control held the same data in a single stacked 4D leaf with an eight-rung
`stream` ladder. Its nominal structure term was 15 first-pass and 40 converged;
measured locally, where first paint equalled convergence, it used 39 requests at
`archive` and 62 as built. The one-request miss falsifies the nominal 40 floor: at
least one assumed read was absent, for example a rung with `ordering: "none"`, which
short-circuits the `chunk_bounds` probe.

Only the low end is a real test because extra chunks can only add requests; 880 is
not a ceiling. Like chunk regime for like, the 704-node store cost 15–18× the
eight-node store. At fixed structure, re-chunking moved first paint 25–37% (922 →
689; 62 → 39), a real second-order term that cannot change the eager structure.

Carry no fitted constant forward. Across these measured points, `requests/node`
spans 0.98 to 7.75, a 7.9× range. The ratio combines two independent variables: what
fraction of the tree is eager and how many chunks each array spans. Derive from eager
rungs rather than extrapolating a per-node rate.

## Measuring a scene

Open the viewer with `?debug` and read
`window.__luxarDebug.cache.getStats().network.requestCount` immediately after the
ready poll. This counts actual byte fetches through the chunk source, including
prefetch; `demand.networkRequests` excludes prefetch. `isLoading` deliberately omits
the refinement drain, so there is no converged flag: sample `requestCount` again once
it plateaus.

Do not use the data-loading monitor's visible "N reqs": it is
`totalRequestsServed`, which includes cache hits across every tier and is not an HTTP
count. `requestCount` itself excludes the document, bundle, WASM, and zip-open probes,
so it remains a lower bound. For an on-disk store, `luxar info --stats` is the offline
companion that projects full-load chunk count.
