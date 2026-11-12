# Choosing a LOD recipe: the measurements behind "ask about the VIEW"

Loaded from the `luxar-gsplat-pipeline` skill ("Choosing a LOD recipe"), which
carries the three deciding questions. This file holds the measured cases, the
per-timepoint partition recipe, and the chunk-profile rules that come with
`stream`.

**Default to `stream`, and make anything costlier justify itself.** Element count
alone is a poor guide: a 128M-splat timelapse wants `stream`, while a 0.7M-splat
galaxy legitimately wants `levels`. Three questions decide it, and all three
matter (measured 2026-08-26):

**Is it ONE object, always fully in frame?** Then there is nothing to
frustum-cull and parts cost a request per node to bootstrap for no return.
Measured first paint: **63 requests** for a single stacked leaf (8 nodes) vs
**689** for a `kind=partition` of 44 parts x 4 levels x 4 rungs (704 nodes) —
~15-18x, like-for-like on chunking. The scaling is SUBLINEAR (88x nodes, ~16x
requests), so halving a node count does not halve the cost.

**Is it viewed WHOLE, or at range?** The LOD selector is screen-occupancy based,
so at a full-frame view the *finest* substitutive level shows and coarse levels
are bytes nobody fetches. They pay off only when the object is genuinely small
on screen. `cryoem_virus` (a compact particle, always full-frame) moved
`levels -> stream` for **-28%** when regenerated (16.03 -> 11.50 MB, 16 -> 4
nodes; publishing the regenerated archive remains #1879);
`milky_way_dust` keeps `levels` because the galaxy really is orbited at range,
and pays +39% on purpose.

**Does the RESIDENT set fit?** `MAX_SPLATS_PER_GSPLATS_NODE` is 4,194,304 on a
4096-class GPU (8.38M at 8192). Above it the viewer reports the clamp at load
time and drops the tail; since storage is Hilbert-ordered that tail is one
contiguous lobe — a clean-edged hole, not noise. **Only the resident slice
counts**: an nD node sliced on a hidden axis is measured per-slice, so a
500-timepoint node holding 128M splats at ~256k/frame is fine. The compiler also
warns on the node total rather than the resident slice, so that warning is
expected for a sliced nD node that satisfies the runtime limit. For a STATIC
object over the cap, parts are load-bearing.

**On a time-stacked 4D node, a SPATIAL partition buys nothing.** The writer
lexsorts by the time barrier (compound ordering: discrete hidden dims first,
Hilbert within a slice), so a timepoint's splats are already contiguous and
chunk-local without any partition. Measured: spatial parts cost +9% bytes for
zero request benefit; substitutive levels cost +49%.

**A partition per TIMEPOINT is a different animal, and it won (2026-09-10).**
Measured on the h2afva 51-timepoint archive at identical chunking, cold cache,
stepping the time axis: the single sliced leaf with a global 12-rung ladder
re-streams that ladder from its bottom on every slice and leaves **3-27%** of
the frame resident while a step loads; 44 spatial parts keep 68-85% resident
but pay ~30 MB per step because every part is still sliced by time; **one part
per timepoint** with its own equal-count ladder fetches exactly one part per
step (~27 MB, nothing from other timepoints). Each part uses eight
equal-count rungs, so the 2,217,045-splat reference frame paints 277,131 splats
(12.5%, ~8.31 MB at 30 B/splat) in rung 0 rather than the former 20.8 K download-
budget rung. Its cost is the full part per step with no reuse across timepoints,
and one live GPU buffer per part (51 here) — pair it with
next-timepoint prefetch and watch the GPU byte budget (the refinement residency
gate stops rungs when the pool is near its ceiling). `equi-energy` ladders lost
on the same data: first rungs of 2-7 K splats let the viewer's stop-early rules
end a pass at e(k)=0.33. Reach for equi-energy where a few heavy elements carry
most of the light, not on a uniform nuclei field. Recipe, from Python: take the
finest content, split on the time column, `make_additive_lod(part,
method="self_energy", breakpoints=time_part_ladders([n], part_count=51)[0])` per part,
write a `GSplatPartition` with `barrier_dims=[time_col]`, graft it with
`add_gsplats_from_file` (see `demo_gsplats_4d_h2afva_timelapse.py`).

**One catch that comes with `stream`:** a flat store needs re-chunking or
scrubbing gets WORSE. Measured per timepoint step on a 4D leaf: **173 requests
as-built, 2 after `luxar optimize --profile archive`** — the as-built figure is
worse than a partitioned store's re-chunked 12. Additive-only and re-chunking
are a package, not alternatives.

**But judge a chunk profile on the node's ACCESS PATTERN, not on the store.**
`optimize` sizes each array from a byte budget, then uses scene animation and
index metadata for a warn-only playback check. Three regimes (2026-08-30 and
2026-09-10 measurements):

- *Laddered, played* (drosophila, h2afva): a coarse rung that fits ONE chunk
  stays cache-resident and the viewer serves every later timepoint from it
  (#2377). 1 MB wins; 64 KB made playback 13x worse there.
- *Un-laddered, played* (`collision_animated`, an animated Lines node): the store
  is already slice-major, so a 1 MB chunk simply holds ~6 frames. Every ~6
  frames the playhead crosses into the next chunk of each array. The viewer now
  prefetches the nearest next chunk boundary while the preceding frames play,
  adding about 2.9 frames of mean lead at 1 MB on this node. Here the number to
  compute is **frames-per-chunk**: group consecutive `chunk_bounds` atoms as the
  planned zarr chunks will group them, then measure each group's inclusive
  hidden-axis span. `hosting` (~1.5 frames per chunk) is the
  middle of the request/byte trade; the request cap of the hosted site
  (Cloudflare Functions) rules out `local`'s 3,500 chunks. `optimize` now warns
  when an actively played, un-laddered node would exceed two frames per chunk
  across multiple chunks.
- *Not animated* (keypress-navigated, refine pass): smaller is a pure win.

The slice-major ordering is NOT something to add — it already exists
(`io/_ordering/compound.py`); do not re-propose it.

Three refinements from applying the rule to a whole hosted wave (2026-09-11):

- *Gate on whether the axis is actually PLAYED.* Only a store whose
  `viewer_config.animation` entry has `playing: true` (or whose users play it)
  is the frames-per-chunk case; a node sliced by a keypress-navigated axis
  (ocean's time, forest's season/growth, hilbert's order) is the refine case,
  and `archive` stays right. Camera `auto_rotate` is not playback.
- *One chunk for the whole array is the BENIGN shape*, not the risk: there is no
  boundary to cross, every frame is resident after one larger fetch. The stall
  needs MANY chunks holding several frames each; boundary-aware lookahead now
  warms the next crossing while the preceding frames play.
- *Read the atom's hidden-axis span from `chunk_bounds`, not the scene
  dimension's range.* A store that partitions NODES by the hidden axis (one
  node per Hilbert order) holds one coordinate per node; dividing the scene
  range by the node's chunk count reports frames-per-chunk that do not exist.

Measured on the wave: `collision_animated` on `hosting` = 143 chunks over 250
frames = 1.8 frames/chunk (target 1-2); ocean's tentacles sit at 28 frames per
`archive` chunk (~7 on `hosting`) — harmless while nothing plays that axis,
but "make ocean play" is not a free change.
