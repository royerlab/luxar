# Cross-Node Depth Ordering — Depth-Shard Interleaving (Route B)

> **Status**: SPEC ONLY — nothing implemented, no branch. The feasibility question that gated the design ("what does splitting one instanced draw into S actually cost?") has been **measured** — see §7. The design below is written against those numbers, not against an estimate.
> **Scope**: Viewer-only. No `.luxar.zarr` / `.gsplats.zarr` format change, no Python-side change, **no shader change**.
> **Goal**: correct back-to-front compositing *between* two or more overlapping order-dependent nodes — of any instanced geometry type, in any mix of order-dependent blending modes — to a bounded, tunable ordering error.
> **Non-goals**: per-pixel ordering (the StopThePop error class — a single per-element depth cannot fix it, and no global sort can); order-independent transparency; one globally merged draw (Spark's architecture — §8.2); Mesh (§8.1).

Related reading: `GSPLAT_DEPTH_SORTING_SPEC.md` (the within-node subsystem this builds on — §1's "Overlapping-node authoring rule" states the limitation this document removes), `VOLUMETRIC_BLENDING_SPEC.md`, `src/rendering/depth-sort-coordinator/README.md`.

Probe source + raw results: `delme/draw-split-probe/` (throwaway; not a repo artifact).

---

## 1. Problem statement (facts on the ground)

- **A draw call is atomic in draw order.** Within one call the element order is fully controllable (`aSortedIndex` permutes draw slot → storage slot). Between two calls there is no interleaving: the whole of one draw composites before the whole of the next.
- **three.js's only inter-object lever is `renderOrder`**, and it is compared *before* `z` — verified in the pinned r184 on both backends (`three.module.js:8142`, `three.webgpu.nodes.js:33131`, `reversePainterSortStable`). One integer per object is the entire expressive budget.
- So `rendering/depth-sort-coordinator/render-order.ts` is already at the ceiling of what per-object ordering permits: group by partition wrapper, order groups by mean member view-z with a strict-containment override (PR #843), order within a group by exact BSP painter rank (Fuchs–Kedem–Naylor) where a `bsp_tree` is available and by view-z otherwise, then write sequential integers `1..M` (0 reserved for never-committed empty parts).
- **Two overlapping order-dependent nodes therefore cannot be composited correctly, ever, by that mechanism** — and they exchange relative order as the camera moves. Today's answer is an authoring rule (`GSPLAT_DEPTH_SORTING_SPEC.md` §1): merge them into one node and `partition={"max_elements": N}` so BSP cells give an exact order. Measured on the nuclear-pore demo (9.87M atoms): 6 semantic nodes → visible artefacts, 60 → worse, 1 node + partition → exact.
  The sharper statement is not "the heuristic picks wrongly" but that **no valid whole-object order exists**: for two *concave interpenetrating* objects, no assignment of one integer per object is correct from every viewpoint. That is why shard granularity is the fix rather than a better heuristic — a shard is a thin depth interval, and thin intervals of interpenetrating objects *do* admit an order.
- **This is type-agnostic, so the narrow version of the ask is not cheaper.** Sharding operates on draw calls, not on element data, so "order all splats, points and lines against each other" costs exactly the same as "order within each geometry type separately". There is no cheaper per-type variant to fall back to.
- That rule has three costs the viewer should not be imposing: it forfeits per-node Layers rows, it requires the nodes to share geometry type, blending mode and effective opacity, and it collides with the per-node capacity clamp (Points: `floor(4096/3) × maxTextureSize` = 5,591,040 elements, then silent truncation).
- **The bug is reproducible on demand.** The §7 probe renders the same geometry three ways — one draw per node, S=16 node-major, S=16 interleaved — and gets **three different images with the same lit coverage** (identical to 4 decimal places). Same fragments, same count, different composite: a pure ordering difference.

### What is already favourable

| Fact | Where | Why it matters here |
| --- | --- | --- |
| All three instanced types draw the **same** base quad (4 verts / 6 indices, identical `aQuadCorner`) | `gsplat-geometry.ts:62`, `point-geometry.ts:81`, `line-geometry.ts:89` | the mechanism is type-agnostic by construction |
| Per-element data lives in an element texture; the **only** per-instance attribute is the `aSortedIndex`/`aSortedIndexB` pair | `element-storage.ts`, `element-texture-layout.ts` | re-ordering costs 4 B/element, and a shard is a *slice of that one attribute* |
| The sort kernel is geometry-neutral: projected centers in → back-to-front permutation out | `workers/sort-worker/sorting.ts` | shards fall out of a permutation the subsystem **already computes** — no new sort of any kind |
| gsplat `normal` **and every** `volumetric` share one framebuffer state — `(One, OneMinusSrcAlpha)`, `depthWrite: false`; Points/Lines `normal` differs | `blending-state.ts:363`, `:403` | Route B needs **no** blend uniformity (state is per draw), so the mode mix is free here — the divergence is only a Route C problem (decision 8) |
| One node already renders as **many sibling meshes ordered per frame** (partition parts) | `render-order.ts`, `load-partition-group-node.ts` | the multi-mesh interleave machinery exists and is load-bearing in production |

---

## 2. Why not the alternatives

- **Route A — authoring (merge + partition).** Exact, zero viewer work, and stays the right answer when you control authoring. It is not a viewer fix, and its three costs are listed above. This spec does not remove it.
- **Route C — one globally merged draw** (Spark's `SparkRenderer`: traverse the scene, compile every splat into one buffer, sort globally, one instanced draw). Asymptotically the correct architecture and proven in three.js — but in Luxar it requires a texture-array/atlas redesign of the per-node element-texture pool against a background of continuous LOD-ladder commits, per-element indirection for ~20 per-node uniforms including textures, one unified blend state (so per-node `depthWrite` dies), and reworked picking, LOD fade and per-layer visibility. Multi-month, high blast radius, and it *removes* capabilities that exist today. See §8.2.
- **Route D — order-independent transparency** (weighted-blended, moment-based, hybrid). Sidesteps ordering entirely and is already an explicit deferral in `GSPLAT_DEPTH_SORTING_SPEC.md` §8. It is an *additional mode*, not a fix to `normal`/`volumetric`, and the exact variants (hybrid transparency's K-nearest core + OIT tail) need fragment-shader atomics — WebGPU only. Orthogonal and composable; not this document.

Route B is chosen because it is the only option that reduces the error *in the modes that already exist*, keeps every per-node material property, and is type- and mode-agnostic.

---

## 3. Target architecture

### 3.1 The mechanism

A node's committed permutation is *already* back-to-front. Therefore **any contiguous range of it is a depth interval**, for free.

Split each order-dependent node's permutation into `S` contiguous, equal-population ranges — a **depth shard** — and draw each shard as its own draw call. Then merge all `(node, shard)` intervals from all nodes globally by depth and hand out sequential `renderOrder` integers.

```
today:    [node A → 1] [node B → 2]                  1 integer per node
sharded:   A0 B0 B1 A1 A2 B2 A3 B3 …                 S integers per node, merged by depth
```

Cross-node ordering error falls from *whole-node extent* to *shard extent* — an S-fold reduction, converging to exact as `S → N`. The name in the literature is **depth bucketing** — e.g. MLABDB (multi-layer alpha blending over disjoint depth buckets), surveyed for transparent scientific geometry in Kern et al., *A Comparison of Rendering Techniques for Large 3D Line Sets with Transparency* (arXiv:1912.08485); "shard" is used here because "bucket" is already the depth-sort kernel's counting-sort vocabulary and "slab" already means an nD data slice in the monitor.

### 3.2 What does not change

No shader source. No blend state. No element texture. No sort kernel algorithm. No per-node material, uniform, colormap or opacity. No format. `aSortedIndex` values stay absolute storage slots, so picking's `vElementId` derivation is untouched. `S = 1` must be byte-identical to today, and is the default.

**The rendering work is invariant, provably.** Shards *partition* a node's elements, so every element is still drawn exactly once: total vertex invocations (`4N`), total fragments, blend/ROP traffic and overdraw are unchanged — nothing is double-blended and no overdraw is added. Total ordering bytes and upload bytes are unchanged (the same permutation, written in `S` pieces). And note what is *not* introduced: there is **no global union sort**. Every node keeps its own existing per-node sort, unchanged; the only new global step is a merge over `K × S` shard intervals — a few hundred items, `O(KS log KS)`, microseconds. This is a strictly cheaper proposition than Route C, which must sort the whole union every time any node moves. Everything Route B costs is therefore *per-draw overhead* — which is exactly, and only, what §7 measures.

### 3.3 Pinned design decisions

1. **Shards are equal-population contiguous ranges, not equal-depth intervals.** Population is fixed at `ceil(N/S)`, so per-shard attribute capacity is known, never overflows, and — critically — **`instanceCount` per shard is invariant across re-sorts** (it changes only when `N` changes, which re-registers the node anyway). Equal-*depth* shards would have data-dependent populations, forcing per-sort resizes and a spill path. Equal population also puts resolution where the elements are.
2. **Shard meshes are CHILDREN of the node's render mesh**, not siblings. Visibility, world matrix, dispose listeners and `traverse`-based enumeration then inherit for free, which is what keeps the "one mesh per node" assumption from having to be unwound everywhere (§5, Phase 1). The parent keeps its geometry and material and simply draws nothing (`instanceCount = 0`) while sharded.
3. **Shard geometry comes from the worker, in LOCAL space, and is transformed per frame.** `SortResult` gains a per-shard centroid + radius, accumulated **inside the kernel's existing pass-3 scatter** — that pass already knows each element's destination slot, hence its shard, so the spheres cost no extra pass over the data (`sort_splats_by_depth` is a three-pass counting sort: z + min/max, uint16 key + histogram, stable scatter). The main thread transforms those `S` spheres through the model-view every frame, exactly as `collectRenderOrderSlot` already does for a node's bounding sphere — so shard ordering tracks camera motion *between* re-sorts instead of going stale under the 3°/5% hysteresis.

   Two kernel behaviours the merge must tolerate rather than trust: behind-camera elements are all clamped into the **far bucket**, so shard 0 can carry the entire behind-camera set and its sphere is then not a tight depth proxy (harmless — those elements are off-screen — but shard 0's radius must not be read as meaningful); and a degenerate depth range (a single depth plane, ≤1 in-front element, everything behind the camera, or NaN centers) makes the kernel fall back to **identity ordering**, in which case that node's shards carry no depth meaning at all and it must be merged as one whole-node interval — i.e. exactly today's behaviour.
4. **The merge is a total order over shard intervals, subject to the containment partial order.** The existing group-major structure (order groups, then members within a group) must become a genuine global merge, because group-major by construction keeps a node's shards contiguous — which is precisely what sharding exists to break. The strict-containment override from PR #843 is preserved as a constraint on the merge, not discarded: a strict container's shards all precede the contained node's. (Its rationale: with one integer per node, a huge cloud containing a small embedded marker sorts *nearer* than the marker for roughly half of all camera orientations, and drawing the container last multiplies the marker's pixels by the container's whole transmittance — erasing it. Forcing container-first makes the embedded content composite on top, because under-attenuation is the lesser error against blinking out on orbit.)

   **Sharding plausibly subsumes that special case**, and this is worth testing once Phase 3 is live: the reason a single integer could not express containment is precisely the granularity problem shards remove — a container's shards can interleave with an embedded node's, which is the correct answer rather than a chosen-lesser-error one. Keep the override until measured redundant on the #843 fixture, then consider retiring it.
5. **View-space z is the merge key, so the merge needs no display-dimension mapping.** This is a strict improvement over the BSP path, which bails to a centroid heuristic when a stored split axis is not in the live `displayDims` (`bspAxisToComponent` → `null`). Shard ordering degrades gracefully exactly where BSP ordering currently gives up.
6. **Per-shard sorted-index attributes are separate fixed-size `InstancedBufferAttribute`s** (`Uint32Array` of length `ceil(N/S)`). Total ordering bytes and total upload bytes are unchanged — the same permutation, written in `S` pieces instead of one. The zero-copy variant (one `InstancedInterleavedBuffer`, `S` `InterleavedBufferAttribute` views at different offsets — three keys GL buffers by `attribute.data` and forwards `offset` verbatim, `three.module.js:1952-1956`) is **deferred behind a measured win and a three-arm spike**: it is exactly the "half-supported r184 vertex path" class that has burned this repo twice.
7. **Publish-on-flip stays the atomicity primitive.** The existing double buffer (`aSortedIndex`/`aSortedIndexB` + the `uSortedIndexSlot` uniform) generalises unchanged: all `S` inactive-slot attributes are written, then one uniform flip on the shared material publishes the whole set. A partially written inactive set is therefore never observable, which also means the L8 time-sliced apply keeps working across shards without a new invariant. (Note the vocabulary collision: L8's "chunked ordering apply" is a *time* slicing of the write; a shard is a *depth* slicing of the draw.)
8. **Each shard draw carries its own node's material, uniforms and blend state** — which is why mixed modes and mixed geometry types interleave *correctly* rather than approximately. This is concrete, not theoretical: the order-dependent set already spans **two different blend equations** (gsplat `normal` and every `volumetric` are premultiplied `(One, OneMinusSrcAlpha)`; Points/Lines `normal` is `(SrcAlpha, OneMinusSrcAlpha)`) and two `depthWrite` policies (Points `normal` forces it off unconditionally, #1002; Lines `normal` keeps the `opacity >= 0.99` predicate). A merged-draw architecture must unify all of that and lose the per-node distinctions; sharding never touches them, because GL state is per draw call. Route B's structural advantage over Route C, not an incidental one.
9. **`S = 1` is the default and the identity.** Sharding is opt-in per node via the policy in §4, so a scene with no overlapping order-dependent nodes pays literally nothing.

### 3.4 Mechanisms considered and rejected

Recorded so they are not re-proposed — each is something a three.js reader reaches for first:

- **A uniform depth-range test in the vertex shader**, collapsing out-of-range quads to zero area so one geometry can be drawn `S` times without splitting the attribute. Rejected: it costs `S ×` the vertex invocations (5M elements at `S = 16` is 320M vertex-shader runs per frame) to avoid an attribute split that is already free.
- **`BufferGeometry` groups** (`addGroup` + a material array). Rejected: groups slice the *index* buffer, and all of an object's groups render consecutively inside that object's single render item — so they cannot interleave with *another* object. `renderOrder` is per-`Object3D`; no finer granularity exists.
- **Ranged instanced draws off one shared ordering buffer** (a base-instance offset instead of `S` attributes). Rejected for now, not on merit: three's WebGL path exposes no first-instance offset (`WEBGL_multi_draw_instanced_base_vertex_base_instance` is unwrapped) while WebGPU has `firstInstance` natively, so this would be a backend-asymmetric fast path. The same win is available portably as decision 6's offset views, which is where it is deferred.
- **Merging the nodes at the data level.** That is Route A (§2) — an authoring change, not a viewer mechanism, and it forfeits the per-node identity this design exists to preserve.

---

## 4. Shard-count policy

The measured cost (§7) has two components: a small term linear in draw count, and a larger term that follows *node alternation* and saturates. So the primary knob is **which nodes interleave at all**, with a soft ceiling on interleaved draws as a backstop:

1. A node that overlaps no other order-dependent node's bounds gets `S = 1`. Today's behaviour, zero cost. Bounds are already computed per frame by the render-order pass, and the Python compiler already emits the co-visible-overlap diagnostic at authoring time.
2. A `kind=partition` node already has **exact** BSP ordering internally; sharding only buys ordering against *foreign* nodes. Shard only the parts whose bounds overlap a foreign order-dependent node, and at low `S`.
3. `S` should follow the geometry rather than a constant: the shard count a pair of nodes needs is roughly *(their overlap extent along the view axis) ÷ (the ordering error you accept)*, so a deeply interpenetrating pair warrants a high `S` and a pair that merely grazes warrants 2.
4. Shards are only needed where the depth intervals actually interleave. Outside the overlap interval one coarse shard suffices, so a nominal `S = 16` typically resolves to 3–5 real shards.
5. **Gate coarsely, then be generous with `S`.** §7 measures the cost as *sublinear* in draw count at realistic element counts: at 2M splats, going from 128 to 512 interleaved draws added only ~1 ms, while interleaving *at all* cost +5.5 ms. The dominant decision is therefore **which nodes interleave**, not how finely they are sharded — so spend the effort on rules 1–4 and let `S = 16` stand once a node qualifies. Keep a soft ceiling of **512 interleaved draws** to bound the linear per-draw term (which does dominate at low element counts, where it is also affordable), and drop `S` on the least-overlapping nodes first when it is exceeded.
6. Config lives in a new `config/sections/depth-shards/` beside `depth-sort/`, with `?depthShards=N` to pin and `?depthShards=0` to disable the subsystem outright (the `?depthSort=0` precedent — a determinism escape hatch must pin output, not merely stop scheduling).

**Real scenes land near the cheap end by construction.** The global merge emits long same-node runs wherever nodes are depth-separated and alternates finely only where they genuinely interpenetrate — so the cost scales with actual overlap, which is the same quantity that determines the benefit.

---

## 5. Phases

Each phase leaves `main` shippable; `S = 1` keeps every phase before 3 visually inert.

**Phase 0 — per-shard geometry from the worker.** `SortResult` gains `shardCount`, `shardCentroids` (`Float32Array[3S]`, local space) and `shardRadii` (`Float32Array[S]`); the kernel computes them in the pass it already makes over the sorted output. WASM + TS twin, so the parity suite covers it. Ships inert (nothing consumes the new fields).

**Phase 1 — shard meshes (the mechanical phase).** `attachElementStorage` grows an `S`-way sorted-index set; `writeSortedIndex*` writes per shard; the node's render mesh gains `S − 1` child meshes sharing its material and base-quad attributes. Behavior-preserving at `S = 1`, so review is a pure refactor review. The real work is the checklist of surfaces that assume one drawing mesh per node — enumerate and adapt each: picking registration (`picking-system.registerNode` / `userData.pickNode` / `invalidateRenderObjectFor`'s eager re-point), LayersPanel row/material state, monitor + `scene-stats` element counts, LOD fade and pool eviction, `releaseDepthSortNode` on dispose, and per-node visibility toggles. Picking itself does **not** shard: its pass is order-independent and runs on the parallel pick mesh.

**Phase 2 — the global merge.** `render-order.ts` slots become per-shard; group-major ordering becomes a total merge over shard intervals under the containment partial order; `assignGlobalRenderOrder` writes `1..M` over shards instead of meshes. A golden test pins that `S = 1` reproduces today's integers exactly.

**Phase 3 — the policy (the feature).** Overlap detection between order-dependent nodes, the draw budget and per-node `S` allocation, config section, URL escape hatch, monitor line (shard count + interleaved draw count).

**Phase 4 — optional, measured-win only.** The shared-buffer offset-view attribute layout (decision 6), behind its own three-arm spike.

---

## 6. Testing and ship gates

| Phase | Hard gate before merge |
| --- | --- |
| 0 | WASM/TS parity on the new outputs; shard spheres bound their own elements (property test) |
| 1 | Pixel-identical E2E at `S = 1`; shard slicing is a **partition** of the permutation (property test); `instanceCount` sum equals the node count; no leak on dispose (the grow-leak probe pattern) |
| 2 | Golden: `S = 1` reproduces today's `renderOrder` integers; containment constraint preserved (the #843 embedded-marker fixture); merge total-order sanity under random shard intervals |
| 3 | Non-vacuous E2E on a **two-node** overlapping fixture — the parent spec's `test_gsplats_normal_overlap_reversed` (`tests/fixtures/generate_test_data.py`) is the single-node front-to-back case and needs a two-node sibling — **fail-first verified by pinning `S = 1`**; draw-budget assertion; `?depthShards=0` determinism |
| 4 | Measured upload/frame win on all three renderer surfaces, else drop |

Cross-cutting requirements: the perf gate records interleaved draw count and frame time against the §7 budget; **all three renderer arms** (WebGL, WebGPU-native, `webgpu-force-webgl`) are required for Phases 1–2 — §7 measured WebGL only, and per-draw cost is exactly the kind of quantity that differs across backends; full E2E before each merge per repo policy; `render-order.ts` and `element-storage.ts` README updates ride their phase.

---

## 7. Measured cost (probe, 2026-08-30/31)

Rig: **Apple M4 Max via ANGLE Metal**, headed Chromium (Playwright), 1280×800, `dpr=1`, WebGL backend, three r184, synthetic gsplat/points nodes injected through `__luxarDebug.injectSyntheticScene` — real node-factory, real materials, real element textures. Arms per node count `K`: `S=1` baseline, `S=16` node-major (splitting *without* node alternation), `S=16` interleaved (this design), swept over `K ∈ {1, 2, 8, 32}` nodes at a **constant** total element count. The probe needs **no viewer source change** — it drives `__luxarDebug` only, so it can be re-run on any branch or backend. Raw data and script: `delme/draw-split-probe/`.

**Both geometry types were measured and agree within noise** (`gsplats` and `points`, 20k arm: 0.40–0.68 µs per added draw node-major, 0.47–2.18 interleaved; 512-draw all-in +1.6/+2.0 ms for gsplats vs +3.1/+3.5 ms for points). That is the direct evidence for the type-agnostic claim in §1 and §3.1 — the mechanism does not care which of the three types it is splitting.

**Validation first.** At `K = 1` the framebuffer hash is **identical across all three arms**, at both 20k and 2M elements — contiguous shards of the sorted permutation are exactly order-preserving, as claimed in §3.1. At `K ≥ 2` the three arms give **three distinct images with the same lit coverage** — §1's reproduction of the defect.

**CPU submission per added draw** — independent of element count (same value at 20k and 2M):

| Material situation | µs per added draw |
| --- | --- |
| Shared material (one node, S=16) | 0.40–0.53 |
| Distinct materials, identical uniform *values* | 0.47–1.15 |
| Distinct materials, perturbed uniform values (**realistic**) | **1.24–1.40** |

The third row is the one to design against: three.js caches uniform values per program and skips unchanged `gl.uniform*` calls, so identically-valued clones under-measure a real multi-node scene. Forcing cache misses roughly doubles the cost — and it is still ~1 µs.

**All-in frame cost at 2M gsplats** (`readPixels` barrier; base frame 26–33 ms, GPU-bound):

| Nodes | Draws | node-major Δ | interleaved Δ |
| --- | --- | --- | --- |
| 1 | 1 → 16 | −0.6 ms (noise) | −1.5 ms (noise) |
| 8 | 8 → 128 | **+1.7 ms** | **+5.5 ms** |
| 32 | 32 → 512 | **+3.9 ms** | **+6.4 ms** |

**The dominant term is not JS.** CPU submission accounts for ~0.5 ms at 512 draws; the rest is GPU/driver-side. The *interleaving premium* (interleaved − node-major) is **+3.8 ms at 128 draws and +2.5 ms at 512** — i.e. splitting alone costs +1.7 to +3.9 ms and alternating nodes on top of it roughly doubles that, most plausibly element-texture locality plus pipeline state: at 2M splats across 8 nodes each element texture holds 250k splats × 4 RGBA32F texels × 16 B = **16 MB**, and the interleaved arm alternates between all 8 of them on every draw while the node-major arm keeps one hot for 16 consecutive draws. This effect was **not predicted** by the pre-probe analysis, which expected 10–20 µs/draw of three.js uniform-upload cost; that estimate was ~10× too pessimistic on CPU and blind to the GPU-side term.

**The low-element-count regime behaves differently, and it is the one that looks alarming in ratios.** At 20k elements the base frame is 0.3–0.5 ms, so 512 draws (+1.6 to +2.0 ms) makes a trivially cheap frame 5–6× more expensive — but in *absolute* terms it is still only ~2 ms, and a 20k-element scene has the headroom. Per-draw cost is element-count-independent, so it is always worst as a *ratio* exactly where the frame was cheapest.

**The cost is sublinear in draw count at scale.** At 2M the interleaved arm cost +5.5 ms at 128 draws and +6.4 ms at 512 — roughly 20% of a 26–33 ms frame either way, and only ~1 ms apart despite 4× the draws. At 20k, by contrast, the same step scales ~linearly (+0.4 ms → +2.0 ms), i.e. the pure per-draw term is real but small, and the large term at scale is a saturating state/locality effect. Hence §4.5: gate *which* nodes interleave, and stop treating `S` as the expensive knob.

Two methodology traps, recorded so the probe is reproducible: Chrome coarsens `performance.now()` to **100 µs**, so a single render quantizes to zero — time a batch per clock read and use the *minimum* (a tight render loop hits GPU backpressure, which inflates the median toward the GPU frame time). And `gl.finish()` does **not** reliably block here; a 1×1 `readPixels` does. A third artifact cost one run: a late-landing sort flipped `uSortedIndexSlot` mid-measurement, so the arms read different slots and diverged in *pixels* — timing arms therefore run with `?depthSort=0`.

---

## 8. Explicit deferrals

1. **Mesh.** Mesh sorting permutes `geometry.index` rather than `aSortedIndex` (`MESH_NODE_SPEC.md` §6.3), so a shard is a contiguous range of the index buffer — a valid triangle subset, i.e. mechanically easier than the instanced path. Deferred only to keep Phase 1's blast radius to the three instanced types.
2. **Route C (one merged draw).** The endgame, and Spark demonstrates it works in three.js: `SparkRenderer` traverses the scene graph, compiles every splat from every `SplatMesh` into one `PackedSplats` buffer (16 B/splat), bucket-sorts it in a worker, and issues a single instanced draw — accepting **one frame of sort lag**, and replacing per-node uniforms with a *per-splat programmable GPU data pipeline* (transforms, recolour, SH, edits baked per splat) rather than per-object state.
   Sizing note for whoever picks this up: **a per-type merge is only marginally cheaper than a cross-type one.** The expensive parts — global element storage (texture array or atlas), per-element indirection for the ~20 per-node uniforms, and invalidating the global buffer on every LOD commit — are shared regardless; the cross-type extra is one shader branch over three texel layouts (gsplats 4 texels/element, points 3, lines 6), which is the easy part. Doing "just gsplats" first buys perhaps 15%.
   Kept compatible either way: if the interleaving premium ever binds, the fix is a shared element-texture atlas or `sampler2DArray` — which *is* Route C's substrate. Sharding does not foreclose it.
3. **Order-independent transparency.** Unchanged deferral from `GSPLAT_DEPTH_SORTING_SPEC.md` §8; the WebGPU-only exact variants (hybrid transparency, per-pixel k-buffer) are the long-term ceiling.
4. **Per-pixel ordering.** Out of reach of any per-element global sort by construction — a single per-element depth is what causes popping under rotation (StopThePop), and the sorting-free stochastic alternative (StochasticSplats) trades it for sample noise. Both are outside this design's error class.
5. **Extending the merge to commutative-mode nodes.** `additive` and `luminous` sit outside the coordinator's sorted set today and keep `renderOrder` 0, so they draw *before* the globally-farthest sorted mesh; `additive` additionally has `depthTest: false` and paints through everything. Interleaving them into the shard merge would fix `additive`-vs-`normal` layering as a side effect (additive is commutative with itself but **not** with `over`), and the machinery would support it unchanged. Deliberately out of scope: it changes the appearance of scenes that render acceptably today, which is a separate decision from fixing the ones that do not.
6. **`opaque` at fractional opacity.** Still order-dependent, still documented rather than fixed.

---

## 9. Risk register (ranked)

1. **Interleaving costs ~20% of frame time at 2M, and it is GPU-side, scene-dependent, and nearly independent of `S` (measured, §7).** Splitting alone costs +1.7 to +3.9 ms; interleaving costs +5.5 to +6.4 ms; the premium for alternation alone is +2.5 to +3.8 ms — on one GPU, one backend, one synthetic scene shape. Because the term saturates rather than scaling with draws, it cannot be tuned away with a smaller `S`; it can only be *avoided* by not interleaving a node at all. Mitigation: §4's coarse gating (rules 1–4). Revisit criterion: if a real scene pays this on nodes that genuinely need ordering, the fix is a shared element-texture array or `sampler2DArray` (Route C's substrate, §8.2), not a smaller shard count.
2. **Unmeasured on WebGPU.** §7 covers WebGL/ANGLE-Metal only. Per-draw cost, render-object count and bind-group churn differ on the WebGPU backend, and `S` multiplies all three. Mitigation: Phase 1 and 2 gates require all three arms; an NVIDIA cross-check (obsidian) is queued for the same reason — its driver's per-draw and texture-locality behaviour is not ANGLE's.
3. **"One drawing mesh per node" is load-bearing in more places than the ordering code.** This is where the work actually is, not in the merge math. Mitigation: shards as *children* (decision 2) so inheritance covers visibility/matrix/dispose/traversal; the Phase 1 checklist enumerates the rest explicitly; `S = 1` default keeps the refactor reviewable in isolation.
4. **Ordering staleness between re-sorts now affects cross-node order too.** Under the 3° / 5%-of-bounding-radius hysteresis a node's permutation — and so its shard boundaries — lags the camera. Mitigation: decision 3 transforms per-shard spheres every frame, so the *merge* is always fresh even when the *within-shard* order is not. Residual: shard boundaries themselves are stale, bounded by the same hysteresis the subsystem already accepts.
5. **`depthWrite` interaction.** Lines `normal` writes depth at opacity ≥ 0.99 (`normalModeDepthWrite`); interleaving such a node mid-sequence lets it depth-reject later shards behind it. This is *more* correct than today, but it is a visible change in a configuration that currently renders differently — needs a fixture and a CHANGELOG note, not a silent flip.
6. **Publish atomicity across S attributes.** A flip must publish all shards or none (decision 7). The failure mode is a one-frame mixed-generation composite, which is exactly the class the generation contract was built for; enforce in the single apply function, and property-test the state machine with a mock worker as Phase 2 of the parent spec did.
7. **The error is bounded, not zero.** Two elements from different nodes inside the same depth interval remain arbitrarily ordered. Honest ceiling: "no visible cross-node artefact on orbit", not "exact". Do not let the diagnostic in `finalize` stop warning about overlapping sorted nodes just because their ordering improved.

---

## 10. Provenance

No GitHub issue exists for this yet (an open-issue sweep on 2026-08-31 found none matching depth/order/overlap/compositing). The limitation is recorded in `GSPLAT_DEPTH_SORTING_SPEC.md` §1 as an authoring rule, in `render-order.ts`'s own header and README as a documented approximation, and as the deferred "cross-type interleaving" item in `TODO.md`'s `[POST]` residue ledger. The authoring-rule diagnostics are also known to under-serve this case: the compiler's overlap warning proposes making a node `additive` or separating bounds — both appearance/science changes — and omits merge-and-partition (#2222), and an unpartitioned merge can silently truncate past the per-node clamp (#2221).

Prior art in-repo: PR #843 (containment override), PR #591 (`render-order.ts` extraction), PR #596 (renderOrder pass survives a dead SortWorker), PR #658 (volumetric phase 4 — the third geometry type joining the sorted set with zero coordinator change).

The NVIDIA cross-check named in risk 2 is **pending on infrastructure, not skipped**: the intended host (obsidian, RTX PRO 6000) has had no loaded NVIDIA kernel module since a kernel bump to `6.8.0-138-generic` left the DKMS module unbuilt, and its 16-core load average of ~20 from the agent fleet would invalidate µs-scale timing regardless. Re-run `delme/draw-split-probe/` there once the driver is restored.

---

## 11. References

External work this design leans on or explicitly declines:

- Kern, Neuhauser, Maack, Han, Usher, Westermann, *A Comparison of Rendering Techniques for Large 3D Line Sets with Transparency* — [arXiv:1912.08485](https://arxiv.org/abs/1912.08485). The closest domain match (transparent scientific geometry); source of the depth-bucketing framing.
- *Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency* — [arXiv:2410.08129](https://arxiv.org/abs/2410.08129). K=16 exactly-sorted per-pixel core + order-independent tail; **skips global presorting entirely** and still reports 457 vs 317 FPS against 3DGS at equal-or-better quality. The strongest long-term argument against *any* global sort — and WebGPU-only, since it needs fragment-shader atomics that WebGL2 does not have.
- Radl, Steiner et al., *StopThePop: Sorted Gaussian Splatting for View-Consistent Real-Time Rendering* — [arXiv:2402.00525](https://arxiv.org/abs/2402.00525). Hierarchical approximate per-pixel sort for ~4% overhead; defines the popping error class this design does not address.
- *StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting* — [arXiv:2503.24366](https://arxiv.org/abs/2503.24366). The sorting-free extreme.
- McGuire & Bavoil, *Weighted Blended Order-Independent Transparency* (JCGT 2013); Münstermann et al., *Moment-Based Order-Independent Transparency* (2018) — [project page](https://cg.ivd.kit.edu/mboit.php); [nvpro-samples/vk_order_independent_transparency](https://github.com/nvpro-samples/vk_order_independent_transparency) for a seven-technique side-by-side.
- Spark — [system design](https://sparkjs.dev/docs/system-design/), [PackedSplats](https://sparkjs.dev/docs/packed-splats/). The production precedent for Route C in three.js.
