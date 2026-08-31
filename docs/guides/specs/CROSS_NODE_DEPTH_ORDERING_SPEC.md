# Cross-Node Depth Ordering — Depth-Shard Interleaving (Route B)

> **Status**: **ARCHIVED, 2026-08-31 — not landing.** Phases S, 0, 1, 2 and 3 are implemented and green on branch `feat/cross-node-depth-shards` (holding PR #2407, closed unmerged, branch retained). The mechanism works and is measured (§7.3: 84% of the cross-node ordering error removed at 64 shards for no measurable frame cost, 93% at 256 for ~1-2 ms). It was archived anyway, on the judgement that the visible payoff on real data does not justify the permanent complexity — see §12, which is the section to read first and the only one written with the outcome known.
> **Scope**: Viewer-only. No `.luxar.zarr` / `.gsplats.zarr` format change, no Python-side change, **no shader change**.
> **Goal**: correct back-to-front compositing *between* two or more overlapping order-dependent nodes — of any instanced geometry type, in any mix of order-dependent blending modes — to a bounded, tunable ordering error.
> **Non-goals**: per-pixel ordering (the StopThePop error class — a single per-element depth cannot fix it, and no global sort can); order-independent transparency; one globally merged draw (Spark's architecture — §8.2); Mesh (§8.1).

Related reading: `GSPLAT_DEPTH_SORTING_SPEC.md` (the within-node subsystem this builds on — §1's "Overlapping-node authoring rule" states the limitation this document removes), `VOLUMETRIC_BLENDING_SPEC.md`, `src/rendering/depth-sort-coordinator/README.md`.

Probe sources + raw results (throwaway, not repo artifacts): `delme/draw-split-probe/` (per-draw cost, §7) and `delme/shard-attr-spike/` (attribute layout, §7.1).

**three.js version**: the pin is `~0.185.1` = **r185** (`packages/luxar-viewer/package.json`). Every three.js line citation below was read from that installed build.

---

## 1. Problem statement (facts on the ground)

- **A draw call is atomic in draw order.** Within one call the element order is fully controllable (`aSortedIndex` permutes draw slot → storage slot). Between two calls there is no interleaving: the whole of one draw composites before the whole of the next.
- **three.js's only inter-object lever is `renderOrder`**, and it is compared *before* `z` — verified in the pinned r185 on both backends (`three.module.js:8142`, `three.webgpu.nodes.js:33131`, `reversePainterSortStable`, whose key order is `groupOrder` → `renderOrder` → `z` → `id`). One integer per object is the entire expressive budget.
- **`groupOrder` outranks `renderOrder`, so it bounds the merge domain.** `projectObject` sets `groupOrder = object.renderOrder` at every `isGroup` ancestor (`three.module.js:17833-17835`), and it is the *first* sort key. Cross-node ordering therefore cannot cross a `groupOrder` boundary. In the standalone viewer every Group keeps the default 0, so the domain is the whole scene; in embedded mode `LuxarLayer` stamps ONE value on every Group it owns (`core/layer/luxar-layer.ts:717-725`, default 10) precisely so a layer is internally comparable. Two `LuxarLayer`s with different `renderOrder` cannot interleave — correctly, since that is the embedder's declared stacking. Not a defect; a documented scope limit.
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
2. **Shard meshes are CHILDREN of the node's render mesh, and the parent IS shard 0.** Children (not siblings) so visibility, world matrix, dispose listeners and `traverse`-based enumeration inherit for free, which is what keeps the "one mesh per node" assumption from having to be unwound everywhere (§5, Phase 1).

   The parent **draws shard 0** — it keeps its geometry, material, element texture and every `userData` stamp, and only its `instanceCount` narrows from `N` to the shard size. An earlier revision of this spec had the parent draw *nothing* (`instanceCount = 0`); that is wrong, and four production surfaces say so:
   - `scene/lod-fade.ts:224-226` is literally `if (obj.material) visit(root); else root.traverse(visit)` — "a mesh that has a material IS the whole node". A parent that keeps its material *and* gains drawing children is the one shape this branch cannot express, so cross-fade and streaming energy compensation would silently skip every shard.
   - `scene/scene-manager/camera/camera-framing.ts:163` returns early on `instanceCount <= 0`, so a zero-instance parent contributes nothing to auto-framing bounds.
   - `rendering/webgl-blend-warmup.ts:273-292` gates on the node's non-zero element stamp, so it would warm the parent's program while the meshes that actually draw never get warmed — reinstating the first-frame compile hitch.
   - `data/stats/scene-stats.ts:73-94` sums `instanceCount`.

   Two invariants follow, and both are load-bearing rather than stylistic. **Shards share the parent's material OBJECT and never a clone** — that is what makes `lod-fade` and the LayersPanel correct, since both mutate the one shared material. And **shards carry `userData.depthShardOf` and NO `nodeType` / `visible*Count` stamps**, so every stats, monitor, warmup and picking traversal skips them by construction instead of double-counting.
3. **Shard geometry comes from the worker as per-shard AABBs, in LOCAL space, and is transformed per frame.** `SortResult` gains a per-shard min/max box, accumulated **inside the kernel's existing pass-3 scatter** — that pass is the only point that knows both an element's destination slot (hence its shard) and its source index, so the boxes cost no extra pass over the data (`sort_splats_by_depth` is a three-pass counting sort: z + min/max, uint16 key + histogram, stable scatter — `wasm/rust/src/depth_sort.rs:168-173`). The main thread derives centroid + radius from each box and transforms them through the model-view every frame, exactly as `collectRenderOrderSlot` already does for a node's bounding sphere — so shard ordering tracks camera motion *between* re-sorts instead of going stale under the 3°/5% hysteresis.

   **AABBs rather than centroid+radius, deliberately.** Min/max over f32 requires no rounding, so WASM/TS parity is exact *by construction* — the new kernel code needs none of the `Math.fround` discipline the key math carries (`wasm/typescript/depth-sort.ts:58-71`, where parity is exact-permutation). Centroid and radius are derived main-thread, where there is no parity contract.

   **The merge key is the projected shard CENTROID's view-z; the radius feeds only the containment test.** A thin depth slab's bounding sphere is wide in x/y, so its radius badly overstates the slab's depth thickness and is not a usable interval bound. Decision 4 is what makes key imprecision harmless.

   Three kernel behaviours the merge must tolerate rather than trust:
   - Behind-camera elements are all clamped into the **far bucket** (key 0 — `depth_sort.rs:148`), so shard 0 can carry the entire behind-camera set and its box is then not a tight depth proxy. Harmless, since those elements are off-screen, but shard 0's extent must not be read as meaningful.
   - A **NaN center keys to the NEAR bucket 65535**, not the far one (`f32::min(NaN, 65535.0) == 65535`; pinned by `depth_sort.rs`'s `test_nan_center_keys_to_near_bucket`), so it lands in the LAST shard and would poison that box. Guard it the way `render-order.ts:554-562` already guards a non-finite bounding sphere: a non-finite shard box yields the `radius = -1` sentinel.
   - A degenerate depth range (a single depth plane, ≤1 in-front element, everything behind the camera, or NaN throughout) makes the kernel fall back to **identity ordering**, in which case that node's shards carry no depth meaning and it must merge as one whole-node interval — i.e. exactly today's behaviour. **The signal already exists**: `sort_splats_by_depth` returns `0` on that path (`depth_sort.rs:138`) and `sortNode` currently discards the return value (`workers/sort-worker/sorting.ts:95`). Surfacing it as `shardCount = 0` gives the merge this case for free.
4. **The merge is a k-way merge of MONOTONE STREAMS, not a flat sort of shard intervals.** The existing group-major structure (order groups, then members within a group) must become a genuine global merge, because group-major by construction keeps a node's shards contiguous — which is precisely what sharding exists to break. But a *flat* sort by depth key would be worse than group-major, for two concrete reasons:
   - A partition wrapper's parts are ordered **exactly** by Fuchs–Kedem–Naylor BSP rank (`render-order.ts:97`), valid from any camera pose including inside the volume. A flat key sort would discard that exactness in favour of a centroid heuristic.
   - `render-order.ts:660-663` sorts a fully-ranked group by `partRank` with **no tiebreak**, so all shards of one part would compare equal and draw in arbitrary order — buying nothing at all.

   Instead: **each order group contributes a stream of shard intervals already in its own correct order, and the assignment is a k-way merge that may only consume stream heads.** Two properties follow, and they are the reason to prefer this shape:
   - Within-group exactness is preserved *by construction* — no BSP rank is ever reordered, because a stream is only ever consumed front-to-back.
   - The merge becomes **robust to depth-key noise**: an imprecise or stale key can change which stream is picked next, but can never mis-order two shards of the *same* node. That is what makes decision 3's centroid key sufficient.

   Cost is `O(KS log K)` over a few hundred items — microseconds.

   The strict-containment override from PR #843 is preserved as a constraint on the merge, not discarded: a contained group's stream is **blocked until its container's stream is exhausted**. That is Kahn's algorithm over streams, emitting the farthest ready head first — structurally the same priority-topological shape as today's `orderGroupsWithContainment` (`render-order.ts:430-504`), including its deliberate double-bound edge rule (an edge must hold under BOTH the Ritter and the centroid+max-reach sphere, `:465-486`, so the tighter bound can only ever REMOVE false edges). Its rationale: with one integer per node, a huge cloud containing a small embedded marker sorts *nearer* than the marker for roughly half of all camera orientations, and drawing the container last multiplies the marker's pixels by the container's whole transmittance — erasing it. Forcing container-first makes the embedded content composite on top, because under-attenuation is the lesser error against blinking out on orbit.

   **Sharding plausibly subsumes that special case**, and this is worth testing once Phase 3 is live: the reason a single integer could not express containment is precisely the granularity problem shards remove — a container's shards can interleave with an embedded node's, which is the correct answer rather than a chosen-lesser-error one. Keep the override until measured redundant on the #843 fixture, then consider retiring it.
5. **View-space z is the merge key, so the merge needs no display-dimension mapping.** This is a strict improvement over the BSP path, which bails to a centroid heuristic when a stored split axis is not in the live `displayDims` (`bspAxisToComponent` → `null`). Shard ordering degrades gracefully exactly where BSP ordering currently gives up.
6. **Per-shard sorted-index attributes are `InstancedBufferAttribute`s over SUBARRAY VIEWS of the node's one contiguous ordering array** — `new InstancedBufferAttribute(parentArray.subarray(k*size, (k+1)*size), 1)`. Chosen by measurement on all three renderer arms (§7.1), over the two alternatives this spec previously considered:

   | Layout | CPU arrays | GPU buffers | Verdict |
   | --- | --- | --- | --- |
   | separate copied arrays | `S` (2× memory) | `S` | works, but every one of the six existing writers becomes S-way |
   | **subarray views** | **1** | `S` | **green on all three arms** |
   | interleaved offset views | 1 | 1 | **RED on WebGPU-native** — unusable |

   Why this matters far beyond byte counts: one contiguous CPU array means **the six existing sorted-index writers keep writing exactly as they do today** (`writeSortedIndexIdentity`, `…IdentityRange`, `…OrderingLive`, `repairSortedIndexForCount`, `writeSortedIndexOrdering`, and the only function that writes streamed permutation bytes, `applyNextSortedIndexChunk`), because a subarray shares memory with its parent. One geometry still owns the chunked-apply stream, the back-pressure latch and the draw-acknowledgement hook. And the double buffer still publishes with ONE uniform flip, because `uSortedIndexSlot` lives on the shared **material**, not per geometry.

   The one addition it does require: after a write, each overlapping shard attribute needs its own `addUpdateRange` + `needsUpdate`, since each has its own GPU buffer. That fans out at a single chokepoint rather than spreading through the writers.

   Note on upload bytes: a naive version where the parent keeps the FULL-length attribute while also drawing only shard 0 would upload ≈`2N` ordering entries per re-sort (the parent's whole array plus every shard's slice). Give the parent a shard-0 **view** as well, and keep the full-length array as the CPU-side staging buffer behind an accessor in `element-storage.ts`, so total upload stays exactly `N` — the 13×-better figure the parent spec's architecture argument rests on. At `S = 1` this is byte-for-byte today's layout.
7. **Publish-on-flip stays the atomicity primitive.** The existing double buffer (`aSortedIndex`/`aSortedIndexB` + the `uSortedIndexSlot` uniform) generalises unchanged, and decision 6's layout is what keeps it cheap: the inactive slot is ONE contiguous array that all `S` views alias, so the write is a single write, and `uSortedIndexSlot` is ONE uniform on the ONE shared material, so the publish is a single flip. What the shards add is only that every view's GPU buffer must be dirtied *before* that flip, never after. A partially published set is therefore never observable, and the L8 time-sliced apply keeps working across shards without a new invariant. (Note the vocabulary collision: L8's "chunked ordering apply" is a *time* slicing of the write; a shard is a *depth* slicing of the draw.)
8. **Each shard draw carries its own node's material, uniforms and blend state** — which is why mixed modes and mixed geometry types interleave *correctly* rather than approximately. This is concrete, not theoretical: the order-dependent set already spans **two different blend equations** (gsplat `normal` and every `volumetric` are premultiplied `(One, OneMinusSrcAlpha)`; Points/Lines `normal` is `(SrcAlpha, OneMinusSrcAlpha)`) and two `depthWrite` policies (Points `normal` forces it off unconditionally, #1002; Lines `normal` keeps the `opacity >= 0.99` predicate). A merged-draw architecture must unify all of that and lose the per-node distinctions; sharding never touches them, because GL state is per draw call. Route B's structural advantage over Route C, not an incidental one.
9. **`S = 1` is the default and the identity.** Sharding is opt-in per node via the policy in §4, so a scene with no overlapping order-dependent nodes pays literally nothing.

### 3.4 Mechanisms considered and rejected

Recorded so they are not re-proposed — each is something a three.js reader reaches for first:

- **A uniform depth-range test in the vertex shader**, collapsing out-of-range quads to zero area so one geometry can be drawn `S` times without splitting the attribute. Rejected: it costs `S ×` the vertex invocations (5M elements at `S = 16` is 320M vertex-shader runs per frame) to avoid an attribute split that is already free.
- **`BufferGeometry` groups** (`addGroup` + a material array). Rejected: groups slice the *index* buffer, and all of an object's groups render consecutively inside that object's single render item — so they cannot interleave with *another* object. `renderOrder` is per-`Object3D`; no finer granularity exists.
- **One `InstancedInterleavedBuffer` with `S` `InterleavedBufferAttribute` views at different offsets** (the zero-copy, one-GPU-buffer ideal). **Rejected on measurement — it is RED on WebGPU-native (§7.1).** WebGPU requires an attribute's `offset + format size <= arrayStride`; here `arrayStride = data.stride × 4 = 4 bytes`, so any shard past the first is invalid. Chromium's own validation message names the only fix — *"Offsets larger than the maximum vertex buffer stride are accommodated by setting buffer offsets when calling setVertexBuffer"* — and three never passes one: `passEncoderGPU.setVertexBuffer( i, buffer )`, `three.webgpu.js:85502`, no offset argument. It works fine on both WebGL arms, where `gl.vertexAttribIPointer` takes an arbitrary byte start (`three.module.js:1894-1898`; the interleaved branch forwards `offset × bytesPerElement` verbatim and computes `integer` from the array type, `:1948-1996`), which is exactly what makes it a **backend-asymmetric trap** rather than an optimisation: it renders correctly on two arms out of three and silently drops geometry on the third. Fixing it would mean patching three.
- **Ranged instanced draws off one shared ordering buffer** (a base-instance offset instead of `S` attributes). Rejected: three's WebGL path exposes no first-instance offset (`WEBGL_multi_draw_instanced_base_vertex_base_instance` is unwrapped) while WebGPU has `firstInstance` natively — the same backend asymmetry as above, from the other direction.
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

Each phase leaves `main` shippable; `S = 1` keeps every phase before 3 visually inert. Phases land on the draft holding PR #2407, which merges once as a merge commit so each phase stays bisectable.

**Phase S — attribute-layout spike. DONE (§7.1).** Verdict: subarray views, green on all three arms; interleaved offset views rejected as backend-asymmetric. This ran *first*, ahead of Phase 0, because its outcome decides how wide Phase 1 is — the layout question is a route-viability gate, not an optimisation.

**Phase 0 — per-shard geometry from the worker.** `SortResult` gains `shardCount`, `shardBoundsMin` and `shardBoundsMax` (`Float32Array[3S]` each, local space); the kernel accumulates them inside its existing pass-3 scatter. WASM + TS twin, so the parity suite covers it — and parity is exact by construction (decision 3). `SortParams` gains `shardCount`, defaulting to 1, which makes `S = 1` a non-vacuous self-check: the single box must agree with the geometry's own bounding sphere. Ships inert (nothing consumes the new fields).

**Phase 1 — shard meshes (the mechanical phase).** A new `rendering/depth-sort-coordinator/depth-shards.ts` — the companion-directory pattern beside `render-order.ts` / `triangle-ordering.ts` — owns shard create / resize / teardown, driven from `noteDepthSortCommit` and `releaseDepthSortNode`.

**Shards are owned by the coordinator, NOT by the scene graph.** This is what bounds the blast radius. A shard is a rendering-time detail of a sorted node, so it gets no scene path, no `attrs`, no loader, no commit, no pool entry and no pick registration — and therefore every path that resolves a node by `rootGroup.getObjectByName(path)` (all of `commit-*-geometry.ts`, `run-loader-updates.ts`, each geometry handler and its `lod-refinement.ts`, `geometry-descriptors.ts`, `retry.ts`) is untouched, with `releaseDepthSortNode` as the single teardown point. This is deliberately *not* the partition-part precedent: parts are real scene-graph nodes with their own paths, loaders, commits and pick registrations, and a shard can never be one.

Behavior-preserving at `S = 1`, so review is a pure refactor review. The real work is the checklist of surfaces that assume one drawing mesh per node:

| Surface | Where | What it needs |
| --- | --- | --- |
| **Picking** — the highest-consequence case, and it fails *silently* | `picking/picking-system.ts:752-757` and `commit/invalidate-render-object.ts:135` both force `pick.geometry = main.geometry` | with the parent narrowed to shard 0, only shard 0 would be pickable and hover just stops working with no error. Give the pick node its **own** `InstancedBufferGeometry` sharing every attribute at `instanceCount = N`. Picking does not shard — its pass is order-independent |
| Element counts | `data/stats/scene-stats.ts:73-94`, `core/app/debug/debug-state.ts:334-438` | read the committed `userData` stamp rather than `instanceCount`; `monitor/visible-counts.ts:70` (`readVisibleElementCount`) already does this correctly and is the thing to reuse |
| Draw-order rows | `monitor/draw-order-provider.ts:56` keys by `object.name`, last write wins — a shard would clobber its node's row | skip shard children, or aggregate them under the parent path with a shard index |
| Disposal | `render-pipeline/scene-disposal.ts:40-64`, `core/layer/luxar-layer.ts:732-739` | skip `userData.depthShardOf` geometries: they share attribute objects, so disposing one frees the parent's GPU buffers, and the shared material would otherwise be disposed `N+1` times |
| Material identity | `layer-apply.ts:107-113` clone-on-first-use **replaces** `parent.material` | re-point shards from an idempotent per-frame `syncShardMaterials`, beside the `syncSortedIndexSlot` re-assert that `pumpChunkedOrderingApplies` already performs every frame (`coordinator.ts:1680`) — a pointer compare per shard, immune to missing a mutation site |
| LOD fade | `scene/lod-fade.ts:186-226` | correct *because* shards share the material object (decision 2). Implicit mechanisms need an explicit test |
| Update-range fan-out | `element-storage.ts::applyNextSortedIndexChunk` | one chokepoint marks each overlapping shard view dirty (decision 6) |

**Phase 2 — the global merge.** Inside `render-order.ts`, which owns the `renderOrder` domain: `OrderSlot` gains a shard index plus the per-shard projected centroid/radius, `collectRenderOrderSlot` emits `S` slots, order groups become monotone streams, and `assignGlobalRenderOrder` becomes the k-way stream merge of decision 4, reusing `groupContains` / `groupEnclosingSphere` / `centroidMaxReachSphere` and the double-bound edge rule unchanged. The `mesh.renderOrder = 0` reset at `coordinator.ts:1818` must also clear shard ranks, or switching a node away from a sorted mode strands stale positive ranks. A golden test pins that `S = 1` reproduces today's integers exactly.

**Phase 3 — the policy (the feature).** Overlap detection between order-dependent nodes, the draw budget and per-node `S` allocation, config section, URL escape hatch, monitor line (shard count + interleaved draw count).

---

## 6. Testing and ship gates

| Phase | Hard gate before merge |
| --- | --- |
| S | **DONE** — pixel equivalence vs the copy control on all three arms, against a *discriminating* anti-vacuity control (§7.1) |
| 0 | WASM/TS parity on the new outputs, mirroring the existing `depth_sort` cases (behind-camera, degenerate identity, NaN/±Inf, 100k random — `tests/unit/wasm/wasm-vs-typescript.test.ts:406-500`); property test that each shard box bounds exactly its own elements; the `S = 1` box agrees with `geometry.boundingSphere` |
| 1 | Pixel-identical E2E at `S = 1`; shard views are a **partition** of the permutation (property test); `Σ instanceCount == N`; **the picking E2E still passes** (the silent-failure case); no leak on dispose (extend the existing shard-free leak assertions in `tests/unit/rendering/depth-sort-coordinator.test.ts`, whose real-THREE + mocked-Comlink harness is the right home) |
| 2 | Golden: `S = 1` reproduces today's `renderOrder` integers; containment constraint preserved (the #843 embedded-marker fixture); merge total-order sanity under random shard intervals **and** that no stream is ever consumed out of order |
| 3 | Non-vacuous E2E on a **two-node** overlapping fixture — the parent spec's `test_gsplats_normal_overlap_reversed` (`tests/fixtures/generate_test_data.py`) is the single-node front-to-back case and needs a two-node sibling — **fail-first verified by pinning `S = 1`**; draw-budget assertion; `?depthShards=0` determinism |

Cross-cutting requirements: the perf gate records interleaved draw count and frame time against the §7 budget; **all three renderer arms** (WebGL, WebGPU-native, `webgpu-force-webgl`) are required for Phases 1–2 — §7's *cost* numbers are WebGL-only, and §7.1 is the standing proof that a vertex-path assumption can hold on two arms and fail on the third; full E2E before each merge per repo policy; `render-order.ts` and `element-storage.ts` README updates ride their phase.

Two evidence rules carried over from §7.1, because they are general and each one caught a false green: **"the mesh was drawn" does not mean "the mesh contributed pixels"** (an `onAfterRender` hook fires even when pipeline creation failed), and **a pixel-equality assertion is meaningless without a control that fails it** — quote the discriminating number alongside the passing one.

---

## 7. Measured cost (probe, 2026-08-30/31)

Rig: **Apple M4 Max via ANGLE Metal**, headed Chromium (Playwright), 1280×800, `dpr=1`, WebGL backend, three r185, synthetic gsplat/points nodes injected through `__luxarDebug.injectSyntheticScene` — real node-factory, real materials, real element textures. Arms per node count `K`: `S=1` baseline, `S=16` node-major (splitting *without* node alternation), `S=16` interleaved (this design), swept over `K ∈ {1, 2, 8, 32}` nodes at a **constant** total element count. The probe needs **no viewer source change** — it drives `__luxarDebug` only, so it can be re-run on any branch or backend. Raw data and script: `delme/draw-split-probe/`.

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

## 7.1 Measured attribute layout (spike, 2026-08-31)

The question: can a node's ordering permutation be split into `S` contiguous shard draws **without copying it**, on every renderer backend? Rig as §7 (M4 Max / ANGLE Metal, headed Chromium, `?depthSort=0` so the ordering is a pinned identity), `K = 2` synthetic gsplat nodes at 20k splats each, `S = 4`, shard 0 drawn by the parent at a narrowed `instanceCount`. Script + raw JSON: `delme/shard-attr-spike/`.

Each candidate is scored against the **`copy` control** (per-shard copied arrays — the layout §7's draw-split probe already proved renders correctly), because that isolates the attribute layout from everything else. Mean absolute per-channel difference over the canvas:

| Layout | WebGL | WebGPU-native | `webgpu-force-webgl` |
| --- | --- | --- | --- |
| `copy` (control) | — | — | — |
| **`subarray` views** | **0.0000** | **0.0002** | **0.0000** |
| `offsetview` (interleaved) | 0.0000 | **9.2795 — FAILS** | 0.0000 |

The WebGPU failure is explicit, not inferred: `THREE.WebGPURenderer: Render pipeline creation failed — Attribute offset (20000) + format size (4 for VertexFormat::Uint32) must be <= the maximum vertex buffer stride (2048)`. Its pixel signature (lit fraction 0.8876 → 0.7398) matches the shards-missing control almost exactly.

**Three methodology points, each of which changed a verdict:**

1. **"All meshes drew" is NOT sufficient evidence.** The `onAfterRender` counter reported `8/8` shard meshes drawn on the failing WebGPU arm — the hook fires even though pipeline creation failed and the draw contributed nothing. Only the pixel comparison caught it. By the same token `renderer.info.render.calls` is unusable here: three resets `info` at the start of every `render()`, and the viewer runs post-processing, so a post-render read reports just the final fullscreen pass (it read `1` for a 2-node scene).
2. **An anti-vacuity control is mandatory, and it moved the threshold.** Sharding and then *hiding* the shard children (so only shard 0 draws) must visibly diverge; it scores mad ≈ 9.5–10.4. Without that number there is no scale on which 0.0000 means anything — and the first version of this probe used a lit-fraction metric so saturated (0.89) that it could not distinguish a correct split from a silently dropped one.
3. **A residual difference against the *unsharded* baseline is a measurement artifact, and was proved so rather than assumed.** Every sharded arm sits ≈1.19–1.55 mad from the unsharded frame. Re-running at **`S = 1`** — where the "sharded" arm is structurally identical to the baseline, no children at all — reproduces the *same* 1.190. So it is TAA re-convergence between capture points, uniform across arms, and the split itself contributes zero. Screenshot captures must also confirm a real frame was produced (`info.render.frame` advanced): once the on-demand loop settles, `renderOnce()` is a no-op and a naive capture silently re-photographs the previous arm's image.

---

## 7.2 Measured popping, and the shard-count limit (2026-08-31)

Reported from real use: "crazy amount of popping when rotating". Measured and
confirmed, then narrowed to a cause that is **not** what §3.3 first assumed.

Method: orbit the camera in 2.5-degree steps and, separately, drag continuously
with no settling; record mean per-channel image change per step and how often the
cross-node draw sequence changes. Popping shows up as *spikiness* — max change
over median — because a smooth rotation changes the image by a roughly constant
amount per step.

| Scene | `?depthShards` | median | max | spikiness | order changes |
| --- | --- | --- | --- | --- | --- |
| two-comb fixture (settled) | 0 | 2.79 | 5.08 | 1.82 | 0 / 24 |
| two-comb fixture (settled) | 16 | 5.35 | 27.37 | **5.12** | 14 / 24 |
| two-comb fixture (drag) | 0 | 4.34 | 10.36 | 2.39 | 0 / 331 frames |
| two-comb fixture (drag) | 16 | 8.18 | 32.56 | **3.98** | 33 / 401 frames |
| two-comb fixture (drag) | 256 | 4.97 | 35.79 | — | **0** / 347 frames |
| neuromast, 2 volumetric nodes (drag) | 0 | 1.41 | 1.47 | 1.04 | 0 / 564 frames |
| neuromast, 2 volumetric nodes (drag) | 16 | 1.51 | 1.70 | **1.13** | 32 / 637 frames |

**The emitted order is correct.** An independent check — recomputing each drawn
mesh's mean view-z from its own `aSortedIndex` range and the element texture,
then testing that `renderOrder` ascends with depth — finds 0–4 inversions out of
32 draws across a 66-degree orbit, worst inversion 0.1–1% of the depth span. The
merge is not mis-ordering anything.

**The cause is shard COUNT versus how finely the two nodes interleave.** The
fixture is two combs of 5,000 splats offset by 0.0002 along z, so the two nodes
are co-extensive and perfectly interleaved at element scale. At 16 shards each
shard spans ~1/16 of the depth range, so node A's shard *k* and node B's shard
*k* overlap almost completely — no ordering of those two intervals is meaningfully
correct, and which one wins flips on tiny camera changes. Raising the count until
the intervals separate removes it: at 256 shards the median per-frame change
returns to the unsharded baseline (4.97 vs 4.34) with **zero** order changes
across the drag. This is §4 rule 3 ("S ≈ overlap extent ÷ accepted error")
asserting itself — and §4's implementation does not honour it, using a constant
`shardsPerNode` regardless of overlap.

**A too-coarse split is worse than no split**, which is the part worth designing
against: it replaces a stable (if approximate) whole-node order with an
oscillating one. Real data does not hit this — the neuromast's two channels are
co-extensive in *bounds* but spatially segregated at fine scale (nuclei are blobs,
membranes are shells), so coarse shards do separate them: spikiness 1.13 vs 1.04.
Bounding-sphere overlap therefore cannot distinguish the two cases; only the
shard intervals themselves can, and those exist only after a sort.

**Two corrections to earlier claims in this document.** The view-z interval
replaced the re-projected AABB as the merge key (§3.3 decision 3) on the theory
that the box degenerates off-axis. The box *does* degenerate — that reasoning
stands — but swapping the key back gives **identical** numbers on real data
(spikiness 1.07 either way), so it was not the cause of the reported popping and
must not be credited with fixing it. And the settled-pose harness in §7.1 hides
transients by construction; the drag numbers above are the ones that correspond
to what a user sees.

**Open**: gate on separability rather than on bounds — after a sort, compare one
node's shard intervals against the other's and fall back to whole-node ordering
for that pair when they overlap too heavily to order meaningfully. Until that
lands, `shardsPerNode` is a blunt instrument and the feature stays off by default.

---

## 7.3 Does it actually help? Quality and cost against ground truth (2026-08-31)

Everything in §7.1/§7.2 measures whether sharding BREAKS anything. This measures
whether it helps, which nothing before it did — and it is the experiment that
decides whether the idea is worth its complexity.

**Ground truth** is the same 10,000 splats of the two-comb fixture in ONE node,
where within-node sorting is already exact, so the correct composite is
computable. Both two-node arms are approximations of it. Error is the mean
per-channel image difference against that reference, averaged over five camera
angles (0/20/45/70/110 degrees). Cost is the median frame time during a
continuous drag.

| shards | draws | error vs truth | error reduction | fixture frame p50 | neuromast frame p50 |
| --- | --- | --- | --- | --- | --- |
| off | — | 19.62 | — | 3.0 ms | 6.1 ms |
| 16 | 32 | 10.17 | 48% | 2.9 ms | 5.1 ms |
| 64 | 128 | **3.07** | **84%** | **3.3 ms** | **5.2 ms** |
| 256 | 512 | 1.29 | 93% | 4.7 ms | 7.0 ms |

**It works, and it converges.** The error falls monotonically toward the exact
composite. 64 shards removes 84% of it at a frame cost inside this machine's
run-to-run variance (repeat runs of the same configuration moved p50 by up to
~1.5 ms, so treat the 16/64 rows as "not measurably more expensive" rather than
as exact figures); 256 removes 93% for a clear but modest ~1–2 ms.

That settles the doubt §7.2 raised. Sharding is not merely non-destructive on
real data — on a scene where cross-node ordering genuinely matters it recovers
most of the error the authoring workaround (merge + partition) would, without
merging the nodes.

**It also found a silent-disable bug, which had corrupted §7.2's conclusion.**
Shards hold `ceil(N / S)` elements each, so a count that does not divide the
element count is covered by FEWER shards than requested: 256 over 5,000 needs
only 250, and the last six are never created. The coordinator then asked the sort
for 256 boxes, `shardOrderInputFor`'s mesh-count guard rejected the mismatch, and
the node fell back to whole-node ordering — the feature quietly switching itself
off. That is why 256 first measured WORSE than 64 (16.37 vs 3.07), and why §7.2
recorded "raising the count removes the popping": it removed it by disabling the
feature. Fixed by requesting bounds for the ESTABLISHED count
(`establishedShardCount`), pinned by a unit test. §7.2's popping numbers at 16
shards stand; its claim about high counts does not.

**What §7.2's real finding reduces to**, with this in hand: the shard count has
to match how finely the two nodes interleave, and the policy's fixed 16 is simply
too low for a co-extensive pair. The fixture wants 64+. That is a tuning problem
with a measured curve behind it, not a wall — and the still-open separability
gate is what should choose the count per pair instead of a constant.

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
2. **Per-draw COST is still unmeasured on WebGPU** — correctness there is now measured (§7.1), cost is not. §7's numbers are WebGL/ANGLE-Metal only. Per-draw cost, render-object count and bind-group churn all differ on the WebGPU backend and `S` multiplies all three; §7.1 found the two backends diverging on a *correctness* question, which is direct evidence they can diverge on a cost one. Mitigation: Phase 1 and 2 gates require all three arms; an NVIDIA cross-check (obsidian) is queued for the same reason — its driver's per-draw and texture-locality behaviour is not ANGLE's.
3. **"One drawing mesh per node" is load-bearing in far more places than the ordering code — this is where the work actually is**, not in the merge math. Mitigations, in descending order of how much each buys: shards owned by the coordinator rather than the scene graph, so nothing that resolves a node by path changes at all; shards as *children* (decision 2) so visibility, matrix, dispose and traversal inherit; the Phase 1 table enumerating every remaining surface with its adaptation; and the `S = 1` default keeping the refactor reviewable in isolation. **Picking is the member of that set that fails SILENTLY** — no error path, hover simply stops working — so its E2E runs on every phase, not only Phase 1.
4. **Ordering staleness between re-sorts now affects cross-node order too.** Under the 3° / 5%-of-bounding-radius hysteresis a node's permutation — and so its shard boundaries — lags the camera. Mitigation: decision 3 re-projects the per-shard boxes every frame, so the *merge* is always fresh even when the *within-shard* order is not, and decision 4's stream merge means a stale key can only pick the wrong stream, never mis-order one node's own shards. Residual: shard boundaries themselves are stale, bounded by the same hysteresis the subsystem already accepts.
5. **`depthWrite` interaction.** Lines `normal` writes depth at opacity ≥ 0.99 (`normalModeDepthWrite`); interleaving such a node mid-sequence lets it depth-reject later shards behind it. This is *more* correct than today, but it is a visible change in a configuration that currently renders differently — needs a fixture and a CHANGELOG note, not a silent flip.
6. **Publish atomicity across S shard views.** A flip must publish all shards or none (decision 7). Decision 6's single-array layout makes this structurally easy — `uSortedIndexSlot` is one uniform on one shared material, so the flip is still a single write — but each shard view still owns a GPU buffer, so all `S` must be dirtied before the flip, never after. The failure mode is a one-frame mixed-generation composite, exactly the class the generation contract was built for; enforce it in the single apply function, and property-test the state machine with a mock worker as Phase 2 of the parent spec did.
7. **The error is bounded, not zero.** Two elements from different nodes inside the same depth interval remain arbitrarily ordered. Honest ceiling: "no visible cross-node artefact on orbit", not "exact". Do not let the diagnostic in `finalize` stop warning about overlapping sorted nodes just because their ordering improved.

---

## 10. Provenance

No GitHub issue exists for this yet (an open-issue sweep on 2026-08-31 found none matching depth/order/overlap/compositing). The limitation is recorded in `GSPLAT_DEPTH_SORTING_SPEC.md` §1 as an authoring rule, and in `render-order.ts:588` / its README as a documented approximation in those exact words ("exact inter-group ordering does not exist for arbitrarily interleaved groups"). The authoring-rule diagnostics are also known to under-serve this case: the compiler's overlap warning proposes making a node `additive` or separating bounds — both appearance/science changes — and omits merge-and-partition (#2222), and an unpartitioned merge can silently truncate past the per-node clamp (#2221).

Prior art in-repo: PR #843 (containment override), PR #591 (`render-order.ts` extraction), PR #596 (renderOrder pass survives a dead SortWorker), PR #658 (volumetric phase 4 — the third geometry type joining the sorted set with zero coordinator change).

The NVIDIA cross-check named in risk 2 is **pending on infrastructure, not skipped**: the intended host (obsidian, RTX PRO 6000) has had no loaded NVIDIA kernel module since a kernel bump to `6.8.0-138-generic` left the DKMS module unbuilt, and its 16-core load average of ~20 from the agent fleet would invalidate µs-scale timing regardless. Re-run `delme/draw-split-probe/` there once the driver is restored.

---

## 12. Why this was archived (2026-08-31)

Read this before anything above it. Everything earlier was written while the
outcome was still open, so it argues for the design; this section is the only
part written knowing how it turned out.

### The verdict

The mechanism is sound, implemented, and measured. It was archived because on
**real data it is not visibly convincing**, and a permanent feature has to earn
its complexity in what people actually see, not in what a fixture can be built
to show.

Both halves of that sentence matter, so neither should be quoted alone:

- **It works.** Against an exact ground truth — the same 10,000 splats in one
  node, where within-node sorting is already exact — sharding monotonically
  converges on the correct composite: 48% of the error removed at 16 shards,
  84% at 64, 93% at 256 (§7.3). The emitted order was independently verified
  correct (0-4 inversions out of 32 draws across a 66° orbit). This is not a
  feature that failed to function.
- **It does not visibly pay on the scene we checked.** Driven interactively on
  the neuromast (two channels, ~110k splats, forced to `volumetric`) at 250
  shards, the result was smooth and cheap — 500 interleaved draws, frame p50
  6.6 ms vs 6.1 ms unsharded, popping indistinguishable from baseline — and the
  reviewer's judgement was "ok, but not super convincing". That is the finding.

### What the gap between those two means

The fixture is two 5,000-splat combs offset by 1/5000 of their extent:
maximally interleaved, co-extensive, equal density, two saturated complementary
colours chosen so any ordering error shows as a hue shift. It was built to make
the effect *measurable*, and it succeeded. But a construction that makes an
effect measurable is not evidence that the effect is *large in the wild* — it is
evidence that the mechanism operates. Those are different claims, and the whole
arc rested on conflating them until real data separated them.

Real scenes weaken the effect at every step, and the neuromast weakens it at all
of them at once. Its two channels are anatomically nested rather than finely
interleaved, so much of each node is unambiguously in front of or behind the
other and the whole-node order is already nearly right. Both channels are
authored `additive`, which is commutative — the feature is *inert* as shipped,
and only forcing `volumetric` engages it at all. And under volumetric
compositing, moderate absorption makes the difference between two orderings
small per pixel even where the order genuinely is wrong.

So the honest summary is: the error this fixes is real, but on the data we have,
it is mostly small, mostly hidden behind commutative blending, and mostly not
where the eye is.

### The cost of keeping it

Against that, the standing price is not small. Every node becomes potentially
several draw calls, and *every* traversal that assumed one drawing mesh per node
becomes a place where a future change can silently break: picking (which fails
with no error path — hover simply stops working), element counts, draw-order
rows, disposal, material identity, LOD fade, context restore. §5 Phase 1 lists
nine such surfaces; each is adapted and tested here, but each is also a
permanent invariant that a contributor who has never heard of depth shards can
violate without any gate going red.

This arc produced two bugs of exactly that shape, both silent, both found only
by measurement: a parent redrawing its whole node on top of its own shards
(9,375 elements drawn for a 5,000-element node), and a shard count that did not
divide the element count causing the feature to **quietly switch itself off**
while still reporting as enabled — which corrupted a conclusion in §7.2 that
stood for several hours. A subsystem that fails silently twice during its own
construction will fail silently again during someone else's.

That is the trade that was declined: a permanent, silent-failure-prone
invariant across nine subsystems, in exchange for an improvement that a careful
reviewer looking straight at it called "not super convincing".

### What is preserved, and what is not

Branch `feat/cross-node-depth-shards` (tip `039897772`) holds the complete
implementation — WASM+TS kernel with per-shard AABBs and exact parity, shard
meshes, the k-way stream merge, the policy, config section, URL parameter, 60
depth-shard unit tests, and a two-node E2E spec verified fail-first. It is
green: 13,439 viewer unit tests, typecheck, lint, warning-fatal Sphinx. Nothing
needs redoing to resume; the branch is retained deliberately and must not be
pruned.

Two things are *not* in the branch and are recorded here instead.

**The separability gate, designed but unbuilt.** The one clear gap at archive
time was that the shard count is a constant, when §7.2 and §7.3 together show it
has to track how deeply a specific pair interpenetrates. The derivation, for
whoever picks this up:

> To order two nodes that interpenetrate over a depth extent `P`, a node of
> depth extent `E` must be cut into ranges thinner than `P/k` to resolve that
> region `k` ways — so it needs `k·E/P` shards. Take `P = rA + rB − d` (the
> overlap of their depth ranges along the axis through their centres: the
> direction in which they are most separated, hence the smallest such overlap
> over all view directions and the conservative requirement) and `E = 2·rA`.
> A node's requirement is the max over its overlapping foreign partners, since
> one set of draw calls serves all of them.

Two properties fall out, and both are the wanted behaviour. A **co-extensive**
pair (`P = E`) needs exactly `k` — so the configured constant becomes "shards
for a fully co-extensive pair", and §7.3's curve is its calibration. A pair that
**barely grazes** needs `k·E/P → ∞`, cannot be afforded, and is left unsharded —
correct on both counts, since the mis-composited lens is tiny and a split too
coarse to resolve it would only make that lens flap as the camera moves. Under a
draw budget, admit cheapest-first and **drop** a node whose requirement does not
fit rather than scaling it down: under-sharding is the failure mode the gate
exists to prevent, so serving fewer nodes properly beats serving all of them too
coarsely.

**The ground-truth method, which outlived the feature.** Merge the nodes under
test into a single node and photograph that: within-node sorting is exact, so
the correct composite becomes computable and "does this help?" stops being an
argument. It cost about an hour and it overturned two of my own diagnoses. It
generalises to any cross-node compositing question and should be the first move
next time, not the last. The generator lived at `delme/groundtruth/make_merged.py`
(gitignored, not preserved — it is ~60 lines and reproducing it from this
paragraph is faster than recovering it).

### What would justify reopening this

Not a better implementation — the implementation is done. Reopen it on
**evidence of demand**, in roughly this order of strength:

1. A real dataset where the mis-composite is visible without being pointed out,
   and whose authors cannot merge the nodes (§2's existing answer: merge them,
   `partition={"max_elements": N}`). The `volumetric` × genuinely-interpenetrating
   × two-separate-layers-required combination is the case; we did not find one.
2. Repeated user reports of wrong compositing between layers.
3. Mesh with transparency landing, which would add order-dependent surfaces that
   *cannot* be merged the way splats can.

### Measure the ceiling before building any successor

This applies to **Route C** as much as to Route B, and it is the cheapest useful
thing anyone can do here.

Every mechanism in this family — depth shards, a shared per-type render list, one
globally merged draw — is an attempt to approach the same target: the composite
you would get if the overlapping nodes were a single node. That target is
directly constructible today, by merging the nodes and compiling one node from
the union. So the ceiling on the entire family is measurable in about an hour,
without building any of them:

> Build the scene twice from identical element data — once as two nodes, once
> merged into one — and compare the renders at several camera angles.

If the merged render is not convincingly better than the two-node one on a given
dataset, then **no** ordering mechanism can be, because merged *is* the exact
answer that all of them approximate. That closes the question for that dataset
in an hour instead of a quarter.

Route B was archived on an approximation (250 shards on the neuromast) rather
than on this ceiling, which is the one loose end in §12: it is possible, though
on the evidence unlikely, that the exact answer is visibly better than the
250-shard approximation was. Anyone reopening this should measure the ceiling on
the neuromast first — and if it *is* convincing, note that the successor to build
is Route C, not Route B, since Route C reaches the ceiling exactly and *reduces*
draw calls instead of multiplying them.

One correction for whoever reads "same geometry type" as the tractable subset:
per §8.2 it is not where the cost lives. The expensive parts — global element
storage, per-element indirection for the ~20 per-node uniforms, and invalidating
the global buffer on every LOD commit — are required whether or not the merge
spans types. The cross-type extra is one shader branch over three texel layouts.
Restricting to one type buys perhaps 15% of the work, not an order of magnitude.

Absent those, the authoring rule remains the right answer, and its limitation is
documented where users meet it (`render-order.ts:588`, the depth-sort README).

## 11. References

External work this design leans on or explicitly declines:

- Kern, Neuhauser, Maack, Han, Usher, Westermann, *A Comparison of Rendering Techniques for Large 3D Line Sets with Transparency* — [arXiv:1912.08485](https://arxiv.org/abs/1912.08485). The closest domain match (transparent scientific geometry); source of the depth-bucketing framing.
- *Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency* — [arXiv:2410.08129](https://arxiv.org/abs/2410.08129). K=16 exactly-sorted per-pixel core + order-independent tail; **skips global presorting entirely** and still reports 457 vs 317 FPS against 3DGS at equal-or-better quality. The strongest long-term argument against *any* global sort — and WebGPU-only, since it needs fragment-shader atomics that WebGL2 does not have.
- Radl, Steiner et al., *StopThePop: Sorted Gaussian Splatting for View-Consistent Real-Time Rendering* — [arXiv:2402.00525](https://arxiv.org/abs/2402.00525). Hierarchical approximate per-pixel sort for ~4% overhead; defines the popping error class this design does not address.
- *StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting* — [arXiv:2503.24366](https://arxiv.org/abs/2503.24366). The sorting-free extreme.
- McGuire & Bavoil, *Weighted Blended Order-Independent Transparency* (JCGT 2013); Münstermann et al., *Moment-Based Order-Independent Transparency* (2018) — [project page](https://cg.ivd.kit.edu/mboit.php); [nvpro-samples/vk_order_independent_transparency](https://github.com/nvpro-samples/vk_order_independent_transparency) for a seven-technique side-by-side.
- Spark — [system design](https://sparkjs.dev/docs/system-design/), [PackedSplats](https://sparkjs.dev/docs/packed-splats/). The production precedent for Route C in three.js.
