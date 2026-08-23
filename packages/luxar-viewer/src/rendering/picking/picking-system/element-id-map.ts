/**
 * Visible-buffer slot → on-disk element index resolution for picking.
 *
 * A pick shader can only report where an element sits in the buffer that was
 * uploaded to the GPU — its **storage slot**. That is not the same number as
 * the element's **on-disk index**, which is what the per-element label CSR
 * (`label_offsets` / `label_bytes`) is keyed by, whenever the visible buffer's
 * index space diverges from the on-disk one. For Points that happens two ways:
 * spatial range loading (only the visible on-disk ranges are concatenated) and
 * effective-radius compaction (zero-radius points are dropped in place). A
 * hover tooltip that reads a label at the raw slot then shows a
 * wrong-but-plausible neighbour's label (issue #1421).
 *
 * The fix is a published map: whichever stage knows the slot → on-disk mapping
 * composes it (`data/loaders/element-ids.ts`), the commit pipeline stamps it
 * onto the node in lockstep with `committedData`
 * (`types/committed-data::setElementIdMap`), and this helper reads it back at
 * the single `PickResult` construction site. The stamp is a MESH-level slot,
 * not a field on the loaded payload: a payload can be a SliceCache-owned
 * snapshot whose byte size was measured at store time, so writing the map onto
 * it would both under-count the cache and break its never-mutated invariant.
 *
 * Who populates the map today:
 *  - **Points** — `data/points/projection.ts::projectPointsTo3D`, for a node
 *    declaring `has_labels` / `has_image_labels` / `has_keys` and only on the non-identity
 *    path (a single range starting at 0 with no compaction emits nothing, so
 *    picking stays allocation-free in the common Points case). The commit
 *    (`commit-points-geometry.ts`) forwards `LoadedPointsData.elementIds` to
 *    the mesh stamp.
 *  - **Points additive ladders** — composed as well, into the ladder's union
 *    index space. `add_points(..., labels=…, additive_lod=True)` writes ONE
 *    union label CSR on the PARENT node, keyed by
 *    `additive_0 || additive_1 || …` (each level in its stored order) and
 *    declared by the parent's `has_labels` (#1422); the per-level maps are
 *    composed into that space. Because a sub-LOD has
 *    no CSR a reader could key by, `createProgressivePointsLoader` overrides
 *    each `additive_<i>` node's label flags with the PARENT's: with a union CSR
 *    every level builds its own level-space map, and
 *    `points-progressive-loader.ts::concatenatePointsData` offsets level `i` by
 *    the preceding levels' ON-DISK counts (`n_points`) to land it in that union
 *    space — a level that published no map (projection identity fast path)
 *    contributing `offset + slot`. Each id is BOUNDED by its own level's row
 *    count as well as shifted: an index past it would name a real row belonging
 *    to a sibling level, which is a confidently wrong label rather than a
 *    missing one. Nothing is emitted when the parent declares
 *    no labels, when the whole resident ladder is complete and unculled (slot
 *    IS the union index), or when the inputs are inconsistent. That last case
 *    is NOT a suppression: with no map this helper returns the slot, so on a
 *    sliced ladder the tooltip still shows whatever CSR row the slot hits. What
 *    refusing buys is that the wrong id is never one this code COMPOSED out of
 *    data it knows is inconsistent — no worse than the pre-#1439 behaviour —
 *    issue #1439.
 *  - **Mesh** — nothing to do: its pick shader already reports the on-disk
 *    vertex ordinal via `gl_VertexID`, so the lookup correctly no-ops.
 *  - **GSplats** — composed at PROJECTION time, not load time
 *    (`data/scene-loader/process/data-processor-gsplats.ts::toProcessed`) and
 *    stamped by `commit-gsplats-geometry.ts`. The two halves are the loader's
 *    visible `ranges` (published only for a node declaring `has_labels` /
 *    `has_image_labels` / `has_keys`) and the surviving source indices the fused kernel now
 *    records (`project_gsplats_nd_to_3d`'s `out_source_indices`, since
 *    hidden-dim visibility compaction happens inside it). The standard-3D fast
 *    path emits every splat in order, so it records nothing and the
 *    range-offset path alone applies — issue #1423. Note the Points
 *    "allocation-free in the common case" framing does NOT carry over: on the
 *    general (compacting) projection path the composer's identity fast path is
 *    effectively unreachable, because whenever `ranges` is published there,
 *    source indices are always supplied — so even a labelled node with a
 *    single `[0, N)` range and zero culling allocates a full N-element
 *    identity map. (Only the standard-3D fast path, which supplies none, can
 *    still reach the identity.) Neither is the COST the same as Points': the
 *    retained map is the same 4 B/element, but the general path additionally
 *    allocates the `splatCount`-sized `Uint32Array` per projection that
 *    `emitSourceIndices` documents as 4 B/splat, and round-trips it through
 *    wasm-bindgen, which copies it into linear memory and back out — so a
 *    transient allocation plus ~8 B/splat of memcpy on top, none of which
 *    Points pays. All of it is gated on `has_labels` / `has_image_labels` / `has_keys`.
 *  - **Lines** — the longest chain, because on top of range loading it has a
 *    GRANULARITY mismatch: the pick shader reports a visible SEGMENT slot while
 *    line labels are per-VERTEX. Four spaces, composed in
 *    `data/scene-loader/process/data-processor-lines.ts` and stamped by
 *    `commit-lines-geometry.ts` (issue #1424): **E** visible segment slot → **D**
 *    loaded segment row (the projection's `sourceSegmentIndices`, derived from
 *    the same `visibility` mask every clipping kernel compacts against) → **C**
 *    loaded-local vertex index (`LoadedLinesData.segments[2·D]`) → **A** on-disk
 *    sorted vertex row (the loader's flat `vertexRangeBounds` pairs through
 *    `buildElementIdMap`).
 *    A segment has TWO endpoints and the pick id is a `flat` vertex-stage
 *    varying, so exactly one can be reported: by convention it is the **START**
 *    vertex. Gated on `has_labels` / `has_image_labels` / `has_keys` like the others, and it
 *    fails closed to the raw slot on any inconsistency.
 *  - **Lines additive ladders** — still out. The per-node indirection above
 *    exists now, but nothing composes the LEVELS: `lines-progressive-loader.ts`
 *    has no counterpart to the Points ladder offsetting above, and a level's
 *    `vertexRangeBounds` are in that level's own vertex space.
 *  - **GSplat additive ladders** — nothing to compose: the authoring path has
 *    no `labels` channel for a gsplat ladder, so no level of one carries
 *    labels at all.
 *  - **Partitioned points, gsplats and lines** — composed, no longer a gap.
 *    `core/group/adders/points.py`, `core/group/adders/gsplats.py` and
 *    `core/group/adders/lines.py` slice the
 *    CSR onto each `part_<i>` leaf, so a partitioned labelled node publishes a
 *    map in the PART's local on-disk space — which is exactly the space that
 *    leaf's sliced CSR is keyed by. Since #1415/#1420,
 *    `core/app/picking/pick-result-handler.ts` looks labels up on the hit LEAF
 *    (`lookupPath = result.mainNode.name`; the outermost `kind=partition`
 *    wrapper is now the reported path only). The two halves cannot drift apart,
 *    because they name the same object: `picking-system.ts::readbackAndVote`
 *    builds the result with `elementId: resolveOnDiskElementId(nodeEntry.main,
 *    …)` and `mainNode: nodeEntry.main`, so the node this helper reads the map
 *    from is the node whose `name` becomes the CSR path. The
 *    `partition=`-forwards-`additive_lod=` composition
 *    (`core/group/adders/points.py`) is readable too since #1422: the part's
 *    ladder parent — which IS the part's scene node — carries one union CSR
 *    spanning its `additive_<i>` levels and declares `has_labels` or `has_keys`. For POINTS
 *    it is MAPPED as well, by exactly the ladder composition above — that
 *    composition reads the parent flags off the part group, so a partitioned
 *    ladder needs no separate path. A LINES part stays readable-but-unmapped,
 *    like any lines ladder, and resolves at the raw committed slot.
 *
 * **Identity fallback is not always a safe answer**, so the map's lifetime is
 * decoupled from the no-op stamp's. `rendering/depth-sort-coordinator.ts::
 * noteDepthSortBlendingModeSwitch` drops `committedData` on any sortable node
 * switched from a commutative mode TO `normal` / `volumetric` (the LayersPanel
 * compose chain) — and all three map-publishing geometries are exposed: POINTS
 * (drawn as instanced quads on a `THREE.Mesh`), GSPLATS — the geometry most
 * likely to be switched to `normal` / `volumetric` in the first place — and
 * LINES, which is depth-sortable too (`types/geometry-capabilities.ts`) and
 * whose commit calls `noteDepthSortCommit` (`commit-lines-geometry.ts`). There
 * the object stays drawn and pickable: the stamp is cleared purely to defeat
 * the commit no-op gate, and the `requestReprocess?.()` that re-stamps it is
 * async, so a pick in that window would resolve through the raw slot — a
 * silently WRONG label rather than "no answer", and for LINES strictly worse
 * than for the other two, because the raw slot is a SEGMENT number handed to a
 * per-VERTEX label array (a GRANULARITY error, not merely an offset error). It
 * therefore calls `invalidateCommittedDataStamp`, which leaves this map in
 * place (the buffers it describes are untouched); only `clearCommittedData`,
 * used where the geometry is genuinely released (LOD demotion, dataset
 * teardown), drops both.
 *
 * @module rendering/picking/picking-system/element-id-map
 */

import type * as THREE from 'three';
import { getElementIdMap } from '../../../types/committed-data';

/**
 * Translate a picked storage slot into the on-disk element index the label
 * CSR is keyed by.
 *
 * Identity is the safe default: with no published map, a stamp of the wrong
 * type, or a slot outside the map, the slot is returned unchanged. This never
 * throws and never returns a sentinel — picking runs on the hover path and
 * must not become a source of exceptions.
 *
 * @param mainNode - The picked node's main (visual) object.
 * @param slot - The element index the pick buffer reported.
 * @returns The on-disk element index, or `slot` when no map applies.
 */
export function resolveOnDiskElementId(mainNode: THREE.Object3D, slot: number): number {
  const ids = getElementIdMap(mainNode);
  if (ids === undefined) return slot;
  if (!Number.isInteger(slot) || slot < 0 || slot >= ids.length) return slot;
  return ids[slot];
}
