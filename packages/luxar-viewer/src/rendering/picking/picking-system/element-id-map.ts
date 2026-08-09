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
 *    declaring `has_labels` / `has_image_labels` and only on the non-identity
 *    path (a single range starting at 0 with no compaction emits nothing, so
 *    picking stays allocation-free in the common Points case). The commit
 *    (`commit-points-geometry.ts`) forwards `LoadedPointsData.elementIds` to
 *    the mesh stamp.
 *  - **Mesh** — nothing to do: its pick shader already reports the on-disk
 *    vertex ordinal via `gl_VertexID`, so the lookup correctly no-ops.
 *  - **GSplats** — composed at PROJECTION time, not load time
 *    (`data/scene-loader/process/data-processor-gsplats.ts::toProcessed`) and
 *    stamped by `commit-gsplats-geometry.ts`. The two halves are the loader's
 *    visible `ranges` (published only for a node declaring `has_labels` /
 *    `has_image_labels`) and the surviving source indices the fused kernel now
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
 *    Points pays. All of it is gated on `has_labels` / `has_image_labels`.
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
 *    vertex. Gated on `has_labels` / `has_image_labels` like the others, and it
 *    fails closed to the raw slot on any inconsistency.
 *  - **Partitioned points, gsplats and lines** — a known gap for ALL THREE.
 *    `core/group/adders/points.py`, `core/group/adders/gsplats.py` and
 *    `core/group/adders/lines.py` each slice the CSR onto every `part_<i>` leaf,
 *    so a partitioned labelled node publishes a map in the PART's local on-disk
 *    space while `core/app/picking/pick-result-handler.ts` still resolves labels
 *    against the outermost `kind=partition` wrapper (which has no CSR, so the
 *    tooltip is null either way today). #1415/#1420 moves the lookup to the
 *    leaf, and the two then compose.
 *
 * **Identity fallback is not always a safe answer.**
 * `rendering/depth-sort-coordinator.ts::noteDepthSortBlendingModeSwitch`
 * calls `clearCommittedData` — which now drops this map too — on any sortable
 * node switched from a commutative mode TO `normal` / `volumetric` (the
 * LayersPanel compose chain). All three map-publishing geometries are exposed:
 * POINTS (drawn as instanced quads on a `THREE.Mesh`), GSPLATS — the geometry
 * most likely to be switched to `normal` / `volumetric` in the first place — and
 * LINES, which is depth-sortable too (`types/geometry-capabilities.ts`) and
 * whose commit calls `noteDepthSortCommit` (`commit-lines-geometry.ts`). The
 * object stays drawn and pickable; the stamp is cleared purely to defeat the
 * commit no-op gate, and the `requestReprocess?.()` that re-stamps it is async.
 * Absence therefore does not mean the GPU buffers were released — in that window
 * a pick still resolves, this helper returns the raw slot, and for a
 * range-loaded (or compacted) labelled node that is a silently WRONG label
 * rather than "no answer". For LINES the fallback is strictly worse than for the
 * other two: the raw slot is a SEGMENT number handed to a per-VERTEX label
 * array, so it is a GRANULARITY error, not merely an offset error. Not fixed
 * here.
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
