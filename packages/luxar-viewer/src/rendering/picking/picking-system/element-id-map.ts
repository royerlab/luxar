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
 * The fix is a published map: a loader that knows its slots are not on-disk
 * indices emits `LoadedPointsData.elementIds`, the commit pipeline stamps the
 * loaded payload onto the node (`types/committed-data`), and this helper reads
 * it back at the single `PickResult` construction site.
 *
 * Who populates the map today:
 *  - **Points** — `data/points/projection.ts::projectPointsTo3D`, for a node
 *    declaring `has_labels` / `has_image_labels` and only on the non-identity
 *    path (a single range starting at 0 with no compaction emits nothing, so
 *    picking stays allocation-free in the common case).
 *  - **Mesh** — nothing to do: its pick shader already reports the on-disk
 *    vertex ordinal via `gl_VertexID`, so the lookup correctly no-ops.
 *  - **GSplats / Lines** — same class of divergence, handled separately
 *    (gsplats needs a WASM kernel change; lines has a segment-vs-vertex
 *    granularity mismatch on top).
 *  - **Partitioned points** — a known gap. `core/group/adders/points.py`
 *    writes the sliced CSR on each `part_<i>` leaf, so a partitioned labelled
 *    node publishes a map in the PART's local on-disk space while
 *    `pick-result-handler.ts` still resolves labels against the outermost
 *    `kind=partition` wrapper (which has no CSR, so the tooltip is null
 *    either way today). #1415/#1420 moves the lookup to the leaf, and the two
 *    then compose.
 *
 * **Identity fallback is not always a safe answer.**
 * `rendering/depth-sort-coordinator.ts::noteDepthSortBlendingModeSwitch`
 * calls `clearCommittedData` on a `THREE.Mesh` — the three.js class, not the
 * Mesh geometry type above; the case that matters here is a POINTS node (also
 * drawn as a `THREE.Mesh`, instanced quads) switched to `normal` or
 * `volumetric` blending, since Mesh publishes no map and so cannot hit this.
 * The object stays drawn and pickable; the stamp is cleared purely to defeat
 * the commit no-op gate, and the `requestReprocess?.()` that re-stamps it is
 * async. Absence of `committedData` therefore does not always mean the GPU
 * buffers were released — in that window a pick still resolves, this helper
 * returns the raw slot, and for a range-loaded labelled points node that is a
 * silently WRONG label rather than "no answer". Not fixed here.
 *
 * @module rendering/picking/picking-system/element-id-map
 */

import type * as THREE from 'three';
import { getCommittedData } from '../../../types/committed-data';

/**
 * Translate a picked storage slot into the on-disk element index the label
 * CSR is keyed by.
 *
 * Identity is the safe default: with no committed data, no published map, a
 * map of the wrong type, or a slot outside the map, the slot is returned
 * unchanged. This never throws and never returns a sentinel — picking runs on
 * the hover path and must not become a source of exceptions.
 *
 * @param mainNode - The picked node's main (visual) object.
 * @param slot - The element index the pick buffer reported.
 * @returns The on-disk element index, or `slot` when no map applies.
 */
export function resolveOnDiskElementId(mainNode: THREE.Object3D, slot: number): number {
  const data = getCommittedData(mainNode);
  if (typeof data !== 'object' || data === null) return slot;
  const ids = (data as { elementIds?: unknown }).elementIds;
  if (!(ids instanceof Uint32Array)) return slot;
  if (!Number.isInteger(slot) || slot < 0 || slot >= ids.length) return slot;
  return ids[slot];
}
