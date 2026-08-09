/**
 * Pick-result handler for the GPU picking → hover-overlay pipeline.
 *
 * Builds the async callback handed to `new PickingSystem(...)` in
 * `core/app.ts::initPicking`. The handler maps a `PickResult | null`
 * to a label + image-URL lookup and forwards the composed payload to
 * `OverlayManager.updateHoverContent`.
 *
 * Extracted from `core/app.ts` so the branch logic (null result,
 * label-only, image-only, both, neither, fetch failure, missing
 * loaders) can be unit-tested with stub ports instead of a real
 * `PickingSystem` + zarr-backed loaders.
 *
 * @module core/app/picking/pick-result-handler
 */

import type * as THREE from 'three';
import { log, Modules } from '../../../utils/log';
import type { PickResult } from '../../../rendering/picking/picking-system';

/**
 * Walk up the parent chain of ``mainNode`` looking for the
 * **outermost** ancestor whose ``userData.kind === 'partition'`` and
 * return its ``name`` (= zarr path). This mirrors how the layers panel
 * treats a ``kind=partition`` wrapper specifically — the wrapper is the
 * user-facing layer, not its ``part_<i>`` children — so when a
 * kind=partition layer wraps other kind=partition or kind=lod groups,
 * hover/click reports the topmost partition wrapper. It is not a general
 * outermost-ancestor rule: the panel also admits ``kind=lod`` wrappers as
 * layers (``ui/layers/layer-state.ts``), and those are deliberately NOT
 * matched here, so a substitutive-LOD layer still reports its internal
 * level path.
 *
 * REPORTING ONLY — never a lookup path. The wrapper is a bare group: the
 * label/image CSR arrays live on each ``part_<i>`` leaf (the Python
 * ``add_points`` / ``add_lines`` / ``add_gsplats`` / ``add_mesh``
 * partition wrappers slice ``labels`` per part), and ``elementId`` is the
 * leaf's own index. See {@link buildPickResultHandler}.
 *
 * Returns ``null`` when no kind=partition ancestor exists — caller falls
 * back to the leaf node's own name.
 */
function findOutermostPartitionWrapperName(mainNode: THREE.Object3D): string | null {
  let outermost: string | null = null;
  let cur: THREE.Object3D | null = mainNode.parent ?? null;
  while (cur) {
    if (cur.userData?.kind === 'partition' && cur.name) {
      outermost = cur.name;
    }
    cur = cur.parent;
  }
  return outermost;
}

/**
 * Narrow port interfaces — the handler only reads the methods it needs
 * from each collaborator, so tests can stub with plain `vi.fn()`s.
 */
export interface PickResultHandlerPorts {
  labelLoader?: { getLabel: (path: string, idx: number) => Promise<string | null> };
  imageLabelLoader?: { getImageUrl: (path: string, idx: number) => Promise<string | null> };
  overlayManager?: {
    updateHoverContent: (
      r: {
        label?: string | null;
        imageUrl?: string | null;
        nodeName: string;
        elementIndex: number;
      } | null
    ) => void;
  };
  /**
   * Optional sink for the public `selection` embedder event. Fires with the
   * picked element on every (non-superseded) hover-pick, and `null` when the
   * hover clears. Independent of whether a label/image tooltip exists —
   * reports what is currently picked. Inline shape (not the embedder type) to
   * keep this handler decoupled from the public event module — keep it in sync
   * with `SelectionPayload` in `core/app/embedder/events.ts`.
   */
  onSelection?: (
    sel: { nodeName: string; elementIndex: number; hitNodeName: string } | null
  ) => void;
}

/**
 * Build the pick-result callback. Dependencies are captured at call
 * time — fine because `initPicking` reassigns the loader / overlay
 * references *before* constructing the `PickingSystem`, and each
 * dataset reload reruns `initPicking` end-to-end (disposing the old
 * picking system and its captured closure with it).
 *
 * Behavior contract:
 *
 * - Two node paths, deliberately split (#1415). The **reported** path —
 *   the selection's `nodeName` and the overlay's title — is the
 *   outermost `kind=partition` wrapper when the hit sits under one,
 *   mirroring how the layers panel treats a partition wrapper as the
 *   user-facing layer. The **queried** path — what the label / image
 *   loaders are handed — is the hit leaf *scene node*,
 *   `result.mainNode.name`, which is the CSR owner for a flat node and
 *   for a `part_<i>` of a partition. Using the wrapper for the lookup
 *   fails twice over: it is a bare group with no `label_offsets` /
 *   `label_bytes` (the CSR is written per `part_<i>`), and
 *   `result.elementId` is the leaf's own `aSortedIndex` / `gl_VertexID`,
 *   meaningless against a whole-node array. Before the split, every
 *   hover on a partitioned layer resolved to an empty tooltip, silently
 *   — `LabelLoader` demotes the missing array to an info log and caches
 *   `[]`. Of the two queries only `getLabel` is reachable under a
 *   partition today (all four adders refuse `image_labels` alongside
 *   `partition=`, so no `part_<i>` ever owns an image CSR);
 *   `getImageUrl` moves with it for consistency, not because it is
 *   broken today.
 *
 *   The selection event carries both paths so the split is resolvable
 *   from outside: `nodeName` is `reportPath` (display) and
 *   `hitNodeName` is `lookupPath` (the node `elementIndex` is local to,
 *   and what an embedder should index against).
 *
 *   Two known limits survive this fix, both outside the handler:
 *   (i) under an *additive ladder* the CSR is scattered per
 *   `additive_<i>` sub-group (which is not a scene node) while the
 *   committed buffer concatenates every loaded level, so no single path
 *   can index it — labels there are unusable regardless of the path
 *   chosen (producer-side gap); and (ii) `result.elementId` is a
 *   visible-buffer storage slot, so it equals the on-disk CSR index only
 *   while no dimension is hidden — a hidden dimension can cull chunks
 *   and triggers effective-radius compaction, either of which shifts the
 *   slot away from the element's on-disk index. Both are pre-existing and
 *   tracked separately.
 * - `null` result → clear hover (`updateHoverContent(null)`); no loader calls.
 * - Non-null result → fetch label + image URL in parallel; emit a
 *   payload only when at least one is truthy. An empty-string label
 *   counts as no content (matches `LabelLoader.getLabel` which
 *   already returns `null` for empty labels, but the falsy gate
 *   here is the second line of defense).
 * - Either fetch rejects → log a warning and clear hover. Errors must
 *   not kill the hover loop.
 * - Ordering: the label/image fetch is async, so a slow fetch from an
 *   older invocation could resolve AFTER a newer one (e.g. an uncached
 *   image while the cursor moves on) and re-show a stale tooltip over
 *   fresher state. A monotonic `latest` token, captured per call, gates
 *   the post-`await` emit: a superseded invocation drops its result. A
 *   `null` (hide) call applies immediately and bumps the token, so it
 *   also cancels any in-flight content fetch.
 */
export function buildPickResultHandler(
  ports: PickResultHandlerPorts
): (result: PickResult | null) => Promise<void> {
  let latest = 0;
  return async (result) => {
    const seq = ++latest;
    try {
      if (!result) {
        ports.overlayManager?.updateHoverContent(null);
        ports.onSelection?.(null);
        return;
      }
      // Partition-aware REPORTING: when the hit's leaf sits under a
      // kind=partition wrapper, the user-facing layer is the wrapper (mirroring
      // how the layers panel treats a partition wrapper). For nested
      // kind=partition-inside-kind=partition, the **outermost** wrapper wins.
      const partitionWrapper = findOutermostPartitionWrapperName(result.mainNode);
      const reportPath = partitionWrapper ?? result.mainNode.name;
      // …but LOOK UP on the leaf. The wrapper holds no label CSR and does not
      // share the leaf's element index space (#1415).
      const lookupPath = result.mainNode.name;
      const [label, imageUrl] = await Promise.all([
        ports.labelLoader?.getLabel(lookupPath, result.elementId) ?? Promise.resolve(null),
        ports.imageLabelLoader?.getImageUrl(lookupPath, result.elementId) ?? Promise.resolve(null),
      ]);
      // A newer pick result (or a fade-to-null) arrived while we were
      // fetching — drop this stale one rather than clobber fresher state.
      if (seq !== latest) return;
      // Selection reflects the picked element itself, independent of whether
      // a label/image tooltip exists for it. Both paths are carried: the
      // wrapper to display, the leaf `elementIndex` is local to so an embedder
      // can actually resolve the element (#1415).
      ports.onSelection?.({
        nodeName: reportPath,
        elementIndex: result.elementId,
        hitNodeName: lookupPath,
      });
      const hasContent = label || imageUrl;
      ports.overlayManager?.updateHoverContent(
        hasContent
          ? { label, imageUrl, nodeName: reportPath, elementIndex: result.elementId }
          : null
      );
    } catch (err) {
      // Don't let label loading errors kill the hover loop. Stay silent if
      // superseded — clearing here would wipe a newer invocation's result.
      if (seq !== latest) return;
      log.warning(Modules.APP, `Picking callback error: ${err}`);
      ports.overlayManager?.updateHoverContent(null);
      ports.onSelection?.(null);
    }
  };
}
