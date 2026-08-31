/**
 * Pick-result handler for the GPU picking → hover-overlay pipeline.
 *
 * Builds the async callback handed to `new PickingSystem(...)` in
 * `core/app.ts::initPicking`. The handler maps a `PickResult | null`
 * to label + key + image-URL lookups and forwards the composed payload to
 * `OverlayManager.updateHoverContent`.
 *
 * Extracted from `core/app.ts` so the branch logic (null result,
 * any combination of label/key/image content, no content, fetch failure,
 * or missing loaders) can be unit-tested with stub ports instead of a real
 * `PickingSystem` + zarr-backed loaders.
 *
 * @module core/app/picking/pick-result-handler
 */

import type * as THREE from 'three';
import { log, Modules } from '../../../utils/log';
import type { PickResult } from '../../../rendering/picking/picking-system';
import { gsplatLabelAt } from '../../../data/gsplats/label-channel';

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
 * label/key CSR arrays live on each ``part_<i>`` leaf (the Python
 * ``add_points`` / ``add_lines`` / ``add_gsplats`` / ``add_mesh``
 * partition wrappers slice ``labels`` / ``keys`` per part), and ``elementId`` is the
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
  /**
   * Reads the per-element `keys` CSR (issue #1917). Same shape as
   * `labelLoader` — it IS a `LabelLoader`, constructed on the `'keys'`
   * channel — so the key is fetched on exactly the same terms as the
   * label: lazily, once per node, coalesced.
   */
  keyLoader?: { getLabel: (path: string, idx: number) => Promise<string | null> };
  imageLabelLoader?: { getImageUrl: (path: string, idx: number) => Promise<string | null> };
  overlayManager?: {
    updateHoverContent: (
      r: {
        label?: string | null;
        key?: string | null;
        imageUrl?: string | null;
        nodeName: string;
        elementIndex: number;
      } | null
    ) => void;
  };
  /**
   * Optional sink for the public `selection` embedder event. Fires with the
   * picked element on every (non-superseded) hover-pick, and `null` when the
   * hover clears. Independent of whether a string/image tooltip exists —
   * reports what is currently picked. Inline shape (not the embedder type) to
   * keep this handler decoupled from the public event module — keep it in sync
   * with `SelectionPayload` in `core/app/embedder/events.ts`.
   */
  onSelection?: (
    sel: { nodeName: string; elementIndex: number; hitNodeName: string } | null
  ) => void;
  /**
   * Optional sink retaining the settled pick so a click can act on it
   * (issue #1917). Written on every non-superseded pick and cleared when the
   * hover clears, mirroring `onSelection` exactly.
   *
   * Independent of whether a tooltip has content: an element with no label can
   * still carry a `link` built from `{hover_index}`, and a right-click on it
   * should still offer something. Inline shape (not the cache type) to keep
   * this handler decoupled — see `core/app/interaction/picked-element-cache.ts`.
   */
  onPicked?: (
    pick: {
      mainNode: THREE.Object3D;
      nodeName: string;
      hitNodeName: string;
      elementIndex: number;
      label: string | null;
      key: string | null;
      screenX: number;
      screenY: number;
    } | null
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
 *   user-facing layer. The **queried** path — what the label / key / image
 *   loaders are handed — is the hit leaf *scene node*,
 *   `result.mainNode.name`, which is the CSR owner for a flat node and
 *   for a `part_<i>` of a partition. Using the wrapper for the lookup fails
 *   twice over: it is a bare group with no label/key CSR arrays (they are
 *   written per `part_<i>`), and `result.elementId` is an index in the leaf's
 *   own element space, meaningless against a whole-node array. Before the
 *   split, every hover on a partitioned layer resolved to an empty tooltip,
 *   silently — `LabelLoader` demotes the missing array to an info log and
 *   caches `[]`. Of the three lookups only the label/key `getLabel` calls are
 *   reachable under a partition today (all four adders refuse `image_labels`
 *   alongside `partition=`, so no `part_<i>` ever owns an image CSR);
 *   `getImageUrl` moves with them for consistency, not because it is broken
 *   today.
 *
 *   The selection event carries both paths so the split is resolvable
 *   from outside: `nodeName` is `reportPath` (display) and
 *   `hitNodeName` is `lookupPath` (the node `elementIndex` is local to,
 *   and what an embedder should index against).
 *
 *   Two known limits survive this fix, both outside the handler:
 *   (i) an *additive ladder* carries ONE union CSR per present labels/keys
 *   channel on its parent node (#1422), spanning the levels in `additive_<i>`
 *   order — the same node `lookupPath` names, and the same space the
 *   progressive loader produces when it concatenates the committed levels, so
 *   the lookup is correct — for POINTS also under slicing, since the loader now
 *   composes each level's slot → on-disk map into that union space, offsetting
 *   level `i` by the preceding levels' on-disk `n_points` (#1439). Not for
 *   LINES: its raw slot is a per-*segment* one while the union CSR is
 *   per-*vertex* (#1424), so a laddered lines node with a string channel is
 *   wrong at the granularity, not merely at an offset, whatever the slicing —
 *   and nothing composes a lines ladder's LEVELS either; gsplat ladders carry
 *   no labels/keys at all; and
 *   (ii) `result.elementId` is only sometimes the on-disk CSR index. It
 *   arrives already resolved wherever the node can resolve one — Points,
 *   GSplats and Lines all do, for a node declaring `has_labels` /
 *   `has_image_labels` / `has_keys`, through a published slot → on-disk map or
 *   trivially where the identity already holds and no map is published,
 *   and Mesh needs none because its `gl_VertexID` already IS the on-disk
 *   vertex ordinal (see
 *   `rendering/picking/picking-system/element-id-map.ts`, which does the
 *   translation at the single `PickResult` construction site). For a
 *   LINES node the resolved value is the picked segment's **start**
 *   vertex row in the on-disk (spatially sorted) VERTEX ordering, not a
 *   segment row: line string channels are per-vertex, and a segment carries a
 *   single `flat` pick id, so exactly one of its two endpoints can be
 *   reported and by convention it is the start (#1424). That resolution
 *   is published for a FLAT lines node only, so any lines node without
 *   it — one with no string channel, and equally a LADDERED one with a
 *   string channel, whose per-level maps a lines ladder's concat still
 *   drops (limit (i), where only Points composes them) — still reports the raw
 *   visible-segment slot, which is neither an on-disk row nor even the
 *   right granularity for a per-vertex CSR. A Points or GSplats node with
 *   no per-element string channel likewise keeps the raw storage slot — no
 *   CSR to miss, but an embedder reading `SelectionPayload.elementIndex`
 *   there is reading a slot.
 * - `null` result → clear hover (`updateHoverContent(null)`); no loader calls.
 * - Non-null result → fetch label + key + image URL in parallel; emit a
 *   payload only when at least one is truthy. An empty string
 *   counts as no content (matches `LabelLoader.getLabel` which
 *   already returns `null` for empty labels, but the falsy gate
 *   here is the second line of defense).
 * - Any fetch rejects → log a warning and clear hover. Errors must
 *   not kill the hover loop.
 * - Ordering: the string/image fetch is async, so a slow fetch from an
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
        ports.onPicked?.(null);
        return;
      }
      // Partition-aware REPORTING: when the hit's leaf sits under a
      // kind=partition wrapper, the user-facing layer is the wrapper (mirroring
      // how the layers panel treats a partition wrapper). For nested
      // kind=partition-inside-kind=partition, the **outermost** wrapper wins.
      const partitionWrapper = findOutermostPartitionWrapperName(result.mainNode);
      const reportPath = partitionWrapper ?? result.mainNode.name;
      // …but LOOK UP on the leaf. The wrapper holds no per-element string CSR
      // and does not share the leaf's element index space (#1415).
      const lookupPath = result.mainNode.name;
      const [loadedLabel, imageUrl, key] = await Promise.all([
        ports.labelLoader?.getLabel(lookupPath, result.elementId) ?? Promise.resolve(null),
        ports.imageLabelLoader?.getImageUrl(lookupPath, result.elementId) ?? Promise.resolve(null),
        // Same lookup path and element id as the label: keys are written per
        // leaf and reordered by the same permutation (#1917).
        ports.keyLoader?.getLabel(lookupPath, result.elementId) ?? Promise.resolve(null),
      ]);
      const categorical =
        result.mainNode.userData?.nodeType === 'gsplats' &&
        result.mainNode.userData.labelIndices &&
        result.mainNode.userData.labelVocabulary
          ? gsplatLabelAt(
              {
                indices: result.mainNode.userData.labelIndices,
                vocabulary: result.mainNode.userData.labelVocabulary,
              },
              result.storageElementId
            )
          : null;
      const label = loadedLabel ?? (categorical ? `${categorical.name} (${categorical.id})` : null);
      // A newer pick result (or a fade-to-null) arrived while we were
      // fetching — drop this stale one rather than clobber fresher state.
      if (seq !== latest) return;
      // Selection reflects the picked element itself, independent of whether
      // a string/image tooltip exists for it. Both paths are carried: the
      // wrapper to display, the leaf `elementIndex` is local to so an embedder
      // can actually resolve the element (#1415).
      ports.onSelection?.({
        nodeName: reportPath,
        elementIndex: result.elementId,
        hitNodeName: lookupPath,
      });
      // Retain the pick so a click can act on it (#1917). Emitted on the same
      // terms as `onSelection` — what is picked, not what has a tooltip — so
      // an unlabelled element carrying a `{hover_index}`-based link is still
      // clickable. Carries `mainNode` because the interaction templates live
      // in its `userData.attrs`, and the pick coordinate so a later click can
      // confirm the cursor never moved.
      ports.onPicked?.({
        mainNode: result.mainNode,
        nodeName: reportPath,
        hitNodeName: lookupPath,
        elementIndex: result.elementId,
        label: label ?? null,
        key: key ?? null,
        screenX: result.screenX,
        screenY: result.screenY,
      });
      const hasContent = label || key || imageUrl;
      ports.overlayManager?.updateHoverContent(
        hasContent
          ? { label, key, imageUrl, nodeName: reportPath, elementIndex: result.elementId }
          : null
      );
    } catch (err) {
      // Don't let string/image loading errors kill the hover loop. Stay silent if
      // superseded — clearing here would wipe a newer invocation's result.
      if (seq !== latest) return;
      log.warning(Modules.APP, `Picking callback error: ${err}`);
      ports.overlayManager?.updateHoverContent(null);
      ports.onSelection?.(null);
      ports.onPicked?.(null);
    }
  };
}
