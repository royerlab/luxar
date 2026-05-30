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
 * **outermost** ancestor whose ``userData.kind === 'split'`` and
 * return its ``name`` (= zarr path). The outermost-as-layer
 * convention matches the layers-panel: when a kind=split layer wraps
 * other kind=split or kind=lod groups, hover/click reports the
 * topmost wrapper, not the inner part_<i>.
 *
 * Returns ``null`` when no kind=split ancestor exists — caller falls
 * back to the leaf node's own name.
 */
function findOutermostSplitWrapperName(mainNode: THREE.Object3D): string | null {
  let outermost: string | null = null;
  let cur: THREE.Object3D | null = mainNode.parent ?? null;
  while (cur) {
    if (cur.userData?.kind === 'split' && cur.name) {
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
}

/**
 * Build the pick-result callback. Dependencies are captured at call
 * time — fine because `initPicking` reassigns the loader / overlay
 * references *before* constructing the `PickingSystem`, and each
 * dataset reload reruns `initPicking` end-to-end (disposing the old
 * picking system and its captured closure with it).
 *
 * Behavior contract (must remain byte-identical to the original
 * inline closure in `core/app.ts:initPicking`):
 *
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
        return;
      }
      // Split-aware reporting: when the hit's leaf sits under a kind=split
      // wrapper, the user-facing layer is the wrapper (matches the
      // layers-panel's outermost-as-layer convention). For nested
      // kind=split-inside-kind=split, the **outermost** wrapper wins.
      const splitWrapper = findOutermostSplitWrapperName(result.mainNode);
      const nodePath = splitWrapper ?? result.mainNode.name;
      const [label, imageUrl] = await Promise.all([
        ports.labelLoader?.getLabel(nodePath, result.elementId) ?? Promise.resolve(null),
        ports.imageLabelLoader?.getImageUrl(nodePath, result.elementId) ?? Promise.resolve(null),
      ]);
      // A newer pick result (or a fade-to-null) arrived while we were
      // fetching — drop this stale one rather than clobber fresher state.
      if (seq !== latest) return;
      const hasContent = label || imageUrl;
      ports.overlayManager?.updateHoverContent(
        hasContent ? { label, imageUrl, nodeName: nodePath, elementIndex: result.elementId } : null
      );
    } catch (err) {
      // Don't let label loading errors kill the hover loop. Stay silent if
      // superseded — clearing here would wipe a newer invocation's result.
      if (seq !== latest) return;
      log.warning(Modules.APP, `Picking callback error: ${err}`);
      ports.overlayManager?.updateHoverContent(null);
    }
  };
}
