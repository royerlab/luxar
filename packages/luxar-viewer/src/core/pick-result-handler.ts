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
 * @module core/pick-result-handler
 */

import { log, Modules } from '../utils/log';
import type { PickResult } from '../rendering/picking/picking-system';

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
 */
export function buildPickResultHandler(
  ports: PickResultHandlerPorts
): (result: PickResult | null) => Promise<void> {
  return async (result) => {
    try {
      if (!result) {
        ports.overlayManager?.updateHoverContent(null);
        return;
      }
      const nodePath = result.mainNode.name;
      const [label, imageUrl] = await Promise.all([
        ports.labelLoader?.getLabel(nodePath, result.elementId) ?? Promise.resolve(null),
        ports.imageLabelLoader?.getImageUrl(nodePath, result.elementId) ?? Promise.resolve(null),
      ]);
      const hasContent = label || imageUrl;
      ports.overlayManager?.updateHoverContent(
        hasContent
          ? { label, imageUrl, nodeName: nodePath, elementIndex: result.elementId }
          : null
      );
    } catch (err) {
      // Don't let label loading errors kill the hover loop.
      log.warning(Modules.APP, `Picking callback error: ${err}`);
      ports.overlayManager?.updateHoverContent(null);
    }
  };
}
