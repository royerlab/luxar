/**
 * Abort signal for a progressive loader's next-rung lookahead prefetch.
 *
 * The lookahead is speculative cache warming for the CURRENT view's next
 * rung. It must stop when the view changes (the loader aborts the controller
 * in its view-change branch and on dispose) and when the update that
 * scheduled it is superseded (the update's own abort signal is linked in).
 * It gets its OWN controller rather than reusing the update signal because
 * the controller is tagged — with the `'lookahead'` decode origin and the
 * `'speculative'` fetch priority — and tagging the update signal would
 * re-attribute (and demote) the foreground's demand reads. Untagged, the
 * warm-up would resolve to `'demand'` in the decode cache and the fetch gate:
 * bypassing the speculative share cap and competing with frame-blocking reads.
 *
 * @module data/loaders/progressive/lookahead-signal
 */

import { tagSignalOrigin } from '../../../cache/decompressed-chunk-cache/decode-origin';
import { tagSignalPriority } from '../../../utils/fetch-concurrency';

/**
 * Create a `'lookahead'`-origin, `'speculative'`-priority controller that also
 * aborts when `updateSignal` aborts.
 *
 * @param updateSignal - The scheduling update's abort signal, when it has one.
 * @returns A fresh controller the caller owns (abort it on view change / dispose).
 */
export function createLookaheadController(updateSignal?: AbortSignal): AbortController {
  const controller = new AbortController();
  tagSignalOrigin(controller.signal, 'lookahead');
  tagSignalPriority(controller.signal, 'speculative');
  if (updateSignal?.aborted) {
    controller.abort();
  } else {
    updateSignal?.addEventListener('abort', () => controller.abort(), { once: true });
  }
  return controller;
}
