/**
 * A child abort controller that follows its parent signal.
 *
 * A pass that starts several reads at once (a pinned pass's concurrent rungs)
 * sometimes has to cancel SOME of them without cancelling the pass: the
 * residency brake keeps rung k and drops k+1..n. Each read therefore runs under
 * its own child controller, which aborts when the parent does and can also be
 * aborted alone.
 *
 * The child inherits the parent's fetch-priority cell (shared, so a later
 * `raise` on the parent reaches the child) and decode-origin tag: a child is the
 * same request class as its parent, and an untagged child would silently
 * resolve to the default `'demand'` priority.
 *
 * @module data/loaders/progressive/child-signal
 */

import {
  signalOrigin,
  tagSignalOrigin,
} from '../../../cache/decompressed-chunk-cache/decode-origin';
import { signalPriority, tagSignalPriority } from '../../../utils/fetch-concurrency';

/** A child controller plus the cleanup that detaches it from its parent. */
export interface ChildController {
  readonly controller: AbortController;
  /** Detach from the parent (call once the child's work has settled). */
  readonly detach: () => void;
}

/** Create a child of `parent` (a plain controller when there is no parent). */
export function createChildController(parent?: AbortSignal | null): ChildController {
  const controller = new AbortController();
  if (!parent) return { controller, detach: () => undefined };
  const priority = signalPriority(parent);
  if (priority) tagSignalPriority(controller.signal, priority);
  const origin = signalOrigin(parent);
  if (origin !== undefined) tagSignalOrigin(controller.signal, origin);
  if (parent.aborted) {
    controller.abort(parent.reason);
    return { controller, detach: () => undefined };
  }
  const onAbort = (): void => controller.abort(parent.reason);
  parent.addEventListener('abort', onAbort, { once: true });
  return { controller, detach: () => parent.removeEventListener('abort', onAbort) };
}
