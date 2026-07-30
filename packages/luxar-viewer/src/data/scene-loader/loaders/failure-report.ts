/**
 * Reporting for loader failures — the aggregate view, as distinct from the
 * per-node logging in `nodes/load-leaf-error-dispatch.ts`.
 *
 * Exists because `loadScene` structurally CANNOT throw on a failed load:
 * `loadLeafNode` catches every `LoaderError` and returns null so the rest of the
 * scene still builds. That is the right behavior, but it meant a scene whose
 * every node failed resolved normally and logged an unconditional
 * "Scene loaded successfully" — a green log over an empty viewport, which is
 * what made a real data bug (a WASM trap in the gsplat projection kernel) look
 * like a mystery.
 *
 * @module data/scene-loader/loaders/failure-report
 */

import { log, Modules } from '../../../utils/log';
import { notifier } from '../../../utils/cross-layer/notifier';

/** How a scene load turned out, once every node has settled. */
export type LoadOutcome = 'clean' | 'partial' | 'total';

/**
 * Duration for the total-failure notification. Long, because nothing rendered
 * and the message is the user's only clue — but still a toast rather than the
 * modal error surface: the most common cause is a transient network outage, and
 * the connectivity-triggered retry may recover it seconds later.
 */
const TOTAL_FAILURE_TOAST_MS = 10_000;

/**
 * Warn that some loaders failed, naming them.
 *
 * Extracted from `SceneLoader.updateView` so the end-of-load path can reuse the
 * exact wording instead of duplicating it. Note the update path is NOT the same
 * moment as end-of-load: `updateView` first runs on user navigation, so before
 * this extraction an initial load with total failure produced no aggregate
 * warning at all.
 */
export function warnFailedLoaders(failedPaths: readonly string[]): void {
  if (failedPaths.length === 0) return;
  const joined = failedPaths.join(', ');
  log.warning(Modules.SCENE_LOADER, `⚠️ ${failedPaths.length} loader(s) failed: ${joined}`);
  log.warning(
    Modules.SCENE_LOADER,
    `Some data could not be loaded. Failed loaders: ${joined}. ` +
      'Check console output for details. Data may be incomplete.'
  );
}

/**
 * Report the outcome of a scene load exactly once, proportionately to how bad it
 * was.
 *
 * - clean → the historical success log, text unchanged (an E2E smoke test
 *   matches it, and it is the signal log filters key on).
 * - partial → one aggregate warning. No user-facing notification: the per-node
 *   failures already logged, and the data-loading monitor's failed-loads banner
 *   is the standing surface.
 * - total → an error log plus a toast, because nothing rendered and the caller
 *   cannot tell: `loadScene` resolves successfully either way.
 *
 * @param failedPaths Scene paths whose load failed.
 * @param registeredPaths Scene paths that registered a loader — the attempted
 *   set. A failed lazy LOD level may fail WITHOUT registering, so grade totality
 *   against this set, not a count.
 */
export function reportLoadOutcome(
  failedPaths: readonly string[],
  registeredPaths: readonly string[]
): LoadOutcome {
  if (failedPaths.length === 0) {
    log.success(Modules.SCENE_LOADER, 'Scene loaded successfully');
    return 'clean';
  }

  const joined = failedPaths.join(', ');

  // "total" means every REGISTERED (attempted) node failed. Grade against the
  // path SET, not a count: a failed lazy substitutive LOD level records a
  // failure without ever registering a loader, so a count comparison would
  // mislabel "one eager node rendered + one lazy level failed" as total. Require
  // at least one registered path AND every one of them in the failed set.
  const failedSet = new Set(failedPaths);
  const allRegisteredFailed =
    registeredPaths.length > 0 && registeredPaths.every((p) => failedSet.has(p));
  if (allRegisteredFailed) {
    log.error(
      Modules.SCENE_LOADER,
      `Scene load FAILED — all ${registeredPaths.length} node(s) failed to load: ${joined}`
    );
    notifier.toast(
      `Scene failed to load: all ${registeredPaths.length} data node(s) failed. ` +
        'See the console for details.',
      TOTAL_FAILURE_TOAST_MS
    );
    return 'total';
  }

  log.warning(
    Modules.SCENE_LOADER,
    `Scene loaded with failures — ${failedPaths.length} node(s) failed to load: ${joined}. ` +
      'Some data may be missing.'
  );
  return 'partial';
}
