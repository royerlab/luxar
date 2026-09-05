/**
 * The parts of a per-geometry refinement wrapper that are not per-geometry.
 *
 * Points, Lines, GSplats and Mesh each have a `lod-refinement.ts` wrapping the
 * generic loop in `./refinement.ts`. All four were 83–91% identical after name
 * normalisation — 965 lines carrying one copy each of the admission gate, the
 * abort/rollback/backoff/toast catch block, the residency bookkeeping and the
 * four loop-progress callbacks. Only the load-and-commit body genuinely differs,
 * and it differs in two shapes: Points and Mesh commit the loaded data
 * directly, while Lines and GSplats have an async process step in between.
 *
 * So this module deliberately does NOT abstract over that body. Parameterising
 * the commit shape would trade four readable flows for one function with a
 * strategy argument, and the flow is the part a reader comes here to follow.
 * What it extracts is only what was *character-for-character identical modulo
 * the geometry's own name*:
 *
 * - {@link admitRefinementCandidate} — the `hasMoreLODs` / exhausted-backoff /
 *   residency-admission gate, whose three parts must agree with the three
 *   exclusions in {@link makeRefinementProgressCallbacks} or the loop re-offers
 *   a declined loader every frame while holding the update lock.
 * - {@link handleRefinementError} — abort-versus-failure classification, pass
 *   rollback, backoff counting, and the user-facing toast on exhaustion.
 * - {@link recordRefinementResidency} — the `finally` measurement.
 * - {@link makeRefinementProgressCallbacks} — `getLoaderProgress`,
 *   `anyHasMoreLODs`, `onNoProgress` and `onError`, which differed only in the
 *   geometry label inside their log lines.
 *
 * Keeping these in one place is also what the project's cross-geometry symmetry
 * rule asks for: a fix to one geometry's refinement becomes a fix to all four,
 * rather than three follow-ups that may or may not get written.
 *
 * @module data/scene-loader/progressive/refinement-wrapper
 */

import { log, Modules } from '../../../utils/log';
import { notifier } from '../../../utils/cross-layer/notifier';
import { isAbortError } from '../../loaders/abort-error';
import { tryRollbackToPassStart } from '../../loaders/progressive/pass-rollback';
import type { LadderResidency, RefinementResidencyBudget } from './residency-budget';
import {
  MAX_CONSECUTIVE_REFINEMENT_FAILURES,
  RefinementFailureTracker,
  type ProgressiveRefinementProgress,
} from './refinement';

/**
 * The optional progressive surface a refinable loader may expose.
 *
 * Every field is optional because the loader maps are typed as the plain
 * per-geometry `*DataLoader`: a single-shot spatial-index loader has no ladder,
 * no pass to unwind and no residency to measure. `hasMoreLODs !== true` is the
 * gate that keeps the rest from being reached for one.
 */
export interface RefinableLoader {
  hasMoreLODs?: boolean;
  loadedLODCount?: number;
  totalLODCount?: number;
  rollbackToPassStart?: () => number;
  ladderResidency?: () => LadderResidency;
}

/** Outcome of {@link admitRefinementCandidate}. */
export type RefinementAdmissionDecision =
  { readonly admitted: false } | { readonly admitted: true; readonly allowanceBytes?: number };

/** Geometry-specific wording for a refinement failure. */
export interface RefinementErrorPresentation {
  label: string;
  degradedState?: string;
}

/**
 * Decide whether one loader may refine this pass.
 *
 * Three reasons to decline, in order of cost: it has no more levels; it has
 * already failed `MAX_CONSECUTIVE_REFINEMENT_FAILURES` times this run; or the
 * shared residency ceiling refused it.
 *
 * The first check also keeps single-level geometry out of the loop. This is
 * especially important for meshes, where that is the overwhelmingly common
 * case and whole-node loaders do not expose a ladder.
 *
 * @param path Node path, used as the key for backoff and admission state.
 * @param loader The loader being offered.
 * @param failures Per-run failure tracker.
 * @param residencyBudget Shared residency ceiling; absent means unbounded.
 * @returns Whether to proceed, and the byte allowance if one was granted.
 */
export function admitRefinementCandidate(
  path: string,
  loader: RefinableLoader,
  failures: RefinementFailureTracker,
  residencyBudget?: RefinementResidencyBudget
): RefinementAdmissionDecision {
  if (loader.hasMoreLODs !== true) return { admitted: false };
  if (failures.isExhausted(path)) return { admitted: false };
  // Declining here is not sufficient on its own: `anyHasMoreLODs` and
  // `getLoaderProgress` must exclude declined paths too, or the loop re-offers
  // this loader every frame forever while holding the update lock. That is why
  // both live in this module.
  const residency = loader.ladderResidency?.();
  const admission = residency ? residencyBudget?.admit(path, residency) : undefined;
  if (admission?.admitted === false) return { admitted: false };
  return { admitted: true, allowanceBytes: admission?.allowanceBytes ?? undefined };
}

/**
 * Classify and report a throw from one loader's refinement step.
 *
 * @param presentation Geometry name and optional degraded-state wording.
 * @param path Node path that failed.
 * @param error The thrown value.
 * @param loader The loader, whose pass is unwound on a real failure.
 * @param failures Per-run failure tracker.
 * @returns Always `false` — the loop's `processLoader` contract for "no
 *   progress made". Returned rather than voided so the call site reads
 *   `return handleRefinementError(...)`.
 */
export function handleRefinementError(
  presentation: RefinementErrorPresentation,
  path: string,
  error: unknown,
  loader: RefinableLoader,
  failures: RefinementFailureTracker
): false {
  const { label, degradedState = 'showing reduced detail' } = presentation;
  // Superseded, not failed: a newer view-state (or dispose) aborted the
  // in-flight read on purpose. Don't count it toward the failure backoff or log
  // an error — the loop's next-pass pending check hands off.
  if (isAbortError(error)) return false;
  // Unwind the levels this pass appended before the throw. `updateView`
  // advances the ladder cursor as each level arrives, so without this the
  // retry resumes from the advanced cursor and attempts a larger allocation;
  // reaching the last rung can then flip `hasMoreLODs` false and strand the
  // node at its last committed prefix without ever reaching the failure cap.
  // See `loaders/progressive/pass-rollback`.
  const unwound = tryRollbackToPassStart(loader);
  const message = (error as Error).message;
  if (failures.recordFailure(path)) {
    log.error(
      Modules.SCENE_LOADER,
      `${label} refinement failed for ${path}: ${message} — ` +
        `giving up after ${MAX_CONSECUTIVE_REFINEMENT_FAILURES} consecutive failures ` +
        `(will retry on the next view change; unwound ${unwound} level(s))`
    );
    // The node silently freezes at its last valid coarse prefix — a
    // console-only error leaves the user staring at a permanently coarse node
    // with no explanation. Same channel as leaf-load failures
    // (load-leaf-error-dispatch).
    notifier.toast(`Refinement failed for ${path} — ${degradedState}`, 5000);
  } else {
    log.error(
      Modules.SCENE_LOADER,
      `${label} refinement failed for ${path}: ${message} (unwound ${unwound} level(s))`
    );
  }
  return false;
}

/**
 * Record what one loader ended the step resident, for the shared ceiling.
 *
 * Belongs in the step's `finally`: the measurement must happen on the failure
 * and abort paths too, or a loader that threw mid-load keeps its pre-load
 * allowance and the budget drifts.
 *
 * @param path Node path.
 * @param loader The loader just stepped.
 * @param residencyBudget Shared ceiling; absent means unbounded.
 */
export function recordRefinementResidency(
  path: string,
  loader: RefinableLoader,
  residencyBudget?: RefinementResidencyBudget
): void {
  const measured = loader.ladderResidency?.();
  if (measured) residencyBudget?.record(path, measured);
}

/** The loop callbacks {@link makeRefinementProgressCallbacks} returns. */
export interface RefinementProgressCallbacks<TLoader> {
  /**
   * `Omit<..., 'path'>` matches the loop's own signature: it re-attaches the
   * path itself when it assembles the progress map, so returning one here
   * would be ignored.
   */
  getLoaderProgress(
    path: string,
    loader: TLoader
  ): Omit<ProgressiveRefinementProgress, 'path'> | null;
  anyHasMoreLODs(): boolean;
  onNoProgress(stalled: Array<{ path: string; loaded: number; total: number }>): void;
  onError(error: unknown): void;
}

/**
 * Build the four loop-progress callbacks for one geometry.
 *
 * The exclusion set here (exhausted OR budget-declined OR no more levels) must
 * stay identical to {@link admitRefinementCandidate}'s. A budget-declined loader
 * still reports `hasMoreLODs`, so omitting it from `anyHasMoreLODs` is not a
 * cosmetic bug — the loop never terminates.
 *
 * @param label Geometry name for the log lines (e.g. `'Points'`).
 * @param loaders The geometry's loader map, read live for `anyHasMoreLODs`.
 * @param failures Per-run failure tracker.
 * @param residencyBudget Shared residency ceiling; absent means unbounded.
 * @returns The callbacks, ready to spread into the loop context.
 */
export function makeRefinementProgressCallbacks<TLoader extends RefinableLoader>(
  label: string,
  loaders: Map<string, TLoader>,
  failures: RefinementFailureTracker,
  residencyBudget?: RefinementResidencyBudget
): RefinementProgressCallbacks<TLoader> {
  const excluded = (path: string, loader: RefinableLoader): boolean =>
    failures.isExhausted(path) ||
    residencyBudget?.isDeclined(path) === true ||
    loader.hasMoreLODs !== true;

  return {
    getLoaderProgress: (path, loader) => {
      if (excluded(path, loader)) return null;
      return {
        loaded: loader.loadedLODCount ?? 0,
        total: loader.totalLODCount ?? 0,
      };
    },
    anyHasMoreLODs: () => [...loaders.entries()].some(([path, l]) => !excluded(path, l)),
    onNoProgress: (stalled) => {
      for (const { path, loaded, total } of stalled) {
        log.warning(
          Modules.SCENE_LOADER,
          `${label} refinement stopped for ${path}: no progress at LOD ${loaded}/${total}`
        );
      }
    },
    onError: (error) =>
      log.error(
        Modules.SCENE_LOADER,
        `${label} refinement loop error: ${(error as Error).message}`
      ),
  };
}
