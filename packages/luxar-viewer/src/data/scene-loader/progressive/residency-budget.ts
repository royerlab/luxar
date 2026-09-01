/**
 * Scene-wide residency budget for progressive LOD refinement.
 *
 * An ADDITIVE ladder is prefix accumulation: its terminal state is 100% of the
 * node, on every node, always. That bounds FIRST PAINT — the first rung is
 * small — but it bounds nothing at rest, and the refinement loop drives every
 * loader to its last rung with no notion of what the scene can afford.
 *
 * On the hosted `cosmicflows_laniakea` demo that is 10 line nodes x ~1.3M
 * segments = 11.4M segments, and the tab dies with `RangeError: Array buffer
 * allocation failed` partway up the ladders (#2426). The eager admission gate
 * in `nodes/load-children-concurrently.ts` already measures each node against a
 * budget and serialises the initial loads — but it releases in the `finally`
 * around each child's load, and refinement is scheduled later
 * (`lifecycle/load-scene.ts`), so by the time the ladders climb, nothing is
 * holding a budget at all. Before this module there was no `usedJSHeapSize`
 * read anywhere in the viewer.
 *
 * This is a REFUSAL gate, never an evictor. It declines to load MORE; it never
 * discards what is already committed. A node that hits the ceiling simply stops
 * at the rung it reached and stays there — a legible partial scene rather than
 * a dead tab. The budget comes from the same
 * {@link computeWorkingSetBudgetBytes} the eager gate uses, so the two agree on
 * what the device can hold instead of drifting apart.
 *
 * Bytes are MEASURED, not modelled. `measureLodBytes` sums the actual
 * `byteLength` of every typed array a loaded ladder holds, so there are no
 * per-geometry constants to get wrong and nothing to re-tune when a payload
 * gains a column. The one estimate is the size of the NEXT rung, for which the
 * largest rung so far is used — see {@link planRefinementAdmission}.
 *
 * DECLINING MUST ALSO RETIRE THE LOADER FROM THE RUN. `runProgressiveRefinement`
 * spins while `anyHasMoreLODs()` is true, and a declined loader still has more
 * LODs. If a wrapper only returned `false` from `processLoader` the loop would
 * re-offer it every frame forever, holding the update lock — the same trap
 * `RefinementFailureTracker` avoids by excluding exhausted paths from the
 * aggregation. So callers MUST consult {@link isDeclined} in their
 * `anyHasMoreLODs` and `getLoaderProgress` closures, exactly as they already do
 * for `failures.isExhausted`.
 *
 * @module data/scene-loader/progressive/residency-budget
 */

import { computeWorkingSetBudgetBytes } from '../../../cache/heap-budget';
import { log, Modules } from '../../../utils/log';

/** Why a refinement step was admitted or refused. */
export type RefinementAdmissionReason =
  /** Budget unknown or disabled — never refuse on an absent signal. */
  | 'unbudgeted'
  /** Fits with the next rung's estimate. */
  | 'ok'
  /** Already at or past the ceiling before this rung. */
  | 'over-budget'
  /** Fits now, but the next rung would cross the ceiling. */
  | 'next-rung-would-exceed';

/** The verdict for one refinement step. */
export interface RefinementAdmission {
  admitted: boolean;
  reason: RefinementAdmissionReason;
  /** Scene-wide measured ladder bytes at the moment of the decision. */
  residentBytes: number;
  budgetBytes: number;
}

/**
 * Decide whether one more rung may be loaded.
 *
 * Two rules, both from measured bytes:
 *
 *  1. a HARD STOP once the scene is already at or past the ceiling, and
 *  2. a PREDICTIVE stop when the next rung would cross it.
 *
 * Rule 2's input comes from {@link estimateNextRungBytes}, which under-estimates
 * on purpose for the ladder shape that actually hurts. Under-estimating means a
 * geometric ladder may cross the ceiling once before rule 1 catches it on the
 * following pass. That is the right direction to err — over-estimating would
 * stop equal-count ladders several rungs early, degrading scenes that were
 * never in danger. The cost of one overshoot is bounded; the cost of
 * systematically under-refining every well-behaved scene is not.
 *
 * A non-positive budget means "no signal" (no `performance.memory`, no
 * override) and admits everything. Refusing to refine because a measurement is
 * unavailable would make Firefox and Safari strictly worse than Chrome at
 * rendering a scene they can hold perfectly well.
 */
export function planRefinementAdmission(
  residentBytes: number,
  nextRungBytes: number,
  budgetBytes: number
): RefinementAdmission {
  const resident = Math.max(0, residentBytes);
  const next = Math.max(0, nextRungBytes);
  if (!(budgetBytes > 0)) {
    return { admitted: true, reason: 'unbudgeted', residentBytes: resident, budgetBytes };
  }
  if (resident >= budgetBytes) {
    return { admitted: false, reason: 'over-budget', residentBytes: resident, budgetBytes };
  }
  if (resident + next > budgetBytes) {
    return {
      admitted: false,
      reason: 'next-rung-would-exceed',
      residentBytes: resident,
      budgetBytes,
    };
  }
  return { admitted: true, reason: 'ok', residentBytes: resident, budgetBytes };
}

/** A loader's measured ladder footprint. */
export interface LadderResidency {
  /** Total bytes of every typed array the loaded rungs hold. */
  residentBytes: number;
  /** How many rungs those bytes represent. */
  loadedRungs: number;
}

/**
 * Estimate the next rung from the mean rung so far.
 *
 * The mean — rather than the largest rung — because a loader may no longer HOLD
 * its individual rungs: since #2427 the Lines loader folds its parts into one
 * cumulative payload and drops them, so per-rung sizes are simply not
 * recoverable, and any estimator that needed them would silently read the
 * merged blob as "one enormous rung" and stop refinement several rungs early.
 * Total-over-count survives the fold because both inputs do.
 *
 * It is exact for an equal-count ladder and an under-estimate for a `stream:C`
 * geometric one (whose next rung is about the size of everything before it) —
 * which is the direction {@link planRefinementAdmission} documents as the safe
 * one. Zero rungs estimates zero: a node that has loaded nothing must never be
 * refused its first rung, or a scene already over budget would never paint at
 * all.
 */
export function estimateNextRungBytes(residentBytes: number, loadedRungs: number): number {
  if (!(loadedRungs > 0) || !(residentBytes > 0)) return 0;
  return residentBytes / loadedRungs;
}

/**
 * One run's scene-wide residency accounting, shared by all four geometry
 * refinement phases (they run sequentially within a run, so one instance sees
 * the whole scene rather than each type budgeting in ignorance of the others —
 * which is the entire point, since Laniakea's ten line nodes are individually
 * affordable and collectively fatal).
 */
export class RefinementResidencyBudget {
  private readonly perPath = new Map<string, number>();
  private readonly declined = new Set<string>();
  private reported = false;

  constructor(readonly budgetBytes: number) {}

  /** Build one sized from the device heap, matching the eager admission gate. */
  static forSession(
    poolOverrideBytes?: number,
    fallbackPoolBytes?: number
  ): RefinementResidencyBudget {
    return new RefinementResidencyBudget(
      computeWorkingSetBudgetBytes(undefined, poolOverrideBytes, fallbackPoolBytes)
    );
  }

  /** Scene-wide measured ladder bytes across every reporting loader. */
  get residentBytes(): number {
    let total = 0;
    for (const bytes of this.perPath.values()) total += bytes;
    return total;
  }

  /**
   * True once `path` has been refused this run. Callers MUST fold this into
   * their `anyHasMoreLODs` / `getLoaderProgress` closures — see the module
   * docstring; omitting it spins the refinement loop at frame rate.
   */
  isDeclined(path: string): boolean {
    return this.declined.has(path);
  }

  /**
   * Record `path`'s current footprint and decide whether it may load one more
   * rung. A refusal is sticky for the rest of the run: re-offering the same
   * loader every pass would re-measure, re-refuse and re-log without making
   * progress. The next view change starts a fresh run and a fresh decision.
   */
  admit(path: string, residency: LadderResidency): RefinementAdmission {
    this.perPath.set(path, Math.max(0, residency.residentBytes));
    const verdict = planRefinementAdmission(
      this.residentBytes,
      estimateNextRungBytes(residency.residentBytes, residency.loadedRungs),
      this.budgetBytes
    );
    if (!verdict.admitted) {
      this.declined.add(path);
      this.reportOnce(path, verdict);
    }
    return verdict;
  }

  /**
   * Log the ceiling ONCE per run rather than once per node. Ten nodes hitting
   * the same ceiling is one fact about the scene, not ten about the nodes, and
   * a per-node line would bury it.
   */
  private reportOnce(path: string, verdict: RefinementAdmission): void {
    if (this.reported) return;
    this.reported = true;
    const mib = (value: number): string => `${(value / (1024 * 1024)).toFixed(1)} MiB`;
    log.warning(
      Modules.SCENE_LOADER,
      `Progressive refinement stopped at the residency ceiling (first: ${path}, ` +
        `${verdict.reason}): resident ${mib(verdict.residentBytes)}, ` +
        `budget ${mib(verdict.budgetBytes)}. Nodes stay at the detail they reached; ` +
        'author a coarser ladder or split the scene to go further.'
    );
  }
}
