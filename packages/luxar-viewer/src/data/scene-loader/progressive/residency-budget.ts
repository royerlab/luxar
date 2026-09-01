/**
 * Residency budget for sweep-registered progressive LOD leaves.
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
 * holding a budget at all. This gate uses the device heap LIMIT already read
 * by the cache sizing path; it does not probe current `usedJSHeapSize`.
 *
 * This is a REFUSAL gate, never an evictor. It declines to load MORE; it never
 * discards what is already committed. A node that hits the ceiling simply stops
 * at the rung it reached and stays there — a legible partial scene rather than
 * a dead tab. Only loaders registered in the four refinement sweep maps are
 * covered; lazy `lod_group` levels are outside those maps and outside this
 * accounting. The budget uses the eager gate's working-set calculation, including
 * its `EAGER_WORKING_SET_CAP_BYTES` ceiling: sufficiently large desktop heaps all
 * resolve to that same cap rather than scaling without bound. This remains an
 * independent settled-residency ceiling, not a shared reservation; eager in-flight
 * bytes and refinement residency may coexist transiently.
 *
 * Decoded payload bytes are measured from the typed arrays. Renderer element
 * rows are derived from each geometry's authoritative layout constant, because
 * those allocations are not present in the decoded payload. The renderer's two
 * Uint32 ordering buffers (8 B/element) are intentionally omitted, as is the
 * separately bounded slice cache. The next pass is estimated from the mean
 * accounted rung so far — see {@link estimateNextRungBytes}. During a
 * fold/commit, the previous cumulative CPU payload can remain reachable while
 * the replacement is allocated, so the transient peak may add nearly one extra
 * decoded cumulative on top of the settled payload + element-row accounting.
 *
 * DECLINING MUST ALSO RETIRE THE LOADER FROM THE RUN. `runProgressiveRefinement`
 * spins while `anyHasMoreLODs()` is true, and a declined loader still has more
 * LODs. If a wrapper only returned `false` from `processLoader` the loop would
 * re-offer it every frame forever, holding the update lock — the same trap
 * `RefinementFailureTracker` avoids by excluding exhausted paths from the
 * aggregation. So callers MUST consult `RefinementResidencyBudget.isDeclined`
 * in their
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

/** The verdict for one loader's refinement pass. */
export interface RefinementAdmission {
  admitted: boolean;
  reason: RefinementAdmissionReason;
  /** Tracked sweep-loader bytes at the moment of the decision. */
  residentBytes: number;
  budgetBytes: number;
}

/**
 * Decide whether one loader may run another refinement pass.
 *
 * Two rules, both from measured bytes:
 *
 *  1. a HARD STOP once tracked residency is already at or past the ceiling, and
 *  2. a PREDICTIVE stop when the next rung would cross it.
 *
 * Rule 2's input comes from {@link estimateNextRungBytes}, which reserves one
 * mean rung but authorises a PASS. A warm-cache pass may append every remaining
 * cache-resident level before the streaming policy yields, so the bound is one
 * pass of overshoot per node per run, not one rung. Admissions reserve their
 * estimate immediately, so later nodes in the same run see that authorised
 * growth. #2432 tracks enforcing the allowance inside the streaming loop.
 * Over-estimating here would stop equal-count ladders several rungs early,
 * degrading scenes that were never in danger.
 *
 * Production construction supplies a device-class fallback when
 * `performance.memory` is unavailable, so Firefox and Safari are budgeted too.
 * The non-positive branch is defensive for direct callers that explicitly supply
 * no usable budget; it admits everything rather than treating invalid input as
 * exhausted capacity.
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
  if (next === 0) {
    return { admitted: true, reason: 'ok', residentBytes: resident, budgetBytes };
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

/**
 * A loader's ladder footprint.
 *
 * TWO TERMS, AND OMITTING THE SECOND BREAKS THE BUDGET ASYMMETRICALLY. The
 * decoded payload is only part of what a committed node costs: each element
 * also occupies a fixed row in the renderer's element texture, and that row's
 * size differs sharply per geometry — 96 B/segment for Lines (6 RGBA32F
 * texels), 64 B/splat for GSplats (4), 48 B/point for Points (3).
 *
 * Measured against the decoded payloads those elements come from (~27 B/vertex
 * for Lines, ~43 B/splat for GSplats), the texture row is ~3.6x the payload for
 * Lines but only ~1.5x for GSplats. So a budget fed the payload alone does not
 * merely under-count — it under-counts LINES BY ~2.4x MORE THAN GSPLATS, and
 * the shared sweep ceiling then declines a 598 MB gsplat node while admitting a
 * 901 MB lines node. That is exactly what was observed when the cap was first
 * measured under forced pressure, and it made the cap useless for the geometry
 * it was written for.
 *
 * The element-row term is derived from each geometry's own authoritative layout
 * constant (`rendering/element-texture-layout`), not estimated. It deliberately
 * excludes the renderer's separate 8 B/element ordering pair documented in the
 * module header.
 */
export interface LadderResidency {
  /** Total bytes of every typed array the loaded rungs hold. */
  residentBytes: number;
  /** How many rungs those bytes represent. */
  loadedRungs: number;
  /** Committed element-texture rows (segments / splats / points; Mesh has none). */
  elementCount: number;
  /** This geometry's element-texture row size, from its layout constant. */
  bytesPerElement: number;
}

/** Total residency a loader accounts for: decoded payload + element rows. */
export function ladderResidentBytes(residency: LadderResidency): number {
  return (
    Math.max(0, residency.residentBytes) +
    Math.max(0, residency.elementCount) * Math.max(0, residency.bytesPerElement)
  );
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

/** Report the residency ceiling once for a scene, across refinement runs. */
export class RefinementResidencyReporter {
  private reported = false;

  reportOnce(path: string, verdict: RefinementAdmission): void {
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

/**
 * One run's residency accounting for sweep-registered progressive leaves,
 * shared by all four geometry refinement phases. The map is seeded from every
 * registered progressive loader, including completed loaders that the wrappers
 * return from before admission, so a new view-triggered run cannot forget bytes
 * already resident and ratchet the ceiling upward.
 */
export class RefinementResidencyBudget {
  private readonly perPath = new Map<string, number>();
  private readonly declined = new Set<string>();

  constructor(
    readonly budgetBytes: number,
    initialResidencies: Iterable<readonly [string, LadderResidency]> = [],
    private readonly reporter = new RefinementResidencyReporter()
  ) {
    for (const [path, residency] of initialResidencies) {
      this.perPath.set(path, ladderResidentBytes(residency));
    }
  }

  /** Build one sized from the device heap, matching the eager admission gate. */
  static forSession(
    poolOverrideBytes?: number,
    fallbackPoolBytes?: number,
    initialResidencies?: Iterable<readonly [string, LadderResidency]>,
    reporter?: RefinementResidencyReporter
  ): RefinementResidencyBudget {
    return new RefinementResidencyBudget(
      computeWorkingSetBudgetBytes(undefined, poolOverrideBytes, fallbackPoolBytes),
      initialResidencies,
      reporter
    );
  }

  /** Measured or optimistically reserved bytes across tracked loaders. */
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
   * progress. An admission immediately reserves its estimated next rung so the
   * remaining loaders in this pass see the growth already authorised.
   */
  admit(path: string, residency: LadderResidency): RefinementAdmission {
    const accounted = ladderResidentBytes(residency);
    const nextRungBytes = estimateNextRungBytes(accounted, residency.loadedRungs);
    this.perPath.set(path, accounted);
    const verdict = planRefinementAdmission(this.residentBytes, nextRungBytes, this.budgetBytes);
    if (verdict.admitted) {
      this.perPath.set(path, accounted + nextRungBytes);
    } else {
      this.declined.add(path);
      this.reporter.reportOnce(path, verdict);
    }
    return verdict;
  }
}
