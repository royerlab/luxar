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
 * a dead tab. Legible to a HUMAN, that is: which nodes won the remaining
 * headroom is not deterministic, so the element counts of a stopped scene are
 * not a property of the store and nothing downstream may compare them across
 * builds. That is why {@link RefinementResidencyReporter} also RECORDS every
 * refusal for {@link RefinementResidencyStop}, which the debug snapshot carries
 * and `core/app/debug/capture-readiness.ts` refuses on (#2508) — a console
 * warning is not something a harness can read. Only loaders registered in the
 * four refinement sweep maps are
 * covered; lazy `lod_group` levels are outside those maps and outside this
 * accounting. The budget uses the eager gate's working-set calculation, including
 * its `EAGER_WORKING_SET_CAP_BYTES` ceiling: sufficiently large desktop heaps all
 * resolve to that same cap rather than scaling without bound. The GPU buffer
 * pool instead takes the full shared non-cache remainder up to its own 2 GB
 * ceiling and covers some of the same renderer element-row bytes. These remain
 * independent ceilings, not additive reservations; eager in-flight bytes,
 * pooled GPU bytes, and refinement residency may coexist transiently.
 *
 * Decoded payload bytes are measured from the typed arrays. Renderer element
 * rows are derived from each geometry's authoritative layout constant, because
 * those allocations are not present in the decoded payload. The renderer's two
 * Uint32 ordering buffers (8 B/element) are intentionally omitted, as is the
 * separately bounded slice cache. The next pass is estimated from the mean
 * accounted rung so far — see {@link estimateNextRungBytes}. An admitted loader
 * receives a per-pass share of the remaining headroom across tracked,
 * non-declined paths, with enough allowance for its estimated next rung, and
 * stops after the first level that spends it. The tracked-set denominator
 * deliberately includes completed or currently filtered paths from the raw
 * sweep maps, so it may under-grant relative to the loaders currently offered.
 * This is not scene-wide round-robin fairness: geometry phases run sequentially,
 * and an earlier phase can consume the ceiling before a later phase is offered.
 * The allowance still prevents a warmed cache from consuming the rest of the
 * ladder under one admission. During a fold/commit, the previous cumulative CPU
 * payload can remain reachable while the replacement is allocated, so the
 * transient peak may add nearly one extra decoded cumulative on top of the
 * settled payload + element-row accounting.
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
import type { RefinementDensityGate } from './density-gate';

/** Why a refinement step was admitted or refused. */
export type RefinementAdmissionReason =
  /** Budget unknown or disabled — never refuse on an absent signal. */
  | 'unbudgeted'
  /** Fits with the next rung's estimate. */
  | 'ok'
  /** Already at or past the ceiling before this rung. */
  | 'over-budget'
  /** Fits now, but the next rung would cross the ceiling. */
  | 'next-rung-would-exceed'
  /** Fits in bytes, but the node already projects denser than its screen cap (`density-gate.ts`). */
  | 'density-cap';

/** The verdict for one loader's refinement pass. */
export interface RefinementAdmission {
  admitted: boolean;
  reason: RefinementAdmissionReason;
  /** Tracked sweep-loader bytes at the moment of the decision. */
  residentBytes: number;
  budgetBytes: number;
}

/** Admission plus this loader's share of additional settled bytes. */
export interface RefinementPassAdmission extends RefinementAdmission {
  /** Per-pass share across tracked paths, lower-bounded by one estimated rung; `null` unbudgeted. */
  allowanceBytes: number | null;
}

/**
 * Decide whether one loader may run another refinement pass.
 *
 * Two rules, both from measured bytes:
 *
 *  1. a HARD STOP once tracked residency is already at or past the ceiling, and
 *  2. a PREDICTIVE stop when the next rung would cross it.
 *
 * Rule 2's input comes from {@link estimateNextRungBytes}, which under-estimates
 * on purpose for the ladder shape that actually hurts. A geometric ladder may
 * cross the ceiling by one level before its pass stops. Admissions reserve
 * their estimate immediately, and measured post-pass bytes replace it before
 * the next loader is considered. Over-estimating here would stop equal-count
 * ladders several rungs early, degrading scenes that were never in danger.
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

/**
 * How many declined paths {@link RefinementResidencyStop.declinedPaths} carries.
 *
 * The snapshot crosses the `page.evaluate` serialisation boundary on its way to
 * a capture tool, and a partitioned scene can decline thousands of paths — one
 * per part — so shipping the whole set would put a multi-megabyte string array
 * in every `getState()` call. `declinedPathCount` is the MEASUREMENT; the sample
 * exists only so a human reading the summary can tell WHICH subtree stopped
 * without re-running the scene with the console open.
 */
export const RESIDENCY_DECLINED_PATH_SAMPLE = 8;

/**
 * A durable record of the scene hitting the refinement BYTE ceiling.
 *
 * Surfaced on `__luxarDebug.getState().refinementResidency` so a capture tool
 * can refuse a scene that stopped short (#2508): the ladder's terminal state is
 * 100% of every node, so a scene that declined is showing a partial one, and
 * WHICH nodes won the remaining headroom is not deterministic. Before this, the
 * only signal was a single console warning, which nothing downstream reads.
 *
 * BYTE CEILING ONLY. A `density-cap` refusal never reaches the reporter — it is
 * camera-dependent, non-sticky across runs, and the density gate's own to report
 * (see `RefinementResidencyBudget.admit`) — so it is absent here by design. A
 * node deferred on density is still going to refine when the user zooms in; a
 * node declined on bytes is not.
 *
 * CUMULATIVE FOR THE LIFE OF THE LOADER, WITH NO RESET, ON PURPOSE. THIS IS THE
 * CANONICAL STATEMENT of that contract; the other places it matters (the debug
 * snapshot field, `core/app/debug/capture-readiness.ts`, the two READMEs) say it
 * in one sentence and point here. The snapshot's OTHER memory-ceiling signal,
 * `gpuPool.byteBudgetEvictions`, is governed by the same contract, because the
 * GPU buffer pool is built and thrown away with the same `SceneLoader`.
 *
 * Once present this record stays present until that loader goes away, so it says
 * "refinement hit the ceiling at some point while this scene was loaded", not
 * "the scene is truncated right now" — a scene that stopped and then refined
 * fully after a view change still carries it. That is the answer a capture tool
 * wants: refinement order is path-dependent, so the run that hit the ceiling
 * settled on a composition the next run would not reproduce. A dataset switch
 * builds a fresh loader (`SceneLoaderManager.createLoaderAsync`) and therefore a
 * fresh reporter and a fresh pool, so the next scene starts from a clean record
 * with no page reload; do not clear either within a loader's life.
 */
export interface RefinementResidencyStop {
  /**
   * The FIRST refusal's reason — `over-budget` or `next-rung-would-exceed`.
   * Named in the capture-readiness message because the two are the difference
   * between "already past the ceiling" and "it fitted, but one more rung would
   * not have", which is the difference between re-authoring the ladder and
   * nudging the budget.
   */
  reason: RefinementAdmissionReason;
  /** Tracked sweep-loader bytes at the moment of that first refusal. */
  residentBytes: number;
  /** The ceiling that first refusal was measured against. */
  budgetBytes: number;
  /** The path that hit the ceiling first. */
  firstPath: string;
  /**
   * DISTINCT paths refused since the scene loaded — never the truncated length
   * of {@link declinedPaths}. The budget is rebuilt per refinement run while the
   * reporter lives for the scene, so a parked path re-declines on every run and
   * must not be counted twice either.
   */
  declinedPathCount: number;
  /**
   * First-seen-order sample of the declined paths, capped at
   * {@link RESIDENCY_DECLINED_PATH_SAMPLE}. Diagnostic only — read
   * `declinedPathCount` for the magnitude.
   */
  declinedPaths: string[];
}

/**
 * Report the residency ceiling once for a scene, across refinement runs, and
 * RECORD every refusal for {@link snapshot}.
 *
 * The two halves are deliberately asymmetric. Logging is once-per-scene because
 * a scene parked at the ceiling starts a fresh run after every view change and
 * would otherwise flood the console during slice playback. Recording is every
 * time, because "how much of the scene declined" is the number a capture tool
 * has to branch on and one console line cannot carry it.
 */
export class RefinementResidencyReporter {
  private reported = false;
  private first: RefinementAdmission & { path: string } = {
    admitted: true,
    reason: 'ok',
    residentBytes: 0,
    budgetBytes: 0,
    path: '',
  };
  /**
   * Distinct declined paths, in first-seen order. A `Set` because the budget is
   * per-run and the reporter is per-scene: the same path re-declines on every
   * subsequent run, and an array would report one stuck node as thousands.
   */
  private readonly declinedPaths = new Set<string>();

  /**
   * Record a byte-ceiling refusal; log the first one. Named for the LOGGING
   * side because `RefinementResidencyBudget.admit` calls it as its
   * report-the-ceiling step, and renaming would be churn across that call site.
   */
  reportOnce(path: string, verdict: RefinementAdmission): void {
    this.declinedPaths.add(path);
    if (this.reported) return;
    this.reported = true;
    this.first = { ...verdict, path };
    const mib = (value: number): string => `${(value / (1024 * 1024)).toFixed(1)} MiB`;
    log.warning(
      Modules.SCENE_LOADER,
      `Progressive refinement stopped at the residency ceiling (first: ${path}, ` +
        `${verdict.reason}): resident ${mib(verdict.residentBytes)}, ` +
        `budget ${mib(verdict.budgetBytes)}. Nodes stay at the detail they reached; ` +
        'author a coarser ladder or split the scene to go further.'
    );
  }

  /**
   * The scene's byte-ceiling stop, or `undefined` when nothing was ever
   * declined.
   *
   * `undefined` rather than a zero-valued object on purpose: the snapshot is
   * read across a serialisation boundary by tools that must be able to tell
   * "refinement never hit the ceiling" from "it hit it and declined nothing",
   * and a `{declinedPathCount: 0}` object is indistinguishable from the field
   * simply being present-but-empty on an older build.
   *
   * Called on every `getState()`, so the sample is taken with a BOUNDED loop
   * rather than `Array.from(set).slice(…)`: the array form materialises all
   * thousands of paths just to keep eight, which is the exact allocation
   * {@link RESIDENCY_DECLINED_PATH_SAMPLE} exists to avoid.
   */
  snapshot(): RefinementResidencyStop | undefined {
    if (this.declinedPaths.size === 0) return undefined;
    const sample: string[] = [];
    for (const path of this.declinedPaths) {
      if (sample.length >= RESIDENCY_DECLINED_PATH_SAMPLE) break;
      sample.push(path);
    }
    return {
      reason: this.first.reason,
      residentBytes: this.first.residentBytes,
      budgetBytes: this.first.budgetBytes,
      firstPath: this.first.path,
      declinedPathCount: this.declinedPaths.size,
      declinedPaths: sample,
    };
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
    private readonly reporter = new RefinementResidencyReporter(),
    /**
     * Optional projected-density gate. Consulted only for a rung the byte
     * ceiling admits; its refusal shares this budget's `declined` set so the
     * loop retires the loader for the run (non-sticky ACROSS runs — the gate
     * is re-evaluated at every run start and re-kicked on zoom).
     */
    private readonly densityGate: RefinementDensityGate | null = null
  ) {
    for (const [path, residency] of initialResidencies) {
      this.perPath.set(path, ladderResidentBytes(residency));
    }
  }

  /**
   * Build one sized from the device heap, matching the eager admission gate.
   *
   * `initialResidencies` is FIRST and REQUIRED, not an optional tail argument,
   * because two separate guarantees rest on it and both fail silently when it
   * is missing: resident bytes of already-complete loaders are forgotten (the
   * ceiling ratchets upward, #2430), and — since the fair-share allowance
   * divides headroom by the paths the budget currently knows — the first node
   * admitted is handed the entire scene's headroom and can spend the whole
   * ladder in one pass. Pass an empty iterable to opt out deliberately.
   */
  static forSession(
    initialResidencies: Iterable<readonly [string, LadderResidency]>,
    poolOverrideBytes?: number,
    fallbackPoolBytes?: number,
    reporter?: RefinementResidencyReporter,
    densityGate?: RefinementDensityGate | null
  ): RefinementResidencyBudget {
    return new RefinementResidencyBudget(
      computeWorkingSetBudgetBytes(undefined, poolOverrideBytes, fallbackPoolBytes),
      initialResidencies,
      reporter,
      densityGate ?? null
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
   * Record `path`'s current footprint and return whether it may refine plus its
   * per-pass `allowanceBytes`. A refusal is sticky for the rest of the run:
   * re-offering the same loader every pass would re-measure, re-refuse and
   * re-log without making progress. An admission immediately reserves only its
   * estimated next rung. `runProgressiveRefinement` awaits loaders serially, so
   * the wrapper's measured `record()` replaces that estimate before the next
   * loader is admitted and the gap to the larger allowance is never observed.
   */
  admit(path: string, residency: LadderResidency): RefinementPassAdmission {
    const accounted = ladderResidentBytes(residency);
    const nextRungBytes = estimateNextRungBytes(accounted, residency.loadedRungs);
    this.perPath.set(path, accounted);
    const verdict = planRefinementAdmission(this.residentBytes, nextRungBytes, this.budgetBytes);
    const headroomBytes = Math.max(0, this.budgetBytes - verdict.residentBytes);
    const eligiblePathCount = Math.max(1, this.perPath.size - this.declined.size);
    const allowanceBytes =
      this.budgetBytes > 0
        ? Math.max(nextRungBytes, Math.floor(headroomBytes / eligiblePathCount))
        : null;
    if (!verdict.admitted) {
      this.declined.add(path);
      this.reporter.reportOnce(path, verdict);
      return { ...verdict, allowanceBytes };
    }
    // Bytes fit; now the screen. A density refusal retires the loader for this
    // run exactly like a byte refusal (same `declined` set, same closures), but
    // it is the gate's to report and it reserves nothing.
    const density = this.densityGate?.admit(path, residency);
    if (density && !density.admitted) {
      this.declined.add(path);
      return { ...verdict, admitted: false, reason: 'density-cap', allowanceBytes };
    }
    this.perPath.set(path, accounted + nextRungBytes);
    return { ...verdict, allowanceBytes };
  }

  /** Replace an admission estimate with the loader's measured post-pass footprint. */
  record(path: string, residency: LadderResidency): void {
    this.perPath.set(path, ladderResidentBytes(residency));
  }
}
