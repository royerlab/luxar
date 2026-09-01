/**
 * Shared fold/residency contract for the four progressive LOD loaders
 * (GSplats, Points, Lines, Mesh).
 *
 * All four now FOLD their ladder: `concatenateMemoized` builds one cumulative
 * payload, drops the per-rung parts, and (where the geometry has one) releases
 * the sub-loader's pooled accumulator. That is what took terminal residency
 * from ~2x the geometry down toward ~1x (#2426, #2427, #2429).
 *
 * WHY THIS IS A STRUCTURAL ASSERTION AND NOT A HEAP MEASUREMENT. A
 * before/after byte number cannot distinguish "released" from "not yet
 * collected", drifts with unrelated changes, and — the point that actually
 * matters — does not FAIL when someone regresses it. The fold's effect is a
 * clean structural prediction (parts 100% + cumulative 100% -> cumulative
 * only), so it is testable exactly. Byte measurements remain the right
 * instrument for sizing the win; they are the wrong one for defending it.
 *
 * WHY SHARED. The pattern is duplicated four times over, and a regression in
 * whichever loader gets touched next would otherwise be invisible — the other
 * three would stay green. Parameterising one contract over all four means a
 * fifth geometry, or a fold rewritten in one place, cannot quietly skip it.
 *
 * The three properties, and what each one catches:
 *
 *  1. FOLDS TO A SINGLE PAYLOAD — the fold happened at all. Catches a loader
 *     that concatenates but forgets to drop the parts, which is the original
 *     ~2x bug.
 *  2. RESIDENCY REPORTS THE CUMULATIVE, IN LOGICAL LEVELS — `ladderResidency()`
 *     feeds the scene-wide budget, and it must report the merged payload's
 *     bytes against the LOGICAL rung count. Reporting `loadedLODs.length` (1,
 *     post-fold) would make the budget's mean-rung estimate read the whole
 *     merged blob as a single enormous next rung and stall refinement several
 *     rungs early.
 *  3. THE MULTI-PASS CLIMB ALSO FOLDS — exercises the real refinement shape,
 *     where each pass can build on the previous cumulative. The single-pass
 *     case concatenates only once and cannot expose retention between passes.
 * @module tests/unit/data/_shared/ladder-fold-contract
 */

import { describe, it, expect } from 'vitest';

/** The surface this contract probes, including two private fields. */
export interface FoldableLadderLoader {
  readonly loadedLODCount: number;
  readonly totalLODCount: number;
  ladderResidency(): { residentBytes: number; loadedRungs: number };
  rollbackToPassStart(): number;
}

/** What a per-geometry test file supplies to run the contract. */
export interface LadderFoldSubject {
  /** A loader wired to `totalLevels` fast, cache-resident sub-LOD stubs. */
  loader: FoldableLadderLoader;
  /** Drive the loader until every rung is loaded and folded. */
  loadAll(): Promise<void>;
  totalLevels: number;
  /**
   * A loader whose sub-LODs report NON-resident, so the streaming
   * policy stops after each level and the ladder is climbed over several
   * passes. That is the shape real refinement takes, and it is materially
   * different from a single resident pass — the single-pass climb concatenates
   * exactly once, so it has no previous cumulative and cannot exhibit the
   * abort-path lineage retention that can pin up to a whole redundant copy at
   * depth (#2426). Committing browser runs retain the same depth curve before
   * and after the release; the multi-pass case protects the uncommitted exit,
   * not a settled deep-rung saving. A contract that only ever ran the
   * single-pass shape looked green through that entire bug.
   */
  multiPass: { loader: FoldableLadderLoader; loadAll(): Promise<void>; totalLevels: number };
}

/**
 * Read a private field for assertion purposes.
 *
 * Deliberate: the fold's whole effect is on internal retention, so a contract
 * that only used the public surface could not tell a folded loader from an
 * unfolded one — which is precisely the regression this exists to catch.
 */
function priv<T>(loader: object, field: string): T {
  return (loader as unknown as Record<string, T>)[field];
}

/**
 * Run the shared fold/residency contract against one geometry's loader.
 *
 * Call from inside the per-geometry test file's own `describe`, passing a
 * factory that builds a loader over N cheap resident sub-LOD stubs.
 */
export function testLadderFoldContract(
  geometry: string,
  makeSubject: () => Promise<LadderFoldSubject>
): void {
  describe(`${geometry} ladder fold contract`, () => {
    it('folds a fully-loaded ladder to a single payload', async () => {
      const { loader, loadAll, totalLevels } = await makeSubject();
      await loadAll();

      expect(loader.loadedLODCount).toBe(totalLevels);
      // The fold: one cumulative payload retained, not one per rung. A loader
      // that concatenates without dropping the parts fails here — that is the
      // ~2x terminal residency this contract defends against.
      expect(priv<unknown[]>(loader, 'loadedLODs')).toHaveLength(1);
    });

    it('reports residency as the cumulative payload against LOGICAL rungs', async () => {
      const { loader, loadAll, totalLevels } = await makeSubject();
      await loadAll();

      const residency = loader.ladderResidency();
      // Logical levels, not payload entries. Reporting 1 here would make the
      // budget's mean-rung estimate treat the merged blob as one enormous next
      // rung and stall refinement several rungs early.
      expect(residency.loadedRungs).toBe(totalLevels);
      expect(residency.residentBytes).toBeGreaterThan(0);
    });
    it('still folds to a single payload when climbed over MANY passes', async () => {
      const { loader: l, loadAll, totalLevels } = (await makeSubject()).multiPass;

      await loadAll();
      while (l.loadedLODCount < totalLevels) await loadAll();

      // The shape real refinement takes. The single-pass case above concatenates
      // exactly once; this one concatenates per pass, which is where retention
      // between passes shows up.
      expect(l.loadedLODCount).toBe(totalLevels);
      expect(priv<unknown[]>(l, 'loadedLODs')).toHaveLength(1);
      expect(l.ladderResidency().loadedRungs).toBe(totalLevels);
    });
  });
}
