/**
 * Unit tests for the progressive-refinement residency budget (#2426).
 *
 * The budget is the only thing that bounds an additive ladder at rest: the
 * ladder's terminal state is 100% of every node, so without a ceiling a scene
 * like Laniakea (10 line nodes x ~1.3M segments) climbs until the tab dies.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  estimateNextRungBytes,
  ladderResidentBytes,
  planRefinementAdmission,
  RefinementResidencyBudget,
  RefinementResidencyReporter,
} from '../../../../../data/scene-loader/progressive/residency-budget';

const MB = 1024 * 1024;

describe('planRefinementAdmission', () => {
  it('admits a rung that fits with room to spare', () => {
    const v = planRefinementAdmission(10 * MB, 5 * MB, 100 * MB);
    expect(v).toMatchObject({ admitted: true, reason: 'ok' });
  });

  it('refuses once already at the ceiling', () => {
    const v = planRefinementAdmission(100 * MB, 1 * MB, 100 * MB);
    expect(v).toMatchObject({ admitted: false, reason: 'over-budget' });
  });

  it('admits a zero-cost first rung even when the scene is already at the ceiling', () => {
    const v = planRefinementAdmission(100 * MB, 0, 100 * MB);
    expect(v).toMatchObject({ admitted: true, reason: 'ok' });
  });

  it('refuses a rung that would cross the ceiling', () => {
    const v = planRefinementAdmission(90 * MB, 20 * MB, 100 * MB);
    expect(v).toMatchObject({ admitted: false, reason: 'next-rung-would-exceed' });
  });

  it('admits a rung that lands exactly on the ceiling', () => {
    // The ceiling is a budget, not a hazard line — spending it exactly is the
    // intended use, and refusing here would waste the last rung of every scene.
    expect(planRefinementAdmission(90 * MB, 10 * MB, 100 * MB).admitted).toBe(true);
  });

  describe('never refuses on an absent signal', () => {
    // Production uses a device-class fallback when `performance.memory` is
    // absent. This branch is defensive for direct callers with no usable budget.
    it.each([0, -1, Number.NaN])('admits when the budget is %p', (budget) => {
      const v = planRefinementAdmission(10 * MB, 5 * MB, budget as number);
      expect(v).toMatchObject({ admitted: true, reason: 'unbudgeted' });
    });
  });

  it('clamps negative inputs rather than trusting them', () => {
    const v = planRefinementAdmission(-5 * MB, -1 * MB, 100 * MB);
    expect(v).toMatchObject({ admitted: true, residentBytes: 0 });
  });
});

describe('ladderResidentBytes — the element-row term', () => {
  // The bug this exists to prevent: budgeting on the decoded payload alone.
  // A Lines element row is 96 B (6 RGBA32F texels) against a ~27 B/vertex
  // payload; a GSplats row is 64 B against ~43 B/splat. Counting only the
  // payload therefore under-reports Lines by ~2.4x MORE than GSplats, and the
  // shared sweep ceiling declined a 598 MB gsplat node while admitting a 901 MB
  // lines node — useless for the geometry the cap was written for.
  it('adds the element rows to the decoded payload', () => {
    expect(
      ladderResidentBytes({
        residentBytes: 27 * 1000,
        loadedRungs: 1,
        elementCount: 1000,
        bytesPerElement: 96,
      })
    ).toBe(27_000 + 96_000);
  });

  it('keeps lines above gsplats for equal element counts, as the layouts require', () => {
    const N = 1_000_000;
    const lines = ladderResidentBytes({
      residentBytes: 27 * N,
      loadedRungs: 4,
      elementCount: N,
      bytesPerElement: 96,
    });
    const gsplats = ladderResidentBytes({
      residentBytes: 43 * N,
      loadedRungs: 4,
      elementCount: N,
      bytesPerElement: 64,
    });
    // Payload alone would rank gsplats HEAVIER (43 vs 27 B/element), which is
    // the inversion that let the cap admit the heavier lines node.
    expect(27 * N).toBeLessThan(43 * N);
    expect(lines).toBeGreaterThan(gsplats);
  });

  it('treats a geometry with no element rows as payload-only', () => {
    // Mesh is not element-texture backed; zero is the honest value.
    expect(
      ladderResidentBytes({
        residentBytes: 500,
        loadedRungs: 2,
        elementCount: 0,
        bytesPerElement: 0,
      })
    ).toBe(500);
  });

  it('clamps negative inputs', () => {
    expect(
      ladderResidentBytes({
        residentBytes: -1,
        loadedRungs: 1,
        elementCount: -5,
        bytesPerElement: -96,
      })
    ).toBe(0);
  });
});

describe('estimateNextRungBytes', () => {
  it('is the mean rung so far', () => {
    expect(estimateNextRungBytes(30 * MB, 3)).toBe(10 * MB);
  });

  it('estimates zero for a node that has loaded nothing', () => {
    // A node with no rungs must never be refused its first one, or a scene
    // already over budget would never paint at all.
    expect(estimateNextRungBytes(0, 0)).toBe(0);
    expect(estimateNextRungBytes(10 * MB, 0)).toBe(0);
  });

  it('under-estimates a geometric ladder, which is the safe direction', () => {
    // `stream:C` doubles: rungs 1+2+4+8 = 15 resident over 4 rungs -> mean
    // 3.75, while the real next rung is 16. Erring low costs at most one
    // overshoot (the hard stop catches it next pass); erring high would stall
    // equal-count ladders that were never in danger.
    const resident = (1 + 2 + 4 + 8) * MB;
    expect(estimateNextRungBytes(resident, 4)).toBeLessThan(16 * MB);
  });

  it('is exact for an equal-count ladder', () => {
    expect(estimateNextRungBytes(4 * MB, 4)).toBe(1 * MB);
  });

  it('survives the Lines fold, where per-rung sizes no longer exist', () => {
    // After #2427 the Lines loader holds ONE merged payload; only the total and
    // the rung count survive, and both are inputs here.
    expect(estimateNextRungBytes(120 * MB, 6)).toBe(20 * MB);
  });
});

describe('RefinementResidencyBudget', () => {
  let warn: ReturnType<typeof vi.spyOn>;

  beforeEach(async () => {
    const { log } = await import('../../../../../utils/log');
    warn = vi.spyOn(log, 'warning').mockImplementation(() => undefined);
  });
  afterEach(() => warn.mockRestore());

  const residency = (residentBytes: number, loadedRungs = 1) => ({
    residentBytes,
    loadedRungs,
    elementCount: 0,
    bytesPerElement: 0,
  });

  it('sums footprints across nodes, which is the whole point', () => {
    // Laniakea's shape: each node is individually affordable and the scene is
    // collectively fatal. A per-node budget would admit all ten.
    const budget = new RefinementResidencyBudget(100 * MB);
    for (let i = 0; i < 4; i++) {
      expect(budget.admit(`/node${i}`, residency(20 * MB, 20)).admitted).toBe(true);
    }
    // Four measured 20 MiB ladders plus one 1 MiB reservation apiece.
    expect(budget.residentBytes).toBe(84 * MB);
    // The fifth crosses the ceiling collectively, though it is the same size.
    expect(budget.admit('/node4', residency(20 * MB, 20))).toMatchObject({
      admitted: false,
      reason: 'over-budget',
    });
  });

  it('re-measures a node before reserving its next rung', () => {
    // `admit` receives the node's CURRENT total, not a delta. Each call replaces
    // the previous reservation before authorising one estimated next rung.
    const budget = new RefinementResidencyBudget(1000 * MB);
    budget.admit('/n', residency(10 * MB, 1));
    budget.admit('/n', residency(20 * MB, 2));
    budget.admit('/n', residency(30 * MB, 3));
    expect(budget.residentBytes).toBe(40 * MB);
  });

  it('returns remaining headroom and reconciles multi-rung pass growth', () => {
    const budget = new RefinementResidencyBudget(100 * MB, [['/other', residency(60 * MB, 3)]]);

    const admission = budget.admit('/n', residency(20 * MB, 2));
    expect(admission).toMatchObject({ admitted: true, allowanceBytes: 20 * MB });
    expect(budget.residentBytes).toBe(90 * MB);

    budget.record('/n', residency(35 * MB, 4));
    expect(budget.residentBytes).toBe(95 * MB);
    expect(budget.admit('/next', residency(10 * MB, 1)).admitted).toBe(false);
  });

  it('seeds completed and pending loaders before the first admission', () => {
    const budget = new RefinementResidencyBudget(
      100 * MB,
      new Map([
        ['/complete', residency(60 * MB, 3)],
        ['/pending', residency(20 * MB, 1)],
      ])
    );

    expect(budget.residentBytes).toBe(80 * MB);
    expect(budget.admit('/pending', residency(20 * MB, 1))).toMatchObject({
      admitted: true,
      reason: 'ok',
    });
    expect(budget.residentBytes).toBe(100 * MB);
  });

  it('reserves an admitted rung so later nodes in the same pass see it', () => {
    const budget = new RefinementResidencyBudget(
      100 * MB,
      new Map([
        ['/a', residency(40 * MB, 2)],
        ['/b', residency(40 * MB, 2)],
      ])
    );

    expect(budget.admit('/a', residency(40 * MB, 2)).admitted).toBe(true);
    expect(budget.residentBytes).toBe(100 * MB);
    expect(budget.admit('/b', residency(40 * MB, 2))).toMatchObject({
      admitted: false,
      reason: 'over-budget',
    });
  });

  it('keeps the ceiling stable when a later run is seeded from current ladders', () => {
    const firstRun = new RefinementResidencyBudget(
      100 * MB,
      new Map([
        ['/a', residency(40 * MB, 2)],
        ['/b', residency(40 * MB, 2)],
      ])
    );
    expect(firstRun.admit('/a', residency(40 * MB, 2)).admitted).toBe(true);

    const secondRun = new RefinementResidencyBudget(
      100 * MB,
      new Map([
        ['/a', residency(60 * MB, 3)],
        ['/b', residency(40 * MB, 2)],
      ])
    );
    expect(secondRun.admit('/a', residency(60 * MB, 3))).toMatchObject({
      admitted: false,
      reason: 'over-budget',
    });
  });

  it('makes a refusal sticky for the rest of the run', () => {
    const budget = new RefinementResidencyBudget(10 * MB);
    expect(budget.isDeclined('/n')).toBe(false);
    budget.admit('/n', residency(50 * MB, 5));
    expect(budget.isDeclined('/n')).toBe(true);
    // Sticky even if the node later reports a footprint that would fit — a
    // loader re-offered every frame would otherwise re-measure and re-refuse
    // forever while holding the update lock.
    budget.admit('/n', residency(1, 1));
    expect(budget.isDeclined('/n')).toBe(true);
  });

  it('does not decline nodes that fit', () => {
    const budget = new RefinementResidencyBudget(100 * MB);
    budget.admit('/small', residency(1 * MB, 4));
    expect(budget.isDeclined('/small')).toBe(false);
  });

  it('never refuses a node its first rung', () => {
    const budget = new RefinementResidencyBudget(
      10 * MB,
      new Map([['/complete', residency(10 * MB, 1)]])
    );
    expect(budget.admit('/fresh', residency(0, 0)).admitted).toBe(true);
  });

  it('reports the ceiling once across view-triggered runs', () => {
    // A scene parked at the ceiling starts a new run after every view change.
    // Sharing the scene-owned reporter keeps that one fact from flooding the
    // console during slice playback or dimension animation.
    const reporter = new RefinementResidencyReporter();
    const firstRun = RefinementResidencyBudget.forSession(30 * MB, undefined, [], reporter);
    const secondRun = RefinementResidencyBudget.forSession(30 * MB, undefined, [], reporter);
    for (let i = 0; i < 5; i++) firstRun.admit(`/first${i}`, residency(50 * MB, 5));
    for (let i = 0; i < 5; i++) secondRun.admit(`/second${i}`, residency(50 * MB, 5));
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][1])).toContain('residency ceiling');
  });

  it('never declines when the budget is unknown', () => {
    const budget = new RefinementResidencyBudget(0);
    expect(budget.admit('/n', residency(500 * MB, 5)).admitted).toBe(true);
    expect(budget.isDeclined('/n')).toBe(false);
    expect(warn).not.toHaveBeenCalled();
  });
});
