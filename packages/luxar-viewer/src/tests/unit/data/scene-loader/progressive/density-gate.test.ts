/**
 * Projected-density rung gate (`density-gate.ts`): the pure admission rule, the
 * session gate's deferral/resume bookkeeping, and its integration with the
 * residency budget's `declined` set (which is what retires a deferred loader
 * for the run instead of re-offering it at frame rate).
 */
import { beforeEach, describe, expect, it } from 'vitest';

import {
  RefinementDensityGate,
  estimateNextRungElements,
  planDensityAdmission,
  type DensityGateCaps,
  type ProjectedDensitySample,
} from '../../../../../data/scene-loader/progressive/density-gate';
import {
  RefinementResidencyBudget,
  type LadderResidency,
} from '../../../../../data/scene-loader/progressive/residency-budget';
import { getLoadTimeline, resetLoadTimeline } from '../../../../../profiling/load-timeline';

const CAPS: DensityGateCaps = { blendable: 4, nonBlendable: 1 };
const MB = 1024 * 1024;

function residency(elementCount: number, loadedRungs: number): LadderResidency {
  return { residentBytes: 43 * elementCount, loadedRungs, elementCount, bytesPerElement: 64 };
}
function sample(areaPx: number, elements: number, blendable = true): ProjectedDensitySample {
  return { areaPx, elements, onScreen: areaPx > 0, blendable };
}

describe('estimateNextRungElements', () => {
  it('is the mean rung so far, zero for an unloaded node', () => {
    expect(estimateNextRungElements(residency(1000, 4))).toBe(250);
    expect(estimateNextRungElements(residency(0, 0))).toBe(0);
    expect(estimateNextRungElements(residency(1000, 0))).toBe(0);
  });
});

describe('planDensityAdmission', () => {
  it('admits while the node plus its next rung stays under the cap', () => {
    // 1000 el in 1000 px, next rung 250 → 1.25 el/px ≤ 4.
    const v = planDensityAdmission(sample(1000, 1000), residency(1000, 4), CAPS);
    expect(v.admitted).toBe(true);
    expect(v.predictedElementsPerPixel).toBeCloseTo(1.25, 9);
    expect(v.cap).toBe(4);
  });

  it('defers once the next rung would cross the cap, and says at what footprint it fits', () => {
    // 1_000_000 el in 1854 px (the audit lattice), next rung 250 000 → 674 el/px.
    const v = planDensityAdmission(sample(1854, 1_000_000), residency(1_000_000, 4), CAPS);
    expect(v.admitted).toBe(false);
    expect(v.predictedElementsPerPixel).toBeGreaterThan(600);
    expect(v.resumeAreaPx).toBeCloseTo(1_250_000 / 4, 6);
  });

  it('uses the tighter cap for a non-blendable node', () => {
    // 2 el/px predicted: fine for additive (cap 4), deferred for max (cap 1).
    const blendable = planDensityAdmission(sample(1000, 1500), residency(1500, 3), CAPS);
    const max = planDensityAdmission(sample(1000, 1500, false), residency(1500, 3), CAPS);
    expect(blendable.admitted).toBe(true);
    expect(max.admitted).toBe(false);
    expect(max.cap).toBe(1);
  });

  it('never refuses a first rung or an off-screen node', () => {
    expect(planDensityAdmission(sample(1, 0), residency(0, 0), CAPS).admitted).toBe(true);
    expect(planDensityAdmission(sample(0, 1e6), residency(1e6, 4), CAPS).admitted).toBe(true);
  });
});

describe('RefinementDensityGate', () => {
  beforeEach(() => resetLoadTimeline());

  it('has no opinion without a measurement, defers with one, and counts the deferral', () => {
    const samples = new Map<string, ProjectedDensitySample>();
    const gate = new RefinementDensityGate((p) => samples.get(p), CAPS);
    expect(gate.admit('/unknown', residency(100, 1))).toBeNull();
    samples.set('/dense', sample(1854, 1_000_000));
    const v = gate.admit('/dense', residency(1_000_000, 4));
    expect(v?.admitted).toBe(false);
    expect(gate.deferredCount).toBe(1);
    expect(getLoadTimeline().refinement.densityDeferred).toBe(1);
  });

  it('resumes a deferred path only once its footprint has grown past the resume area', () => {
    const samples = new Map<string, ProjectedDensitySample>();
    const gate = new RefinementDensityGate((p) => samples.get(p), CAPS);
    samples.set('/dense', sample(1854, 1_000_000));
    gate.admit('/dense', residency(1_000_000, 4)); // resume at 312 500 px
    expect(gate.takeResumable()).toEqual([]);
    samples.set('/dense', sample(200_000, 1_000_000)); // zoomed, still 5 el/px predicted 6.25
    expect(gate.takeResumable()).toEqual([]);
    samples.set('/dense', sample(400_000, 1_000_000)); // 3.1 el/px predicted → fits
    expect(gate.takeResumable()).toEqual(['/dense']);
    expect(gate.deferredCount).toBe(0);
    // Off-screen never resumes (area 0), and a resumed path is not re-reported.
    samples.set('/other', sample(10, 10_000));
    gate.admit('/other', residency(10_000, 2));
    samples.set('/other', { areaPx: 0, elements: 10_000, onScreen: false, blendable: true });
    expect(gate.takeResumable()).toEqual([]);
    expect(gate.deferredCount).toBe(1);
  });

  it('beginRun forgets the previous run’s deferrals', () => {
    const samples = new Map<string, ProjectedDensitySample>([['/d', sample(10, 10_000)]]);
    const gate = new RefinementDensityGate((p) => samples.get(p), CAPS);
    gate.admit('/d', residency(10_000, 2));
    expect(gate.deferredCount).toBe(1);
    gate.beginRun();
    expect(gate.deferredCount).toBe(0);
  });
});

describe('RefinementResidencyBudget × density gate', () => {
  it('refuses with reason density-cap, retires the path for the run, reserves nothing', () => {
    const samples = new Map<string, ProjectedDensitySample>([
      ['/dense', sample(1854, 1_000_000)],
      ['/sparse', sample(500_000, 1000)],
    ]);
    const gate = new RefinementDensityGate((p) => samples.get(p), CAPS);
    const budget = new RefinementResidencyBudget(10_000 * MB, [], undefined, gate);
    const dense = budget.admit('/dense', residency(1_000_000, 4));
    expect(dense.admitted).toBe(false);
    expect(dense.reason).toBe('density-cap');
    expect(budget.isDeclined('/dense')).toBe(true);
    // Nothing reserved beyond the measured footprint.
    expect(budget.residentBytes).toBe(43 * 1_000_000 + 64 * 1_000_000);
    // A node under its cap is admitted by both rules.
    const sparse = budget.admit('/sparse', residency(1000, 2));
    expect(sparse).toMatchObject({ admitted: true, reason: 'ok' });
    expect(budget.isDeclined('/sparse')).toBe(false);
  });

  it('is not consulted for a rung the byte ceiling already refused', () => {
    let asked = 0;
    const gate = new RefinementDensityGate(() => {
      asked += 1;
      return sample(10, 1e6);
    }, CAPS);
    const budget = new RefinementResidencyBudget(1 * MB, [], undefined, gate);
    const v = budget.admit('/big', residency(1_000_000, 4)); // ~107 MB resident ≫ 1 MB
    expect(v.admitted).toBe(false);
    expect(v.reason).toBe('over-budget');
    expect(asked).toBe(0);
  });

  it('without a gate the budget behaves as before', () => {
    const budget = new RefinementResidencyBudget(10_000 * MB, []);
    expect(budget.admit('/dense', residency(1_000_000, 4))).toMatchObject({
      admitted: true,
      reason: 'ok',
    });
  });
});
