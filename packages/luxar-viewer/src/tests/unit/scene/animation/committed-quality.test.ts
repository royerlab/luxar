/**
 * Unit tests for the committed-quality probe (scene/animation/committed-quality).
 *
 * The probe exists so the pacing feedback can tell "the loaders cannot keep up"
 * from "the playhead slowed to wait for them" (#2374). Its contract is a
 * MINIMUM over visible stamped nodes, and a `null` that means "cannot tell"
 * rather than good or bad news.
 */

import { describe, it, expect } from 'vitest';
import {
  worstCommittedEnergy,
  type QualityNode,
} from '../../../../scene/animation/committed-quality';

const stamped = (energy: number, extra: Partial<QualityNode> = {}): QualityNode => ({
  userData: { committedEnergyFraction: energy },
  ...extra,
});

describe('worstCommittedEnergy', () => {
  it('reduces with the MINIMUM, not a mean — the thinnest node is what shows', () => {
    const root: QualityNode = { children: [stamped(0.95), stamped(0.12), stamped(0.8)] };
    expect(worstCommittedEnergy(root)).toBeCloseTo(0.12);
  });

  it('finds stamps at any depth', () => {
    const root: QualityNode = { children: [{ children: [{ children: [stamped(0.3)] }] }] };
    expect(worstCommittedEnergy(root)).toBeCloseTo(0.3);
  });

  it('ignores nodes that are not visible, and their subtrees', () => {
    // A hidden LOD level's stamp describes something nobody can see, so letting
    // it drag the figure down would report a problem the viewer does not have.
    const root: QualityNode = {
      children: [stamped(0.9), stamped(0.01, { visible: false })],
    };
    expect(worstCommittedEnergy(root)).toBeCloseTo(0.9);

    const hiddenParent: QualityNode = { visible: false, children: [stamped(0.01)] };
    expect(worstCommittedEnergy({ children: [stamped(0.9), hiddenParent] })).toBeCloseTo(0.9);
  });

  it('returns null when nothing is stamped — "cannot tell", not "fine"', () => {
    expect(worstCommittedEnergy({ children: [{}, { children: [{}] }] })).toBeNull();
    expect(worstCommittedEnergy(null)).toBeNull();
    expect(worstCommittedEnergy(undefined)).toBeNull();
  });

  it('ignores non-numeric and non-finite stamps rather than propagating them', () => {
    const root: QualityNode = {
      children: [
        { userData: { committedEnergyFraction: 'lots' } },
        { userData: { committedEnergyFraction: Number.NaN } },
        stamped(0.5),
      ],
    };
    expect(worstCommittedEnergy(root)).toBeCloseTo(0.5);
  });

  it('accepts 0 as a real reading, not as absent', () => {
    // A ladder that has committed nothing is the case the feedback most needs to
    // report, so a falsy-but-present 0 must not be skipped.
    expect(worstCommittedEnergy({ children: [stamped(0)] })).toBe(0);
  });
});
