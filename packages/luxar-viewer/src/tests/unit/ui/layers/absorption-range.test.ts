/**
 * Per-layer absorption (κ) slider range.
 *
 * The regression these pin: κ is 1/length, so a fixed 0–10 track tops out
 * at τ = κ·α·width·chord ≈ 0.012 for a 1.5e-3-wide line (a sub-1/255 change
 * — a knob that visibly does nothing), while the same 0–10 is right for a
 * fitted gsplat volume. The derived bound must scale with the geometry
 * thickness the writer records.
 */

import { describe, it, expect } from 'vitest';
import type { SceneNode } from '../../../../data/data-loader-types';
import {
  ABSORPTION_DEFAULT_MAX,
  ABSORPTION_LOG_DECADES,
  ABSORPTION_LOG_DECADES_MAX,
  ABSORPTION_MAX_LIMIT,
  ABSORPTION_TAU_TARGET,
  absorptionMaxForNode,
  absorptionSliderRange,
  formatAbsorption,
} from '../../../../ui/layers/absorption-range';
import { LINE_CHORD_SCALE } from '../../../../rendering/materials/line/math';
import { POINT_CHORD_SCALE } from '../../../../rendering/materials/point/math';

function node(
  type: SceneNode['type'],
  attrs: Record<string, unknown> = {},
  children?: SceneNode[]
): SceneNode {
  return { path: `/${type}`, name: type, type, attrs, children } as unknown as SceneNode;
}

describe('absorptionMaxForNode', () => {
  it('derives the bound from a lines node max_width so the top of the track is opaque', () => {
    const width = 0.0015; // the 3D-Hilbert-curve demo's line width
    const max = absorptionMaxForNode(node('lines', { max_width: width }));

    // τ at the top of the track must be the "clearly opaque" target.
    expect(max * width * LINE_CHORD_SCALE).toBeCloseTo(ABSORPTION_TAU_TARGET, 6);
    // And it must be far above the old fixed bound — that is the bug.
    expect(max).toBeGreaterThan(100 * ABSORPTION_DEFAULT_MAX);
  });

  it('derives the bound from a points node max_radius', () => {
    const radius = 0.02;
    const max = absorptionMaxForNode(node('points', { max_radius: radius }));
    expect(max * radius * POINT_CHORD_SCALE).toBeCloseTo(ABSORPTION_TAU_TARGET, 6);
  });

  it('falls back to the default for gsplats (τ = κ·opacity·rayMass is already O(1)-calibrated)', () => {
    expect(absorptionMaxForNode(node('gsplats', { n_splats: 1000 }))).toBe(ABSORPTION_DEFAULT_MAX);
  });

  it('falls back to the default when the thickness stat is missing or degenerate', () => {
    expect(absorptionMaxForNode(node('lines', {}))).toBe(ABSORPTION_DEFAULT_MAX);
    expect(absorptionMaxForNode(node('lines', { max_width: 0 }))).toBe(ABSORPTION_DEFAULT_MAX);
    expect(absorptionMaxForNode(node('lines', { max_width: -1 }))).toBe(ABSORPTION_DEFAULT_MAX);
    expect(absorptionMaxForNode(node('lines', { max_width: NaN }))).toBe(ABSORPTION_DEFAULT_MAX);
    expect(absorptionMaxForNode(node('points', { max_radius: 'fat' }))).toBe(
      ABSORPTION_DEFAULT_MAX
    );
  });

  it('never drops BELOW the default, so authored κ ≤ 10 stays reachable on fat geometry', () => {
    // A 50-unit-wide line would derive κ_max ≈ 0.086 on its own.
    expect(absorptionMaxForNode(node('lines', { max_width: 50 }))).toBe(ABSORPTION_DEFAULT_MAX);
  });

  it('clamps a degenerate (near-zero) thickness to the hard ceiling', () => {
    expect(absorptionMaxForNode(node('lines', { max_width: 1e-30 }))).toBe(ABSORPTION_MAX_LIMIT);
  });

  it('takes the THINNEST descendant for a group layer (one κ drives the whole subtree)', () => {
    const group = node('group', {}, [
      node('lines', { max_width: 0.5 }),
      node('lines', { max_width: 0.002 }), // thinnest → needs the largest κ
      node('gsplats', {}),
    ]);
    const max = absorptionMaxForNode(group);
    expect(max * 0.002 * LINE_CHORD_SCALE).toBeCloseTo(ABSORPTION_TAU_TARGET, 6);
  });

  it('walks nested kind=lod / kind=partition subtrees', () => {
    const partition = node('group', { kind: 'partition' }, [
      node('group', { kind: 'lod' }, [node('lines', { max_width: 0.004 })]),
    ]);
    const max = absorptionMaxForNode(partition);
    expect(max * 0.004 * LINE_CHORD_SCALE).toBeCloseTo(ABSORPTION_TAU_TARGET, 6);
  });
});

describe('absorptionSliderRange', () => {
  it('spans the configured decades below the layer bound', () => {
    const { min, max } = absorptionSliderRange(2000, 1);
    expect(max).toBe(2000);
    expect(min).toBeCloseTo(2000 / Math.pow(10, ABSORPTION_LOG_DECADES), 9);
  });

  it('widens to keep an authored κ above the derived bound on the track', () => {
    const { max } = absorptionSliderRange(ABSORPTION_DEFAULT_MAX, 250);
    expect(max).toBe(250);
  });

  it('LOWERS the floor to keep a κ below the nominal minimum on the track', () => {
    // Regression: very thin geometry derives a large max, whose nominal
    // 4-decade floor can sit ABOVE the authored default κ = 1. The readout
    // would then show 1.00 while the thumb could not represent it, and the
    // first drag would silently jump κ up to the floor.
    const thin = absorptionMaxForNode(node('lines', { max_width: 6e-4 }));
    expect(thin / Math.pow(10, ABSORPTION_LOG_DECADES)).toBeGreaterThan(1); // nominal floor > κ=1
    const { min, max } = absorptionSliderRange(thin, 1);
    expect(min).toBe(1); // floor lowered exactly onto the current κ
    expect(max).toBe(thin); // top of the track still the opaque point
  });

  it('bounds how far the floor is lowered for a κ that is already ~zero', () => {
    const { min, max } = absorptionSliderRange(4035.77, 1e-30);
    expect(min).toBeCloseTo(max / Math.pow(10, ABSORPTION_LOG_DECADES_MAX), 12);
  });

  it('caps the widening at the hard ceiling so an absurd authored κ keeps a usable track', () => {
    expect(absorptionSliderRange(ABSORPTION_DEFAULT_MAX, 1e30).max).toBe(ABSORPTION_MAX_LIMIT);
  });

  it('floors at the default bound', () => {
    expect(absorptionSliderRange(0.5, 0).max).toBe(ABSORPTION_DEFAULT_MAX);
    expect(absorptionSliderRange(NaN, NaN).max).toBe(ABSORPTION_DEFAULT_MAX);
  });

  describe('at the clamped extremes', () => {
    /**
     * Outside the two deliberate clamps the thumb seats at the clamped end
     * while the readout shows the true κ, so a touch writes the clamp back.
     * These pin that documented trade-off — and the reason it is acceptable:
     * the swapped states are visually identical. τ = κ · thickness · chord
     * for the Hilbert-demo line (thickness 1.5e-3, chord √(π/ln 100)).
     */
    const TAU = (k: number) => k * 0.0015 * Math.sqrt(Math.PI / Math.log(100));

    it('below the span cap: both κ are ≥ 7 decades below visible absorption', () => {
      const { min, max } = absorptionSliderRange(4035.77, 1e-30);
      expect(min).toBeCloseTo(max / Math.pow(10, ABSORPTION_LOG_DECADES_MAX), 12);
      expect(1e-30).toBeLessThan(min); // off-track: a touch would write `min`
      // The value the touch would write is still optically nothing.
      expect(TAU(min)).toBeLessThan(1e-7);
      expect(TAU(1e-30)).toBeLessThan(1e-7);
    });

    it('above the ceiling: both κ are far past opaque', () => {
      const { max } = absorptionSliderRange(ABSORPTION_DEFAULT_MAX, 1e30);
      expect(max).toBe(ABSORPTION_MAX_LIMIT);
      expect(1e30).toBeGreaterThan(max); // off-track: a touch would write `max`
      // 1 − e^(−τ) is 1.0 to double precision on both sides of the clamp.
      expect(1 - Math.exp(-TAU(max))).toBe(1);
      expect(1 - Math.exp(-TAU(1e30))).toBe(1);
    });
  });
});

describe('formatAbsorption', () => {
  it('keeps the readout short and informative across decades', () => {
    expect(formatAbsorption(0)).toBe('0');
    expect(formatAbsorption(-1)).toBe('0');
    expect(formatAbsorption(0.001)).toBe('1.0e-3');
    expect(formatAbsorption(1)).toBe('1.00');
    expect(formatAbsorption(12.34)).toBe('12.3');
    expect(formatAbsorption(2858.5)).toBe('2859');
  });
});
