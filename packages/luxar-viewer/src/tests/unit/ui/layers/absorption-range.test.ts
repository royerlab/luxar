/**
 * Absorption (κ) slider range.
 *
 * What these pin: the track is now **layer-independent**. Every geometry
 * family builds `τ = κ · rayMass` from the same normalised ray mass (the
 * quantity its additive branch emits), so κ ≈ 1 is the useful anchor for
 * points, lines and gsplats alike and one fixed span serves every scene.
 *
 * The regression they guard is the REMOVED behaviour: the range used to be
 * derived per layer as `ABSORPTION_TAU_TARGET / (thickness · chord)` from
 * `max_width` / `max_radius`, because the point and line shaders multiplied τ
 * by a world thickness the gsplat shader had no counterpart for. That made κ a
 * per-unit-length coefficient for two families and dimensionless for the
 * third — a units conversion smuggled into the UI, which could never serve a
 * mixed points→gsplat LOD ladder composing ONE κ over both. Re-deriving bounds
 * from geometry here would mean the shader convention has drifted apart again.
 */

import { describe, it, expect } from 'vitest';
import {
  ABSORPTION_DEFAULT_MAX,
  ABSORPTION_LOG_DECADES,
  ABSORPTION_LOG_DECADES_MAX,
  ABSORPTION_MAX_LIMIT,
  absorptionSliderRange,
  formatAbsorption,
} from '../../../../ui/layers/absorption-range';

describe('absorptionSliderRange', () => {
  it('spans the configured decades below the nominal maximum', () => {
    const { min, max } = absorptionSliderRange(1);
    expect(max).toBe(ABSORPTION_DEFAULT_MAX);
    expect(min).toBeCloseTo(ABSORPTION_DEFAULT_MAX / Math.pow(10, ABSORPTION_LOG_DECADES), 12);
  });

  it('is INDEPENDENT of geometry — the same track for every κ in the nominal span', () => {
    // The load-bearing property. A thin-line layer, a fat-point layer and a
    // fitted-gsplat layer all sit on one track now; nothing about the scene's
    // units or the node's recorded thickness can move it. If this ever needs
    // a per-layer bound again, the shaders have diverged.
    const a = absorptionSliderRange(1);
    const b = absorptionSliderRange(0.05);
    const c = absorptionSliderRange(9.9);
    expect(b).toEqual(a);
    expect(c).toEqual(a);
  });

  it('widens the top to keep an authored κ above the nominal maximum on the track', () => {
    expect(absorptionSliderRange(250).max).toBe(250);
  });

  it('keeps the nominal floor reachable when the top is widened for a large authored κ', () => {
    // Regression guard: widening the top for a large authored κ must NOT drag
    // the floor up with it, or the whole 0.001–10 nominal span becomes
    // unreachable for scenes authored with a large κ. The floor is anchored to
    // the nominal maximum, not the raised one.
    expect(absorptionSliderRange(250).min).toBeCloseTo(
      ABSORPTION_DEFAULT_MAX / Math.pow(10, ABSORPTION_LOG_DECADES),
      12
    );
    expect(absorptionSliderRange(250).max).toBe(250);
  });

  it('LOWERS the floor to keep a κ below the nominal minimum on the track', () => {
    // Without this the readout would show the true κ while the thumb could
    // not represent it, and the first drag would silently jump κ up.
    const belowFloor = ABSORPTION_DEFAULT_MAX / Math.pow(10, ABSORPTION_LOG_DECADES + 1);
    const { min, max } = absorptionSliderRange(belowFloor);
    expect(min).toBe(belowFloor); // floor lowered exactly onto the current κ
    expect(max).toBe(ABSORPTION_DEFAULT_MAX);
  });

  it('bounds how far the floor is lowered for a κ that is already ~zero', () => {
    const { min, max } = absorptionSliderRange(1e-30);
    expect(min).toBeCloseTo(max / Math.pow(10, ABSORPTION_LOG_DECADES_MAX), 12);
  });

  it('caps the widening at the hard ceiling so an absurd authored κ keeps a usable track', () => {
    expect(absorptionSliderRange(1e30).max).toBe(ABSORPTION_MAX_LIMIT);
  });

  it('falls back to the nominal maximum for κ = 0 and for a non-finite κ', () => {
    expect(absorptionSliderRange(0).max).toBe(ABSORPTION_DEFAULT_MAX);
    expect(absorptionSliderRange(NaN).max).toBe(ABSORPTION_DEFAULT_MAX);
  });

  describe('at the clamped extremes', () => {
    /**
     * Outside the two deliberate clamps the thumb seats at the clamped end
     * while the readout shows the true κ, so a touch writes the clamp back.
     * These pin that documented trade-off — and why it is acceptable: the
     * swapped states are visually identical. With the unified convention
     * τ = κ · rayMass and rayMass ≈ 1 at an element's peak, τ ≈ κ.
     */
    it('below the span cap: both κ are far below visible absorption', () => {
      const { min, max } = absorptionSliderRange(1e-30);
      expect(min).toBeCloseTo(max / Math.pow(10, ABSORPTION_LOG_DECADES_MAX), 12);
      expect(1e-30).toBeLessThan(min); // off-track: a touch would write `min`
      expect(min).toBeLessThan(1e-3); // and what it writes is optically nothing
    });

    it('above the ceiling: both κ are far past opaque', () => {
      const { max } = absorptionSliderRange(1e30);
      expect(max).toBe(ABSORPTION_MAX_LIMIT);
      expect(1e30).toBeGreaterThan(max); // off-track: a touch would write `max`
      // 1 − e^(−τ) is 1.0 to double precision on both sides of the clamp.
      expect(1 - Math.exp(-max)).toBe(1);
      expect(1 - Math.exp(-1e30)).toBe(1);
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
