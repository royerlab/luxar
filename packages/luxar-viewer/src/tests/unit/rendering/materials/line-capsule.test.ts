/**
 * Unit pins for the capsule line primitive's shared constants + CPU
 * reference (`_shared/line-capsule.ts`) — the single source both shader
 * backends fold from (#1352).
 */
import { describe, expect, it } from 'vitest';

import { GAUSSIAN_EQUIVALENT_TRUNCATION } from '../../../../rendering/materials/_shared/falloff';
import {
  CAPSULE_CUT_FADE_RADIUS_FRACTION,
  CAPSULE_MIN_RADIUS_PX,
  CAPSULE_RADIUS_PER_QUAD_HALFWIDTH,
  CAPSULE_SUPPORT_SIGMA,
  capsuleProfile,
  capsuleProfileExponent,
} from '../../../../rendering/materials/_shared/line-capsule';
import {
  CAPSULE_LINE_FRAGMENT_SHADER,
  CAPSULE_LINE_VERTEX_SHADER,
} from '../../../../rendering/materials/line/shader-glsl-capsule';
import {
  CAPSULE_LINE_PICK_FRAGMENT_SHADER,
  CAPSULE_LINE_PICK_VERTEX_SHADER,
} from '../../../../rendering/picking/line/shaders-capsule';

describe('capsule constants', () => {
  it('the radius factor is the 2σ fraction of the quad half-width', () => {
    expect(CAPSULE_SUPPORT_SIGMA).toBe(2.0);
    expect(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH).toBeCloseTo(2 / GAUSSIAN_EQUIVALENT_TRUNCATION, 12);
    // The shaders inline it at toFixed(7); pin the literal so a constant
    // change is visible here before it silently reshapes every line.
    expect(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH.toFixed(7)).toBe('0.6590102');
  });

  it('joint constants: quarter-radius cut fade, 1.5 px AA floor', () => {
    expect(CAPSULE_CUT_FADE_RADIUS_FRACTION).toBe(0.25);
    expect(CAPSULE_MIN_RADIUS_PX).toBe(1.5); // matches the quad's AA floor
  });

  it('shaders fold the shared literals (no re-derived magic numbers)', () => {
    for (const src of [CAPSULE_LINE_VERTEX_SHADER, CAPSULE_LINE_PICK_VERTEX_SHADER]) {
      expect(src).toContain(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH.toFixed(7)); // radius factor
      expect(src).toContain('1.5'); // AA radius floor
    }
    for (const src of [CAPSULE_LINE_FRAGMENT_SHADER, CAPSULE_LINE_PICK_FRAGMENT_SHADER]) {
      // The quartic default path, the sharpness exponent map, and the
      // foreign-side cut fade fraction.
      expect(src).toContain('w * w');
      expect(src).toContain('exp2(3.0 - 4.0 * vSharp)');
      expect(src).toContain(CAPSULE_CUT_FADE_RADIUS_FRACTION.toFixed(2));
    }
  });

  it('the two stages agree on the foreign-side fade band', () => {
    // The fragment shades the bend-scaled fade beyond the endpoint, so the
    // vertex stage must reserve stencil for it — reserving only the kept
    // half-disc (|n.y|·rMax) chopped the ramp part-way down at gentle
    // joints and handed back the hard step the fade exists to remove.
    const frac = CAPSULE_CUT_FADE_RADIUS_FRACTION.toFixed(2);
    for (const src of [CAPSULE_LINE_VERTEX_SHADER, CAPSULE_LINE_PICK_VERTEX_SHADER]) {
      expect(src).toContain(`(abs(nLoc.y) + ${frac} * max(abs(nLoc.y), 0.25)) * rMax`);
      expect(src).not.toMatch(/ext[AB] = abs\(nLoc\.y\) \* rMax/);
    }
    for (const src of [CAPSULE_LINE_FRAGMENT_SHADER, CAPSULE_LINE_PICK_FRAGMENT_SHADER]) {
      expect(src).toContain(`${frac} * rPx * max(abs(vCutA2.y), 0.25)`);
      expect(src).toContain(`${frac} * rPx * max(abs(vCutB2.y), 0.25)`);
    }
  });

  it('the cap rule comes from the shared joint-code helper (a hub keeps its cap)', () => {
    // A degree->=3 hub (code -2) and a free end (0) keep the whole round
    // cap; reading `abs(code) > 0.5` instead butt-cuts a hub and notches it.
    for (const src of [CAPSULE_LINE_VERTEX_SHADER, CAPSULE_LINE_PICK_VERTEX_SHADER]) {
      expect(src).toContain('luxarLineJointCapSuppression(lineT4.y)');
      expect(src).toContain('luxarLineJointCapSuppression(lineT4.z)');
    }
  });

  it('a butt cut is hard — nothing draws past the endpoint line', () => {
    // No bisector (slice-clipped end, behind-near joint vertex, degenerate
    // partner projection, exactly straight joint) ⇒ no partner body to fade
    // into, and the vertex stage reserves no fade band there either.
    for (const src of [CAPSULE_LINE_FRAGMENT_SHADER, CAPSULE_LINE_PICK_FRAGMENT_SHADER]) {
      expect(src).toContain('vCutA2.y == 0.0');
      expect(src).toContain('vCutB2.y == 0.0');
    }
  });

  it('no backticks inside the GLSL template literals', () => {
    for (const src of [
      CAPSULE_LINE_VERTEX_SHADER,
      CAPSULE_LINE_FRAGMENT_SHADER,
      CAPSULE_LINE_PICK_VERTEX_SHADER,
      CAPSULE_LINE_PICK_FRAGMENT_SHADER,
    ]) {
      expect(src).not.toContain('`');
    }
  });
});

describe('capsuleProfile (CPU reference)', () => {
  it('default knob: the quartic bump (1 − p²)²', () => {
    expect(capsuleProfileExponent(0.5)).toBe(2);
    expect(capsuleProfile(0)).toBe(1);
    expect(capsuleProfile(0.5)).toBeCloseTo(0.75 * 0.75, 12);
    expect(capsuleProfile(1)).toBe(0); // exact zero at the 2σ rim
    expect(capsuleProfile(1.5)).toBe(0); // compact support beyond
  });

  it('sharpness map runs boxy↔spiky the right way round', () => {
    // Smaller exponents are boxier in w-space: s=1 (boxy) keeps more of
    // the shoulder than s=0 (spiky) at the same radius.
    expect(capsuleProfileExponent(0)).toBe(8);
    expect(capsuleProfileExponent(1)).toBe(0.5);
    expect(capsuleProfile(0.7, 1.0)).toBeGreaterThan(capsuleProfile(0.7, 0.5));
    expect(capsuleProfile(0.7, 0.0)).toBeLessThan(capsuleProfile(0.7, 0.5));
  });

  it('monotone decreasing in p at every knob', () => {
    for (const s of [0, 0.25, 0.5, 0.75, 1]) {
      let prev = Infinity;
      for (let i = 0; i <= 20; i++) {
        const val = capsuleProfile(i / 20, s);
        expect(val).toBeLessThanOrEqual(prev + 1e-12);
        prev = val;
      }
    }
  });
});
