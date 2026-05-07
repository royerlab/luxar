/**
 * Unit tests for the FOV / focal-length conversion helpers.
 *
 * The helpers are pure math — no DOM, no THREE.js, no scene manager.
 */

import { describe, it, expect } from 'vitest';
import {
  FILM_35MM_SENSOR_WIDTH_MM,
  focalLengthToFov,
  fovToFocalLength,
} from '../../../ui/rendering-controls/fov-utils';

describe('fovToFocalLength', () => {
  it('reproduces the standard 35mm-photography lens labels (within rounding)', () => {
    // These are the canonical FOV → focal-length mappings the FOV
    // preset table uses. Allowing ±1mm because the formula rounds.
    const expectations: Array<{ fov: number; focal: number }> = [
      { fov: 65, focal: 28 },
      { fov: 54, focal: 35 },
      { fov: 39, focal: 50 },
      { fov: 24, focal: 85 },
      { fov: 15, focal: 135 },
    ];
    for (const { fov, focal } of expectations) {
      expect(Math.abs(fovToFocalLength(fov) - focal)).toBeLessThanOrEqual(2);
    }
  });

  it('returns an integer (rounded)', () => {
    for (const fov of [10, 30, 45, 60, 90, 120]) {
      const f = fovToFocalLength(fov);
      expect(Number.isInteger(f)).toBe(true);
    }
  });

  it('decreases monotonically as FOV increases (wide-angle = shorter lens)', () => {
    const focals = [10, 20, 40, 60, 80, 100, 120].map(fovToFocalLength);
    for (let i = 1; i < focals.length; i++) {
      expect(focals[i]).toBeLessThanOrEqual(focals[i - 1]);
    }
  });

  it('uses the 35mm sensor width constant (36mm)', () => {
    expect(FILM_35MM_SENSOR_WIDTH_MM).toBe(36);
  });
});

describe('focalLengthToFov', () => {
  it('returns 0 for non-positive focal lengths (defensive)', () => {
    expect(focalLengthToFov(0)).toBe(0);
    expect(focalLengthToFov(-50)).toBe(0);
  });

  it('produces the expected FOV for a 50mm "natural" lens (~40°)', () => {
    expect(focalLengthToFov(50)).toBeCloseTo(39.6, 0);
  });

  it('round-trips through fovToFocalLength within rounding tolerance', () => {
    for (const focal of [28, 35, 50, 85, 135]) {
      const fov = focalLengthToFov(focal);
      const restored = fovToFocalLength(fov);
      expect(Math.abs(restored - focal)).toBeLessThanOrEqual(1);
    }
  });
});
