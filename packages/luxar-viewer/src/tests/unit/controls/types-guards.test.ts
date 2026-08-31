/**
 * Unit tests for the type guards in `controls/types.ts`.
 *
 * The guards use property-presence checks to discriminate between
 * orbit and fly control instances. The tests use object literals
 * with the right property shape — no need to instantiate the real
 * THREE.js-coupled classes.
 */

import { describe, it, expect } from 'vitest';
import {
  dollyAmplitudeFromPercent,
  dollyAmplitudeToPercent,
  isFlyControls,
  isOrbitControls,
  rpmFromSecondsPerTurn,
  secondsPerTurnFromRpm,
  type ControlInstance,
} from '../../../controls/types';

// Test fixtures use type assertion because the guard reads properties
// off a duck-typed object — the actual classes pull in THREE which
// these unit tests deliberately avoid.

const orbitLike = {
  target: { x: 0, y: 0, z: 0 },
  autoRotate: false,
} as unknown as ControlInstance;

const flyLike = {
  inertialMode: false,
  lookSpeed: 1.0,
} as unknown as ControlInstance;

const ambiguous = {
  target: { x: 0, y: 0, z: 0 },
  autoRotate: false,
  inertialMode: false,
  lookSpeed: 1.0,
} as unknown as ControlInstance;

describe('isOrbitControls', () => {
  it('returns true for an object with target + autoRotate', () => {
    expect(isOrbitControls(orbitLike)).toBe(true);
  });

  it('returns false for fly-only shape (missing target/autoRotate)', () => {
    expect(isOrbitControls(flyLike)).toBe(false);
  });

  it('returns false for null', () => {
    expect(isOrbitControls(null)).toBe(false);
  });

  it('also returns true for an object that satisfies both shapes', () => {
    // Property-presence guards are not mutually exclusive; the call
    // site picks one to dispatch on. Lock in the observed behaviour.
    expect(isOrbitControls(ambiguous)).toBe(true);
  });
});

describe('isFlyControls', () => {
  it('returns true for an object with inertialMode + lookSpeed', () => {
    expect(isFlyControls(flyLike)).toBe(true);
  });

  it('returns false for orbit-only shape (missing inertialMode/lookSpeed)', () => {
    expect(isFlyControls(orbitLike)).toBe(false);
  });

  it('returns false for null', () => {
    expect(isFlyControls(null)).toBe(false);
  });

  it('returns true for an object that satisfies both shapes', () => {
    expect(isFlyControls(ambiguous)).toBe(true);
  });
});

/**
 * Unit conversions between what the UI shows and what the settings store.
 *
 * These exist because the two disagree ON PURPOSE: `auto_rotate_speed` is rpm
 * and `auto_dolly_amplitude_percent` is a percent, both because of what is
 * already written to disk, while the rows show a period in seconds and a
 * fraction respectively. A silently wrong conversion here is a 60x or 100x
 * error in camera motion that no type would catch.
 */
describe('display-unit conversions', () => {
  describe('turntable period ↔ rpm', () => {
    it.each([
      [0.25, 240], // the shipped default: one turn every four minutes
      [1, 60],
      [3, 20],
      [5, 12], // the fastest the slider allows
    ])('%f rpm is %d seconds per turn', (rpm, seconds) => {
      expect(secondsPerTurnFromRpm(rpm)).toBeCloseTo(seconds, 9);
      expect(rpmFromSecondsPerTurn(seconds)).toBeCloseTo(rpm, 9);
    });

    it('round-trips an arbitrary value both ways', () => {
      expect(rpmFromSecondsPerTurn(secondsPerTurnFromRpm(0.37))).toBeCloseTo(0.37, 9);
      expect(secondsPerTurnFromRpm(rpmFromSecondsPerTurn(17))).toBeCloseTo(17, 9);
    });

    it.each([0, -1, NaN, Infinity])('falls back instead of dividing by %s', (bad) => {
      // Infinity/NaN reaching the turntable would freeze it or NaN the camera
      // out of existence; the fallbacks are the other unit's identity.
      expect(Number.isFinite(secondsPerTurnFromRpm(bad))).toBe(true);
      expect(Number.isFinite(rpmFromSecondsPerTurn(bad))).toBe(true);
    });
  });

  describe('dolly amplitude percent ↔ fraction', () => {
    it.each([
      [15, 0.15], // default
      [95, 0.95], // the slider ceiling
      [1, 0.01],
    ])('%d%% is a fraction of %f', (percent, fraction) => {
      expect(dollyAmplitudeFromPercent(percent)).toBeCloseTo(fraction, 12);
      expect(dollyAmplitudeToPercent(fraction)).toBeCloseTo(percent, 12);
    });

    it('round-trips', () => {
      expect(dollyAmplitudeToPercent(dollyAmplitudeFromPercent(37))).toBeCloseTo(37, 12);
    });
  });
});
