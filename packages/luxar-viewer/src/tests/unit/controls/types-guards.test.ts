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
  isFlyControls,
  isOrbitControls,
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
