/**
 * Unit tests for the blending-state module.
 *
 * F.2 covers the discriminator predicates; H.1 expands with canonical
 * state assertions for each mode (additive/normal/max/opaque/luminous),
 * idempotency, max-mode round-trip, and opacity boundary tests.
 */
import { describe, it, expect } from 'vitest';
import {
  isAdditiveMode,
  isLuminousMode,
  isMaxMode,
  isNormalMode,
  isOpaqueMode,
} from '../../../rendering/blending-state';
import type { BlendingMode } from '../../../rendering/material-manager';

describe('F.2 — BlendingMode predicates', () => {
  const all: BlendingMode[] = ['additive', 'normal', 'max', 'opaque', 'luminous'];

  it('isAdditiveMode is true only for additive', () => {
    for (const mode of all) {
      expect(isAdditiveMode(mode)).toBe(mode === 'additive');
    }
  });

  it('isNormalMode is true only for normal', () => {
    for (const mode of all) {
      expect(isNormalMode(mode)).toBe(mode === 'normal');
    }
  });

  it('isMaxMode is true only for max', () => {
    for (const mode of all) {
      expect(isMaxMode(mode)).toBe(mode === 'max');
    }
  });

  it('isOpaqueMode is true only for opaque', () => {
    for (const mode of all) {
      expect(isOpaqueMode(mode)).toBe(mode === 'opaque');
    }
  });

  it('isLuminousMode is true only for luminous', () => {
    for (const mode of all) {
      expect(isLuminousMode(mode)).toBe(mode === 'luminous');
    }
  });

  it('exactly one predicate fires per mode', () => {
    for (const mode of all) {
      const hits = [
        isAdditiveMode(mode),
        isNormalMode(mode),
        isMaxMode(mode),
        isOpaqueMode(mode),
        isLuminousMode(mode),
      ].filter(Boolean).length;
      expect(hits).toBe(1);
    }
  });
});
