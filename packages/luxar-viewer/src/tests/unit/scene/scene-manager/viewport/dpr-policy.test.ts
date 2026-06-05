/**
 * Unit tests for the pure DPR-policy helpers used by SceneManager's
 * adaptive/manual pixel-ratio handling.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  computePixelRatioOverride,
  getActivePixelRatio,
  getNormalizedDPRScale,
  syncPostProcessingDPRScale,
} from '../../../../../scene/scene-manager/viewport/dpr-policy';
import type { PostProcessingManager } from '../../../../../rendering';

function withNativeDPR<T>(dpr: number, fn: () => T): T {
  const original = window.devicePixelRatio;
  Object.defineProperty(window, 'devicePixelRatio', {
    value: dpr,
    configurable: true,
    writable: true,
  });
  try {
    return fn();
  } finally {
    Object.defineProperty(window, 'devicePixelRatio', {
      value: original,
      configurable: true,
      writable: true,
    });
  }
}

describe('getActivePixelRatio', () => {
  it('returns the explicit override when one is set', () => {
    withNativeDPR(2, () => {
      expect(getActivePixelRatio(0.5)).toBe(0.5);
    });
  });

  it('falls back to native devicePixelRatio when override is null', () => {
    withNativeDPR(3, () => {
      expect(getActivePixelRatio(null)).toBe(3);
    });
  });

  it('returns 1 when both override and native are 0/falsy', () => {
    withNativeDPR(0, () => {
      expect(getActivePixelRatio(null)).toBe(1);
    });
  });

  // G1: explicit override = 0 is falsy, so the `|| 1` guard maps it to 1
  // (?? only treats null/undefined as "unset", so 0 reaches the || guard).
  it('maps an explicit override of 0 to 1 (falsy-guard)', () => {
    withNativeDPR(2, () => {
      expect(getActivePixelRatio(0)).toBe(1);
    });
  });

  // G1: a positive fractional override is returned verbatim.
  it('returns a positive fractional override verbatim', () => {
    withNativeDPR(2, () => {
      expect(getActivePixelRatio(1.5)).toBe(1.5);
    });
  });
});

describe('computePixelRatioOverride', () => {
  it('clears the override when requested DPR matches native (within 0.01)', () => {
    withNativeDPR(2, () => {
      const r = computePixelRatioOverride(2);
      expect(r.override).toBeNull();
      expect(r.active).toBe(2);
    });
  });

  it('clears the override on near-native DPR (delta < 0.01)', () => {
    withNativeDPR(2, () => {
      const r = computePixelRatioOverride(2.005);
      expect(r.override).toBeNull();
    });
  });

  it('stores the override when requested DPR differs from native', () => {
    withNativeDPR(2, () => {
      const r = computePixelRatioOverride(0.75);
      expect(r.override).toBe(0.75);
      expect(r.active).toBe(0.75);
    });
  });

  it('coerces non-finite / non-positive DPR to native', () => {
    withNativeDPR(1.5, () => {
      // G2: invalid input → override cleared AND active falls back to native,
      // not to the invalid value.
      for (const bad of [NaN, 0, -1, Infinity, -Infinity]) {
        const r = computePixelRatioOverride(bad);
        expect(r.override).toBeNull();
        expect(r.active).toBe(1.5);
      }
    });
  });

  // C3: pin the ~0.01 near-native threshold on BOTH sides. A sub-threshold
  // delta clears the override; a clearly supra-threshold delta stores it.
  // (We avoid asserting exactly at 0.01 — `2.01 - 2` is 0.00999… in IEEE-754,
  // so the exact boundary is float-fuzzy by nature.)
  it('clears the override just below the 0.01 threshold and stores it above', () => {
    withNativeDPR(2, () => {
      // delta ≈ 0.009 < 0.01 → cleared.
      expect(computePixelRatioOverride(2.009).override).toBeNull();
      // delta = 0.02 > 0.01 → stored.
      const stored = computePixelRatioOverride(2.02);
      expect(stored.override).toBeCloseTo(2.02, 10);
      expect(stored.active).toBeCloseTo(2.02, 10);
    });
  });
});

describe('getNormalizedDPRScale', () => {
  it('returns 1 when active DPR equals native', () => {
    withNativeDPR(2, () => {
      expect(getNormalizedDPRScale(2)).toBe(1);
    });
  });

  it('returns < 1 when active is below native (perceptual scale-down)', () => {
    withNativeDPR(2, () => {
      expect(getNormalizedDPRScale(1)).toBe(0.5);
    });
  });

  it('returns > 1 when active is above native', () => {
    withNativeDPR(1, () => {
      expect(getNormalizedDPRScale(2)).toBe(2);
    });
  });
});

describe('syncPostProcessingDPRScale', () => {
  it('forwards the normalized scale to postProcessing.setDPRScale', () => {
    withNativeDPR(2, () => {
      const setDPRScale = vi.fn();
      const pp = { setDPRScale } as unknown as PostProcessingManager;

      syncPostProcessingDPRScale(pp, 1);

      expect(setDPRScale).toHaveBeenCalledTimes(1);
      expect(setDPRScale).toHaveBeenCalledWith(0.5); // active=1 / native=2
    });
  });

  it('is a no-op when postProcessing is null (early init)', () => {
    expect(() => syncPostProcessingDPRScale(null, 1)).not.toThrow();
  });
});
