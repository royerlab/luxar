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
      expect(computePixelRatioOverride(NaN).override).toBeNull();
      expect(computePixelRatioOverride(0).override).toBeNull();
      expect(computePixelRatioOverride(-1).override).toBeNull();
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
