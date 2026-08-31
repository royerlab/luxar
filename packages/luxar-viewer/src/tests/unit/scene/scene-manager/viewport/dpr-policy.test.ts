// @vitest-environment jsdom
/**
 * Unit tests for the pure DPR-policy helpers used by SceneManager's
 * adaptive/manual pixel-ratio handling.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  computePixelRatioOverride,
  getActivePixelRatio,
  getNormalizedDPRScale,
  syncPostProcessingDPRScale,
} from '../../../../../scene/scene-manager/viewport/dpr-policy';
import {
  DEFAULT_MAX_PIXEL_RATIO,
  setMaxPixelRatioCap,
} from '../../../../../rendering/pixel-ratio-cap';
import { allowHighDPR, withNativeDPR } from '../../../../helpers/device-pixel-ratio';
import type { PostProcessingManager } from '../../../../../rendering';

// The blocks below predate the pixel-ratio cap and describe the policy
// against the DISPLAY's DPR, so they run with the cap lifted. The cap's
// own behaviour is covered by the last block in this file, which is the
// one that exercises the shipped DEFAULT.
let restoreCap: () => void;
beforeEach(() => {
  restoreCap = allowHighDPR();
});
afterEach(() => restoreCap());

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

  // C3: pin the ~0.01 near-ceiling threshold on BOTH sides. A
  // sub-threshold delta clears the override; a clearly supra-threshold
  // delta stores it. Both probes sit BELOW the ceiling, because an
  // above-ceiling request is clamped before the threshold is even
  // consulted (see the next test).
  // (We avoid asserting exactly at 0.01 — `2 - 1.99` is 0.01000…2 in
  // IEEE-754, so the exact boundary is float-fuzzy by nature.)
  it('clears the override just below the 0.01 threshold and stores it above', () => {
    withNativeDPR(2, () => {
      // delta ≈ 0.009 < 0.01 → cleared.
      expect(computePixelRatioOverride(1.991).override).toBeNull();
      // delta = 0.02 > 0.01 → stored.
      const stored = computePixelRatioOverride(1.98);
      expect(stored.override).toBeCloseTo(1.98, 10);
      expect(stored.active).toBeCloseTo(1.98, 10);
    });
  });

  // Even uncapped, the ceiling is the display's own DPR: a request above
  // it is clamped rather than stored, so nothing can quietly supersample
  // the backbuffer past what the screen can show. (Supersampling is
  // SSAA's job, and it is a separate, explicit multiplier.)
  it('clamps an above-native request to native instead of supersampling', () => {
    withNativeDPR(2, () => {
      const r = computePixelRatioOverride(2.02);
      expect(r.override).toBeNull();
      expect(r.active).toBe(2);
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

/**
 * The shipped default: high DPR disallowed, so the ceiling is 1.0 even on
 * a HiDPI display and every function above measures against THAT rather
 * than the display's own DPR.
 */
describe('pixel-ratio cap (the default: high DPR disallowed)', () => {
  beforeEach(() => setMaxPixelRatioCap(DEFAULT_MAX_PIXEL_RATIO));

  it('clamps a null override to the cap instead of tracking native', () => {
    withNativeDPR(2, () => {
      expect(getActivePixelRatio(null)).toBe(1);
    });
  });

  it('clamps an explicit above-cap override down to the cap', () => {
    withNativeDPR(2, () => {
      expect(getActivePixelRatio(2)).toBe(1);
      expect(getActivePixelRatio(1.5)).toBe(1);
    });
  });

  it('leaves a below-cap override alone — adaptation still works underneath', () => {
    withNativeDPR(2, () => {
      expect(getActivePixelRatio(0.5)).toBe(0.5);
    });
  });

  it('never RAISES the ratio: a sub-1 display keeps its own DPR', () => {
    withNativeDPR(0.8, () => {
      expect(getActivePixelRatio(null)).toBeCloseTo(0.8, 10);
    });
  });

  /**
   * The fast path the whole design hinges on. A null override means
   * "track the live ceiling", and that only survives if a request AT the
   * ceiling clears the override. Comparing against the raw native DPR
   * would leave the override permanently non-null on any HiDPI display,
   * so every monitor drag would force a render-target reallocation.
   */
  it('clears the override for a request at the CAP, not at native', () => {
    withNativeDPR(2, () => {
      const atCap = computePixelRatioOverride(1);
      expect(atCap.override).toBeNull();
      expect(atCap.active).toBe(1);

      // Asking for native is not asking for something the viewer will do.
      const atNative = computePixelRatioOverride(2);
      expect(atNative.override).toBeNull();
      expect(atNative.active).toBe(1);
    });
  });

  it('still stores a genuine reduction below the cap', () => {
    withNativeDPR(2, () => {
      const r = computePixelRatioOverride(0.75);
      expect(r.override).toBe(0.75);
      expect(r.active).toBe(0.75);
    });
  });

  it('coerces invalid input to the cap, not to native', () => {
    withNativeDPR(2, () => {
      for (const bad of [NaN, 0, -1, Infinity, -Infinity]) {
        const r = computePixelRatioOverride(bad);
        expect(r.override).toBeNull();
        expect(r.active).toBe(1);
      }
    });
  });

  /**
   * Detector noise is scaled by this ratio. Normalizing against native
   * would peg it at 0.5 on every Retina session, silently halving the
   * noise sigma and quartering the photon gain of a scene that asked for
   * neither.
   */
  it('normalizes the DPR scale against the cap, so at the cap it is 1', () => {
    withNativeDPR(2, () => {
      expect(getNormalizedDPRScale(1)).toBe(1);
      expect(getNormalizedDPRScale(0.5)).toBe(0.5);
    });
  });

  it('forwards the cap-relative scale to postProcessing', () => {
    withNativeDPR(2, () => {
      const setDPRScale = vi.fn();
      const pp = { setDPRScale } as unknown as PostProcessingManager;

      syncPostProcessingDPRScale(pp, null);

      // active = the 1.0 ceiling, and the ceiling IS full quality here.
      expect(setDPRScale).toHaveBeenCalledWith(1);
    });
  });
});
