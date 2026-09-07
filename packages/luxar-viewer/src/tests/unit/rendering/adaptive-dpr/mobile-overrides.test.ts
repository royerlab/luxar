/**
 * `mobileAdaptiveDprOverrides()` — the construction-time overrides the app
 * passes to `AdaptiveDPRManager` on a phone or tablet, and nothing elsewhere.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  AdaptiveDPRManager,
  MOBILE_MIN_DPR,
  MOBILE_REFRESH_RATE_CEILING,
  mobileAdaptiveDprOverrides,
} from '../../../../rendering/adaptive-dpr-manager';
import { config } from '../../../../config';

const profile = { deviceClass: 'laptop' as 'mobile' | 'laptop' | 'desktop' };
vi.mock('../../../../utils/input-capabilities', () => ({
  getInputProfile: () => profile,
}));

describe('mobileAdaptiveDprOverrides', () => {
  afterEach(() => {
    profile.deviceClass = 'laptop';
  });

  it('is undefined on a laptop/desktop, so the manager constructs exactly as before', () => {
    expect(mobileAdaptiveDprOverrides()).toBeUndefined();
    profile.deviceClass = 'desktop';
    expect(mobileAdaptiveDprOverrides()).toBeUndefined();
  });

  it('on a phone/tablet raises the floor to 0.75 and caps the refresh estimate at 60 Hz', () => {
    profile.deviceClass = 'mobile';
    expect(mobileAdaptiveDprOverrides()).toEqual({
      minDPR: Math.max(config.adaptiveDPR.minDPR, MOBILE_MIN_DPR),
      refreshRateCeiling: MOBILE_REFRESH_RATE_CEILING,
    });
    expect(MOBILE_MIN_DPR).toBe(0.75);
    expect(MOBILE_REFRESH_RATE_CEILING).toBe(60);
  });

  it('never lowers a configured floor that is already higher', () => {
    profile.deviceClass = 'mobile';
    const original = config.adaptiveDPR.minDPR;
    config.adaptiveDPR.minDPR = 0.9;
    try {
      expect(mobileAdaptiveDprOverrides()?.minDPR).toBe(0.9);
    } finally {
      config.adaptiveDPR.minDPR = original;
    }
  });

  it('keeps a tighter configured refresh ceiling', () => {
    profile.deviceClass = 'mobile';
    const original = config.adaptiveDPR.refreshRateCeiling;
    config.adaptiveDPR.refreshRateCeiling = 30;
    try {
      expect(mobileAdaptiveDprOverrides()?.refreshRateCeiling).toBe(30);
    } finally {
      config.adaptiveDPR.refreshRateCeiling = original;
    }
  });

  it('falls back to structural defaults for partial config mocks', () => {
    profile.deviceClass = 'mobile';
    const original = config.adaptiveDPR;
    config.adaptiveDPR = {} as typeof config.adaptiveDPR;
    try {
      expect(mobileAdaptiveDprOverrides()).toEqual({
        minDPR: MOBILE_MIN_DPR,
        refreshRateCeiling: MOBILE_REFRESH_RATE_CEILING,
      });
    } finally {
      config.adaptiveDPR = original;
    }
  });

  it('the manager accepts the overrides (a 120 Hz mark reports a 60 Hz cap)', () => {
    profile.deviceClass = 'mobile';
    const manager = new AdaptiveDPRManager(mobileAdaptiveDprOverrides());
    // Feed a light-scene 120 fps window; the cap must stay at the ceiling.
    for (let i = 0; i < 6; i++) manager.recordFrame(1000 + i * 8.33);
    expect(manager.getState().refreshRateCap).toBeLessThanOrEqual(60);
    manager.dispose();
  });
});
