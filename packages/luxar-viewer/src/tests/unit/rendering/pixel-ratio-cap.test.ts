// @vitest-environment jsdom
/**
 * Unit tests for `rendering/pixel-ratio-cap.ts` — the max DPR the viewer may
 * render at, and how "high DPR allowed" resolves per device class.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  DEFAULT_MAX_PIXEL_RATIO,
  MOBILE_MAX_PIXEL_RATIO,
  deviceHighDprCeiling,
  getMaxPixelRatio,
  getMaxPixelRatioCap,
  isHighDPRAllowed,
  setHighDPRAllowed,
  setMaxPixelRatioCap,
} from '../../../rendering/pixel-ratio-cap';

const profile = { deviceClass: 'laptop' as 'mobile' | 'laptop' | 'desktop' };
vi.mock('../../../utils/input-capabilities', () => ({
  getInputProfile: () => profile,
}));

describe('pixel-ratio-cap', () => {
  beforeEach(() => {
    profile.deviceClass = 'laptop';
    setHighDPRAllowed(false);
  });
  afterEach(() => setHighDPRAllowed(false));

  it('defaults to CSS resolution (cap 1.0)', () => {
    expect(getMaxPixelRatioCap()).toBe(DEFAULT_MAX_PIXEL_RATIO);
    expect(isHighDPRAllowed()).toBe(false);
  });

  it('allowed on a laptop/desktop = no cap', () => {
    setHighDPRAllowed(true);
    expect(getMaxPixelRatioCap()).toBe(Infinity);
    expect(isHighDPRAllowed()).toBe(true);
    profile.deviceClass = 'desktop';
    expect(deviceHighDprCeiling()).toBe(Infinity);
  });

  it('allowed on a phone/tablet = MOBILE_MAX_PIXEL_RATIO, not Infinity', () => {
    profile.deviceClass = 'mobile';
    expect(deviceHighDprCeiling()).toBe(MOBILE_MAX_PIXEL_RATIO);
    setHighDPRAllowed(true);
    expect(getMaxPixelRatioCap()).toBe(2);
    expect(isHighDPRAllowed()).toBe(true); // still "above CSS resolution"
    setHighDPRAllowed(false);
    expect(getMaxPixelRatioCap()).toBe(DEFAULT_MAX_PIXEL_RATIO);
  });

  it('an explicit cap (a ?dpr= pin, a recording captureDPR) is not clamped by the device class', () => {
    profile.deviceClass = 'mobile';
    setMaxPixelRatioCap(3);
    expect(getMaxPixelRatioCap()).toBe(3);
  });

  it('a non-positive cap falls back to the default rather than disabling the ceiling', () => {
    setMaxPixelRatioCap(0);
    expect(getMaxPixelRatioCap()).toBe(DEFAULT_MAX_PIXEL_RATIO);
    setMaxPixelRatioCap(Number.NaN);
    expect(getMaxPixelRatioCap()).toBe(DEFAULT_MAX_PIXEL_RATIO);
  });

  it('getMaxPixelRatio is the lower of the native DPR and the cap', () => {
    const original = window.devicePixelRatio;
    Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: 3 });
    try {
      expect(getMaxPixelRatio()).toBe(1);
      profile.deviceClass = 'mobile';
      setHighDPRAllowed(true);
      expect(getMaxPixelRatio()).toBe(2); // DPR-3 phone, mobile ceiling binds
      profile.deviceClass = 'laptop';
      setHighDPRAllowed(true);
      expect(getMaxPixelRatio()).toBe(3); // no cap → native
    } finally {
      Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: original });
    }
  });
});
