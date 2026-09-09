/**
 * `getConfiguredWorkerCount` on a mobile device class: the pool is capped at
 * `MOBILE_MAX_WORKERS` regardless of the reported core count. The
 * laptop/desktop arithmetic is covered in `timeout/pure-helpers.test.ts`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  MOBILE_MAX_WORKERS,
  getConfiguredWorkerCount,
} from '../../../../../workers/worker-pool/lifecycle/worker-count';

const profile = { deviceClass: 'laptop' as 'mobile' | 'laptop' | 'desktop' };
vi.mock('../../../../../utils/input-capabilities', () => ({
  getInputProfile: () => profile,
}));

describe('getConfiguredWorkerCount — mobile ceiling', () => {
  const origNavigator = globalThis.navigator;

  beforeEach(() => {
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { hardwareConcurrency: 8 },
    });
  });

  afterEach(() => {
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: origNavigator });
    profile.deviceClass = 'laptop';
  });

  it('laptop/desktop: hardwareConcurrency - 1, as before', () => {
    expect(getConfiguredWorkerCount(0)).toBe(7);
    expect(getConfiguredWorkerCount(100)).toBe(7);
  });

  it('mobile: auto mode caps at MOBILE_MAX_WORKERS even on an 8-core phone', () => {
    profile.deviceClass = 'mobile';
    expect(MOBILE_MAX_WORKERS).toBe(3);
    expect(getConfiguredWorkerCount(0)).toBe(3);
  });

  it('mobile: an explicit count is capped at the mobile ceiling too', () => {
    profile.deviceClass = 'mobile';
    expect(getConfiguredWorkerCount(6)).toBe(3);
    expect(getConfiguredWorkerCount(2)).toBe(2);
  });

  it('mobile: a low core count still wins when it is below the ceiling', () => {
    profile.deviceClass = 'mobile';
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { hardwareConcurrency: 2 },
    });
    expect(getConfiguredWorkerCount(0)).toBe(1);
  });
});
