import { afterEach, describe, it, expect, vi } from 'vitest';
import {
  candidateRules,
  defaultPlatformRules,
  detectBrowserPlatform,
  matchAdapterRule,
  STAGED_WEBGPU_PLATFORM_RULES,
  WEBGPU_PLATFORM_RULES,
} from '../../../../../scene/scene-manager/render-pipeline/backend-policy';

afterEach(() => {
  vi.unstubAllEnvs();
});

describe('detectBrowserPlatform', () => {
  it('reads OS and browser from client hints when present', () => {
    const nav = {
      userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/146.0 Safari/537.36',
      userAgentData: {
        platform: 'Linux',
        brands: [{ brand: 'Chromium' }, { brand: 'Google Chrome' }],
      },
    };
    expect(detectBrowserPlatform(nav)).toEqual({ os: 'linux', browser: 'chromium' });
  });

  it('does not take Android or ChromeOS for desktop Linux', () => {
    const android =
      'Mozilla/5.0 (Linux; Android 15; Pixel 8a) AppleWebKit/537.36 Chrome/146.0 Mobile Safari/537.36';
    const cros =
      'Mozilla/5.0 (X11; CrOS x86_64 16000.0.0) AppleWebKit/537.36 Chrome/146.0 Safari/537.36';
    expect(detectBrowserPlatform({ userAgent: android }).os).toBe('android');
    expect(detectBrowserPlatform({ userAgent: cros }).os).toBe('chromeos');
    expect(
      detectBrowserPlatform({ userAgent: android, userAgentData: { platform: 'Android' } }).os
    ).toBe('android');
  });

  it('tells Firefox and Safari apart from Chromium', () => {
    expect(
      detectBrowserPlatform({
        userAgent: 'Mozilla/5.0 (X11; Linux x86_64; rv:150.0) Gecko/20100101 Firefox/150.0',
      })
    ).toEqual({ os: 'linux', browser: 'firefox' });
    expect(
      detectBrowserPlatform({
        userAgent:
          'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.6 Safari/605.1.15',
      })
    ).toEqual({ os: 'macos', browser: 'safari' });
  });

  it('is unknown outside a browser', () => {
    expect(detectBrowserPlatform(undefined)).toEqual({ os: 'unknown', browser: 'unknown' });
  });
});

describe('platform rules', () => {
  it('admits no platform until cold load is within 0.15 s (#3172)', () => {
    vi.stubEnv('VITE_LUXAR_WEBGPU_STAGED_RULES', '');
    expect(WEBGPU_PLATFORM_RULES).toEqual([]);
    expect(candidateRules({ os: 'linux', browser: 'chromium' })).toEqual([]);
  });

  it('includes the staged platform when explicitly enabled', () => {
    vi.stubEnv('VITE_LUXAR_WEBGPU_STAGED_RULES', '1');
    expect(defaultPlatformRules()).toEqual(STAGED_WEBGPU_PLATFORM_RULES);
  });

  it('stages only Linux + Chromium + NVIDIA', () => {
    expect(STAGED_WEBGPU_PLATFORM_RULES.map((r) => [r.os, r.browser, r.vendor])).toEqual([
      ['linux', 'chromium', 'nvidia'],
    ]);
    const staged = (os: 'macos' | 'windows' | 'linux', browser: 'chromium' | 'firefox') =>
      candidateRules({ os, browser }, STAGED_WEBGPU_PLATFORM_RULES);
    expect(staged('macos', 'chromium')).toEqual([]);
    expect(staged('windows', 'chromium')).toEqual([]);
    expect(staged('linux', 'firefox')).toEqual([]);
  });

  it('matches the adapter vendor case-insensitively and rejects fallback adapters', () => {
    const linux = candidateRules(
      { os: 'linux', browser: 'chromium' },
      STAGED_WEBGPU_PLATFORM_RULES
    );
    expect(matchAdapterRule(linux, { vendor: 'NVIDIA' })?.vendor).toBe('nvidia');
    expect(matchAdapterRule(linux, { vendor: 'intel' })).toBeUndefined();
    expect(matchAdapterRule(linux, { vendor: 'nvidia', isFallbackAdapter: true })).toBeUndefined();
    expect(matchAdapterRule(linux, undefined)).toBeUndefined();
  });
});
