import { describe, it, expect } from 'vitest';
import {
  candidateRules,
  detectBrowserPlatform,
  matchAdapterRule,
  WEBGPU_PLATFORM_RULES,
} from '../../../../../scene/scene-manager/render-pipeline/backend-policy';

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
  it('admits only Linux + Chromium + NVIDIA today', () => {
    expect(WEBGPU_PLATFORM_RULES.map((r) => [r.os, r.browser, r.vendor])).toEqual([
      ['linux', 'chromium', 'nvidia'],
    ]);
    expect(candidateRules({ os: 'macos', browser: 'chromium' })).toEqual([]);
    expect(candidateRules({ os: 'windows', browser: 'chromium' })).toEqual([]);
    expect(candidateRules({ os: 'linux', browser: 'firefox' })).toEqual([]);
  });

  it('matches the adapter vendor case-insensitively and rejects fallback adapters', () => {
    const linux = candidateRules({ os: 'linux', browser: 'chromium' });
    expect(matchAdapterRule(linux, { vendor: 'NVIDIA' })?.vendor).toBe('nvidia');
    expect(matchAdapterRule(linux, { vendor: 'intel' })).toBeUndefined();
    expect(matchAdapterRule(linux, { vendor: 'nvidia', isFallbackAdapter: true })).toBeUndefined();
    expect(matchAdapterRule(linux, undefined)).toBeUndefined();
  });
});
