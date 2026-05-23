/**
 * Unit tests for `utils/platform.ts`.
 *
 * Closes audit finding utils.md G2 — the module had no dedicated test
 * file. `isMacPlatform()` is a leaf that fans out into keyboard-shortcut
 * handling, copy/paste behavior, and the menu accelerator labels; a
 * regression here is a cross-feature regression.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { isMacPlatform } from '../../../utils/platform';

describe('isMacPlatform', () => {
  const realNavigator = globalThis.navigator;

  afterEach(() => {
    vi.unstubAllGlobals();
    // Belt + suspenders: explicitly restore to avoid leaking into sibling tests.
    Object.defineProperty(globalThis, 'navigator', {
      value: realNavigator,
      configurable: true,
      writable: true,
    });
  });

  describe('positive cases (platform starts with "Mac")', () => {
    it('returns true for MacIntel (Intel macOS)', () => {
      vi.stubGlobal('navigator', { platform: 'MacIntel' });
      expect(isMacPlatform()).toBe(true);
    });

    it('returns true for MacPPC (legacy PowerPC macOS)', () => {
      vi.stubGlobal('navigator', { platform: 'MacPPC' });
      expect(isMacPlatform()).toBe(true);
    });

    it('returns true for Mac68K (legacy 68K macOS — defensive)', () => {
      // Source explicitly relies on `startsWith('Mac')`. Any future
      // historical or test-fixture `Mac*` string is intentionally covered.
      vi.stubGlobal('navigator', { platform: 'Mac68K' });
      expect(isMacPlatform()).toBe(true);
    });
  });

  describe('negative cases (non-Mac platforms)', () => {
    it('returns false for Win32', () => {
      vi.stubGlobal('navigator', { platform: 'Win32' });
      expect(isMacPlatform()).toBe(false);
    });

    it('returns false for Linux x86_64', () => {
      vi.stubGlobal('navigator', { platform: 'Linux x86_64' });
      expect(isMacPlatform()).toBe(false);
    });

    it('returns false for iPhone (mobile Safari) — Mac substring must be at start', () => {
      vi.stubGlobal('navigator', { platform: 'iPhone' });
      expect(isMacPlatform()).toBe(false);
    });

    it('returns false for empty platform string', () => {
      vi.stubGlobal('navigator', { platform: '' });
      expect(isMacPlatform()).toBe(false);
    });

    it('returns false when "Mac" appears mid-string (substring, not prefix)', () => {
      // Doc-string says "starts with 'Mac' only on desktop macOS". This
      // test pins that startsWith — not includes — is the contract.
      vi.stubGlobal('navigator', { platform: 'iMac (synthetic)' });
      expect(isMacPlatform()).toBe(false);
    });
  });

  describe('defensive — non-browser context (navigator undefined)', () => {
    beforeEach(() => {
      // Delete navigator entirely (Node-test parity). Use defineProperty
      // so we can restore in afterEach.
      Object.defineProperty(globalThis, 'navigator', {
        value: undefined,
        configurable: true,
        writable: true,
      });
    });

    it('returns false when navigator is undefined (Node/SSR fallback)', () => {
      expect(isMacPlatform()).toBe(false);
    });
  });
});
