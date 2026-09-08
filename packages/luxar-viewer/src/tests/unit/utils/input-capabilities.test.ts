// @vitest-environment jsdom
/**
 * Unit tests for the shared input / device capability profile.
 *
 * `deriveInputProfile` is pure and is exercised table-style over the devices
 * the touch work must tell apart: iPhone, iPad (real UA and the iPadOS
 * "Macintosh" masquerade), iPad with a trackpad, Android, macOS with a mouse,
 * and a Windows touch laptop. The memoised `getInputProfile()` is exercised
 * against a stubbed `matchMedia` / `navigator` for override and invalidation.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  deriveInputProfile,
  getInputProfile,
  inferDeviceClass,
  isTouchLikePointer,
  readInputSignals,
  resetInputProfileForTests,
  setInputProfileOverride,
  type InputSignals,
} from '../../../utils/input-capabilities';

const IPHONE_UA =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1';
const IPAD_UA =
  'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148';
/** iPadOS 13+ default: the desktop Safari UA, indistinguishable from a Mac. */
const MAC_UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15';
const ANDROID_UA =
  'Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Mobile Safari/537.36';
const WINDOWS_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36';

const base: InputSignals = {
  userAgent: MAC_UA,
  platform: 'MacIntel',
  maxTouchPoints: 0,
  coarsePointer: false,
  anyHover: true,
  cores: 8,
};

describe('deriveInputProfile', () => {
  it('iPhone Safari: coarse, no hover, iPhone/iOS, mobile', () => {
    const p = deriveInputProfile({
      ...base,
      userAgent: IPHONE_UA,
      platform: 'iPhone',
      maxTouchPoints: 5,
      coarsePointer: true,
      anyHover: false,
      cores: 6,
    });
    expect(p).toMatchObject({
      coarsePointer: true,
      hoverCapable: false,
      touchPoints: 5,
      isIPhone: true,
      isIPad: false,
      isIOS: true,
      isAndroid: false,
      deviceClass: 'mobile',
      source: 'detected',
    });
  });

  it('iPadOS masquerading as macOS: Mac platform + multi-touch → iPad, mobile', () => {
    const p = deriveInputProfile({
      ...base,
      maxTouchPoints: 5,
      coarsePointer: true,
      anyHover: false,
    });
    expect(p.isIPad).toBe(true);
    expect(p.isIOS).toBe(true);
    expect(p.isIPhone).toBe(false);
    expect(p.deviceClass).toBe('mobile');
    expect(p.hoverCapable).toBe(false);
  });

  it('iPad with a legacy iPad UA is still an iPad', () => {
    const p = deriveInputProfile({
      ...base,
      userAgent: IPAD_UA,
      platform: 'iPad',
      maxTouchPoints: 5,
      coarsePointer: true,
      anyHover: false,
    });
    expect(p.isIPad).toBe(true);
    expect(p.deviceClass).toBe('mobile');
  });

  it('iPad with a trackpad: still coarse-primary and mobile, but hover-capable', () => {
    const p = deriveInputProfile({
      ...base,
      maxTouchPoints: 5,
      coarsePointer: true,
      anyHover: true,
    });
    expect(p.coarsePointer).toBe(true);
    expect(p.hoverCapable).toBe(true);
    expect(p.isIPad).toBe(true);
    expect(p.deviceClass).toBe('mobile');
  });

  it('Android Chrome: coarse, no hover, Android, mobile', () => {
    const p = deriveInputProfile({
      ...base,
      userAgent: ANDROID_UA,
      platform: 'Linux armv8l',
      maxTouchPoints: 5,
      coarsePointer: true,
      anyHover: false,
    });
    expect(p).toMatchObject({
      coarsePointer: true,
      hoverCapable: false,
      isAndroid: true,
      isIOS: false,
      isIPad: false,
      deviceClass: 'mobile',
    });
  });

  it('macOS with a mouse: fine, hover, not iOS, laptop/desktop by core count', () => {
    expect(deriveInputProfile(base)).toMatchObject({
      coarsePointer: false,
      hoverCapable: true,
      touchPoints: 0,
      isIPhone: false,
      isIPad: false,
      isIOS: false,
      isAndroid: false,
      deviceClass: 'laptop',
    });
    expect(deriveInputProfile({ ...base, cores: 16 }).deviceClass).toBe('desktop');
  });

  it('a Mac with a single-touch signal is NOT an iPad', () => {
    // No Mac has a multi-touch screen; the masquerade needs > 1 touch point.
    expect(deriveInputProfile({ ...base, maxTouchPoints: 1 }).isIPad).toBe(false);
  });

  it('Windows touch laptop: touch points but a fine primary pointer → not mobile, hover-capable', () => {
    const p = deriveInputProfile({
      ...base,
      userAgent: WINDOWS_UA,
      platform: 'Win32',
      maxTouchPoints: 10,
      coarsePointer: false,
      anyHover: true,
      cores: 8,
    });
    expect(p.coarsePointer).toBe(false);
    expect(p.hoverCapable).toBe(true);
    expect(p.touchPoints).toBe(10);
    expect(p.isIPad).toBe(false);
    expect(p.deviceClass).toBe('laptop');
  });

  it('no signals at all (node) resolves to a hover-capable fine-pointer laptop', () => {
    const p = deriveInputProfile({
      userAgent: '',
      platform: '',
      maxTouchPoints: 0,
      coarsePointer: false,
      anyHover: false,
      cores: 0,
    });
    expect(p.coarsePointer).toBe(false);
    expect(p.hoverCapable).toBe(true);
    expect(p.deviceClass).toBe('laptop');
  });
});

describe('inferDeviceClass', () => {
  it('mobile UA wins regardless of pointer signals', () => {
    expect(inferDeviceClass({ ...base, userAgent: IPHONE_UA })).toBe('mobile');
    expect(inferDeviceClass({ ...base, userAgent: ANDROID_UA, cores: 16 })).toBe('mobile');
  });

  it('touch + coarse pointer is mobile; touch + fine pointer is not', () => {
    expect(inferDeviceClass({ ...base, maxTouchPoints: 5, coarsePointer: true })).toBe('mobile');
    expect(inferDeviceClass({ ...base, maxTouchPoints: 5, coarsePointer: false })).toBe('laptop');
  });

  it('desktop at or above 12 cores, laptop below (unknown cores → laptop)', () => {
    expect(inferDeviceClass({ ...base, cores: 12 })).toBe('desktop');
    expect(inferDeviceClass({ ...base, cores: 11 })).toBe('laptop');
    expect(inferDeviceClass({ ...base, cores: 0 })).toBe('laptop');
  });
});

// ---------------------------------------------------------------------------
// Browser-backed API: readInputSignals / getInputProfile / override / MQL.
// ---------------------------------------------------------------------------

type ChangeListener = () => void;

interface MatchMediaStub {
  restore: () => void;
  setMatching: (queries: Set<string>) => void;
  fireChange: () => void;
}

function installMatchMedia(matching: Set<string>): MatchMediaStub {
  const original = window.matchMedia;
  const listeners: ChangeListener[] = [];
  let current = matching;
  const stub = vi.fn().mockImplementation((q: string) => ({
    get matches() {
      return current.has(q);
    },
    media: q,
    onchange: null,
    addEventListener: (_type: string, cb: ChangeListener) => listeners.push(cb),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
  Object.defineProperty(window, 'matchMedia', { configurable: true, value: stub });
  return {
    restore: () =>
      Object.defineProperty(window, 'matchMedia', { configurable: true, value: original }),
    setMatching: (queries) => {
      current = queries;
    },
    fireChange: () => listeners.forEach((cb) => cb()),
  };
}

function setNavigator(over: Partial<Navigator>): () => void {
  const originals = new Map<string, PropertyDescriptor | undefined>();
  for (const [key, value] of Object.entries(over)) {
    originals.set(key, Object.getOwnPropertyDescriptor(navigator, key));
    Object.defineProperty(navigator, key, { configurable: true, value });
  }
  return () => {
    for (const [key, desc] of originals) {
      if (desc) Object.defineProperty(navigator, key, desc);
      else delete (navigator as unknown as Record<string, unknown>)[key];
    }
  };
}

describe('readInputSignals / getInputProfile (browser-backed)', () => {
  let mm: MatchMediaStub;
  let restoreNav: () => void;

  beforeEach(() => {
    resetInputProfileForTests();
    mm = installMatchMedia(new Set());
    restoreNav = setNavigator({
      userAgent: MAC_UA,
      platform: 'MacIntel',
      maxTouchPoints: 0,
      hardwareConcurrency: 8,
    });
  });

  afterEach(() => {
    restoreNav();
    mm.restore();
    resetInputProfileForTests();
  });

  it('reads the raw signals from navigator and matchMedia', () => {
    mm.setMatching(new Set(['(any-hover: hover)']));
    expect(readInputSignals()).toEqual({
      userAgent: MAC_UA,
      platform: 'MacIntel',
      maxTouchPoints: 0,
      coarsePointer: false,
      anyHover: true,
      cores: 8,
    });
  });

  it('memoises the profile and re-derives it when a pointer media query changes', () => {
    const first = getInputProfile();
    expect(first.coarsePointer).toBe(false);
    expect(getInputProfile()).toBe(first); // same object → memoised

    // An iPad unmasked by DevTools emulation or a trackpad detaching.
    restoreNav();
    restoreNav = setNavigator({
      userAgent: MAC_UA,
      platform: 'MacIntel',
      maxTouchPoints: 5,
      hardwareConcurrency: 8,
    });
    mm.setMatching(new Set(['(pointer: coarse)']));
    expect(getInputProfile()).toBe(first); // no change event yet → still cached

    mm.fireChange();
    const second = getInputProfile();
    expect(second).not.toBe(first);
    expect(second.coarsePointer).toBe(true);
    expect(second.isIPad).toBe(true);
    expect(second.deviceClass).toBe('mobile');
  });

  it('?input=touch forces a bare-touch profile but keeps the detected platform flags', () => {
    setInputProfileOverride('touch');
    const p = getInputProfile();
    expect(p).toMatchObject({
      coarsePointer: true,
      hoverCapable: false,
      deviceClass: 'mobile',
      source: 'override',
      isIPad: false,
      isIOS: false,
    });
    expect(p.touchPoints).toBeGreaterThanOrEqual(5);
  });

  it('?input=mouse forces a mouse profile on a real iPad but keeps isIPad and the mobile tier', () => {
    restoreNav();
    restoreNav = setNavigator({
      userAgent: MAC_UA,
      platform: 'MacIntel',
      maxTouchPoints: 5,
      hardwareConcurrency: 8,
    });
    mm.setMatching(new Set(['(pointer: coarse)']));
    setInputProfileOverride('mouse');
    const p = getInputProfile();
    expect(p).toMatchObject({
      coarsePointer: false,
      hoverCapable: true,
      touchPoints: 0,
      deviceClass: 'mobile', // detected tier kept — the override is pointer-only
      source: 'override',
      isIPad: true,
      isIOS: true,
    });
  });

  it('?input=mouse on a phone UA keeps the detected (mobile) memory tier', () => {
    restoreNav();
    restoreNav = setNavigator({
      userAgent: IPHONE_UA,
      platform: 'iPhone',
      maxTouchPoints: 5,
      hardwareConcurrency: 8,
    });
    mm.setMatching(new Set(['(pointer: coarse)']));
    setInputProfileOverride('mouse');
    // The tier is a memory budget, not a pointer property: on WebKit it is the
    // operative cache pool, and `mouse` must never raise it.
    expect(getInputProfile().deviceClass).toBe('mobile');
    expect(getInputProfile().coarsePointer).toBe(false);
    expect(getInputProfile().isIPhone).toBe(true);
  });

  it('clearing the override returns to detection', () => {
    setInputProfileOverride('touch');
    expect(getInputProfile().source).toBe('override');
    setInputProfileOverride(null);
    expect(getInputProfile()).toMatchObject({ source: 'detected', coarsePointer: false });
  });

  it('falls back to the deprecated addListener when addEventListener is absent', () => {
    mm.restore();
    const addListener = vi.fn();
    const legacy = vi.fn().mockImplementation((q: string) => ({
      matches: false,
      media: q,
      addListener,
    }));
    Object.defineProperty(window, 'matchMedia', { configurable: true, value: legacy });
    getInputProfile();
    expect(addListener).toHaveBeenCalledTimes(2);
    mm = installMatchMedia(new Set()); // so afterEach restore has something sane to undo
  });
});

describe('isTouchLikePointer', () => {
  let mm: MatchMediaStub;
  let restoreNav: () => void;

  beforeEach(() => {
    resetInputProfileForTests();
    mm = installMatchMedia(new Set());
    restoreNav = setNavigator({ userAgent: MAC_UA, platform: 'MacIntel', maxTouchPoints: 0 });
  });

  afterEach(() => {
    restoreNav();
    mm.restore();
    resetInputProfileForTests();
  });

  it('a finger is always touch-like; a mouse never is', () => {
    expect(isTouchLikePointer({ pointerType: 'touch' })).toBe(true);
    expect(isTouchLikePointer({ pointerType: 'mouse', button: 0 })).toBe(false);
  });

  it('a pen on a fine-pointer desktop keeps the mouse mapping', () => {
    expect(isTouchLikePointer({ pointerType: 'pen', button: 0, buttons: 1 })).toBe(false);
  });

  it('a pen on a coarse-pointer device is touch-like for the primary tip only', () => {
    setInputProfileOverride('touch');
    expect(isTouchLikePointer({ pointerType: 'pen', button: 0, buttons: 1 })).toBe(true);
    // Barrel button held: falls through to the secondary (mouse) mapping.
    expect(isTouchLikePointer({ pointerType: 'pen', button: 0, buttons: 3 })).toBe(false);
    expect(isTouchLikePointer({ pointerType: 'pen', button: 2, buttons: 2 })).toBe(false);
    // Barrel RELEASE: button 2 with no buttons held is still the secondary path.
    expect(isTouchLikePointer({ pointerType: 'pen', button: 2, buttons: 0 })).toBe(false);
  });

  it('a pen drag stays touch-like across the whole gesture (pointermove has button -1)', () => {
    setInputProfileOverride('touch');
    // Measured Chromium shape: down {0,1} → move {-1,1} → up {0,0}.
    expect(isTouchLikePointer({ pointerType: 'pen', button: 0, buttons: 1 })).toBe(true);
    expect(isTouchLikePointer({ pointerType: 'pen', button: -1, buttons: 1 })).toBe(true);
    expect(isTouchLikePointer({ pointerType: 'pen', button: 0, buttons: 0 })).toBe(true);
    // Hover move with no contact (pen in proximity) is also not a mouse drag.
    expect(isTouchLikePointer({ pointerType: 'pen', button: -1, buttons: 0 })).toBe(true);
    // A barrel drag stays on the mouse path for every move too.
    expect(isTouchLikePointer({ pointerType: 'pen', button: -1, buttons: 3 })).toBe(false);
    // An eraser drag also stays on the mouse path for down, move, and release.
    expect(isTouchLikePointer({ pointerType: 'pen', button: 5, buttons: 32 })).toBe(false);
    expect(isTouchLikePointer({ pointerType: 'pen', button: -1, buttons: 32 })).toBe(false);
    expect(isTouchLikePointer({ pointerType: 'pen', button: 5, buttons: 0 })).toBe(false);
  });
});

describe('readInputSignals without browser APIs', () => {
  it('falls back to empty/zero signals when matchMedia and navigator fields are absent', () => {
    const originalMatchMedia = window.matchMedia;
    Object.defineProperty(window, 'matchMedia', { configurable: true, value: undefined });
    const restoreNav = setNavigator({
      userAgent: undefined,
      platform: undefined,
      maxTouchPoints: undefined,
      hardwareConcurrency: undefined,
    } as unknown as Partial<Navigator>);
    try {
      expect(readInputSignals()).toEqual({
        userAgent: '',
        platform: '',
        maxTouchPoints: 0,
        coarsePointer: false,
        anyHover: false,
        cores: 0,
      });
      // No matchMedia → nothing to arm; the profile still resolves.
      resetInputProfileForTests();
      expect(getInputProfile().deviceClass).toBe('laptop');
    } finally {
      restoreNav();
      Object.defineProperty(window, 'matchMedia', {
        configurable: true,
        value: originalMatchMedia,
      });
      resetInputProfileForTests();
    }
  });

  it('a pen event with no button fields counts as the primary tip', () => {
    resetInputProfileForTests();
    setInputProfileOverride('touch');
    expect(isTouchLikePointer({ pointerType: 'pen' })).toBe(true);
    resetInputProfileForTests();
  });
});
