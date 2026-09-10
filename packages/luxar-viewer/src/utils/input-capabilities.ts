/**
 * Input and device capability profile — the single answer to "is this a
 * touch-first device, is it an iPhone or an iPad, can its pointer hover?"
 *
 * Every JS-side touch adaptation in the viewer (gesture routing, long-press
 * menus, mobile rendering budgets, tap-friendly copy) keys off this profile so
 * the decision is made once, from the same signals, and so a test can stub
 * one module instead of `navigator` and `matchMedia` in a dozen places. CSS
 * adaptations do NOT go through here: they use the `(pointer: coarse)` /
 * `(any-hover: none)` media features directly, which is why the `?input=` override
 * below is documented as JS-only.
 *
 * Deliberately import-free: `rendering/pixel-ratio-cap.ts` and other
 * leaf modules depend on this file precisely so they can stay out of every
 * import cycle, and `utils/` is the one layer every other layer may reach.
 *
 * Signal semantics (all best-effort, all false/0/empty outside a browser, so
 * the no-information default is a hover-capable fine-pointer desktop — the
 * historical behaviour):
 *   - `coarsePointer`: the PRIMARY pointer is coarse (`(pointer: coarse)`), i.e.
 *     a finger. A touch-screen laptop with a trackpad reports `false`.
 *   - `anyHover`: SOME pointer can hover (`(any-hover: hover)`). An iPad with a
 *     Magic Keyboard trackpad reports `true` while still being coarse-primary,
 *     which is exactly the case that must keep hover tooltips.
 *   - iPadOS Safari reports a Macintosh user agent (desktop-site mode is the
 *     default since iPadOS 13). The masquerade is unmasked by `platform`
 *     starting with `Mac` together with `maxTouchPoints > 1` — no Mac has a
 *     multi-touch screen.
 *
 * @module utils/input-capabilities
 */

/** Coarse device tier used for memory / GPU / worker budgets. */
export type DeviceClass = 'mobile' | 'laptop' | 'desktop';

/** Raw browser signals the profile is derived from (injectable for tests). */
export interface InputSignals {
  /** `navigator.userAgent` ('' outside a browser). */
  userAgent: string;
  /** `navigator.platform` ('' outside a browser). iPadOS reports `MacIntel`. */
  platform: string;
  /** `navigator.maxTouchPoints` (0 outside a browser / on a mouse-only device). */
  maxTouchPoints: number;
  /** `matchMedia('(pointer: coarse)').matches` — the primary pointer is a finger. */
  coarsePointer: boolean;
  /** `matchMedia('(any-hover: hover)').matches` — at least one pointer can hover. */
  anyHover: boolean;
  /** `navigator.hardwareConcurrency` (0 when unknown). */
  cores: number;
}

/** The derived, memoised profile every touch adaptation reads. */
export interface InputProfile {
  /** Primary pointer is coarse → touch-first interaction and sizing. */
  coarsePointer: boolean;
  /**
   * Some pointer can hover → hover-revealed affordances (tooltips, captions)
   * stay meaningful. True for every mouse/trackpad machine AND for an iPad
   * with a trackpad; false for a bare phone or tablet.
   */
  hoverCapable: boolean;
  /** Number of simultaneous touch contacts the device reports. */
  touchPoints: number;
  /** iPhone / iPod touch (Safari lacks Fullscreen for non-video elements). */
  isIPhone: boolean;
  /**
   * iPad, including iPadOS masquerading as macOS. An iPhone with Safari's
   * "Request Desktop Website" active also presents the macOS UA and therefore
   * reads as an iPad here (the UA carries no phone marker); gate on `isIOS`
   * when the phone/tablet distinction is not load-bearing.
   */
  isIPad: boolean;
  /** `isIPhone || isIPad` — drives WebKit-on-iOS-only workarounds. */
  isIOS: boolean;
  /** Android (Chrome synthesises `contextmenu` on long-press; iOS does not). */
  isAndroid: boolean;
  /** Memory / GPU / worker budget tier. */
  deviceClass: DeviceClass;
  /** `'override'` when `?input=` forced the pointer part of the profile. */
  source: 'detected' | 'override';
}

/** Value of the `?input=` URL override. */
export type InputProfileOverride = 'touch' | 'mouse';

/**
 * Logical-core count at/above which a NON-mobile device is treated as a desktop
 * rather than a laptop. This is a deliberately WEAK proxy: laptop vs desktop is
 * not reliably distinguishable in-browser — there is no RAM API in WebKit
 * (`navigator.deviceMemory` is Chromium-only), no battery API in Safari, and
 * UA / core counts overlap (an 8-core MacBook vs an 8-core Mac mini). Both the
 * laptop and desktop budgets are safe on the 8 GB+ machines that run the
 * desktop app, so a misclassification is low-consequence.
 */
const DESKTOP_CORE_THRESHOLD = 12;

const COARSE_POINTER_QUERY = '(pointer: coarse)';
const ANY_HOVER_QUERY = '(any-hover: hover)';

function mediaMatches(query: string): boolean {
  return typeof matchMedia === 'function' ? matchMedia(query).matches : false;
}

/** Read the raw signals from the browser (best-effort; safe in node/jsdom). */
export function readInputSignals(): InputSignals {
  const nav = typeof navigator !== 'undefined' ? navigator : undefined;
  return {
    userAgent: nav?.userAgent ?? '',
    platform: nav?.platform ?? '',
    maxTouchPoints: nav?.maxTouchPoints ?? 0,
    coarsePointer: mediaMatches(COARSE_POINTER_QUERY),
    anyHover: mediaMatches(ANY_HOVER_QUERY),
    cores: nav?.hardwareConcurrency ?? 0,
  };
}

/**
 * Infer a coarse device class from browser signals. `mobile` is reliable
 * (mobile UA, or a touch + coarse-pointer device — which also catches iPadOS,
 * whose UA masquerades as macOS). `desktop` vs `laptop` is the weak core-count
 * proxy (`DESKTOP_CORE_THRESHOLD`). A Windows touch laptop
 * (`maxTouchPoints > 0` but a fine primary pointer) is NOT mobile.
 */
export function inferDeviceClass(signals: InputSignals): DeviceClass {
  const mobileUA = /Mobi|Android|iPhone|iPod|iPad/i.test(signals.userAgent);
  if (mobileUA || (signals.maxTouchPoints > 0 && signals.coarsePointer)) return 'mobile';
  return signals.cores >= DESKTOP_CORE_THRESHOLD ? 'desktop' : 'laptop';
}

/**
 * Derive the profile from raw signals. Pure — this is the unit-test surface.
 *
 * `hoverCapable` is `anyHover || !coarsePointer`: a fine primary pointer is a
 * mouse or trackpad and can hover even where `matchMedia` is unavailable, so
 * the no-signal default (node, jsdom, old browsers) stays "desktop".
 */
export function deriveInputProfile(
  signals: InputSignals,
  source: InputProfile['source'] = 'detected'
): InputProfile {
  const isIPhone = /iPhone|iPod/i.test(signals.userAgent);
  const isIPad =
    /iPad/i.test(signals.userAgent) ||
    (signals.platform.startsWith('Mac') && signals.maxTouchPoints > 1);
  return {
    coarsePointer: signals.coarsePointer,
    hoverCapable: signals.anyHover || !signals.coarsePointer,
    touchPoints: signals.maxTouchPoints,
    isIPhone,
    isIPad,
    isIOS: isIPhone || isIPad,
    isAndroid: /Android/i.test(signals.userAgent),
    deviceClass: inferDeviceClass(signals),
    source,
  };
}

/**
 * Apply the `?input=touch|mouse` override to detected signals. `touch` makes
 * the device look like a bare phone/tablet (coarse, no hover, mobile budgets);
 * `mouse` makes it look like a mouse-driven machine (fine pointer, hover) while
 * keeping the detected memory tier. The platform flags (`isIPad`, …) stay
 * DETECTED in both cases: they gate WebKit workarounds that remain true
 * regardless of how the user chose to interact.
 */
function applyOverride(signals: InputSignals, mode: InputProfileOverride): InputProfile {
  // Platform flags come from the REAL signals: forcing touch points must not
  // turn a Mac into an iPad, and forcing a mouse must not hide a real iPad.
  const detected = deriveInputProfile(signals, 'override');
  if (mode === 'touch') {
    return {
      ...detected,
      coarsePointer: true,
      hoverCapable: false,
      touchPoints: Math.max(5, signals.maxTouchPoints),
      deviceClass: 'mobile',
    };
  }
  // `deviceClass` is a memory / GPU tier, not a pointer property, so the mouse
  // override leaves it at the DETECTED tier: on WebKit without
  // `?cacheBudgetMB=` the device-class pool is the operative cache budget, and
  // lifting a phone from 384 MB to 1 GB because the user wants mouse-style
  // interaction would be the one budget change with no upside. (`touch`
  // lowering the tier is how a desktop emulates a phone, and stays.)
  return {
    ...detected,
    coarsePointer: false,
    hoverCapable: true,
    touchPoints: 0,
  };
}

let override: InputProfileOverride | null = null;
let cached: InputProfile | undefined;
let invalidationArmed = false;

/**
 * Arm one-time `change` listeners on the two media queries the profile depends
 * on, so an iPad gaining a trackpad (or DevTools toggling device emulation)
 * refreshes the memoised profile instead of serving a stale one.
 */
function armInvalidation(): void {
  if (invalidationArmed || typeof matchMedia !== 'function') return;
  invalidationArmed = true;
  const invalidate = (): void => {
    cached = undefined;
  };
  for (const query of [COARSE_POINTER_QUERY, ANY_HOVER_QUERY]) {
    const mql = matchMedia(query);
    if (typeof mql.addEventListener === 'function') {
      mql.addEventListener('change', invalidate);
    } else if (typeof mql.addListener === 'function') {
      // Safari < 14 exposes only the deprecated form.
      mql.addListener(invalidate);
    }
  }
}

/**
 * Force the pointer half of the profile for the session (`?input=touch|mouse`)
 * or clear it (`null`). A testing and support aid, JS-only: stylesheets keep
 * following the real `(pointer: coarse)` / `(any-hover: none)` media features, so
 * a faithful end-to-end check still needs device emulation or a real device.
 */
export function setInputProfileOverride(mode: InputProfileOverride | null): void {
  override = mode;
  cached = undefined;
}

/**
 * The memoised profile for this browser session. Cheap after the first call;
 * re-derived when the pointer media queries change or the override is set.
 */
export function getInputProfile(): InputProfile {
  if (cached !== undefined) return cached;
  armInvalidation();
  const signals = readInputSignals();
  cached = override === null ? deriveInputProfile(signals) : applyOverride(signals, override);
  return cached;
}

/**
 * Whether a pointer event should take the TOUCH gesture path: a finger, or a
 * pen used as a finger on a touch-first device (iPad + Pencil, with no
 * secondary button held). A pen on a fine-pointer desktop (Wacom) keeps the
 * mouse mapping, and any secondary-button press or drag falls through to it
 * everywhere so the secondary action stays reachable.
 *
 * Must agree across a whole gesture: `pointermove` reports `button === -1`
 * ("no button changed"), so held buttons must come from `buttons` rather than
 * `button` — otherwise a secondary-button drag can change paths on every move.
 */
export function isTouchLikePointer(event: {
  pointerType: string;
  button?: number;
  buttons?: number;
}): boolean {
  if (event.pointerType === 'touch') return true;
  if (event.pointerType !== 'pen') return false;
  if (!getInputProfile().coarsePointer) return false;
  const button = event.button ?? 0; // -1 on pointermove: no button changed
  if (((event.buttons ?? 0) & ~1) !== 0) return false; // any secondary button held
  return button <= 0;
}

/** Reset memoised state between tests. */
export function resetInputProfileForTests(): void {
  override = null;
  cached = undefined;
  invalidationArmed = false;
}
