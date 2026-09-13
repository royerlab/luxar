/**
 * Kiosk mode has two inputs that are not equal partners, and the tests are
 * mostly about that asymmetry: the URL is the operator's channel and must be
 * able to lock a display whose store knows nothing about kiosk mode.
 */

import { describe, expect, it } from 'vitest';

import { DEFAULT_KIOSK_WATCHDOG_S, KIOSK_MODE_OFF, resolveKioskMode } from '../../../config/kiosk';

describe('resolveKioskMode', () => {
  it('is off when nothing asks for it', () => {
    for (const authored of [null, undefined, {}, { enabled: false }]) {
      expect(resolveKioskMode(authored, false)).toEqual(KIOSK_MODE_OFF);
    }
  });

  it('leaves everything permissive when off', () => {
    // The default has to be "an ordinary viewer", or a scene without the block
    // would quietly lose its input.
    const mode = resolveKioskMode(null, false);
    expect(mode.allowPointer).toBe(true);
    expect(mode.allowKeyboard).toBe(true);
    expect(mode.showPanels).toBe(true);
    expect(mode.watchdogReload).toBe(false);
  });

  it('locks down by default once enabled', () => {
    // Enabling kiosk mode without saying more must mean the restrictive thing;
    // otherwise the switch does nothing on its own.
    expect(resolveKioskMode({ enabled: true }, false)).toEqual({
      enabled: true,
      allowPointer: false,
      allowKeyboard: false,
      showPanels: false,
      watchdogReload: false,
      watchdogGraceS: DEFAULT_KIOSK_WATCHDOG_S,
    });
  });

  it('lets ?kiosk lock a store that has never heard of kiosk mode', () => {
    // The case the override exists for: an exhibit borrowed for a display it
    // was not authored for.
    const mode = resolveKioskMode(null, true);
    expect(mode.enabled).toBe(true);
    expect(mode.allowPointer).toBe(false);
    expect(mode.showPanels).toBe(false);
  });

  it('honours authored permissions under the URL override', () => {
    // `?kiosk` turns it on; it does not overrule what the author allowed
    // INSIDE kiosk mode, so a touch-driven exhibit stays touch-driven.
    const mode = resolveKioskMode({ allow_pointer: true }, true);
    expect(mode.enabled).toBe(true);
    expect(mode.allowPointer).toBe(true);
    expect(mode.allowKeyboard).toBe(false);
  });

  it('cannot be turned OFF by the URL', () => {
    // Deliberate: an operator who does not want kiosk mode omits the flag,
    // and a stray query string must not be able to unlock an exhibit.
    expect(resolveKioskMode({ enabled: true }, false).enabled).toBe(true);
  });

  describe('the watchdog', () => {
    it('stays off unless asked for', () => {
      // A reload throws away every warm cache the display has built, and the
      // viewer already recovers from context loss on its own.
      expect(resolveKioskMode({ enabled: true }, false).watchdogReload).toBe(false);
    });

    it('takes an authored grace period', () => {
      expect(resolveKioskMode({ enabled: true, watchdog_grace_s: 30 }, false)).toMatchObject({
        watchdogGraceS: 30,
      });
    });

    it('accepts zero as a real grace period', () => {
      // "Reload immediately" is a legitimate choice for a display nobody can
      // reach, so a falsy check here would silently substitute ten seconds.
      expect(resolveKioskMode({ enabled: true, watchdog_grace_s: 0 }, false).watchdogGraceS).toBe(
        0
      );
    });

    it.each([-1, Number.NaN, Number.POSITIVE_INFINITY, '20', null])(
      'falls back to the default for %p',
      (grace) => {
        expect(
          resolveKioskMode({ enabled: true, watchdog_grace_s: grace as number }, false)
            .watchdogGraceS
        ).toBe(DEFAULT_KIOSK_WATCHDOG_S);
      }
    );
  });

  it('ignores non-boolean flags rather than coercing them', () => {
    // A store is untrusted input; `"false"` is truthy in JavaScript and would
    // otherwise silently unlock a display.
    const mode = resolveKioskMode(
      { enabled: true, allow_pointer: 'false' as unknown as boolean },
      false
    );
    expect(mode.allowPointer).toBe(false);
  });
});
