/**
 * Kiosk mode: what an unattended public display is allowed to do.
 *
 * Not a security boundary — anyone at the host's keyboard can do anything —
 * but the difference between an exhibit that survives a day of visitors and
 * one that ends up showing the rendering-controls panel with the point size at
 * zero. A visitor who drags the camera off the tour cannot put it back.
 *
 * Resolution has two inputs and they are not equal partners:
 *
 * - the scene's authored `viewer_config.ui.kiosk` block, and
 * - `?kiosk` on the URL, which is a HARD override.
 *
 * The URL wins because it is the operator's channel and the store's is the
 * author's: a display whose scene predates the block, or one borrowed for an
 * exhibit it was never authored for, has to be lockable from the launch
 * command without rewriting the store.
 *
 * @module config/kiosk
 */

/** Seconds to wait for WebGL recovery before a watchdog reload. */
export const DEFAULT_KIOSK_WATCHDOG_S = 10;

/** The resolved decision. Every field definite — no "unset" left to interpret. */
export interface KioskMode {
  enabled: boolean;
  allowPointer: boolean;
  allowKeyboard: boolean;
  showPanels: boolean;
  watchdogReload: boolean;
  watchdogGraceS: number;
}

/** The authored block, as it arrives out of the store's attributes. */
export interface ZarrKioskConfig {
  enabled?: boolean;
  allow_pointer?: boolean;
  allow_keyboard?: boolean;
  show_panels?: boolean;
  watchdog_reload?: boolean;
  watchdog_grace_s?: number;
}

/**
 * What a display does when nothing asks for kiosk mode: everything as usual.
 *
 * Spelled out rather than left implicit so a caller can compare against it,
 * and so adding a field to `KioskMode` forces a decision about its default
 * here instead of silently defaulting to `undefined`.
 */
export const KIOSK_MODE_OFF: KioskMode = {
  enabled: false,
  allowPointer: true,
  allowKeyboard: true,
  showPanels: true,
  watchdogReload: false,
  watchdogGraceS: DEFAULT_KIOSK_WATCHDOG_S,
};

function boolOr(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

/**
 * Resolve kiosk mode from the authored block and the URL flag.
 *
 * @param authored The scene's `ui.kiosk`, or null/undefined when it has none.
 *   Untrusted shape: a store is a file someone downloaded, so a non-boolean
 *   field falls back rather than being coerced.
 * @param urlFlag Whether `?kiosk` was present.
 *
 * When kiosk mode is on, the restrictive defaults apply to every field the
 * author did not set — that is the point of the switch. An author who wants a
 * locked display that still takes touch says `allow_pointer=True` explicitly.
 */
export function resolveKioskMode(
  authored: ZarrKioskConfig | null | undefined,
  urlFlag: boolean
): KioskMode {
  const block = authored ?? {};
  // `?kiosk` forces it on; it deliberately cannot force it OFF, because an
  // operator who does not want kiosk mode simply omits the flag, whereas
  // "the store says kiosk but I disagree" is not a case worth a second
  // parameter and would make an exhibit unlockable by a stray query string.
  const enabled = urlFlag || boolOr(block.enabled, false);
  if (!enabled) return { ...KIOSK_MODE_OFF };

  const grace = block.watchdog_grace_s;
  return {
    enabled: true,
    // Restrictive by default once enabled, permissive only where asked.
    allowPointer: boolOr(block.allow_pointer, false),
    allowKeyboard: boolOr(block.allow_keyboard, false),
    showPanels: boolOr(block.show_panels, false),
    // The watchdog is the one thing NOT on by default: reloading a display
    // is destructive of every warm cache it has built, and the viewer already
    // recovers from context loss on its own where it can.
    watchdogReload: boolOr(block.watchdog_reload, false),
    watchdogGraceS:
      typeof grace === 'number' && Number.isFinite(grace) && grace >= 0
        ? grace
        : DEFAULT_KIOSK_WATCHDOG_S,
  };
}
