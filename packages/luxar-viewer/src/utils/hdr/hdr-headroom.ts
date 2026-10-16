/**
 * HDR output headroom (PROTOTYPE, `?hdr`): how far above SDR white the
 * extended-range canvas may go, as a multiple of SDR white.
 *
 * The web exposes no number for this. `(dynamic-range: high)` says only
 * that the display CAN exceed SDR white, not by how much, and the real
 * figure moves with the OS brightness setting (macOS EDR headroom grows as
 * the panel is dimmed). So it is a user setting: shown in the Rendering
 * panel only while HDR output is active, persisted per browser so a display
 * calibrated once stays calibrated, and overridable per URL (`?hdr=4`).
 *
 * @module utils/hdr/hdr-headroom
 */

import { StorageKeys } from '../storage-keys';

/**
 * Default headroom: 2x SDR white (one stop). Deliberately conservative.
 * Overshooting the real headroom makes the OS clip or compress the top of
 * the range (the brightest detail is lost); undershooting only leaves some
 * headroom unused. Entry-level HDR monitors (DisplayHDR 400) at a typical
 * 200-240 nit SDR white have about 1.7-2x; MacBook Pro XDR panels have
 * roughly 2-3x at full brightness and much more when dimmed.
 */
export const DEFAULT_HDR_HEADROOM = 2;

/** Slider range for the headroom control. */
export const HDR_HEADROOM_MIN = 1;
export const HDR_HEADROOM_MAX = 16;

function clampHeadroom(value: number): number {
  return Math.min(HDR_HEADROOM_MAX, Math.max(HDR_HEADROOM_MIN, value));
}

/** The persisted headroom, or null when none is stored / storage is unavailable. */
export function loadStoredHdrHeadroom(): number | null {
  try {
    const raw = globalThis.localStorage?.getItem(StorageKeys.hdrHeadroom);
    if (raw === null || raw === undefined) return null;
    const n = Number.parseFloat(raw);
    return Number.isFinite(n) && n > 0 ? clampHeadroom(n) : null;
  } catch {
    return null;
  }
}

/** Persist the headroom chosen in the Rendering panel. */
export function storeHdrHeadroom(value: number): void {
  try {
    globalThis.localStorage?.setItem(StorageKeys.hdrHeadroom, String(clampHeadroom(value)));
  } catch {
    // Storage disabled (private mode, sandboxed iframe): the setting is per-session.
  }
}

/** Headroom to start with: explicit URL value, else stored, else the default. */
export function resolveHdrHeadroom(urlValue: number | null): number {
  if (urlValue !== null) return clampHeadroom(urlValue);
  return loadStoredHdrHeadroom() ?? DEFAULT_HDR_HEADROOM;
}
