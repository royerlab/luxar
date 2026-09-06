/**
 * Cross-browser wheel-delta normalization.
 *
 * `WheelEvent.deltaY` is only meaningful together with `deltaMode`, and the
 * viewer used to read it as if it were always pixels. One physical notch of
 * the same mouse therefore differed by roughly 32x between browsers:
 * Chromium and WebKit report `deltaMode = DOM_DELTA_PIXEL` with
 * `deltaY = 100`, while Firefox reports `deltaMode = DOM_DELTA_LINE` with
 * `deltaY = 3`. Wheel zoom, Shift+wheel roll and Ctrl+wheel FOV all took that
 * raw number, so on Firefox each of them moved about a thirtieth as far per
 * notch — wheel zoom felt essentially dead.
 *
 * This module converts a wheel event's vertical delta into a
 * pixel-mode-equivalent number so every wheel entry point can keep doing the
 * arithmetic it already does, on a browser-independent input.
 *
 * @module utils/wheel-delta
 */

import { clamp } from './clamp';

/**
 * Pixels one `DOM_DELTA_LINE` unit stands for. The conventional px-per-line
 * used by browsers and scroll libraries; it turns Firefox's 3-line notch into
 * 48 px, the same order as Chromium's 100 px pixel-mode notch.
 */
export const PIXELS_PER_LINE = 16;

/**
 * Fallback page height (px) for `DOM_DELTA_PAGE` when the element's own
 * `clientHeight` is unusable. A detached or not-yet-laid-out element reports
 * 0, and a 0-px page would silently swallow the event.
 */
export const NOMINAL_PAGE_HEIGHT_PX = 800;

/**
 * Magnitude cap (px) applied to CONVERTED (line/page) deltas only. Two
 * Chromium notches: at 200 px a wheel zoom step is `0.95^2` (9.75% of
 * distance) and an FOV step is 10 degrees, which is the largest single notch
 * we want to allow. It exists because a page-mode event on a tall canvas, or a
 * line-mode driver reporting an outlier line count, would otherwise produce
 * one absurd jump. Firefox's ordinary 3-line notch (48 px) is nowhere near it.
 */
export const MAX_NORMALIZED_DELTA_PX = 200;

/**
 * Convert a wheel event's `deltaY` into a pixel-mode-equivalent value.
 *
 * - `DOM_DELTA_PIXEL` (0): returned **verbatim and unclamped**. Chromium and
 *   WebKit behaviour is bit-identical to before this helper existed, so no
 *   sensitivity needed re-tuning and fast trackpad flings — which legitimately
 *   exceed {@link MAX_NORMALIZED_DELTA_PX} — still pass through whole. This
 *   invariant is why the clamp deliberately skips this mode.
 * - `DOM_DELTA_LINE` (1): scaled by {@link PIXELS_PER_LINE}, then
 *   magnitude-clamped.
 * - `DOM_DELTA_PAGE` (2): scaled by `element.clientHeight` when that is a
 *   positive finite number, else {@link NOMINAL_PAGE_HEIGHT_PX}, then
 *   magnitude-clamped.
 * - any other `deltaMode`: returned unchanged. A future spec mode is not
 *   something we can convert, and preserving today's behaviour is the
 *   least-surprise default.
 *
 * A non-finite `deltaY` (a driver or synthetic-event outlier) yields `0`: an
 * `Infinity` reaching the zoom scale or the roll accumulator would poison the
 * camera state permanently. `deltaY === 0` yields `0` in every mode.
 *
 * @param event - The wheel event to read `deltaY` / `deltaMode` from.
 * @param element - Element the wheel is over, used as the page height in
 *   `DOM_DELTA_PAGE`. Optional; omit it to always use the nominal height.
 * @returns A pixel-equivalent vertical delta, sign preserved.
 */
export function normalizeWheelDelta(event: WheelEvent, element?: HTMLElement | null): number {
  const { deltaY } = event;
  if (!Number.isFinite(deltaY)) return 0;
  if (deltaY === 0) return 0;

  let pixels: number;
  if (event.deltaMode === 1) {
    pixels = deltaY * PIXELS_PER_LINE;
  } else if (event.deltaMode === 2) {
    const clientHeight = element?.clientHeight;
    const pageHeight =
      typeof clientHeight === 'number' && Number.isFinite(clientHeight) && clientHeight > 0
        ? clientHeight
        : NOMINAL_PAGE_HEIGHT_PX;
    pixels = deltaY * pageHeight;
  } else {
    // Pixel mode (0) and any unknown mode: pass through untouched.
    return deltaY;
  }

  return clamp(pixels, -MAX_NORMALIZED_DELTA_PX, MAX_NORMALIZED_DELTA_PX);
}
