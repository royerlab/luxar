/**
 * Cross-browser wheel-delta normalization.
 *
 * `WheelEvent.deltaY` is only meaningful together with `deltaMode`, and the
 * viewer used to read it as if it were always pixels. One physical notch of
 * the same mouse therefore differed by ~32x between browsers: Chromium and
 * WebKit report `DOM_DELTA_PIXEL` with `deltaY = 100`, Firefox reports
 * `DOM_DELTA_LINE` with `deltaY = 3`. Wheel zoom, Shift+wheel roll and
 * Ctrl+wheel FOV all consumed that raw number, so on Firefox each moved about
 * a thirtieth as far per notch — wheel zoom felt essentially dead.
 *
 * This module converts a wheel event's selected axis delta into a
 * pixel-mode-equivalent number so every wheel entry point can keep doing the
 * arithmetic it already does, on a browser-independent input. Most callers
 * select `deltaY`; Shift+wheel roll may explicitly fall back to `deltaX`.
 *
 * @module utils/wheel-delta
 */

import { clamp } from './clamp';

/**
 * Pixels one `DOM_DELTA_LINE` unit stands for. This is the factor THREE.js
 * OrbitControls applies for `DOM_DELTA_LINE`
 * (`three/examples/jsm/controls/OrbitControls.js`, `newEvent.deltaY *= 16`),
 * and our zoom math is vendored from that same control
 * (`controls/luxar-orbit-controls/math/zoom.ts`), so we stay consistent with
 * it. It is NOT the browsers' own step: Blink and WebKit use a
 * `pixelsPerLineStep` of 40, as does Facebook's `normalize-wheel`
 * (`LINE_HEIGHT = 40`, `PAGE_HEIGHT = 800` — the source of our nominal page
 * height). Consequence, deliberately accepted: Firefox's 3-line notch becomes
 * 48 px against Chromium's 100 px, closing the gap from ~32x to ~2x rather
 * than reaching parity. 40 would in fact land closer in absolute terms
 * (3 × 40 = 120 px, 1.2x over, against 48 px at 2.08x under); we keep 16 so
 * the zoom math and the delta feeding it come from the same vendored source.
 */
export const PIXELS_PER_LINE = 16;

/**
 * Fallback page height (px) for `DOM_DELTA_PAGE` when the element's own
 * `clientHeight` is unusable. A detached or not-yet-laid-out element reports
 * 0, and a 0-px page would silently swallow the event.
 */
export const NOMINAL_PAGE_HEIGHT_PX = 800;

/**
 * Magnitude cap (px) applied to CONVERTED (line/page) deltas only, because a
 * page-mode event on a tall canvas or a line-mode driver reporting an outlier
 * line count would otherwise produce one absurd jump. Two Chromium notches: at
 * 200 px a wheel zoom step is `0.95^2` (9.75% of distance) and an FOV step is
 * 10 degrees — the largest single notch worth allowing. Firefox's ordinary
 * 3-line notch (48 px) is nowhere near it.
 */
export const MAX_NORMALIZED_DELTA_PX = 200;

function normalizeDelta(delta: number, deltaMode: number, element?: HTMLElement | null): number {
  if (!Number.isFinite(delta)) return 0;
  if (delta === 0) return 0;

  let pixels: number;
  if (deltaMode === 1) {
    pixels = delta * PIXELS_PER_LINE;
  } else if (deltaMode === 2) {
    const clientHeight = element?.clientHeight;
    const pageHeight =
      typeof clientHeight === 'number' && Number.isFinite(clientHeight) && clientHeight > 0
        ? clientHeight
        : NOMINAL_PAGE_HEIGHT_PX;
    pixels = delta * pageHeight;
  } else {
    // Pixel mode (0) and any unknown mode: pass through untouched.
    return delta;
  }

  return clamp(pixels, -MAX_NORMALIZED_DELTA_PX, MAX_NORMALIZED_DELTA_PX);
}

/**
 * Convert a wheel event's `deltaY` into a pixel-mode-equivalent value.
 *
 * - `DOM_DELTA_PIXEL` (0): returned **verbatim and unclamped**, so Chromium
 *   and WebKit behaviour is unchanged — no sensitivity needed re-tuning, and a
 *   fast trackpad fling (which legitimately exceeds
 *   {@link MAX_NORMALIZED_DELTA_PX}) still passes through whole. The one
 *   exception is a non-finite `deltaY` (below), which is caught in every mode.
 * - `DOM_DELTA_LINE` (1): scaled by {@link PIXELS_PER_LINE}, then
 *   magnitude-clamped.
 * - `DOM_DELTA_PAGE` (2): scaled by `element.clientHeight` when that is a
 *   positive finite number, else {@link NOMINAL_PAGE_HEIGHT_PX}, then
 *   magnitude-clamped. On any realistic canvas (`clientHeight >= 200`) a
 *   whole-page notch — `deltaY` of ±1, which is what "scroll one screen at a
 *   time" emits — saturates the cap whatever the height, so the height only
 *   shows through for a fractional page.
 * - any other `deltaMode`: returned unchanged. A future spec mode is not
 *   something we can convert, and preserving today's behaviour is the
 *   least-surprise default.
 *
 * A non-finite `deltaY` (a driver or synthetic-event outlier) yields `0` in
 * every mode, pixel included: pre-fix, a `+Infinity` delta made
 * `computeZoomScale` return `Math.pow(0.95, Infinity) === 0` and so jumped the
 * camera 100% of its distance in one event.
 *
 * A zero delta of either sign returns `+0`. That is contract hygiene, not a
 * consumer requirement — consumers either branch on `< 0` / `> 0` or
 * accumulate, and none can tell `+0` from `-0`.
 *
 * @param event - The wheel event to read `deltaY` / `deltaMode` from.
 * @param element - Element the wheel is over, used as the page height in
 *   `DOM_DELTA_PAGE`. Optional; omit it to always use the nominal height.
 * @returns A pixel-equivalent vertical delta, sign preserved.
 */
export function normalizeWheelDelta(event: WheelEvent, element?: HTMLElement | null): number {
  return normalizeDelta(event.deltaY, event.deltaMode, element);
}

/**
 * Normalize the axis carrying a Shift+wheel gesture.
 *
 * Some browsers move a standard mouse wheel's delta from `deltaY` to
 * `deltaX` while Shift is held. Prefer `deltaY` whenever it is non-zero, then
 * fall back to `deltaX`; this preserves genuine two-axis events while making
 * the swapped-axis gesture usable. Callers must opt in deliberately: ordinary
 * horizontal trackpad scrolling must not become zoom, FOV, or forward motion.
 *
 * @param event - The wheel event whose carrying axis should be normalized.
 * @param element - Optional page-height source, as in {@link normalizeWheelDelta}.
 * @returns A pixel-equivalent delta from `deltaY`, or `deltaX` when `deltaY` is zero.
 */
export function normalizeWheelDeltaWithAxisFallback(
  event: WheelEvent,
  element?: HTMLElement | null
): number {
  const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
  return normalizeDelta(delta, event.deltaMode, element);
}
