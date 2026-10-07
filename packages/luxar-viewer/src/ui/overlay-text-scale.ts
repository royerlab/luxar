/**
 * Overlay text scale: one multiplier on the type of every screen-space overlay.
 *
 * It lives in a CSS custom property on the viewer container rather than in each
 * overlay's config, so it reaches authored HTML as well as text overlays, and so
 * changing it is one property write instead of a rebuild. Only font sizes read
 * it: overlay positions, widths, anchors and paddings stay as authored, which
 * is the point — the same layout, read from further away or closer up.
 *
 * Text overlays pick it up through {@link scaledFontSize}, unless authored with
 * `scale_text: false` (a title that belongs to the layout). Authored HTML opts in
 * by writing its sizes as `calc(1.3vh * var(--luxar-text-scale, 1))`; the
 * fallback keeps that HTML correct in any page that never sets the property.
 */
import { getViewerContainer } from '../utils/viewer-container';

/** The custom property every scaled font size multiplies by. */
export const TEXT_SCALE_PROPERTY = '--luxar-text-scale';

/** Accepted range; anything outside is clamped, a non-number means 1. */
export const TEXT_SCALE_RANGE = Object.freeze({ min: 0.25, max: 4 });

/**
 * The session's text scale: `?textScale=` beats the scene's authored
 * `viewer_config.text_scale`, and neither means 1.
 */
export function resolveTextScale(urlScale: number | null | undefined, authored: unknown): number {
  const raw = urlScale ?? authored;
  if (typeof raw !== 'number' || !Number.isFinite(raw) || raw <= 0) return 1;
  return Math.min(TEXT_SCALE_RANGE.max, Math.max(TEXT_SCALE_RANGE.min, raw));
}

/** Write the scale onto the viewer container, where every overlay inherits it. */
export function applyTextScale(scale: number, container: HTMLElement = getViewerContainer()): void {
  // A styleless container (an embedder's stand-in, a test double) has nowhere
  // to hold the property; its overlays simply keep their authored size.
  const style = container.style as CSSStyleDeclaration | undefined;
  if (!style) return;
  if (scale === 1) style.removeProperty(TEXT_SCALE_PROPERTY);
  else style.setProperty(TEXT_SCALE_PROPERTY, String(scale));
}

/** The scale an element currently inherits (1 when none is set). */
export function textScaleOf(el: Element): number {
  const value = Number.parseFloat(getComputedStyle(el).getPropertyValue(TEXT_SCALE_PROPERTY));
  return Number.isFinite(value) && value > 0 ? value : 1;
}

/** CSS font size for a viewport-height fraction, scaled by the session's text scale. */
export function scaledFontSize(fraction: number): string {
  return `calc(${fraction * 100}vh * var(${TEXT_SCALE_PROPERTY}, 1))`;
}
