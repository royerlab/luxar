/**
 * Small DOM helpers shared by the control-rail class and its overlay.
 *
 * @module ui/control-rail/dom-helpers
 */

/** A panel counts as "open" when its root element is present and rendered. */
export function isPanelVisible(selector: string, scope: ParentNode): boolean {
  const el = scope.querySelector(selector) ?? document.querySelector(selector);
  if (!(el instanceof HTMLElement)) return false;
  if (el.getClientRects().length === 0) return false;
  const s = getComputedStyle(el);
  return s.visibility !== 'hidden' && Number(s.opacity) > 0.01;
}

/**
 * Escape a string for safe interpolation into tooltip `innerHTML`.
 *
 * Re-exports the canonical {@link escapeHtml} (escapes `&<>"'` — the
 * single-quote superset), which is safe in the text/`<kbd>` interpolation
 * positions used by `index.ts` and `rail-overlay.ts`.
 */
export { escapeHtml } from '../../utils/escape-html';
