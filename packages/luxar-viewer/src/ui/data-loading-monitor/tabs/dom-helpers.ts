/**
 * Small DOM helpers shared by the per-tab updater modules in
 * `ui/data-loading-monitor/tabs/`.
 *
 * The Monitor uses a `data-field="<key>"` selector pattern for
 * incremental DOM updates: each value cell on a tab carries a stable
 * key, the renderer paints the structure once, and the per-tick
 * updater patches the values in place by selector. These helpers
 * encapsulate the patch + color-class plumbing so the per-tab
 * updaters don't have to.
 */

/**
 * Update every element carrying the given `data-field` attribute,
 * scoped to `container`. A field key may appear more than once (the
 * Cache tab renders each value both in the full metric card AND in the
 * collapsed-header compact summary) — all copies get the same text.
 * Returns false if no matching element was found (the caller can use
 * the boolean to decide whether the DOM structure needs a full
 * rebuild).
 */
export function patchField(container: HTMLElement | null, field: string, text: string): boolean {
  const els = container?.querySelectorAll(`[data-field="${field}"]`);
  if (!els || els.length === 0) return false;
  els.forEach((el) => {
    el.textContent = text;
  });
  return true;
}

/**
 * Replace any existing `luxar-color--*` class on `el` with
 * `newColorClass`. Pass an empty string to clear all color classes
 * without adding a new one.
 */
export function updateColorClass(el: HTMLElement, newColorClass: string): void {
  const classes = el.className.split(' ').filter((c) => c && !c.startsWith('luxar-color--'));
  if (newColorClass) {
    classes.push(newColorClass);
  }
  el.className = classes.join(' ');
}

/**
 * Apply `updateColorClass` to EVERY element carrying the given
 * `data-field` attribute within `container`. Companion to `patchField`
 * for fields that render in more than one place (full metric card +
 * compact collapsed-header summary on the Cache tab).
 */
export function updateColorClassByField(
  container: HTMLElement | null,
  field: string,
  newColorClass: string
): void {
  container?.querySelectorAll(`[data-field="${field}"]`).forEach((el) => {
    updateColorClass(el as HTMLElement, newColorClass);
  });
}
