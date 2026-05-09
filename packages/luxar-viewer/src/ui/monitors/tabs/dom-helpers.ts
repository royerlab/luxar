/**
 * Phase 19A — small DOM helpers shared by the per-tab updater
 * modules in `ui/monitors/tabs/`.
 *
 * The Monitor uses a `data-field="<key>"` selector pattern for
 * incremental DOM updates: each value cell on a tab carries a stable
 * key, the renderer paints the structure once, and the per-tick
 * updater patches the values in place by selector. These helpers
 * encapsulate the patch + color-class plumbing so the per-tab
 * updaters don't have to.
 */

/**
 * Update a single element's textContent by `data-field` attribute,
 * scoped to `container`. Returns false if no matching element was
 * found (the caller can use the boolean to decide whether the DOM
 * structure needs a full rebuild).
 */
export function patchField(
  container: HTMLElement | null,
  field: string,
  text: string
): boolean {
  const el = container?.querySelector(`[data-field="${field}"]`);
  if (!el) return false;
  el.textContent = text;
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
