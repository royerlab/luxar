/**
 * Pure DOM-focus helpers extracted from input-handler.ts so they're
 * unit-testable against a real DOM (no mocks needed) and can be reused
 * by other input components.
 *
 * Both helpers operate on a passed-in `Element | null` rather than
 * reading `document.activeElement` directly, so callers control the
 * DOM context and tests don't have to programmatically focus elements
 * to assert behaviour.
 *
 * @module utils/dom/focus
 */

/**
 * `true` when keyboard focus is on a "typing surface" — an editable
 * field, textarea, select, or contenteditable element. Used to suppress
 * navigation shortcuts (e.g. `[`/`]`, number keys) so the user can type
 * into a search box without flipping dimensions.
 *
 * Intentionally returns `false` for `<input type="range|checkbox|radio">`
 * — those don't capture text input, so navigation keys should still fire
 * even if focus has landed on them after a click.
 *
 * Canonical typing-detection helper. Both
 * `InputHandler.isTypingInInput` and
 * `InputContextManager.isTypingContext` delegate here so the
 * two paths cannot diverge — a fix to this function propagates to
 * every typing-aware code path.
 *
 * @param activeElement - Result of `document.activeElement`, or any
 *   `Element | null` you want to classify.
 */
export function isTypingInInput(activeElement: Element | null): boolean {
  if (!activeElement) return false;

  const tagName = activeElement.tagName.toLowerCase();

  if (tagName === 'input') {
    const inputType = (activeElement as HTMLInputElement).type?.toLowerCase();
    if (inputType === 'range' || inputType === 'checkbox' || inputType === 'radio') {
      return false;
    }
    return true;
  }

  return (
    tagName === 'textarea' ||
    tagName === 'select' ||
    activeElement.getAttribute('contenteditable') === 'true'
  );
}

/**
 * `true` when focus is on either the document body or the scene canvas,
 * i.e. the user is "interacting with the scene" rather than a UI panel.
 * Used to gate global keys like Space (fullscreen toggle) so they don't
 * fire when the user has clicked into a panel button.
 *
 * Body-or-canvas instead of just canvas because most pages start with
 * focus on body and never put it on the canvas explicitly.
 *
 * @param activeElement - Currently focused element, typically
 *   `document.activeElement`.
 * @param canvas - The scene's WebGL canvas (renderer.domElement).
 */
export function isFocusOnSceneCanvas(
  activeElement: Element | null,
  canvas: Element | null
): boolean {
  if (!activeElement) return false;
  if (typeof document !== 'undefined' && activeElement === document.body) return true;
  return canvas !== null && activeElement === canvas;
}
