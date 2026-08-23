/**
 * Predicates that run INSIDE the browser page, via `page.evaluate()`.
 *
 * Playwright serializes the function source and evaluates it in the page, so
 * everything here must be self-contained: no imports, no closure variables,
 * no references to module scope. That is also why these are duplicates of
 * production helpers rather than imports of them — and why they get a parity
 * unit test (`src/tests/unit/tests/e2e-page-predicates.test.ts`) that asserts
 * they still agree with the production originals.
 *
 * @module tests/e2e/page-predicates
 */

/**
 * `true` when keyboard focus is on a "typing surface" in the page.
 *
 * Mirrors `input/input-handler/commands/focus-utils.ts::isTypingInInput`, the
 * exact predicate `InputHandler.onKeyDown` guards on — it drops every key but
 * Escape while this is true. A modal panel that leaves focus on a text field
 * therefore makes its own toggle key one-way (issue #1922), so panel toggle
 * tests assert this is `false`.
 */
export function isTypingSurfaceInPage(): boolean {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName.toLowerCase();
  if (tag === 'input') {
    const type = (el as HTMLInputElement).type?.toLowerCase();
    return type !== 'range' && type !== 'checkbox' && type !== 'radio';
  }
  return tag === 'textarea' || tag === 'select' || el.getAttribute('contenteditable') === 'true';
}
