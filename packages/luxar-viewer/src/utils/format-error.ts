/**
 * Turning an unknown thrown value into something readable.
 *
 * `catch (error)` gives `unknown`, and the two obvious things to do with it are
 * both wrong in the same place: `String(error)` on an `Error` yields
 * `"Error: msg"` but on a `DOMException` loses the name, and `JSON.stringify`
 * yields `"{}"` because `name` / `message` / `stack` are all non-enumerable.
 * That second one is not hypothetical — it is why a real bug report read
 * `OPFSStore metadata save failed {}` with the cause entirely absent.
 *
 * The idiom was inlined ~31 times and file-private in a 32nd
 * (`data/points/points-spatial-index-loader.ts`); this is that helper promoted
 * so the console formatters and any log site can share one definition.
 *
 * @module utils/format-error
 */

/**
 * The message from an unknown thrown value.
 *
 * `DOMException` (what the File System Access API throws) satisfies
 * `instanceof Error` per WebIDL, so one branch covers it.
 *
 * Never throws. `String(x)` raises `TypeError: Cannot convert object to
 * primitive value` for a null-prototype object, and a hostile `message`
 * accessor can throw too; this runs inside `catch` blocks and logging paths —
 * a helper whose job is "make this loggable" must not become the thing that
 * breaks the error handler. Same hazard the debug console's object branch
 * already guards.
 */
export function getErrorMessage(error: unknown): string {
  try {
    if (error instanceof Error) return error.message;
    return String(error);
  } catch {
    return '[unprintable error]';
  }
}

/**
 * One-line `name: message` rendering, the shape devtools shows.
 *
 * An empty message renders as just the name rather than leaving a dangling
 * colon, so `new Error()` reads `Error` and not `Error: `.
 *
 * Never throws — a throwing `name`/`message` accessor must not break the
 * console renderers, which previously guarded every property read via the
 * try/catch around `JSON.stringify`.
 */
export function formatErrorForDisplay(error: Error): string {
  try {
    const name = error.name || 'Error';
    return error.message ? `${name}: ${error.message}` : name;
  } catch {
    return '[unprintable error]';
  }
}

/**
 * The stack from an unknown thrown value, or `undefined` when there isn't one.
 *
 * The duck-typed branch is deliberate: some Firefox `DOMException`s carry a
 * `stack` without being reported as an `Error`, and a thrown plain object may
 * carry one too.
 *
 * Never throws. `stack` can be a throwing accessor and `'stack' in x` can hit
 * a throwing Proxy trap — and this runs inside the patched `console.warn` /
 * `console.error` BEFORE the original console call, so a throw here would
 * swallow the very diagnostic being logged and break the calling code.
 */
export function getErrorStack(error: unknown): string | undefined {
  try {
    if (error instanceof Error) return error.stack;
    if (typeof error === 'object' && error !== null && 'stack' in error) {
      const stack = (error as { stack?: unknown }).stack;
      return typeof stack === 'string' ? stack : undefined;
    }
    return undefined;
  } catch {
    return undefined;
  }
}
