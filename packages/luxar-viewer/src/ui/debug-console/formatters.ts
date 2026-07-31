/**
 * Pure formatters for the debug console.
 *
 * Extracted from debug-console.ts so the arg-formatting logic, the
 * filter matcher, and the timestamp formatter are unit-testable
 * without instantiating the panel's DOM.
 *
 * @module ui/debug-console/formatters
 */

import { formatErrorForDisplay, isErrorLike } from '../../utils/format-error';

/**
 * Convert an arbitrary console-arg list (strings, numbers, booleans,
 * objects, null, undefined) into a single space-separated display
 * string. Used both for the user-visible row content and as the
 * filter haystack — keeping the same output for both means the
 * "filter by visible text" promise holds.
 *
 * Objects are pretty-printed with `JSON.stringify(_, null, 2)`. If
 * stringify throws (e.g. a circular reference), falls back to
 * `String(arg)`.
 *
 * `Error` (and `DOMException`) is checked FIRST, before the object branch:
 * `name` / `message` / `stack` are non-enumerable, so `JSON.stringify(err)` is
 * `"{}"` — and the `catch → String(arg)` fallback never fires, because
 * stringify succeeds at producing that empty object. Every `log.*(…, error)`
 * call site therefore lost its message here, which is exactly what made a real
 * bug report read `OPFSStore metadata save failed {}`. Ordering matters beyond
 * plain Errors too: a subclass that assigns own enumerable fields (`LoaderError`
 * sets `name`/`cause`) stringifies to a non-empty but still message-LESS object.
 *
 * KEEP IN LOCKSTEP with `debug-console.ts::formatArgAsDOMElement`, which renders
 * the same values into DOM. The two must agree for Errors or the visible row and
 * the copied text disagree. (They differ for plain strings by design — the DOM
 * renderer quotes them.)
 */
export function formatArgs(args: readonly unknown[]): string {
  return args
    .map((arg) => {
      if (arg === undefined) return 'undefined';
      if (arg === null) return 'null';
      if (typeof arg === 'string') return arg;
      if (typeof arg === 'number') return arg.toString();
      if (typeof arg === 'boolean') return arg.toString();
      // `isErrorLike` also matches a cross-realm Error, which fails
      // `instanceof Error` yet still stringifies to `{}`.
      if (isErrorLike(arg)) return formatErrorForDisplay(arg as Error);
      if (typeof arg === 'object') {
        try {
          return JSON.stringify(arg, null, 2);
        } catch {
          return String(arg);
        }
      }
      return String(arg);
    })
    .join(' ');
}

/**
 * Case-insensitive substring matcher used by the console's filter input.
 *
 * Returns `true` when:
 *   - `filter` is empty or whitespace-only (no filter active), OR
 *   - the formatted text contains the filter string (case-insensitive).
 *
 * Both arguments are normalised with `.toLowerCase()` before comparison
 * so a user typing `ERROR` matches a row containing `error`.
 */
export function messageMatchesFilter(formatted: string, filter: string): boolean {
  if (!filter) return true;
  const trimmed = filter.trim();
  if (trimmed.length === 0) return true;
  return formatted.toLowerCase().includes(trimmed.toLowerCase());
}

/**
 * 24-hour HH:mm:ss.SSS timestamp matching the en-US locale formatter
 * the panel was using inline. The Date instance becomes the input so
 * tests are deterministic — caller-side `new Date()` is the only
 * source of nondeterminism.
 */
export function formatConsoleTimestamp(date: Date): string {
  return date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    fractionalSecondDigits: 3,
  });
}
