/**
 * Unit tests for the pure debug-console formatter helpers.
 */

import { describe, it, expect } from 'vitest';
import {
  formatArgs,
  formatConsoleTimestamp,
  messageMatchesFilter,
} from '../../../ui/debug-console/formatters';

describe('formatArgs', () => {
  it('returns empty string for empty arg list', () => {
    expect(formatArgs([])).toBe('');
  });

  it('joins primitive args with single spaces', () => {
    expect(formatArgs(['a', 1, true])).toBe('a 1 true');
  });

  it('writes "undefined" and "null" for those values', () => {
    expect(formatArgs([undefined])).toBe('undefined');
    expect(formatArgs([null])).toBe('null');
    expect(formatArgs([null, undefined, 'x'])).toBe('null undefined x');
  });

  it('passes strings through unquoted (vs the DOM formatter which quotes them)', () => {
    expect(formatArgs(['hello'])).toBe('hello');
  });

  it('pretty-prints objects with two-space indentation', () => {
    const out = formatArgs([{ a: 1, b: 'x' }]);
    expect(out).toContain('{\n  "a": 1,\n  "b": "x"\n}');
  });

  it('renders an Error as name: message, not {}', () => {
    // The headline regression: Error's name/message/stack are non-enumerable, so
    // JSON.stringify produced '{}' and every `log.*(…, error)` site lost its
    // message. The stringify `catch` never fired, because it SUCCEEDED at
    // producing that empty object.
    expect(formatArgs([new Error('boom')])).toBe('Error: boom');
  });

  it('reproduces the reported cache line with its cause intact', () => {
    // Verbatim shape of the user's bug report, which read
    // `OPFSStore metadata save failed {}`.
    const out = formatArgs([
      '[⚠️] [Cache] OPFSStore metadata save failed',
      new Error('A requested file or directory could not be found'),
    ]);
    expect(out).toContain('could not be found');
    expect(out).not.toContain('{}');
  });

  it('keeps the name of an Error subclass', () => {
    // A subclass assigning own enumerable fields (as LoaderError does with
    // name/cause) stringifies to a non-empty but message-LESS object, so it would
    // regress differently from a plain Error if the branch order were wrong.
    class Sub extends Error {
      constructor(message: string) {
        super(message);
        this.name = 'Sub';
      }
    }
    expect(formatArgs([new Sub('boom')])).toBe('Sub: boom');
  });

  it('renders an Error with no message as just the name', () => {
    expect(formatArgs([new Error()])).toBe('Error');
  });

  it('renders a cross-realm Error as name: message, not {}', () => {
    // An Error from another realm fails a same-realm `instanceof Error` but
    // still stringifies to {}. `isErrorLike` catches it structurally.
    const crossRealm = { [Symbol.toStringTag]: 'Error', name: 'TypeError', message: 'boom' };
    expect(crossRealm instanceof Error).toBe(false);
    expect(formatArgs([crossRealm])).toBe('TypeError: boom');
  });

  it('still renders an ordinary {name, message} object as JSON', () => {
    // Guard against isErrorLike false-positiving on legitimate data payloads.
    const out = formatArgs([{ name: 'Widget', message: 'hello' }]);
    expect(out).toContain('"name": "Widget"');
    expect(out).toContain('"message": "hello"');
  });

  it('falls back to String() when JSON.stringify throws (circular ref)', () => {
    const obj: { self?: object } = {};
    obj.self = obj;
    const out = formatArgs([obj]);
    // Circular fallback hits `String(arg)`, which is "[object Object]".
    expect(out).toBe('[object Object]');
  });

  it('handles a mix of types in one call', () => {
    const out = formatArgs(['msg', 42, false, null, { k: 'v' }]);
    expect(out.startsWith('msg 42 false null ')).toBe(true);
    expect(out).toContain('"k": "v"');
  });
});

describe('messageMatchesFilter', () => {
  it('returns true when filter is empty', () => {
    expect(messageMatchesFilter('anything here', '')).toBe(true);
  });

  it('returns true when filter is whitespace-only', () => {
    expect(messageMatchesFilter('anything here', '   ')).toBe(true);
    expect(messageMatchesFilter('anything here', '\t\n')).toBe(true);
  });

  it('case-insensitive substring match', () => {
    expect(messageMatchesFilter('Loading scene now', 'scene')).toBe(true);
    expect(messageMatchesFilter('Loading scene now', 'SCENE')).toBe(true);
    expect(messageMatchesFilter('LOADING SCENE NOW', 'scene')).toBe(true);
  });

  it('returns false on no match', () => {
    expect(messageMatchesFilter('hello world', 'goodbye')).toBe(false);
  });

  it('trims the filter before matching (extra padding by accident)', () => {
    expect(messageMatchesFilter('error: thing failed', '  error  ')).toBe(true);
  });
});

describe('formatConsoleTimestamp', () => {
  it('formats HH:mm:ss.SSS in 24-hour form with exactly 3 ms digits and either . or , separator', () => {
    // [ui.md/W10][P2] Previously the regex was permissive but the test
    // name promised "HH:mm:ss.SSS" strictness. Strengthen by pinning the
    // millisecond digit count to exactly 3, the locale-dependent decimal
    // separator to one of two known values, and the exact ms string '042'
    // (preserves leading zero). The hour:minute:second portion is locale-
    // sensitive depending on the runner's timezone; just verify it has
    // the right shape and is not the empty string.
    const date = new Date('2026-05-07T14:09:08.042Z');
    const result = formatConsoleTimestamp(date);
    expect(result.length).toBeGreaterThan(0);
    expect(result).toMatch(/^\d{2}:\d{2}:\d{2}[.,]\d{3}$/);
    // ms portion must be exactly '042' (3 digits, leading zero preserved).
    expect(result.slice(-4)).toMatch(/^[.,]042$/);
    // Decimal separator is one of '.' or ',' — no other punctuation.
    const sep = result.slice(-4, -3);
    expect(['.', ',']).toContain(sep);
  });

  it('preserves leading zeros for hours / minutes / seconds', () => {
    const date = new Date('2026-05-07T03:04:05.006Z');
    const result = formatConsoleTimestamp(date);
    expect(result).toMatch(/^\d{2}:\d{2}:\d{2}[.,]006$/);
  });

  it('different Date inputs produce different output strings', () => {
    const a = formatConsoleTimestamp(new Date('2026-05-07T10:00:00.000Z'));
    const b = formatConsoleTimestamp(new Date('2026-05-07T11:00:00.000Z'));
    expect(a).not.toBe(b);
  });
});
