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
  it('formats HH:mm:ss.SSS in 24-hour form', () => {
    // 14:09:08 with 042ms — locale formatter uses comma as decimal in some
    // jsdom builds, so allow either '.' or ',' as the millisecond separator.
    const date = new Date('2026-05-07T14:09:08.042Z');
    const result = formatConsoleTimestamp(date);
    // shape: HH:mm:ss[.,]SSS
    expect(result).toMatch(/^\d{2}:\d{2}:\d{2}[.,]\d{3}$/);
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
