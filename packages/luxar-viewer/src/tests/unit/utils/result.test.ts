/**
 * Tests for the Result<T, E> utility.
 */

import { describe, it, expect } from 'vitest';
import {
  ok,
  err,
  isOk,
  isErr,
  match,
  mapOk,
  mapErr,
  unwrap,
  unwrapOr,
  tryAsync,
  type Result,
} from '../../../utils/result';

describe('ok / err / isOk / isErr', () => {
  it('ok constructs a successful result', () => {
    const r = ok(42);
    expect(r.ok).toBe(true);
    expect((r as { value: number }).value).toBe(42);
  });

  it('err constructs a failed result', () => {
    const r = err<string>('boom');
    expect(r.ok).toBe(false);
    expect((r as { error: string }).error).toBe('boom');
  });

  it('isOk narrows correctly', () => {
    const r: Result<number, string> = ok(1);
    if (isOk(r)) {
      // TypeScript should now know r.value is number
      expect(r.value + 1).toBe(2);
    } else {
      throw new Error('should not branch here');
    }
  });

  it('isErr narrows correctly', () => {
    const r: Result<number, string> = err('nope');
    if (isErr(r)) {
      expect(r.error.toUpperCase()).toBe('NOPE');
    } else {
      throw new Error('should not branch here');
    }
  });
});

describe('match', () => {
  it('routes to ok handler on success', () => {
    expect(
      match(ok(7), {
        ok: (v) => v * 2,
        err: () => -1,
      })
    ).toBe(14);
  });

  it('routes to err handler on failure', () => {
    expect(
      match(err<string>('nope'), {
        ok: () => 'hi',
        err: (e) => `oops: ${e}`,
      })
    ).toBe('oops: nope');
  });
});

describe('mapOk / mapErr', () => {
  it('mapOk applies the function on success', () => {
    expect(mapOk(ok(3), (n) => n * 4)).toEqual(ok(12));
  });

  it('mapOk passes through errors unchanged', () => {
    const e: Result<number, string> = err('bad');
    expect(mapOk(e, (n) => n * 4)).toBe(e);
  });

  it('mapErr applies the function on error', () => {
    expect(mapErr(err<string>('bad'), (e) => e.toUpperCase())).toEqual(err('BAD'));
  });

  it('mapErr passes through successes unchanged', () => {
    const o: Result<number, string> = ok(5);
    expect(mapErr(o, () => 'X')).toBe(o);
  });
});

describe('unwrap / unwrapOr', () => {
  it('unwrap returns the success value', () => {
    expect(unwrap(ok('hi'))).toBe('hi');
  });

  it('unwrap throws on error', () => {
    expect(() => unwrap(err('boom'))).toThrow(/unwrap on Err: boom/);
  });

  it('unwrap stringifies non-string errors', () => {
    expect(() => unwrap(err({ code: 42 }))).toThrow(/code/);
  });

  it('unwrapOr returns the value on success', () => {
    expect(unwrapOr(ok(1), 99)).toBe(1);
  });

  it('unwrapOr returns the fallback on error', () => {
    expect(unwrapOr(err<string, number>('bad'), 99)).toBe(99);
  });
});

describe('tryAsync', () => {
  it('wraps a successful async function as ok', async () => {
    const r = await tryAsync(
      async () => 42,
      (_e) => 'mapped'
    );
    expect(r).toEqual(ok(42));
  });

  it('wraps a thrown error via the mapper', async () => {
    const r = await tryAsync(
      async () => {
        throw new Error('boom');
      },
      (e) => (e instanceof Error ? e.message : 'unknown')
    );
    expect(r).toEqual(err('boom'));
  });
});
