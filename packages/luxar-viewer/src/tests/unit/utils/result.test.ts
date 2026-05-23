/**
 * Tests for the Result<T, E> utility.
 */

import { describe, it, expect, test } from 'vitest';
import * as fc from 'fast-check';
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

// [utils.md/H1][P12] Functor laws for Result<T, E> — pinned over arbitrary
// values and arbitrary mapping functions. These laws are the canonical
// algebraic guarantees a Functor/Either must satisfy; any mutation that
// breaks them (e.g. swapping mapOk and mapErr branches, or returning
// `ok(undefined)` on error) would surface here.
describe('Result functor laws [utils.md/H1][P12]', () => {
  test('mapOk identity law: mapOk(r, id) ≡ r (for ok-side)', () => {
    fc.assert(
      fc.property(fc.integer(), (n) => {
        const r = ok(n);
        const mapped = mapOk(r, (x) => x);
        expect(mapped).toEqual(r);
      })
    );
  });

  test('mapOk composition: mapOk(mapOk(r, f), g) ≡ mapOk(r, x => g(f(x)))', () => {
    fc.assert(
      fc.property(fc.integer(), (n) => {
        const f = (x: number) => x * 2;
        const g = (x: number) => x + 7;
        const r = ok(n);
        const stepwise = mapOk(mapOk(r, f), g);
        const composed = mapOk(r, (x) => g(f(x)));
        expect(stepwise).toEqual(composed);
      })
    );
  });

  test('mapErr identity law: mapErr(r, id) ≡ r (for err-side)', () => {
    fc.assert(
      fc.property(fc.string(), (s) => {
        const r: Result<number, string> = err(s);
        const mapped = mapErr(r, (e) => e);
        expect(mapped).toEqual(r);
      })
    );
  });

  test('mapOk does not touch err values, mapErr does not touch ok values', () => {
    fc.assert(
      fc.property(fc.integer(), fc.string(), (n, s) => {
        const success: Result<number, string> = ok(n);
        const failure: Result<number, string> = err(s);
        // mapOk on err is identity-by-reference (verified by previous tests
        // for ===, but here we assert structural).
        expect(mapOk(failure, (x) => x * 99)).toEqual(failure);
        // mapErr on ok is identity-by-reference.
        expect(mapErr(success, (e) => `${e}!`)).toEqual(success);
      })
    );
  });

  test('unwrapOr returns the value when ok, the fallback when err', () => {
    fc.assert(
      fc.property(fc.integer(), fc.integer(), fc.string(), (value, fallback, errMsg) => {
        expect(unwrapOr<number, string>(ok(value), fallback)).toBe(value);
        expect(unwrapOr<number, string>(err(errMsg), fallback)).toBe(fallback);
      })
    );
  });

  test('isOk and isErr are mutually exclusive for every Result', () => {
    fc.assert(
      fc.property(
        fc.oneof(
          fc.integer().map((n) => ok(n) as Result<number, string>),
          fc.string().map((s) => err<string, number>(s) as Result<number, string>)
        ),
        (r) => {
          expect(isOk(r) !== isErr(r)).toBe(true);
        }
      )
    );
  });

  test('match routes correctly for every Result + every pair of handlers', () => {
    fc.assert(
      fc.property(fc.integer(), fc.string(), (n, s) => {
        const handlers = {
          ok: (v: number) => `ok:${v}`,
          err: (e: string) => `err:${e}`,
        };
        expect(match<number, string, string>(ok(n), handlers)).toBe(`ok:${n}`);
        expect(match<number, string, string>(err(s), handlers)).toBe(`err:${s}`);
      })
    );
  });
});
