import { describe, expect, it } from 'vitest';

import {
  judge,
  quantile,
  scoreFloatBuffers,
  scorePickBuffers,
  ulp16,
  ulpDistance,
} from './exactness.mjs';

/** Round a float to the nearest half-float, as a HalfFloat render target stores it. */
function toHalf(value) {
  const buf = new Float32Array([value]);
  // Math.f16round is Node 24+; fall back to exact ULP16 rounding otherwise.
  if (typeof Math.f16round === 'function') return Math.f16round(buf[0]);
  const u = ulp16(value);
  return Math.round(value / u) * u;
}

function rgba(pixels, value) {
  return new Float32Array(pixels * 4).fill(value);
}

describe('ulp16', () => {
  it('is 2^-10 of the binade for normal values', () => {
    expect(ulp16(1)).toBe(2 ** -10);
    expect(ulp16(1.99)).toBe(2 ** -10);
    expect(ulp16(2)).toBe(2 ** -9);
    expect(ulp16(-0.5)).toBe(2 ** -11);
  });

  it('bottoms out at the subnormal step', () => {
    expect(ulp16(0)).toBe(2 ** -24);
    expect(ulp16(1e-7)).toBe(2 ** -24);
  });

  it('matches the spacing of adjacent half-floats (calibration)', () => {
    // The scorer must see one half-float step as exactly 1 ULP16, or its
    // thresholds mean nothing: take a value, step to the next half-float.
    for (const v of [0.001, 0.37, 1, 3.3, 250]) {
      const h = toHalf(v);
      const next = h + ulp16(h);
      expect(ulpDistance(h, next)).toBeCloseTo(1, 12);
      expect(toHalf(next)).toBe(next);
    }
  });
});

describe('ulpDistance', () => {
  it('treats NaN pairs as equal and a lone NaN as infinite', () => {
    expect(ulpDistance(NaN, NaN)).toBe(0);
    expect(ulpDistance(NaN, 1)).toBe(Infinity);
    expect(ulpDistance(Infinity, 1)).toBe(Infinity);
    expect(ulpDistance(Infinity, Infinity)).toBe(0);
  });
});

describe('quantile', () => {
  it('uses nearest rank', () => {
    expect(quantile([5, 1, 3, 2, 4], 0.5)).toBe(3);
    expect(quantile([5, 1, 3, 2, 4], 1)).toBe(5);
    expect(quantile([], 0.5)).toBe(0);
  });
});

describe('scoreFloatBuffers', () => {
  it('reports identical buffers as identical', () => {
    const a = rgba(100, 0.5);
    const s = scoreFloatBuffers(a, a.slice());
    expect(s.differing).toBe(0);
    expect(s.maxUlp).toBe(0);
    expect(judge('IDENTICAL', s, s, null).pass).toBe(true);
  });

  it('classifies a one-step change everywhere as drift, not flips', () => {
    const a = rgba(1000, 0.5);
    const b = a.map((v) => v + ulp16(v));
    const s = scoreFloatBuffers(a, b);
    expect(s.differing).toBe(1000);
    expect(s.flips).toBe(0);
    expect(s.maxDriftUlp).toBeCloseTo(1, 12);
    expect(judge('IDENTICAL', s, s, null).pass).toBe(false);
    expect(judge('ULP', s, s, null).pass).toBe(true);
  });

  it('counts an isolated large jump as a flip with its size relative to peak', () => {
    const a = rgba(200_000, 0.1);
    a[0] = 2; // peak
    const b = a.slice();
    b[4 * 7] = 0.6; // one pixel jumps by 0.5 (a discard flip)
    const s = scoreFloatBuffers(a, b);
    expect(s.flips).toBe(1);
    expect(s.maxFlipRel).toBeCloseTo(0.25, 6);
    expect(s.maxDriftUlp).toBe(0);
    expect(judge('ULP', s, null, null).pass).toBe(true); // 5e-6 < 3e-3
  });

  /** `count` pixels of a 10k frame jump by `ulps` ULP16. */
  function withJumps(count, ulps) {
    const a = rgba(10_000, 0.1);
    const b = a.slice();
    for (let i = 0; i < count; i++) b[i * 4] = 0.1 + ulps * ulp16(0.1);
    return scoreFloatBuffers(a, b);
  }

  it('accepts HDR flips up to the calibrated 3e-3 budget and fails above it', () => {
    // Small jumps (3 ULP16) so p99.99 stays under its limit and only the flip
    // fraction decides.
    expect(judge('ULP', withJumps(30, 3), null, null).pass).toBe(true); // 3e-3
    const over = judge('ULP', withJumps(31, 3), null, null);
    expect(over.pass).toBe(false);
    expect(over.failures.join()).toMatch(/hdr flips/);
  });

  it('fails the ULP class on a large p99.99 even within the flip budget', () => {
    // 1e-3 of pixels 100 ULP16 off: flips pass (1e-3 < 3e-3), p99.99 does not.
    const s = withJumps(10, 100);
    expect(s.flipFraction).toBeCloseTo(1e-3, 12);
    const v = judge('ULP', s, null, null);
    expect(v.pass).toBe(false);
    expect(v.failures.join()).toMatch(/hdr p99.99/);
  });

  it('applies the looser LDR guard to the LDR buffer', () => {
    const ldr = withJumps(50, 3); // 5e-3: over the HDR budget, under the LDR one
    const clean = scoreFloatBuffers(rgba(10, 0.1), rgba(10, 0.1));
    expect(judge('ULP', clean, ldr, null).pass).toBe(true);
    expect(judge('ULP', clean, withJumps(101, 3), null).pass).toBe(false); // 1.01e-2
  });

  it('rejects mismatched sizes', () => {
    expect(() => scoreFloatBuffers(rgba(2, 0), rgba(3, 0))).toThrow();
  });
});

describe('scorePickBuffers', () => {
  it('counts any id difference as a mismatch', () => {
    const a = rgba(4, 0);
    const b = a.slice();
    b[5] = 1e-7;
    const s = scorePickBuffers(a, b);
    expect(s.mismatches).toBe(1);
    expect(s.mismatchFraction).toBe(0.25);
    expect(judge('IDENTICAL', scoreFloatBuffers(a, a), null, s).pass).toBe(false);
    expect(s.hits).toBe(0); // the id is in the candidate; hits count the baseline
  });

  it('counts baseline pixels carrying an id as hits', () => {
    const a = rgba(4, 0);
    a[0] = 7; // pixel 0 holds an id
    a[9] = 3; // pixel 2 holds an id
    const s = scorePickBuffers(a, a.slice());
    expect(s.hits).toBe(2);
    expect(s.mismatches).toBe(0);
  });
});
