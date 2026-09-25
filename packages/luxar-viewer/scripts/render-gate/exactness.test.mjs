import { describe, expect, it } from 'vitest';

import {
  judge,
  quantile,
  scoreBlocks,
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
  });

  it('rejects mismatched sizes', () => {
    expect(() => scoreFloatBuffers(rgba(2, 0), rgba(3, 0))).toThrow();
  });
});

/** A W x H RGBA frame, every pixel at `value` in RGB. */
function frame(w, h, value) {
  const f = new Float32Array(w * h * 4);
  for (let i = 0; i < w * h; i++) f.fill(value, i * 4, i * 4 + 3);
  return f;
}

describe('scoreBlocks and the ULP class', () => {
  const W = 64;
  const H = 32;

  it('sees sub-pixel jitter as nothing and a systematic change as light', () => {
    const base = frame(W, H, 0.2);
    // Jitter: move light between two neighbouring pixels of every tile.
    const jitter = base.slice();
    for (let y = 0; y < H; y += 4) {
      for (let x = 0; x < W; x += 4) {
        const i = (y * W + x) * 4;
        jitter[i] += 0.05;
        jitter[i + 4] -= 0.05;
      }
    }
    const j = scoreBlocks(base, jitter, W, H);
    expect(Math.abs(j.energyRel)).toBeLessThan(1e-9);
    expect(j.blockMaxRel).toBeLessThan(1e-9);
    // Systematic: every pixel 1e-4 brighter.
    const bright = base.map((v, i) => (i % 4 === 3 ? v : v * (1 + 1e-4)));
    const b = scoreBlocks(base, bright, W, H);
    expect(b.energyRel).toBeCloseTo(1e-4, 7);
    expect(b.blockP999Rel).toBeCloseTo(1e-4, 7);
  });

  it('judges ULP on the calibrated energy and tile limits', () => {
    const base = frame(W, H, 0.2);
    const hdr = scoreFloatBuffers(base, base);
    const scaled = (eps) =>
      scoreBlocks(
        base,
        base.map((v, i) => (i % 4 === 3 ? v : v * (1 + eps))),
        W,
        H
      );
    expect(judge('ULP', hdr, null, null, scaled(2e-6)).pass).toBe(true); // rounding level
    const v = judge('ULP', hdr, null, null, scaled(1e-4)); // a real error
    expect(v.pass).toBe(false);
    // A UNIFORM 1e-4 brightening moves every tile by exactly 1e-4 of the peak
    // tile, under the tile limit: energy is what catches it (the tile rule
    // catches local errors, next test).
    expect(v.failures.join()).toMatch(/energy/);
  });

  it('catches a local error that leaves the frame energy flat', () => {
    // Light moved from one region to another: energy unchanged, tiles not.
    const base = frame(W, H, 0.2);
    const moved = base.slice();
    for (let i = 0; i < (W * H) / 2; i++) moved[i * 4] *= 1.001;
    for (let i = (W * H) / 2; i < W * H; i++) moved[i * 4] *= 0.999;
    const s = scoreBlocks(base, moved, W, H);
    expect(Math.abs(s.energyRel)).toBeLessThan(1e-9);
    const v = judge('ULP', scoreFloatBuffers(base, moved), null, null, s);
    expect(v.pass).toBe(false);
    expect(v.failures.join()).toMatch(/tile p99.9/);
  });

  it('requires the block score for the ULP class', () => {
    const base = frame(4, 4, 0.1);
    expect(() => judge('ULP', scoreFloatBuffers(base, base), null, null)).toThrow();
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

  it('judges ULP on node identity and IDENTICAL on everything', () => {
    // 1000 pixels of node 2; the candidate changes the ELEMENT (G) at half of
    // them (a tie flip among overlapping elements) and the NODE (R) at 5.
    const n = 1000;
    const a = rgba(n, 0);
    for (let p = 0; p < n; p++) {
      a[p * 4] = 2;
      a[p * 4 + 1] = p % 7;
    }
    const b = a.slice();
    for (let p = 0; p < n; p += 2) b[p * 4 + 1] += 1;
    const clean = scoreFloatBuffers(rgba(10, 0.1), rgba(10, 0.1));
    const noLight = scoreBlocks(rgba(16, 0.1), rgba(16, 0.1), 4, 4);
    const elementOnly = scorePickBuffers(a, b);
    expect(elementOnly.nodeMismatches).toBe(0);
    expect(elementOnly.mismatches).toBe(500);
    expect(judge('ULP', clean, null, elementOnly, noLight).pass).toBe(true);
    expect(judge('IDENTICAL', clean, null, elementOnly).pass).toBe(false);
    for (let p = 1; p < 11; p += 2) b[p * 4] = 3; // 5 node changes = 5e-3
    const nodeChanged = scorePickBuffers(a, b);
    expect(nodeChanged.nodeMismatches).toBe(5);
    const v = judge('ULP', clean, null, nodeChanged, noLight);
    expect(v.pass).toBe(false);
    expect(v.failures.join()).toMatch(/pick node mismatches/);
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
