/**
 * Unit tests for the generic per-channel dequantizers
 * (`ArrayDecoder.makePerChannelDequant`) — the viewer inverse of the Python
 * `log_perchannel_*` / `signed_log_perchannel_*` encoders. These are geometry-
 * agnostic; the gsplat loader is just the first consumer.
 */

import { describe, it, expect } from 'vitest';
import { ArrayDecoder } from '../../../../data/array-decoder/decoder';

// Forward quantizers mirroring the Python encoder, so the test exercises a true
// encode→decode round-trip rather than re-asserting the decode formula.
function quantLog(values: number[][], bits: number) {
  const cols = values[0].length;
  const levels = (1 << bits) - 1;
  const y = values.map((row) => row.map((x) => Math.log1p(Math.max(x, 0))));
  const lo = Array.from({ length: cols }, (_, c) => Math.min(...y.map((r) => r[c])));
  const hi = Array.from({ length: cols }, (_, c) => Math.max(...y.map((r) => r[c])));
  const u = y.map((row) =>
    row.map((v, c) => Math.round(((v - lo[c]) / Math.max(hi[c] - lo[c], 1e-30)) * levels))
  );
  return { u, lo, hi };
}

function quantSignedLog(values: number[][], bits: number) {
  const cols = values[0].length;
  const levels = (1 << bits) - 1;
  const y = values.map((row) => row.map((x) => Math.sign(x) * Math.log1p(Math.abs(x))));
  const lo = Array.from({ length: cols }, (_, c) => Math.min(...y.map((r) => r[c])));
  const hi = Array.from({ length: cols }, (_, c) => Math.max(...y.map((r) => r[c])));
  const u = y.map((row) =>
    row.map((v, c) => Math.round(((v - lo[c]) / Math.max(hi[c] - lo[c], 1e-30)) * levels))
  );
  return { u, lo, hi };
}

function quantLinear(values: number[][], bits: number) {
  const cols = values[0].length;
  const levels = (1 << bits) - 1;
  const lo = Array.from({ length: cols }, (_, c) => Math.min(...values.map((r) => r[c])));
  const hi = Array.from({ length: cols }, (_, c) => Math.max(...values.map((r) => r[c])));
  const u = values.map((row) =>
    row.map((v, c) => Math.round(((v - lo[c]) / Math.max(hi[c] - lo[c], 1e-30)) * levels))
  );
  return { u, lo, hi };
}

describe('makePerChannelDequant', () => {
  it('linear: round-trips a signed per-axis coordinate array (uint16), sub-unit', () => {
    // COORDINATE positions: identity transform, per-axis fixed-point, negatives OK.
    const vals = [
      [-1200.5, 3.0, 1990.25],
      [400.0, -50.0, 0.0],
      [-3.0, 1800.0, 42.5],
      [393.9, 119.0, 37.0],
    ];
    const { u, lo, hi } = quantLinear(vals, 16);
    const deq = ArrayDecoder.makePerChannelDequant(
      { name: 'linear_perchannel_u16', bits: 16, col_lo: lo, col_hi: hi },
      3
    );
    for (let i = 0; i < vals.length; i++) {
      for (let c = 0; c < 3; c++) {
        const got = deq(u[i][c], c);
        const extent = hi[c] - lo[c];
        expect(Math.abs(got - vals[i][c])).toBeLessThan(extent / 65535 + 1e-6); // sub-unit
        if (Math.abs(vals[i][c]) > 1) expect(Math.sign(got)).toBe(Math.sign(vals[i][c]));
      }
    }
  });

  it('log: round-trips a positive per-channel array (uint8)', () => {
    const vals = [
      [0.5, 12.0],
      [1.0, 3.0],
      [8.0, 0.6],
      [2.5, 18.0],
    ];
    const { u, lo, hi } = quantLog(vals, 8);
    const deq = ArrayDecoder.makePerChannelDequant(
      { name: 'log_perchannel_u8', bits: 8, col_lo: lo, col_hi: hi },
      2
    );
    for (let i = 0; i < vals.length; i++) {
      for (let c = 0; c < 2; c++) {
        const got = deq(u[i][c], c);
        expect(Math.abs(got - vals[i][c])).toBeLessThan(0.2); // u8 over [0.5,18]
      }
    }
  });

  it('signed-log: round-trips a signed per-channel array and preserves sign (uint16)', () => {
    const vals = [
      [-0.3, 5.0],
      [0.1, -2.0],
      [4.0, 0.0],
      [-5.0, 0.05],
    ];
    const { u, lo, hi } = quantSignedLog(vals, 16);
    const deq = ArrayDecoder.makePerChannelDequant(
      { name: 'signed_log_perchannel_u16', bits: 16, col_lo: lo, col_hi: hi },
      2
    );
    for (let i = 0; i < vals.length; i++) {
      for (let c = 0; c < 2; c++) {
        const got = deq(u[i][c], c);
        expect(Math.abs(got - vals[i][c])).toBeLessThan(1e-2);
        if (Math.abs(vals[i][c]) > 1e-3) expect(Math.sign(got)).toBe(Math.sign(vals[i][c]));
      }
    }
  });

  it('uses per-channel scales (a coarse channel does not blunt a fine one)', () => {
    const vals = [
      [0.5, 5.0],
      [0.6, 50.0],
      [0.55, 25.0],
    ];
    const { u, lo, hi } = quantLog(vals, 8);
    expect(hi[1]).toBeGreaterThan(hi[0]); // channel 1 has the larger log-range
    const deq = ArrayDecoder.makePerChannelDequant(
      { name: 'log_perchannel_u8', bits: 8, col_lo: lo, col_hi: hi },
      2
    );
    // the fine channel 0 stays precise despite channel 1's much larger range
    expect(Math.abs(deq(u[0][0], 0) - 0.5)).toBeLessThan(0.02);
  });

  it('throws (does not silently zero-fill) on missing or mismatched col scales', () => {
    // Missing scales entirely
    expect(() =>
      ArrayDecoder.makePerChannelDequant({ name: 'log_perchannel_u8', bits: 8 }, 3)
    ).toThrow(/col_lo\/col_hi/);
    // Length mismatch (2 scales for a 3-column array)
    expect(() =>
      ArrayDecoder.makePerChannelDequant(
        { name: 'signed_log_perchannel_u16', bits: 16, col_lo: [0, 0], col_hi: [1, 1] },
        3
      )
    ).toThrow(/3 entries/);
  });

  it('throws on non-finite or inverted (hi < lo) per-channel scales', () => {
    // Non-finite scale (would propagate NaN into every covariance) — matches
    // the Python decoder's _perchannel_scales finiteness check.
    expect(() =>
      ArrayDecoder.makePerChannelDequant(
        { name: 'log_perchannel_u8', bits: 8, col_lo: [0, 0], col_hi: [1, Infinity] },
        2
      )
    ).toThrow(/finite/);
    // Inverted range hi < lo would silently invert the scale.
    expect(() =>
      ArrayDecoder.makePerChannelDequant(
        { name: 'signed_log_perchannel_u16', bits: 16, col_lo: [5, 0], col_hi: [1, 1] },
        2
      )
    ).toThrow(/col_hi >= col_lo/);
    // hi === lo (constant column) is VALID — must not throw.
    expect(() =>
      ArrayDecoder.makePerChannelDequant(
        { name: 'log_perchannel_u8', bits: 8, col_lo: [2, 0], col_hi: [2, 1] },
        2
      )
    ).not.toThrow();
  });

  it('returns identity for float32 / direct / unknown encodings', () => {
    for (const enc of [undefined, { name: 'float32' }, { name: 'something_else' }]) {
      const deq = ArrayDecoder.makePerChannelDequant(enc, 3);
      expect(deq(123.5, 0)).toBe(123.5);
      expect(deq(-7, 2)).toBe(-7);
    }
  });

  it('classifies all six per-channel encodings as known, self-decoded, not global-quantized', () => {
    for (const n of [
      'log_perchannel_u8',
      'log_perchannel_u16',
      'signed_log_perchannel_u8',
      'signed_log_perchannel_u16',
      'linear_perchannel_u8',
      'linear_perchannel_u16',
    ]) {
      expect(ArrayDecoder.isPerChannelQuantEncodingName(n)).toBe(true);
      expect(ArrayDecoder.isKnownEncodingName(n)).toBe(true);
      expect(ArrayDecoder.isQuantizedEncodingName(n)).toBe(false); // not the global path
      // isEncoded=true → the RangeLoader fully decodes them to float32 (consumers
      // never dequant themselves), and points allocates a Float32 output buffer.
      expect(ArrayDecoder.isEncoded({ encoding: { name: n } })).toBe(true);
    }
  });
});
