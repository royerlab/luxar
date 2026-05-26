/**
 * Tests for `src/wasm/typescript/decode.ts` (quantized, LUT, broadcast decoders).
 *
 * Extracted from `tests/unit/wasm/typescript-reference.test.ts` per the
 * restructuring plan (wasm.md O1 / Phase D6). Combines the three
 * sibling `decode: *` describes (quantized, LUT, broadcast) since all
 * three test the same source module.
 */

import { describe, it, expect } from 'vitest';
import {
  decode_quantized_u8,
  decode_quantized_u16,
  decode_log_scalar_u8,
  decode_log_scalar_u16,
  decode_lut_scalar_u8,
  decode_lut_scalar_u16,
  decode_lut_row_u8,
  decode_lut_row_u16,
  decode_broadcasted,
} from '../../../../wasm/typescript';

describe('decode: quantized functions', () => {
  // wasm.md O6 / Phase E4: the four `decode: quantized` tests form a
  // 2x2 grid of {u8, u16} × {linear, log-space}. Each previously used
  // its own `it` with hand-tuned tolerances, so a regression in just
  // the u16 log-space path surfaced as a generic
  // "should decode log-space quantization uint16" failure that doesn't
  // name which of the 4 grid cells regressed. Split via two
  // `it.each` tables (one per math family, since linear vs log have
  // different decoder signatures and tolerance shapes) so each row
  // names its specific variant.

  // --- Linear quantization (u8 / u16): decode(data, lo, hi, out) ---
  it.each<{
    label: string;
    decode: (data: Uint8Array | Uint16Array, lo: number, hi: number, out: Float32Array) => void;
    data: Uint8Array | Uint16Array;
    lo: number;
    hi: number;
    expected: [number, number, number];
    precision: [number, number, number];
  }>([
    {
      label: 'u8, range [0, 10]',
      decode: decode_quantized_u8 as never,
      data: new Uint8Array([0, 128, 255]),
      lo: 0.0,
      hi: 10.0,
      expected: [0.0, 5.02, 10.0], // 128/255 * 10 ≈ 5.02
      precision: [2, 1, 2],
    },
    {
      label: 'u16, range [-1, 1]',
      decode: decode_quantized_u16 as never,
      data: new Uint16Array([0, 32768, 65535]),
      lo: -1.0,
      hi: 1.0,
      expected: [-1.0, 0.0, 1.0],
      precision: [2, 2, 2],
    },
  ])('linear quantization: $label', ({ decode, data, lo, hi, expected, precision }) => {
    const output = new Float32Array(3);
    decode(data, lo, hi, output);
    expect(output[0]).toBeCloseTo(expected[0], precision[0]);
    expect(output[1]).toBeCloseTo(expected[1], precision[1]);
    expect(output[2]).toBeCloseTo(expected[2], precision[2]);
  });

  // --- Log-space quantization (u8 / u16): decode(data, maxLog, out) ---
  // expm1(0)=0; expm1(~2.5)≈11.2-11.3 (between 10 and 13); expm1(maxLog)
  // at index N-1. The middle assertion uses bounds (not toBeCloseTo)
  // because the exact value differs slightly between u8 (255/255 * 5)
  // and u16 (32768/65535 * 5).
  it.each<{
    label: string;
    decode: (data: Uint8Array | Uint16Array, maxLog: number, out: Float32Array) => void;
    data: Uint8Array | Uint16Array;
  }>([
    { label: 'u8', decode: decode_log_scalar_u8 as never, data: new Uint8Array([0, 128, 255]) },
    {
      label: 'u16',
      decode: decode_log_scalar_u16 as never,
      data: new Uint16Array([0, 32768, 65535]),
    },
  ])('log-space quantization: $label, maxLog=5', ({ decode, data }) => {
    const maxLog = 5.0;
    const output = new Float32Array(3);
    decode(data, maxLog, output);
    expect(output[0]).toBeCloseTo(0, 2);
    expect(output[1]).toBeGreaterThan(10);
    expect(output[1]).toBeLessThan(13);
    expect(output[2]).toBeCloseTo(Math.expm1(maxLog), 1);
  });
});

describe('decode: LUT functions', () => {
  it('should decode scalar LUT indices uint8', () => {
    const indices = new Uint8Array([0, 2, 1]);
    const lut = new Float32Array([1.0, 2.0, 3.0]);
    const output = new Float32Array(3);

    decode_lut_scalar_u8(indices, lut, output);

    expect(output[0]).toBe(1.0);
    expect(output[1]).toBe(3.0);
    expect(output[2]).toBe(2.0);
  });

  it('should decode scalar LUT indices uint16', () => {
    const indices = new Uint16Array([0, 2, 1]);
    const lut = new Float32Array([1.0, 2.0, 3.0]);
    const output = new Float32Array(3);

    decode_lut_scalar_u16(indices, lut, output);

    expect(output[0]).toBe(1.0);
    expect(output[1]).toBe(3.0);
    expect(output[2]).toBe(2.0);
  });

  it('should decode row LUT indices (vector attributes) uint8', () => {
    const indices = new Uint8Array([0, 1]);
    // LUT with 2 entries, each with 3 values (rgb)
    const lut = new Float32Array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]); // red, green
    const output = new Float32Array(6);

    decode_lut_row_u8(indices, lut, 3, output);

    // First row: red
    expect(output[0]).toBe(1.0);
    expect(output[1]).toBe(0.0);
    expect(output[2]).toBe(0.0);
    // Second row: green
    expect(output[3]).toBe(0.0);
    expect(output[4]).toBe(1.0);
    expect(output[5]).toBe(0.0);
  });

  it('should decode row LUT indices (vector attributes) uint16', () => {
    const indices = new Uint16Array([0, 1]);
    // LUT with 2 entries, each with 3 values (rgb)
    const lut = new Float32Array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]); // red, green
    const output = new Float32Array(6);

    decode_lut_row_u16(indices, lut, 3, output);

    // First row: red
    expect(output[0]).toBe(1.0);
    expect(output[1]).toBe(0.0);
    expect(output[2]).toBe(0.0);
    // Second row: green
    expect(output[3]).toBe(0.0);
    expect(output[4]).toBe(1.0);
    expect(output[5]).toBe(0.0);
  });
});

describe('decode: broadcast function', () => {
  it('should broadcast scalar to all points', () => {
    const value = new Float32Array([0.5]);
    const output = new Float32Array(5);

    decode_broadcasted(value, 5, 1, output);

    for (let i = 0; i < 5; i++) {
      expect(output[i]).toBe(0.5);
    }
  });

  it('should broadcast vector to all points', () => {
    const value = new Float32Array([0.5, 0.6, 0.7]); // rgb
    const output = new Float32Array(9); // 3 points * 3 elements

    decode_broadcasted(value, 3, 3, output);

    for (let i = 0; i < 3; i++) {
      expect(output[i * 3]).toBeCloseTo(0.5, 5);
      expect(output[i * 3 + 1]).toBeCloseTo(0.6, 5);
      expect(output[i * 3 + 2]).toBeCloseTo(0.7, 5);
    }
  });

  // Regression: MED-19 — reject ambiguous middle-length inputs. Previously
  // `value.length=2, elementsPerPoint=3` would silently produce a row of
  // `[v0, v1, v0]` (mixed broadcast). Now it throws.
  it('should throw on ambiguous value.length between 1 and elementsPerPoint', () => {
    const value = new Float32Array([0.1, 0.2]); // length 2
    const output = new Float32Array(9); // 3 points * 3 elements

    expect(() => decode_broadcasted(value, 3, 3, output)).toThrow(
      /value\.length must be 1.*or elementsPerPoint \(3\), got 2/
    );
  });

  it('should throw on value.length greater than elementsPerPoint', () => {
    const value = new Float32Array([0.1, 0.2, 0.3, 0.4]); // length 4
    const output = new Float32Array(6); // 2 points * 3 elements

    expect(() => decode_broadcasted(value, 2, 3, output)).toThrow(
      /value\.length must be 1.*or elementsPerPoint \(3\), got 4/
    );
  });
});
