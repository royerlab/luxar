/**
 * Tests for the geometric-log scalar decode path (geolog_scalar_uint8/uint16).
 *
 * The rescale-first encoding for wide-dynamic-range positive scalars (gsplat
 * amplitudes): min/max-anchored true-log grid, level 0 RESERVED for exact
 * zeros — mirrors Python `_decode_geolog_scalar` and the WASM kernels.
 */

import { describe, expect, it } from 'vitest';
import { ArrayDecoder, ArrayRefRegistry } from '../../../../data/array-decoder/decoder';
import type { ArrayMetadata } from '../../../../data/array-decoder/types';

const MIN_LOG = Math.log(5e-4);
const MAX_LOG = Math.log(2e4);

function attrs(overrides: Record<string, unknown> = {}): ArrayMetadata {
  return {
    encoding: {
      name: 'geolog_scalar_uint16',
      min_log: MIN_LOG,
      max_log: MAX_LOG,
      bits: 16,
      original_dtype: 'float32',
      ...overrides,
    },
  } as ArrayMetadata;
}

function expected(u: number, top = 65535): number {
  return u === 0 ? 0 : Math.exp(MIN_LOG + ((u - 1) / (top - 1)) * (MAX_LOG - MIN_LOG));
}

describe('geolog_scalar decode', () => {
  it('is a known, quantized encoding with its own mode label', () => {
    expect(ArrayDecoder.isGeologScalarEncodingName('geolog_scalar_uint8')).toBe(true);
    expect(ArrayDecoder.isGeologScalarEncodingName('geolog_scalar_uint16')).toBe(true);
    expect(ArrayDecoder.isGeologScalarEncodingName('log_scalar_uint16')).toBe(false);
    expect(ArrayDecoder.isKnownEncodingName('geolog_scalar_uint16')).toBe(true);
    expect(ArrayDecoder.isQuantizedEncodingName('geolog_scalar_uint16')).toBe(true);
    expect(ArrayDecoder.getEncodingMode(attrs())).toBe('geolog_scalar');
  });

  it('getQuantizationMetadata returns the geolog variant with [min_log, max_log]', () => {
    const meta = ArrayDecoder.getQuantizationMetadata(attrs(), 'uint16');
    expect(meta).not.toBeNull();
    expect(meta!.isGeologSpace).toBe(true);
    expect(meta!.isLogSpace).toBe(false);
    expect(meta!.bounds[0]).toBeCloseTo(MIN_LOG, 10);
    expect(meta!.bounds[1]).toBeCloseTo(MAX_LOG, 10);
    expect(meta!.dtype).toBe('uint16');
  });

  it('dequantizeRange decodes with the reserved zero level', () => {
    const decoder = new ArrayDecoder(new ArrayRefRegistry());
    const meta = ArrayDecoder.getQuantizationMetadata(attrs(), 'uint16')!;
    const codes = new Uint16Array([0, 1, 32768, 65535]);
    const out = decoder.dequantizeRange(codes, meta);
    expect(out[0]).toBe(0); // reserved zero level -> exact 0
    expect(out[1]).toBeCloseTo(expected(1), 6); // = exp(min_log), grid endpoint
    expect(out[2]).toBeCloseTo(expected(32768), 0);
    expect(out[3]).toBeCloseTo(expected(65535), -1); // ~2e4, f32 tolerance
    // no nonzero code may decode to zero
    for (let i = 1; i < out.length; i++) expect(out[i]).toBeGreaterThan(0);
  });

  it('uint8 variant uses the 254-interval grid', () => {
    const decoder = new ArrayDecoder(new ArrayRefRegistry());
    const meta = ArrayDecoder.getQuantizationMetadata(
      attrs({ name: 'geolog_scalar_uint8', bits: 8 }),
      'uint8'
    )!;
    const out = decoder.dequantizeRange(new Uint8Array([0, 1, 255]), meta);
    expect(out[0]).toBe(0);
    expect(out[1]).toBeCloseTo(Math.exp(MIN_LOG), 6);
    expect(out[2]).toBeCloseTo(Math.exp(MAX_LOG), -1);
  });

  it('validateEncodingMetadata accepts geolog and rejects malformed anchors', () => {
    expect(() => ArrayDecoder.validateEncodingMetadata(attrs().encoding)).not.toThrow();
    expect(() =>
      ArrayDecoder.validateEncodingMetadata(attrs({ min_log: undefined }).encoding)
    ).toThrow(/min_log/);
    expect(() =>
      ArrayDecoder.validateEncodingMetadata(attrs({ min_log: 5, max_log: 1 }).encoding)
    ).toThrow(/max_log >= min_log/);
    expect(() =>
      ArrayDecoder.validateEncodingMetadata(attrs({ min_log: Number.NaN }).encoding)
    ).toThrow(/Invalid geolog_scalar/);
    // min_log remains geolog-only
    expect(() =>
      ArrayDecoder.validateEncodingMetadata({
        name: 'log_scalar_uint16',
        max_log: 5,
        min_log: 1,
        original_dtype: 'float32',
      })
    ).toThrow(/min_log metadata is only valid/);
  });

  it('constant-value arrays (min_log == max_log) decode without error', () => {
    const decoder = new ArrayDecoder(new ArrayRefRegistry());
    const meta = ArrayDecoder.getQuantizationMetadata(
      attrs({ min_log: 2.0, max_log: 2.0 }),
      'uint16'
    )!;
    const out = decoder.dequantizeRange(new Uint16Array([0, 1, 9]), meta);
    expect(out[0]).toBe(0);
    expect(out[1]).toBeCloseTo(Math.exp(2.0), 5);
    expect(out[2]).toBeCloseTo(Math.exp(2.0), 5);
  });
});
