import { describe, expect, it } from 'vitest';
import { coerceColorsToFloat32, coerceScalarsToFloat32 } from '../../../workers/data-worker';

describe('coerceColorsToFloat32', () => {
  it('Float32 input passes through unchanged (same reference)', () => {
    const input = new Float32Array([0.0, 0.5, 1.0, 0.25]);
    const out = coerceColorsToFloat32(input);
    expect(out).toBe(input);
  });

  it('Uint8 input is normalized by 1/255', () => {
    const input = new Uint8Array([0, 128, 255, 64]);
    const out = coerceColorsToFloat32(input);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(input.length);
    expect(out[0]).toBeCloseTo(0, 6);
    expect(out[1]).toBeCloseTo(128 / 255, 6);
    expect(out[2]).toBeCloseTo(1, 6);
    expect(out[3]).toBeCloseTo(64 / 255, 6);
  });

  it('Uint16 input is normalized by 1/65535', () => {
    const input = new Uint16Array([0, 32768, 65535, 16384]);
    const out = coerceColorsToFloat32(input);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(input.length);
    expect(out[0]).toBeCloseTo(0, 6);
    expect(out[1]).toBeCloseTo(32768 / 65535, 6);
    expect(out[2]).toBeCloseTo(1, 6);
    expect(out[3]).toBeCloseTo(16384 / 65535, 6);
  });

  it('handles empty arrays', () => {
    expect(coerceColorsToFloat32(new Uint8Array(0)).length).toBe(0);
    expect(coerceColorsToFloat32(new Uint16Array(0)).length).toBe(0);
    expect(coerceColorsToFloat32(new Float32Array(0)).length).toBe(0);
  });

  it('worker output matches main-thread reference for Uint8 RGB', () => {
    // Mirrors the reference normalization in
    // data/gsplats/projection.ts:processGSplats3DOnly and
    // data/lines/projection.ts:buildInstanceBuffers — both apply
    // colors[i] * (1/255) before passing to WASM. The worker path
    // must produce byte-equal output.
    const input = new Uint8Array([0, 64, 128, 192, 255, 1]);
    const workerOut = coerceColorsToFloat32(input);
    const mainOut = new Float32Array(input.length);
    const norm = 1 / 255;
    for (let i = 0; i < input.length; i++) mainOut[i] = input[i] * norm;
    expect(Array.from(workerOut)).toEqual(Array.from(mainOut));
  });

  it('worker output matches main-thread reference for Uint16 RGB', () => {
    const input = new Uint16Array([0, 16384, 32768, 49152, 65535, 1]);
    const workerOut = coerceColorsToFloat32(input);
    const mainOut = new Float32Array(input.length);
    const norm = 1 / 65535;
    for (let i = 0; i < input.length; i++) mainOut[i] = input[i] * norm;
    expect(Array.from(workerOut)).toEqual(Array.from(mainOut));
  });
});

describe('coerceScalarsToFloat32', () => {
  it('Float32 input passes through unchanged (same reference)', () => {
    const input = new Float32Array([0.0, 0.5, 1.0, 0.25]);
    const out = coerceScalarsToFloat32(input);
    expect(out).toBe(input);
  });

  it('Uint8 input is normalized by 1/255', () => {
    const input = new Uint8Array([0, 128, 255, 64]);
    const out = coerceScalarsToFloat32(input);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(input.length);
    expect(out[0]).toBeCloseTo(0, 6);
    expect(out[1]).toBeCloseTo(128 / 255, 6);
    expect(out[2]).toBeCloseTo(1, 6);
    expect(out[3]).toBeCloseTo(64 / 255, 6);
  });

  it('Float16 input is element-wise expanded (no normalization)', () => {
    // Float16Array support is environment-dependent in 2026; skip
    // gracefully when unavailable. The runtime contract still holds:
    // when the constructor exists, the coerce path expands element-
    // wise without any normalization (Float16 carries real-valued
    // scalars, not normalized indices).
    const F16 = (globalThis as { Float16Array?: typeof Float16Array }).Float16Array;
    if (!F16) return;
    const input = new F16([0.0, 0.5, 1.0, 2.5]);
    const out = coerceScalarsToFloat32(input);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(input.length);
    expect(out[0]).toBeCloseTo(0, 5);
    expect(out[1]).toBeCloseTo(0.5, 5);
    expect(out[2]).toBeCloseTo(1, 5);
    expect(out[3]).toBeCloseTo(2.5, 3);
  });

  it('handles empty arrays', () => {
    expect(coerceScalarsToFloat32(new Uint8Array(0)).length).toBe(0);
    expect(coerceScalarsToFloat32(new Float32Array(0)).length).toBe(0);
  });
});
