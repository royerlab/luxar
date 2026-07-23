/**
 * Unit tests for the dtype-widening helper consumed by the texel
 * writers (`rendering/widen-to-float32.ts`) — extracted from the
 * retired interleaved-attributes suite when the lines texture-storage
 * migration deleted the packing machinery.
 */
import { describe, it, expect } from 'vitest';
import { widenToFloat32 } from '../../../rendering/widen-to-float32';

describe('widenToFloat32', () => {
  it('returns the same reference when already Float32 and no divisor', () => {
    const src = new Float32Array([1, 2, 3]);
    expect(widenToFloat32(src)).toBe(src);
  });

  it('widens a Uint8Array to Float32 verbatim', () => {
    const out = widenToFloat32(new Uint8Array([0, 128, 255]));
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out)).toEqual([0, 128, 255]);
  });

  it('divides by `divisor` to preserve the normalized [0,1] range', () => {
    const out = widenToFloat32(new Uint8Array([0, 128, 255]), 255);
    expect(out[0]).toBeCloseTo(0.0, 5);
    expect(out[1]).toBeCloseTo(128 / 255, 5);
    expect(out[2]).toBeCloseTo(1.0, 5);
  });

  it('widens Uint16 sources as well', () => {
    const out = widenToFloat32(new Uint16Array([0, 32768, 65535]));
    expect(Array.from(out)).toEqual([0, 32768, 65535]);
  });

  it('copies (never aliases) a Float32 source when a divisor is supplied', () => {
    const src = new Float32Array([2, 4]);
    const out = widenToFloat32(src, 2);
    expect(out).not.toBe(src);
    expect(Array.from(out)).toEqual([1, 2]);
  });
});
