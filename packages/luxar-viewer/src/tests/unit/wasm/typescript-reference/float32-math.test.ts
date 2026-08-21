import { describe, expect, it } from 'vitest';
import { expf, logf } from '../../../../wasm/typescript/float32-math';

const bitBuffer = new ArrayBuffer(4);
const bitView = new DataView(bitBuffer);

function fromBits(bits: number): number {
  bitView.setUint32(0, bits, true);
  return bitView.getFloat32(0, true);
}

function toBits(value: number): number {
  bitView.setFloat32(0, value, true);
  return bitView.getUint32(0, true);
}

const EXP_VECTORS: ReadonlyArray<readonly [number, number]> = [
  [0xff800000, 0x00000000],
  [0xc2d00000, 0x00000000],
  [0xc2cfc000, 0x00000001],
  [0xc2a90000, 0x0288742e],
  [0xc1a00000, 0x310da433],
  [0xc0f20000, 0x3a083411],
  [0xbf800000, 0x3ebc5ab2],
  [0xbe800000, 0x3f475f7d],
  [0xb8000000, 0x3f7ffe00],
  [0x80000000, 0x3f800000],
  [0x00000000, 0x3f800000],
  [0x38000000, 0x3f800100],
  [0x39000001, 0x3f800400],
  [0x3e800000, 0x3fa45af2],
  [0x3eb17218, 0x3fb504f3],
  [0x3f800000, 0x402df854],
  [0x41200000, 0x46ac14ee],
  [0x42b00000, 0x7ef882b7],
  [0x42b20000, 0x7f800000],
  [0x7f800000, 0x7f800000],
];

const LOG_VECTORS: ReadonlyArray<readonly [number, number]> = [
  [0x00000000, 0xff800000],
  [0x80000000, 0xff800000],
  [0x00000001, 0xc2ce8ed0],
  [0x007fffff, 0xc2aeac50],
  [0x00800000, 0xc2aeac50],
  [0x2fb2ebb1, 0xc1aec431],
  [0x3dcccccd, 0xc0135d8e],
  [0x3f000000, 0xbf317218],
  [0x3f800000, 0x00000000],
  [0x40000000, 0x3f317218],
  [0x4028ee5f, 0x3f7879c6],
  [0x4f000001, 0x41abe687],
  [0x7f7fffff, 0x42b17218],
  [0x7f800000, 0x7f800000],
];

describe('compiler-builtins float32 math', () => {
  // Expected bits were captured from direct temporary exports in the real
  // rustc 1.92 wasm32 build, rather than from another JavaScript math library.
  it('matches Rust expf bit-for-bit across normal, subnormal, and special values', () => {
    for (const [inputBits, expectedBits] of EXP_VECTORS) {
      expect(toBits(expf(fromBits(inputBits))), inputBits.toString(16)).toBe(expectedBits);
    }
  });

  it('matches Rust logf bit-for-bit across normal, subnormal, and special values', () => {
    for (const [inputBits, expectedBits] of LOG_VECTORS) {
      expect(toBits(logf(fromBits(inputBits))), inputBits.toString(16)).toBe(expectedBits);
    }
  });

  it('covers inputs where rounded JavaScript transcendentals choose another float', () => {
    const expInput = fromBits(0xc07fff04);
    const logInput = fromBits(0x0001a2a2);
    expect(expf(expInput)).not.toBe(Math.fround(Math.exp(expInput)));
    expect(logf(logInput)).not.toBe(Math.fround(Math.log(logInput)));
  });

  it('preserves the Rust special-value contract without pinning NaN payloads', () => {
    expect(Number.isNaN(expf(Number.NaN))).toBe(true);
    expect(Number.isNaN(logf(-1))).toBe(true);
    expect(Number.isNaN(logf(Number.NaN))).toBe(true);
  });
});
