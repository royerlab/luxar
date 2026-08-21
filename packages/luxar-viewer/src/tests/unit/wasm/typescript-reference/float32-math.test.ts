import { describe, expect, it } from 'vitest';
import { expm1f } from '../../../../wasm/typescript/float32-math';

const bits = new Uint32Array(1);
const values = new Float32Array(bits.buffer);

function fromBits(value: number): number {
  bits[0] = value;
  return values[0];
}

function toBits(value: number): number {
  values[0] = value;
  return bits[0];
}

describe('expm1f', () => {
  it.each([
    [0x00000000, 0x00000000],
    [0x80000000, 0x80000000],
    [0x00000001, 0x00000001],
    [0x007fffff, 0x007fffff],
    [0x00800000, 0x00800000],
    [0x33000000, 0x33000000],
    [0x33000001, 0x33000001],
    [0x3e800000, 0x3e916bc8],
    [0xbe800000, 0xbe62820c],
    [0x3eb17218, 0x3ed413cd],
    [0x3eb17219, 0x3ed413ce],
    [0x3f851591, 0x3fea09e4],
    [0x3f851592, 0x3fea09e6],
    [0x4195b843, 0x4cffffd9],
    [0x4195b844, 0x4cfffff9],
    [0xc195b844, 0xbf800000],
    [0x42b17180, 0x7f7fb40f],
    [0x42b17181, 0x7f800000],
    [0x7f800000, 0x7f800000],
    [0xff800000, 0xbf800000],
  ])('maps float32 bits %#x to %#x', (inputBits, expectedBits) => {
    expect(toBits(expm1f(fromBits(inputBits)))).toBe(expectedBits);
  });

  it('returns NaN without pinning its payload', () => {
    expect(Number.isNaN(expm1f(NaN))).toBe(true);
  });
});
