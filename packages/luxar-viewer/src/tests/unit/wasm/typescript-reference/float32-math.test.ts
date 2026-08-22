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
    { input: '0x00000000', expected: '0x00000000' },
    { input: '0x80000000', expected: '0x80000000' },
    { input: '0x00000001', expected: '0x00000001' },
    { input: '0x007fffff', expected: '0x007fffff' },
    { input: '0x00800000', expected: '0x00800000' },
    { input: '0x33000000', expected: '0x33000000' },
    { input: '0x33000001', expected: '0x33000001' },
    { input: '0x3e800000', expected: '0x3e916bc8' },
    { input: '0xbe800000', expected: '0xbe62820c' },
    { input: '0x3eb17218', expected: '0x3ed413cd' },
    { input: '0x3eb17219', expected: '0x3ed413ce' },
    { input: '0x3f851591', expected: '0x3fea09e4' },
    { input: '0x3f851592', expected: '0x3fea09e6' },
    { input: '0xbf333333', expected: '0xbf00dfc9' },
    { input: '0xc0a00000', expected: '0xbf7e466c' },
    { input: '0x4195b843', expected: '0x4cffffd9' },
    { input: '0x4195b844', expected: '0x4cfffff9' },
    { input: '0xc195b844', expected: '0xbf800000' },
    { input: '0x42340000', expected: '0x5ff267bb' },
    { input: '0x42700000', expected: '0x6abcede5' },
    { input: '0x42b17180', expected: '0x7f7fb40f' },
    { input: '0x42b17181', expected: '0x7f800000' },
    { input: '0x7f800000', expected: '0x7f800000' },
    { input: '0xff800000', expected: '0xbf800000' },
  ])('maps float32 bits $input to $expected', ({ input, expected }) => {
    expect(toBits(expm1f(fromBits(Number(input))))).toBe(Number(expected));
  });

  it('returns NaN without pinning its payload', () => {
    expect(Number.isNaN(expm1f(NaN))).toBe(true);
  });
});
