/**
 * Float32 transcendental math matching Rust/WASM's `libm` operation order.
 *
 * Ported from `libm` 0.2.15 `expf.rs` and `logf.rs`, which carry the FreeBSD
 * msun `e_expf.c` and `e_logf.c` implementations. Kernels evaluating `exp` or
 * `log` on an f32 must use `expf` or `logf`, never host `Math.exp` / `Math.log`.
 *
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

const floatBitsView = new DataView(new ArrayBuffer(4));

function floatToBits(value: number): number {
  floatBitsView.setFloat32(0, value, true);
  return floatBitsView.getUint32(0, true);
}

function bitsToFloat(bits: number): number {
  floatBitsView.setUint32(0, bits, true);
  return floatBitsView.getFloat32(0, true);
}

function scalbnf(value: number, exponent: number): number {
  let scaled = value;
  let remaining = exponent;
  if (remaining > 127) {
    scaled = Math.fround(scaled * bitsToFloat(0x7f000000));
    remaining -= 127;
    if (remaining > 127) {
      scaled = Math.fround(scaled * bitsToFloat(0x7f000000));
      remaining = Math.min(remaining - 127, 127);
    }
  } else if (remaining < -126) {
    const minimumNormalTimesMantissa = bitsToFloat(0x0c800000);
    scaled = Math.fround(scaled * minimumNormalTimesMantissa);
    remaining += 102;
    if (remaining < -126) {
      scaled = Math.fround(scaled * minimumNormalTimesMantissa);
      remaining = Math.max(remaining + 102, -126);
    }
  }
  return Math.fround(scaled * bitsToFloat((127 + remaining) << 23));
}

const EXP_LN2_HI = bitsToFloat(0x3f317200);
const EXP_LN2_LO = bitsToFloat(0x35bfbe8e);
const EXP_INV_LN2 = bitsToFloat(0x3fb8aa3b);
const EXP_P1 = bitsToFloat(0x3e2aaa8f);
const EXP_P2 = bitsToFloat(0xbb355215);

/** Rust compiler-builtins `expf`, evaluated in the same f32 operation order. */
export function expf(value: number): number {
  let x = Math.fround(value);
  let magnitudeBits = floatToBits(x);
  const sign = magnitudeBits >>> 31;
  magnitudeBits &= 0x7fffffff;

  if (magnitudeBits >= 0x42aeac50) {
    if (magnitudeBits > 0x7f800000) return x;
    if (magnitudeBits >= 0x42b17218 && sign === 0) {
      return Math.fround(x * bitsToFloat(0x7f000000));
    }
    if (sign !== 0 && magnitudeBits >= 0x42cff1b5) return 0;
  }

  let exponent: number;
  let high: number;
  let low: number;
  if (magnitudeBits > 0x3eb17218) {
    if (magnitudeBits > 0x3f851592) {
      exponent = Math.trunc(Math.fround(Math.fround(EXP_INV_LN2 * x) + (sign === 0 ? 0.5 : -0.5)));
    } else {
      exponent = 1 - sign - sign;
    }
    const exponentFloat = Math.fround(exponent);
    high = Math.fround(x - Math.fround(exponentFloat * EXP_LN2_HI));
    low = Math.fround(exponentFloat * EXP_LN2_LO);
    x = Math.fround(high - low);
  } else if (magnitudeBits > 0x39000000) {
    exponent = 0;
    high = x;
    low = 0;
  } else {
    return Math.fround(1 + x);
  }

  const squared = Math.fround(x * x);
  const polynomial = Math.fround(EXP_P1 + Math.fround(squared * EXP_P2));
  const correction = Math.fround(x - Math.fround(squared * polynomial));
  const quotient = Math.fround(Math.fround(x * correction) / Math.fround(2 - correction));
  const result = Math.fround(1 + Math.fround(Math.fround(quotient - low) + high));
  return exponent === 0 ? result : scalbnf(result, exponent);
}

const LOG_LN2_HI = bitsToFloat(0x3f317180);
const LOG_LN2_LO = bitsToFloat(0x3717f7d1);
const LOG_LG1 = bitsToFloat(0x3f2aaaaa);
const LOG_LG2 = bitsToFloat(0x3eccce13);
const LOG_LG3 = bitsToFloat(0x3e91e9ee);
const LOG_LG4 = bitsToFloat(0x3e789e26);

/** Rust compiler-builtins `logf`, evaluated in the same f32 operation order. */
export function logf(value: number): number {
  let x = Math.fround(value);
  let bits = floatToBits(x);
  let exponent = 0;

  if (bits < 0x00800000 || bits >>> 31 !== 0) {
    if (bits << 1 === 0) return -Infinity;
    if (bits >>> 31 !== 0) return NaN;
    exponent -= 25;
    x = Math.fround(x * bitsToFloat(0x4c000000));
    bits = floatToBits(x);
  } else if (bits >= 0x7f800000) {
    return x;
  } else if (bits === 0x3f800000) {
    return 0;
  }

  bits = (bits + (0x3f800000 - 0x3f3504f3)) >>> 0;
  exponent += (bits >>> 23) - 0x7f;
  bits = (bits & 0x007fffff) + 0x3f3504f3;
  x = bitsToFloat(bits);

  const difference = Math.fround(x - 1);
  const ratio = Math.fround(difference / Math.fround(2 + difference));
  const ratioSquared = Math.fround(ratio * ratio);
  const ratioFourth = Math.fround(ratioSquared * ratioSquared);
  const term1 = Math.fround(
    ratioFourth * Math.fround(LOG_LG2 + Math.fround(ratioFourth * LOG_LG4))
  );
  const term2 = Math.fround(
    ratioSquared * Math.fround(LOG_LG1 + Math.fround(ratioFourth * LOG_LG3))
  );
  const remainder = Math.fround(term2 + term1);
  const halfDifferenceSquared = Math.fround(Math.fround(0.5 * difference) * difference);
  const exponentFloat = Math.fround(exponent);

  let result = Math.fround(
    Math.fround(ratio * Math.fround(halfDifferenceSquared + remainder)) +
      Math.fround(exponentFloat * LOG_LN2_LO)
  );
  result = Math.fround(result - halfDifferenceSquared);
  result = Math.fround(result + difference);
  return Math.fround(result + Math.fround(exponentFloat * LOG_LN2_HI));
}
