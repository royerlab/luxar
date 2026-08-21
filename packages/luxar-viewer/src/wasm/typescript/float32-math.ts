const floatBitsView = new DataView(new ArrayBuffer(4));

function floatToBits(value: number): number {
  floatBitsView.setFloat32(0, value, true);
  return floatBitsView.getUint32(0, true);
}

function bitsToFloat(bits: number): number {
  floatBitsView.setUint32(0, bits, true);
  return floatBitsView.getFloat32(0, true);
}

const EXPM1_OVERFLOW_THRESHOLD = bitsToFloat(0x42b17180);
const EXPM1_LN2_HI = bitsToFloat(0x3f317180);
const EXPM1_LN2_LO = bitsToFloat(0x3717f7d1);
const EXPM1_INV_LN2 = bitsToFloat(0x3fb8aa3b);
const EXPM1_Q1 = bitsToFloat(0xbd088868);
const EXPM1_Q2 = bitsToFloat(0x3acf3010);

/** Rust compiler-builtins `expm1f`, evaluated in the same f32 operation order. */
export function expm1f(value: number): number {
  let x = Math.fround(value);
  let magnitudeBits = floatToBits(x);
  const negative = magnitudeBits >>> 31 !== 0;
  magnitudeBits &= 0x7fffffff;

  if (magnitudeBits >= 0x4195b844) {
    if (magnitudeBits > 0x7f800000) return x;
    if (negative) return -1;
    if (x > EXPM1_OVERFLOW_THRESHOLD) {
      return Math.fround(x * bitsToFloat(0x7f000000));
    }
  }

  let exponent: number;
  let high: number;
  let low: number;
  let correction = 0;
  if (magnitudeBits > 0x3eb17218) {
    if (magnitudeBits < 0x3f851592) {
      if (!negative) {
        high = Math.fround(x - EXPM1_LN2_HI);
        low = EXPM1_LN2_LO;
        exponent = 1;
      } else {
        high = Math.fround(x + EXPM1_LN2_HI);
        low = -EXPM1_LN2_LO;
        exponent = -1;
      }
    } else {
      exponent = Math.trunc(
        Math.fround(Math.fround(EXPM1_INV_LN2 * x) + Math.fround(negative ? -0.5 : 0.5))
      );
      const exponentFloat = Math.fround(exponent);
      high = Math.fround(x - Math.fround(exponentFloat * EXPM1_LN2_HI));
      low = Math.fround(exponentFloat * EXPM1_LN2_LO);
    }
    x = Math.fround(high - low);
    correction = Math.fround(Math.fround(high - x) - low);
  } else if (magnitudeBits < 0x33000000) {
    return x;
  } else {
    exponent = 0;
  }

  const half = Math.fround(0.5 * x);
  const squaredHalf = Math.fround(x * half);
  const polynomial = Math.fround(
    1 + Math.fround(squaredHalf * Math.fround(EXPM1_Q1 + Math.fround(squaredHalf * EXPM1_Q2)))
  );
  const denominatorTerm = Math.fround(3 - Math.fround(polynomial * half));
  let error = Math.fround(
    squaredHalf *
      Math.fround(
        Math.fround(polynomial - denominatorTerm) /
          Math.fround(6 - Math.fround(x * denominatorTerm))
      )
  );

  if (exponent === 0) {
    return Math.fround(x - Math.fround(Math.fround(x * error) - squaredHalf));
  }

  error = Math.fround(Math.fround(x * Math.fround(error - correction)) - correction);
  error = Math.fround(error - squaredHalf);
  if (exponent === -1) {
    return Math.fround(Math.fround(0.5 * Math.fround(x - error)) - 0.5);
  }
  if (exponent === 1) {
    if (x < -0.25) {
      return Math.fround(-2 * Math.fround(error - Math.fround(x + 0.5)));
    }
    return Math.fround(1 + Math.fround(2 * Math.fround(x - error)));
  }

  const twoToExponent = bitsToFloat(((0x7f + exponent) << 23) >>> 0);
  if (exponent < 0 || exponent > 56) {
    let result = Math.fround(Math.fround(x - error) + 1);
    result =
      exponent === 128
        ? Math.fround(Math.fround(result * 2) * bitsToFloat(0x7f000000))
        : Math.fround(result * twoToExponent);
    return Math.fround(result - 1);
  }

  const twoToNegativeExponent = bitsToFloat(((0x7f - exponent) << 23) >>> 0);
  if (exponent < 23) {
    return Math.fround(
      Math.fround(Math.fround(x - error) + Math.fround(1 - twoToNegativeExponent)) * twoToExponent
    );
  }
  return Math.fround(
    Math.fround(Math.fround(x - Math.fround(error + twoToNegativeExponent)) + 1) * twoToExponent
  );
}
