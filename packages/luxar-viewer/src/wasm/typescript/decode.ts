/**
 * Data decoding functions for quantized and LUT-encoded arrays.
 *
 * TypeScript reference implementation matching decode.rs
 */

/**
 * Decode quantized uint8 data to float32.
 * Maps uint8 [0, 255] to [minVal, maxVal] linearly.
 */
export function decode_quantized_u8(
  data: Uint8Array,
  minVal: number,
  maxVal: number,
  output: Float32Array
): void {
  const scale = (maxVal - minVal) / 255;
  for (let i = 0; i < data.length; i++) {
    output[i] = minVal + data[i] * scale;
  }
}

/**
 * Decode quantized uint16 data to float32.
 * Maps uint16 [0, 65535] to [minVal, maxVal] linearly.
 */
export function decode_quantized_u16(
  data: Uint16Array,
  minVal: number,
  maxVal: number,
  output: Float32Array
): void {
  const scale = (maxVal - minVal) / 65535;
  for (let i = 0; i < data.length; i++) {
    output[i] = minVal + data[i] * scale;
  }
}

/**
 * Decode log-space quantized uint8 data to float32.
 * Decoding: expm1(normalized * maxLog)
 */
export function decode_log_scalar_u8(data: Uint8Array, maxLog: number, output: Float32Array): void {
  const invMax = maxLog / 255;
  for (let i = 0; i < data.length; i++) {
    const normalized = data[i] * invMax;
    output[i] = Math.expm1(normalized);
  }
}

/**
 * Decode log-space quantized uint16 data to float32.
 */
export function decode_log_scalar_u16(
  data: Uint16Array,
  maxLog: number,
  output: Float32Array
): void {
  const invMax = maxLog / 65535;
  for (let i = 0; i < data.length; i++) {
    const normalized = data[i] * invMax;
    output[i] = Math.expm1(normalized);
  }
}

/**
 * Decode geometric-log quantized uint8 data to float32.
 * Reserved zero level: 0 -> exactly 0; levels [1,255] ->
 * exp(minLog + (u-1)/254 * (maxLog - minLog)). Mirrors the Rust kernel 1:1.
 */
export function decode_geolog_scalar_u8(
  data: Uint8Array,
  minLog: number,
  maxLog: number,
  output: Float32Array
): void {
  // fround the anchors: the WASM kernel receives them as f32, so the TS
  // reference must quantize them identically before the f64 math.
  const lo = Math.fround(minLog);
  const inv = Math.max(Math.fround(maxLog) - lo, 0) / 254;
  for (let i = 0; i < data.length; i++) {
    const u = data[i];
    output[i] = u === 0 ? 0 : Math.exp(lo + (u - 1) * inv);
  }
}

/**
 * Decode geometric-log quantized uint16 data to float32.
 */
export function decode_geolog_scalar_u16(
  data: Uint16Array,
  minLog: number,
  maxLog: number,
  output: Float32Array
): void {
  const lo = Math.fround(minLog);
  const inv = Math.max(Math.fround(maxLog) - lo, 0) / 65534;
  for (let i = 0; i < data.length; i++) {
    const u = data[i];
    output[i] = u === 0 ? 0 : Math.exp(lo + (u - 1) * inv);
  }
}

/**
 * Shared per-channel dequantization loop (linear / log / signed-log).
 * Mirrors the Rust `decode_perchannel_impl` 1:1 — see decode.rs for the
 * layout contract (rolling column counter, `colOffset` phase, `zeroLevel`
 * reserved-zero vs legacy all-levels mapping, 1e-30 constant-column clamp).
 * JS numbers ARE f64, so no fround is needed: the scales arrive as
 * Float64Array and the math matches the Rust kernel and the Python decoder
 * bit-for-bit.
 */
function decodePerChannelImpl(
  data: ArrayLike<number>,
  colLo: Float64Array,
  colHi: Float64Array,
  levels: number,
  zeroLevel: boolean,
  colOffset: number,
  transform: (y: number) => number,
  output: Float32Array
): void {
  const cols = colLo.length;
  if (cols === 0) {
    return;
  }
  const rng = new Float64Array(cols);
  for (let c = 0; c < cols; c++) {
    rng[c] = Math.max(colHi[c] - colLo[c], 1e-30);
  }
  const denom = Math.max(levels - 1, 1);
  let c = colOffset % cols;
  for (let i = 0; i < data.length; i++) {
    const u = data[i];
    if (zeroLevel && u === 0) {
      output[i] = 0;
    } else {
      const y = zeroLevel
        ? colLo[c] + ((u - 1) / denom) * rng[c]
        : colLo[c] + (u / levels) * rng[c];
      output[i] = transform(y);
    }
    c += 1;
    if (c === cols) {
      c = 0;
    }
  }
}

/** `sign(y) * expm1(|y|)` — the signed-log inverse compand. */
function signedExpm1(y: number): number {
  return y < 0 ? -Math.expm1(-y) : Math.expm1(y);
}

const identity = (y: number): number => y;

/**
 * Decode per-channel LINEAR (fixed-point) uint8 codes to float32.
 * Identity transform, per-column scales — the COORDINATE encoding. No
 * reserved zero level (mirrors Python's `_decode_linear_perchannel`).
 */
export function decode_linear_perchannel_u8(
  data: Uint8Array,
  colLo: Float64Array,
  colHi: Float64Array,
  colOffset: number,
  output: Float32Array
): void {
  decodePerChannelImpl(data, colLo, colHi, 255, false, colOffset, identity, output);
}

/** Decode per-channel LINEAR (fixed-point) uint16 codes to float32. */
export function decode_linear_perchannel_u16(
  data: Uint16Array,
  colLo: Float64Array,
  colHi: Float64Array,
  colOffset: number,
  output: Float32Array
): void {
  decodePerChannelImpl(data, colLo, colHi, 65535, false, colOffset, identity, output);
}

/**
 * Decode per-channel LOG uint8 codes to float32 (`x = expm1(y)`).
 * The Cholesky-diagonal encoding; `zeroLevel` per the current-vs-legacy
 * layout contract in decode.rs.
 */
export function decode_log_perchannel_u8(
  data: Uint8Array,
  colLo: Float64Array,
  colHi: Float64Array,
  zeroLevel: boolean,
  colOffset: number,
  output: Float32Array
): void {
  decodePerChannelImpl(data, colLo, colHi, 255, zeroLevel, colOffset, Math.expm1, output);
}

/** Decode per-channel LOG uint16 codes to float32. */
export function decode_log_perchannel_u16(
  data: Uint16Array,
  colLo: Float64Array,
  colHi: Float64Array,
  zeroLevel: boolean,
  colOffset: number,
  output: Float32Array
): void {
  decodePerChannelImpl(data, colLo, colHi, 65535, zeroLevel, colOffset, Math.expm1, output);
}

/**
 * Decode per-channel SIGNED-LOG uint8 codes to float32
 * (`x = sign(y)·expm1(|y|)`). The Cholesky off-diagonal encoding.
 */
export function decode_signed_log_perchannel_u8(
  data: Uint8Array,
  colLo: Float64Array,
  colHi: Float64Array,
  zeroLevel: boolean,
  colOffset: number,
  output: Float32Array
): void {
  decodePerChannelImpl(data, colLo, colHi, 255, zeroLevel, colOffset, signedExpm1, output);
}

/** Decode per-channel SIGNED-LOG uint16 codes to float32. */
export function decode_signed_log_perchannel_u16(
  data: Uint16Array,
  colLo: Float64Array,
  colHi: Float64Array,
  zeroLevel: boolean,
  colOffset: number,
  output: Float32Array
): void {
  decodePerChannelImpl(data, colLo, colHi, 65535, zeroLevel, colOffset, signedExpm1, output);
}

/**
 * Decode per-channel TRUE-log uint8 codes to float32 (`x = exp(y)`).
 * HDR colors: ln-domain per-column anchors, uniform relative precision.
 * The reserved zero level is part of the name contract (always on).
 */
export function decode_geolog_perchannel_u8(
  data: Uint8Array,
  colLo: Float64Array,
  colHi: Float64Array,
  colOffset: number,
  output: Float32Array
): void {
  decodePerChannelImpl(data, colLo, colHi, 255, true, colOffset, Math.exp, output);
}

/** Decode per-channel TRUE-log uint16 codes to float32. */
export function decode_geolog_perchannel_u16(
  data: Uint16Array,
  colLo: Float64Array,
  colHi: Float64Array,
  colOffset: number,
  output: Float32Array
): void {
  decodePerChannelImpl(data, colLo, colHi, 65535, true, colOffset, Math.exp, output);
}

/**
 * Decode LUT-encoded uint8 indices to float32 (scalar mode).
 * Each index maps to a single float value from the LUT.
 */
export function decode_lut_scalar_u8(
  indices: Uint8Array,
  lut: Float32Array,
  output: Float32Array
): void {
  for (let i = 0; i < indices.length; i++) {
    output[i] = lut[indices[i]];
  }
}

/**
 * Decode LUT-encoded uint16 indices to float32 (scalar mode).
 */
export function decode_lut_scalar_u16(
  indices: Uint16Array,
  lut: Float32Array,
  output: Float32Array
): void {
  for (let i = 0; i < indices.length; i++) {
    output[i] = lut[indices[i]];
  }
}

/**
 * Decode LUT-encoded uint8 indices to float32 (row mode).
 * Each index maps to k consecutive float values from the LUT.
 */
export function decode_lut_row_u8(
  indices: Uint8Array,
  lut: Float32Array,
  k: number,
  output: Float32Array
): void {
  for (let i = 0; i < indices.length; i++) {
    const lutOffset = indices[i] * k;
    const outOffset = i * k;
    for (let j = 0; j < k; j++) {
      output[outOffset + j] = lut[lutOffset + j];
    }
  }
}

/**
 * Decode LUT-encoded uint16 indices to float32 (row mode).
 */
export function decode_lut_row_u16(
  indices: Uint16Array,
  lut: Float32Array,
  k: number,
  output: Float32Array
): void {
  for (let i = 0; i < indices.length; i++) {
    const lutOffset = indices[i] * k;
    const outOffset = i * k;
    for (let j = 0; j < k; j++) {
      output[outOffset + j] = lut[lutOffset + j];
    }
  }
}

/**
 * Broadcast a single value to all points.
 *
 * Contract: `value.length` must be either 1 (scalar broadcast to all
 * `elementsPerPoint` slots) or exactly `elementsPerPoint` (per-element
 * vector replicated across every point). Any other length is rejected
 * because the previous "mixed broadcast" semantics produced surprising
 * rows like `[v0, v1, v0]` for `value.length=2, elementsPerPoint=3`.
 */
export function decode_broadcasted(
  value: Float32Array,
  numPoints: number,
  elementsPerPoint: number,
  output: Float32Array
): void {
  if (value.length !== 1 && value.length !== elementsPerPoint) {
    throw new Error(
      `decode_broadcasted: value.length must be 1 (broadcast) or elementsPerPoint (${elementsPerPoint}), got ${value.length}`
    );
  }
  // Two distinct hot paths — split them so neither carries the
  // surprising "mixed broadcast" fallback (`j < value.length ? value[j]
  // : value[0]`) the old impl exposed when value.length fell between 1
  // and elementsPerPoint. The length guard above already rejects every
  // other shape; the ternary at the read site was dead defensive code
  // that obscured intent.
  if (value.length === 1) {
    const v = value[0];
    output.fill(v, 0, numPoints * elementsPerPoint);
  } else {
    for (let i = 0; i < numPoints; i++) {
      const outOffset = i * elementsPerPoint;
      for (let j = 0; j < elementsPerPoint; j++) {
        output[outOffset + j] = value[j];
      }
    }
  }
}
