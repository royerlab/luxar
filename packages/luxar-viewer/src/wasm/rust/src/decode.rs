//! Data decoding functions for quantized and LUT-encoded arrays.
//!
//! These functions dequantize compressed data formats back to float32.
//! Optimized for large arrays. Note: Manual SIMD was tested but showed no benefit
//! over simple loops when compiled to WASM (see WASM_ANALYSIS.md for benchmarks).
//!
//! ## Optimization Status
//! ✅ All functions are already optimally implemented:
//! - No divisions in hot loops (scale precomputed)
//! - Minimal branching
//! - Direct array access with no bounds checks in release mode
//! - LLVM auto-vectorizes these patterns effectively for WASM

use wasm_bindgen::prelude::*;

/// Decode quantized uint8 data to float32.
///
/// Maps uint8 [0, 255] to [min_val, max_val] linearly.
///
/// # Arguments
/// * `data` - Quantized uint8 values
/// * `min_val` - Minimum output value
/// * `max_val` - Maximum output value
/// * `output` - Output float32 buffer (same length as data)
#[wasm_bindgen]
pub fn decode_quantized_u8(data: &[u8], min_val: f32, max_val: f32, output: &mut [f32]) {
    debug_assert!(
        output.len() >= data.len(),
        "output too small: {} < {}",
        output.len(),
        data.len()
    );

    let scale = (max_val - min_val) / 255.0;

    for i in 0..data.len() {
        output[i] = min_val + (data[i] as f32) * scale;
    }
}

/// Decode quantized uint16 data to float32.
///
/// Maps uint16 [0, 65535] to [min_val, max_val] linearly.
///
/// # Arguments
/// * `data` - Quantized uint16 values
/// * `min_val` - Minimum output value
/// * `max_val` - Maximum output value
/// * `output` - Output float32 buffer (same length as data)
#[wasm_bindgen]
pub fn decode_quantized_u16(data: &[u16], min_val: f32, max_val: f32, output: &mut [f32]) {
    debug_assert!(
        output.len() >= data.len(),
        "output too small: {} < {}",
        output.len(),
        data.len()
    );

    let scale = (max_val - min_val) / 65535.0;

    for i in 0..data.len() {
        output[i] = min_val + (data[i] as f32) * scale;
    }
}

/// Decode log-space quantized uint8 data to float32.
///
/// Used for positive scalars with wide dynamic range (e.g., radii).
/// Decoding: expm1(normalized * max_log)
///
/// # Arguments
/// * `data` - Quantized uint8 values
/// * `max_log` - Maximum log value (from encoding)
/// * `output` - Output float32 buffer
#[wasm_bindgen]
pub fn decode_log_scalar_u8(data: &[u8], max_log: f32, output: &mut [f32]) {
    debug_assert!(
        output.len() >= data.len(),
        "output too small: {} < {}",
        output.len(),
        data.len()
    );

    let inv_max = max_log / 255.0;

    for i in 0..data.len() {
        let normalized = (data[i] as f32) * inv_max;
        // expm1(x) = e^x - 1, more accurate than exp(x) - 1 for small x
        output[i] = normalized.exp_m1();
    }
}

/// Decode log-space quantized uint16 data to float32.
#[wasm_bindgen]
pub fn decode_log_scalar_u16(data: &[u16], max_log: f32, output: &mut [f32]) {
    debug_assert!(
        output.len() >= data.len(),
        "output too small: {} < {}",
        output.len(),
        data.len()
    );

    let inv_max = max_log / 65535.0;

    for i in 0..data.len() {
        let normalized = (data[i] as f32) * inv_max;
        output[i] = normalized.exp_m1();
    }
}

/// Decode geometric-log quantized uint8 data to float32.
///
/// Min/max-anchored true-log quantization with a RESERVED ZERO LEVEL:
/// level 0 decodes to exactly 0.0; levels [1, 255] decode to
/// `exp(min_log + (u - 1)/254 * (max_log - min_log))` — uniform relative
/// precision over the array's own nonzero range. Mirrors the Python
/// `_decode_geolog_scalar` and the TS reference exactly.
#[wasm_bindgen]
pub fn decode_geolog_scalar_u8(data: &[u8], min_log: f32, max_log: f32, output: &mut [f32]) {
    debug_assert!(
        output.len() >= data.len(),
        "output too small: {} < {}",
        output.len(),
        data.len()
    );

    // f64 internals: matches the Python decoder and the TS reference (JS
    // number math) so all three backends agree to the last f32 ULP even at
    // the top of a 7-decade range.
    let min_log = min_log as f64;
    let inv = ((max_log as f64) - min_log).max(0.0) / 254.0;

    for i in 0..data.len() {
        let u = data[i];
        output[i] = if u == 0 {
            0.0
        } else {
            (min_log + ((u - 1) as f64) * inv).exp() as f32
        };
    }
}

/// Decode geometric-log quantized uint16 data to float32.
#[wasm_bindgen]
pub fn decode_geolog_scalar_u16(data: &[u16], min_log: f32, max_log: f32, output: &mut [f32]) {
    debug_assert!(
        output.len() >= data.len(),
        "output too small: {} < {}",
        output.len(),
        data.len()
    );

    let min_log = min_log as f64;
    let inv = ((max_log as f64) - min_log).max(0.0) / 65534.0;

    for i in 0..data.len() {
        let u = data[i];
        output[i] = if u == 0 {
            0.0
        } else {
            (min_log + ((u - 1) as f64) * inv).exp() as f32
        };
    }
}

/// Decode LUT-encoded uint8 indices to float32 (scalar mode).
///
/// Each index maps to a single float value from the LUT.
///
/// # Arguments
/// * `indices` - uint8 indices into the LUT
/// * `lut` - Lookup table of float values
/// * `output` - Output float32 buffer (same length as indices)
#[wasm_bindgen]
pub fn decode_lut_scalar_u8(indices: &[u8], lut: &[f32], output: &mut [f32]) {
    debug_assert!(
        output.len() >= indices.len(),
        "output too small: {} < {}",
        output.len(),
        indices.len()
    );

    for i in 0..indices.len() {
        output[i] = lut[indices[i] as usize];
    }
}

/// Decode LUT-encoded uint16 indices to float32 (scalar mode).
#[wasm_bindgen]
pub fn decode_lut_scalar_u16(indices: &[u16], lut: &[f32], output: &mut [f32]) {
    debug_assert!(
        output.len() >= indices.len(),
        "output too small: {} < {}",
        output.len(),
        indices.len()
    );

    for i in 0..indices.len() {
        output[i] = lut[indices[i] as usize];
    }
}

/// Decode LUT-encoded uint8 indices to float32 (row mode).
///
/// Each index maps to k consecutive float values from the LUT.
/// Used for vector attributes like positions (xyz) or colors (rgb).
///
/// # Arguments
/// * `indices` - uint8 indices into the LUT
/// * `lut` - Lookup table of float values [numEntries * k]
/// * `k` - Number of values per LUT entry
/// * `output` - Output float32 buffer [indices.len() * k]
#[wasm_bindgen]
pub fn decode_lut_row_u8(indices: &[u8], lut: &[f32], k: usize, output: &mut [f32]) {
    debug_assert!(
        output.len() >= indices.len() * k,
        "output too small: {} < {}",
        output.len(),
        indices.len() * k
    );

    for i in 0..indices.len() {
        let lut_offset = (indices[i] as usize) * k;
        let out_offset = i * k;
        output[out_offset..out_offset + k].copy_from_slice(&lut[lut_offset..lut_offset + k]);
    }
}

/// Decode LUT-encoded uint16 indices to float32 (row mode).
#[wasm_bindgen]
pub fn decode_lut_row_u16(indices: &[u16], lut: &[f32], k: usize, output: &mut [f32]) {
    debug_assert!(
        output.len() >= indices.len() * k,
        "output too small: {} < {}",
        output.len(),
        indices.len() * k
    );

    for i in 0..indices.len() {
        let lut_offset = (indices[i] as usize) * k;
        let out_offset = i * k;
        output[out_offset..out_offset + k].copy_from_slice(&lut[lut_offset..lut_offset + k]);
    }
}

/// Broadcast a value to all points.
///
/// Strict contract (mirrors `wasm/typescript/decode.ts::decode_broadcasted`):
/// `value.len()` must be either `1` (a scalar broadcast to every slot) or
/// exactly `elements_per_point` (a per-element vector replicated across every
/// point). Any other length is INVALID and rejected upstream by the worker
/// wrapper (`workers/data-worker/decode/broadcasted.ts`) with a catchable error
/// before this kernel is reached. The old "mixed broadcast" pad/truncate
/// fallback (e.g. `value=[v0,v1], epp=3 -> [v0,v1,v0]`) was removed because it
/// silently diverged from the TS reference.
///
/// # Arguments
/// * `value` - `1` value (scalar) or `elements_per_point` values (vector)
/// * `num_points` - Number of points
/// * `elements_per_point` - Elements per point (1 for scalar, 3 for vec3, etc.)
/// * `output` - Output buffer [num_points * elements_per_point]
#[wasm_bindgen]
pub fn decode_broadcasted(
    value: &[f32],
    num_points: usize,
    elements_per_point: usize,
    output: &mut [f32],
) {
    debug_assert!(
        output.len() >= num_points * elements_per_point,
        "output too small: {} < {}",
        output.len(),
        num_points * elements_per_point
    );

    if value.len() == 1 {
        // Scalar broadcast: fill every slot with the single value.
        let v = value[0];
        for slot in output[..num_points * elements_per_point].iter_mut() {
            *slot = v;
        }
    } else {
        // Per-element vector: value.len() == elements_per_point (contract).
        for i in 0..num_points {
            let out_offset = i * elements_per_point;
            output[out_offset..out_offset + elements_per_point]
                .copy_from_slice(&value[..elements_per_point]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_quantized_u8() {
        let data = vec![0u8, 128, 255];
        let mut output = vec![0.0f32; 3];

        decode_quantized_u8(&data, 0.0, 10.0, &mut output);

        assert!((output[0] - 0.0).abs() < 0.01);
        assert!((output[1] - 5.02).abs() < 0.1); // 128/255 * 10 ≈ 5.02
        assert!((output[2] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_quantized_u16() {
        let data = vec![0u16, 32768, 65535];
        let mut output = vec![0.0f32; 3];

        decode_quantized_u16(&data, -1.0, 1.0, &mut output);

        assert!((output[0] - (-1.0)).abs() < 0.01);
        assert!((output[1] - 0.0).abs() < 0.01); // ~32768/65535 * 2 - 1 ≈ 0
        assert!((output[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_geolog_scalar_u8() {
        // level 0 = reserved exact zero; level 1 = exp(min_log); top = exp(max_log)
        let data = vec![0u8, 1, 255];
        let (min_log, max_log) = (-2.0f32, 3.0f32);
        let mut output = vec![0.0f32; 3];
        decode_geolog_scalar_u8(&data, min_log, max_log, &mut output);
        assert_eq!(output[0], 0.0);
        assert!((output[1] - (-2.0f32).exp()).abs() < 1e-6);
        assert!((output[2] - 3.0f32.exp()).abs() < 1e-4);
    }

    #[test]
    fn test_decode_geolog_scalar_u16() {
        let data = vec![0u16, 1, 65535];
        let (min_log, max_log) = (-7.5f32, 9.9f32);
        let mut output = vec![0.0f32; 3];
        decode_geolog_scalar_u16(&data, min_log, max_log, &mut output);
        assert_eq!(output[0], 0.0);
        assert!((output[1] - (-7.5f32).exp()).abs() < 1e-9);
        assert!((output[2] - 9.9f32.exp()).abs() < 10.0); // ~2e4, f32 rel tol
    }

    #[test]
    fn test_decode_log_scalar_u8() {
        let data = vec![0u8, 128, 255];
        let max_log = 5.0f32; // log1p(~147)
        let mut output = vec![0.0f32; 3];

        decode_log_scalar_u8(&data, max_log, &mut output);

        // 0 -> expm1(0) = 0
        assert!(output[0].abs() < 0.01);
        // 128 -> expm1(128/255 * 5) ≈ expm1(2.5) ≈ 11.2
        assert!(output[1] > 10.0 && output[1] < 13.0);
        // 255 -> expm1(5) ≈ 147.4
        assert!(output[2] > 145.0 && output[2] < 150.0);
    }

    #[test]
    fn test_decode_lut_scalar_u8() {
        let indices = vec![0u8, 2, 1];
        let lut = vec![1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];

        decode_lut_scalar_u8(&indices, &lut, &mut output);

        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 3.0);
        assert_eq!(output[2], 2.0);
    }

    #[test]
    fn test_decode_lut_row_u8() {
        let indices = vec![0u8, 1];
        // LUT with 2 entries, each with 3 values (rgb)
        let lut = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // red, green
        let mut output = vec![0.0f32; 6];

        decode_lut_row_u8(&indices, &lut, 3, &mut output);

        assert_eq!(output[0], 1.0); // red.r
        assert_eq!(output[1], 0.0); // red.g
        assert_eq!(output[2], 0.0); // red.b
        assert_eq!(output[3], 0.0); // green.r
        assert_eq!(output[4], 1.0); // green.g
        assert_eq!(output[5], 0.0); // green.b
    }

    #[test]
    fn test_decode_lut_row_u16() {
        let indices = vec![0u16, 2, 1];
        // LUT with 3 entries, each with 2 values (xy)
        let lut = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // (1,2), (3,4), (5,6)
        let mut output = vec![0.0f32; 6];

        decode_lut_row_u16(&indices, &lut, 2, &mut output);

        // Index 0 → (1, 2)
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 2.0);
        // Index 2 → (5, 6)
        assert_eq!(output[2], 5.0);
        assert_eq!(output[3], 6.0);
        // Index 1 → (3, 4)
        assert_eq!(output[4], 3.0);
        assert_eq!(output[5], 4.0);
    }

    #[test]
    fn test_decode_broadcasted_scalar() {
        // Scalar broadcast (value.len() == 1): the single value fills every
        // slot of every point, regardless of elements_per_point.
        let value = vec![0.5f32]; // scalar
        let mut output = vec![0.0f32; 6]; // 2 points * 3 elements

        decode_broadcasted(&value, 2, 3, &mut output);

        for i in 0..6 {
            assert_eq!(output[i], 0.5, "output[{}] should be 0.5", i);
        }
    }

    #[test]
    fn test_decode_broadcasted() {
        let value = vec![0.5f32, 0.6, 0.7]; // rgb
        let mut output = vec![0.0f32; 9]; // 3 points * 3 elements

        decode_broadcasted(&value, 3, 3, &mut output);

        // All 3 points should have the same color
        for i in 0..3 {
            assert_eq!(output[i * 3], 0.5);
            assert_eq!(output[i * 3 + 1], 0.6);
            assert_eq!(output[i * 3 + 2], 0.7);
        }
    }
}
