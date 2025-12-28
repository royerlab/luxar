//! Data decoding functions for quantized and LUT-encoded arrays.
//!
//! These functions dequantize compressed data formats back to float32.
//! Optimized for large arrays. Note: Manual SIMD was tested but showed no benefit
//! over simple loops when compiled to WASM (see WASM_ANALYSIS.md for benchmarks).

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
    let inv_max = max_log / 65535.0;

    for i in 0..data.len() {
        let normalized = (data[i] as f32) * inv_max;
        output[i] = normalized.exp_m1();
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
    for i in 0..indices.len() {
        output[i] = lut[indices[i] as usize];
    }
}

/// Decode LUT-encoded uint16 indices to float32 (scalar mode).
#[wasm_bindgen]
pub fn decode_lut_scalar_u16(indices: &[u16], lut: &[f32], output: &mut [f32]) {
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
    for i in 0..indices.len() {
        let lut_offset = (indices[i] as usize) * k;
        let out_offset = i * k;
        for j in 0..k {
            output[out_offset + j] = lut[lut_offset + j];
        }
    }
}

/// Decode LUT-encoded uint16 indices to float32 (row mode).
#[wasm_bindgen]
pub fn decode_lut_row_u16(indices: &[u16], lut: &[f32], k: usize, output: &mut [f32]) {
    for i in 0..indices.len() {
        let lut_offset = (indices[i] as usize) * k;
        let out_offset = i * k;
        for j in 0..k {
            output[out_offset + j] = lut[lut_offset + j];
        }
    }
}

/// Broadcast a single value to all points.
///
/// # Arguments
/// * `value` - Value(s) to broadcast [elements_per_point]
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
    for i in 0..num_points {
        let out_offset = i * elements_per_point;
        for j in 0..elements_per_point {
            // Use value[j] if available, otherwise value[0]
            output[out_offset + j] = if j < value.len() { value[j] } else { value[0] };
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
