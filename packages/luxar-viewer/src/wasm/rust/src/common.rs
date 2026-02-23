//! Shared constants and validation utilities for the WASM module.
//!
//! All dimension-limited functions should use these constants and the
//! `validate_ndim` helper to ensure consistent behavior and error messages.

/// Maximum number of dimensions supported by WASM functions.
/// This limit exists because fixed-size arrays are used for performance.
/// If you need more dimensions, the TypeScript fallback is used automatically.
pub const MAX_SUPPORTED_DIMS: usize = 16;

/// Maximum packed Cholesky size for MAX_SUPPORTED_DIMS dimensions.
/// Formula: n * (n + 1) / 2 = 16 * 17 / 2 = 136
pub const MAX_PACKED_CHOLESKY_SIZE: usize = (MAX_SUPPORTED_DIMS * (MAX_SUPPORTED_DIMS + 1)) / 2;

/// Epsilon for degenerate diagonal detection during forward substitution.
pub const CHOLESKY_EPSILON: f32 = 1e-10;

/// Validate that the number of dimensions is within the supported limit.
/// Panics with a clear error message if the limit is exceeded.
#[inline]
pub fn validate_ndim(ndim: usize, function_name: &str) {
    if ndim > MAX_SUPPORTED_DIMS {
        panic!(
            "[WASM] {}: ndim={} exceeds maximum supported dimensions ({}). \
             Luxar WASM functions support up to {} dimensions. \
             For higher dimensions, use TypeScript fallback or reduce dataset dimensionality.",
            function_name, ndim, MAX_SUPPORTED_DIMS, MAX_SUPPORTED_DIMS
        );
    }
}

/// Compute the packed index for a Cholesky element L[row, col].
/// Packed lower-triangular: [L00, L10, L11, L20, L21, L22, ...]
/// Formula: row * (row + 1) / 2 + col (for col <= row)
#[inline]
pub fn packed_index(row: usize, col: usize) -> usize {
    (row * (row + 1)) / 2 + col
}
