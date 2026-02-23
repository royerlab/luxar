//! Luxar WASM Module - Spatial Queries and nD Visibility Computation
//!
//! This module provides high-performance implementations of:
//! - Spatial index chunk queries (bounding box intersection tests)
//! - nD visibility computation for Points, Lines, and GSplats
//! - Effective radius calculation for nD hypersphere slicing
//! - Data decoding (quantized, log-space, LUT)
//! - nD to 3D projection and bounds calculation
//!
//! ## Performance Optimizations
//!
//! - Minimal allocations (fixed-size arrays for common cases)
//! - Cache-friendly access patterns
//! - Zero-copy data transfer via wasm-bindgen
//!
//! ## Dimension Limits
//!
//! **Maximum supported dimensions: 16**
//!
//! Functions that process nD data use fixed-size arrays for performance.
//! If your dataset has more than 16 dimensions, the TypeScript fallback
//! will be used automatically (via automatic fallback detection).
//!
//! ## Module Organization
//!
//! - `spatial` - Chunk AABB queries for spatial indexing
//! - `points` - Point visibility using hypersphere intersection
//! - `lines` - Line segment visibility (endpoint-based)
//! - `gsplats` - Gaussian splat visibility using ellipsoid extent
//! - `effective_radii` - Effective radius calculation for nD slicing
//! - `decode` - Data decoding (quantized, LUT, log-space)
//! - `projection` - nD to 3D projection and bounds

// Shared constants and utilities
pub mod common;

// Declare modules
mod decode;
mod effective_radii;
mod gsplats;
mod gsplats_processing;
mod lines;
mod lines_clipping;
mod points;
mod projection;
mod spatial;

// Re-export all public functions for WASM binding
pub use decode::{
    decode_broadcasted, decode_log_scalar_u16, decode_log_scalar_u8, decode_lut_row_u16,
    decode_lut_row_u8, decode_lut_scalar_u16, decode_lut_scalar_u8, decode_quantized_u16,
    decode_quantized_u8,
};
pub use effective_radii::calculate_effective_radii;
pub use gsplats::compute_nd_visibility_gsplats;
pub use gsplats_processing::{
    compact_attenuated_amplitudes, compute_gsplats_attenuation, extract_cholesky_submatrix,
    extract_visible_cholesky_3d, mahalanobis_distance,
};
pub use lines::compute_nd_visibility_lines;
pub use lines_clipping::{
    calculate_segment_lengths, clip_segment_single, clip_segments_batch,
    interpolate_clipped_positions, interpolate_colors_batch, interpolate_scalars_batch, lerp,
    lerp_vec3, mark_clipped_endpoints, distance_3d,
};
pub use points::compute_nd_visibility_points;
pub use projection::{
    calculate_bounds_3d, compact_by_mask, count_visible, extract_3d_positions,
    radii_to_visibility_mask,
};
pub use spatial::query_chunks_for_view;
