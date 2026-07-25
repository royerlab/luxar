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
//! ## Dimension Limits (and the >16D fallback)
//!
//! **WASM fast path: up to 16 dimensions.**
//!
//! Functions that process nD data use fixed-size arrays for performance and call
//! `common::validate_ndim`, which panics (the crate is built `panic = "abort"`)
//! for `ndim > 16`. These kernels must therefore never be invoked above 16D.
//!
//! **>16D is still fully supported** — automatically and transparently. The
//! TypeScript reference implementations in `wasm/typescript/` are uncapped and
//! handle arbitrary ndim, and the worker's `pickBackend(ctx, ndim)`
//! (`workers/data-worker/state.ts`) routes any `ndim > 16` operation to that TS
//! backend instead of the WASM kernel. So high-dimensional datasets run on the
//! (slower but correct) TS path rather than being rejected — keep both backends
//! in sync so this fallback stays a faithful substitute.
//!
//! ## Module Organization
//!
//! - `effective_radii` - Effective radius calculation for nD slicing
//! - `decode` - Data decoding (quantized, LUT, log-space)
//! - `projection` - nD to 3D projection and bounds
//! - `gsplats_processing` - GSplat nD->3D projection, attenuation, Cholesky
//! - `lines_clipping` - Line segment clipping for nD slicing
//! - `depth_sort` - Back-to-front splat ordering (depth-sorting Phase 2)

// Clippy lint configuration for this numerical WASM crate.
//
// - `too_many_arguments`: the `#[wasm_bindgen]` kernels take many flat
//   slice/scalar parameters (input buffers + pre-allocated output buffers +
//   shape scalars). wasm-bindgen exports cannot accept aggregate structs for
//   the zero-copy typed-array contract, so collapsing them is not an option.
// - `needless_range_loop`: the kernels are in-place numerical algorithms
//   (forward/back substitution, Cholesky factorization, mask compaction over
//   parallel arrays). Index loops are the natural — and often the only
//   borrow-checker-valid — form (e.g. reading `y[k]` while writing `y[i]`),
//   and indexing by a shared `i` across several arrays is clearer than
//   zipping. The iterator rewrites clippy suggests are infeasible or less
//   readable here.
//
// `manual_memcpy` is intentionally NOT allowed — that lint flags genuinely
// improvable slice copies (use `copy_from_slice`), which we fix in place.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

// Shared constants and utilities
pub mod common;

// Declare modules
mod decode;
mod depth_sort;
mod effective_radii;
mod gsplats_processing;
mod lines_clipping;
mod projection;

// Re-export all public functions for WASM binding
pub use decode::{
    decode_broadcasted, decode_geolog_perchannel_u16, decode_geolog_perchannel_u8,
    decode_geolog_scalar_u16, decode_geolog_scalar_u8, decode_linear_perchannel_u16,
    decode_linear_perchannel_u8, decode_log_perchannel_u16, decode_log_perchannel_u8,
    decode_log_scalar_u16, decode_log_scalar_u8, decode_lut_row_u16, decode_lut_row_u8,
    decode_lut_scalar_u16, decode_lut_scalar_u8, decode_quantized_u16, decode_quantized_u8,
    decode_signed_log_perchannel_u16, decode_signed_log_perchannel_u8,
};
pub use depth_sort::{create_depth_sorter, sort_splats_by_depth, DepthSorter};
pub use effective_radii::calculate_effective_radii;
pub use gsplats_processing::{
    compact_attenuated_amplitudes, compute_gsplats_attenuation, extract_cholesky_submatrix,
    extract_visible_cholesky_3d, mahalanobis_distance, project_gsplats_nd_to_3d,
};
pub use lines_clipping::{
    calculate_segment_lengths, clip_segment_single, clip_segments_batch, distance_3d,
    interpolate_clipped_positions, interpolate_colors_batch, interpolate_scalars_batch, lerp,
    lerp_vec3, mark_clipped_endpoints,
};
pub use projection::{
    calculate_bounds_3d, compact_by_mask, count_visible, extract_3d_positions,
    radii_to_visibility_mask,
};
