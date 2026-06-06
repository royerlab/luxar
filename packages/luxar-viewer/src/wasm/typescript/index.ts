/**
 * TypeScript reference implementations for WASM functions.
 *
 * These implementations mirror the Rust WASM module exactly and serve as:
 * 1. Reference implementations for testing
 * 2. Fallback when WASM is unavailable
 * 3. Documentation of expected behavior
 *
 * Structure mirrors src/wasm/rust/src/:
 * - spatial.ts     -> spatial.rs
 * - points.ts      -> points.rs
 * - lines.ts       -> lines.rs
 * - gsplats.ts     -> gsplats.rs
 * - effective-radii.ts -> effective_radii.rs
 * - decode.ts      -> decode.rs
 * - lines-clipping.ts -> lines_clipping.rs
 */

export { query_chunks_for_view } from './spatial';
export { compute_nd_visibility_points } from './points';
export { compute_nd_visibility_lines } from './lines';
export { compute_nd_visibility_gsplats } from './gsplats';
export { calculate_effective_radii } from './effective-radii';
export {
  decode_quantized_u8,
  decode_quantized_u16,
  decode_log_scalar_u8,
  decode_log_scalar_u16,
  decode_lut_scalar_u8,
  decode_lut_scalar_u16,
  decode_lut_row_u8,
  decode_lut_row_u16,
  decode_broadcasted,
} from './decode';
export {
  extract_3d_positions,
  calculate_bounds_3d,
  compact_by_mask,
  count_visible,
  radii_to_visibility_mask,
} from './projection';
export {
  mahalanobis_distance,
  extract_cholesky_submatrix,
  computeMarginalCholesky,
  compute_gsplats_attenuation,
  extract_visible_cholesky_3d,
  compact_attenuated_amplitudes,
  project_gsplats_nd_to_3d,
} from './gsplats-processing';
export {
  clip_segment_single,
  clip_segments_batch,
  interpolate_clipped_positions,
  lerp,
  lerp_vec3,
  distance_3d,
  interpolate_scalars_batch,
  interpolate_colors_batch,
  calculate_segment_lengths,
  mark_clipped_endpoints,
} from './lines-clipping';

// Re-export as a module class for compatibility with WasmModule interface
import type { WasmModule } from '../types';
import { query_chunks_for_view } from './spatial';
import { compute_nd_visibility_points } from './points';
import { compute_nd_visibility_lines } from './lines';
import { compute_nd_visibility_gsplats } from './gsplats';
import { calculate_effective_radii } from './effective-radii';
import {
  decode_quantized_u8,
  decode_quantized_u16,
  decode_log_scalar_u8,
  decode_log_scalar_u16,
  decode_lut_scalar_u8,
  decode_lut_scalar_u16,
  decode_lut_row_u8,
  decode_lut_row_u16,
  decode_broadcasted,
} from './decode';
import {
  extract_3d_positions,
  calculate_bounds_3d,
  compact_by_mask,
  count_visible,
  radii_to_visibility_mask,
} from './projection';
import {
  mahalanobis_distance,
  extract_cholesky_submatrix,
  compute_gsplats_attenuation,
  extract_visible_cholesky_3d,
  compact_attenuated_amplitudes,
  project_gsplats_nd_to_3d,
} from './gsplats-processing';
import {
  clip_segment_single,
  clip_segments_batch,
  interpolate_clipped_positions,
  lerp,
  lerp_vec3,
  distance_3d,
  interpolate_scalars_batch,
  interpolate_colors_batch,
  calculate_segment_lengths,
  mark_clipped_endpoints,
} from './lines-clipping';

/**
 * TypeScript fallback class implementing WasmModule interface.
 * Used when WASM is unavailable or fails to load.
 */
export class TypeScriptFallback implements WasmModule {
  query_chunks_for_view = query_chunks_for_view;
  compute_nd_visibility_points = compute_nd_visibility_points;
  compute_nd_visibility_lines = compute_nd_visibility_lines;
  compute_nd_visibility_gsplats = compute_nd_visibility_gsplats;
  calculate_effective_radii = calculate_effective_radii;
  decode_quantized_u8 = decode_quantized_u8;
  decode_quantized_u16 = decode_quantized_u16;
  decode_log_scalar_u8 = decode_log_scalar_u8;
  decode_log_scalar_u16 = decode_log_scalar_u16;
  decode_lut_scalar_u8 = decode_lut_scalar_u8;
  decode_lut_scalar_u16 = decode_lut_scalar_u16;
  decode_lut_row_u8 = decode_lut_row_u8;
  decode_lut_row_u16 = decode_lut_row_u16;
  decode_broadcasted = decode_broadcasted;
  extract_3d_positions = extract_3d_positions;
  calculate_bounds_3d = calculate_bounds_3d;
  compact_by_mask = compact_by_mask;
  count_visible = count_visible;
  radii_to_visibility_mask = radii_to_visibility_mask;
  mahalanobis_distance = mahalanobis_distance;
  extract_cholesky_submatrix = extract_cholesky_submatrix;
  compute_gsplats_attenuation = compute_gsplats_attenuation;
  extract_visible_cholesky_3d = extract_visible_cholesky_3d;
  compact_attenuated_amplitudes = compact_attenuated_amplitudes;
  project_gsplats_nd_to_3d = project_gsplats_nd_to_3d;
  clip_segment_single = clip_segment_single;
  clip_segments_batch = clip_segments_batch;
  interpolate_clipped_positions = interpolate_clipped_positions;
  lerp = lerp;
  lerp_vec3 = lerp_vec3;
  distance_3d = distance_3d;
  interpolate_scalars_batch = interpolate_scalars_batch;
  interpolate_colors_batch = interpolate_colors_batch;
  calculate_segment_lengths = calculate_segment_lengths;
  mark_clipped_endpoints = mark_clipped_endpoints;
}
