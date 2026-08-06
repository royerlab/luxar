/**
 * TypeScript reference implementations for WASM functions.
 *
 * These implementations mirror the Rust WASM module exactly and serve as:
 * 1. Reference implementations for testing
 * 2. Fallback when WASM is unavailable
 * 3. Documentation of expected behavior
 *
 * Structure mirrors src/wasm/rust/src/:
 * - effective-radii.ts -> effective_radii.rs
 * - decode.ts      -> decode.rs
 * - lines-clipping.ts -> lines_clipping.rs
 * - mesh-culling.ts -> mesh_culling.rs
 * - depth-sort.ts  -> depth_sort.rs
 */

export { calculate_effective_radii } from './effective-radii';
export { sort_splats_by_depth } from './depth-sort';
export {
  decode_quantized_u8,
  decode_quantized_u16,
  decode_log_scalar_u8,
  decode_log_scalar_u16,
  decode_geolog_scalar_u8,
  decode_geolog_scalar_u16,
  decode_linear_perchannel_u8,
  decode_linear_perchannel_u16,
  decode_log_perchannel_u8,
  decode_log_perchannel_u16,
  decode_signed_log_perchannel_u8,
  decode_signed_log_perchannel_u16,
  decode_geolog_perchannel_u8,
  decode_geolog_perchannel_u16,
  decode_lut_scalar_u8,
  decode_lut_scalar_u16,
  decode_lut_row_u8,
  decode_lut_row_u16,
  decode_broadcasted,
} from './decode';
export { extract_3d_positions } from './projection';
export {
  mahalanobis_distance,
  /** Exposed for the parity tests; the Rust twin is private. */
  computeMarginalCholesky,
  project_gsplats_nd_to_3d,
} from './gsplats-processing';
export {
  clip_segment_single,
  clip_segments_batch,
  interpolate_clipped_positions,
  interpolate_scalars_batch,
  interpolate_colors_batch,
  calculate_segment_lengths,
  compute_joint_codes,
  JOINT_FREE_END,
  JOINT_CLIPPED,
  JOINT_HUB,
} from './lines-clipping';
export { mesh_vertex_visibility_mask, compact_visible_faces } from './mesh-culling';

// Re-export as a module class for compatibility with WasmModule interface
import type { WasmModule } from '../types';
import { calculate_effective_radii } from './effective-radii';
import { sort_splats_by_depth } from './depth-sort';
import {
  decode_quantized_u8,
  decode_quantized_u16,
  decode_log_scalar_u8,
  decode_log_scalar_u16,
  decode_geolog_scalar_u8,
  decode_geolog_scalar_u16,
  decode_linear_perchannel_u8,
  decode_linear_perchannel_u16,
  decode_log_perchannel_u8,
  decode_log_perchannel_u16,
  decode_signed_log_perchannel_u8,
  decode_signed_log_perchannel_u16,
  decode_geolog_perchannel_u8,
  decode_geolog_perchannel_u16,
  decode_lut_scalar_u8,
  decode_lut_scalar_u16,
  decode_lut_row_u8,
  decode_lut_row_u16,
  decode_broadcasted,
} from './decode';
import { extract_3d_positions } from './projection';
import { mahalanobis_distance, project_gsplats_nd_to_3d } from './gsplats-processing';
import {
  clip_segment_single,
  clip_segments_batch,
  interpolate_clipped_positions,
  interpolate_scalars_batch,
  interpolate_colors_batch,
  calculate_segment_lengths,
  compute_joint_codes,
} from './lines-clipping';
import { mesh_vertex_visibility_mask, compact_visible_faces } from './mesh-culling';

/**
 * TypeScript fallback class implementing WasmModule interface.
 * Used when WASM is unavailable or fails to load.
 */
export class TypeScriptFallback implements WasmModule {
  calculate_effective_radii = calculate_effective_radii;
  sort_splats_by_depth = sort_splats_by_depth;
  decode_quantized_u8 = decode_quantized_u8;
  decode_quantized_u16 = decode_quantized_u16;
  decode_log_scalar_u8 = decode_log_scalar_u8;
  decode_geolog_scalar_u8 = decode_geolog_scalar_u8;
  decode_geolog_scalar_u16 = decode_geolog_scalar_u16;
  decode_linear_perchannel_u8 = decode_linear_perchannel_u8;
  decode_linear_perchannel_u16 = decode_linear_perchannel_u16;
  decode_log_perchannel_u8 = decode_log_perchannel_u8;
  decode_log_perchannel_u16 = decode_log_perchannel_u16;
  decode_signed_log_perchannel_u8 = decode_signed_log_perchannel_u8;
  decode_signed_log_perchannel_u16 = decode_signed_log_perchannel_u16;
  decode_geolog_perchannel_u8 = decode_geolog_perchannel_u8;
  decode_geolog_perchannel_u16 = decode_geolog_perchannel_u16;
  decode_log_scalar_u16 = decode_log_scalar_u16;
  decode_lut_scalar_u8 = decode_lut_scalar_u8;
  decode_lut_scalar_u16 = decode_lut_scalar_u16;
  decode_lut_row_u8 = decode_lut_row_u8;
  decode_lut_row_u16 = decode_lut_row_u16;
  decode_broadcasted = decode_broadcasted;
  extract_3d_positions = extract_3d_positions;
  mahalanobis_distance = mahalanobis_distance;
  project_gsplats_nd_to_3d = project_gsplats_nd_to_3d;
  clip_segment_single = clip_segment_single;
  clip_segments_batch = clip_segments_batch;
  interpolate_clipped_positions = interpolate_clipped_positions;
  interpolate_scalars_batch = interpolate_scalars_batch;
  interpolate_colors_batch = interpolate_colors_batch;
  calculate_segment_lengths = calculate_segment_lengths;
  compute_joint_codes = compute_joint_codes;
  mesh_vertex_visibility_mask = mesh_vertex_visibility_mask;
  compact_visible_faces = compact_visible_faces;
}
