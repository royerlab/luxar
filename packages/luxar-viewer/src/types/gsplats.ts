/**
 * GSplats type definitions for luxar-viewer.
 *
 * These types mirror Python luxar.gsplats and enable type-safe
 * handling of Gaussian splat data throughout the viewer.
 *
 * GSplats are volumetric primitives representing oriented, anisotropic
 * Gaussian density functions with standard Gaussian falloff.
 *
 * @module types/gsplats
 */

import type { BlendingMode } from './blending';
import type { ViewState } from '../data/data-loader-types';
import type { LoaderMetrics, MonitorEventListener, QueryInfo } from './data-monitor-types';

// ============================================================================
// Metadata Types (from zarr .zattrs)
// ============================================================================

/**
 * Range with min and max values.
 */
export interface ValueRange {
  min: number;
  max: number;
}

/**
 * Bounding box with min and max coordinates per dimension.
 */
export interface CoordinateBounds {
  min: number[];
  max: number[];
}

/**
 * GSplats node metadata from zarr .zattrs
 *
 * This interface defines all metadata stored with a GSplats node in the zarr archive.
 * It enables type-safe access to splat properties throughout the viewer.
 */
export interface GSplatsMetadata {
  /** Node type identifier */
  type: 'gsplats';

  /** Total splat count */
  n_splats: number;

  /** Position dimensionality */
  ndim: number;

  /** Whether colors array is present */
  has_colors: boolean;

  /** Whether per-element string labels exist (CSR-encoded, for hover tooltips) */
  has_labels?: boolean;

  /** Whether per-element image labels exist (CSR-encoded, for hover thumbnails) */
  has_image_labels?: boolean;

  /** Elements per chunk */
  chunk_size: number;

  /** Amplitude value range */
  amplitude_range: ValueRange;

  /** Center coordinate bounds */
  center_bounds: CoordinateBounds;

  /**
   * Axis-aligned bounds stamped by the v3.0 writer on every node (a leaf's is
   * its center bounds; a group's is the union of its children). Used for
   * framing/clipping a bare-node file on load. Optional for back-compat with
   * scene leaves that surfaced only `center_bounds`.
   */
  position_bounds?: CoordinateBounds;

  /** Spatial ordering method */
  ordering: 'morton' | 'hilbert' | 'none';

  /** Min bounds for coordinate normalization (when ordering != 'none') */
  ordering_min?: number[];

  /** Max bounds for coordinate normalization (when ordering != 'none') */
  ordering_max?: number[];

  /** Bits per dimension for space-filling curve encoding (when ordering != 'none') */
  ordering_bits_per_dim?: number;

  /** 4x4 transform matrix (column-major for THREE.js) */
  transform?: number[];

  /** Opacity multiplier */
  opacity?: number;

  /** Absorption coefficient κ (volumetric blending mode; identity 1.0) */
  absorption?: number;

  /** Gamma correction */
  gamma?: number;

  /** Intensity (linear color multiplier / gain) */
  intensity?: number;

  /** Offset (additive brightness shift / black level subtraction) */
  offset?: number;

  /** Blending mode */
  blending_mode?: BlendingMode;

  /** Whether this node is exposed as a layer in the Layers panel */
  layer?: boolean;

  /**
   * List of dimension names to extend visibility across.
   * GSplats with extend_to_all will be visible regardless of slice position
   * in the specified dimensions. Allows splats to appear at all values of
   * non-displayed dimensions (e.g., all times, all channels).
   *
   * Example: ["Time", "Channel"] makes splats visible at all time points
   * and all channels regardless of the current slice position.
   */
  extend_to_all?: string[];

  /**
   * Number of additive sub-LODs. When > 1 the splats group contains
   * ``additive_0/`` … ``additive_<N-1>/`` subgroups for progressive
   * (prefix-sum) loading. Absent or 1 means single-LOD (flat) layout.
   */
  n_additive_sublods?: number;

  /** Gaussian truncation radius in standard deviations (default 3.0 if absent). */
  truncation_radius?: number;
}

// ============================================================================
// Spatial Index Types
// ============================================================================

// ============================================================================
// Loaded Data Types
// ============================================================================

/**
 * Splat range for partial loading.
 */
export interface SplatRange {
  /** Start splat index (inclusive) */
  start: number;

  /** End splat index (exclusive) */
  end: number;
}

/**
 * Raw gsplats data loaded from zarr before nD projection.
 *
 * At this stage:
 * - Centers are in full nD space
 * - Cholesky factors are packed (k = ndim * (ndim + 1) / 2 elements per splat)
 * - Attributes may need broadcasting (scalar → per-splat)
 */
export interface LoadedGSplatsData {
  /** Splat positions (N splats * ndim dimensions), flattened row-major */
  positions: Float32Array;

  /** Splat amplitudes (N,) */
  amplitudes: Float32Array;

  /**
   * Packed Cholesky factors (N splats * k elements), flattened row-major.
   * k = ndim * (ndim + 1) / 2
   *
   * Packing order: [L00, L10, L11, L20, L21, L22, ...]
   * Forms lower-triangular L where covariance Σ = L @ Lᵀ
   */
  choleskyFactors: Float32Array;

  /** Splat colors (N * 3) RGB, null if not present
   * Supports Float32Array (HDR), Uint8Array (SDR), or Uint16Array */
  colors: Float32Array | Uint8Array | Uint16Array | null;

  /** Number of splats loaded */
  splatCount: number;

  /** Dimensionality for interpreting centers and cholesky arrays */
  ndim: number;
}

/**
 * Processed gsplats data ready for GPU rendering.
 *
 * After nD → 3D slicing:
 * - Centers are in 3D display space
 * - Cholesky factors are 3D (6 elements per splat)
 * - Amplitudes are attenuated based on distance to hyperplane in hidden dimensions
 * - Per-splat attributes ready for instanced rendering
 */
export interface ProcessedGSplatsData {
  /** Splat centers in 3D display space (M * 3) */
  centers3D: Float32Array;

  /** Attenuated amplitudes (M,), reduced based on nD slice distance */
  amplitudes: Float32Array;

  /**
   * 3D Cholesky factors (M * 6), packed as [L00, L10, L11, L20, L21, L22]
   * Extracted from nD Cholesky by taking display dimension submatrix.
   */
  choleskyFactors3D: Float32Array;

  /** Splat colors RGB (M * 3) */
  colors: Float32Array;

  /** Number of visible splats after nD clipping */
  splatCount: number;
}

// ============================================================================
// Scene Integration Types
// ============================================================================

import type { UpdateSession } from '../profiling/update-profiler';

/**
 * Data loader interface for GSplats nodes.
 *
 * Mirrors the DataLoader interface from points but specialized for gsplats.
 */
export interface GSplatsDataLoader {
  /** Load gsplats data for the given view state */
  loadGSplats(viewState: GSplatsViewState, session?: UpdateSession): Promise<LoadedGSplatsData>;

  /** Update existing data for a new view state (see DataLoader.updateView for `signal`). */
  updateView(
    viewState: GSplatsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedGSplatsData>;

  /** Clean up resources */
  dispose(): void;

  /**
   * Whether this loader has more LOD levels to load for the current view state.
   * Used by the scene loader to schedule refinement passes after the initial commit.
   * Always false for non-progressive (single-LOD) loaders.
   */
  readonly hasMoreLODs?: boolean;

  /**
   * LoaderMonitor surface (optional, for the data-loading-monitor UI).
   * Mirrors the surface that `points-spatial-index-loader.ts` exposes —
   * implementations that don't track metrics may omit these methods.
   */
  addEventListener?(listener: MonitorEventListener): void;
  removeEventListener?(listener: MonitorEventListener): void;
  getMetrics?(): LoaderMetrics;
  getActiveQueries?(): QueryInfo[];
}

/**
 * View state for gsplats loading.
 *
 * Extends the points ViewState pattern with gsplats-specific information.
 */
export type GSplatsViewState = ViewState;

/**
 * User data attached to THREE.Mesh for GSplats in scene.
 *
 * GSplats use THREE.Mesh with InstancedBufferGeometry for rendering
 * oriented quads that are ray-integrated in the fragment shader.
 *
 * Enables runtime type checking and provides access to loader/metadata.
 * Note: Dimension info is NOT stored here - it's only at the Scene level.
 */
export interface GSplatsUserData {
  /** Node type identifier for runtime type checking */
  nodeType: 'gsplats';

  /** Data loader instance */
  loader: GSplatsDataLoader;

  /** Zarr group attributes */
  attrs: GSplatsMetadata;

  /** Currently visible splat count after nD slicing (updated on view change) */
  visibleSplatCount?: number;

  /**
   * The ``SceneLoader._updateVersion`` this mesh's committed geometry was
   * loaded for. Written by ``commitGSplatsGeometry`` on every commit and read
   * by the LOD registry (``lod-group-registry.ts``) to decide whether a level
   * is *fresh for the current view (slice/displayDims) version* — distinct from
   * ``ready`` (geometry committed), since a re-slice overwrites geometry in
   * place without changing readiness. ``undefined`` ⇒ never committed ⇒ treated
   * as stale (the registry falls back to a coarser fresh level, never blank).
   */
  loadedViewVersion?: number;

  /**
   * The ``uTruncate`` value baked into the splat texels currently on the GPU
   * (depth-sorting Phase 4 Stage 2, the append fast path). ``truncate`` is a
   * material uniform, NOT part of the loader's view state, so a change to it is
   * invisible to the prefix-lineage / view-equality checks. The append gate
   * requires ``committedTruncate === readTruncate(mesh)`` so an append never
   * leaves a prefix rendered under the old truncate while the suffix uses the
   * new one. ``undefined`` ⇒ never committed ⇒ append rejected.
   */
  committedTruncate?: number;

  /**
   * True when the GPU splat buffers hold the exact texels of the last commit
   * — i.e. the append fast path may skip re-uploading the prefix. Cleared to
   * ``false`` by ``NodeFactory.rebuildAfterContextRestore`` after a WebGL
   * context loss (the CPU mirror survives but the GPU buffers are gone), which
   * forces the next commit to a full rewrite. Re-enabled by every full commit.
   */
  gpuPrefixIntact?: boolean;

  /** Pick ID assigned by PickingSystem for GPU picking (undefined if picking disabled) */
  pickId?: number;
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Check if metadata is for a GSplats node.
 *
 * @param attrs - Unknown attributes object
 * @returns True if attrs is GSplatsMetadata
 */
export function isGSplatsMetadata(attrs: unknown): attrs is GSplatsMetadata {
  return (
    typeof attrs === 'object' &&
    attrs !== null &&
    (attrs as Record<string, unknown>).type === 'gsplats'
  );
}

/**
 * Check if userData indicates a GSplats object.
 *
 * @param userData - THREE.Object3D userData
 * @returns True if userData is GSplatsUserData
 */
export function isGSplatsUserData(userData: unknown): userData is GSplatsUserData {
  return (
    typeof userData === 'object' &&
    userData !== null &&
    (userData as Record<string, unknown>).nodeType === 'gsplats'
  );
}

// ============================================================================
// Cholesky Utility Types
// ============================================================================

/**
 * Number of packed Cholesky elements for a given dimensionality.
 *
 * @param ndim - Number of dimensions
 * @returns Number of elements in packed lower-triangular form
 */
export function choleskyPackedSize(ndim: number): number {
  return (ndim * (ndim + 1)) / 2;
}

/**
 * Packed-vector positions of the diagonal elements of a d×d lower-triangular
 * matrix, in row-major packing `[L00, L10, L11, L20, L21, L22, …]`.
 *
 * The diagonal element `(i, i)` lives at packed position `(i+1)*(i+2)/2 - 1`.
 * Mirrors `luxar.gsplats.utils.trils.diag_indices` (Python). Used to recombine
 * the v3.1 split Cholesky arrays (`cholesky_factors_diag` +
 * `cholesky_factors_offdiag`) into the packed form the GPU geometry expects.
 *
 * @param ndim - Number of dimensions (d)
 * @returns Diagonal positions, e.g. d=3 → [0, 2, 5]
 */
export function choleskyDiagIndices(ndim: number): number[] {
  const out: number[] = [];
  let pos = 0;
  for (let i = 0; i < ndim; i++) {
    pos += i + 1; // running tril_size(i+1)
    out.push(pos - 1);
  }
  return out;
}

/**
 * Packed-vector positions of the strictly-lower off-diagonal elements —
 * the complement of {@link choleskyDiagIndices} within `range(k)`, preserving
 * row-major order. Empty for `ndim === 1`. Mirrors
 * `luxar.gsplats.utils.trils.offdiag_indices` (Python).
 *
 * @param ndim - Number of dimensions (d)
 * @returns Off-diagonal positions, e.g. d=3 → [1, 3, 4]
 */
export function choleskyOffdiagIndices(ndim: number): number[] {
  const diag = new Set(choleskyDiagIndices(ndim));
  const out: number[] = [];
  for (let i = 0; i < choleskyPackedSize(ndim); i++) {
    if (!diag.has(i)) out.push(i);
  }
  return out;
}

/**
 * Standard packed Cholesky sizes for common dimensions.
 *
 * Extended to 16D (the WASM 16-dimension ceiling — see CLAUDE.md
 * "WASM 16-Dimension Limit"). For arbitrary ndim, prefer calling
 * {@link choleskyPackedSize} directly. The values here are `n*(n+1)/2`.
 */
export const CHOLESKY_SIZES = {
  '1D': 1, // [L00] — degenerate but legal: LoadedGSplatsData permits ndim=1
  '2D': 3, // [L00, L10, L11]
  '3D': 6, // [L00, L10, L11, L20, L21, L22]
  '4D': 10, // [L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]
  '5D': 15,
  '6D': 21,
  '7D': 28,
  '8D': 36,
  '9D': 45,
  '10D': 55,
  '11D': 66,
  '12D': 78,
  '13D': 91,
  '14D': 105,
  '15D': 120,
  '16D': 136,
} as const;
