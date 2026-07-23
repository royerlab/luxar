/**
 * Lines type definitions for luxar-viewer.
 *
 * These types mirror Python luxar.core.Lines and enable type-safe
 * handling of line data throughout the viewer.
 *
 * @module types/lines
 */

import type { BlendingMode } from './blending';
import type { ViewState } from '../data/data-loader-types';
import type { ScalarArray } from './points';
import type { LoaderMetrics, MonitorEventListener, QueryInfo } from './data-monitor-types';

// ============================================================================
// Metadata Types (from zarr .zattrs)
// ============================================================================

/**
 * Ordering metadata structure (shared between vertices and segments)
 *
 * Lines use dual spatial indexing:
 * - Vertices: Ordered in D-dimensional space using space-filling curves
 * - Segments: Ordered in (2×D)-dimensional space (start+end concatenated)
 */
export interface OrderingMetadata {
  /** Discrete dimension indices for compound ordering */
  slice_dims: number[];

  /** Spatial dimension indices for curve ordering */
  ordering_dims: number[];

  /** Min bounds for coordinate normalization */
  ordering_min: number[];

  /** Max bounds for coordinate normalization */
  ordering_max: number[];

  /** Elements per chunk */
  chunk_size: number;

  /** Bits per dimension for space-filling curve encoding */
  ordering_bits_per_dim?: number;
}

/**
 * Original line type from Python API.
 *
 * - 'segments': Independent line segments (pairs of vertices)
 * - 'polyline': Connected vertices forming a continuous line
 * - 'loop': Polyline with first and last vertex connected
 * - 'indexed': Explicit vertex/segment indices (most flexible)
 */
export type LineType = 'segments' | 'polyline' | 'loop' | 'indexed';

/**
 * Lines node metadata from zarr .zattrs
 *
 * This interface defines all metadata stored with a Lines node in the zarr archive.
 * It enables type-safe access to line properties throughout the viewer.
 */
export interface LinesMetadata {
  /** Node type identifier */
  type: 'lines';

  /** Total vertex count */
  n_vertices: number;

  /** Total segment count */
  n_segments: number;

  /** Vertex position dimensionality */
  ndim: number;

  /** Original line type from Python API */
  original_line_type: LineType;

  /** Maximum line width in world units */
  max_width: number;

  /** Whether colors array is present */
  has_colors: boolean;

  /** Whether sharpness array is present */
  has_sharpness: boolean;

  /**
   * Whether per-vertex scalar values for colormap lookup exist.
   *
   * gates whether the loader opens the `scalars` zarr array and
   * whether `NodeFactory.createLinesNode` enables `USE_COLORMAP`.
   */
  has_scalars?: boolean;

  /**
   * Named colormap or 'custom' (paired with `colormap_lut` zarr array).
   * viewer reads this to choose the LUT texture; when set without
   * `has_scalars`, the colormap path is suppressed (fail-closed).
   */
  colormap?: string;

  /**
   * `[min, max]` for normalising scalars before LUT lookup.
   * passed to `material.updateScalarRange` so a value of `min`
   * samples LUT index 0 and `max` samples 255.
   */
  scalar_data_range?: [number, number];

  /** Whether spatial index exists (redundant with ordering !== 'none', but explicit) */
  has_spatial_index?: boolean;

  /** Whether per-element string labels exist (CSR-encoded, for hover tooltips) */
  has_labels?: boolean;

  /** Whether per-element image labels exist (CSR-encoded, for hover thumbnails) */
  has_image_labels?: boolean;

  /** Spatial ordering method */
  ordering: 'morton' | 'hilbert' | 'none';

  /** Vertex ordering metadata (when ordering != 'none') */
  vertex_ordering?: OrderingMetadata;

  /** Segment ordering metadata (when ordering != 'none') */
  segment_ordering?: OrderingMetadata;

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
   * Lines with extend_to_all will be visible regardless of slice position
   * in the specified dimensions (e.g., ['time'] makes geometry visible
   * at all time values).
   */
  extend_to_all?: string[];
}

// ============================================================================
// Loaded Data Types
// ============================================================================

/**
 * Segment range for partial loading (mirrors PointRange pattern)
 */
export interface SegmentRange {
  /** Start segment index (inclusive) */
  start: number;

  /** End segment index (exclusive) */
  end: number;
}

/**
 * Raw lines data loaded from zarr before nD projection.
 *
 * At this stage:
 * - Vertices are in full nD space
 * - Segments use LOCAL indices into the loaded vertex arrays
 * - Attributes may need broadcasting (scalar → per-vertex)
 */
export interface LoadedLinesData {
  /** Vertex positions (N vertices * ndim dimensions), flattened row-major */
  positions: Float32Array;

  /** Segment index pairs (M segments * 2) - local indices into positions array */
  segments: Uint32Array;

  /** Vertex widths (N,) or (1,) if broadcast */
  widths: Float32Array;

  /** Vertex colors (N * 3) RGB, null if not present
   * Supports Float32Array (HDR), Uint8Array (SDR), or Uint16Array */
  colors: Float32Array | Uint8Array | Uint16Array | null;

  /** Vertex sharpness (N,) null if not present */
  sharpness: Float32Array | null;

  /**
   * Per-vertex scalar values for colormap lookup (N,). Optional —
   * when omitted (undefined), the line shader's `USE_COLORMAP` path
   * stays inactive and `ProcessedLinesData.startScalars`/`endScalars`
   * are also undefined.
   *
   * Aligned with `LoadedPointsData.scalars?: ScalarArray` (both
   * optional + undefined absence) so consumer truthy-checks read
   * uniformly across node types.
   *
   * Dtype aligned with `ScalarArray` (Float32/Float16/Uint8) so
   * Uint8 loaders can keep native dtype through the accumulator and
   * pay a single widening at projection time rather than 4× memory up
   * front.
   */
  scalars?: ScalarArray;

  /** Number of segments loaded */
  segmentCount: number;

  /** Number of vertices loaded */
  vertexCount: number;

  /** Dimensionality for interpreting vertices array */
  ndim: number;
}

/**
 * Processed lines data ready for GPU rendering.
 *
 * After nD clipping and 3D projection:
 * - Positions are in 3D display space
 * - Per-segment attributes ready for instanced rendering
 * - Clipped endpoints have interpolated attributes
 */
export interface ProcessedLinesData {
  /** Segment start positions in 3D display space (M * 3) */
  startPositions: Float32Array;

  /** Segment end positions in 3D display space (M * 3) */
  endPositions: Float32Array;

  /** Start vertex colors RGB (M * 3), interpolated if clipped */
  startColors: Float32Array;

  /** End vertex colors RGB (M * 3), interpolated if clipped */
  endColors: Float32Array;

  /** Start vertex widths (M,), interpolated if clipped */
  startWidths: Float32Array;

  /** End vertex widths (M,), interpolated if clipped */
  endWidths: Float32Array;

  /** Start vertex sharpness (M,), interpolated if clipped */
  startSharpness: Float32Array;

  /** End vertex sharpness (M,), interpolated if clipped */
  endSharpness: Float32Array;

  /** 3D segment lengths (M,), used for cap factor calculation */
  segmentLengths: Float32Array;

  /** Whether start endpoint was clipped (M,), 1=clipped, 0=original */
  startClipped: Uint8Array;

  /** Whether end endpoint was clipped (M,), 1=clipped, 0=original */
  endClipped: Uint8Array;

  /**
   * Start-vertex scalar values (M,), interpolated if clipped, optional.
   *
   * present when the source `LoadedLinesData.scalars` is non-null.
   * Written into texel5.x of the line texture and used by the line
   * vertex shader under `USE_COLORMAP` (presence rides the
   * `userData.hasScalars` stamp — the fixed 6-texel layout always has
   * the slot).
   */
  startScalars?: Float32Array;

  /**
   * End-vertex scalar values (M,), interpolated if clipped, optional.
   * Written into texel5.y.
   */
  endScalars?: Float32Array;

  /** Number of visible segments after clipping */
  segmentCount: number;
}

// ============================================================================
// nD Clipping Types
// ============================================================================

/**
 * Result of clipping a segment to the current nD slice.
 *
 * Segments may:
 * - Be fully visible (both endpoints in slice)
 * - Be partially visible (clipped to slice boundary)
 * - Be invisible (both endpoints outside slice, same side)
 */
export interface ClippedSegment {
  /** Clipped start position in 3D display space */
  p1: number[];

  /** Clipped end position in 3D display space */
  p2: number[];

  /** Parameter t at clipped start (0-1, for attribute interpolation) */
  t1: number;

  /** Parameter t at clipped end (0-1) */
  t2: number;

  /** Whether segment should be rendered */
  visible: boolean;
}

// ============================================================================
// Scene Integration Types
// ============================================================================

import type { UpdateSession } from '../profiling/update-profiler';

/**
 * Data loader interface for Lines nodes.
 *
 * Mirrors the DataLoader interface from points but specialized for lines.
 */
export interface LinesDataLoader {
  /** Load lines data for the given view state */
  loadLines(viewState: LinesViewState, session?: UpdateSession): Promise<LoadedLinesData>;

  /** Update existing data for a new view state (see DataLoader.updateView for `signal`). */
  updateView(
    viewState: LinesViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedLinesData>;

  /** Clean up resources */
  dispose(): void;

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
 * View state for lines loading.
 *
 * Identical shape to the points + gsplats ViewState — kept as a named
 * alias for documentation (a function signature reading
 * `viewState: LinesViewState` is more self-documenting than the
 * generic `ViewState`).
 */
export type LinesViewState = ViewState;

/**
 * User data attached to THREE.Mesh for Lines in scene.
 *
 * Lines use THREE.Mesh with InstancedBufferGeometry (not InstancedMesh)
 * to avoid exceeding WebGL's 16 attribute location limit.
 *
 * Enables runtime type checking and provides access to loader/metadata.
 * Note: Dimension info is NOT stored here - it's only at the Scene level.
 */
export interface LinesUserData {
  /** Node type identifier for runtime type checking */
  nodeType: 'lines';

  /** Data loader instance */
  loader: LinesDataLoader;

  /** Zarr group attributes */
  attrs: LinesMetadata;

  /** Maximum line width (for bounding box expansion) */
  maxWidth: number;

  /** Currently visible segment count after nD clipping (updated on view change) */
  visibleSegmentCount?: number;

  /**
   * True when the GPU interleaved buffer holds the exact data of the last
   * commit — i.e. the append fast path may skip re-uploading the prefix.
   * Cleared to ``false`` by ``NodeFactory.rebuildAfterContextRestore`` after
   * a WebGL context loss (the CPU mirror survives but the GPU buffers are
   * gone), which forces the next commit to a full rewrite. Re-enabled by
   * every full commit. Mirrors ``GSplatsUserData.gpuPrefixIntact``.
   */
  gpuPrefixIntact?: boolean;

  /**
   * The ``SceneLoader`` view-update version this mesh's committed geometry was
   * loaded for. Written at commit (``commit-lines-geometry.ts`` via
   * ``stampLoadedViewVersion``) and read by the LOD registry to tell whether a
   * level is *fresh for the current view (slice/displayDims)* vs merely
   * committed — a re-slice overwrites geometry in place without changing
   * readiness. ``undefined`` ⇒ never committed ⇒ treated as stale.
   */
  loadedViewVersion?: number;

  /** Pick ID assigned by PickingSystem for GPU picking (undefined if picking disabled) */
  pickId?: number;
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Check if metadata is for a Lines node.
 *
 * @param attrs - Unknown attributes object
 * @returns True if attrs is LinesMetadata
 */
export function isLinesMetadata(attrs: unknown): attrs is LinesMetadata {
  return (
    typeof attrs === 'object' &&
    attrs !== null &&
    (attrs as Record<string, unknown>).type === 'lines'
  );
}

/**
 * Check if userData indicates a Lines object.
 *
 * @param userData - THREE.Object3D userData
 * @returns True if userData is LinesUserData
 */
export function isLinesUserData(userData: unknown): userData is LinesUserData {
  return (
    typeof userData === 'object' &&
    userData !== null &&
    (userData as Record<string, unknown>).nodeType === 'lines'
  );
}

/**
 * Check if a line type is valid.
 *
 * @param lineType - String to check
 * @returns True if lineType is a valid LineType
 */
export function isValidLineType(lineType: unknown): lineType is LineType {
  return (
    lineType === 'segments' ||
    lineType === 'polyline' ||
    lineType === 'loop' ||
    lineType === 'indexed'
  );
}
