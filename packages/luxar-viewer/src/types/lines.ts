/**
 * Lines type definitions for luxar-viewer.
 *
 * These types mirror Python luxar.core.Lines and enable type-safe
 * handling of line data throughout the viewer.
 *
 * @module types/lines
 */

import type { BlendingMode } from './blending';
import { LINE_TYPES, type LineTypeName, type OrderingMethodName } from './format-contract';
import type { ViewState } from '../data/data-loader-types';
import type { ScalarArray } from './points';
import type { PositionBounds } from './zarr';
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
export type LineType = LineTypeName;

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

  /**
   * Authored nD axis-aligned vertex bounds. Written unconditionally on
   * every lines node by the compiler (`geometry_writers/lines.py`), and
   * on an additive-ladder parent as the union over its levels — so it is
   * the one authored extent that survives when a node has no spatial
   * index. Optional only for hand-written metadata.
   */
  position_bounds?: PositionBounds;

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

  /** Whether per-element stable string keys exist (CSR-encoded, for element actions) */
  has_keys?: boolean;

  /** Spatial ordering method */
  ordering: OrderingMethodName;

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

  /** Vertex colors (N * colorComponents) RGB or RGBA, null if not present
   * Supports Float32Array (HDR), Uint8Array (SDR), or Uint16Array */
  colors: Float32Array | Uint8Array | Uint16Array | null;

  /**
   * Number of channels per color entry: 3 (RGB) or 4 (RGBA — the alpha
   * column is per-vertex opacity, volumetric phase 4). Defaults to 3
   * when omitted. Mirrors `LoadedPointsData.colorComponents`.
   */
  colorComponents?: 3 | 4;

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

  /**
   * The ascending, pairwise-disjoint ON-DISK VERTEX ranges that the loaded
   * per-vertex arrays — `positions`, `widths`, `colors`, `sharpness`,
   * `scalars` — concatenate, FLATTENED as
   * `[start0, end0, start1, end1, …]`: each range is a half-open
   * `[start, end)` pair, so the array's length is always even and
   * `length / 2` is the range count. This is index space **A** in the lines
   * picking chain (see `rendering/picking/picking-system/element-id-map.ts`):
   * line labels are PER-VERTEX and their CSR is keyed by the on-disk *sorted*
   * vertex row, so these ranges are the second half of the slot → on-disk map
   * picking resolves labels through (`data/loaders/element-ids.ts`).
   *
   * A FLAT `Uint32Array` rather than an object array precisely because this
   * payload is cached: `data/loaders/progressive/slice-cache-helper.ts` bills
   * only own typed-array properties, so an object array would be retained by
   * the LRU at a billed 0 bytes (a fragmented labelled slice — the loader logs a
   * fragmentation "efficiency" diagnostic, so high range counts are expected —
   * could hold roughly twice the bytes the budget believes), and
   * `cloneLodSnapshot` would pass the very same mutable objects into the stored
   * snapshot by reference. A typed array is measured and deep-copied for free,
   * and costs ~6x fewer retained bytes per range.
   *
   * Note this is a VERTEX space, not the segment space `SegmentRange` names:
   * `LinesSpatialIndexLoader` queries visible SEGMENT ranges, then derives the
   * vertex ranges those segments reference and loads only those.
   *
   * Published ONLY when the node declares a per-element string/image CSR
   * (`has_labels` / `has_image_labels` / `has_keys`): nothing else reads the
   * map, and the field would otherwise ride along in every SliceCache snapshot
   * for free.
   * Absent ⇒ no map is composed and picking falls back to the raw slot.
   */
  vertexRangeBounds?: Uint32Array;
}

/**
 * Processed lines data ready for GPU rendering.
 *
 * After nD clipping and 3D projection:
 * - Positions are in 3D display space
 * - Per-segment attributes ready for instanced rendering
 * - Clipped endpoints have interpolated attributes
 */
export interface LinesProjectionBounds {
  /** AABB min corner over start+end positions [x, y, z]. */
  min: [number, number, number];
  /** AABB max corner over start+end positions [x, y, z]. */
  max: [number, number, number];
  /**
   * Max finite half-width over start/end widths — the conservative
   * per-segment footprint used to expand the cull box (matches
   * `computeLineBounds`' expansion semantics exactly).
   */
  maxWidth: number;
}

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

  /**
   * Per-endpoint joint code for the start endpoint (M,): how the line shader
   * should treat it, and — at an ordinary two-segment joint — which segment it
   * joins. `0` free polyline end (keep the soft cap); `-1` slice-clipped
   * (suppress the cap; no neighbour will arrive); `-2` degree->=3 hub (keep the
   * cap); `+(slot + 1)` joins visible segment `slot` at that segment's START;
   * `-(slot + 3)` joins it at that segment's END.
   *
   * `slot` indexes the VISIBLE stream, which is exactly a line-texture storage
   * slot, so it needs no adjustment when the depth-sort worker permutes draw
   * order. Purely topological: the bend angle is measured in screen space by
   * the vertex stage, per frame, so it tracks the camera.
   *
   * Produced by `compute_joint_codes` — see
   * `wasm/rust/src/lines_clipping.rs` for the derivation.
   */
  startJointCode: Float32Array;

  /** Same for the end endpoint (M,). */
  endJointCode: Float32Array;

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

  /**
   * Start-vertex opacity alphas (M,), interpolated if clipped,
   * optional. Present when the source colors carry an RGBA alpha
   * column (`LoadedLinesData.colorComponents === 4`); normalized to
   * [0, 1] float regardless of the source color dtype. Written into
   * texel5.z; presence rides the `userData.hasElementAlpha` stamp
   * (gates only the volumetric w(a) optical-depth map).
   */
  startAlphas?: Float32Array;

  /**
   * End-vertex opacity alphas (M,), interpolated if clipped, optional.
   * Written into texel5.w.
   */
  endAlphas?: Float32Array;

  /** Number of visible segments after clipping */
  segmentCount: number;

  /**
   * Fused-scan cull metadata from projection (AABB over start+end
   * positions + max finite width — see {@link LinesProjectionBounds}).
   * Mirrors the Points `metadata.bounds` / gsplats `bounds` pattern:
   * when present, `computeLineBounds` skips its O(N) per-segment scan;
   * absent (zero visible segments, or a producer that didn't scan) ⇒
   * scan fallback. Plain scalars so the object survives the worker
   * structured clone.
   */
  bounds?: LinesProjectionBounds;

  /**
   * Slot → ON-DISK element index map for per-element label lookups (M,),
   * composed at projection time by `data-processor-lines.ts`.
   *
   * Indexed by the VISIBLE SEGMENT slot the lines pick shader reports (the
   * line-texture texel row); the value is the on-disk *sorted VERTEX* row of
   * that segment's **START** vertex, because line labels are per-vertex and
   * their CSR is keyed by that row. A segment has two endpoints and the pick id
   * is a `flat` vertex-stage varying, so exactly one of them can be reported —
   * see the convention note in `data-processor-lines.ts`.
   *
   * Absent ⇒ identity / unavailable: picking falls back to the raw slot. That
   * happens for a node with no label CSR (no `vertexRangeBounds` published), and
   * whenever the composition inputs were inconsistent (fail closed). The commit
   * stamps it onto the MESH (`types/committed-data::setElementIdMap`), never
   * onto the loaded payload — that payload may be a SliceCache-owned snapshot.
   */
  elementIds?: Uint32Array;
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
    signal?: AbortSignal,
    residencyAllowanceBytes?: number
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

  /** Elements requested by the latest successful commit, before the GPU capacity clamp. */
  requestedElementCount?: number;

  /** Elements omitted by the latest successful commit's GPU capacity clamp. */
  droppedElementCount?: number;

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
  return typeof lineType === 'string' && (LINE_TYPES as readonly string[]).includes(lineType);
}
