/**
 * Points type definitions for luxar-viewer.
 *
 * These types mirror Python luxar.core.Points and enable type-safe
 * handling of point data throughout the viewer.
 *
 * This module follows the same pattern as types/lines.ts and types/gsplats.ts
 * for consistency across all node types — every node-type's loaded /
 * processed / view-state / metadata interfaces, plus the point-specific
 * array-element type aliases (Float16Array support is points-only),
 * live in types/{node-type}.ts.
 *
 * @module types/points
 */

import type { BlendingMode } from './blending';
import * as THREE from 'three';
import type { DimensionMetadata } from './dims';
import type { LoaderMetrics, MonitorEventListener, QueryInfo } from './data-monitor-types';
import type { UpdateSession } from '../profiling/update-profiler';

// ============================================================================
// Effective-Radius Config (shared between data/points/ and workers/)
// ============================================================================

/**
 * Configuration for effective radius calculation.
 *
 * Canonical definition lives here in `types/`. The Points effective-radius
 * code consumes this shape: `data/points/effective-radius-calculator.ts`
 * (query-tolerance helpers) and `data/points/projection.ts` (the
 * main-thread, WASM-accelerated projection). Re-exported from
 * `effective-radius-calculator.ts` for backward compatibility — existing
 * importers don't need to change.
 */
export interface EffectiveRadiusConfig {
  /** Which dimensions points extend through spatially */
  spatialExtendDims: boolean[];
  /** Maximum radius in the dataset for query optimization */
  maxRadius: number;
}

// ============================================================================
// Element Array Types (Float16 is points-specific)
// ============================================================================

/** Position array variants. Float16 support is points-only. */
export type PositionArray = Float32Array | Float16Array;

/** Color array variants. Uint8 / Uint16 are dtype-preserved (255 vs 1.0 semantics). */
export type ColorArray = Float32Array | Uint8Array | Uint16Array;

/** Scalar attribute array variants (radii, sharpness). */
export type ScalarArray = Float32Array | Float16Array | Uint8Array;

/**
 * Scalar attribute variants that additionally keep a native Uint16Array.
 * Points sharpness/scalars are dtype-preserved through the accumulator (like
 * colors) and normalized at the GPU upload site (÷255 / ÷65535); Float16
 * widens value-preserving to Float32. Kept distinct from `ScalarArray` so the
 * Lines path (which has no native-uint16 scalar buffer) is unaffected.
 */
export type PointScalarArray = ScalarArray | Uint16Array;

// ============================================================================
// Loaded / Range Types (post-spatial-index slice, ready for GPU upload)
// ============================================================================

/**
 * Loaded points data ready for GPU rendering.
 * All arrays are properly aligned with the same point ordering.
 * Arrays can be in different data types for memory efficiency.
 *
 * Named with "Loaded" prefix for consistency with LoadedLinesData and
 * LoadedGSplatsData.
 */
export interface LoadedPointsData {
  /** 3D positions extracted from nD space (size: numPoints * 3) */
  positions: PositionArray;

  /** RGB or RGBA colors (size: numPoints * colorComponents, optional) */
  colors?: ColorArray;

  /**
   * Components per color item: 3 = RGB, 4 = RGBA — the alpha column is
   * per-point opacity (VOLUMETRIC_BLENDING_SPEC.md phase 3). Absent ⇒ 3.
   * Mirrors `LoadedGSplatsData.colorComponents`.
   */
  colorComponents?: 3 | 4;

  /** Point radii in world units (size: numPoints, optional) */
  radii?: ScalarArray;

  /** Point sharpness values (size: numPoints, optional). Native Uint16 is
   *  kept dtype-preserving (like colors) and normalized on GPU upload. */
  sharpness?: PointScalarArray;

  /**
   * Per-point scalar values for colormap lookup (size: numPoints, optional).
   *
   * when present, the geometry binds a `scalar` attribute and the
   * Point shader's USE_COLORMAP path samples the LUT at
   * `(scalar - uScalarMin) * uScalarScale`. Without scalars, colormap
   * mode falls back to vertex colours. The dtype matches the source
   * zarr array (Float32 / Float16 / Uint8 / Uint16); the shader reads
   * `radius`-style normalised attributes when the dtype is integer.
   */
  scalars?: PointScalarArray;

  /**
   * Visible-buffer slot → ON-DISK element index map (size: numPoints, optional).
   *
   * Produced by `data/points/projection.ts::projectPointsTo3D` and consumed at
   * commit by `data/scene-loader/commit/commit-points-geometry.ts`, which
   * forwards it to `types/committed-data::setElementIdMap`. Picking reads that
   * MESH-level stamp — not this field — via
   * `rendering/picking/picking-system/element-id-map.ts`, so hover string
   * channels index their per-element CSR by the on-disk index rather than by
   * the storage slot the pick shader reports.
   * The two diverge after spatial range loading (only the visible ranges are
   * concatenated) or effective-radius compaction (zero-radius points are
   * dropped in place).
   *
   * OMITTED in three cases: when the node declares none of `has_labels`,
   * `has_image_labels`, or `has_keys` (the map costs 4 B/point on the
   * zero-allocation accumulator path and no string/image reader exists — the
   * embedder `selection` event can still fire on such a node and keeps
   * reporting the slot); when
   * the identity holds — a single range starting at 0 with no compaction, i.e.
   * the common plain-3D case; and when the map could not be built at all, in
   * which case {@link LoadedPointsData.elementIdsUnavailable} is set (the slot
   * is NOT a valid substitute — only the first two cases let a consumer use it
   * directly).
   *
   * On an additive LOD ladder each level's map is in that LEVEL's on-disk
   * index space. `points-progressive-loader.ts::concatenatePointsData` either
   * composes the levels into the PARENT node's union string-channel CSR space
   * (when the parent declares that CSR — the `levelOffsets` path, #1439) or
   * publishes nothing at all.
   */
  elementIds?: Uint32Array;

  /**
   * Set (only when true) when a slot → on-disk map was WANTED — the node
   * declares `has_labels` / `has_image_labels` / `has_keys` — but could not be
   * built: the slot is NOT the on-disk index and there is no map to say what is.
   *
   * Distinct from a plain missing {@link LoadedPointsData.elementIds}, which
   * usually means the identity holds. A consumer that composes this payload
   * into a wider index space (the additive-ladder concat) must not substitute
   * identity for it — it publishes no map instead. ONE exemption: a payload
   * with `pointCount === 0` contributes no slots to a composition, so it cannot
   * corrupt one and does not veto its siblings' maps (both ladder-concat
   * branches skip the flag there). That is not a suppression:
   * with no map, picking reports the raw slot, which on a sliced payload is
   * itself a wrong CSR row. It only guarantees the reported id is never one
   * composed from data known to be inconsistent.
   */
  elementIdsUnavailable?: boolean;

  /** Number of points loaded (top-level for consistency with Lines/GSplats) */
  pointCount: number;

  /** Original nD dimensionality (top-level for consistency with Lines/GSplats) */
  ndim: number;

  /** Metadata about the loaded data */
  metadata: {
    /** Total points in the full dataset */
    totalPoints: number;

    /** Number of points actually loaded (also available as top-level pointCount) */
    loadedPoints: number;

    /** Bounding box of loaded points */
    bounds: THREE.Box3;

    /** Whether spatial index was used */
    usedSpatialIndex: boolean;

    /** Whether effective radius calculation was applied */
    usedEffectiveRadius?: boolean;

    /** Original data types from zarr (for proper conversion) */
    dtypes?: {
      positions?: string;
      colors?: string;
      radii?: string;
      sharpness?: string;
      scalars?: string;
    };
  };
}

/**
 * Range of points to load (for spatial index queries). Mirrors
 * `SegmentRange` in `types/lines.ts` and `SplatRange` in
 * `types/gsplats.ts`.
 */
export interface PointRange {
  /** Starting index (inclusive) */
  start: number;

  /** Ending index (exclusive) */
  end: number;
}

// ============================================================================
// Metadata Types (from zarr .zattrs)
// ============================================================================

/**
 * Points node metadata from zarr .zattrs
 *
 * This interface defines all metadata stored with a Points node in the zarr archive.
 * It enables type-safe access to point properties throughout the viewer.
 */
export interface PointsMetadata {
  /** Node type identifier */
  type: 'points';

  /** Total point count */
  n_points: number;

  /** Position dimensionality */
  ndim: number;

  /** Maximum point radius in world units */
  max_radius?: number;

  /** Whether colors array is present */
  has_colors?: boolean;

  /** Whether per-element scalar values are present (drives colormap). */
  has_scalars?: boolean;

  /** Colormap name applied to per-element scalars (e.g. 'viridis', 'plasma'). */
  colormap?: string;

  /** [min, max] of the scalar data range used to remap into the colormap LUT. */
  scalar_data_range?: [number, number];

  /** Whether radii array is present */
  has_radii?: boolean;

  /** Whether sharpness array is present */
  has_sharpness?: boolean;

  /** Whether spatial index exists */
  has_spatial_index?: boolean;

  /** Whether per-element string labels exist (CSR-encoded, for hover tooltips) */
  has_labels?: boolean;

  /** Whether per-element image labels exist (CSR-encoded, for hover thumbnails) */
  has_image_labels?: boolean;

  /** Whether per-element stable string keys exist (CSR-encoded, for element actions) */
  has_keys?: boolean;

  /** Spatial ordering method */
  ordering?: 'morton' | 'hilbert' | 'none';

  /** Elements per chunk */
  chunk_size?: number;

  /** Dimensions using space-filling curve ordering (spatial dims) */
  ordering_dims?: number[];

  /** Dimensions using lexicographic ordering (discrete/slice dims) */
  slice_dims?: number[];

  /** Bits per dimension for space-filling curve encoding */
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
   * Points with extend_to_all will be visible regardless of slice position
   * in the specified dimensions (e.g., ['time'] makes points visible
   * at all time values).
   */
  extend_to_all?: string[];

  /**
   * bytes loaded from the node's `colormap_lut` zarr array when
   * `colormap === 'custom'`. Populated at scene-graph build time;
   * runtime-only field, not authored at the zarr level.
   */
  customLutBytes?: Uint8Array;

  /** Position array dtype (for proper conversion) */
  position_dtype?: string;

  /** Color array dtype */
  color_dtype?: string;

  /** Radius array dtype */
  radius_dtype?: string;

  /** Sharpness array dtype */
  sharpness_dtype?: string;
}

// ============================================================================
// View State Types
// ============================================================================

/**
 * Represents the current view state for data loading.
 * This determines what portion of the nD dataset should be loaded.
 *
 * Canonical definition lives here in `types/` (moved from
 * `data/data-loader-types.ts`, which re-exports it — `data/` may import
 * from `types/`, never the reverse).
 *
 * Array fields are typed `readonly number[]` so the *contents* cannot
 * be mutated (`arr[i] = …`, `arr.push(…)`). The fields themselves are
 * still re-assignable, so callers that need to update the view state
 * should construct a fresh array (`tolerance: [...prev, x]`) rather
 * than mutating in place. This prevents the cross-method mutation
 * races flagged in the 2026-05-06 review.
 */
export interface ViewState {
  /** Which dimensions to display (max 3, indices into nD space) */
  displayDims: readonly number[];

  /** Current position in nD space (one value per dimension) */
  slicePosition: readonly number[];

  /** Tolerance for slicing in each dimension (radius in non-displayed dims, 0 for displayed) */
  tolerance: readonly number[];

  /**
   * Dimension metadata for the dataset.
   *
   * **IMPORTANT**: This field is REQUIRED for the `extend_to_all` feature to work.
   * If `extend_to_all` is configured on a node but dimensions is undefined,
   * the optimization will silently be skipped.
   *
   * Always provide dimensions when using nD datasets with extend_to_all.
   *
   * Same shape as `LinesViewState.dimensions`, `GSplatsViewState.dimensions`,
   * `PointsViewState.dimensions`, and `BaseViewState.dimensions`. Callers
   * that need richer dimension state (selected dim, animation state, etc.)
   * read it off `sceneDimsManager` directly rather than reaching for a
   * different shape on this field.
   */
  dimensions?: DimensionMetadata[];

  /**
   * Per-tick LOD time budget (ms) for progressive loaders during dimension
   * ANIMATION playback, or absent for normal full-refinement behavior.
   *
   * **PER-PASS DIRECTIVE, not state.** This field rides only the
   * `updateView(partial)` call: the scene loader destructures it OUT before
   * merging into its persistent view state (a stale budget would leave
   * loaders capped after playback ends), threads it through the per-type
   * handler ctxs, and injects it into the DERIVED per-node view state right
   * before `loader.updateView` — so refinement and retry passes (which
   * derive independently) are budget-free by construction. It must never
   * enter `viewStatesEqual` (would reset ladders on play/pause) nor the
   * SliceCache key (`buildSliceViewSig`).
   */
  frameBudgetMs?: number;

  /**
   * Pinned additive-ladder depth for progressive loaders during dimension
   * playback and scrubbing — the "playback detail" setting. A number makes a
   * pass load EXACTLY `min(ladderDepth, nLods)` rungs, cold or not, ignoring the
   * time budget, so every frame of a time-lapse is drawn at the same rung and
   * the tick waits for the data instead of showing whatever happened to be
   * resident. `'auto'` lets each loader resolve its own depth from its energy
   * stamps (`resolveLadderDepth`); a ladder without stamps stays time-budgeted.
   * Absent = the time-budgeted behaviour.
   *
   * Like `frameBudgetMs` this is a **PER-PASS DIRECTIVE, not state**: the scene
   * loader strips it before persisting the view state, threads it through the
   * handler ctxs into the DERIVED per-node view state, and the SlicePrefetcher
   * hands the same value to its shadow passes so the t+1 S-cache entry carries
   * the pinned prefix. It must never enter `viewStatesEqual` nor the SliceCache
   * key.
   */
  ladderDepth?: number | 'auto';

  /**
   * Set only on the SlicePrefetcher's shadow pass: marks stores as PREFETCH so
   * the loader pins the cached ladder until the foreground tick restores it
   * (see `SliceCache.set({ pin })`). Without the pin, the just-stored t+1 is the
   * MRU entry and the FIRST victim of a subsequent scan-eviction under budget
   * pressure — defeating the prefetch exactly in the thrash regime. Like
   * `frameBudgetMs` this is a per-pass directive: it must never enter
   * `viewStatesEqual` nor the SliceCache key (`buildSliceViewSig`).
   */
  prefetch?: boolean;

  /**
   * Set by `deriveNodeViewState` when this node's composed `nd_transform` maps
   * the current WORLD slice to a position that no local value can occupy — a
   * DISCRETE dimension whose inverse image falls between grid points (e.g.
   * `scale: 2` at an odd world frame). The node must then render NOTHING.
   *
   * **DERIVED PER-NODE, not global state.** It is computed from the node's own
   * transform in `invertNdTransformForQuery` (which documents the rule), so it
   * only ever appears on the derived per-node view state, never on the scene
   * loader's shared one. Each geometry's range query honours it by returning an
   * empty range list, whose existing "no visible elements" path clears the
   * geometry. It must never enter `viewStatesEqual` nor the SliceCache key —
   * both already vary with `slicePosition`, from which this is a pure function.
   */
  noPreimage?: boolean;
}

/**
 * View state for points loading.
 *
 * Identical shape to the lines + gsplats ViewState — kept as a named
 * alias for documentation (a function signature reading
 * `viewState: PointsViewState` is more self-documenting than the
 * generic `ViewState`).
 *
 * @internal — a type-only documentation alias; no runtime caller depends on it.
 */
export type PointsViewState = ViewState;

// ============================================================================
// Scene Integration Types
// ============================================================================

/**
 * Core interface for points data loaders. Implementations handle
 * different loading strategies (spatial index vs fallback).
 *
 * Canonical definition lives here in `types/` (moved from
 * `data/data-loader-types.ts`, which re-exports it).
 *
 * Mirrors `LinesDataLoader` and `GSplatsDataLoader` in
 * `types/{lines,gsplats}.ts`. Aliased as `PointsDataLoader` below
 * (using a slightly different `PointsViewState` shape).
 */
export interface DataLoader {
  /** Load points data for the given view state */
  loadPoints(viewState: ViewState, session?: UpdateSession): Promise<LoadedPointsData>;

  /**
   * Update existing data for a new view state.
   * @param signal - Optional per-update abort signal. When it fires (a newer
   *   view-state superseded this one), in-flight chunk reads/decodes bail with
   *   an `AbortError` instead of running to completion.
   */
  updateView(
    viewState: ViewState,
    session?: UpdateSession,
    signal?: AbortSignal,
    residencyAllowanceBytes?: number
  ): Promise<LoadedPointsData>;

  /** Clean up resources */
  dispose(): void;

  /**
   * LoaderMonitor surface (optional, for the data-loading-monitor UI).
   * Mirrors the surface that `points-spatial-index-loader.ts` exposes —
   * implementations that don't track metrics may omit these methods.
   * Symmetric with `LinesDataLoader` / `GSplatsDataLoader`.
   */
  addEventListener?(listener: MonitorEventListener): void;
  removeEventListener?(listener: MonitorEventListener): void;
  getMetrics?(): LoaderMetrics;
  getActiveQueries?(): QueryInfo[];
}

/**
 * Data loader interface for Points nodes. Identical shape to the
 * `DataLoader` interface above; aliased for
 * naming symmetry with `LinesDataLoader` and `GSplatsDataLoader`.
 */
export type PointsDataLoader = DataLoader;

/**
 * User data attached to THREE.Points objects in the scene.
 *
 * Enables runtime type checking and provides access to loader/metadata.
 * Note: Dimension info is NOT stored here - it's only at the Scene level.
 *
 * This follows the same pattern as LinesUserData and GSplatsUserData.
 */
export interface PointsUserData {
  /** Node type identifier for runtime type checking */
  nodeType: 'points';

  /** Data loader instance */
  loader: PointsDataLoader;

  /** Zarr group attributes */
  attrs: PointsMetadata;

  /** Maximum point radius (for bounding box expansion) */
  maxRadius?: number;

  /** Currently visible point count after nD slicing (updated on view change) */
  visiblePointCount?: number;

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
   * loaded for. Written at commit (``commit-points-geometry.ts`` via
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
 * Check if metadata is for a Points node.
 *
 * @param attrs - Unknown attributes object
 * @returns True if attrs is PointsMetadata
 */
export function isPointsMetadata(attrs: unknown): attrs is PointsMetadata {
  return (
    typeof attrs === 'object' &&
    attrs !== null &&
    (attrs as Record<string, unknown>).type === 'points'
  );
}

/**
 * Check if userData indicates a Points object.
 *
 * @param userData - THREE.Object3D userData
 * @returns True if userData is PointsUserData
 */
export function isPointsUserData(userData: unknown): userData is PointsUserData {
  return (
    typeof userData === 'object' &&
    userData !== null &&
    (userData as Record<string, unknown>).nodeType === 'points'
  );
}
