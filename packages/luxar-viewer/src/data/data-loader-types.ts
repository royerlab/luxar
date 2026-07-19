/**
 * Core types and interfaces for the new data loading architecture.
 *
 * This module defines the clean abstractions for loading nD points data
 * with proper spatial indexing support and aligned attribute loading.
 */

import type { DimensionMetadata } from '../types/dims';

/**
 * Represents the current view state for data loading.
 * This determines what portion of the nD dataset should be loaded.
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
   * Set only on the SlicePrefetcher's shadow pass: marks stores as PREFETCH so
   * the loader pins the cached ladder until the foreground tick restores it
   * (see `SliceCache.set({ pin })`). Without the pin, the just-stored t+1 is the
   * MRU entry and the FIRST victim of a subsequent scan-eviction under budget
   * pressure — defeating the prefetch exactly in the thrash regime. Like
   * `frameBudgetMs` this is a per-pass directive: it must never enter
   * `viewStatesEqual` nor the SliceCache key (`buildSliceViewSig`).
   */
  prefetch?: boolean;
}

// Re-export the points-specific types (LoadedPointsData, PointRange,
// PositionArray, ColorArray, ScalarArray) from `types/points.ts` for
// consumers that import through `data/data-loader-types`. New code
// should import from `types/points` directly.
export type {
  LoadedPointsData,
  PointRange,
  PositionArray,
  ColorArray,
  ScalarArray,
} from '../types/points';

import type { UpdateSession } from '../profiling/update-profiler';
import type { LoadedPointsData, PointRange } from '../types/points';
import type { LoaderMetrics, MonitorEventListener, QueryInfo } from '../types/data-monitor-types';

/**
 * Core interface for points data loaders. Implementations handle
 * different loading strategies (spatial index vs fallback).
 *
 * Mirrors `LinesDataLoader` and `GSplatsDataLoader` in
 * `types/{lines,gsplats}.ts`. Aliased as `PointsDataLoader` in
 * `types/points.ts` (using a slightly different `PointsViewState`
 * shape).
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
    signal?: AbortSignal
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
 * Configuration for data loader behavior
 */
export interface LoaderConfig {
  /** Enable debug logging */
  debug?: boolean;

  /** Enable data loading monitor UI */
  enableMonitor?: boolean;

  /** Disable ALL cache tiers: SliceCache, L0 decompressed, and L1/L2. */
  noCache?: boolean;

  /** Disable only the SliceCache / S-cache (L0/L1/L2 stay on). */
  noSliceCache?: boolean;

  /**
   * Explicit total in-memory cache pool (L0+L1+S-cache) in MB, from
   * `?cacheBudgetMB=` / the native launcher. Used where `performance.memory` is
   * absent (WKWebView/Safari) so heap-aware sizing still gets a real budget.
   */
  cacheBudgetMB?: number | null;

  /** Verbose cache logging. */
  cacheDebug?: boolean;

  /** Clear caches on init. */
  clearCache?: boolean;

  /** Disable adjacent-chunk prefetching. */
  noPrefetch?: boolean;

  /** Verbose prefetch logging. */
  prefetchDebug?: boolean;
}

/**
 * Node in the scene graph with loading information
 */
export interface SceneNode {
  /** Path in the zarr store */
  path: string;

  /** Node type (group, points, etc.) */
  type: string;

  /** Rendering and node attributes from Zarr */
  attrs: {
    /**
     * Transformation matrix (16 elements for a column-major 4x4 matrix).
     * Typed as `readonly number[]` because zarr metadata is parsed
     * dynamically; see {@link import('../types/zarr').Matrix4x4} for the
     * narrowed 16-tuple shape.
     */
    transform?: readonly number[];

    /** Per-dimension transforms for non-displayed dimensions */
    nd_transform?: import('../types/zarr').NdTransformMap;

    /** Rendering attributes */
    opacity?: number;
    absorption?: number;
    gamma?: number;
    intensity?: number;
    offset?: number;
    blending_mode?: string;
    max_radius?: number;
    n_points?: number;

    /** Whether this node is exposed as a layer in the Layers panel */
    layer?: boolean;

    /** Initial visibility when the scene loads (default: true). Authoring-time only. */
    visible?: boolean;

    /** Min/max of color data, computed at encoding time */
    color_data_range?: [number, number];

    /** Min/max of amplitude data (GSplats), computed at encoding time */
    amplitude_data_range?: [number, number];

    /** Min/max of scalar data (Points/Lines), computed at encoding time */
    scalar_data_range?: [number, number];

    /** Colormap name for scalar-to-color mapping (e.g., "viridis", "green", "custom") */
    colormap?: string;

    /** Whether this node has scalar values for colormap lookup */
    has_scalars?: boolean;

    /** Dimensions to extend visibility across (points visible at all values) */
    extend_to_all?: string[];

    /** Any additional attributes */
    [key: string]: unknown;
  };

  /** Whether this node has a spatial index */
  hasSpatialIndex: boolean;

  /** Child nodes */
  children?: SceneNode[];
}

/**
 * Result of a spatial index query
 *
 * @internal — reserved extension shape; no current consumer.
 */
export interface SpatialQueryResult {
  /** Point ranges that match the query */
  ranges: PointRange[];

  /** Total number of points in the ranges */
  totalPoints: number;

  /** Grid cells that were queried */
  queriedCells: number;
}

/**
 * Loader statistics for monitoring
 *
 * @internal — reserved extension shape; no current consumer.
 */
export interface LoaderStats {
  /** Time taken to load (ms) */
  loadTime: number;

  /** Amount of data loaded (bytes) */
  bytesLoaded: number;

  /** Number of zarr chunks accessed */
  chunksAccessed: number;

  /** Whether cache was used */
  fromCache: boolean;
}

/**
 * Tag identifying which of the three first-class geometry kinds a node
 * or handler operates on. Used by the per-type registry that replaces
 * switch/case dispatch on `geometry_type` strings in scene-loader.
 */
export type GeometryKind = 'points' | 'lines' | 'gsplats';
