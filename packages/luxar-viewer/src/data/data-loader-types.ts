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
}

// Re-export the points-specific types (LoadedPointsData, PointRange,
// PositionArray, ColorArray, ScalarArray) from `types/points.ts` —
// kept here for back-compat with consumers using the historical
// `from '../data/data-loader-types'` import path. New code should
// import from `types/points` directly.
export type {
  LoadedPointsData,
  PointRange,
  PositionArray,
  ColorArray,
  ScalarArray,
} from '../types/points';

import type { UpdateSession } from '../profiling/update-profiler';
import type { LoadedPointsData, PointRange } from '../types/points';

/**
 * Core interface for points data loaders. Implementations handle
 * different loading strategies (spatial index vs fallback).
 *
 * Mirrors `LinesDataLoader` and `GSplatsDataLoader` in
 * `types/{lines,gsplats}.ts`. Aliased as `PointsDataLoader` in
 * `types/points.ts` (using a slightly different `PointsViewState`
 * shape) — collapsing the two is tracked as Phase 11.4.
 */
export interface DataLoader {
  /** Load points data for the given view state */
  loadPoints(viewState: ViewState, session?: UpdateSession): Promise<LoadedPointsData>;

  /** Update existing data for a new view state */
  updateView(viewState: ViewState, session?: UpdateSession): Promise<LoadedPointsData>;

  /** Clean up resources */
  dispose(): void;
}

/**
 * Configuration for data loader behavior
 */
export interface LoaderConfig {
  /** Enable debug logging */
  debug?: boolean;

  /** Enable data loading monitor UI */
  enableMonitor?: boolean;

  /** Disable both L1/L2 cache tiers and L0 decompressed cache. */
  noCache?: boolean;

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
