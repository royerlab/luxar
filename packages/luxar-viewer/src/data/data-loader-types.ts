/**
 * Core types and interfaces for the new data loading architecture.
 *
 * This module defines the clean abstractions for loading nD points data
 * with proper spatial indexing support and aligned attribute loading.
 */

// Re-export the points-specific types (ViewState, DataLoader,
// LoadedPointsData, PointRange, PositionArray, ColorArray, ScalarArray)
// from `types/points.ts` for consumers that import through
// `data/data-loader-types`. The `ViewState` and `DataLoader` definitions
// moved to `types/points.ts` (types/ must not import from data/, so the
// canonical definitions live in types/). New code should import from
// `types/points` directly.
export type {
  ViewState,
  DataLoader,
  LoadedPointsData,
  PointRange,
  PositionArray,
  ColorArray,
  ScalarArray,
  PointScalarArray,
} from '../types/points';

import type { PointRange } from '../types/points';
import type { GeometryTypeName } from '../types/format-contract';

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
 * Tag identifying which first-class geometry kind a node or handler
 * operates on. Used by the per-type registry that replaces switch/case
 * dispatch on `geometry_type` strings in scene-loader.
 *
 * Single-sourced from the cross-language format contract
 * (`format-contract/contract.yaml` → `geometry_types`), which the generator
 * checks is a subset of `node_types`. Declaring the members here by hand
 * would let the viewer's dispatch vocabulary drift from the on-disk one.
 */
export type GeometryKind = GeometryTypeName;
