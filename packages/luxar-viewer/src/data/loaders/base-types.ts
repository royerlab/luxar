/**
 * Base types for unified spatial index loader architecture.
 *
 * These types define the common interfaces shared across Points, Lines, and GSplats
 * loaders, enabling code reuse and consistent patterns.
 *
 * @module data/loaders/base-types
 */

import type * as zarr from '../zarr';
import type * as THREE from 'three';
import type { DimensionMetadata } from '../../types/dims';
import type { UpdateSession } from '../../profiling/update-profiler';
import type { SceneNode } from '../data-loader-types';
import type { ArrayRefRegistry } from '../array-decoder/decoder';

// ============================================================================
// Common View State
// ============================================================================

/**
 * Base view state interface shared by all loader types.
 *
 * All three data types (Points, Lines, GSplats) share this core view state:
 * - displayDims: Which 3 dimensions to project to 3D
 * - slicePosition: nD position for slicing
 * - tolerance: Visibility tolerance per dimension
 * - dimensions: Optional dimension metadata
 */
export interface BaseViewState {
  /** Which dimensions to display (max 3, indices into nD space) */
  displayDims: readonly number[];

  /** Current position in nD space (one value per dimension) */
  slicePosition: readonly number[];

  /** Tolerance for slicing in each dimension */
  tolerance: readonly number[];

  /**
   * Dimension metadata for the dataset (raw metadata array). Same
   * shape as `ViewState.dimensions`.
   */
  dimensions?: DimensionMetadata[];
}

// ============================================================================
// Range Types
// ============================================================================

/**
 * Generic range for loading array subsets.
 * Used by all loader types (Points, Lines, GSplats).
 */
export interface LoadRange {
  /** Start index (inclusive) */
  start: number;

  /** End index (exclusive) */
  end: number;
}

// ============================================================================
// Chunk Spatial Index
// ============================================================================

/**
 * Base interface for chunk-based spatial indices.
 *
 * All three data types use similar chunk-based spatial indexing:
 * - Morton/Hilbert ordering for locality
 * - Bounding boxes per chunk
 * - Query by nD slice position + tolerance
 *
 * @internal — preserved for future use; no current consumer.
 */
export interface BaseChunkSpatialIndex {
  /** Chunk bounding boxes (num_chunks * ndim * 2), flattened row-major */
  chunkBounds: Float32Array;

  /** Number of chunks */
  chunkCount: number;

  /** Base metadata fields shared across all types */
  metadata: {
    ndim: number;
    ordering: 'morton' | 'hilbert' | 'none';
  };
}

// ============================================================================
// Loader Configuration
// ============================================================================

/**
 * Common loader configuration options.
 *
 * @internal — preserved for future use; no current consumer.
 */
export interface LoaderOptions {
  /** Worker threshold - minimum elements to offload to worker */
  workerThreshold?: number;

  /** Enable debug logging */
  debug?: boolean;
}

/**
 * Dependencies injected into loaders.
 *
 * @internal — preserved for future use; no current consumer.
 */
export interface LoaderDependencies {
  /** Zarr location for reading arrays */
  zarrLocation: zarr.Location<zarr.Readable>;

  /** Scene node with attributes */
  node: SceneNode;

  /** Array reference registry for encoding resolution */
  refRegistry?: ArrayRefRegistry;

  /** Zarr store for direct access */
  zarrStore?: zarr.Readable;
}

// ============================================================================
// Loader Interface
// ============================================================================

/**
 * Base loader interface.
 *
 * All loader types implement:
 * - initialize(): Async initialization
 * - dispose(): Cleanup resources
 */
export interface BaseLoader {
  /** Initialize the loader (load spatial index, open arrays) */
  initialize(): Promise<void>;

  /** Clean up resources */
  dispose(): void;
}

/**
 * Data loader interface with load method.
 * TViewState: The view state type (PointsViewState, LinesViewState, GSplatsViewState)
 * TLoadedData: The loaded data type (LoadedPointsData, LoadedLinesData, LoadedGSplatsData)
 *
 * @internal — preserved for future use; no current consumer.
 */
export interface SpatialDataLoader<
  TViewState extends BaseViewState,
  TLoadedData,
> extends BaseLoader {
  /**
   * Load data for the given view state.
   *
   * This is the unified entry point that:
   * 1. Queries spatial index for visible ranges
   * 2. Loads raw data from zarr
   * 3. Projects to 3D (with optional worker offload)
   *
   * @param viewState - Current view state
   * @param session - Optional profiler session
   * @returns Loaded and projected data
   */
  loadData(viewState: TViewState, session?: UpdateSession): Promise<TLoadedData>;

  /** Update view (may be more efficient than full reload) */
  updateView(viewState: TViewState, session?: UpdateSession): Promise<TLoadedData>;
}

// ============================================================================
// Loaded Data Metadata
// ============================================================================

/**
 * Common metadata structure for loaded data.
 *
 * @internal — preserved for future use; no current consumer.
 */
export interface LoadedDataMetadata {
  /** Total items in the full dataset */
  totalItems: number;

  /** Number of items actually loaded */
  loadedItems: number;

  /** Bounding box of loaded data */
  bounds: THREE.Box3;

  /** Whether spatial index was used */
  usedSpatialIndex: boolean;
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Check if a view state has dimension metadata.
 */
export function hasDimensionMetadata(
  viewState: BaseViewState
): viewState is BaseViewState & { dimensions: DimensionMetadata[] } {
  return !!viewState.dimensions && viewState.dimensions.length > 0;
}

/**
 * Get display dimension count (always 3 or less).
 */
export function getDisplayDimCount(viewState: BaseViewState): number {
  return Math.min(viewState.displayDims.length, 3);
}

/**
 * Check if a dimension is hidden (not displayed).
 */
export function isHiddenDimension(viewState: BaseViewState, dimIndex: number): boolean {
  return !viewState.displayDims.includes(dimIndex);
}
