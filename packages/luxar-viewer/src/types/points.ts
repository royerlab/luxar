/**
 * Points type definitions for luxar-viewer.
 *
 * These types mirror Python luxar.core.Points and enable type-safe
 * handling of point data throughout the viewer.
 *
 * This module follows the same pattern as types/lines.ts and types/gsplats.ts
 * for consistency across all node types.
 *
 * @module types/points
 */

import type { DimensionMetadata } from './dims';
import type { LoadedPointsData } from '../data/data-loader-types';
import type { ChunkSpatialIndex } from '../data/chunk-spatial-index';
import type { UpdateSession } from '../profiling/update-profiler';

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

  /** Whether radii array is present */
  has_radii?: boolean;

  /** Whether sharpness array is present */
  has_sharpness?: boolean;

  /** Whether spatial index exists */
  has_spatial_index?: boolean;

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

  /** Gamma correction */
  gamma?: number;

  /** Blending mode */
  blending_mode?: 'additive' | 'normal' | 'max';

  /**
   * List of dimension names to extend visibility across.
   * Points with extend_to_all will be visible regardless of slice position
   * in the specified dimensions (e.g., ['time'] makes points visible
   * at all time values).
   */
  extend_to_all?: string[];

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
// Spatial Index Types
// ============================================================================

/**
 * Re-export ChunkSpatialIndex as PointsChunkSpatialIndex for consistency.
 *
 * Points use chunk-based spatial indexing with Morton/Hilbert ordering.
 * The index contains bounding boxes for each chunk enabling efficient nD queries.
 */
export type PointsChunkSpatialIndex = ChunkSpatialIndex;

// ============================================================================
// View State Types
// ============================================================================

/**
 * View state for points loading.
 *
 * Mirrors the pattern used by LinesViewState and GSplatsViewState.
 * Note: Uses DimensionMetadata[] for consistency with Lines/GSplats.
 */
export interface PointsViewState {
  /** Which dimensions to display (max 3, indices into nD space) */
  displayDims: number[];

  /** Current position in nD space (one value per dimension) */
  slicePosition: number[];

  /** Tolerance for slicing in each dimension */
  tolerance: number[];

  /** Dimension metadata for the dataset */
  dimensions?: DimensionMetadata[];
}

// ============================================================================
// Scene Integration Types
// ============================================================================

/**
 * Data loader interface for Points nodes.
 *
 * Mirrors the pattern from LinesDataLoader and GSplatsDataLoader.
 * This is a standalone interface (doesn't extend DataLoader) to allow
 * PointsViewState to use DimensionMetadata[] for consistency with Lines/GSplats.
 */
export interface PointsDataLoader {
  /** Load points data for the given view state */
  loadPoints(viewState: PointsViewState, session?: UpdateSession): Promise<LoadedPointsData>;

  /** Update existing data for a new view state */
  updateView(viewState: PointsViewState, session?: UpdateSession): Promise<LoadedPointsData>;

  /** Clean up resources */
  dispose(): void;
}

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

  /** Spatial index for queries (optional, may not exist for 3D data) */
  spatialIndex?: PointsChunkSpatialIndex;

  /** Currently visible point count after nD slicing (updated on view change) */
  visiblePointCount?: number;
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
