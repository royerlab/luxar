/**
 * Core types and interfaces for the new data loading architecture.
 *
 * This module defines the clean abstractions for loading nD points data
 * with proper spatial indexing support and aligned attribute loading.
 */

import { SimpleDims } from '../types/dims';
import * as THREE from 'three';

/**
 * Represents the current view state for data loading.
 * This determines what portion of the nD dataset should be loaded.
 */
export interface ViewState {
  /** Which dimensions to display (max 3, indices into nD space) */
  displayDims: number[];

  /** Current position in nD space (one value per dimension) */
  slicePosition: number[];

  /** Tolerance for slicing in each dimension (radius in non-displayed dims, 0 for displayed) */
  tolerance: number[];

  /** Optional camera frustum for view-dependent loading */
  cameraFrustum?: THREE.Frustum;

  /** Dimension metadata for the dataset */
  dimensions?: SimpleDims;
}

/**
 * Supported TypedArray types for points attributes
 * Note: Float16Array is supported in modern browsers (2024+)
 * We include it in the type but handle fallback at runtime
 */
export type PositionArray = Float32Array | Float16Array;
export type ColorArray = Float32Array | Uint8Array | Uint16Array;
export type ScalarArray = Float32Array | Float16Array | Uint8Array;

/**
 * Points data ready for GPU rendering.
 * All arrays are properly aligned with the same point ordering.
 * Arrays can be in different data types for memory efficiency.
 */
export interface PointsData {
  /** 3D positions extracted from nD space (size: numPoints * 3) */
  positions: PositionArray;

  /** RGB colors (size: numPoints * 3, optional) */
  colors?: ColorArray;

  /** Point radii in world units (size: numPoints, optional) */
  radii?: ScalarArray;

  /** Point sharpness values (size: numPoints, optional) */
  sharpness?: ScalarArray;

  /** Metadata about the loaded data */
  metadata: {
    /** Total points in the full dataset */
    totalPoints: number;

    /** Number of points actually loaded */
    loadedPoints: number;

    /** Bounding box of loaded points */
    bounds: THREE.Box3;

    /** Original nD dimensionality */
    ndim: number;

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
    };
  };
}

/**
 * Core interface for data loaders.
 * Implementations handle different loading strategies (spatial index vs fallback).
 */
export interface DataLoader {
  /** Load points data for the given view state */
  loadPoints(viewState: ViewState): Promise<PointsData>;

  /** Update existing data for a new view state */
  updateView(viewState: ViewState): Promise<PointsData>;

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
}

/**
 * Range of points to load (for spatial index queries)
 */
export interface PointRange {
  /** Starting index (inclusive) */
  start: number;

  /** Ending index (exclusive) */
  end: number;
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
    /** Transformation matrix (16 elements for 4x4 matrix) */
    transform?: number[];

    /** Rendering attributes */
    opacity?: number;
    gamma?: number;
    blending_mode?: string;
    max_radius?: number;
    num_points?: number;

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
