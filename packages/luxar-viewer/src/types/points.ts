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
import type { ViewState, DataLoader } from '../data/data-loader-types';

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

  /** RGB colors (size: numPoints * 3, optional) */
  colors?: ColorArray;

  /** Point radii in world units (size: numPoints, optional) */
  radii?: ScalarArray;

  /** Point sharpness values (size: numPoints, optional) */
  sharpness?: ScalarArray;

  /**
   * Per-point scalar values for colormap lookup (size: numPoints, optional).
   *
   * when present, the geometry binds a `scalar` attribute and the
   * Point shader's USE_COLORMAP path samples the LUT at
   * `(scalar - uScalarMin) * uScalarScale`. Without scalars, colormap
   * mode falls back to vertex colours. The dtype matches the source
   * zarr array (Float32 / Float16 / Uint8); the shader reads `radius`-
   * style normalised attributes when the dtype is integer.
   */
  scalars?: ScalarArray;

  /**
   * Per-point original (node-global) element IDs (size: numPoints, optional).
   *
   * when picking labels are enabled, this carries the node-global
   * element index that label/image-label loaders expect. Without this,
   * `gl_VertexID` is used — but that's a visible-buffer-local index
   * after spatial range loading or nD compaction, not a global index.
   */
  elementIds?: Uint32Array;

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
 * View state for points loading.
 *
 * Identical shape to the lines + gsplats ViewState — kept as a named
 * alias for documentation (a function signature reading
 * `viewState: PointsViewState` is more self-documenting than the
 * generic `ViewState`).
 *
 * @internal — re-exported via types/index.ts for type-only consumers;
 * no runtime caller depends on this alias.
 */
export type PointsViewState = ViewState;

// ============================================================================
// Scene Integration Types
// ============================================================================

/**
 * Data loader interface for Points nodes. Identical shape to the
 * `DataLoader` interface in `data/data-loader-types.ts`; aliased for
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
