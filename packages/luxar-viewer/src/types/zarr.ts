/**
 * Type definitions for Zarr store attributes and metadata.
 *
 * These types provide proper typing for Zarr group and array attributes,
 * eliminating the need for 'as any' type assertions throughout the codebase.
 */

/**
 * Scene-level dimension information stored in Zarr attributes
 */
export interface SceneDimensionAttrs {
  dimensions: Array<{
    name: string;
    unit: string;
    scale?: number;
    range?: [number, number];
    display: boolean;
    discrete?: boolean;
    step?: number;
    cyclic?: boolean;
    spatial?: boolean;
    description?: string;
  }>;
}

/**
 * Zarr group attributes for the root scene
 */
export interface ZarrSceneAttrs {
  /** Scene format version */
  luxar_version?: string;

  /** Scene-level dimensions */
  scene_dimensions?: SceneDimensionAttrs;

  /** Scene type identifier */
  type?: 'scene' | string;

  /** Physical units */
  units?: string;

  /** Any additional metadata */
  [key: string]: unknown;
}

/**
 * Zarr group attributes for nodes in the scene graph
 */
export interface ZarrNodeAttrs {
  /** Node type (points, group, etc.) */
  type?: 'points' | 'group' | string;

  /** Transformation matrix (16 elements for 4x4 matrix) */
  transform?: number[];

  /** Rendering attributes */
  opacity?: number;
  gamma?: number;
  blending_mode?: string;
  point_size?: number;
  depth_test?: boolean;
  depth_write?: boolean;

  /** Points metadata */
  num_points?: number;
  max_radius?: number;

  /** Broadcasting dimensions */
  broadcast_dims?: string[];

  /** Arrays in this group */
  arrays?: string[];

  /** Physical units */
  units?: string;

  /** Any additional attributes */
  [key: string]: unknown;
}

/**
 * Zarr store interface with contents method
 */
export interface ZarrStoreWithContents {
  /** List contents of the store */
  contents(): Promise<Array<{ path: string; kind: 'group' | 'array' }>>;

  /** Other store properties */
  [key: string]: unknown;
}

/**
 * Type guard to check if a store has contents method
 */
export function hasContentsMethod(store: unknown): store is ZarrStoreWithContents {
  return (
    typeof store === 'object' &&
    store !== null &&
    typeof (store as Record<string, unknown>).contents === 'function'
  );
}

/**
 * Type guard to check if attributes contain a transform
 */
export function hasTransform(
  attrs: ZarrNodeAttrs
): attrs is ZarrNodeAttrs & { transform: number[] } {
  return (
    attrs.transform !== undefined && Array.isArray(attrs.transform) && attrs.transform.length === 16
  );
}

/**
 * Type guard to check if attributes are for a points node
 */
export function isPointsNode(attrs: ZarrNodeAttrs): boolean {
  return attrs.type === 'points';
}

/**
 * Type guard to check if attributes contain scene dimensions
 */
export function hasSceneDimensions(
  attrs: ZarrSceneAttrs
): attrs is ZarrSceneAttrs & { scene_dimensions: SceneDimensionAttrs } {
  return attrs.scene_dimensions !== undefined && attrs.scene_dimensions.dimensions !== undefined;
}
