/**
 * Partition-kind Group type definitions for luxar-viewer.
 *
 * A `Group` whose `kind === 'partition'` is a compile-time decomposition of
 * a single large geometry node (10M+ elements) into multiple smaller
 * child nodes for per-child frustum culling, per-child LOD, etc. The
 * user does not see the decomposition: the layers panel presents one
 * logical layer of `display_type`. All children render simultaneously
 * (no per-frame selector — unlike `kind === 'lod'`).
 *
 * @module types/partition-group
 */

/**
 * Metadata stored on a kind=`partition` `Group` node's `.zattrs`.
 */
export interface PartitionGroupMetadata {
  /** Node type identifier. */
  type: 'group';

  /** Specialized-group discriminant. */
  kind: 'partition';

  /**
   * Geometry type the user sees this layer as. All children must
   * resolve to this same display type (homogeneity is mandatory for
   * Partition — you cannot decompose a single logical layer into mixed-type
   * parts).
   */
  display_type: 'points' | 'lines' | 'gsplats';

  /**
   * Per-part element cap that drove the BSP recursion. Recorded for
   * diagnostics and for future partition-aware tools.
   */
  max_elements: number;

  /**
   * Union of children's `position_bounds`. Lets picking / framing /
   * scene-bounds-cache treat the layer as one logical entity rather
   * than as N independent meshes.
   */
  position_bounds?: {
    min: number[];
    max: number[];
  };

  /** 4x4 transform matrix (column-major for THREE.js). */
  transform?: number[];

  /** Optional compositing rendering attrs (inherited via scene-graph composition). */
  opacity?: number;
  gamma?: number;
  intensity?: number;
  offset?: number;
  blending_mode?: 'additive' | 'normal' | 'max' | 'opaque' | 'luminous';
  layer?: boolean;
  visible?: boolean;

  /** Optional nD transform (per-dimension affine / permutation). */
  nd_transform?: Record<string, unknown>;

  /** Visibility extension across non-displayed dimensions. */
  extend_to_all?: string[];
}

