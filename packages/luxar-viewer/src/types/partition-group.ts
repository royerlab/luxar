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
 * A node of the serialized BSP tree (`bsp_tree` partition attr).
 *
 * An INTERNAL node carries the split plane: `axis` (a spatial axis `0`/`1`/`2`)
 * and `split` coordinate (in the parts' local center space), with `left` = the
 * side where `coord < split` and `right` = `coord >= split`. A LEAF node holds
 * `part`, the index of the `part_<i>` child it represents.
 *
 * Because the parts are BSP cells, the tree gives an EXACT back-to-front
 * ordering for painter's-algorithm (alpha-over) compositing, correct for any
 * camera pose incl. inside the volume: at each node recurse the far side of
 * `split` first. See {@link ../../rendering/depth-sort-coordinator}.
 */
export type BspTreeNode =
  | { part: number; axis?: undefined }
  | { axis: number; split: number; left: BspTreeNode; right: BspTreeNode; part?: undefined };

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

  /**
   * Split-plane record of the BSP that produced the parts (present only when
   * they came from a single `to_spatial_partition` split — the `tiles` /
   * `adaptive` recipes; absent for streamed grid/content merges). Enables the
   * depth-sort coordinator's exact back-to-front part ordering; absent → it
   * falls back to a per-part centroid-distance heuristic. See {@link BspTreeNode}.
   */
  bsp_tree?: BspTreeNode;

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
