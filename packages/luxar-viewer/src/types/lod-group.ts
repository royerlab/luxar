/**
 * LOD-kind Group type definitions for luxar-viewer.
 *
 * A kind=`lod` `Group` scene-graph node selects one of N alternative
 * children at runtime based on the projected bbox diagonal in pixels and
 * each child's `min_pixel_size` threshold. Children are arbitrary
 * geometry subtrees (points / lines / gsplats / nested specialized groups).
 *
 * On disk, the node has `type: "group"` (no longer a distinct `lod_group`
 * type) and an additional `kind: "lod"` discriminant attribute. The
 * collapse from `LODGroup` subclass to flagged `Group` lets future
 * specialized-group kinds slot in without a new node type per kind.
 *
 * @module types/lod-group
 */

/**
 * Metadata stored on a kind=`lod` `Group` node's `.zattrs`.
 */
export interface LODGroupMetadata {
  /** Node type identifier. Always `"group"` after the kind-flag refactor. */
  type: 'group';

  /** Specialized-group discriminant. */
  kind: 'lod';

  /**
   * Geometry type the user sees this layer as. Resolved at write time
   * (typically the finest child's type, walking through nested
   * specialized groups). The viewer uses this for the layers-panel label.
   */
  display_type?: 'points' | 'lines' | 'gsplats';

  /**
   * Selector mode. Currently only `"pixel_size"` is supported; the field
   * is carried in the format so future modes (distance, screen-coverage,
   * ...) can be added without breaking existing scenes.
   */
  selector: 'pixel_size';

  /**
   * Initial active level index for the manual-override UI. Stored 0-based
   * in **child insertion order** (i.e., coarsest-first). The runtime
   * defaults to `auto` (view-driven) selection; this value only seeds
   * the override widget.
   */
  default_level?: number;

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

/**
 * Per-child attribute that must appear on each `lod_group` child's
 * `.zattrs`. Strictly monotonic increasing in coarsest→finest order;
 * conventionally `0.0` on the coarsest child.
 *
 * The selector picks the **finest** child whose `min_pixel_size`
 * threshold is satisfied by the current projected bbox diagonal in
 * pixels.
 */
export type ChildMinPixelSize = number;

/**
 * Runtime selector mode held by the registry per lod_group node.
 *
 * `auto` — view-driven pixel-size selection (the default).
 * `{ lockLevel: i }` — user has locked to child index `i` (0-based in
 *   coarsest→finest order); the auto selector is bypassed.
 */
export type LODGroupSelectorMode = 'auto' | { lockLevel: number };

/**
 * Type guard for `LODGroupMetadata`. Matches the new `{ type: "group",
 * kind: "lod" }` shape on disk.
 */
export function isLODGroupMetadata(attrs: unknown): attrs is LODGroupMetadata {
  if (typeof attrs !== 'object' || attrs === null) return false;
  const record = attrs as Record<string, unknown>;
  return record.type === 'group' && record.kind === 'lod';
}
