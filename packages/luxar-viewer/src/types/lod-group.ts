import type { BlendingMode } from './blending';
/**
 * LOD-kind Group type definitions for luxar-viewer.
 *
 * A `Group` whose `kind === 'lod'` selects one of N alternative
 * children at runtime based on the projected bbox diagonal in pixels and
 * each child's `coverage_fraction` threshold — a viewport-relative fraction
 * (0..1) the viewer multiplies by the viewport diagonal, so the finest child
 * (coverage 1.0) activates when the object fills the screen. Children are
 * arbitrary geometry subtrees (points / lines / gsplats / nested specialized
 * groups).
 *
 * @module types/lod-group
 */

/**
 * Metadata stored on a kind=`lod` `Group` node's `.zattrs`.
 */
export interface LODGroupMetadata {
  /** Node type identifier. */
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
   * Selector mode. Currently only `"coverage"` is supported; the field
   * is carried in the format so future modes (distance, ...) can be added
   * without breaking existing scenes.
   */
  selector: 'coverage';

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
  blending_mode?: BlendingMode;
  layer?: boolean;
  visible?: boolean;

  /** Optional nD transform (per-dimension affine / permutation). */
  nd_transform?: Record<string, unknown>;

  /** Visibility extension across non-displayed dimensions. */
  extend_to_all?: string[];
}

/**
 * Runtime selector mode held by the registry per lod_group node.
 *
 * `auto` — view-driven coverage-fraction selection (the default).
 * `{ lockLevel: i }` — user has locked to child index `i` (0-based in
 *   coarsest→finest order); the auto selector is bypassed.
 */
export type LODGroupSelectorMode = 'auto' | { lockLevel: number };
