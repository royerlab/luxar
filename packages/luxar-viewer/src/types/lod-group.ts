import type { BlendingMode } from './blending';
import type { LodSelectorName } from './format-contract';
/**
 * LOD-kind Group type definitions for luxar-viewer.
 *
 * A `Group` whose `kind === 'lod'` selects one of N alternative children at
 * runtime by comparing the group's on-screen size against each child's
 * `coverage_fraction` threshold. The group's `selector` attr names the UNITS
 * of those thresholds:
 *
 * - `'screen-area'` (what every derived ladder stamps): a threshold is a
 *   literal screen-area fraction — the group's projected bbox rect area over
 *   the viewport area. The derived whole-object ladder is [0, …, 1/8, 1/4,
 *   1/2] (full detail while the node occupies at least half the screen, one
 *   level coarser per halving of occupied area); a partition tile anchors at
 *   1.0 (the tile alone fills the screen).
 * - `'coverage'` (legacy, and explicit `coverage_fractions=[...]` lists): the
 *   diagonal metric — projected bbox diagonal / (FILL_FACTOR=0.5 ×
 *   min(viewport.width, viewport.height), the fitted screen axis — see the
 *   FILL_FACTOR doc in `scene/lod-group-registry.ts`), thresholds in 0..4
 *   (whole-object derived ladders spanned 0..1, partition-bound ones reach
 *   4.0).
 *
 * Children are arbitrary geometry subtrees (points / lines / gsplats /
 * nested specialized groups).
 *
 * The Python mirror of the selector vocabulary is
 * `luxar.typing_utils.constants.LOD_SELECTORS`; keep the spellings in step.
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
   *
   * Spelled out rather than `GeometryTypeName`: this is the LOD-CAPABLE subset
   * of the vocabulary, so do not widen it when a geometry type is added —
   * declare that type's `lod` capability in `types/geometry-capabilities`
   * instead, and extend this union only if it comes out `true`.
   */
  display_type?: 'points' | 'lines' | 'gsplats' | 'mesh';

  /**
   * Units of the children's `coverage_fraction` thresholds (see the module
   * doc): `'screen-area'` = literal screen-area fractions (what derived
   * ladders stamp), `'coverage'` = the legacy diagonal metric. An unknown /
   * missing value falls back to `'coverage'` so older stores keep rendering.
   * Single-sourced from `format-contract/contract.yaml::lod_selectors`.
   */
  selector: LodSelectorName;

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
  absorption?: number;
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
