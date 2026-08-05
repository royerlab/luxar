/**
 * Attribute composition across the scene graph hierarchy.
 *
 * Rendering attributes compose from root to leaf rather than override:
 *
 *   effective_opacity    = clamp(∏ opacity_i,  0, 1)
 *   effective_absorption = max(0, ∏ absorption_i)   (volumetric κ; identity 1)
 *   effective_gamma      = clamp(∏ gamma_i,    0.1, 10)
 *   effective_intensity  = max(0, ∏ intensity_i)
 *   effective_offset     = Σ offset_i
 *   effective_blending   = nearest ancestor (root-to-leaf) that sets blending_mode,
 *                          else undefined (each consumer applies its own per-type
 *                          default: `additive` for points/lines/gsplats, `opaque`
 *                          for mesh — docs/specs/MESH_NODE_SPEC.md §6.3)
 *
 * Note: the offset composition is additive per the spec. This is mathematically
 * different from chaining the shader's `color * I + O` model through successive
 * nodes (which would yield `O_parent * I_child + O_child`). The additive form
 * is simpler to reason about at the authoring layer and is what the Python
 * API documents; viewer and spec match intentionally.
 */

import type { SceneNode } from './data-loader-types';
import { clamp } from '../utils/clamp';
import { normalizeBlendingMode } from '../rendering/blending-state';
import type { BlendingMode } from '../types/blending';
import { isGeometryType } from '../types/geometry-capabilities';

export interface ComposableAttrs {
  opacity?: number;
  /** Absorption coefficient κ (volumetric mode); multiplicative, identity 1. */
  absorption?: number;
  gamma?: number;
  intensity?: number;
  offset?: number;
  /** Raw (unvalidated) mode string as authored; normalized at compose time. */
  blending_mode?: string;
}

export interface EffectiveAttrs {
  opacity: number;
  /** Composed absorption coefficient κ (≥ 0; only read in volumetric mode). */
  absorption: number;
  gamma: number;
  intensity: number;
  offset: number;
  /**
   * `undefined` when no level of the chain set a mode (each consumer applies
   * its own per-type default); otherwise canonical — a set value (even a
   * malformed one) is run through `normalizeBlendingMode`.
   */
  blending_mode: BlendingMode | undefined;
}

/**
 * Compose a root-to-leaf chain of attribute records.
 *
 * The chain is ordered from the outermost ancestor (root) to the leaf node
 * whose effective attributes we want. Unset values are treated as identity:
 * opacity/gamma/intensity = 1, offset = 0. A set `blending_mode` at any
 * level overrides the cumulative choice; a set winning string is validated
 * through `normalizeBlendingMode` (unknown string → 'normal' + one-time
 * warning), so every consumer sees a canonical mode. An unset chain yields
 * `undefined` — each consumer then applies its own per-type default (spec
 * §6.3: `additive` for points/lines/gsplats, `opaque` for mesh).
 */
export function composeAttrs(chainRootToLeaf: readonly ComposableAttrs[]): EffectiveAttrs {
  let opacity = 1.0;
  let absorption = 1.0;
  let gamma = 1.0;
  let intensity = 1.0;
  let offset = 0.0;
  let blending_mode: string | undefined;

  for (const a of chainRootToLeaf) {
    if (a.opacity !== undefined) opacity *= a.opacity;
    if (a.absorption !== undefined) absorption *= a.absorption;
    if (a.gamma !== undefined) gamma *= a.gamma;
    if (a.intensity !== undefined) intensity *= a.intensity;
    if (a.offset !== undefined) offset += a.offset;
    if (a.blending_mode !== undefined) blending_mode = a.blending_mode;
  }

  // Clamp per spec
  opacity = clamp(opacity, 0, 1);
  absorption = Math.max(0, absorption); // κ is unbounded above
  gamma = clamp(gamma, 0.1, 10);
  intensity = Math.max(0, intensity);

  return {
    opacity,
    absorption,
    gamma,
    intensity,
    offset,
    // Preserve the unset state: only a truly-unset chain yields `undefined`
    // (consumers apply their per-type default). A set-but-malformed value
    // (e.g. '') is NOT undefined, so it still normalizes → 'normal'.
    blending_mode: blending_mode === undefined ? undefined : normalizeBlendingMode(blending_mode),
  };
}

/**
 * Walk from `root` toward the node whose path equals `targetPath`,
 * returning the chain of SceneNodes in root-to-leaf order. The scene-root
 * node is included iff its own `type` is not `'scene'` — scene roots are
 * carriers-only and should contribute no rendering attrs.
 *
 * This implementation is path-format-agnostic: it matches children by
 * direct prefix comparison on the already-stored `path` field, so it
 * works for both the leading-slash format produced by the real scene
 * loader (`'/'`, `'/group/pts'`) and the plain format used in unit tests
 * (`''`, `'group/pts'`).
 */
export function collectAncestorNodes(root: SceneNode, targetPath: string): SceneNode[] {
  const chain: SceneNode[] = [];

  // Empty target means "nothing requested" — return nothing even if the
  // root itself is a data node (callers that want root should pass its
  // path explicitly).
  if (!targetPath) return chain;

  // Target equals root: include root iff it carries attrs (non-scene).
  if (targetPath === root.path) {
    if (root.type !== 'scene') chain.push(root);
    return chain;
  }

  if (root.type !== 'scene') chain.push(root);

  let cursor: SceneNode = root;
  while (cursor.path !== targetPath) {
    const next: SceneNode | undefined = cursor.children?.find(
      (c) => c.path === targetPath || targetPath.startsWith(c.path + '/')
    );
    if (!next) break;
    chain.push(next);
    cursor = next;
  }

  return chain;
}

/**
 * Walk the scene graph from `root` toward `targetPath`, collecting the
 * composable attrs of every node on the path.
 */
export function collectAncestorAttrs(root: SceneNode, targetPath: string): ComposableAttrs[] {
  return collectAncestorNodes(root, targetPath).map((n) => toComposable(n.attrs));
}

/**
 * Convenience: compose effective attrs for a target path in the scene graph.
 */
export function getEffectiveAttrs(root: SceneNode, targetPath: string): EffectiveAttrs {
  return composeAttrs(collectAncestorAttrs(root, targetPath));
}

/**
 * Collect every data-leaf descended from `start`.
 * Used by group-layer controls that need to fan out to actual materials.
 *
 * "Data-leaf" is the whole geometry vocabulary, not a fixed list: a group-level
 * opacity / gamma / colormap control must reach every leaf underneath it, so a
 * geometry type omitted here would silently ignore its ancestors' attributes.
 */
export function collectDataDescendants(start: SceneNode): SceneNode[] {
  const result: SceneNode[] = [];
  const visit = (n: SceneNode): void => {
    if (isGeometryType(n.type)) {
      result.push(n);
    }
    if (n.children) for (const c of n.children) visit(c);
  };
  visit(start);
  return result;
}

function toComposable(attrs: SceneNode['attrs']): ComposableAttrs {
  return {
    opacity: attrs.opacity as number | undefined,
    absorption: attrs.absorption as number | undefined,
    gamma: attrs.gamma as number | undefined,
    intensity: attrs.intensity as number | undefined,
    offset: attrs.offset as number | undefined,
    blending_mode: attrs.blending_mode as string | undefined,
  };
}
