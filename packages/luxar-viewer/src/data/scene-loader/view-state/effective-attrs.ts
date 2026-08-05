/**
 * Compose effective rendering attributes for a scene-graph node.
 *
 * Pure helper: given a scene graph and a node, return a copy of the
 * node's attrs with `opacity`, `absorption`, `gamma`, `intensity`,
 * `offset`, and `blending_mode` replaced by the values from
 * {@link getEffectiveAttrs}. Falls back to the raw attrs when the
 * scene graph is unavailable.
 *
 * The composed `blending_mode` uses the NODE'S OWN type default when nothing in the
 * ancestry set one — `additive` for points/lines/gsplats, `opaque` for mesh.
 *
 * Centralized here so the scene loader's class method becomes a
 * one-line delegation and the composition rules can be tested
 * without touching the loader.
 *
 * @module data/scene-loader/view-state/effective-attrs
 */

import { getEffectiveAttrs } from '../../attrs-composer';
import { defaultBlendingModeFor } from '../../../types/geometry-capabilities';
import type { SceneNode } from '../../data-loader-types';

/**
 * Return a node-attrs record with rendering attributes replaced by
 * the effective values composed along the scene-graph ancestry
 * (root → leaf).
 */
export function applyEffectiveAttrs(
  sceneGraph: SceneNode | null | undefined,
  node: SceneNode
): SceneNode['attrs'] {
  if (!sceneGraph) return node.attrs;
  // The node's own type supplies the fallback for an ancestry that sets no mode
  // (§6.3): `additive` for the three emissive types, `opaque` for mesh. It has to be
  // decided HERE, at the compose call, because `composeAttrs` is the last place that
  // knows the difference between "unset" and "set to additive" —
  // `normalizeBlendingMode(undefined)` collapses the two, which is what made
  // `createMeshNode`'s `?? 'opaque'` dead code and left every mesh rendering additive.
  const eff = getEffectiveAttrs(sceneGraph, node.path, defaultBlendingModeFor(node.type));
  return {
    ...node.attrs,
    opacity: eff.opacity,
    absorption: eff.absorption,
    gamma: eff.gamma,
    intensity: eff.intensity,
    offset: eff.offset,
    blending_mode: eff.blending_mode,
  };
}
