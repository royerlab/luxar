/**
 * Compose effective rendering attributes for a scene-graph node.
 *
 * Pure helper: given a scene graph and a node, return a copy of the
 * node's attrs with `opacity`, `absorption`, `gamma`, `intensity`,
 * `offset`, `blending_mode`, and `join` replaced by the values from
 * {@link getEffectiveAttrs}. Falls back to the raw attrs when the
 * scene graph is unavailable.
 *
 * Centralized here so the scene loader's class method becomes a
 * one-line delegation and the composition rules can be tested
 * without touching the loader.
 *
 * @module data/scene-loader/view-state/effective-attrs
 */

import { getEffectiveAttrs } from '../../attrs-composer';
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
  const eff = getEffectiveAttrs(sceneGraph, node.path);
  return {
    ...node.attrs,
    opacity: eff.opacity,
    absorption: eff.absorption,
    gamma: eff.gamma,
    intensity: eff.intensity,
    offset: eff.offset,
    blending_mode: eff.blending_mode,
    join: eff.join,
  };
}
