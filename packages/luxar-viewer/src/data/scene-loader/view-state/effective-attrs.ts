/**
 * Compose effective rendering attributes for a scene-graph node.
 *
 * Pure helper: given a scene graph and a node, return a copy of the
 * node's attrs with `opacity`, `absorption`, `gamma`, `intensity`,
 * `offset`, `blending_mode`, `join`, `layer_order`, and `colormap` (with its
 * `customLutBytes`) replaced by the values from
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
    // `colormap` used to survive only inside the `...node.attrs` spread, so
    // every consumer read the node's OWN value and an ancestor-authored
    // palette never arrived (#1600). The LUT bytes come from the same node the
    // winning name did — never mixed with this node's own leftovers.
    colormap: eff.colormap,
    customLutBytes: eff.customLutBytes,
    // The authored cross-layer draw order. Each `create-*-node.ts` factory
    // picks this composed value from its effective attrs argument and stamps
    // it onto the mesh's dedicated `userData.layerOrder` render-state slot
    // (`LAYER_ORDER_SPEC.md` §7).
    layer_order: eff.layer_order,
  };
}
