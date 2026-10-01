/**
 * Pick materials on #2992's shared TSL node graphs: ONE graph per pick
 * configuration (`applySharedTSLGraph`, `materials/_shared/shared-graph-tsl.ts`)
 * instead of one per material, so N pick materials of one configuration cost
 * one WebGPU node build, as their visual twins do.
 *
 * The pick factories end by stamping the pick pass's fixed material state onto
 * the material they build into. Under the shared graph they build into a
 * scratch material once per configuration, so that state is re-applied here on
 * every material: an opaque ID buffer (no blending, depth-tested and written,
 * not tone mapped) whose `opacity` is pinned to exactly 1 — NodeMaterial
 * appends `DiffuseColor.w *= material.opacity` to every fragment, and the
 * element id's HIGH half rides in alpha (see the factories' comments).
 *
 * @module rendering/picking/_shared/shared-pick-graph-tsl
 */
import * as THREE from 'three';
import type { NodeMaterial } from 'three/webgpu';
import { applySharedTSLGraph, type TSLLeafSet } from '../../materials/_shared/shared-graph-tsl';

/** The pick pass's fixed material state (what every pick factory's tail sets). */
export function applyPickMaterialState(material: NodeMaterial): void {
  material.toneMapped = false;
  material.depthTest = true;
  material.depthWrite = true;
  material.transparent = false;
  material.opacity = 1;
  material.blending = THREE.NoBlending;
}

/**
 * Point `material` at the shared pick graph of its configuration (built on
 * first use by `build` into a scratch material) and re-apply the pick state.
 *
 * @param key - everything the factory reads at build time that selects code
 *   (the baked element-texture width, the line projection/join/primitive…);
 *   leaf presence and texture types are added automatically.
 */
export function applySharedPickGraph<T extends TSLLeafSet>(
  material: NodeMaterial,
  family: string,
  key: unknown,
  leaves: T,
  build: (inputs: T, scratch: NodeMaterial) => void
): void {
  applySharedTSLGraph(material, family, key, leaves, build);
  applyPickMaterialState(material);
}
