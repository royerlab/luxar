/**
 * Registration helpers for PickingSystem.
 *
 * Contains pure utilities operating on a registered-pick-node entry
 * without back-references to the orchestrator.
 *
 * @module rendering/picking/picking-system/registration
 */

import * as THREE from 'three';
import { isCameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import { materialManager } from '../../material-manager';

/** Internal tracking of a registered node pair. */
export interface PickNodeEntry {
  main: THREE.Object3D;
  pick: THREE.Object3D;
}

/**
 * True when the node AND all its ancestors are visible. `node.visible`
 * alone misses a hidden ancestor: the LOD-group registry toggles
 * `visible` on the LEVEL object, which can be a GROUP (partition tiles)
 * — the member meshes' own flags stay true. Mirrors the depth-sort
 * coordinator's private helper of the same name (`depth-sort-coordinator.ts`).
 *
 * Load-bearing for the pick pass: a hidden/demoted LOD level must never
 * reach the pick buffer. Rendering it would (a) produce phantom picks —
 * the user hovers what they SEE, not hidden levels — and (b) resurrect a
 * demoted level's pool-released geometry: `renderPickBuffer` force-syncs
 * `pick.geometry = main.geometry`, so a byte-evicted DISPOSED geometry
 * (or another node's adopted pool data) would be re-uploaded under the
 * old pickId.
 */
export function isEffectivelyVisible(node: THREE.Object3D): boolean {
  for (let o: THREE.Object3D | null = node; o; o = o.parent) {
    if (!o.visible) return false;
  }
  return true;
}

/** Dispose any materials on the pick mesh (either a Material or Material[]). */
export function disposePickMaterial(mesh: THREE.Mesh): void {
  const material = mesh.material;
  if (Array.isArray(material)) {
    for (const m of material) m?.dispose?.();
  } else {
    material?.dispose?.();
  }
}

/**
 * Unregister every pick material from `materialManager` without
 * disposing the underlying GPU object. Used after WebGL context-loss
 * — see `PickingSystem.clearRegistrationsForRebuild` for the contract.
 */
export function unregisterAllPickMaterials(nodeMap: ReadonlyMap<number, PickNodeEntry>): void {
  for (const entry of nodeMap.values()) {
    const material = (entry.pick as THREE.Mesh).material;
    const list = Array.isArray(material) ? material : [material];
    for (const m of list) {
      if (m && isCameraAwareMaterial(m)) {
        materialManager.unregister(m);
      }
    }
  }
}
