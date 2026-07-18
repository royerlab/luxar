/**
 * The `committedData` mesh stamp — typed accessors for the memoized-concat
 * no-op contract.
 *
 * The progressive loaders MEMOIZE their LOD concatenation: an update whose
 * view state is unchanged and that loaded no new LODs returns the SAME
 * object reference as the previous update. The commit pipeline stamps that
 * reference onto the mesh (`mesh.userData.committedData`) whenever geometry
 * actually reaches the GPU, so a handler seeing
 * `getCommittedData(mesh) === data` knows the GPU already holds this exact
 * data and can skip the expensive process (nD→3D projection) and upload
 * steps entirely.
 *
 * The stamp is load-bearing in BOTH directions:
 * - PRESENCE + identity is the no-op fast-path key (above).
 * - ABSENCE signals LOD demotion: the demoted level's geometry returned to
 *   the evictable pool (and may since belong to another node), so consumers
 *   like the depth-sort coordinator treat a missing stamp as "this mesh's
 *   geometry no longer holds the commit's data".
 * Presence therefore tests `!== undefined`, and clearing DELETES the
 * property (never assigns `undefined`) — the stamp would otherwise keep the
 * loader-returned source arrays reachable.
 *
 * Lives in `types/` (the bottom layer) so both `rendering/` (depth-sort
 * coordinator) and `data/` (commit pipeline, scene loader) share one
 * definition without violating layer direction.
 *
 * @module types/committed-data
 */

import type * as THREE from 'three';

/** Mesh userData slot recording the last data reference committed to the GPU. */
export interface CommittedDataUserData {
  committedData?: unknown;
}

/**
 * True when the object carries a committed-data stamp — i.e. its GPU
 * buffers hold a real commit's data. Absence (`undefined`) is the LOD
 * demotion / never-committed signal.
 */
export function hasCommittedData(obj: THREE.Object3D): boolean {
  return (obj.userData as CommittedDataUserData).committedData !== undefined;
}

/**
 * The last data reference committed to this object's GPU buffers
 * (`undefined` when never committed or demoted). Used for the no-op
 * identity comparison — reference equality is sufficient because the
 * memoized loaders return the same object for unchanged data.
 */
export function getCommittedData(obj: THREE.Object3D): unknown {
  return (obj.userData as CommittedDataUserData).committedData;
}

/**
 * Record `ref` as the data now held by the object's GPU buffers. Written
 * only when geometry actually reaches the GPU (a real, non-noop commit).
 */
export function setCommittedData(obj: THREE.Object3D, ref: unknown): void {
  (obj.userData as CommittedDataUserData).committedData = ref;
}

/**
 * Remove the stamp entirely — DELETE semantics, not `= undefined`: the
 * property's absence is the demotion signal, and deleting also drops the
 * only reference pinning the (potentially huge) source arrays in memory.
 */
export function clearCommittedData(obj: THREE.Object3D): void {
  delete (obj.userData as CommittedDataUserData).committedData;
}
