/**
 * The `committedData` mesh stamp — typed accessors for the memoized-concat
 * no-op contract — plus its sibling `elementIdMap` stamp.
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
 * **The `elementIdMap` sibling stamp.** Picking needs to translate the
 * STORAGE SLOT its shaders report (a position in the buffer now on the GPU)
 * into the ON-DISK element index the per-element label CSR is keyed by. That
 * map is a property of the committed buffers, not of any loader payload, so
 * it lives here as a second mesh-level stamp rather than riding on the data
 * object — a loaded payload can be a SliceCache-owned snapshot whose byte size
 * was measured at store time, and commit-time mutation of it would both
 * falsify that accounting and break the cache's never-mutated invariant.
 * Contract:
 * - indexed by STORAGE slot; the value is the on-disk element index,
 * - written in LOCKSTEP with `committedData` so it always describes the
 *   buffers currently on the GPU (and cleared with it whenever those buffers
 *   are actually released — see `clearCommittedData` vs
 *   `invalidateCommittedDataStamp`),
 * - OPTIONAL: absent ⇒ the identity holds, the slot IS the on-disk index.
 *
 * Lives in `types/` (the bottom layer) so both `rendering/` (depth-sort
 * coordinator, picking) and `data/` (commit pipeline, scene loader) share one
 * definition without violating layer direction.
 *
 * @module types/committed-data
 */

import type * as THREE from 'three';

/**
 * Mesh userData slots recording the last data reference committed to the GPU
 * and the slot → on-disk element index map describing those same buffers.
 */
export interface CommittedDataUserData {
  committedData?: unknown;
  elementIdMap?: Uint32Array;
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
 * Record the slot → on-disk element index map describing the buffers this
 * commit just uploaded, or clear it when the commit produced none.
 *
 * Called next to {@link setCommittedData} on every real commit — passing
 * `undefined` DELETES the key, so a previous commit's map can never outlive
 * the geometry it described (a stale map is a silently WRONG label, strictly
 * worse than falling back to the slot).
 */
export function setElementIdMap(obj: THREE.Object3D, map: Uint32Array | undefined): void {
  if (map) {
    (obj.userData as CommittedDataUserData).elementIdMap = map;
  } else {
    delete (obj.userData as CommittedDataUserData).elementIdMap;
  }
}

/**
 * The slot → on-disk element index map for the buffers currently on the GPU,
 * or `undefined` when none applies (identity: slot IS the on-disk index).
 * Type-checks the stored value so a malformed stamp reads as absent rather
 * than throwing on the hover path.
 */
export function getElementIdMap(obj: THREE.Object3D): Uint32Array | undefined {
  const map = (obj.userData as CommittedDataUserData).elementIdMap;
  return map instanceof Uint32Array ? map : undefined;
}

/**
 * Remove the stamp entirely — DELETE semantics, not `= undefined`: the
 * property's absence is the demotion signal, and deleting also drops the
 * only reference pinning the (potentially huge) source arrays in memory.
 *
 * For use when the geometry itself is GONE (LOD demotion returned it to the
 * evictable pool, or the whole scene is being torn down). Drops the
 * `elementIdMap` sibling too: the map describes the very buffers whose stamp
 * is being dropped, so keeping it would leave picking resolving slots through
 * a map for geometry that is no longer vouched for. To merely defeat the
 * no-op gate on geometry that STAYS on the GPU, use
 * {@link invalidateCommittedDataStamp} instead.
 */
export function clearCommittedData(obj: THREE.Object3D): void {
  delete (obj.userData as CommittedDataUserData).committedData;
  delete (obj.userData as CommittedDataUserData).elementIdMap;
}

/**
 * Drop ONLY the no-op memoization stamp, keeping the `elementIdMap` sibling.
 *
 * For the caller that is forcing a re-process of geometry which REMAINS on
 * the GPU and pickable — `rendering/depth-sort-coordinator.ts`'s blending-mode
 * switch, which clears the stamp purely so the memoized-concat fast path
 * cannot skip re-projection. The buffers are untouched, so the map still
 * describes them exactly; dropping it would make every hover in the (async)
 * window before the reprocess commits fall back to the raw slot — the
 * silently-wrong label this map exists to prevent. The next real commit
 * rewrites both in lockstep.
 */
export function invalidateCommittedDataStamp(obj: THREE.Object3D): void {
  delete (obj.userData as CommittedDataUserData).committedData;
}
