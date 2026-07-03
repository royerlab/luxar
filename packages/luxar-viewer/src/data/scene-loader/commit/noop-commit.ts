/**
 * No-op staged commit — the "unchanged data" fast path shared by all
 * three geometry types.
 *
 * The progressive loaders MEMOIZE their LOD concatenation: an update whose
 * view state is unchanged and that loaded no new LODs returns the SAME
 * object reference as the previous update. The commit pipeline stamps that
 * reference onto the mesh (`mesh.userData.committedData`) whenever geometry
 * actually reaches the GPU, so a handler seeing
 * `mesh.userData.committedData === data` knows the GPU already holds this
 * exact data and can skip the expensive process (nD→3D projection) and
 * upload steps entirely.
 *
 * The skip must NOT be a plain `null` staged result: the LOD freshness
 * registry requires `loadedViewVersion` to be re-stamped every update
 * (`scene/lod-freshness.ts` checks exact equality), and the stamp is only
 * written by commits. A `StagedNoopCommit` therefore flows through the
 * atomic commit stage like any staged result — still under the superseded-
 * update abort guard — and the commit helper performs a STAMP-ONLY commit:
 * `loadedViewVersion` is refreshed, geometry is untouched.
 *
 * @module data/scene-loader/commit/noop-commit
 */

/**
 * Stamp-only staged commit: `sourceData` is reference-identical to what the
 * mesh's GPU buffers already hold.
 */
export interface StagedNoopCommit<TData> {
  path: string;
  noop: true;
  /** The loader-returned data reference that matched `committedData`. */
  sourceData: TData;
}

/** Mesh userData slot recording the last data reference committed to the GPU. */
export interface CommittedDataUserData {
  committedData?: unknown;
}

/**
 * True when `data` is reference-identical to the last data committed to
 * this mesh's GPU buffers (see module doc for why identity is sufficient).
 */
export function isAlreadyCommitted(userData: unknown, data: unknown): boolean {
  return (
    userData !== null &&
    typeof userData === 'object' &&
    (userData as CommittedDataUserData).committedData === data
  );
}
