/**
 * No-op staged commit — the "unchanged data" fast path shared by all
 * three geometry types.
 *
 * The progressive loaders MEMOIZE their LOD concatenation: an update whose
 * view state is unchanged and that loaded no new LODs returns the SAME
 * object reference as the previous update. The commit pipeline stamps that
 * reference onto the mesh (the `committedData` stamp — accessors and
 * contract in `types/committed-data.ts`) whenever geometry actually reaches
 * the GPU, so a handler seeing `isAlreadyCommitted(mesh, data)` knows the
 * GPU already holds this exact data and can skip the expensive process
 * (nD→3D projection) and upload steps entirely.
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

import type * as THREE from 'three';
import { getCommittedData } from '../../../types/committed-data';

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

/**
 * True when `data` is reference-identical to the last data committed to
 * this mesh's GPU buffers (see module doc for why identity is sufficient).
 * Tolerates a missing mesh (node not attached yet) — never committed.
 */
export function isAlreadyCommitted(
  mesh: THREE.Object3D | null | undefined,
  data: unknown
): boolean {
  return mesh != null && getCommittedData(mesh) === data;
}
