/**
 * Mesh data-processor concern — the Mesh sibling of
 * `data-processor-lines.ts` / `data-processor-gsplats.ts`.
 *
 * Runs the display-space projection: `extract_3d_positions`, the whole-triangle nD
 * cull, and the winding post-pass (`data/mesh/projection.ts`). Unlike its two
 * async siblings there is no worker RPC — mesh projects **in-process**, and for a
 * different reason than Points does.
 *
 * Points runs in-process because its projection is memory-bound per update, so
 * offloading would pay transfer cost both ways for negligible compute savings.
 * Mesh runs in-process because it is whole-node resident, so its payload is ONE
 * large transfer rather than lines' many small ones — a genuinely different
 * benefit profile that should be measured before a worker is built for it, not
 * assumed. That keeps the measure-first performance doctrine intact rather than
 * copying the lines shape because it exists.
 *
 * It is still `async`, because selecting the backend is (`pickBackend` may await
 * the WASM module's first load).
 *
 * @module data/scene-loader/process/data-processor-mesh
 */

import { getMeshBackend } from '../../../workers/data-worker/projection/in-process';
import {
  projectMesh,
  noticeUndecidableWinding,
  type ProjectedMeshData,
} from '../../mesh/projection';
import type { LoadedMeshData, MeshMetadata, MeshViewState } from '../../../types/mesh';

/** Everything `commitMeshGeometry` needs, staged and ready for GPU upload. */
export interface StagedMeshCommit {
  path: string;
  /** The whole loaded mesh — held so the commit can read colours and counts. */
  data: LoadedMeshData;
  /** The projection for this epoch. */
  projected: ProjectedMeshData;
}

/**
 * Nodes already warned about undecidable winding.
 *
 * Module-scoped so the notice is once per node for the lifetime of the tab rather
 * than once per rebuild — the projection runs on every slice move, and a per-call
 * warning would flood the console during a scrub. A `Set` of paths is enough: paths
 * are unique per scene, and re-warning after a dataset switch is harmless.
 */
const noticedWinding = new Set<string>();

/**
 * Project loaded mesh data for the given view state and stage it for commit.
 *
 * @param attrs - The node's metadata; `normal_dims` supplies the winding frame and
 *   `double_sided` the authored side.
 */
export async function processMeshData(
  path: string,
  data: LoadedMeshData,
  viewState: MeshViewState,
  attrs: Pick<MeshMetadata, 'normal_dims' | 'double_sided'>
): Promise<StagedMeshCommit> {
  const backend = await getMeshBackend(data.ndim);
  const projected = projectMesh(data, viewState, attrs.normal_dims, attrs.double_sided, backend);

  // Reported here rather than inside `resolveWinding`, which stays pure so it can
  // be called on every index build without a logging side effect.
  if (projected.undecidableReason) {
    noticeUndecidableWinding(path, projected.undecidableReason, noticedWinding);
  }

  return { path, data, projected };
}

/** Test seam: forget which nodes have been warned about. */
export function resetWindingNoticesForTesting(): void {
  noticedWinding.clear();
}
