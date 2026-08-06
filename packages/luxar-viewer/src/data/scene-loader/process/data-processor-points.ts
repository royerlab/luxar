/**
 * Points data-processor concern — the Points sibling of
 * `data-processor-lines.ts` / `data-processor-gsplats.ts`.
 *
 * Points differ from the other two in where projection happens: their loader
 * already returns display-space data, so there is no worker RPC and no
 * projection output to carry. `data/points/handler.ts::pointsLoadAndStage`
 * therefore loads and stages in one step, and owns the
 * {@link StagedPointsCommit} shape.
 *
 * This module exposes the *staging half* of that on its own, so callers that
 * already hold loader output — `nodes/load-points-node.ts` and
 * `lifecycle/retry.ts`, which each run their own `updateView` — can stage it
 * through the same `process` → `commit` pair that lines and gsplats use. That
 * is what lets `NodeBuildCtx` carry one uniform pair per geometry kind rather
 * than a points-only one-shot alongside two pairs.
 *
 * The staged type is deliberately re-exported from the handler rather than
 * redeclared here: a second same-named type with a different field would type
 * check (the two never meet) while silently diverging.
 *
 * Note the no-op fast path stays in `commitPointsGeometry`. `handler.ts`
 * documents why — lines and gsplats must short-circuit *before* their worker
 * projection, whereas points has no projection to skip, so its
 * reference-identity check lives at the commit, where it also covers the
 * atomic-commit, refinement, retry and lazy paths uniformly.
 *
 * @module data/scene-loader/process/data-processor-points
 */

import type { LoadedPointsData } from '../../data-loader-types';
import type { StagedPointsCommit } from '../../points/handler';

export type { StagedPointsCommit };

/**
 * Stage loader-returned points data for commit.
 *
 * Synchronous and non-nullable, unlike the lines/gsplats processors: those are
 * async because they await a worker projection, and nullable because they take
 * the `rootGroup` and look the target `THREE.Mesh` up themselves (a missing root
 * or wrong `nodeType` has nothing to project against). This one takes neither —
 * the caller has already resolved the node — so widening it to match would force
 * callers to `await` and null-check something that can never be pending or absent.
 */
export function processPointsData(path: string, data: LoadedPointsData): StagedPointsCommit {
  return { path, data };
}
