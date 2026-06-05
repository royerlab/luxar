# Scene-Loader Process Step

Async nD→3D projection for Lines and GSplats. This is the **process**
half of the SceneLoader's `process → commit` split: each function in
here runs the per-geometry projection (worker preferred, main-thread
fallback) and returns a `Staged*Commit` payload **without mutating any
mesh**. The matching `scene-loader/commit/commit-*-geometry.ts` helper
then applies those staged buffers to the THREE.js geometry inside the
atomic commit phase.

## Files

| File                        | Role                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data-processor-lines.ts`   | Lines pipeline. Computes per-dim tolerance via `loaders/tolerance-computer`, applies `EXTEND_TO_ALL_TOLERANCE` for any `extend_to_all` dim, then projects via worker (`projectLinesTo3DUsingWorker`) when `useWebWorkers && segmentCount > 1000`, else `projectLinesTo3D` on the main thread. Returns `StagedLinesCommit`.                                                          |
| `data-processor-gsplats.ts` | GSplats pipeline. Reads the mesh's `uTruncate` uniform (or `DEFAULT_TRUNCATE = 3.0`), derives `discreteDims` / `discreteSteps` / `extendToAllDims` from `viewState.dimensions`, projects via worker (`projectGSplatsTo3DUsingWorker`) when `useWebWorkers && splatCount > 1000 && ndim > 3`, packs Cholesky factors via `packCholeskyForShader`, and returns `StagedGSplatsCommit`. |

Both files also **re-export** their sibling commit helper
(`commitLinesGeometry`, `commitGSplatsGeometry`) from
`../commit/commit-*-geometry.ts` so callers that historically imported
the commit function from `data-processor-*` keep working.

## Public surface

```typescript
// Lines
export interface StagedLinesCommit {
  path: string;
  processed: ProcessedLinesData;
}
export async function processLinesData(
  path: string,
  data: LoadedLinesData,
  viewState: LinesViewState,
  rootGroup: THREE.Group | null,
  updateVersion: number,
  session?: UpdateSession
): Promise<StagedLinesCommit | null>;
export async function projectLinesTo3DUsingWorker(
  data: LoadedLinesData,
  viewState: LinesViewState,
  tolerance: readonly number[],
  updateVersion: number
): Promise<ProcessedLinesData>;
export { commitLinesGeometry } from '../commit/commit-lines-geometry';

// GSplats
export interface StagedGSplatsCommit {
  path: string;
  processed: ReturnType<typeof projectGSplats>;
  cholesky01: Float32Array;
  cholesky23: Float32Array;
  cholesky45: Float32Array;
}
export async function processGSplatsData(
  path: string,
  data: LoadedGSplatsData,
  viewState: GSplatsViewState,
  rootGroup: THREE.Group | null,
  updateVersion: number,
  session?: UpdateSession
): Promise<StagedGSplatsCommit | null>;
export async function projectGSplatsTo3DUsingWorker(
  data: LoadedGSplatsData,
  viewState: GSplatsViewState,
  truncate: number,
  updateVersion: number
): Promise<ReturnType<typeof projectGSplats>>;
export { commitGSplatsGeometry } from '../commit/commit-gsplats-geometry';
```

Consumers: `SceneLoader.processLinesData` / `processGSplatsData`
(`data/scene-loader.ts`), `scene-loader/lifecycle/retry.ts`, and the per-node
loaders in `scene-loader/nodes/load-{lines,gsplats}-node.ts` — all of
which reach these functions through the `LoadCtx` interface assembled
in `nodes/build-ctx.ts`, never via direct import.

## Invariants

- **Process never mutates geometry.** Both `processLinesData` and
  `processGSplatsData` return a `Staged*Commit` (or `null` on skip).
  The mesh's buffers are touched only inside the matching
  `commit-*-geometry.ts` helper during the atomic commit phase. This
  is what keeps multi-node updates frame-atomic.
- **Worker thresholds are geometry-specific.** Lines use the worker
  when `segmentCount > 1000`; GSplats add `ndim > 3` because 3D-only
  splat projection is already cheap on the main thread. Both gate on
  `appConfig.dataLoading.performance.useWebWorkers`.
- **Worker failures fall back to the main thread**, except for
  `WorkerAbortError` (dataset-switch) which is re-thrown — running
  stale main-thread work after a switch burns CPU for nothing.
- **First-update logs are gated by `updateVersion <= 1`** so long
  sessions don't spam the console with per-projection info logs.
- **`extend_to_all` is geometry-aware.** Lines mutate the tolerance
  array (`EXTEND_TO_ALL_TOLERANCE` per matching dim name from
  `mesh.userData.attrs.extend_to_all`). GSplats instead pass an
  `extendToAllDims` index list to the worker so the kernel can skip
  hidden-dim attenuation for those axes.
- **No `data-processor-points.ts`.** Points have no async projection
  step — the loader already returns 3D-ready buffers because the
  effective-radius math runs inside the spatial-index loader itself
  (`data/points/`). The points commit goes straight from
  `LoadedPointsData` to `commitPointsGeometry`. See the comment block
  at the top of `../commit/commit-points-geometry.ts` for the full
  rationale and the conditions under which a future
  `data-processor-points.ts` would be added.

## See also

- `../commit/` — atomic GPU commit phase consumed by these stages.
- `../nodes/build-ctx.ts` — assembles the `LoadCtx` that exposes
  `processLinesData` / `processGSplatsData` to the per-node loaders.
- `../lifecycle/retry.ts` — re-runs the same `process → commit` pair
  when a previous attempt failed.
- `../../lines/projection.ts`, `../../gsplats/projection.ts` —
  main-thread projection kernels used as the worker fallback.
- `../../loaders/spatial-query/tolerance-computer.ts` — geometry-aware
  per-dim tolerance used by `processLinesData` (imported via the
  `../../loaders` index).
- `../view-state/extend-tolerance.ts` — `EXTEND_TO_ALL_TOLERANCE` sentinel.
- `../../../workers/worker-pool.ts` — `runWithTimeout` dispatch used
  by both worker projection paths.
- `../../../rendering/gsplat-geometry.ts` — `packCholeskyForShader`
  used to lay out the three `cholesky*` Float32Arrays.
