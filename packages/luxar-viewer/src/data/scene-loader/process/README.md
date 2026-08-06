# Scene-Loader Process Step

Per-geometry nD→3D projection / staging — one `data-processor-<kind>.ts`
per geometry type, and they are **not** all async or worker-based. This
is the **process** half of the SceneLoader's `process → commit` split:
each function here produces a `Staged*Commit` payload **without mutating
any mesh**, and the matching `scene-loader/commit/commit-*-geometry.ts`
helper then applies those staged buffers to the THREE.js geometry inside
the atomic commit phase. Lines and GSplats run an async projection
(worker preferred, main-thread fallback); Points is a synchronous
staging half (its loader already returns display-space data, so there is
no projection to run); Mesh projects in-process and is `async` only
because backend selection may await the WASM module's first load.

## Files

| File                        | Role                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data-processor-lines.ts`   | Lines pipeline. Computes per-dim tolerance via `loaders/tolerance-computer`, applies `EXTEND_TO_ALL_TOLERANCE` for any `extend_to_all` dim, then projects via worker (`projectLinesTo3DUsingWorker`) when `useWebWorkers && segmentCount > 1000`, else `projectLinesInProcess` (the same dispatcher kernel run in-process, from `workers/data-worker/projection/in-process.ts`). Returns `StagedLinesCommit`.                 |
| `data-processor-gsplats.ts` | GSplats pipeline. Reads the mesh's `uTruncate` uniform (or `DEFAULT_TRUNCATE = 3.0`), derives `discreteDims` / `discreteSteps` / `extendToAllDims` from `viewState.dimensions`, projects via worker (`projectGSplatsTo3DUsingWorker`) when `useWebWorkers && splatCount > 1000 && ndim > 3`, and returns `StagedGSplatsCommit` (the projection's 6-stride `choleskyFactors3D` + fused-scan `bounds` flow through unmodified). |
| `data-processor-points.ts`  | Points pipeline. The **synchronous staging half** — the points loader already returns display-space (3D-ready) data, so there is no worker RPC and no projection output to carry. Exposes staging on its own so callers holding loader output stage it through the same `process → commit` pair lines/gsplats use; returns `StagedPointsCommit`. The no-op reference-identity fast path stays in `commitPointsGeometry`. |
| `data-processor-mesh.ts`    | Mesh pipeline. Runs the in-process display-space projection (`extract_3d_positions`, whole-triangle nD cull, winding post-pass). `async` **only** because backend selection (`pickBackend`) may await the WASM module's first load — not because it uses a worker; mesh projects in-process (whole-node resident). Returns `StagedMeshCommit`. |

The lines and gsplats processors also **re-export** their sibling commit
helper (`commitLinesGeometry`, `commitGSplatsGeometry`) from
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
  // `processed.choleskyFactors3D` (6-stride) flows straight to the GPU
  // commit — no split/re-interleave pass; `processed.bounds` carries the
  // projection's fused-scan cull metadata (AABB + max Cholesky row norm).
  processed: ProcessedGSplatsData;
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
): Promise<ProcessedGSplatsData>;
export { commitGSplatsGeometry } from '../commit/commit-gsplats-geometry';
```

Consumers: `SceneLoader.processLinesData` / `processGSplatsData`
(`data/scene-loader.ts`), `scene-loader/lifecycle/retry.ts`, and the per-node
loaders in `scene-loader/nodes/load-{lines,gsplats}-node.ts` — all of
which reach these functions through the `LoadCtx` interface assembled
in `nodes/build-ctx.ts`, never via direct import.

## Invariants

- **Process never mutates geometry.** Every `process*Data` function
  returns a `Staged*Commit` (or `null` on skip).
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
- **`data-processor-points.ts` is the synchronous staging half.**
  Points have no async projection step — the loader already returns
  display-space (3D-ready) buffers, so there is no worker RPC and no
  projection output to carry. The module exposes the staging half on
  its own so callers that already hold loader output stage it through
  the same `process → commit` pair lines/gsplats use; the no-op
  reference-identity fast path stays in `commitPointsGeometry`. See the
  comment block at the top of `../commit/commit-points-geometry.ts` for
  the full rationale.

## See also

- `../commit/` — atomic GPU commit phase consumed by these stages.
- `../nodes/build-ctx.ts` — assembles the `LoadCtx` that exposes
  `processLinesData` / `processGSplatsData` to the per-node loaders.
- `../lifecycle/retry.ts` — re-runs the same `process → commit` pair
  when a previous attempt failed.
- `../../../workers/data-worker/projection/in-process.ts` —
  `projectLinesInProcess` / `projectGSplatsInProcess`, the same
  dispatcher kernels run in-process as the worker fallback
  (`../../lines/projection.ts` / `../../gsplats/projection.ts` export
  only the empty-payload factories).
- `../../loaders/spatial-query/tolerance-computer.ts` — geometry-aware
  per-dim tolerance used by `processLinesData` (imported via the
  `../../loaders` index).
- `../view-state/extend-tolerance.ts` — `EXTEND_TO_ALL_TOLERANCE` sentinel.
- `../../../workers/worker-pool.ts` — `runWithTimeout` dispatch used
  by both worker projection paths.
