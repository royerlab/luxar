/**
 * GSplats data-processor concern extracted from `scene-loader.ts`.
 *
 * Three pure-ish functions for GSplat projection and GPU commit. They
 * take the SceneLoader's per-call inputs (rootGroup, gpuBufferPool,
 * updateVersion) as parameters instead of reading them off `this`.
 *
 * Behavior summary:
 *   - worker projection is used when `useWebWorkers && splatCount > 1000 && ndim > 3`,
 *   - truncation radius is read from the mesh material,
 *   - Cholesky factors are packed after projection,
 *   - worker args derive discreteDims / discreteSteps / extendToAllDims from `viewState.dimensions`,
 *   - the non-worker path and worker failures run the shared dispatcher
 *     in-process (`workers/data-worker/projection/in-process`),
 *   - first-update info logs are gated by `updateVersion <= 1`,
 *   - commits use the GPU buffer pool when enabled, otherwise
 *     `updateInstancedGSplatsMesh`.
 *
 * @module data/scene-loader/process/data-processor-gsplats
 */

import * as THREE from 'three';
import { packCholeskyForShader } from '../../../rendering/gsplat-geometry';
import type {
  LoadedGSplatsData,
  GSplatsViewState,
  ProcessedGSplatsData,
} from '../../../types/gsplats';
import { config as appConfig } from '../../../config';
import { log, Modules } from '../../../utils/log';
import { getWorkerPool } from '../../../workers/worker-pool';
import { projectGSplatsInProcess } from '../../../workers/data-worker/projection/in-process';
import { isExtendToAll } from '../../../workers/data-worker/projection/hidden-dims';
import { SHIFTED_GAUSSIAN_DEFAULT_TRUNCATE } from '../../../workers/data-worker/projection/constants';
import type { UpdateSession } from '../../../profiling/update-profiler';

/** Default truncation radius if the mesh material doesn't expose one. */
const DEFAULT_TRUNCATE = SHIFTED_GAUSSIAN_DEFAULT_TRUNCATE;

import type { StagedNoopCommit } from '../commit/noop-commit';

/** Staged data carried between async processing and the GPU commit. */
export interface StagedGSplatsGeometryCommit {
  path: string;
  noop?: undefined;
  /** Raw loader-returned data — stamped as `committedData` on commit. */
  sourceData: LoadedGSplatsData;
  processed: ProcessedGSplatsData;
  cholesky01: Float32Array;
  cholesky23: Float32Array;
  cholesky45: Float32Array;
}

/**
 * Either a real geometry commit or the stamp-only no-op fast path (data
 * reference-identical to what the GPU already holds — see noop-commit.ts).
 */
export type StagedGSplatsCommit = StagedGSplatsGeometryCommit | StagedNoopCommit<LoadedGSplatsData>;

/**
 * Build the projection params consumed by both the worker RPC and the
 * in-process dispatcher (they share the same dispatcher signature). The
 * worker needs a flat description of which non-display dimensions are
 * extend_to_all and which are discrete; derive it once here from the
 * scene's dimension metadata so the dispatcher stays dim-agnostic.
 */
function buildGSplatsParams(
  data: LoadedGSplatsData,
  viewState: GSplatsViewState,
  truncate: number
): Parameters<typeof projectGSplatsInProcess>[0] {
  const discreteDims: number[] = [];
  const discreteSteps: Record<number, number> = {};
  const extendToAllDims: number[] = [];
  if (viewState.dimensions) {
    for (let d = 0; d < viewState.dimensions.length; d++) {
      if (viewState.displayDims.includes(d)) continue;
      if (isExtendToAll(viewState.tolerance[d])) {
        extendToAllDims.push(d);
      } else if (viewState.dimensions[d]?.discrete) {
        discreteDims.push(d);
        discreteSteps[d] = viewState.dimensions[d].step ?? 1.0;
      }
    }
  }
  return {
    positions: data.positions,
    choleskyFactors: data.choleskyFactors,
    amplitudes: data.amplitudes,
    colors: data.colors,
    sharpness: null,
    viewState: {
      displayDims: viewState.displayDims,
      slicePosition: viewState.slicePosition,
      // GSplats doesn't consume tolerance in the dispatcher — hidden-dim
      // attenuation is computed from the cholesky factors. The field is
      // required for the uniform `ProjectionViewState` shape; pass the
      // loader's tolerance through for parity.
      tolerance: viewState.tolerance,
    },
    ndim: data.ndim,
    splatCount: data.splatCount,
    discreteDims,
    discreteSteps,
    extendToAllDims,
    truncate,
  };
}

/** Map a dispatcher result (worker or in-process) to `ProcessedGSplatsData`. */
function toProcessed(
  result: Awaited<ReturnType<typeof projectGSplatsInProcess>>
): ProcessedGSplatsData {
  return {
    centers3D: result.centers3D,
    choleskyFactors3D: result.choleskyFactors3D,
    amplitudes: result.amplitudes,
    colors: result.colors,
    splatCount: result.visibleCount,
  };
}

/**
 * Read the `uTruncate` uniform from the mesh material, falling back to
 * the default. Used both before projection (as a worker arg) and at
 * commit time (for frustum-culling sizing).
 */
function readTruncate(mesh: THREE.Mesh): number {
  return (
    (mesh.material as { uniforms?: { uTruncate?: { value: number } } })?.uniforms?.uTruncate
      ?.value ?? DEFAULT_TRUNCATE
  );
}

/**
 * Project GSplats to 3D on a worker thread. On worker-infrastructure
 * failure (anything but a dataset-switch abort) it degrades to the
 * in-process dispatcher — the *same* projection kernel run on the main
 * thread — rather than a separate hand-written copy.
 *
 * `updateVersion` gates the first-update info logs.
 */
export async function projectGSplatsTo3DUsingWorker(
  data: LoadedGSplatsData,
  viewState: GSplatsViewState,
  truncate: number,
  updateVersion: number
): Promise<ProcessedGSplatsData> {
  const params = buildGSplatsParams(data, viewState, truncate);
  try {
    if (updateVersion <= 1) {
      log.info(
        Modules.SCENE_LOADER,
        `Projecting ${data.splatCount} gsplats to 3D using worker (ndim=${data.ndim})`
      );
    }

    const workerResult = await getWorkerPool().runWithTimeout(
      'projectGSplatsTo3D',
      'projection',
      (api) => api.projectGSplatsTo3D(params)
    );

    if (updateVersion <= 1) {
      log.info(
        Modules.SCENE_LOADER,
        `Worker projection complete: ${workerResult.visibleCount}/${data.splatCount} visible splats`
      );
    }

    return toProcessed(workerResult);
  } catch (error) {
    // Dataset-switch abort: don't burn CPU on stale in-process work.
    if (error instanceof Error && error.name === 'WorkerAbortError') {
      throw error;
    }
    log.warning(
      Modules.SCENE_LOADER,
      'Worker GSplats projection failed, falling back to in-process dispatcher:',
      error
    );
    return toProcessed(await projectGSplatsInProcess(params));
  }
}

/**
 * Async process step for a single gsplats node: project nD → 3D
 * (worker or main thread), pack the Cholesky factors for the shader,
 * and return staged commit data — without mutating any mesh geometry.
 */
export async function processGSplatsData(
  path: string,
  data: LoadedGSplatsData,
  viewState: GSplatsViewState,
  rootGroup: THREE.Group | null,
  updateVersion: number,
  session?: UpdateSession
): Promise<StagedGSplatsCommit | null> {
  if (!rootGroup) return null;

  const mesh = rootGroup.getObjectByName(path) as THREE.Mesh;
  if (!mesh || mesh.userData?.nodeType !== 'gsplats') {
    log.warning(
      Modules.SCENE_LOADER,
      `GSplats update skipped for ${path}: ${
        !mesh ? 'mesh not found in scene' : `unexpected nodeType=${mesh.userData?.nodeType}`
      }. Data had ${data.splatCount} splats.`
    );
    return null;
  }

  // Worker only worth using for nD projections that are large enough
  // to amortize the postMessage cost; 3D-only data short-circuits.
  const useWorkerProjection =
    appConfig.dataLoading.performance.useWebWorkers && data.splatCount > 1000 && data.ndim > 3;

  const truncate = readTruncate(mesh);

  let processed: ProcessedGSplatsData;
  let cholesky01: Float32Array;
  let cholesky23: Float32Array;
  let cholesky45: Float32Array;

  const project = async () => {
    if (useWorkerProjection) {
      return projectGSplatsTo3DUsingWorker(data, viewState, truncate, updateVersion);
    }
    // Non-worker path (useWebWorkers off, small, or 3D-only data): run
    // the same dispatcher in-process. The standard-3D fast path inside
    // the dispatcher handles the ndim===3 case efficiently.
    return toProcessed(
      await projectGSplatsInProcess(buildGSplatsParams(data, viewState, truncate))
    );
  };

  if (session) {
    const projectSession = session.begin('Project to 3D');
    try {
      processed = await project();
      const packed = packCholeskyForShader(processed.choleskyFactors3D, processed.splatCount);
      cholesky01 = packed.cholesky01;
      cholesky23 = packed.cholesky23;
      cholesky45 = packed.cholesky45;
    } finally {
      projectSession.end();
    }
  } else {
    processed = await project();
    const packed = packCholeskyForShader(processed.choleskyFactors3D, processed.splatCount);
    cholesky01 = packed.cholesky01;
    cholesky23 = packed.cholesky23;
    cholesky45 = packed.cholesky45;
  }

  // All-loaded-but-none-visible is unusual enough to warrant a warning;
  // most often it indicates a slice/displayDims combination that doesn't
  // intersect any splat.
  if (data.splatCount > 0 && processed.splatCount === 0) {
    log.warning(
      Modules.SCENE_LOADER,
      `GSplats ${path}: all ${data.splatCount} loaded splats were filtered out during nD→3D processing. ` +
        `slicePosition=[${viewState.slicePosition.join(', ')}], displayDims=[${viewState.displayDims.join(', ')}], ndim=${data.ndim}`
    );
  }

  return { path, sourceData: data, processed, cholesky01, cholesky23, cholesky45 };
}

// Re-export the commit helper from its focused module so existing
// data-processor imports keep working unchanged.
export { commitGSplatsGeometry } from '../commit/commit-gsplats-geometry';
