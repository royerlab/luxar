/**
 * GSplats data-processor concern extracted from `scene-loader.ts`.
 *
 * Three pure-ish functions matching the inline `processGSplatsData`,
 * `commitGSplatsGeometry`, and `projectGSplatsTo3DUsingWorker` methods
 * on the SceneLoader class. They take the SceneLoader's per-call inputs
 * (rootGroup, gpuBufferPool, updateVersion) as parameters instead of
 * reading them off `this`.
 *
 * Behavior is identical to the inline original:
 *   - same worker / main-thread split (worker iff
 *     `useWebWorkers && splatCount > 1000 && ndim > 3`),
 *   - same truncate-uniform read from the mesh material,
 *   - same Cholesky packing call after projection,
 *   - same worker arg derivation (discreteDims / discreteSteps /
 *     extendToAllDims) computed from `viewState.dimensions`,
 *   - same fallback to `processGSplats` on worker failure,
 *   - same first-update logging gate (`updateVersion <= 1`),
 *   - same GPU-buffer-pool vs `updateInstancedGSplatsMesh` commit branch.
 *
 * @module data/scene-loader/data-processor-gsplats
 */

import * as THREE from 'three';
import { processGSplats } from '../gsplats/gsplats-processor';
import {
  updateInstancedGSplatsMesh,
  packCholeskyForShader,
} from '../../rendering/gsplat-geometry';
import type {
  LoadedGSplatsData,
  GSplatsUserData,
  GSplatsViewState,
} from '../../types/gsplats';
import { config as appConfig } from '../../config';
import { log, Modules } from '../../utils/log';
import { getWorkerPool } from '../../workers/worker-pool';
import type { UpdateSession } from '../../profiling/update-profiler';
import type { GPUBufferPool } from '../../rendering/gpu-buffer-pool';

/** Default truncation radius if the mesh material doesn't expose one. */
const DEFAULT_TRUNCATE = 3.0;

/** Staged data carried between the async process phase and the GPU commit. */
export interface StagedGSplatsCommit {
  path: string;
  processed: ReturnType<typeof processGSplats>;
  cholesky01: Float32Array;
  cholesky23: Float32Array;
  cholesky45: Float32Array;
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
 * Project GSplats to 3D on a worker thread. Falls back to the main
 * thread `processGSplats` on worker failure with a warning log.
 *
 * `updateVersion` gates the first-update info logs.
 */
export async function projectGSplatsTo3DUsingWorker(
  data: LoadedGSplatsData,
  viewState: GSplatsViewState,
  truncate: number,
  updateVersion: number
): Promise<ReturnType<typeof processGSplats>> {
  try {
    const worker = await getWorkerPool().getWorker();

    if (updateVersion <= 1) {
      log.info(
        Modules.SCENE_LOADER,
        `Projecting ${data.splatCount} gsplats to 3D using worker (ndim=${data.ndim})`
      );
    }

    // Worker needs a flat description of which non-display dimensions
    // are extend_to_all and which are discrete. Computed once here so
    // the worker side can stay dim-agnostic.
    const discreteDims: number[] = [];
    const discreteSteps: Record<number, number> = {};
    const extendToAllDims: number[] = [];
    if (viewState.dimensions) {
      for (let d = 0; d < viewState.dimensions.length; d++) {
        if (viewState.displayDims.includes(d)) continue;
        if (viewState.tolerance[d] >= 1e9) {
          extendToAllDims.push(d);
        } else if (viewState.dimensions[d]?.discrete) {
          discreteDims.push(d);
          discreteSteps[d] = viewState.dimensions[d].step ?? 1.0;
        }
      }
    }

    const workerResult = await worker.projectGSplatsTo3D({
      positions: data.positions,
      choleskyFactors: data.choleskyFactors,
      amplitudes: data.amplitudes,
      colors: data.colors,
      sharpness: null,
      displayDims: viewState.displayDims,
      slicePosition: viewState.slicePosition,
      ndim: data.ndim,
      splatCount: data.splatCount,
      discreteDims,
      discreteSteps,
      extendToAllDims,
      truncate,
    });

    if (updateVersion <= 1) {
      log.info(
        Modules.SCENE_LOADER,
        `Worker projection complete: ${workerResult.visibleCount}/${data.splatCount} visible splats`
      );
    }

    return {
      centers3D: workerResult.centers3D,
      choleskyFactors3D: workerResult.choleskyFactors3D,
      amplitudes: workerResult.amplitudes,
      colors: workerResult.colors,
      splatCount: workerResult.visibleCount,
    };
  } catch (error) {
    log.warning(
      Modules.SCENE_LOADER,
      'Worker GSplats projection failed, falling back to main thread:',
      error
    );
    return processGSplats(data, viewState, truncate);
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
    appConfig.dataLoading.performance.useWebWorkers &&
    data.splatCount > 1000 &&
    data.ndim > 3;

  const truncate = readTruncate(mesh);

  let processed: ReturnType<typeof processGSplats>;
  let cholesky01: Float32Array;
  let cholesky23: Float32Array;
  let cholesky45: Float32Array;

  const project = async () => {
    if (useWorkerProjection) {
      return projectGSplatsTo3DUsingWorker(data, viewState, truncate, updateVersion);
    }
    return processGSplats(data, viewState, truncate);
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

  return { path, processed, cholesky01, cholesky23, cholesky45 };
}

/**
 * Synchronous GPU commit step: write the staged buffers into the
 * mesh's geometry, either via the GPU buffer pool (if enabled) or via
 * `updateInstancedGSplatsMesh`. Updates `visibleSplatCount` on the
 * mesh's user-data and logs an info line on a zero-splat frame.
 *
 * Must run synchronously inside the atomic commit phase.
 */
export function commitGSplatsGeometry(
  staged: StagedGSplatsCommit,
  rootGroup: THREE.Group | null,
  gpuBufferPool: GPUBufferPool | null
): void {
  if (!rootGroup) return;

  const mesh = rootGroup.getObjectByName(staged.path) as THREE.Mesh;
  if (!mesh || mesh.userData?.nodeType !== 'gsplats') return;

  const { processed, cholesky01, cholesky23, cholesky45 } = staged;

  if (gpuBufferPool) {
    const geometry = gpuBufferPool.acquireGSplatsGeometry(staged.path, processed.splatCount);
    const truncationRadius = readTruncate(mesh);
    gpuBufferPool.updateGSplatsGeometry(
      geometry,
      {
        centers3D: processed.centers3D,
        amplitudes: processed.amplitudes,
        cholesky01,
        cholesky23,
        cholesky45,
        colors: processed.colors,
        splatCount: processed.splatCount,
      },
      processed.splatCount,
      truncationRadius
    );
    mesh.geometry = geometry;
  } else {
    updateInstancedGSplatsMesh(mesh, {
      centers: processed.centers3D,
      cholesky01,
      cholesky23,
      cholesky45,
      amplitudes: processed.amplitudes,
      colors: processed.colors,
      splatCount: processed.splatCount,
    });
  }

  if (mesh.userData) {
    (mesh.userData as GSplatsUserData).visibleSplatCount = processed.splatCount;
  }

  if (processed.splatCount === 0) {
    log.info(
      Modules.SCENE_LOADER,
      `Clearing gsplats for ${staged.path} (no visible splats at current slice)`
    );
  }
}
