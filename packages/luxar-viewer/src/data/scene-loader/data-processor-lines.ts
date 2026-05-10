/**
 * Lines data-processor concern extracted from `scene-loader.ts`.
 *
 * Three pure-ish functions for line projection and GPU commit. They
 * take the SceneLoader's per-call inputs as parameters (rootGroup,
 * gpuBufferPool, updateVersion) instead of reading them off `this`, so
 * the lines pipeline is testable in isolation and the SceneLoader
 * facade stays small.
 *
 * Behavior summary:
 *   - tolerance computation delegates to `tolerance-computer`,
 *   - `extend_to_all` flips covered dimensions to infinite tolerance,
 *   - worker projection is used when `useWebWorkers && segmentCount > 1000`,
 *   - worker failures fall back to `buildInstanceBuffers`,
 *   - first-update info logs are gated by `updateVersion <= 1`,
 *   - commits use the GPU buffer pool when enabled, otherwise
 *     `updateInstancedLinesMesh`.
 *
 * @module data/scene-loader/data-processor-lines
 */

import * as THREE from 'three';
import { buildInstanceBuffers } from '../lines/projection';
import type { LinesViewState, LoadedLinesData, ProcessedLinesData } from '../../types/lines';
import { isLinesUserData } from '../../types/lines';
import { computeTolerance } from '../utils/tolerance-computer';
import { EXTEND_TO_ALL_TOLERANCE } from './extend-tolerance';
import { config as appConfig } from '../../config';
import { log, Modules } from '../../utils/log';

/**
 * C.1: track whether the lines-scalar-worker-fallback warning has been
 * emitted this session so we surface the cliff once per process,
 * not per frame. Reset on hot-module-reload by reload, not in tests.
 */
let _linesScalarWorkerFallbackWarned = false;
import { getWorkerPool } from '../../workers/worker-pool';
import type { UpdateSession } from '../../profiling/update-profiler';
import type { GPUBufferPool } from '../../rendering/gpu-buffer-pool';
import { updateInstancedLinesMesh } from '../../rendering/line-geometry';

/** Staged data carried between async processing and the GPU commit. */
export interface StagedLinesCommit {
  path: string;
  processed: ProcessedLinesData;
}

/**
 * Project nD lines to a 3D instance-buffer set on a worker thread.
 * Falls back to the main-thread `buildInstanceBuffers` on worker
 * failure with a warning log — the user-visible behavior is identical
 * either way; only timing differs.
 *
 * `updateVersion` gates the first-update info logs to avoid noisy long
 * sessions.
 */
export async function projectLinesTo3DUsingWorker(
  data: LoadedLinesData,
  viewState: LinesViewState,
  tolerance: readonly number[],
  updateVersion: number
): Promise<ProcessedLinesData> {
  // When the dataset has per-vertex scalars, use the main-thread
  // buildInstanceBuffers path. The worker payload does not carry
  // scalar arrays, and the main-thread path preserves colormap
  // correctness end-to-end.
  //
  // C.1: TODO(viewer-code-review-rerun) — extend the worker schema to
  // carry per-vertex scalar buffers so colormap-enabled line datasets
  // don't pay the main-thread cost. For now, emit a one-shot warning
  // so users see the performance cliff exists. Reset semantics: the
  // module-scoped flag stays set for the lifetime of the JS context.
  if (data.scalars) {
    if (!_linesScalarWorkerFallbackWarned) {
      _linesScalarWorkerFallbackWarned = true;
      log.warning(
        Modules.SCENE_LOADER,
        'Lines with scalar colormap fall back to main-thread projection ' +
          '(worker schema does not yet carry per-vertex scalars). Large line ' +
          'datasets may stutter on view updates until the worker path is extended.'
      );
    }
    return buildInstanceBuffers(
      data,
      viewState.slicePosition,
      tolerance,
      viewState.displayDims
    );
  }

  try {
    if (updateVersion <= 1) {
      log.info(
        Modules.SCENE_LOADER,
        `Projecting ${data.segmentCount} line segments to 3D using worker`
      );
    }

    const workerResult = await getWorkerPool().runWithTimeout(
      'projectLinesTo3D',
      'projection',
      (api) =>
        api.projectLinesTo3D({
          positions: data.positions,
          segments: data.segments,
          widths: data.widths,
          colors: data.colors,
          sharpness: data.sharpness,
          slicePosition: viewState.slicePosition,
          tolerance,
          displayDims: viewState.displayDims,
          ndim: data.ndim,
          segmentCount: data.segmentCount,
        })
    );

    if (updateVersion <= 1) {
      log.info(
        Modules.SCENE_LOADER,
        `Worker projection complete: ${workerResult.visibleSegmentCount}/${data.segmentCount} visible segments`
      );
    }

    return {
      startPositions: workerResult.startPositions,
      endPositions: workerResult.endPositions,
      startColors: workerResult.startColors,
      endColors: workerResult.endColors,
      startWidths: workerResult.startWidths,
      endWidths: workerResult.endWidths,
      startSharpness: workerResult.startSharpness,
      endSharpness: workerResult.endSharpness,
      segmentLengths: workerResult.segmentLengths,
      startClipped: workerResult.startClipped,
      endClipped: workerResult.endClipped,
      segmentCount: workerResult.visibleSegmentCount,
    };
  } catch (error) {
    log.warning(
      Modules.SCENE_LOADER,
      'Worker lines projection failed, falling back to main thread:',
      error
    );
    return buildInstanceBuffers(
      data,
      viewState.slicePosition,
      tolerance,
      viewState.displayDims
    );
  }
}

/**
 * Async process step for a single lines node: compute tolerance, apply
 * `extend_to_all` if any, project to 3D (worker or main thread), and
 * return staged commit data — without mutating any mesh geometry. The
 * SceneLoader collects these stages, then runs all commits together so
 * the frame is atomic.
 */
export async function processLinesData(
  path: string,
  data: LoadedLinesData,
  viewState: LinesViewState,
  rootGroup: THREE.Group | null,
  updateVersion: number,
  session?: UpdateSession
): Promise<StagedLinesCommit | null> {
  if (!rootGroup) return null;

  const mesh = rootGroup.getObjectByName(path) as THREE.Mesh;
  if (!mesh || !isLinesUserData(mesh.userData)) return null;

  // Compute base tolerance, then mutate per-dim for extend_to_all
  // dimensions. Copy first to avoid mutating shared arrays.
  const ndim = data.ndim;
  let tolerance = computeTolerance(
    'lines',
    viewState.displayDims,
    ndim,
    viewState.dimensions
  );

  const attrs = mesh.userData.attrs as { extend_to_all?: string[] };
  const extendDims: string[] = attrs.extend_to_all || [];
  if (extendDims.length > 0 && viewState.dimensions) {
    tolerance = [...tolerance];
    for (const dimName of extendDims) {
      const dimIndex = viewState.dimensions.findIndex(
        (d: { name?: string }) => d.name === dimName
      );
      if (dimIndex >= 0 && dimIndex < tolerance.length) {
        tolerance[dimIndex] = EXTEND_TO_ALL_TOLERANCE;
      }
    }
  }

  // Use the worker only when the dataset is large enough to amortize
  // the postMessage cost.
  const useWorkerProjection =
    appConfig.dataLoading.performance.useWebWorkers && data.segmentCount > 1000;

  let processed: ProcessedLinesData;
  const project = async () => {
    if (useWorkerProjection) {
      return projectLinesTo3DUsingWorker(data, viewState, tolerance, updateVersion);
    }
    return buildInstanceBuffers(
      data,
      viewState.slicePosition,
      tolerance,
      viewState.displayDims
    );
  };

  if (session) {
    const buildSession = session.begin('Project to 3D');
    try {
      processed = await project();
    } finally {
      buildSession.end();
    }
  } else {
    processed = await project();
  }

  if (updateVersion <= 1) {
    log.info(
      Modules.SCENE_LOADER,
      `[GEOM] lines ${path}: ${processed.segmentCount}/${data.segmentCount} visible after projection`
    );
  }

  return { path, processed };
}

/**
 * Synchronous GPU commit step: write the staged buffers into the
 * mesh's geometry, either via the GPU buffer pool (if enabled) or via
 * `updateInstancedLinesMesh`. Updates `visibleSegmentCount` on the
 * mesh's user-data and logs an info line on a zero-segment frame
 * (slice with no visible content).
 *
 * Must run synchronously inside the atomic commit stage — no async
 * operations allowed.
 */
export function commitLinesGeometry(
  staged: StagedLinesCommit,
  rootGroup: THREE.Group | null,
  gpuBufferPool: GPUBufferPool | null
): void {
  if (!rootGroup) return;

  const mesh = rootGroup.getObjectByName(staged.path) as THREE.Mesh;
  if (!mesh || !isLinesUserData(mesh.userData)) return;

  const { processed } = staged;

  if (gpuBufferPool) {
    const geometry = gpuBufferPool.acquireLinesGeometry(staged.path, processed.segmentCount);
    gpuBufferPool.updateLinesGeometry(geometry, processed, processed.segmentCount);
    mesh.geometry = geometry;
  } else {
    updateInstancedLinesMesh(mesh, processed);
  }

  if (isLinesUserData(mesh.userData)) {
    mesh.userData.visibleSegmentCount = processed.segmentCount;
  }

  if (processed.segmentCount === 0) {
    log.info(
      Modules.SCENE_LOADER,
      `Clearing lines for ${staged.path} (no visible segments at current slice)`
    );
  }
}
