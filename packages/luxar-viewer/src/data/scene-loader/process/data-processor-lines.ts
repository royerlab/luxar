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
 *   - per-vertex scalars (colormap mode) ride the worker payload too —
 *     forwarded as a transferable typed-array and interpolated via
 *     `interpolate_scalars_batch` on the worker side,
 *   - the non-worker path and worker-UNAVAILABLE failures (the pool has no
 *     worker at all) run the shared dispatcher in-process
 *     (`workers/data-worker/projection/in-process`); any other failure —
 *     a rejection from the worker, a timeout — propagates instead, since the
 *     fallback shares the kernel and would only block the UI thread,
 *   - first-update info logs are gated by `updateVersion <= 1`,
 *   - commits use the GPU buffer pool when enabled, otherwise
 *     `updateInstancedLinesMesh`.
 *
 * @module data/scene-loader/process/data-processor-lines
 */

import * as THREE from 'three';
import type { LinesViewState, LoadedLinesData, ProcessedLinesData } from '../../../types/lines';
import { isLinesUserData } from '../../../types/lines';
import { assertColorLayout, computeTolerance } from '../../loaders';
import { EXTEND_TO_ALL_TOLERANCE } from '../view-state/extend-tolerance';
import { config as appConfig } from '../../../config';
import { log, Modules } from '../../../utils/log';
import { getWorkerPool } from '../../../workers/worker-pool';
// From the leaf module, not the `worker-pool` barrel: the barrel eagerly imports
// `./data-worker?worker`, and this predicate must stay free of that dependency.
import { isWorkerInfrastructureError } from '../../../workers/worker-pool/errors';
import { projectLinesInProcess } from '../../../workers/data-worker/projection/in-process';
import type { UpdateSession } from '../../../profiling/update-profiler';

import type { StagedNoopCommit } from '../commit/noop-commit';

/** Staged data carried between async processing and the GPU commit. */
export interface StagedLinesGeometryCommit {
  path: string;
  noop?: undefined;
  /** Raw loader-returned data — stamped as `committedData` on commit. */
  sourceData: LoadedLinesData;
  processed: ProcessedLinesData;
}

/**
 * Either a real geometry commit or the stamp-only no-op fast path (data
 * reference-identical to what the GPU already holds — see noop-commit.ts).
 */
export type StagedLinesCommit = StagedLinesGeometryCommit | StagedNoopCommit<LoadedLinesData>;

/**
 * Build the projection params consumed by both the worker RPC and the
 * in-process dispatcher (they share the same dispatcher signature).
 * Per-vertex scalars (colormap-mode Lines) ride along as a typed array;
 * `undefined → null` so Comlink doesn't strip the field.
 */
function buildLinesParams(
  data: LoadedLinesData,
  viewState: LinesViewState,
  tolerance: readonly number[]
): Parameters<typeof projectLinesInProcess>[0] {
  // Strict layout check at the chokepoint where the exact vertex count
  // is known: catches an RGBA array whose producer forgot to declare
  // colorComponents (see assertColorLayout).
  assertColorLayout(data.colors, data.vertexCount, data.colorComponents ?? 3, 'buildLinesParams');
  return {
    positions: data.positions,
    segments: data.segments,
    widths: data.widths,
    colors: data.colors,
    sharpness: data.sharpness,
    scalars: data.scalars ?? null,
    colorComponents: data.colorComponents,
    viewState: {
      displayDims: viewState.displayDims,
      slicePosition: viewState.slicePosition,
      tolerance,
    },
    ndim: data.ndim,
    segmentCount: data.segmentCount,
  };
}

/** True when the loaded colors carry an RGBA alpha column. */
function hasRGBAColors(data: LoadedLinesData): boolean {
  return !!data.colors && (data.colorComponents ?? 3) === 4;
}

/**
 * Map a dispatcher result (worker or in-process) to `ProcessedLinesData`.
 *
 * The worker returns an empty `startScalars` when input scalars were
 * null; the loader treats absent source scalars as the "no colormap"
 * signal. We pass `hasSourceScalars` (truthiness of `data.scalars`) so
 * undefined and null inputs are handled uniformly — and presence
 * follows the SOURCE alone, not the visible count: at a slice with 0
 * visible segments the worker's scalar arrays are empty but the node
 * still HAS scalars, and the empty-but-defined fields keep the
 * geometry's `hasScalars` stamp true (matching the points accumulator's
 * empty-subarray semantics). Gating on `length > 0` here used to flip
 * the stamp false on an empty slice, silently suppressing a colormap
 * picked while empty — with nothing re-applying it when segments
 * returned.
 */
function toProcessedLines(
  result: Awaited<ReturnType<typeof projectLinesInProcess>>,
  hasSourceScalars: boolean,
  hasSourceAlpha: boolean
): ProcessedLinesData {
  const hasScalars = hasSourceScalars;
  return {
    startPositions: result.startPositions,
    endPositions: result.endPositions,
    startColors: result.startColors,
    endColors: result.endColors,
    startWidths: result.startWidths,
    endWidths: result.endWidths,
    startSharpness: result.startSharpness,
    endSharpness: result.endSharpness,
    startScalars: hasScalars ? result.startScalars : undefined,
    endScalars: hasScalars ? result.endScalars : undefined,
    // Alpha presence follows the SOURCE color layout alone (RGBA), for
    // the same empty-slice reason as scalars above: 0 visible segments
    // must not flip the geometry's hasElementAlpha stamp.
    startAlphas: hasSourceAlpha ? result.startAlphas : undefined,
    endAlphas: hasSourceAlpha ? result.endAlphas : undefined,
    segmentLengths: result.segmentLengths,
    startCapSuppression: result.startCapSuppression,
    endCapSuppression: result.endCapSuppression,
    segmentCount: result.visibleSegmentCount,
    // Fused-scan cull metadata (AABB + max width) — lets computeLineBounds
    // skip its O(N) main-thread scan per commit (see types/lines.ts).
    bounds: result.bounds,
  };
}

/**
 * Project nD lines to a 3D instance-buffer set on a worker thread. Only when
 * the pool has no worker at all (`WorkerUnavailableError`) does it fall back
 * to the in-process dispatcher (the same kernel run on the main thread) with
 * a warning log.
 *
 * Every other failure propagates. A rejection that came back FROM the worker
 * would only reproduce the fault on the UI thread (the fallback shares the
 * kernel), and a timeout means work the main thread cannot afford either —
 * see the twin note in `data-processor-gsplats.ts`.
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
  // Per-vertex scalars (colormap-mode Lines) ride the worker path:
  // they are forwarded as a transferable typed-array and the worker
  // calls `interpolate_scalars_batch` on them just like widths /
  // sharpness. The empty-scalar (no colormap) case is preserved by
  // passing `null`.
  const params = buildLinesParams(data, viewState, tolerance);
  try {
    if (updateVersion <= 1) {
      log.info(
        Modules.SCENE_LOADER,
        `Projecting ${data.segmentCount} line segments to 3D using worker`
      );
    }

    // NOTE: `params` (positions / segments / widths / colors / scalars) is
    // passed WITHOUT a Comlink transfer list, so its buffers are structured-
    // cloned into the worker, not detached. The SliceCache relies on this: a
    // restored slice hands the loader's cached arrays straight into projection,
    // and transferring them would neuter (detach) the cached snapshot. Do not
    // add a transfer list for the inputs here. (Mirrors the gsplats processor.)
    const workerResult = await getWorkerPool().runWithTimeout(
      'projectLinesTo3D',
      'projection',
      (api) => api.projectLinesTo3D(params)
    );

    if (updateVersion <= 1) {
      log.info(
        Modules.SCENE_LOADER,
        `Worker projection complete: ${workerResult.visibleSegmentCount}/${data.segmentCount} visible segments`
      );
    }

    return toProcessedLines(workerResult, !!data.scalars, hasRGBAColors(data));
  } catch (error) {
    // Dataset-switch abort: don't burn CPU on stale in-process work.
    if (error instanceof Error && error.name === 'WorkerAbortError') {
      throw error;
    }
    // Only worker UNAVAILABILITY justifies the in-process retry. The fallback
    // runs the SAME kernel through the same `pickBackend`, so a rejection that
    // came back FROM the worker (a WASM trap, a validation throw) fails
    // identically here — except on the UI thread, where it blocks the frame —
    // and a timeout (hung kernel, or work slower than the budget) re-run
    // in-process blocks the frame at least as long again. Propagate instead:
    // the caller's `runLoaderUpdates` catch records the failure and the node
    // stays retryable. Fails closed on an unknown error.
    if (!isWorkerInfrastructureError(error)) {
      throw error;
    }
    log.warning(
      Modules.SCENE_LOADER,
      'No worker available, falling back to in-process lines projection:',
      error
    );
    return toProcessedLines(
      await projectLinesInProcess(params),
      !!data.scalars,
      hasRGBAColors(data)
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
  // MEMBERSHIP role: this tolerance is the per-dimension visibility slab
  // used by clip_segments_batch, not the chunk-fetch reach — discrete dims
  // get the half-cell gate matching points/gsplats, while the loader's
  // fetch path keeps the quarter-cell query role.
  const ndim = data.ndim;
  let tolerance = computeTolerance('lines', viewState.displayDims, ndim, viewState.dimensions, {
    discreteRole: 'membership',
  });

  const attrs = mesh.userData.attrs as { extend_to_all?: string[] };
  const extendDims: string[] = attrs.extend_to_all || [];
  if (extendDims.length > 0 && viewState.dimensions) {
    tolerance = [...tolerance];
    for (const dimName of extendDims) {
      const dimIndex = viewState.dimensions.findIndex((d: { name?: string }) => d.name === dimName);
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
    // Non-worker path (useWebWorkers off or small data): run the same
    // dispatcher in-process rather than a separate main-thread copy.
    return toProcessedLines(
      await projectLinesInProcess(buildLinesParams(data, viewState, tolerance)),
      !!data.scalars,
      hasRGBAColors(data)
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

  return { path, sourceData: data, processed };
}

// Re-export the commit helper from its focused module so existing
// data-processor imports keep working unchanged.
export { commitLinesGeometry } from '../commit/commit-lines-geometry';
