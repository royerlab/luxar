/**
 * GSplats data-processor concern extracted from `scene-loader.ts`.
 *
 * Three pure-ish functions for GSplat projection and GPU commit. They
 * take the SceneLoader's per-call inputs (rootGroup, gpuBufferPool,
 * updateVersion) as parameters instead of reading them off `this`.
 *
 * Behavior summary:
 *   - worker projection is used when `useWebWorkers` and the splat count exceeds
 *     `WORKER_MIN_SPLATS_ND` (1 000) for nD data or `WORKER_MIN_SPLATS_3D`
 *     (100 000) for 3-D data (whose fast path is cheap per splat, so only a
 *     large node amortizes the clone),
 *   - truncation radius is read from the mesh material,
 *   - the projection's 6-stride `choleskyFactors3D` flows straight to the
 *     commit (no split/re-interleave pass),
 *   - worker args derive discreteDims / discreteSteps / extendToAllDims from `viewState.dimensions`,
 *   - the non-worker path and worker-UNAVAILABLE failures (the pool has no
 *     worker at all) run the shared dispatcher in-process
 *     (`workers/data-worker/projection/in-process`); any other failure —
 *     a rejection from the worker, a timeout — propagates instead, since the
 *     fallback shares the kernel and would only block the UI thread,
 *   - a slice the loader restored from the S-cache reuses the projection
 *     cached on its entry when every other projection input is unchanged
 *     (post-projection stage cache, `projection-stage-cache.ts`); the output is
 *     then marked `sharedBuffers` and the commit must not transfer it,
 *   - first-update info logs are gated by `updateVersion <= 1`,
 *   - commits use the GPU buffer pool when enabled, otherwise
 *     `updateInstancedGSplatsMesh`.
 *
 * @module data/scene-loader/process/data-processor-gsplats
 */

import * as THREE from 'three';
import type {
  LoadedGSplatsData,
  GSplatsViewState,
  ProcessedGSplatsData,
  SplatRange,
} from '../../../types/gsplats';
import { assertColorLayout, buildElementIdMap } from '../../loaders';
import { config as appConfig } from '../../../config';
import { log, Modules } from '../../../utils/log';
import { getWorkerPool } from '../../../workers/worker-pool';
// From the leaf module, not the `worker-pool` barrel: the barrel eagerly imports
// `./data-worker?worker`, and this predicate must stay free of that dependency.
import { isWorkerInfrastructureError } from '../../../workers/worker-pool/errors';
import { projectGSplatsInProcess } from '../../../workers/data-worker/projection/in-process';
import { isExtendToAll } from '../../../workers/data-worker/projection/hidden-dims';
import { isStandardGSplats3D } from '../../../workers/data-worker/projection/gsplats';
import { projectGSplatLabelIndices } from '../../gsplats/label-channel';
import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../../config/constants';
import type { UpdateSession } from '../../../profiling/update-profiler';
import { perfCounters } from '../../../profiling/perf-counters';
import type { StagedNoopCommit } from '../commit/noop-commit';
import {
  canStoreStageOutput,
  getSliceCacheOrigin,
  lookupStageOutput,
  storeStageOutput,
} from '../../../cache/slice-cache-origin';
import {
  buffersIntact,
  hasWastefulBuffers,
  projectionStageSig,
  retainedBufferBytes,
  tightCopy,
} from './projection-stage-cache';

/** Counter: gsplats projections dispatched to a worker (`runWithTimeout`). */
const P_WORKER_CALLS = perfCounters.slot('projection.gsplats.worker');
/** Counter: gsplats projections answered by the post-projection stage cache. */
const P_STAGE_HITS = perfCounters.slot('projection.gsplats.stageHits');

/**
 * Worker-projection thresholds (splat count, exclusive). nD projections run a
 * per-splat attenuation/compaction kernel and amortize the worker round trip
 * early; 3-D projections take the standard fast path and only pay off off-thread
 * once the copy itself is long enough to block frames (see `processGSplatsData`).
 */
export const WORKER_MIN_SPLATS_ND = 1000;
export const WORKER_MIN_SPLATS_3D = 100_000;

/** Default truncation radius if the mesh material doesn't expose one. */
const DEFAULT_TRUNCATE = GSPLAT_DEFAULT_TRUNCATION_RADIUS;

/** Staged data carried between async processing and the GPU commit. */
export interface StagedGSplatsGeometryCommit {
  path: string;
  noop?: undefined;
  /** Raw loader-returned data — stamped as `committedData` on commit. */
  sourceData: LoadedGSplatsData;
  /**
   * Projection output, committed as-is: `choleskyFactors3D` (6-stride)
   * flows straight into the texel writer — no split/re-interleave pass.
   */
  processed: ProcessedGSplatsData;
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
  // Strict layout check at the chokepoint where the exact splat count
  // is known: catches an RGBA array whose producer forgot to declare
  // colorComponents (see assertColorLayout).
  assertColorLayout(data.colors, data.splatCount, data.colorComponents ?? 3, 'buildGSplatsParams');
  if (data.labelIndices && data.labelIndices.length !== data.splatCount) {
    throw new Error(
      `[buildGSplatsParams] labelIndices length ${data.labelIndices.length} does not match splat count ${data.splatCount}`
    );
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
    colorComponents: data.colorComponents ?? 3,
    discreteDims,
    discreteSteps,
    extendToAllDims,
    truncate,
    // Ask the kernel to record which source splat each emitted slot came from
    // (issue #1423) only when there is a map to build AND the projection will
    // actually compact. The standard-3D fast path emits every splat in order,
    // so it produces no source indices — the shared predicate keeps this site
    // and the dispatcher from disagreeing about which inputs take it.
    emitSourceIndices:
      (data.ranges !== undefined || data.labelIndices !== undefined) &&
      !isStandardGSplats3D(data.ndim, viewState.displayDims),
  };
}

/**
 * Map a dispatcher result (worker or in-process) to `ProcessedGSplatsData`.
 *
 * `ranges` are the loader's visible on-disk ranges (present only for a
 * label-carrying node); combined with the kernel's recorded `sourceIndices`
 * they compose the slot → on-disk element index map picking resolves labels
 * through. `sourceIndices` is absent on the standard-3D fast path, where no
 * compaction happened — passing `null` there puts the composer on its own
 * identity/range-offset path, which is exactly right.
 */
function toProcessed(
  result: Awaited<ReturnType<typeof projectGSplatsInProcess>>,
  data: LoadedGSplatsData,
  ranges: readonly SplatRange[] | undefined
): ProcessedGSplatsData {
  return {
    centers3D: result.centers3D,
    choleskyFactors3D: result.choleskyFactors3D,
    amplitudes: result.amplitudes,
    colors: result.colors,
    colorComponents: data.colorComponents ?? 3,
    labelIndices: data.labelIndices
      ? projectGSplatLabelIndices(data.labelIndices, result.sourceIndices, result.visibleCount)
      : undefined,
    labelVocabulary: data.labelVocabulary,
    splatCount: result.visibleCount,
    // Fused-scan cull metadata (AABB + max Cholesky row norm) — lets the
    // GPU commit skip its two O(N) main-thread scans (see types/gsplats.ts).
    bounds: result.bounds,
    elementIds: ranges
      ? buildElementIdMap(
          ranges,
          result.sourceIndices ?? null,
          result.visibleCount,
          Modules.SCENE_LOADER
        )
      : undefined,
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
 * Project GSplats to 3D on a worker thread. Only when the pool has no worker
 * at all (`WorkerUnavailableError`) does it degrade to the in-process
 * dispatcher — the *same* projection kernel run on the main thread — rather
 * than a separate hand-written copy.
 *
 * Every other failure propagates. A rejection that came back FROM the worker
 * would only reproduce the fault on the UI thread (the fallback shares the
 * kernel); a WASM trap there blocks the frame. A timeout means work the main
 * thread cannot afford either — a hung kernel or a genuinely-slow projection
 * re-run in-process blocks the frame at least as long again, and the pool has
 * already evicted the wedged worker, so a retry gets a fresh one.
 *
 * `updateVersion` gates the first-update info logs.
 *
 * `signal` is the per-update abort signal: a superseded pass rejects promptly
 * with a WorkerAbortError instead of holding the update lock through a whole
 * projection (50-150 ms at 1-3M elements). The worker still finishes the
 * abandoned task; the pool keeps that worker's slot busy until it does.
 */
export async function projectGSplatsTo3DUsingWorker(
  data: LoadedGSplatsData,
  viewState: GSplatsViewState,
  truncate: number,
  updateVersion: number,
  signal?: AbortSignal
): Promise<ProcessedGSplatsData> {
  const params = buildGSplatsParams(data, viewState, truncate);
  try {
    if (updateVersion <= 1) {
      log.info(
        Modules.SCENE_LOADER,
        `Projecting ${data.splatCount} gsplats to 3D using worker (ndim=${data.ndim})`
      );
    }

    // NOTE: `params` (positions / choleskyFactors / amplitudes / colors) is
    // passed WITHOUT a Comlink transfer list, so its buffers are structured-
    // cloned into the worker, not detached. The SliceCache relies on this: a
    // restored slice hands the loader's cached arrays straight into projection,
    // and transferring them would neuter (detach) the cached snapshot. Do not
    // add a transfer list for the inputs here.
    perfCounters.add(P_WORKER_CALLS);
    const workerResult = await getWorkerPool().runWithTimeout(
      'projectGSplatsTo3D',
      'projection',
      (api) => api.projectGSplatsTo3D(params),
      signal
    );

    if (updateVersion <= 1) {
      log.info(
        Modules.SCENE_LOADER,
        `Worker projection complete: ${workerResult.visibleCount}/${data.splatCount} visible splats`
      );
    }

    return toProcessed(workerResult, data, data.ranges);
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
      'No worker available, falling back to in-process GSplats projection:',
      error
    );
    return toProcessed(await projectGSplatsInProcess(params), data, data.ranges);
  }
}

/** Every typed array a gsplats stage output keeps alive. */
function gsplatsStageArrays(p: ProcessedGSplatsData): Array<ArrayBufferView | undefined> {
  return [p.centers3D, p.choleskyFactors3D, p.amplitudes, p.colors, p.labelIndices, p.elementIds];
}

/**
 * The cached projection for `data` under `sig`, if one is attached to its
 * S-cache entry and its buffers are intact (a detached buffer would commit an
 * empty frame — the commit contract forbids it, this is the backstop).
 */
function lookupGSplatsStage(
  data: LoadedGSplatsData,
  sig: string
): ProcessedGSplatsData | undefined {
  const cached = lookupStageOutput(data, sig) as ProcessedGSplatsData | undefined;
  if (!cached) return undefined;
  const n = cached.splatCount;
  const intact = buffersIntact([
    [cached.centers3D, n * 3],
    [cached.choleskyFactors3D, n * 6],
    [cached.amplitudes, n],
    [cached.colors, n * (cached.colorComponents ?? 3)],
  ]);
  if (intact) perfCounters.add(P_STAGE_HITS);
  return intact ? cached : undefined;
}

/**
 * Offer a fresh projection to `data`'s S-cache entry. Returns what the commit
 * should use: the retained (`sharedBuffers`) output when it was admitted —
 * tight copies when the projection culled enough that its worst-case buffers
 * would pin dead tail — else `processed` untouched (the commit then owns it).
 */
function retainGSplatsStage(
  data: LoadedGSplatsData,
  sig: string,
  processed: ProcessedGSplatsData,
  scan: boolean
): ProcessedGSplatsData {
  const arrays = gsplatsStageArrays(processed);
  const tighten = hasWastefulBuffers(arrays);
  const bytes = tighten
    ? arrays.reduce((s, a) => s + (a?.byteLength ?? 0), 0)
    : retainedBufferBytes(arrays);
  if (!canStoreStageOutput(data, bytes, { scan })) return processed;
  const shared: ProcessedGSplatsData = tighten
    ? {
        ...processed,
        centers3D: tightCopy(processed.centers3D),
        choleskyFactors3D: tightCopy(processed.choleskyFactors3D),
        amplitudes: tightCopy(processed.amplitudes),
        colors: tightCopy(processed.colors),
        labelIndices: tightCopy(processed.labelIndices),
        elementIds: tightCopy(processed.elementIds),
        sharedBuffers: true,
      }
    : { ...processed, sharedBuffers: true };
  return storeStageOutput(data, { sig, value: shared, bytes }, { scan }) ? shared : processed;
}

/**
 * Async process step for a single gsplats node: project nD → 3D
 * (worker or main thread) and return staged commit data — without
 * mutating any mesh geometry. `signal` (the per-update abort signal) reaches
 * the worker projection so a superseded pass rejects promptly.
 */
export async function processGSplatsData(
  path: string,
  data: LoadedGSplatsData,
  viewState: GSplatsViewState,
  rootGroup: THREE.Group | null,
  updateVersion: number,
  session?: UpdateSession,
  signal?: AbortSignal
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

  // Worker only worth using when the projection is large enough to amortize
  // the structured-clone round trip. nD data pays a real per-splat kernel
  // (hidden-dim attenuation + compaction), so 1 000 splats already amortize
  // it. 3D data takes the standard fast path (a copy plus the fused bounds
  // scan) and used to stay on the main thread at ANY size — which on a
  // 14.8 M-splat 2-D pathology slide meant several seconds of main-thread long
  // tasks per load (2026-09 audit, finding 4). Above WORKER_MIN_SPLATS_3D the
  // clone is cheaper than blocking the frame loop for that long.
  const useWorkerProjection =
    appConfig.dataLoading.performance.useWebWorkers &&
    data.splatCount > (data.ndim > 3 ? WORKER_MIN_SPLATS_ND : WORKER_MIN_SPLATS_3D);

  const truncate = readTruncate(mesh);

  // Post-projection stage cache (#2944 B2): a slice restored from the S-cache
  // with unchanged projection params reuses the projection stored on its
  // entry. The signature is built from the very params the kernel receives
  // (see projection-stage-cache.ts), so it cannot miss a projection input.
  const stageSig = getSliceCacheOrigin(data)
    ? projectionStageSig('gsplats', buildGSplatsParams(data, viewState, truncate))
    : null;
  const cached = stageSig !== null ? lookupGSplatsStage(data, stageSig) : undefined;

  let processed: ProcessedGSplatsData;

  const project = async () => {
    if (cached) return cached;
    if (useWorkerProjection) {
      return projectGSplatsTo3DUsingWorker(data, viewState, truncate, updateVersion, signal);
    }
    // Non-worker path (useWebWorkers off, small, or 3D-only data): run
    // the same dispatcher in-process. The standard-3D fast path inside
    // the dispatcher handles the ndim===3 case efficiently.
    return toProcessed(
      await projectGSplatsInProcess(buildGSplatsParams(data, viewState, truncate)),
      data,
      data.ranges
    );
  };

  if (session) {
    const projectSession = session.begin('Project to 3D');
    try {
      processed = await project();
    } finally {
      projectSession.end();
    }
  } else {
    processed = await project();
  }
  if (!cached && stageSig !== null)
    processed = retainGSplatsStage(data, stageSig, processed, viewState.frameBudgetMs != null);

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

  return { path, sourceData: data, processed };
}

// Re-export the commit helper from its focused module so existing
// data-processor imports keep working unchanged.
export { commitGSplatsGeometry } from '../commit/commit-gsplats-geometry';
