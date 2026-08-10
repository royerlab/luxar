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
import { assertColorLayout, buildElementIdMap, computeTolerance } from '../../loaders';
import type { ElementIdRange } from '../../loaders/element-ids';
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
    // Ask the projection to record which loaded segment row each emitted slot
    // came from (issue #1424) only when there is a map to build — i.e. when the
    // loader published the on-disk vertex ranges for a label-carrying node.
    // Mirrors the gsplats processor's `emitSourceIndices` gate.
    emitSourceIndices: data.vertexRangeBounds !== undefined,
  };
}

/** True when the loaded colors carry an RGBA alpha column. */
function hasRGBAColors(data: LoadedLinesData): boolean {
  return !!data.colors && (data.colorComponents ?? 3) === 4;
}

/**
 * Compose the lines picking map: visible segment SLOT → on-disk sorted VERTEX
 * row (issue #1424). Four index spaces are involved, and the chain walks all of
 * them:
 *
 *  - **A** on-disk (sorted) VERTEX row — what the per-vertex label CSR is keyed
 *    by. `io/_ordering/lines.py` rewrites the stored segment entries through
 *    `argsort(vertex_sort_indices)` (original index → sorted row), while
 *    `io/_compiler/labels/text_labels.py` gathers the label strings by the
 *    FORWARD `vertex_sort_indices` (sorted row → original index). Those are
 *    duals, which is exactly why the two land in the same space: a stored
 *    segment entry indexes the label CSR directly.
 *  - **C** loaded-local VERTEX index — what `remapSegmentIndices` wrote into
 *    `LoadedLinesData.segments`, i.e. a row of `positions` / `widths`.
 *  - **D** loaded SEGMENT row — row `r` of `segments`, holding two **C** values
 *    at `[2r]`, `[2r + 1]`.
 *  - **E** visible SEGMENT slot — what the pick shader reports (the
 *    `ProcessedLinesData` array index == the line-texture texel row).
 *
 * `sourceSegmentIndices` (from the projection) is E → D; `segments` is D → C;
 * `buildElementIdMap(vertexRangeBounds, null, …)` is C → A. Composing them is
 * the whole fix: indexing the label CSR with the raw slot **E** is wrong twice
 * over — wrong granularity (segment vs vertex) and wrong index space (visible
 * vs on-disk).
 *
 * **The START vertex is the one reported.** A segment has two endpoints, but the
 * pick id is a `flat` vertex-stage varying (`rendering/picking/line/shaders.ts`,
 * `pick.tsl.ts`), so the fragment stage cannot choose the nearer endpoint
 * without the shader emitting two ids and a barycentric split. Reporting the
 * start endpoint consistently is the honest single-id answer. Two consequences
 * are visible to a user and worth stating: on a PARTIALLY CLIPPED segment the
 * reported start vertex can lie entirely outside the visible slice (what is
 * drawn starts at `p1 + t1·(p2 - p1)`, so with `t1 → 1` the visible geometry
 * sits at the far end), and ANY vertex that is never a segment's start is not
 * reportable at all. Which ones those are is set by the authored line type
 * (`io/_ordering/lines.py::convert_to_indexed`): for `polyline` only the final
 * vertex, for `loop` none, for `indexed` whatever subset the supplied indices
 * never place first — and, the surprising case, for `segments` EVERY
 * odd-numbered vertex, since its pairs are consecutive and disjoint
 * (`(0,1), (2,3), …`), so half the label array is unreachable.
 *
 * Fails CLOSED (returns `undefined`, one warning) on any inconsistency: picking
 * then falls back to the raw slot, exactly as it behaved before this map
 * existed. Writing a garbage index would instead surface a wrong-but-plausible
 * label.
 */
function composeLinesElementIds(
  data: LoadedLinesData,
  sourceSegmentIndices: Uint32Array | undefined,
  visibleSegmentCount: number
): Uint32Array | undefined {
  const bounds = data.vertexRangeBounds;
  if (bounds === undefined) return undefined;
  // Shape check on the flat `[start0, end0, …]` pair layout. An odd length is
  // malformed by construction, and the arithmetic below would silently read
  // `undefined` for the missing `end` — so reject it explicitly rather than
  // leaning on the NaN the subtraction would produce.
  if (bounds.length % 2 !== 0) {
    log.warning(
      Modules.LINES_LOADER,
      'Lines element-ID map skipped: the vertex range bounds have an odd length ' +
        `(${bounds.length}); they must be [start, end) pairs. Picking labels fall ` +
        'back to the visible-buffer slot.'
    );
    return undefined;
  }
  if (visibleSegmentCount <= 0) return undefined; // no slots to map
  if (sourceSegmentIndices === undefined || sourceSegmentIndices.length !== visibleSegmentCount) {
    // EXACT length, not "at least": the projection publishes a table that
    // describes precisely the visible stream (`workers/data-worker/projection/
    // lines.ts` sizes it `visibleCount` and only publishes it when its own set-bit
    // count agrees), so a longer table is not a superset to read a prefix of — it
    // is an UNCOMPACTED `segmentCount`-long table from a producer that ignored the
    // clip, and a `>=` check would accept it and map every slot `s` to segment row
    // `s`: exactly the wrong-but-plausible answer this map exists to prevent.
    // Every other guard here is an exact-match rejection for the same reason.
    //
    // Deliberately does not assert what the projection did: the table is also
    // dropped when the clip kernel's visibility mask and returned count
    // disagree, and that case logs its own warning at the detection site in
    // `workers/data-worker/projection/lines.ts`.
    log.warning(
      Modules.LINES_LOADER,
      'Lines element-ID map skipped: no usable source segment indices for ' +
        `${visibleSegmentCount} visible segments (got ${
          sourceSegmentIndices?.length ?? 'none'
        }). Picking labels fall back to the visible-buffer slot.`
    );
    return undefined;
  }

  // The ranges must describe EXACTLY the loaded vertices before the C → A link
  // means anything, so assert the total up front. This is also what keeps the
  // two identity conditions from disagreeing: `buildElementIdMap` takes its
  // identity fast path BEFORE its own count guard, so a single `[0, k)` range
  // with `k !== vertexCount` returned `undefined` with nothing logged, and the
  // "already warned" bail below was a lie.
  let rangeTotal = 0;
  for (let i = 0; i < bounds.length; i += 2) rangeTotal += bounds[i + 1] - bounds[i];
  if (rangeTotal !== data.vertexCount) {
    log.warning(
      Modules.LINES_LOADER,
      `Lines element-ID map skipped: the vertex ranges cover ${rangeTotal} vertices ` +
        `but ${data.vertexCount} were loaded. Picking labels fall back to the ` +
        'visible-buffer slot.'
    );
    return undefined;
  }

  // C → A. With the total asserted above, "one range anchored at 0" is now
  // EXACTLY `buildElementIdMap`'s own identity condition (a total of
  // `vertexCount` from a single range starting at 0 forces `end ===
  // vertexCount`), so the two can no longer diverge: on that shape the local
  // index IS the on-disk row, and any other `undefined` from the composer is a
  // genuine rejection it warned about.
  const identityCA = bounds.length === 2 && bounds[0] === 0;
  let localToGlobal: Uint32Array | undefined;
  if (!identityCA) {
    // The object array materializes HERE and nowhere else: `buildElementIdMap`
    // takes `readonly ElementIdRange[]`, and it stays the shared composer
    // precisely for its guards. Nothing cached ever sees these objects — the
    // payload keeps the flat typed array.
    const ranges: ElementIdRange[] = new Array<ElementIdRange>(bounds.length / 2);
    for (let i = 0; i < bounds.length; i += 2) {
      ranges[i / 2] = { start: bounds[i], end: bounds[i + 1] };
    }
    localToGlobal = buildElementIdMap(ranges, null, data.vertexCount, Modules.LINES_LOADER);
    if (localToGlobal === undefined) return undefined; // already warned
  }

  const out = new Uint32Array(visibleSegmentCount);
  for (let s = 0; s < visibleSegmentCount; s++) {
    const r = sourceSegmentIndices[s]; // E → D
    // Both limits matter. `r >= segmentCount` catches a row past the loaded
    // segments; the `segments.length` half catches a SHORT `segments` array,
    // where `data.segments[2 * r]` would be `undefined` — and `undefined >=
    // vertexCount` is FALSE, so the vertex guard below would wave it through and
    // `out[s]` would store a wrong-but-plausible 0. A NaN/undefined comparison
    // cannot be relied on to fail closed, which is why the explicit length check
    // is the correct form. (Unreachable today only because
    // `validateLineSegmentReferences` throws upstream first.)
    if (r >= data.segmentCount || 2 * r + 1 >= data.segments.length) {
      log.warning(
        Modules.LINES_LOADER,
        `Lines element-ID map skipped: source segment row ${r} is outside the ` +
          `${data.segmentCount} loaded segments (${data.segments.length} segment ` +
          'entries). Picking labels fall back to the visible-buffer slot.'
      );
      return undefined;
    }
    // D → C (start vertex). Note this reads `data.segments`' CONTENTS after the
    // awaited projection: on a cold plain-leaf load that array aliases
    // `LinesDataAccumulator`'s reused buffer, which `loadLinesInternal` refills IN
    // PLACE. Safe only because no second load can be in flight against it —
    // `SceneLoader.updateView` is single-flight, `SlicePrefetcher` uses shadow
    // loaders with their own accumulators, and a slice-cache hit returns a clone
    // (the same aliasing `setCommittedData` already retains). Loosen the
    // single-flight assumption and this needs a snapshot first.
    const c = data.segments[2 * r];
    // Bounds-checked for BOTH C → A branches. On the identity branch `c` is
    // written straight through as an on-disk row, so an out-of-range segment
    // entry would surface exactly the wrong-but-plausible label this map exists
    // to remove. `vertexCount` is the right limit either way — `localToGlobal`,
    // when it is built at all, is precisely that long.
    if (c >= data.vertexCount) {
      log.warning(
        Modules.LINES_LOADER,
        `Lines element-ID map skipped: local vertex index ${c} is outside the ` +
          `${data.vertexCount} loaded vertices. Picking labels fall back to the ` +
          'visible-buffer slot.'
      );
      return undefined;
    }
    out[s] = localToGlobal === undefined ? c : localToGlobal[c]; // C → A
  }
  return out;
}

/**
 * Map a dispatcher result (worker or in-process) to `ProcessedLinesData`.
 *
 * Takes the LOADED payload (not just derived booleans) because the picking map
 * composition below reads `segments` / `vertexRangeBounds` / `vertexCount` off it.
 *
 * The worker returns an empty `startScalars` when input scalars were
 * null; the loader treats absent source scalars as the "no colormap"
 * signal. Scalar presence is read off `data.scalars` (truthiness) so
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
  data: LoadedLinesData
): ProcessedLinesData {
  const hasScalars = !!data.scalars;
  const hasSourceAlpha = hasRGBAColors(data);
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
    startJointCode: result.startJointCode,
    endJointCode: result.endJointCode,
    segmentCount: result.visibleSegmentCount,
    // Fused-scan cull metadata (AABB + max width) — lets computeLineBounds
    // skip its O(N) main-thread scan per commit (see types/lines.ts).
    bounds: result.bounds,
    // Visible slot → on-disk START-vertex row, for per-vertex label lookups
    // on hover (see composeLinesElementIds).
    elementIds: composeLinesElementIds(
      data,
      result.sourceSegmentIndices,
      result.visibleSegmentCount
    ),
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

    return toProcessedLines(workerResult, data);
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
    return toProcessedLines(await projectLinesInProcess(params), data);
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
      data
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
