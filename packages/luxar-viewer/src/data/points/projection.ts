/**
 * nD → 3D projection helpers for the points spatial-index loader.
 *
 * Three behaviors live here, extracted from
 * `data/points-spatial-index-loader.ts`:
 *
 *   - `projectPointsTo3D` — main-thread projection, with optional
 *     accumulator-buffer write-through (zero-allocation when the loader
 *     enabled the accumulator path) and effective-radius filtering.
 *   - `projectPointsTo3DUsingWorker` — same projection on a worker thread
 *     via the worker pool, with a main-thread fallback on worker failure.
 *   - `createEmptyPointsData` — factory for the "no visible points" return
 *     path, shared between both projectors.
 *
 * All three are pure with respect to the loader: the loader-side state
 * (`chunkIndex`, `effectiveRadiusConfig`, `accumulator`, node attrs) is
 * passed in via a small `ProjectionContext` so the helpers can be
 * unit-tested without instantiating a full `PointsSpatialIndexLoader`.
 *
 * Behavior is identical to the inlined methods — bit-for-bit on the
 * happy path and on every "early return" branch (missing positions,
 * accumulator filtering, all-zero-radius fallback, worker failure).
 *
 * @module data/point-loader/projection
 */

import * as THREE from 'three';
import { log, Modules } from '../../utils/log';
import {
  calculateEffectiveRadii,
  shouldApplyEffectiveRadius,
  type EffectiveRadiusConfig,
} from './effective-radius-calculator';
import type {
  ViewState,
  LoadedPointsData,
  PointRange,
  PositionArray,
  ColorArray,
  ScalarArray,
} from '../data-loader-types';
import { LoadedPointsDataAccumulator } from '../utils/data-accumulator';
import type { PointsMetadata } from '../../types/points';
import type { PointsChunkIndex } from './chunk-index-loader';
import { getWorkerPool } from '../../workers/worker-pool';

/** Buffers an accumulator owns; writing directly into them avoids allocations. */
export interface ProjectionTargetBuffers {
  positions3D: Float32Array;
  colors: ColorArray;
  radii: ScalarArray;
  sharpness: ScalarArray;
}

/**
 * Loader-side context the projectors need. Built once per call from the
 * loader's instance fields (`chunkIndex`, `_effectiveRadiusConfig`,
 * `_accumulator`, `node.attrs`).
 */
export interface ProjectionContext {
  chunkIndex: PointsChunkIndex | null;
  effectiveRadiusConfig: EffectiveRadiusConfig | null;
  accumulator: LoadedPointsDataAccumulator | null;
  nodeAttrs: PointsMetadata;
}

/** Pull dtype metadata off the node attrs in the shape both projectors return. */
function dtypesFromAttrs(nodeAttrs: PointsMetadata): {
  positions: string | undefined;
  colors: string | undefined;
  radii: string | undefined;
  sharpness: string | undefined;
} {
  return {
    positions: nodeAttrs.position_dtype as string | undefined,
    colors: nodeAttrs.color_dtype as string | undefined,
    radii: nodeAttrs.radius_dtype as string | undefined,
    sharpness: nodeAttrs.sharpness_dtype as string | undefined,
  };
}

/**
 * Build the "no visible points" `LoadedPointsData` payload. The empty
 * payload still carries the dataset's `ndim` (from the chunk index when
 * present, else 3) and `totalPoints` (from `node.attrs.n_points`) so
 * downstream metrics don't read undefined.
 */
export function createEmptyPointsData(
  ctx: ProjectionContext,
  viewState: ViewState
): LoadedPointsData {
  // viewState parameter retained for future ndim-aware empty payloads.
  void viewState;

  return {
    positions: new Float32Array(0) as PositionArray,
    pointCount: 0,
    ndim: ctx.chunkIndex?.metadata.ndim || 3,
    metadata: {
      totalPoints: ctx.nodeAttrs.n_points || 0,
      loadedPoints: 0,
      bounds: new THREE.Box3(),
      usedSpatialIndex: true,
      dtypes: dtypesFromAttrs(ctx.nodeAttrs),
    },
  };
}

/**
 * Project nD points to 3D display space on the main thread.
 *
 * Two execution paths share this entry point:
 *
 *  - **Accumulator path** (`targetBuffers` provided): the supported
 *    hot path. Writes through preallocated accumulator buffers for
 *    zero-allocation operation. Production code always takes this
 *    path via `LoadedPointsDataAccumulator`.
 *  - **Fallback path** (`targetBuffers` null/undefined): allocates
 *    fresh arrays. Used by tests, the explicit no-accumulator opt-out,
 *    and the worker-error rescue route in
 *    `projectPointsTo3DUsingWorker`. Color/sharpness inputs pass
 *    through by reference; positions3D and (when filtering applies)
 *    radii are freshly allocated.
 *
 * @param targetBuffers - Optional accumulator buffers for zero-allocation
 *                        operation. When provided, writes directly
 *                        through; when null/undefined, allocates new
 *                        arrays for the result.
 */
export function projectPointsTo3D(
  positions: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  colors: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  radii: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  sharpness: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  viewState: ViewState,
  ranges: PointRange[],
  ctx: ProjectionContext,
  targetBuffers?: ProjectionTargetBuffers | null
): LoadedPointsData {
  const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

  if (!positions) {
    throw new Error('[PointsProjection] Positions data is required for points');
  }

  // CRITICAL: Calculate ndim from actual positions array, not chunk index metadata.
  const ndim =
    totalPoints > 0
      ? Math.round(positions.length / totalPoints)
      : ctx.chunkIndex?.metadata.ndim || 3;

  // Validate the calculation
  if (totalPoints > 0 && positions.length !== totalPoints * ndim) {
    log.error(
      Modules.SPATIAL_INDEX_LOADER,
      `Position data size mismatch: ${positions.length} elements for ${totalPoints} points ` +
        `doesn't divide evenly (calculated ndim=${ndim}). This may indicate encoding metadata issues.`
    );
  }

  let numPoints = totalPoints;

  // Use target buffers if provided (zero allocations).
  const { displayDims } = viewState;

  // Use target buffer or allocate new (zero-allocation when targetBuffers provided)
  let positions3D = targetBuffers ? targetBuffers.positions3D : new Float32Array(numPoints * 3);

  // Extract 3D positions from nD data (write directly to buffer)
  for (let i = 0; i < numPoints; i++) {
    // Extract displayed dimensions
    for (let j = 0; j < Math.min(3, displayDims.length); j++) {
      const dimIdx = displayDims[j];
      positions3D[i * 3 + j] = positions[i * ndim + dimIdx];
    }
    // Fill remaining with zeros
    for (let j = displayDims.length; j < 3; j++) {
      positions3D[i * 3 + j] = 0;
    }
  }

  // Calculate bounds
  const bounds = new THREE.Box3();
  const point = new THREE.Vector3();
  for (let i = 0; i < numPoints; i++) {
    point.set(positions3D[i * 3], positions3D[i * 3 + 1], positions3D[i * 3 + 2]);
    bounds.expandByPoint(point);
  }

  // Calculate effective radii if configuration exists and radii are provided
  let finalRadii: Float32Array | Uint8Array | undefined;
  let usedEffectiveRadius = false;

  if (radii) {
    // Use target buffer or allocate (zero-allocation when targetBuffers provided)
    if (targetBuffers && targetBuffers.radii) {
      // Deep integration: Write directly to accumulator radii buffer
      if (radii instanceof Uint8Array && targetBuffers.radii instanceof Uint8Array) {
        (targetBuffers.radii as Uint8Array).set(radii);
        finalRadii = targetBuffers.radii as Uint8Array;
      } else if (radii instanceof Float32Array && targetBuffers.radii instanceof Float32Array) {
        (targetBuffers.radii as Float32Array).set(radii as Float32Array);
        finalRadii = targetBuffers.radii as Float32Array;
      } else {
        // Type mismatch (rare): fallback to conversion
        const float32Radii = radii instanceof Float32Array ? radii : new Float32Array(radii);
        (targetBuffers.radii as Float32Array).set(float32Radii);
        finalRadii = targetBuffers.radii as Float32Array;
      }
    } else {
      // Fallback: Allocate if needed
      finalRadii = radii instanceof Float32Array ? radii : new Float32Array(radii);
    }

    // Normalize uint8 radii to world units before effective radius calculation
    let effectiveRadiusConfig = ctx.effectiveRadiusConfig;
    if (finalRadii instanceof Uint8Array) {
      // Convert Uint8 to Float32 for effective radius calculation
      const float32Radii = new Float32Array(finalRadii.length);
      for (let i = 0; i < finalRadii.length; i++) {
        float32Radii[i] = finalRadii[i] / 255.0;
      }
      // Write to target buffer or use temp array
      if (targetBuffers && targetBuffers.radii instanceof Float32Array) {
        (targetBuffers.radii as Float32Array).set(float32Radii);
        finalRadii = targetBuffers.radii as Float32Array;
      } else {
        finalRadii = float32Radii;
      }

      // Scale max_radius for effective radius calculation
      if (effectiveRadiusConfig) {
        effectiveRadiusConfig = {
          ...effectiveRadiusConfig,
          maxRadius: effectiveRadiusConfig.maxRadius / 255.0,
        };
      }
    }

    if (effectiveRadiusConfig && finalRadii instanceof Float32Array) {
      // Check if we should apply effective radius
      if (shouldApplyEffectiveRadius(effectiveRadiusConfig, viewState.displayDims, true)) {
        const effectiveRadii = calculateEffectiveRadii(
          positions,
          finalRadii,
          viewState,
          effectiveRadiusConfig,
          ndim
        );

        // Write result to target buffer (if using) or replace
        if (targetBuffers && targetBuffers.radii instanceof Float32Array) {
          (targetBuffers.radii as Float32Array).set(effectiveRadii);
          finalRadii = targetBuffers.radii as Float32Array;
        } else {
          finalRadii = effectiveRadii;
        }
        usedEffectiveRadius = true;
      }
    }
  }

  // Filter out zero-radius points to avoid sending them to GPU
  // This significantly improves performance for nD slicing
  // IMPORTANT: Only filter when we actually calculated effective radii
  if (usedEffectiveRadius && finalRadii) {
    const threshold = 0.0001; // Small threshold for floating point precision
    const validIndices: number[] = [];

    // Find indices of points with non-zero radius
    for (let i = 0; i < numPoints; i++) {
      if (finalRadii[i] > threshold) {
        validIndices.push(i);
      }
    }

    const filteredCount = validIndices.length;

    // Only filter if we're actually removing points AND we have valid points left
    if (filteredCount < numPoints && filteredCount > 0) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Filtering out ${numPoints - filteredCount} zero-radius points (keeping ${filteredCount})`
      );

      if (targetBuffers) {
        // In-place compaction into the target buffers (zero
        // allocations). Compact valid points to the buffer start.
        let writeIdx = 0;
        for (let i = 0; i < validIndices.length; i++) {
          const readIdx = validIndices[i];

          // Only copy if read index != write index (avoid redundant copy)
          if (writeIdx !== readIdx) {
            // Compact positions
            positions3D[writeIdx * 3] = positions3D[readIdx * 3];
            positions3D[writeIdx * 3 + 1] = positions3D[readIdx * 3 + 1];
            positions3D[writeIdx * 3 + 2] = positions3D[readIdx * 3 + 2];

            // Compact radii
            if (finalRadii) {
              if (finalRadii instanceof Float32Array) {
                (finalRadii as Float32Array)[writeIdx] = (finalRadii as Float32Array)[readIdx];
              } else {
                (finalRadii as Uint8Array)[writeIdx] = (finalRadii as Uint8Array)[readIdx];
              }
            }

            // Compact colors (type-preserving)
            if (colors && targetBuffers.colors) {
              if (colors instanceof Uint8Array && targetBuffers.colors instanceof Uint8Array) {
                const cb = targetBuffers.colors as Uint8Array;
                cb[writeIdx * 3] = cb[readIdx * 3];
                cb[writeIdx * 3 + 1] = cb[readIdx * 3 + 1];
                cb[writeIdx * 3 + 2] = cb[readIdx * 3 + 2];
              } else if (
                colors instanceof Uint16Array &&
                targetBuffers.colors instanceof Uint16Array
              ) {
                const cb = targetBuffers.colors as Uint16Array;
                cb[writeIdx * 3] = cb[readIdx * 3];
                cb[writeIdx * 3 + 1] = cb[readIdx * 3 + 1];
                cb[writeIdx * 3 + 2] = cb[readIdx * 3 + 2];
              } else if (
                colors instanceof Float32Array &&
                targetBuffers.colors instanceof Float32Array
              ) {
                const cb = targetBuffers.colors as Float32Array;
                cb[writeIdx * 3] = cb[readIdx * 3];
                cb[writeIdx * 3 + 1] = cb[readIdx * 3 + 1];
                cb[writeIdx * 3 + 2] = cb[readIdx * 3 + 2];
              }
            }

            // Compact sharpness (type-preserving)
            if (sharpness && targetBuffers.sharpness) {
              if (
                sharpness instanceof Uint8Array &&
                targetBuffers.sharpness instanceof Uint8Array
              ) {
                (targetBuffers.sharpness as Uint8Array)[writeIdx] = (
                  targetBuffers.sharpness as Uint8Array
                )[readIdx];
              } else if (
                sharpness instanceof Float32Array &&
                targetBuffers.sharpness instanceof Float32Array
              ) {
                (targetBuffers.sharpness as Float32Array)[writeIdx] = (
                  targetBuffers.sharpness as Float32Array
                )[readIdx];
              }
            }
          }

          writeIdx++;
        }

        // Update count to filtered count (arrays already compacted in-place!)
        numPoints = filteredCount;
      } else {
        // Fallback: Create filtered arrays (allocations when accumulator disabled)
        const filteredPositions3D = new Float32Array(filteredCount * 3);
        const filteredRadii = new Float32Array(filteredCount);

        // Filter colors if present
        let filteredColors: Float32Array | Uint8Array | Uint16Array | undefined;
        if (colors) {
          if (colors instanceof Float32Array) {
            filteredColors = new Float32Array(filteredCount * 3);
          } else if (colors instanceof Uint8Array) {
            filteredColors = new Uint8Array(filteredCount * 3);
          } else if (colors instanceof Uint16Array) {
            filteredColors = new Uint16Array(filteredCount * 3);
          }
        }

        // Filter sharpness if present
        let filteredSharpness: Float32Array | Uint8Array | Uint16Array | undefined;
        if (sharpness) {
          if (sharpness instanceof Float32Array) {
            filteredSharpness = new Float32Array(filteredCount);
          } else if (sharpness instanceof Uint8Array) {
            filteredSharpness = new Uint8Array(filteredCount);
          } else if (sharpness instanceof Uint16Array) {
            filteredSharpness = new Uint16Array(filteredCount);
          }
        }

        // Copy only valid points
        for (let i = 0; i < filteredCount; i++) {
          const srcIdx = validIndices[i];

          // Copy position (3 components)
          filteredPositions3D[i * 3] = positions3D[srcIdx * 3];
          filteredPositions3D[i * 3 + 1] = positions3D[srcIdx * 3 + 1];
          filteredPositions3D[i * 3 + 2] = positions3D[srcIdx * 3 + 2];

          // Copy radius
          filteredRadii[i] = finalRadii![srcIdx];

          // Copy colors if present (3 components)
          if (colors && filteredColors) {
            filteredColors[i * 3] = colors[srcIdx * 3];
            filteredColors[i * 3 + 1] = colors[srcIdx * 3 + 1];
            filteredColors[i * 3 + 2] = colors[srcIdx * 3 + 2];
          }

          // Copy sharpness if present
          if (sharpness && filteredSharpness) {
            filteredSharpness[i] = sharpness[srcIdx];
          }
        }

        // Replace arrays with filtered versions
        positions3D = filteredPositions3D;
        finalRadii = filteredRadii;
        colors = filteredColors || colors;
        sharpness = filteredSharpness || sharpness;

        // Update point count
        numPoints = filteredCount;
      }

      // Recalculate bounds for filtered points only
      bounds.makeEmpty();
      for (let i = 0; i < filteredCount; i++) {
        point.set(positions3D[i * 3], positions3D[i * 3 + 1], positions3D[i * 3 + 2]);
        bounds.expandByPoint(point);
      }
    } else if (filteredCount === 0) {
      // All points were filtered out - this is correct behavior for points outside the hyperplane!
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `All ${numPoints} points have zero effective radius - no points visible at this slice`
      );
      // Return empty points - this is the correct behavior
      return createEmptyPointsData(ctx, viewState);
    }
  }

  // Return from accumulator when using target buffers (zero
  // allocations).
  if (targetBuffers && ctx.accumulator) {
    // Data is already in accumulator buffers (written directly during processing)
    // Just update metadata and return (ZERO allocations!)
    ctx.accumulator.updateMetadata({
      bounds,
      usedSpatialIndex: true,
    });

    // Return from accumulator (subarrays are views into accumulator buffers)
    return ctx.accumulator.getData(numPoints);
  }

  // Fallback: Create new LoadedPointsData object (when accumulator disabled)
  return {
    positions: positions3D as PositionArray,
    colors: colors as ColorArray | undefined,
    radii: finalRadii as ScalarArray | undefined,
    sharpness: sharpness as ScalarArray | undefined,
    pointCount: numPoints,
    ndim,
    metadata: {
      totalPoints: ctx.nodeAttrs.n_points || totalPoints,
      loadedPoints: numPoints,
      bounds,
      usedSpatialIndex: true,
      usedEffectiveRadius,
      dtypes: dtypesFromAttrs(ctx.nodeAttrs),
    },
  };
}

/**
 * Project nD points to 3D display space using a web worker.
 *
 * Offloads CPU-intensive projection work (nD → 3D extraction,
 * effective-radius computation, zero-radius filter, bounds calc) to a
 * worker thread via the global worker pool. Uses `Comlink.transfer()`
 * under the hood for zero-copy `ArrayBuffer` transfer.
 *
 * On any worker failure (`runWithTimeout` rejection — including the
 * {@link import('../../workers/worker-pool').WorkerTimeoutError} fired
 * after `workerProjectionTimeoutMs` — RPC error, or structured clone
 * failure) the call falls back to `projectPointsTo3D` on the main
 * thread with `targetBuffers=null` — no accumulator path on fallback.
 */
export async function projectPointsTo3DUsingWorker(
  positions: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  colors: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  radii: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  sharpness: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  viewState: ViewState,
  ranges: PointRange[],
  ctx: ProjectionContext
): Promise<LoadedPointsData> {
  const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

  if (!positions) {
    throw new Error('[PointsProjection] Positions data is required for points');
  }

  // Calculate ndim from actual positions array
  const ndim =
    totalPoints > 0
      ? Math.round(positions.length / totalPoints)
      : ctx.chunkIndex?.metadata.ndim || 3;

  // Convert positions to Float32Array if needed (positions must be Float32)
  const positionsFloat32 =
    positions instanceof Float32Array ? positions : new Float32Array(positions);

  // Colors: Keep native type! Worker and GPU buffer pool support multi-type (Uint8/Uint16/Float32)
  // THREE.js handles normalization in shader via normalized attribute flag
  // Float16Array needs conversion to Float32Array (worker doesn't support Float16)
  // Note: Float16 values are already in float range, no normalization needed
  let colorsMultiType: Float32Array | Uint8Array | Uint16Array | null = null;
  if (colors) {
    if (colors instanceof Float16Array) {
      // Convert Float16 to Float32 (no normalization - already in float range)
      colorsMultiType = new Float32Array(colors);
    } else {
      colorsMultiType = colors;
    }
  }

  // Radii/sharpness: Convert to Float32Array (no normalization needed - already world units)
  const radiiFloat32 = radii
    ? radii instanceof Float32Array
      ? radii
      : new Float32Array(radii)
    : null;

  const sharpnessFloat32 = sharpness
    ? sharpness instanceof Float32Array
      ? sharpness
      : new Float32Array(sharpness)
    : null;

  // Build effective radius config for worker
  let workerEffectiveRadiusConfig: { spatialExtendDims: boolean[]; maxRadius: number } | null =
    null;
  if (ctx.effectiveRadiusConfig && radiiFloat32) {
    // Check if we should apply effective radius
    if (shouldApplyEffectiveRadius(ctx.effectiveRadiusConfig, viewState.displayDims, true)) {
      workerEffectiveRadiusConfig = {
        spatialExtendDims: ctx.effectiveRadiusConfig.spatialExtendDims,
        maxRadius: ctx.effectiveRadiusConfig.maxRadius,
      };
    }
  }

  try {
    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `Projecting ${totalPoints} points to 3D using worker (ndim=${ndim})`
    );

    const workerResult = await getWorkerPool().runWithTimeout(
      'projectPointsTo3D',
      'projection',
      (api) =>
        api.projectPointsTo3D({
          positions: positionsFloat32,
          colors: colorsMultiType,
          radii: radiiFloat32,
          sharpness: sharpnessFloat32,
          viewState: {
            displayDims: viewState.displayDims,
            slicePosition: viewState.slicePosition,
            tolerance: viewState.tolerance,
          },
          effectiveRadiusConfig: workerEffectiveRadiusConfig,
          ndim,
          numPoints: totalPoints,
        })
    );

    // Handle empty result
    if (workerResult.visibleCount === 0) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Worker projection: all ${totalPoints} points have zero effective radius`
      );
      return createEmptyPointsData(ctx, viewState);
    }

    // Build THREE.Box3 from worker bounds
    const bounds = new THREE.Box3(
      new THREE.Vector3(...workerResult.bounds.min),
      new THREE.Vector3(...workerResult.bounds.max)
    );

    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `Worker projection complete: ${workerResult.visibleCount}/${totalPoints} visible points`
    );

    return {
      positions: workerResult.positions3D as PositionArray,
      colors: workerResult.colors as ColorArray | undefined,
      radii: workerResult.radii as ScalarArray | undefined,
      sharpness: workerResult.sharpness as ScalarArray | undefined,
      pointCount: workerResult.visibleCount,
      ndim,
      metadata: {
        totalPoints: ctx.nodeAttrs.n_points || totalPoints,
        loadedPoints: workerResult.visibleCount,
        bounds,
        usedSpatialIndex: true,
        usedEffectiveRadius: !!workerEffectiveRadiusConfig,
        dtypes: dtypesFromAttrs(ctx.nodeAttrs),
      },
    };
  } catch (error) {
    // Fallback to main thread on worker failure
    log.warning(
      Modules.SPATIAL_INDEX_LOADER,
      'Worker projection failed, falling back to main thread:',
      error
    );
    return projectPointsTo3D(positions, colors, radii, sharpness, viewState, ranges, ctx, null);
  }
}
