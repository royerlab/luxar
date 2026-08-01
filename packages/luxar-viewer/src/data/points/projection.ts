/**
 * nD → 3D projection for the points spatial-index loader.
 *
 * Two behaviors live here, extracted from
 * `data/points-spatial-index-loader.ts`:
 *
 *   - `projectPointsTo3D` — the single, **WASM-accelerated** Points
 *     projection (nD→3D extraction + effective-radius via the WASM
 *     kernels), with optional accumulator-buffer write-through
 *     (zero-allocation) and effective-radius filtering. Points run on the
 *     main thread by design (memory-bandwidth-bound + zero-alloc
 *     accumulator); the former worker-dispatcher copy was deleted in W4b.
 *   - `createEmptyPointsData` — factory for the "no visible points" return.
 *
 * Both are pure with respect to the loader: the loader-side state
 * (`chunkIndex`, `effectiveRadiusConfig`, `accumulator`, node attrs) is
 * passed in via a small `ProjectionContext`, and the WASM backend is
 * supplied as an argument (via `getPointsBackend`), so the helpers can be
 * unit-tested without instantiating a full `PointsSpatialIndexLoader`.
 *
 * @module data/point-loader/projection
 */

import * as THREE from 'three';
import { assertColorLayout } from '../loaders/color-loader';
import { log, Modules } from '../../utils/log';
import {
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
  PointScalarArray,
} from '../data-loader-types';
import { LoadedPointsDataAccumulator } from '../accumulators/points';
import type { PointsMetadata } from '../../types/points';
import type { PointsChunkIndex } from './chunk-index-loader';
import type { WasmModule } from '../../wasm/types';
import { validateProjectionInputs } from '../../workers/data-worker/validation';

/** Buffers an accumulator owns; writing directly into them avoids allocations. */
export interface ProjectionTargetBuffers {
  positions3D: Float32Array;
  colors: ColorArray;
  radii: ScalarArray;
  // sharpness/scalars are kept in their native dtype through the accumulator
  // (like colors) so the GPU upload site can normalize uint8/uint16 (÷255 /
  // ÷65535); PointScalarArray adds the native Uint16Array variant.
  sharpness: PointScalarArray;
  /** optional scalar target — present when the accumulator's
   *  hasScalars flag is true so the projection can write through. */
  scalars?: PointScalarArray;
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
  scalars: string | undefined;
} {
  return {
    positions: nodeAttrs.position_dtype as string | undefined,
    colors: nodeAttrs.color_dtype as string | undefined,
    radii: nodeAttrs.radius_dtype as string | undefined,
    sharpness: nodeAttrs.sharpness_dtype as string | undefined,
    scalars: (nodeAttrs as { scalar_dtype?: string }).scalar_dtype,
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
 * Project nD points to 3D display space on the main thread, **WASM-accelerated**.
 *
 * The two expensive steps — nD→3D extraction and effective-radius
 * computation — run through the compiled WASM kernels (`extract_3d_positions`,
 * `calculate_effective_radii`) on the `wasm` module the caller supplies
 * (compiled WASM, or the uncapped TS reference for `ndim > 16` — see
 * `getPointsBackend`). The zero-radius filter, multi-type compaction,
 * uint8-radius `/255` normalization, scalar handling, and bounds stay in
 * TypeScript: they are cheap, multi-type, and tightly coupled to the
 * accumulator's in-place reuse contract.
 *
 * Points run on the main thread (not a worker) by design: projection is
 * memory-bandwidth-bound and pairs with the zero-allocation accumulator,
 * so offloading would pay transfer cost both ways for little compute
 * gain. This is the single Points projection implementation — the former
 * worker dispatcher copy was deleted in W4b.
 *
 * Two execution paths share this entry point:
 *
 *  - **Accumulator path** (`targetBuffers` provided): the supported
 *    hot path. Writes through preallocated accumulator buffers for
 *    zero-allocation operation. Production code always takes this
 *    path via `LoadedPointsDataAccumulator`.
 *  - **Fallback path** (`targetBuffers` null/undefined): allocates
 *    fresh arrays. Used by tests and the explicit no-accumulator opt-out.
 *    Color/sharpness inputs pass through by reference; positions3D and
 *    (when filtering applies) radii are freshly allocated.
 *
 * @param wasm - WASM backend (compiled or TS-reference fallback) supplying
 *               the projection kernels. Obtain via `getPointsBackend(ndim)`.
 * @param targetBuffers - Optional accumulator buffers for zero-allocation
 *                        operation. When provided, writes directly
 *                        through; when null/undefined, allocates new
 *                        arrays for the result.
 */
export function projectPointsTo3D(
  wasm: WasmModule,
  positions: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  colors: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  radii: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  sharpness: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  viewState: ViewState,
  ranges: PointRange[],
  ctx: ProjectionContext,
  targetBuffers?: ProjectionTargetBuffers | null,
  /** optional per-point scalars for colormap lookup. */
  scalars?: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
  /**
   * Components per color item: 3 = RGB (default), 4 = RGBA (the alpha
   * column is per-point opacity). Strides every color guard/compaction
   * below — a hardcoded 3 would truncate + misalign RGBA data (the
   * gsplat colorK lesson).
   */
  colorComponents: 3 | 4 = 3
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

  // Validate the calculation. Fail fast on size mismatch — proceeding
  // with the rounded `ndim` produces silently wrong projections
  // downstream (the per-point inner loop reads `positions[i * ndim + d]`
  // with the wrong stride and writes garbage into positions3D).
  if (totalPoints > 0 && positions.length !== totalPoints * ndim) {
    throw new Error(
      `[PointsProjection] Position data size mismatch: ${positions.length} elements ` +
        `for ${totalPoints} points doesn't divide evenly (calculated ndim=${ndim}). ` +
        'This indicates encoding metadata issues.'
    );
  }

  // WASM-boundary guards. This function now feeds the WASM kernels
  // (`extract_3d_positions`, `calculate_effective_radii`) directly, so
  // out-of-range displayDims or short radii / spatialExtendDims / tolerance
  // would cause OOB reads inside WASM. Reject them here — the same guards
  // the former worker dispatcher applied at the RPC boundary. Skipped for
  // the empty (totalPoints === 0) early-out, where no kernel reads occur.
  if (totalPoints > 0) {
    // requireSlicePosition=false: extract_3d_positions doesn't read
    // slicePosition (a pure-3D view may carry a short/empty one); the
    // effective-radius branch below validates its length where the kernel
    // actually consumes it.
    validateProjectionInputs(
      'projectPointsTo3D',
      positions,
      viewState.displayDims,
      viewState.slicePosition,
      ndim,
      totalPoints,
      ndim,
      false
    );
    if (radii && radii.length < totalPoints) {
      throw new Error(
        `projectPointsTo3D: radii too short (got ${radii.length}, expected ≥ ${totalPoints})`
      );
    }
    if (sharpness && sharpness.length < totalPoints) {
      throw new Error(
        `projectPointsTo3D: sharpness too short (got ${sharpness.length}, expected ≥ ${totalPoints})`
      );
    }
    // RGB(A) tuple per point — STRICT equality, not a minimum: loaders
    // emit exact-length views, and a minimum check (4N ≥ 3N) would let
    // an RGBA array with an undeclared colorComponents silently
    // mis-stride every point after the first (see assertColorLayout).
    assertColorLayout(colors, totalPoints, colorComponents, 'projectPointsTo3D');
    if (ctx.effectiveRadiusConfig) {
      // The effective-radius kernel reads slicePosition[d] for d < ndim.
      if (viewState.slicePosition.length < ndim) {
        throw new Error(
          'projectPointsTo3D: slicePosition too short for effective radius ' +
            `(got ${viewState.slicePosition.length}, expected ≥ ${ndim})`
        );
      }
      if (ctx.effectiveRadiusConfig.spatialExtendDims.length < ndim) {
        throw new Error(
          'projectPointsTo3D: effectiveRadiusConfig.spatialExtendDims too short ' +
            `(got ${ctx.effectiveRadiusConfig.spatialExtendDims.length}, expected ≥ ${ndim})`
        );
      }
      if (!Number.isFinite(ctx.effectiveRadiusConfig.maxRadius)) {
        throw new Error(
          `projectPointsTo3D: effectiveRadiusConfig.maxRadius=${ctx.effectiveRadiusConfig.maxRadius} must be a finite number`
        );
      }
      if (!viewState.tolerance || viewState.tolerance.length < ndim) {
        throw new Error(
          'projectPointsTo3D: viewState.tolerance too short for effective radius ' +
            `(got ${viewState.tolerance?.length ?? 0}, expected ≥ ${ndim})`
        );
      }
    }
  }

  // Validate scalar length matches point count. Mismatch suppresses the
  // scalar branch (fail-closed) — geometry renders without colormap
  // rather than carrying truncated/over-large scalar arrays into the GPU
  // pool and producing garbage LUT lookups.
  if (scalars && scalars.length !== totalPoints) {
    log.warning(
      Modules.SPATIAL_INDEX_LOADER,
      `Scalar length mismatch: ${scalars.length} scalars for ${totalPoints} points ` +
        '(expected one scalar per point). Suppressing scalar projection — ' +
        'colormap mode will be off until the source data is fixed.'
    );
    scalars = null;
  }

  let numPoints = totalPoints;

  // Use target buffers if provided (zero allocations).
  const { displayDims } = viewState;

  // Use target buffer or allocate new (zero-allocation when targetBuffers provided)
  let positions3D = targetBuffers ? targetBuffers.positions3D : new Float32Array(numPoints * 3);

  // Extract 3D positions from nD data via WASM (extract_3d_positions writes
  // displayed dims and zero-fills any remaining slots, matching the former
  // TS loop). The kernel requires Float32 input; positions are Float32 in
  // production but coerce defensively for non-Float32 sources.
  const positionsF32 = positions instanceof Float32Array ? positions : new Float32Array(positions);
  const displayDimsU32 = new Uint32Array(displayDims);
  wasm.extract_3d_positions(positionsF32, displayDimsU32, ndim, numPoints, positions3D);

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
  // Float effective radii in WORLD units, retained for the zero-radius cull
  // decision below. On the uint8-accumulator path `finalRadii` is re-encoded
  // to uint8 (so the buffer getData() returns is the one compaction shuffles
  // — issue #740), but the cull threshold (`radiusThresholdAnchor`) is a
  // world-unit anchor, so the keep/remove decision must read these floats,
  // not the re-encoded uint8.
  let effectiveRadiiFloat: Float32Array | undefined;
  // World-unit max radius (attrs.max_radius). For uint8 radii the on-disk
  // value is a normalized [0,255] encoding of `(u8/255)·max_radius` world
  // units; we keep this to (a) decode radii to WORLD units before the kernel
  // (so radius and slice distance share units) and (b) re-encode the
  // world-unit effective radii back to uint8 with the SAME divisor the
  // renderer multiplies by. Defaults 1.0 (no effectiveRadiusConfig → the
  // temp is unused; the kernel won't run).
  let maxRadiusWorld = 1.0;
  // Scale anchor for the boundary-dust filter below, in WORLD units (equals
  // effectiveRadiusConfig.maxRadius, the same units as effectiveRadiiFloat).
  let radiusThresholdAnchor = 1.0;

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

    // Decode uint8 radii to WORLD units before effective radius calculation.
    const effectiveRadiusConfig = ctx.effectiveRadiusConfig;
    if (finalRadii instanceof Uint8Array) {
      // Uint8 radii are a normalized [0,255] encoding of world-unit radii:
      // world_radius = (u8/255)·max_radius. Decode to WORLD units (not just
      // /255) so the effective-radius kernel compares radius and slice
      // distance in the SAME units — D stays in world/data units. Leaving
      // radii normalized while D was world-unit mis-scaled the attenuation
      // whenever max_radius != 1 (issue #740 revision).
      maxRadiusWorld = effectiveRadiusConfig?.maxRadius ?? 1.0;
      const float32Radii = new Float32Array(finalRadii.length);
      for (let i = 0; i < finalRadii.length; i++) {
        float32Radii[i] = (finalRadii[i] / 255.0) * maxRadiusWorld;
      }
      // Write to target buffer or use temp array
      if (targetBuffers && targetBuffers.radii instanceof Float32Array) {
        (targetBuffers.radii as Float32Array).set(float32Radii);
        finalRadii = targetBuffers.radii as Float32Array;
      } else {
        finalRadii = float32Radii;
      }
      // effectiveRadiusConfig.maxRadius is intentionally kept in WORLD units
      // (no /255 rescale) so the kernel sees world-unit radii matching D.
    }

    if (effectiveRadiusConfig && finalRadii instanceof Float32Array) {
      // Check if we should apply effective radius
      if (shouldApplyEffectiveRadius(effectiveRadiusConfig, viewState.displayDims, true)) {
        // WASM effective-radius kernel. extend_to_all dims (tolerance ≥ 1e9)
        // are folded into the display-dims set so the kernel skips them
        // entirely (no discrete check, no distance) — the same construction
        // the worker dispatcher used, matching the deleted TS
        // `calculateEffectiveRadii`'s internal `tolerance >= 1e9` skip.
        const slicePositionF32 = new Float32Array(viewState.slicePosition);
        const spatialExtendDimsU8 = new Uint8Array(
          effectiveRadiusConfig.spatialExtendDims.map((b) => (b ? 1 : 0))
        );
        const extendToAllDims: number[] = [];
        for (let d = 0; d < ndim; d++) {
          const tol = viewState.tolerance[d];
          if (!displayDims.includes(d) && Number.isFinite(tol) && tol >= 1e9) {
            extendToAllDims.push(d);
          }
        }
        const effDisplayDims =
          extendToAllDims.length > 0
            ? new Uint32Array([...displayDims, ...extendToAllDims])
            : displayDimsU32;

        const effectiveRadii = new Float32Array(numPoints);
        wasm.calculate_effective_radii(
          positionsF32,
          finalRadii,
          effDisplayDims,
          slicePositionF32,
          spatialExtendDimsU8,
          ndim,
          numPoints,
          effectiveRadii
        );

        // Keep the float (WORLD-unit) effective radii for the cull decision
        // below regardless of the accumulator dtype.
        effectiveRadiiFloat = effectiveRadii;

        // Write result to target buffer (if using) or replace
        if (targetBuffers && targetBuffers.radii instanceof Float32Array) {
          (targetBuffers.radii as Float32Array).set(effectiveRadii);
          finalRadii = targetBuffers.radii as Float32Array;
        } else if (targetBuffers && targetBuffers.radii instanceof Uint8Array) {
          // Uint8 accumulator radii (issue #740): keep radii dtype-preserving
          // as uint8 THROUGH the accumulator so (a) the renderer's
          // (u8/255)·maxRadius contract stays valid and (b) the buffer
          // getData() returns is the SAME one the in-place compaction below
          // shuffles. Pre-fix, effective radii were computed into a temp
          // Float32 array that getData() never saw, so surviving points
          // rendered with raw/misaligned uint8 radii. effectiveRadii are in
          // WORLD units, so re-encode with the SAME divisor the renderer
          // multiplies by: u8 = round(R_eff_world / max_radius · 255). Since
          // R_eff_world ≤ R_world ≤ max_radius the ratio is in [0,1].
          const u8 = targetBuffers.radii as Uint8Array;
          const invMax = maxRadiusWorld > 0 ? 255 / maxRadiusWorld : 0;
          for (let i = 0; i < numPoints; i++) {
            const q = Math.round(effectiveRadii[i] * invMax);
            u8[i] = q < 0 ? 0 : q > 255 ? 255 : q;
          }
          finalRadii = u8;
        } else {
          finalRadii = effectiveRadii;
        }
        usedEffectiveRadius = true;
        radiusThresholdAnchor = effectiveRadiusConfig.maxRadius;
      }
    }
  }

  // Filter out zero-radius points to avoid sending them to GPU
  // This significantly improves performance for nD slicing
  // IMPORTANT: Only filter when we actually calculated effective radii
  if (usedEffectiveRadius && finalRadii) {
    // SCALE-FREE boundary-dust threshold: relative to the dataset's own
    // max radius, never an absolute world-unit constant — an absolute
    // 1e-4 discarded EVERY point of scenes authored in units where radii
    // are sub-1e-4 (e.g. meter-unit data with micron-scale points; found
    // by a real-GPU unit-extremes probe). 1e-6 of maxRadius keeps
    // everything except the true slice-boundary sliver.
    const threshold = radiusThresholdAnchor * 1e-6;
    const validIndices: number[] = [];

    // Decide kept/removed from the float effective radii (WORLD units,
    // matching `threshold`), NOT the possibly-re-encoded `finalRadii`: on the
    // uint8 path finalRadii holds 0..255 values that would compare wrongly
    // against the tiny world-unit threshold (issue #740). Falls back to
    // finalRadii for any path that didn't retain the floats.
    const cullRadii = effectiveRadiiFloat ?? finalRadii;
    // On the re-encoded uint8 path a world-unit radius can clear the dust
    // threshold (maxRadius·1e-6) yet still quantize to u8 = 0 — anything
    // below maxRadius/510. Such a point is invisible at uint8 resolution
    // but would survive the cull and occupy pointCount/GPU slots, defeating
    // the compaction at exactly the slice boundary. Require the ENCODED
    // value to be nonzero too.
    const encodedU8 = finalRadii instanceof Uint8Array ? finalRadii : null;

    // Find indices of points with non-zero radius
    for (let i = 0; i < numPoints; i++) {
      if (cullRadii[i] > threshold && (!encodedU8 || encodedU8[i] > 0)) {
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

            // Compact colors (type-preserving, strided by the RGB(A) layout)
            if (colors && targetBuffers.colors) {
              if (colors instanceof Uint8Array && targetBuffers.colors instanceof Uint8Array) {
                const cb = targetBuffers.colors as Uint8Array;
                for (let c = 0; c < colorComponents; c++) {
                  cb[writeIdx * colorComponents + c] = cb[readIdx * colorComponents + c];
                }
              } else if (
                colors instanceof Uint16Array &&
                targetBuffers.colors instanceof Uint16Array
              ) {
                const cb = targetBuffers.colors as Uint16Array;
                for (let c = 0; c < colorComponents; c++) {
                  cb[writeIdx * colorComponents + c] = cb[readIdx * colorComponents + c];
                }
              } else if (
                colors instanceof Float32Array &&
                targetBuffers.colors instanceof Float32Array
              ) {
                const cb = targetBuffers.colors as Float32Array;
                for (let c = 0; c < colorComponents; c++) {
                  cb[writeIdx * colorComponents + c] = cb[readIdx * colorComponents + c];
                }
              }
            }

            // Compact sharpness (type-preserving). The shuffle only touches
            // the accumulator target buffer, so the SOURCE dtype is irrelevant
            // — gating on `sharpness instanceof ...` here would skip the move
            // for a Float16/Uint16 source and corrupt compaction. Uint8Array,
            // Uint16Array, and Float32Array all support numeric index assignment.
            if (sharpness && targetBuffers.sharpness) {
              const sb = targetBuffers.sharpness as Uint8Array | Uint16Array | Float32Array;
              sb[writeIdx] = sb[readIdx];
            }

            // compact scalars (type-preserving) — symmetric with the
            // colors / sharpness paths above. The accumulator owns the
            // target buffer; we just shuffle indices in place regardless of
            // the source dtype.
            if (scalars && targetBuffers.scalars) {
              const sb = targetBuffers.scalars as Uint8Array | Uint16Array | Float32Array;
              sb[writeIdx] = sb[readIdx];
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

        // Filter colors if present (strided by the RGB(A) layout)
        let filteredColors: Float32Array | Uint8Array | Uint16Array | undefined;
        if (colors) {
          if (colors instanceof Float32Array) {
            filteredColors = new Float32Array(filteredCount * colorComponents);
          } else if (colors instanceof Uint8Array) {
            filteredColors = new Uint8Array(filteredCount * colorComponents);
          } else if (colors instanceof Uint16Array) {
            filteredColors = new Uint16Array(filteredCount * colorComponents);
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
          } else {
            // Float16Array (or any other) source → widen value-preserving to
            // Float32. Without this else the array stayed undefined, the copy
            // loop was skipped, and `sharpness = filteredSharpness || sharpness`
            // kept the ORIGINAL full-length array while numPoints=filteredCount
            // → misaligned attributes (issue #751, Finding 2).
            filteredSharpness = new Float32Array(filteredCount);
          }
        }

        // filter scalars if present (no-accumulator fallback path).
        let filteredScalars: Float32Array | Uint8Array | Uint16Array | undefined;
        if (scalars) {
          if (scalars instanceof Float32Array) {
            filteredScalars = new Float32Array(filteredCount);
          } else if (scalars instanceof Uint8Array) {
            filteredScalars = new Uint8Array(filteredCount);
          } else if (scalars instanceof Uint16Array) {
            filteredScalars = new Uint16Array(filteredCount);
          } else {
            // Float16Array source → widen value-preserving to Float32 (same
            // misaligned-attribute hazard as sharpness above).
            filteredScalars = new Float32Array(filteredCount);
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

          // Copy colors if present (RGB(A) — colorComponents per point)
          if (colors && filteredColors) {
            for (let c = 0; c < colorComponents; c++) {
              filteredColors[i * colorComponents + c] = colors[srcIdx * colorComponents + c];
            }
          }

          // Copy sharpness if present
          if (sharpness && filteredSharpness) {
            filteredSharpness[i] = sharpness[srcIdx];
          }

          // copy scalar if present
          if (scalars && filteredScalars) {
            filteredScalars[i] = scalars[srcIdx];
          }
        }

        // Replace arrays with filtered versions
        positions3D = filteredPositions3D;
        finalRadii = filteredRadii;
        colors = filteredColors || colors;
        sharpness = filteredSharpness || sharpness;
        scalars = filteredScalars || scalars;

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
    // Note: `bounds` intentionally omitted — accumulator.getData() will
    // recompute bounds from the position buffer, so passing the local
    // `bounds` here would be a no-op (and would warn).
    ctx.accumulator.updateMetadata({
      usedSpatialIndex: true,
    });

    // Return from accumulator (subarrays are views into accumulator buffers)
    return ctx.accumulator.getData(numPoints);
  }

  // Fallback: Create new LoadedPointsData object (when accumulator disabled)
  return {
    positions: positions3D as PositionArray,
    colors: colors as ColorArray | undefined,
    colorComponents: colors ? colorComponents : undefined,
    radii: finalRadii as ScalarArray | undefined,
    sharpness: sharpness as PointScalarArray | undefined,
    // pass scalars through. They are already type-compacted above
    // (or unchanged when no filtering occurred).
    scalars: (scalars as PointScalarArray | null | undefined) ?? undefined,
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
