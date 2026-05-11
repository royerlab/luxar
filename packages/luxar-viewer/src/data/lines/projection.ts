/**
 * Pure projection math + WASM dispatch for the lines geometry.
 *
 * Extracted from `lines-spatial-index-loader.ts` so the clipping math,
 * the per-vertex interpolation primitives, and the WASM/TS instance-
 * buffer builders can be unit-tested without an active zarr store or a
 * loaded scene. Mirrors the layout of `points/projection.ts` for the
 * points geometry.
 *
 * - `clipSegmentToSlice`, `lerp`, `lerpVec3`, `distance3D` —
 *   per-segment math primitives (TS path).
 * - `buildInstanceBuffers` — TS path: clip + interpolate + build the
 *   GPU instance buffers from a `LoadedLinesData` blob.
 * - `buildInstanceBuffersWASM` — WASM-accelerated equivalent that
 *   batches clipping, interpolation, and length calculations through
 *   the compiled module.
 * - `initLinesWASM` — eager warm-up of the WASM module so the first
 *   `buildInstanceBuffersWASM` call doesn't pay the load cost.
 *
 * @module data/lines/projection
 */

import { coerceColorsToFloat32 } from '../../workers/color-utils';
import { config as appConfig } from '../../config';
import { log, Modules, LogEmoji } from '../../utils/log';
import type {
  LinesMetadata,
  LoadedLinesData,
  ProcessedLinesData,
  ClippedSegment,
} from '../../types/lines';
import { initWasm, getFallback } from '../../wasm';
import type { WasmModule } from '../../wasm/types';

/**
 * Construct the canonical "no visible lines at this slice" payload.
 * Mirrors `createEmptyPointsData` in `points/projection.ts` and
 * `createEmptyGSplatsData` in `gsplats/projection.ts`.
 */
export function createEmptyLinesData(attrs: LinesMetadata): LoadedLinesData {
  return {
    positions: new Float32Array(0),
    segments: new Uint32Array(0),
    widths: new Float32Array(0),
    colors: null,
    sharpness: null,
    scalars: null, // C4
    segmentCount: 0,
    vertexCount: 0,
    ndim: attrs.ndim,
  };
}

// ============================================================================
// WASM Module Caching for Hot Path Optimization
// ============================================================================

/** Cached WASM module instance (lazily initialized) */
let wasmModuleCache: WasmModule | null = null;
let wasmInitPromise: Promise<WasmModule> | null = null;

/**
 * Get the WASM module, initializing if necessary.
 * Uses caching to avoid repeated initialization overhead.
 */
async function getWasmModule(): Promise<WasmModule> {
  if (wasmModuleCache) {
    return wasmModuleCache;
  }

  if (!wasmInitPromise) {
    wasmInitPromise = initWasm().then((module) => {
      wasmModuleCache = module;
      return module;
    });
  }

  return wasmInitPromise;
}

/**
 * Get the WASM module synchronously (returns fallback if not yet initialized).
 * Used in hot paths where async is not desirable.
 */
function getWasmModuleSync(): WasmModule {
  if (wasmModuleCache) {
    return wasmModuleCache;
  }
  // Return fallback if WASM not yet loaded
  return getFallback();
}

// ============================================================================
// Pure math primitives
// ============================================================================

/**
 * Clip a single segment to the visible nD slice along all hidden
 * dimensions. Returns the 3D-projected endpoints and the parametric
 * positions `t1, t2` along the original segment so per-vertex
 * attributes can be interpolated to match.
 *
 * The five-case algorithm:
 *
 * - A: Both endpoints inside the slice → render the original segment.
 * - B: One inside, one outside → clip the outside endpoint.
 * - C: One inside (at the boundary), one outside → clip the boundary side.
 * - D: Both outside, opposite sides → clip both (segment crosses slice).
 * - E: Both outside, same side → don't render.
 *
 * @param p1 - Start vertex (nD)
 * @param p2 - End vertex (nD)
 * @param slicePosition - Current slice position in nD
 * @param tolerance - Per-dimension tolerance
 * @param displayDims - Which dimensions to display [d0, d1, d2]
 * @returns Clipped segment with interpolation parameters
 */
export function clipSegmentToSlice(
  p1: number[],
  p2: number[],
  slicePosition: readonly number[],
  tolerance: readonly number[],
  displayDims: readonly number[]
): ClippedSegment {
  let t1 = 0.0; // Parameter at start
  let t2 = 1.0; // Parameter at end

  for (let dim = 0; dim < p1.length; dim++) {
    if (displayDims.includes(dim)) continue; // Skip displayed dimensions

    const tol = Number.isFinite(tolerance[dim]) ? tolerance[dim] : 0;
    const slicePos = slicePosition[dim] ?? 0;
    const sliceMin = slicePos - tol;
    const sliceMax = slicePos + tol;

    const v1 = p1[dim];
    const v2 = p2[dim];

    // Classify endpoints relative to slice
    const p1In = v1 >= sliceMin && v1 <= sliceMax;
    const p2In = v2 >= sliceMin && v2 <= sliceMax;

    if (p1In && p2In) {
      // Both in - no clipping needed for this dimension
      continue;
    }

    if (!p1In && !p2In) {
      // Both out - check if on same side (Case E)
      if ((v1 < sliceMin && v2 < sliceMin) || (v1 > sliceMax && v2 > sliceMax)) {
        return { p1: [], p2: [], t1: 0, t2: 0, visible: false };
      }
      // Opposite sides - clip both (Case D)
    }

    // Compute intersection parameters
    const dv = v2 - v1;
    if (Math.abs(dv) < 1e-10) continue; // Parallel to slice

    // t where line crosses sliceMin and sliceMax
    const tMin = (sliceMin - v1) / dv;
    const tMax = (sliceMax - v1) / dv;

    // Clip t1 (entry) and t2 (exit) to valid range
    if (dv > 0) {
      // Moving from low to high
      t1 = Math.max(t1, tMin);
      t2 = Math.min(t2, tMax);
    } else {
      // Moving from high to low
      t1 = Math.max(t1, tMax);
      t2 = Math.min(t2, tMin);
    }

    if (t1 >= t2) {
      return { p1: [], p2: [], t1: 0, t2: 0, visible: false }; // No valid range
    }
  }

  // Interpolate clipped positions (in full nD space)
  const clippedP1 = p1.map((v, i) => v + t1 * (p2[i] - v));
  const clippedP2 = p1.map((v, i) => v + t2 * (p2[i] - v));

  // Project to 3D display space
  const display1 = displayDims.map((d) => clippedP1[d]);
  const display2 = displayDims.map((d) => clippedP2[d]);

  // Pad to 3D if fewer than 3 display dims
  while (display1.length < 3) display1.push(0);
  while (display2.length < 3) display2.push(0);

  return { p1: display1, p2: display2, t1, t2, visible: true };
}

/** Linear interpolation between two values. */
export function lerp(a: number, b: number, t: number): number {
  return a + t * (b - a);
}

/** Linear interpolation between two 3-component vectors. */
export function lerpVec3(a: number[], b: number[], t: number): number[] {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

/** Calculate 3D Euclidean distance. */
export function distance3D(a: number[], b: number[]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = b[2] - a[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// ============================================================================
// Instance buffer builders
// ============================================================================

/**
 * Build GPU instance buffers from loaded lines data (TS path).
 *
 * Transforms per-vertex data into per-segment instance attributes:
 * - Clips segments to nD slice
 * - Interpolates attributes for clipped endpoints
 * - Calculates 3D segment lengths
 * - Tracks which endpoints were clipped
 *
 * **Allocation profile (accepted trade-off for the fallback path).**
 * This TS path runs only as a fallback when the WASM module is
 * unavailable or when a worker call has timed out and recovery is on
 * the main thread. It allocates per-segment temporaries (color lerps,
 * slice copies) — fine for the ≤10k–100k-segment workloads that hit
 * this path in practice. The hot path is the WASM batch builder
 * (`workers/data-worker.ts::projectLinesTo3D`), which writes through
 * pre-allocated output buffers, uses `coerceColorsToFloat32` for
 * zero-copy on Float32 inputs, and handles per-vertex scalars via
 * `interpolate_scalars_batch` so colormap-enabled Lines stay on the
 * worker thread. To measure the fallback overhead vs the WASM path
 * run `src/tests/benchmarks/lines-ts-fallback-alloc-bench.ts`.
 *
 * @param loadedData - Raw lines data from loader
 * @param slicePosition - Current position in nD space
 * @param tolerance - Per-dimension tolerance
 * @param displayDims - Which dimensions to display
 * @returns Processed data ready for GPU
 */
export function buildInstanceBuffers(
  loadedData: LoadedLinesData,
  slicePosition: readonly number[],
  tolerance: readonly number[],
  displayDims: readonly number[]
): ProcessedLinesData {
  const { positions, segments, widths, colors, sharpness, ndim, segmentCount } = loadedData;
  // A.3: validate per-vertex scalar length matches vertex count. Mismatch
  // suppresses the scalar branch (fail-closed) — line geometry renders
  // without colormap rather than carrying mismatched per-vertex values
  // through the WASM batch interpolator (which would silently read OOB).
  const vertexCount = ndim > 0 ? Math.floor(positions.length / ndim) : 0;
  let scalars: typeof loadedData.scalars = loadedData.scalars;
  if (scalars && scalars.length !== vertexCount) {
    log.warning(
      Modules.LINES_LOADER,
      `Scalar length mismatch: ${scalars.length} scalars for ${vertexCount} vertices ` +
        '(expected one scalar per vertex). Suppressing scalar projection — ' +
        'colormap mode will be off until the source data is fixed.'
    );
    scalars = null;
  }

  // Diagnostic: log input shape for large-segment-count Lines updates.
  if (segmentCount > 10000 && appConfig.dataLoading.performance.enablePerformanceMonitoring) {
    log.custom(LogEmoji.DEBUG, Modules.LINES_LOADER, 'buildInstanceBuffers input');
    log.custom(LogEmoji.DEBUG, Modules.LINES_LOADER, 'segmentCount', segmentCount);
    log.custom(
      LogEmoji.DEBUG,
      Modules.LINES_LOADER,
      'positions.length',
      positions.length,
      '(should be',
      segmentCount * 2 * ndim,
      ')'
    );
    log.custom(
      LogEmoji.DEBUG,
      Modules.LINES_LOADER,
      'segments.length',
      segments.length,
      '(should be',
      segmentCount * 2,
      ')'
    );
    log.custom(LogEmoji.DEBUG, Modules.LINES_LOADER, 'ndim', ndim);
    log.custom(LogEmoji.DEBUG, Modules.LINES_LOADER, 'slicePosition', slicePosition);
    log.custom(LogEmoji.DEBUG, Modules.LINES_LOADER, 'tolerance', tolerance);
    log.custom(LogEmoji.DEBUG, Modules.LINES_LOADER, 'displayDims', displayDims);
    log.custom(
      LogEmoji.DEBUG,
      Modules.LINES_LOADER,
      'First vertex',
      Array.from(positions.slice(0, ndim))
    );
    log.custom(
      LogEmoji.DEBUG,
      Modules.LINES_LOADER,
      'First segment indices',
      segments[0],
      segments[1]
    );
  }

  // Pre-allocate output arrays (may be smaller after clipping)
  const maxSegments = segmentCount;
  const startPositions = new Float32Array(maxSegments * 3);
  const endPositions = new Float32Array(maxSegments * 3);
  const startColors = new Float32Array(maxSegments * 3);
  const endColors = new Float32Array(maxSegments * 3);
  const startWidths = new Float32Array(maxSegments);
  const endWidths = new Float32Array(maxSegments);
  const startSharpness = new Float32Array(maxSegments);
  const endSharpness = new Float32Array(maxSegments);
  const segmentLengths = new Float32Array(maxSegments);
  const startClipped = new Uint8Array(maxSegments);
  const endClipped = new Uint8Array(maxSegments);
  // per-segment scalar pairs (only allocated when source scalars
  // are present, so the no-colormap path pays no extra memory cost).
  const startScalars = scalars ? new Float32Array(maxSegments) : null;
  const endScalars = scalars ? new Float32Array(maxSegments) : null;

  // Shared with the worker projection path; Float32 inputs pass through
  // (zero alloc), Uint8/Uint16 trigger an upfront normalization. Audited
  // for sparse-visible-segment alloc cost — sub-1ms at 100k segments;
  // (Historical benchmark file removed in J.1 of viewer-code-review-rerun.)
  const colorsF32 = colors ? coerceColorsToFloat32(colors) : null;

  let outIdx = 0;
  let firstClippedReason = null;

  for (let i = 0; i < segmentCount; i++) {
    // Get vertex indices (local space)
    const v0 = segments[i * 2];
    const v1 = segments[i * 2 + 1];

    // Extract nD positions
    const p1 = Array.from(positions.slice(v0 * ndim, (v0 + 1) * ndim));
    const p2 = Array.from(positions.slice(v1 * ndim, (v1 + 1) * ndim));

    // Diagnostic: log the first few segments to spot-check input shape.
    if (
      segmentCount > 10000 &&
      i < 3 &&
      appConfig.dataLoading.performance.enablePerformanceMonitoring
    ) {
      log.custom(
        LogEmoji.DEBUG,
        Modules.LINES_LOADER,
        `Segment ${i}: v0=${v0}, v1=${v1}, p1=[${p1}], p2=[${p2}]`
      );
    }

    // Clip to slice
    const clipped = clipSegmentToSlice(p1, p2, slicePosition, tolerance, displayDims);
    if (!clipped.visible) {
      if (
        segmentCount > 10000 &&
        !firstClippedReason &&
        appConfig.dataLoading.performance.enablePerformanceMonitoring
      ) {
        firstClippedReason = `Segment ${i} clipped: p1=[${p1}], p2=[${p2}]`;
      }
      continue;
    }

    // Write 3D positions
    startPositions.set(clipped.p1, outIdx * 3);
    endPositions.set(clipped.p2, outIdx * 3);

    // Interpolate and write colors. colorsF32 is already normalized
    // to [0, 1] by coerceColorsToFloat32 above.
    const c0 = colorsF32
      ? [colorsF32[v0 * 3], colorsF32[v0 * 3 + 1], colorsF32[v0 * 3 + 2]]
      : [1, 1, 1];
    const c1 = colorsF32
      ? [colorsF32[v1 * 3], colorsF32[v1 * 3 + 1], colorsF32[v1 * 3 + 2]]
      : [1, 1, 1];
    const startC = lerpVec3(c0, c1, clipped.t1);
    const endC = lerpVec3(c0, c1, clipped.t2);
    startColors.set(startC, outIdx * 3);
    endColors.set(endC, outIdx * 3);

    // Interpolate widths
    const w0 = widths[v0];
    const w1 = widths[v1];
    startWidths[outIdx] = lerp(w0, w1, clipped.t1);
    endWidths[outIdx] = lerp(w0, w1, clipped.t2);

    // Interpolate sharpness (default 1.0 if not present)
    const s0 = sharpness ? sharpness[v0] : 1.0;
    const s1 = sharpness ? sharpness[v1] : 1.0;
    startSharpness[outIdx] = lerp(s0, s1, clipped.t1);
    endSharpness[outIdx] = lerp(s0, s1, clipped.t2);

    // interpolate per-vertex scalar at the clipped endpoints so
    // the colormap LUT lookup uses the correct value at the slice
    // boundary (matching the colors / widths / sharpness pattern).
    if (scalars && startScalars && endScalars) {
      const sc0 = scalars[v0];
      const sc1 = scalars[v1];
      startScalars[outIdx] = lerp(sc0, sc1, clipped.t1);
      endScalars[outIdx] = lerp(sc0, sc1, clipped.t2);
    }

    // Calculate 3D segment length
    segmentLengths[outIdx] = distance3D(clipped.p1, clipped.p2);

    // Track clipping for cap factor adjustment
    startClipped[outIdx] = clipped.t1 > 0 ? 1 : 0;
    endClipped[outIdx] = clipped.t2 < 1 ? 1 : 0;

    outIdx++;
  }

  // Diagnostic: log post-clipping output shape for large-segment-count updates.
  if (segmentCount > 10000 && appConfig.dataLoading.performance.enablePerformanceMonitoring) {
    log.custom(LogEmoji.DEBUG, Modules.LINES_LOADER, 'buildInstanceBuffers output');
    log.custom(
      LogEmoji.DEBUG,
      Modules.LINES_LOADER,
      'Input segments',
      segmentCount,
      '→ Output segments:',
      outIdx
    );
    log.custom(
      LogEmoji.DEBUG,
      Modules.LINES_LOADER,
      'Clipped rate',
      `${((1 - outIdx / segmentCount) * 100).toFixed(1)}%`
    );
    if (firstClippedReason) {
      log.custom(LogEmoji.DEBUG, Modules.LINES_LOADER, 'First clip reason', firstClippedReason);
    }
    if (outIdx > 0) {
      log.custom(LogEmoji.DEBUG, Modules.LINES_LOADER, 'First output segment', {
        p1: Array.from(startPositions.slice(0, 3)),
        p2: Array.from(endPositions.slice(0, 3)),
      });
    }
  }

  // Trim arrays to actual size
  return {
    startPositions: startPositions.slice(0, outIdx * 3),
    endPositions: endPositions.slice(0, outIdx * 3),
    startColors: startColors.slice(0, outIdx * 3),
    endColors: endColors.slice(0, outIdx * 3),
    startWidths: startWidths.slice(0, outIdx),
    endWidths: endWidths.slice(0, outIdx),
    startSharpness: startSharpness.slice(0, outIdx),
    endSharpness: endSharpness.slice(0, outIdx),
    segmentLengths: segmentLengths.slice(0, outIdx),
    startClipped: startClipped.slice(0, outIdx),
    endClipped: endClipped.slice(0, outIdx),
    // Include scalar pairs only when the source had scalars. Output
    // omits these fields when no scalars are present so the fail-closed
    // colormap guard passes unconditionally for non-colormap datasets.
    ...(startScalars && endScalars
      ? {
          startScalars: startScalars.slice(0, outIdx),
          endScalars: endScalars.slice(0, outIdx),
        }
      : {}),
    segmentCount: outIdx,
  };
}

/**
 * WASM-accelerated version of buildInstanceBuffers.
 *
 * Uses batch WASM functions for significantly faster nD clipping:
 * - clip_segments_batch: Process all segments at once
 * - interpolate_clipped_positions: Batch position interpolation
 * - interpolate_scalars_batch: Batch width/sharpness interpolation
 * - interpolate_colors_batch: Batch color interpolation
 * - calculate_segment_lengths: Batch length calculation
 * - mark_clipped_endpoints: Batch endpoint marking
 *
 * @param loadedData - Raw lines data from loader
 * @param slicePosition - Current position in nD space
 * @param tolerance - Per-dimension tolerance
 * @param displayDims - Which dimensions to display
 * @returns Processed data ready for GPU
 */
export function buildInstanceBuffersWASM(
  loadedData: LoadedLinesData,
  slicePosition: number[],
  tolerance: number[],
  displayDims: number[]
): ProcessedLinesData {
  const { positions, segments, widths, colors, sharpness, ndim, segmentCount } = loadedData;
  // A.3: same fail-closed scalar length validation as the synchronous
  // path. Mismatch suppresses scalar projection; WASM path otherwise
  // would read OOB inside interpolate_scalars_batch.
  const vertexCount = ndim > 0 ? Math.floor(positions.length / ndim) : 0;
  let scalars: typeof loadedData.scalars = loadedData.scalars;
  if (scalars && scalars.length !== vertexCount) {
    log.warning(
      Modules.LINES_LOADER,
      `Scalar length mismatch (WASM path): ${scalars.length} scalars for ${vertexCount} vertices. ` +
        'Suppressing scalar projection.'
    );
    scalars = null;
  }

  // Get WASM module (uses cached instance or fallback)
  const wasm = getWasmModuleSync();

  // Convert inputs to typed arrays for WASM
  const slicePosF32 = new Float32Array(slicePosition);
  const toleranceF32 = new Float32Array(tolerance);
  const displayDimsU32 = new Uint32Array(displayDims);

  // Step 1: Batch-clip all segments
  const visibility = new Uint8Array(segmentCount);
  const t1Params = new Float32Array(segmentCount);
  const t2Params = new Float32Array(segmentCount);

  const visibleCount = wasm.clip_segments_batch(
    positions,
    segments,
    slicePosF32,
    toleranceF32,
    displayDimsU32,
    ndim,
    segmentCount,
    visibility,
    t1Params,
    t2Params
  );

  // Early exit if no visible segments
  if (visibleCount === 0) {
    return {
      startPositions: new Float32Array(0),
      endPositions: new Float32Array(0),
      startColors: new Float32Array(0),
      endColors: new Float32Array(0),
      startWidths: new Float32Array(0),
      endWidths: new Float32Array(0),
      startSharpness: new Float32Array(0),
      endSharpness: new Float32Array(0),
      segmentLengths: new Float32Array(0),
      startClipped: new Uint8Array(0),
      endClipped: new Uint8Array(0),
      segmentCount: 0,
    };
  }

  // Step 2: Allocate output buffers for visible segments
  const startPositions = new Float32Array(visibleCount * 3);
  const endPositions = new Float32Array(visibleCount * 3);
  const startColors = new Float32Array(visibleCount * 3);
  const endColors = new Float32Array(visibleCount * 3);
  const startWidths = new Float32Array(visibleCount);
  const endWidths = new Float32Array(visibleCount);
  const startSharpness = new Float32Array(visibleCount);
  const endSharpness = new Float32Array(visibleCount);
  const segmentLengths = new Float32Array(visibleCount);
  const startClipped = new Uint8Array(visibleCount);
  const endClipped = new Uint8Array(visibleCount);

  // Step 3: Interpolate clipped positions to 3D
  wasm.interpolate_clipped_positions(
    positions,
    segments,
    visibility,
    t1Params,
    t2Params,
    displayDimsU32,
    ndim,
    segmentCount,
    startPositions,
    endPositions
  );

  // Interpolate colors. The shared coerceColorsToFloat32 helper
  // handles dtype + normalization in one step.
  if (colors) {
    wasm.interpolate_colors_batch(
      coerceColorsToFloat32(colors),
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startColors,
      endColors
    );
  } else {
    // Default white color
    startColors.fill(1.0);
    endColors.fill(1.0);
  }

  // Step 4: Interpolate widths
  wasm.interpolate_scalars_batch(
    widths,
    segments,
    visibility,
    t1Params,
    t2Params,
    segmentCount,
    startWidths,
    endWidths
  );

  // Step 5: Interpolate sharpness
  if (sharpness) {
    wasm.interpolate_scalars_batch(
      sharpness,
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startSharpness,
      endSharpness
    );
  } else {
    // Default sharpness of 1.0
    startSharpness.fill(1.0);
    endSharpness.fill(1.0);
  }

  // interpolate per-vertex scalars when present. Reuses the same
  // batch helper as widths/sharpness — the WASM op is type-agnostic
  // and treats each per-vertex value identically.
  //
  // A.2: source scalars may be Uint8 / Float16 / Float32. The WASM
  // signature requires Float32, so widen up front for non-Float32
  // sources. Float32 inputs pass through (zero alloc).
  let startScalars: Float32Array | null = null;
  let endScalars: Float32Array | null = null;
  if (scalars) {
    startScalars = new Float32Array(visibleCount);
    endScalars = new Float32Array(visibleCount);
    const scalarsF32 =
      scalars instanceof Float32Array
        ? scalars
        : (() => {
            const out = new Float32Array(scalars.length);
            for (let i = 0; i < scalars.length; i++) out[i] = scalars[i];
            return out;
          })();
    wasm.interpolate_scalars_batch(
      scalarsF32,
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startScalars,
      endScalars
    );
  }

  // Step 6: Calculate segment lengths
  wasm.calculate_segment_lengths(startPositions, endPositions, visibleCount, segmentLengths);

  // Step 7: Mark clipped endpoints
  wasm.mark_clipped_endpoints(
    visibility,
    t1Params,
    t2Params,
    segmentCount,
    startClipped,
    endClipped
  );

  return {
    startPositions,
    endPositions,
    startColors,
    endColors,
    startWidths,
    endWidths,
    startSharpness,
    endSharpness,
    segmentLengths,
    startClipped,
    endClipped,
    // include scalar pairs only when source had scalars.
    ...(startScalars && endScalars ? { startScalars, endScalars } : {}),
    segmentCount: visibleCount,
  };
}

// ============================================================================
// Eager WASM warmup
// ============================================================================

/**
 * Initialize WASM module for hot path optimization.
 * Call this early in application startup to ensure WASM is ready when needed.
 *
 * @internal — invoked at viewer startup; not part of the public API.
 */
export async function initLinesWASM(): Promise<void> {
  await getWasmModule();
  log.info(Modules.LINES_LOADER, 'WASM module initialized for lines clipping');
}
