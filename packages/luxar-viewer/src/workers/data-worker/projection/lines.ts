/**
 * Project Lines from nD to 3D with segment clipping.
 *
 * The single Lines projection implementation — run on a worker, or on the
 * main thread via `in-process.ts` (the former hand-written main-thread
 * copy in `data/lines/projection.ts` was deleted in W4). Uses WASM batch
 * functions:
 * 1. Clips all segments to the nD slice
 * 2. Projects clipped endpoints to 3D
 * 3. Interpolates per-vertex attributes (colors, widths, sharpness, scalars)
 * 4. Calculates segment lengths and tracks clipping
 *
 * WASM batch functions provide 3-5x speedup over per-segment TypeScript loops.
 */

import { transfer } from 'comlink';
import { pickBackend, type WasmCtx } from '../state';
import { validateProjectionInputs, validateLineSegmentReferences } from '../validation';
import { coerceColorsToFloat32, coerceScalarsToFloat32, fillColorsWhite } from '../../color-utils';
import type { ProjectionViewState } from '../types';
import type { LinesProjectionBounds } from '../../../types/lines';

/**
 * Fused output scan: AABB over start+end positions + max finite width,
 * in ONE pass over the projected arrays — computed here, where the data
 * is already hot (and, for the nD worker path, OFF the main thread), so
 * `computeLineBounds` doesn't re-run its O(N) per-segment scan on the
 * main thread per commit (see `LinesProjectionBounds`). Float semantics
 * match `computeLineBounds`' fallback scan exactly: `Math.min`/`Math.max`
 * per component (Box3.expandByPoint) and the `Number.isFinite` width
 * guard.
 *
 * Exported for the fused-vs-brute-force unit tests.
 *
 * @param count - Visible segment count (must be > 0; callers skip the
 *   scan and omit `bounds` for empty results)
 */
export function computeLinesProjectionBounds(
  startPositions: Float32Array,
  endPositions: Float32Array,
  startWidths: Float32Array,
  endWidths: Float32Array,
  count: number
): LinesProjectionBounds {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let maxWidth = 0;
  for (let i = 0; i < count; i++) {
    const si = i * 3;
    const sx = startPositions[si];
    const sy = startPositions[si + 1];
    const sz = startPositions[si + 2];
    const ex = endPositions[si];
    const ey = endPositions[si + 1];
    const ez = endPositions[si + 2];
    minX = Math.min(minX, sx, ex);
    minY = Math.min(minY, sy, ey);
    minZ = Math.min(minZ, sz, ez);
    maxX = Math.max(maxX, sx, ex);
    maxY = Math.max(maxY, sy, ey);
    maxZ = Math.max(maxZ, sz, ez);

    const sw = startWidths[i];
    const ew = endWidths[i];
    if (Number.isFinite(sw) && sw > maxWidth) maxWidth = sw;
    if (Number.isFinite(ew) && ew > maxWidth) maxWidth = ew;
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ], maxWidth };
}

export async function projectLinesTo3D(
  ctx: WasmCtx,
  params: {
    positions: Float32Array;
    segments: Uint32Array;
    widths: Float32Array;
    colors: Float32Array | Uint8Array | Uint16Array | null;
    sharpness: Float32Array | null;
    /**
     * Optional per-vertex scalar attribute for colormap-mode Lines.
     * Accepted dtypes: `Float32Array` (passes through zero-copy),
     * `Float16Array` (element-wise expand), `Uint8Array` (normalized by
     * `1/255` to align with the colormap shader's `[0, 1]` contract).
     * When `null` the worker emits empty `startScalars`/`endScalars`
     * arrays and the loader leaves the geometry's scalar attribute
     * unallocated.
     */
    scalars: Float32Array | Float16Array | Uint8Array | null;
    /**
     * View state for projection. Same `ProjectionViewState` shape every
     * `project*To3D` worker function accepts — keeps the worker API
     * uniform across node types. All three fields are consumed: the
     * WASM `clip_segments_batch` step reads `slicePosition` + `tolerance`
     * + `displayDims` to decide per-segment visibility and to clip
     * partially-visible segments at the slice boundary.
     */
    viewState: ProjectionViewState;
    ndim: number;
    segmentCount: number;
    /**
     * Channels per color entry: 3 (RGB, default) or 4 (RGBA — the alpha
     * column is per-vertex opacity, volumetric phase 4). When 4, the
     * worker de-interleaves the coerced colors into an RGB stride-3
     * array for `interpolate_colors_batch` (an RGB-only kernel) plus a
     * stride-1 alpha column routed through `interpolate_scalars_batch`
     * (the widths/sharpness/scalars kernel — alpha interpolates
     * linearly like any scalar), emitting `startAlphas`/`endAlphas`.
     */
    colorComponents?: 3 | 4;
  }
): Promise<{
  startPositions: Float32Array;
  endPositions: Float32Array;
  startColors: Float32Array;
  endColors: Float32Array;
  startWidths: Float32Array;
  endWidths: Float32Array;
  startSharpness: Float32Array;
  endSharpness: Float32Array;
  /** Per-segment start scalar (empty Float32Array when input scalars=null). */
  startScalars: Float32Array;
  /** Per-segment end scalar (empty Float32Array when input scalars=null). */
  endScalars: Float32Array;
  /** Per-segment start alpha (empty Float32Array unless colorComponents=4). */
  startAlphas: Float32Array;
  /** Per-segment end alpha (empty Float32Array unless colorComponents=4). */
  endAlphas: Float32Array;
  segmentLengths: Float32Array;
  startJointCode: Float32Array;
  endJointCode: Float32Array;
  visibleSegmentCount: number;
  /**
   * Fused-scan cull metadata (AABB over start+end positions + max
   * finite width) — present whenever `visibleSegmentCount > 0`. Plain
   * scalars, structured-clone safe.
   */
  bounds?: LinesProjectionBounds;
}> {
  const wasmModule = pickBackend(ctx, params.ndim); // >16D -> uncapped TS reference

  const { positions, segments, widths, colors, sharpness, scalars, viewState, ndim, segmentCount } =
    params;
  const colorK = colors ? (params.colorComponents ?? 3) : 3;
  const { displayDims, slicePosition, tolerance } = viewState;

  // The shared projection validator handles displayDims and the basic
  // ndim/slicePosition sanity checks. numItems is 0 because the real
  // positions invariant for lines — "covers the max vertex referenced by
  // segments" — is enforced by validateLineSegmentReferences below, and
  // the canonical empty payload (segmentCount 0, zero-length positions)
  // is a legitimate input that must project to an empty result so the
  // commit step can CLEAR stale geometry (a hardcoded numItems=1 here
  // rejected it, leaving out-of-slice Lines rendered forever on
  // non-displayed-dimension scrubs).
  validateProjectionInputs(
    'projectLinesTo3D',
    positions,
    displayDims,
    slicePosition,
    ndim,
    0,
    ndim
  );
  if (tolerance.length < ndim) {
    throw new Error(
      `projectLinesTo3D: tolerance too short (got ${tolerance.length}, expected ≥ ${ndim})`
    );
  }
  validateLineSegmentReferences('projectLinesTo3D', segments, segmentCount, positions, ndim, {
    widths,
    colors: colors ?? undefined,
    colorComponents: colorK,
    sharpness: sharpness ?? undefined,
    scalars: scalars ?? undefined,
  });

  // Convert input arrays to WASM-compatible formats
  const slicePosF32 = new Float32Array(slicePosition);
  const toleranceF32 = new Float32Array(tolerance);
  const displayDimsU32 = new Uint32Array(displayDims);

  // Step 1: Batch clip all segments using WASM
  const visibility = new Uint8Array(segmentCount);
  const t1Params = new Float32Array(segmentCount);
  const t2Params = new Float32Array(segmentCount);

  const visibleCount = wasmModule.clip_segments_batch(
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
    const emptyPositions = new Float32Array(0);
    const emptyColors = new Float32Array(0);
    const emptyScalars = new Float32Array(0);
    const emptyFlags = new Float32Array(0);

    return transfer(
      {
        startPositions: emptyPositions,
        endPositions: new Float32Array(0),
        startColors: emptyColors,
        endColors: new Float32Array(0),
        startWidths: emptyScalars,
        endWidths: new Float32Array(0),
        startSharpness: new Float32Array(0),
        endSharpness: new Float32Array(0),
        startScalars: new Float32Array(0),
        endScalars: new Float32Array(0),
        startAlphas: new Float32Array(0),
        endAlphas: new Float32Array(0),
        segmentLengths: new Float32Array(0),
        startJointCode: emptyFlags,
        endJointCode: new Float32Array(0),
        visibleSegmentCount: 0,
      },
      [emptyPositions.buffer, emptyColors.buffer, emptyScalars.buffer, emptyFlags.buffer]
    );
  }

  // Step 2: Interpolate clipped positions to 3D using WASM
  const startPositions = new Float32Array(visibleCount * 3);
  const endPositions = new Float32Array(visibleCount * 3);

  wasmModule.interpolate_clipped_positions(
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

  // Step 3: Interpolate colors using WASM
  // Convert colors to Float32Array if needed (WASM expects Float32Array).
  // `interpolate_colors_batch` is an RGB stride-3 kernel; an RGBA input
  // (colorK === 4) is de-interleaved once into an RGB array + a stride-1
  // alpha column, and the alpha rides the SAME scalar kernel widths /
  // sharpness / scalars already use (alpha interpolates linearly like
  // any scalar; both WASM and the >16D TS backend have it, so no kernel
  // change and no parity surface is added).
  const startColors = new Float32Array(visibleCount * 3);
  const endColors = new Float32Array(visibleCount * 3);
  let startAlphas: Float32Array;
  let endAlphas: Float32Array;

  if (colors) {
    let colorsRGB = coerceColorsToFloat32(colors);
    let alphaColumn: Float32Array | null = null;
    if (colorK === 4) {
      const numVertices = colorsRGB.length / 4;
      const rgb = new Float32Array(numVertices * 3);
      alphaColumn = new Float32Array(numVertices);
      for (let v = 0; v < numVertices; v++) {
        rgb[v * 3] = colorsRGB[v * 4];
        rgb[v * 3 + 1] = colorsRGB[v * 4 + 1];
        rgb[v * 3 + 2] = colorsRGB[v * 4 + 2];
        alphaColumn[v] = colorsRGB[v * 4 + 3];
      }
      colorsRGB = rgb;
    }
    wasmModule.interpolate_colors_batch(
      colorsRGB,
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startColors,
      endColors
    );
    if (alphaColumn) {
      startAlphas = new Float32Array(visibleCount);
      endAlphas = new Float32Array(visibleCount);
      wasmModule.interpolate_scalars_batch(
        alphaColumn,
        segments,
        visibility,
        t1Params,
        t2Params,
        segmentCount,
        startAlphas,
        endAlphas
      );
    } else {
      startAlphas = new Float32Array(0);
      endAlphas = new Float32Array(0);
    }
  } else {
    fillColorsWhite(startColors, visibleCount);
    fillColorsWhite(endColors, visibleCount);
    startAlphas = new Float32Array(0);
    endAlphas = new Float32Array(0);
  }

  // Step 4: Interpolate widths using WASM
  const startWidths = new Float32Array(visibleCount);
  const endWidths = new Float32Array(visibleCount);

  wasmModule.interpolate_scalars_batch(
    widths,
    segments,
    visibility,
    t1Params,
    t2Params,
    segmentCount,
    startWidths,
    endWidths
  );

  // Step 5: Interpolate sharpness using WASM
  const startSharpness = new Float32Array(visibleCount);
  const endSharpness = new Float32Array(visibleCount);

  if (sharpness) {
    wasmModule.interpolate_scalars_batch(
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
    // Default sharpness knob is 0.5 -> beta=2 (Gaussian)
    startSharpness.fill(0.5);
    endSharpness.fill(0.5);
  }

  // Step 6: Interpolate per-vertex scalars using WASM (colormap mode).
  // Empty arrays are returned when `scalars` is null so the loader can
  // skip allocating the geometry's scalar attribute. We coerce
  // Float16/Uint8 to Float32 first since `interpolate_scalars_batch`
  // expects Float32 inputs (same path widths/sharpness take).
  let startScalars: Float32Array;
  let endScalars: Float32Array;
  if (scalars) {
    const scalarsF32 = coerceScalarsToFloat32(scalars);
    startScalars = new Float32Array(visibleCount);
    endScalars = new Float32Array(visibleCount);
    wasmModule.interpolate_scalars_batch(
      scalarsF32,
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startScalars,
      endScalars
    );
  } else {
    startScalars = new Float32Array(0);
    endScalars = new Float32Array(0);
  }

  // Step 7: Calculate segment lengths using WASM
  const segmentLengths = new Float32Array(visibleCount);
  wasmModule.calculate_segment_lengths(startPositions, endPositions, visibleCount, segmentLengths);

  // Step 8: Per-endpoint joint codes using WASM. Purely topological — it reads
  // connectivity and the clip parameters, never positions, because the bend
  // angle is now measured in SCREEN space by the vertex stage (which is what
  // lets it track the camera; the stored data-space angle could not, #795).
  // The visible-stream index it emits is a line-texture storage slot, so it
  // must be computed over the same visible ordering the texel writer consumes.
  const startJointCode = new Float32Array(visibleCount);
  const endJointCode = new Float32Array(visibleCount);
  const vertexCount = Math.floor(positions.length / ndim);

  wasmModule.compute_joint_codes(
    segments,
    visibility,
    t1Params,
    t2Params,
    segmentCount,
    vertexCount,
    startJointCode,
    endJointCode
  );

  // Build transferable list
  const transferables: ArrayBuffer[] = [
    startPositions.buffer as ArrayBuffer,
    endPositions.buffer as ArrayBuffer,
    startColors.buffer as ArrayBuffer,
    endColors.buffer as ArrayBuffer,
    startWidths.buffer as ArrayBuffer,
    endWidths.buffer as ArrayBuffer,
    startSharpness.buffer as ArrayBuffer,
    endSharpness.buffer as ArrayBuffer,
    startScalars.buffer as ArrayBuffer,
    endScalars.buffer as ArrayBuffer,
    startAlphas.buffer as ArrayBuffer,
    endAlphas.buffer as ArrayBuffer,
    segmentLengths.buffer as ArrayBuffer,
    startJointCode.buffer as ArrayBuffer,
    endJointCode.buffer as ArrayBuffer,
  ];

  return transfer(
    {
      startPositions,
      endPositions,
      startColors,
      endColors,
      startWidths,
      endWidths,
      startSharpness,
      endSharpness,
      startScalars,
      endScalars,
      startAlphas,
      endAlphas,
      segmentLengths,
      startJointCode,
      endJointCode,
      visibleSegmentCount: visibleCount,
      bounds: computeLinesProjectionBounds(
        startPositions,
        endPositions,
        startWidths,
        endWidths,
        visibleCount
      ),
    },
    transferables
  );
}
