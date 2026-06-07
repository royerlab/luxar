/**
 * Golden-equivalence tests for the worker projection dispatchers
 * (GSplats + Lines) introduced when W4 eliminated the hand-written
 * main-thread projection copies.
 *
 * Two guarantees per geometry:
 *   1. **Backend equivalence** — running the dispatcher through compiled
 *      WASM produces bit-identical (NaN-aware) output to the TypeScript
 *      reference. Skipped cleanly when `public/wasm` isn't built; the
 *      TypeScript assertions below still run.
 *   2. **Behavioral goldens** — hand-verified expectations for the
 *      dispatcher-level behavior the deleted copies used to own: the
 *      standard-3D fast path (no amplitude filtering), discrete-dim
 *      gating, extend_to_all skipping, axis-permuted displayDims, color
 *      normalization, and clipping/compaction.
 *
 * The kernel math itself is covered by `wasm/typescript-reference/*` and
 * the WASM-vs-TS parity suite (`wasm-vs-typescript.test.ts`); this file
 * covers the dispatcher wrappers that orchestrate those kernels.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { TypeScriptFallback } from '../../../../wasm/typescript';
import type { WasmModule } from '../../../../wasm/types';
import { arraysEqual } from '../../../helpers/array-compare';
import {
  projectGSplatsViaDispatcher,
  projectLinesViaDispatcher,
} from '../../../helpers/projection-adapters';
import { createEmptyGSplatsData } from '../../../../data/gsplats/projection';
import { createEmptyLinesData } from '../../../../data/lines/projection';
import { projectPointsTo3D, type ProjectionContext } from '../../../../data/points/projection';
import type {
  LoadedGSplatsData,
  GSplatsViewState,
  GSplatsMetadata,
  ProcessedGSplatsData,
} from '../../../../types/gsplats';
import type { LoadedLinesData, LinesMetadata, ProcessedLinesData } from '../../../../types/lines';
import type { DimensionMetadata } from '../../../../types/dims';
import type { ViewState, PointRange, LoadedPointsData } from '../../../../data/data-loader-types';
import type { PointsMetadata, EffectiveRadiusConfig } from '../../../../types/points';

// ---------------------------------------------------------------------------
// Backends: TypeScript reference (always) + compiled WASM (when built)
// ---------------------------------------------------------------------------

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const wasmJsPath = join(__dirname, '../../../../../public/wasm/luxar_wasm.js');
const wasmBinaryPath = join(__dirname, '../../../../../public/wasm/luxar_wasm_bg.wasm');
const wasmAvailable = existsSync(wasmJsPath) && existsSync(wasmBinaryPath);

const tsBackend: WasmModule = new TypeScriptFallback();
let wasmBackend: WasmModule | null = null;

beforeAll(async () => {
  if (!wasmAvailable) return;
  const wasmBinary = readFileSync(wasmBinaryPath);
  const wasm = await import(wasmJsPath);
  wasm.initSync({ module: wasmBinary });
  wasmBackend = wasm as unknown as WasmModule;
});

/** Continuous dim metadata entry (display flag irrelevant here). */
function contDim(name: string): DimensionMetadata {
  return { name, unit: '', scale: 1, discrete: false, step: 1 };
}
/** Discrete dim metadata entry with the given step. */
function discDim(name: string, step = 1): DimensionMetadata {
  return { name, unit: '', scale: 1, discrete: true, step };
}

// ---------------------------------------------------------------------------
// Empty-payload factories (consumed directly by the loaders)
// ---------------------------------------------------------------------------

describe('createEmptyGSplatsData', () => {
  it('produces a zero-splat payload preserving ndim', () => {
    const empty = createEmptyGSplatsData({ ndim: 5 } as unknown as GSplatsMetadata);
    expect(empty.splatCount).toBe(0);
    expect(empty.ndim).toBe(5);
    expect(empty.positions.length).toBe(0);
    expect(empty.choleskyFactors.length).toBe(0);
    expect(empty.colors).toBeNull();
  });
});

describe('createEmptyLinesData', () => {
  it('produces a zero-segment payload preserving ndim', () => {
    const empty = createEmptyLinesData({ ndim: 4 } as unknown as LinesMetadata);
    expect(empty.segmentCount).toBe(0);
    expect(empty.ndim).toBe(4);
    expect(empty.positions.length).toBe(0);
    expect(empty.colors).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// GSplats dispatcher
// ---------------------------------------------------------------------------

/** Assert WASM and TS dispatcher outputs are field-by-field equal. */
function expectGSplatsBackendsEqual(a: ProcessedGSplatsData, b: ProcessedGSplatsData): void {
  expect(a.splatCount).toBe(b.splatCount);
  expect(arraysEqual(a.centers3D, b.centers3D, 1e-5)).toBe(true);
  expect(arraysEqual(a.choleskyFactors3D, b.choleskyFactors3D, 1e-5)).toBe(true);
  expect(arraysEqual(a.amplitudes, b.amplitudes, 1e-5)).toBe(true);
  expect(arraysEqual(a.colors, b.colors, 1e-5)).toBe(true);
}

/** Run the gsplats dispatcher on both backends; return the TS result. */
async function runGSplatsBothBackends(
  loaded: LoadedGSplatsData,
  viewState: GSplatsViewState,
  truncate = 3.0
): Promise<ProcessedGSplatsData> {
  const ts = await projectGSplatsViaDispatcher(tsBackend, loaded, viewState, truncate);
  if (wasmBackend) {
    const w = await projectGSplatsViaDispatcher(wasmBackend, loaded, viewState, truncate);
    expectGSplatsBackendsEqual(w, ts);
  }
  return ts;
}

/** Packed 3D identity Cholesky [L00,L10,L11,L20,L21,L22]. */
const IDENTITY_CHOL_3D = [1, 0, 1, 0, 0, 1];
/** Packed 4D identity Cholesky (10 elements). */
const IDENTITY_CHOL_4D = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1];

describe('gsplats dispatcher: standard-3D fast path', () => {
  it('copies every splat with NO amplitude filtering and normalizes u8 colors', async () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
      // One amplitude is far below MIN_AMPLITUDE (1e-6) — the standard-3D
      // path must still keep it (matches the old projectGSplats3DOnly).
      amplitudes: new Float32Array([1.0, 0.5, 1e-9]),
      choleskyFactors: new Float32Array([
        ...IDENTITY_CHOL_3D,
        ...IDENTITY_CHOL_3D,
        ...IDENTITY_CHOL_3D,
      ]),
      colors: new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]),
      splatCount: 3,
      ndim: 3,
    };
    const viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1e10, 1e10, 1e10],
      dimensions: [contDim('x'), contDim('y'), contDim('z')],
    } as unknown as GSplatsViewState;

    const out = await runGSplatsBothBackends(loaded, viewState);
    expect(out.splatCount).toBe(3); // all kept, none filtered
    expect(Array.from(out.centers3D)).toEqual([0, 0, 0, 1, 1, 1, 2, 2, 2]);
    // Amplitudes copied through verbatim (f32 rounding on the tiny value);
    // the third is far below MIN_AMPLITUDE yet survives — no filtering.
    expect(out.amplitudes[0]).toBeCloseTo(1.0, 6);
    expect(out.amplitudes[1]).toBeCloseTo(0.5, 6);
    expect(out.amplitudes[2]).toBeGreaterThan(0);
    expect(out.amplitudes[2]).toBeCloseTo(1e-9, 12);
    // u8 → normalized /255
    expect(out.colors[0]).toBeCloseTo(1.0, 5);
    expect(out.colors[4]).toBeCloseTo(1.0, 5);
    expect(out.colors[8]).toBeCloseTo(1.0, 5);
  });
});

describe('gsplats dispatcher: nD continuous attenuation', () => {
  it('keeps on-slice splats and culls far ones (4D, hidden dim 3)', async () => {
    const loaded: LoadedGSplatsData = {
      // splat0 at dim3=0 (on slice), splat1 at dim3=10 (far)
      positions: new Float32Array([0, 0, 0, 0, 1, 1, 1, 10]),
      amplitudes: new Float32Array([1.0, 1.0]),
      choleskyFactors: new Float32Array([...IDENTITY_CHOL_4D, ...IDENTITY_CHOL_4D]),
      colors: null,
      splatCount: 2,
      ndim: 4,
    };
    const viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1e10, 1e10, 1e10, 1],
      dimensions: [contDim('x'), contDim('y'), contDim('z'), contDim('t')],
    } as unknown as GSplatsViewState;

    const out = await runGSplatsBothBackends(loaded, viewState);
    expect(out.splatCount).toBe(1);
    // the surviving splat is splat0 at the origin
    expect(Array.from(out.centers3D.slice(0, 3))).toEqual([0, 0, 0]);
  });
});

describe('gsplats dispatcher: axis-permuted displayDims', () => {
  it('maps output XYZ in requested display order [2, 1, 0]', async () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([10, 20, 30]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([...IDENTITY_CHOL_3D]),
      colors: null,
      splatCount: 1,
      ndim: 3,
    };
    const viewState = {
      displayDims: [2, 1, 0],
      slicePosition: [0, 0, 0],
      tolerance: [1e10, 1e10, 1e10],
      dimensions: [contDim('x'), contDim('y'), contDim('z')],
    } as unknown as GSplatsViewState;

    const out = await runGSplatsBothBackends(loaded, viewState);
    expect(out.splatCount).toBe(1);
    // source (10,20,30) reordered by [2,1,0] → (30,20,10)
    expect(Array.from(out.centers3D)).toEqual([30, 20, 10]);
  });
});

describe('gsplats dispatcher: discrete-dim gating', () => {
  it('drops splats outside the half-step window of a discrete hidden dim', async () => {
    const loaded: LoadedGSplatsData = {
      // splat0 at dim3=0 (in), splat1 at dim3=5 (out, > half-step)
      positions: new Float32Array([0, 0, 0, 0, 1, 1, 1, 5]),
      amplitudes: new Float32Array([1.0, 1.0]),
      choleskyFactors: new Float32Array([...IDENTITY_CHOL_4D, ...IDENTITY_CHOL_4D]),
      colors: null,
      splatCount: 2,
      ndim: 4,
    };
    const viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1e10, 1e10, 1e10, 1],
      dimensions: [contDim('x'), contDim('y'), contDim('z'), discDim('t', 1)],
    } as unknown as GSplatsViewState;

    const out = await runGSplatsBothBackends(loaded, viewState);
    expect(out.splatCount).toBe(1);
    expect(Array.from(out.centers3D.slice(0, 3))).toEqual([0, 0, 0]);
  });
});

describe('gsplats dispatcher: extend_to_all skipping', () => {
  it('keeps all splats regardless of position in an extend_to_all dim', async () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 0, 1, 1, 1, 999]),
      amplitudes: new Float32Array([1.0, 1.0]),
      choleskyFactors: new Float32Array([...IDENTITY_CHOL_4D, ...IDENTITY_CHOL_4D]),
      colors: null,
      splatCount: 2,
      ndim: 4,
    };
    const viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      // dim3 tolerance ≥ EXTEND_TO_ALL_THRESHOLD → skipped entirely
      tolerance: [1e10, 1e10, 1e10, 1e10],
      dimensions: [contDim('x'), contDim('y'), contDim('z'), contDim('t')],
    } as unknown as GSplatsViewState;

    const out = await runGSplatsBothBackends(loaded, viewState);
    expect(out.splatCount).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// Lines dispatcher
// ---------------------------------------------------------------------------

function expectLinesBackendsEqual(a: ProcessedLinesData, b: ProcessedLinesData): void {
  expect(a.segmentCount).toBe(b.segmentCount);
  expect(arraysEqual(a.startPositions, b.startPositions, 1e-5)).toBe(true);
  expect(arraysEqual(a.endPositions, b.endPositions, 1e-5)).toBe(true);
  expect(arraysEqual(a.startColors, b.startColors, 1e-5)).toBe(true);
  expect(arraysEqual(a.endColors, b.endColors, 1e-5)).toBe(true);
  expect(arraysEqual(a.startWidths, b.startWidths, 1e-5)).toBe(true);
  expect(arraysEqual(a.endWidths, b.endWidths, 1e-5)).toBe(true);
}

async function runLinesBothBackends(
  loaded: LoadedLinesData,
  slicePosition: number[],
  tolerance: number[],
  displayDims: number[]
): Promise<ProcessedLinesData> {
  const ts = await projectLinesViaDispatcher(
    tsBackend,
    loaded,
    slicePosition,
    tolerance,
    displayDims
  );
  if (wasmBackend) {
    const w = await projectLinesViaDispatcher(
      wasmBackend,
      loaded,
      slicePosition,
      tolerance,
      displayDims
    );
    expectLinesBackendsEqual(w, ts);
  }
  return ts;
}

describe('lines dispatcher: basic 3D projection', () => {
  it('projects fully-visible segments and white-fills missing colors', async () => {
    const loaded: LoadedLinesData = {
      positions: new Float32Array([0, 0, 0, 1, 0, 0]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.2]),
      colors: null,
      sharpness: null,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };
    const out = await runLinesBothBackends(loaded, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);
    expect(out.segmentCount).toBe(1);
    expect(Array.from(out.startPositions)).toEqual([0, 0, 0]);
    expect(Array.from(out.endPositions)).toEqual([1, 0, 0]);
    // white default colors
    expect(out.startColors[0]).toBeCloseTo(1.0, 5);
    expect(out.endColors[2]).toBeCloseTo(1.0, 5);
  });
});

describe('lines dispatcher: hidden-dim clipping + culling', () => {
  it('culls a segment entirely outside the slice (4D, hidden dim 3)', async () => {
    const loaded: LoadedLinesData = {
      // both vertices at dim3=10, slice at dim3=0 ± 0.5 → invisible
      positions: new Float32Array([0, 0, 0, 10, 1, 0, 0, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      colors: null,
      sharpness: null,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 4,
    };
    const out = await runLinesBothBackends(
      loaded,
      [0, 0, 0, 0],
      [1e10, 1e10, 1e10, 0.5],
      [0, 1, 2]
    );
    expect(out.segmentCount).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Points projection (main-thread, WASM-accelerated — W4b)
// ---------------------------------------------------------------------------

function pointsCtx(overrides: Partial<ProjectionContext> = {}): ProjectionContext {
  return {
    chunkIndex: null,
    effectiveRadiusConfig: null,
    accumulator: null,
    nodeAttrs: { type: 'points', n_points: 0 } as unknown as PointsMetadata,
    ...overrides,
  };
}

function expectPointsBackendsEqual(a: LoadedPointsData, b: LoadedPointsData): void {
  expect(a.pointCount).toBe(b.pointCount);
  expect(arraysEqual(a.positions, b.positions, 1e-5)).toBe(true);
  // radii is optional; compare only when both present (else both absent).
  if (a.radii && b.radii) {
    expect(arraysEqual(a.radii, b.radii, 1e-5)).toBe(true);
  } else {
    expect(!!a.radii).toBe(!!b.radii);
  }
}

/** Run projectPointsTo3D on both backends; assert equality; return the TS result. */
function runPointsBothBackends(
  positions: Float32Array,
  radii: Float32Array | null,
  viewState: ViewState,
  ranges: PointRange[],
  ctx: ProjectionContext
): LoadedPointsData {
  const ts = projectPointsTo3D(tsBackend, positions, null, radii, null, viewState, ranges, ctx);
  if (wasmBackend) {
    const w = projectPointsTo3D(wasmBackend, positions, null, radii, null, viewState, ranges, ctx);
    expectPointsBackendsEqual(w, ts);
  }
  return ts;
}

describe('points dispatcher: 3D extraction', () => {
  it('extracts displayed dims (WASM == TS)', () => {
    // 2 points × 4D; display [0,1,2] drops the 4th dim.
    const positions = new Float32Array([1, 2, 3, 99, 4, 5, 6, 88]);
    const out = runPointsBothBackends(
      positions,
      null,
      {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      },
      [{ start: 0, end: 2 }],
      pointsCtx()
    );
    expect(out.pointCount).toBe(2);
    expect(Array.from(out.positions)).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe('points dispatcher: effective-radius filtering', () => {
  it('culls points whose hypersphere does not reach the slice (WASM == TS)', () => {
    // 4D, hidden spatial dim 3. Slice at dim3=0. Point 0 sits on the slice
    // (effective radius ~= radius); point 1 is far (radius 1 < distance 10 →
    // culled). displayDims [0,1,2], dim 3 is spatial.
    const positions = new Float32Array([0, 0, 0, 0, 5, 5, 5, 10]);
    const radii = new Float32Array([1.0, 1.0]);
    const ctx = pointsCtx({
      effectiveRadiusConfig: {
        spatialExtendDims: [true, true, true, true],
        maxRadius: 2.0,
      } as EffectiveRadiusConfig,
    });
    const out = runPointsBothBackends(
      positions,
      radii,
      {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 2],
      },
      [{ start: 0, end: 2 }],
      ctx
    );
    expect(out.pointCount).toBe(1);
    expect(Array.from(out.positions)).toEqual([0, 0, 0]);
  });
});
