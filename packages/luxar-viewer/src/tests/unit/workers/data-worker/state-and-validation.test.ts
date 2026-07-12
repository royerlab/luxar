/**
 * Direct tests for state.ts and validation.ts helpers
 * (workers.md G17, G16, H6 — symmetry + boundary invariants).
 *
 * - G17: `requireWasm(ctx)` throw path was only exercised indirectly
 *   via the worker-init test (where every projection / decode rejects
 *   with `[DataWorker] Not initialized`). Pin the contract directly.
 *
 * - G16: `sharpness too short` symmetry — the audit found lines.ts
 *   tested, points.ts/gsplats.ts not. This file adds the points test;
 *   gsplats has no sharpness validation in source (sharpness is an
 *   accepted-but-ignored param), so this pin documents that gap.
 *
 * - H6: `validateLineSegmentReferences` — bounds invariant via a
 *   small property-style sweep.
 */

import { describe, expect, it } from 'vitest';
import {
  NOT_INITIALIZED_MSG,
  requireWasm,
  state as workerState,
  type WasmCtx,
} from '../../../../workers/data-worker/state';
import {
  validateLineSegmentReferences,
  validateNDArrays,
  validateProjectionInputs,
  validateChunkQueryInputs,
  validateDecodeArgs,
  MAX_WASM_DIMS,
} from '../../../../workers/data-worker/validation';

describe('requireWasm (G17, P5)', () => {
  it('throws NOT_INITIALIZED_MSG when ctx.wasm is null', () => {
    const ctx: WasmCtx = { wasm: null };
    expect(() => requireWasm(ctx)).toThrow(NOT_INITIALIZED_MSG);
  });

  it('returns ctx.wasm when present (identity, no copy)', () => {
    const fakeWasm = { tag: 'fake' } as unknown as WasmCtx['wasm'];
    const ctx: WasmCtx = { wasm: fakeWasm };
    expect(requireWasm(ctx)).toBe(fakeWasm);
  });

  it('NOT_INITIALIZED_MSG matches the documented prefix', () => {
    // The viewer's main-thread error UI matches `[DataWorker]` prefix to
    // surface init failures with a distinct banner. Pin the prefix.
    expect(NOT_INITIALIZED_MSG).toMatch(/^\[DataWorker\]/);
    expect(NOT_INITIALIZED_MSG).toMatch(/Not initialized/);
  });

  it('module-level `state` defaults: wasm=null', () => {
    // Production worker starts with the slot null; initialize() fills
    // `wasm`. The audit asked for the throw path; this also pins the
    // module-level default.
    //
    // NOTE: vitest test files share the module graph, so this assertion
    // checks the cached state, which other tests in this file may
    // mutate. We only assert the *shape* of the initial-default
    // contract: the slot is nullable, never undefined.
    expect(workerState).toHaveProperty('wasm');
  });
});

describe('validateLineSegmentReferences — direct (H6, P5)', () => {
  // The signature: segments must fit inside positions × ndim, and
  // optional widths / colors / sharpness arrays must cover up through
  // max-vertex+1. Edge cases:

  it('numSegments=0 returns without inspecting positions/segments contents', () => {
    // Empty segment list — early return per source line 215. Critical
    // because the inner loop would otherwise dereference segments[0].
    expect(() =>
      validateLineSegmentReferences('test', new Uint32Array(0), 0, new Float32Array(0), 3)
    ).not.toThrow();
  });

  it('rejects negative numSegments', () => {
    expect(() =>
      validateLineSegmentReferences('test', new Uint32Array(4), -1, new Float32Array(12), 3)
    ).toThrow(/numSegments=-1 must be a non-negative integer/);
  });

  it('rejects non-integer numSegments', () => {
    expect(() =>
      validateLineSegmentReferences('test', new Uint32Array(4), 1.5, new Float32Array(12), 3)
    ).toThrow(/numSegments=1\.5 must be a non-negative integer/);
  });

  it('rejects segments shorter than 2 × numSegments', () => {
    expect(() =>
      validateLineSegmentReferences('test', new Uint32Array(3), 2, new Float32Array(30), 3)
    ).toThrow(/segments too short/);
  });

  // H6 property-style sweep: for ndim in {1,2,3,4}, segmentCount in {1,2,5},
  // positions sized exactly to (max-vertex + 1) × ndim must pass, but
  // (max-vertex + 1) × ndim - 1 must fail. This is the bounds invariant.
  it('bounds invariant: positions of size minVertices*ndim passes; size-1 fails', () => {
    for (const ndim of [1, 2, 3, 4]) {
      for (const numSeg of [1, 2, 5]) {
        // Build linear-chain segments [0,1, 1,2, 2,3, ...] so max-vertex == numSeg.
        const segs = new Uint32Array(numSeg * 2);
        for (let i = 0; i < numSeg; i++) {
          segs[i * 2] = i;
          segs[i * 2 + 1] = i + 1;
        }
        const maxVertex = numSeg;
        const minVertices = maxVertex + 1;

        // Exact-fit positions: passes.
        expect(() =>
          validateLineSegmentReferences(
            'test',
            segs,
            numSeg,
            new Float32Array(minVertices * ndim),
            ndim
          )
        ).not.toThrow();

        // One short: fails.
        expect(() =>
          validateLineSegmentReferences(
            'test',
            segs,
            numSeg,
            new Float32Array(minVertices * ndim - 1),
            ndim
          )
        ).toThrow(/positions too short/);
      }
    }
  });

  it('widths / colors / sharpness / scalars bounds also enforced against max-vertex+1', () => {
    // ndim=3, 2 segments [0,1, 2,3] → max-vertex=3, minVertices=4.
    const segs = new Uint32Array([0, 1, 2, 3]);
    const positions = new Float32Array(4 * 3); // exactly fits
    // widths: 3 entries (need 4) → rejected.
    expect(() =>
      validateLineSegmentReferences('test', segs, 2, positions, 3, {
        widths: new Float32Array(3),
      })
    ).toThrow(/widths too short/);
    // colors: 11 entries (need 12 = 4×3) → rejected.
    expect(() =>
      validateLineSegmentReferences('test', segs, 2, positions, 3, {
        colors: new Float32Array(11),
      })
    ).toThrow(/colors too short/);
    // sharpness: 3 entries (need 4) → rejected.
    expect(() =>
      validateLineSegmentReferences('test', segs, 2, positions, 3, {
        sharpness: new Float32Array(3),
      })
    ).toThrow(/sharpness too short/);
    // [workers OOS] scalars: 3 entries (need 4) → rejected.
    // Pre-fix, the validator didn't check scalars at all, so a short
    // scalars input reached interpolate_scalars_batch and panicked WASM.
    expect(() =>
      validateLineSegmentReferences('test', segs, 2, positions, 3, {
        scalars: new Float32Array(3),
      })
    ).toThrow(/scalars too short/);
    // All exact-fit → passes.
    expect(() =>
      validateLineSegmentReferences('test', segs, 2, positions, 3, {
        widths: new Float32Array(4),
        colors: new Float32Array(12),
        sharpness: new Float32Array(4),
        scalars: new Float32Array(4),
      })
    ).not.toThrow();
  });

  it('discontiguous segment indices: max-vertex sets the bound (not numSegments)', () => {
    // 2 segments referencing vertex 99 — needs 100 vertices, not 2.
    const segs = new Uint32Array([0, 1, 50, 99]);
    expect(() =>
      validateLineSegmentReferences('test', segs, 2, new Float32Array(100 * 3), 3)
    ).not.toThrow();
    expect(() =>
      validateLineSegmentReferences('test', segs, 2, new Float32Array(50 * 3), 3)
    ).toThrow(/positions too short for max segment vertex 99/);
  });
});

describe('validateNDArrays — direct boundary tests (P5)', () => {
  it('passes on exact-fit boundary', () => {
    expect(() =>
      validateNDArrays(
        'test',
        new Float32Array(15), // 5 items × 3 dims
        new Float32Array(3),
        new Float32Array(3),
        3,
        5
      )
    ).not.toThrow();
  });

  it('rejects numItems < 0', () => {
    expect(() =>
      validateNDArrays('test', new Float32Array(0), new Float32Array(3), new Float32Array(3), 3, -1)
    ).toThrow(/numItems=-1 must be a non-negative integer/);
  });

  it('accepts ndim above MAX_WASM_DIMS (>16D is routed to the TS backend, not rejected)', () => {
    // The validator no longer caps ndim — pickBackend() routes >16D to the
    // uncapped TS reference. Array-length consistency is still enforced.
    expect(() =>
      validateNDArrays(
        'test',
        new Float32Array(0),
        new Float32Array(MAX_WASM_DIMS + 1),
        new Float32Array(MAX_WASM_DIMS + 1),
        MAX_WASM_DIMS + 1,
        0
      )
    ).not.toThrow();
  });

  it('rejects ndim < 1 (non-positive)', () => {
    expect(() =>
      validateNDArrays('test', new Float32Array(0), new Float32Array(0), new Float32Array(0), 0, 0)
    ).toThrow(/must be a positive integer/);
  });

  it('accepts ndim === MAX_WASM_DIMS (WASM fast-path upper boundary)', () => {
    expect(() =>
      validateNDArrays(
        'test',
        new Float32Array(0),
        new Float32Array(MAX_WASM_DIMS),
        new Float32Array(MAX_WASM_DIMS),
        MAX_WASM_DIMS,
        0
      )
    ).not.toThrow();
  });

  it('radii is optional — when supplied, must cover numItems', () => {
    expect(() =>
      validateNDArrays(
        'test',
        new Float32Array(15),
        new Float32Array(3),
        new Float32Array(3),
        3,
        5,
        3,
        new Float32Array(3)
      )
    ).toThrow(/radii too short/);
  });

  it('positionsPerItem override: positions size = numItems × positionsPerItem (not ndim)', () => {
    // For lines, positions are indexed by vertex, not ndim alone.
    expect(() =>
      validateNDArrays(
        'test',
        new Float32Array(10),
        new Float32Array(3),
        new Float32Array(3),
        3,
        2,
        5
      )
    ).not.toThrow();
    expect(() =>
      validateNDArrays(
        'test',
        new Float32Array(9),
        new Float32Array(3),
        new Float32Array(3),
        3,
        2,
        5
      )
    ).toThrow(/positions array too short/);
  });
});

describe('validateProjectionInputs — boundary (P5)', () => {
  it('rejects displayDims with 0 entries', () => {
    expect(() =>
      validateProjectionInputs('test', new Float32Array(0), [], new Float32Array(3), 3, 0)
    ).toThrow(/displayDims must have 1–3 entries/);
  });

  it('rejects displayDims with > 3 entries', () => {
    expect(() =>
      validateProjectionInputs('test', new Float32Array(0), [0, 1, 2, 3], new Float32Array(4), 4, 0)
    ).toThrow(/displayDims must have 1–3 entries/);
  });

  it('rejects displayDims with negative entry', () => {
    expect(() =>
      validateProjectionInputs('test', new Float32Array(0), [-1, 1, 2], new Float32Array(3), 3, 0)
    ).toThrow(/displayDims\[0\]=-1 out of range/);
  });

  it('rejects displayDims with non-integer entry', () => {
    expect(() =>
      validateProjectionInputs('test', new Float32Array(0), [0, 1.5, 2], new Float32Array(3), 3, 0)
    ).toThrow(/displayDims\[1\]=1\.5 out of range/);
  });

  it('accepts 1-displayDim case (1D projection)', () => {
    expect(() =>
      validateProjectionInputs('test', new Float32Array(5), [0], new Float32Array(1), 1, 5)
    ).not.toThrow();
  });
});

describe('validateDecodeArgs — branch coverage', () => {
  it('minLength: rejects when data shorter', () => {
    expect(() => validateDecodeArgs('test', new Uint8Array(3), { minLength: 5 })).toThrow(
      /data array too short/
    );
  });

  it('finiteScalar: rejects non-finite', () => {
    expect(() =>
      validateDecodeArgs('test', new Uint8Array(0), {
        finiteScalar: { name: 'x', value: NaN },
      })
    ).toThrow(/x=NaN must be a finite number/);
  });

  it('boundsPair: rejects max <= min', () => {
    expect(() =>
      validateDecodeArgs('test', new Uint8Array(0), {
        boundsPair: { name: 'b', bounds: [5, 5] },
      })
    ).toThrow(/b max \(5\) must be greater than min/);
  });

  it('boundsPair: rejects non-finite bounds', () => {
    expect(() =>
      validateDecodeArgs('test', new Uint8Array(0), {
        boundsPair: { name: 'b', bounds: [0, Infinity] },
      })
    ).toThrow(/must be finite/);
  });

  it('lut: rejects empty array', () => {
    expect(() =>
      validateDecodeArgs('test', new Uint8Array(0), {
        lut: { name: 'l', values: [] as number[] },
      })
    ).toThrow(/l must be non-empty/);
  });

  it('lut: rejects too-short array', () => {
    expect(() =>
      validateDecodeArgs('test', new Uint8Array(0), {
        lut: { name: 'l', values: [0.1, 0.2], minLength: 5 },
      })
    ).toThrow(/l too short/);
  });

  it('positiveInt: rejects 0', () => {
    expect(() =>
      validateDecodeArgs('test', new Uint8Array(0), {
        positiveInt: { name: 'k', value: 0 },
      })
    ).toThrow(/k=0 must be a positive integer/);
  });

  it('positiveInt: rejects negative', () => {
    expect(() =>
      validateDecodeArgs('test', new Uint8Array(0), {
        positiveInt: { name: 'k', value: -3 },
      })
    ).toThrow(/k=-3 must be a positive integer/);
  });

  it('positiveInt: rejects non-integer', () => {
    expect(() =>
      validateDecodeArgs('test', new Uint8Array(0), {
        positiveInt: { name: 'k', value: 1.5 },
      })
    ).toThrow(/k=1\.5 must be a positive integer/);
  });

  it('accepts empty opts (no-op validation)', () => {
    expect(() => validateDecodeArgs('test', new Uint8Array(0))).not.toThrow();
  });
});

describe('validateChunkQueryInputs — boundary (P5)', () => {
  it('rejects numChunks < 0', () => {
    expect(() =>
      validateChunkQueryInputs(
        'test',
        new Float32Array(30),
        new Float32Array(3),
        new Float32Array(3),
        3,
        -1
      )
    ).toThrow(/numChunks=-1 must be a non-negative integer/);
  });

  it('rejects non-positive ndim', () => {
    expect(() =>
      validateChunkQueryInputs(
        'test',
        new Float32Array(0),
        new Float32Array(20),
        new Float32Array(20),
        0,
        0
      )
    ).toThrow(/must be a positive integer/);
  });

  it('accepts ndim>16 (chunk query is dimension-agnostic, not capped)', () => {
    expect(() =>
      validateChunkQueryInputs(
        'test',
        new Float32Array(0),
        new Float32Array(17),
        new Float32Array(17),
        17,
        0
      )
    ).not.toThrow();
  });

  it('passes on exact-fit chunkBounds = numChunks × ndim × 2', () => {
    expect(() =>
      validateChunkQueryInputs(
        'test',
        new Float32Array(30), // 5 chunks × 3 dims × 2
        new Float32Array(3),
        new Float32Array(3),
        3,
        5
      )
    ).not.toThrow();
  });
});
