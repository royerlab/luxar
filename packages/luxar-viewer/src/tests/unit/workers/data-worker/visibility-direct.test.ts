/**
 * Direct unit tests for data-worker/visibility/{points,lines,gsplats}.ts
 * (workers.md G1, G16, P8).
 *
 * The visibility helpers take a `WasmCtx` (state.ts) directly and call
 * `requireWasm(ctx)` to dereference the underlying module. Because the
 * ctx is supplied by the caller, no module-graph mocking is required:
 * we hand-roll a `WasmCtx` with a stub module and inspect the calls.
 *
 * Production WASM math is validated by the Rust unit suite and by the
 * WASM-vs-TS parity tests (LUXAR_REQUIRE_WASM_TESTS=1). These tests
 * cover the JS-side contract:
 *   - input validation at the worker boundary
 *   - pooled-buffer growth / reuse across calls
 *   - return-shape symmetry (visibilityMask = subarray(0, numItems))
 *   - three-geometry symmetry (P8): same name shape, same buffer reuse.
 *
 * No `vi.doMock` — the helpers under test are pure functions of their
 * `ctx` arg, so we just inject a fake module.
 */

import { describe, expect, it, vi } from 'vitest';
import { computeNDVisibilityPoints } from '../../../../workers/data-worker/visibility/points';
import { computeNDVisibilityLines } from '../../../../workers/data-worker/visibility/lines';
import { computeNDVisibilityGSplats } from '../../../../workers/data-worker/visibility/gsplats';
import type { WasmCtx } from '../../../../workers/data-worker/state';
import { NOT_INITIALIZED_MSG } from '../../../../workers/data-worker/state';

// Minimal WASM-module stub. Each visibility entry point uses ONLY its
// own visibility primitive; we leave the other slots as `vi.fn()` so
// the typing satisfies the production `WasmModule` shape.
function makeWasmStub(opts: {
  pointsReturn?: number;
  linesReturn?: number;
  gsplatsReturn?: number;
  // Per-call hook so a test can assert mask writes.
  onPoints?: (...args: unknown[]) => void;
  onLines?: (...args: unknown[]) => void;
  onGSplats?: (...args: unknown[]) => void;
} = {}) {
  return {
    compute_nd_visibility_points: vi.fn((...args: unknown[]) => {
      opts.onPoints?.(...args);
      return opts.pointsReturn ?? 0;
    }),
    compute_nd_visibility_lines: vi.fn((...args: unknown[]) => {
      opts.onLines?.(...args);
      return opts.linesReturn ?? 0;
    }),
    compute_nd_visibility_gsplats: vi.fn((...args: unknown[]) => {
      opts.onGSplats?.(...args);
      return opts.gsplatsReturn ?? 0;
    }),
  };
}

function makeCtx(wasmStub: unknown): WasmCtx {
  return {
    // Cast through unknown: production WasmModule has many slots but the
    // visibility helpers only touch one each, and the stub is opaque
    // to those callsites.
    wasm: wasmStub as WasmCtx['wasm'],
    visibilityMaskBuffer: null,
  };
}

describe('computeNDVisibilityPoints — direct unit (G1)', () => {
  it('throws NOT_INITIALIZED_MSG when ctx.wasm is null (P5)', async () => {
    const ctx: WasmCtx = { wasm: null, visibilityMaskBuffer: null };
    await expect(
      computeNDVisibilityPoints(ctx, {
        positions: new Float32Array(3),
        radii: new Float32Array(1),
        slicePosition: new Float32Array(3),
        tolerance: new Float32Array(3),
        ndim: 3,
        numPoints: 1,
      })
    ).rejects.toThrow(NOT_INITIALIZED_MSG);
  });

  it('happy path: invokes WASM with caller-supplied buffers + returns visibleCount (P2)', async () => {
    const onPoints = vi.fn();
    const wasm = makeWasmStub({ pointsReturn: 2, onPoints });
    const ctx = makeCtx(wasm);

    const positions = new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]);
    const radii = new Float32Array([0.5, 0.5, 0.5]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([1, 1, 1]);

    const result = await computeNDVisibilityPoints(ctx, {
      positions,
      radii,
      slicePosition,
      tolerance,
      ndim: 3,
      numPoints: 3,
    });

    expect(wasm.compute_nd_visibility_points).toHaveBeenCalledTimes(1);
    // Argument shape: positions, radii, slicePosition, tolerance, ndim, numPoints, buf.
    const args = onPoints.mock.calls[0];
    expect(args[0]).toBe(positions);
    expect(args[1]).toBe(radii);
    expect(args[2]).toBe(slicePosition);
    expect(args[3]).toBe(tolerance);
    expect(args[4]).toBe(3);
    expect(args[5]).toBe(3);
    expect(args[6]).toBeInstanceOf(Uint8Array);
    // visibilityMask is a subarray of the pooled buffer, length === numPoints.
    expect(result.visibilityMask).toBeInstanceOf(Uint8Array);
    expect(result.visibilityMask.length).toBe(3);
    expect(result.visibleCount).toBe(2);
  });

  it('grows pooled buffer when numPoints exceeds capacity (1.5x reservation) (P5)', async () => {
    const wasm = makeWasmStub({ pointsReturn: 0 });
    const ctx = makeCtx(wasm);
    // Initial buffer (small).
    ctx.visibilityMaskBuffer = new Uint8Array(2);

    await computeNDVisibilityPoints(ctx, {
      positions: new Float32Array(30),
      radii: new Float32Array(10),
      slicePosition: new Float32Array(3),
      tolerance: new Float32Array(3),
      ndim: 3,
      numPoints: 10,
    });

    // Post-condition: buffer reallocated. The contract is `length >= numPoints`
    // AND the implementation grows with 1.5x to avoid thrashing.
    expect(ctx.visibilityMaskBuffer).not.toBeNull();
    expect(ctx.visibilityMaskBuffer!.length).toBeGreaterThanOrEqual(10);
    // 1.5x growth: ceil(10 * 1.5) = 15 expected.
    expect(ctx.visibilityMaskBuffer!.length).toBe(15);
  });

  it('reuses the pooled buffer when it already has capacity (no realloc) (P5)', async () => {
    const wasm = makeWasmStub({ pointsReturn: 0 });
    const ctx = makeCtx(wasm);
    const big = new Uint8Array(100);
    ctx.visibilityMaskBuffer = big;

    await computeNDVisibilityPoints(ctx, {
      positions: new Float32Array(9),
      radii: new Float32Array(3),
      slicePosition: new Float32Array(3),
      tolerance: new Float32Array(3),
      ndim: 3,
      numPoints: 3,
    });

    // Identity check: same buffer reference, no reallocation.
    expect(ctx.visibilityMaskBuffer).toBe(big);
    expect(ctx.visibilityMaskBuffer!.length).toBe(100);
  });

  it('returned visibilityMask is a subarray of the pooled buffer (zero-copy view) (P2)', async () => {
    const wasm = makeWasmStub({
      pointsReturn: 0,
      onPoints: (..._args: unknown[]) => {
        const buf = _args[6] as Uint8Array;
        // Write a sentinel pattern; result.visibilityMask must reflect it.
        for (let i = 0; i < 4; i++) buf[i] = i + 1;
      },
    });
    const ctx = makeCtx(wasm);

    const result = await computeNDVisibilityPoints(ctx, {
      positions: new Float32Array(12),
      radii: new Float32Array(4),
      slicePosition: new Float32Array(3),
      tolerance: new Float32Array(3),
      ndim: 3,
      numPoints: 4,
    });

    expect(Array.from(result.visibilityMask)).toEqual([1, 2, 3, 4]);
    // The mask is a subarray view — mutations on the pool propagate.
    ctx.visibilityMaskBuffer![0] = 99;
    expect(result.visibilityMask[0]).toBe(99);
  });

  it('rejects radii too short for numPoints (boundary, P5)', async () => {
    const wasm = makeWasmStub();
    const ctx = makeCtx(wasm);
    await expect(
      computeNDVisibilityPoints(ctx, {
        positions: new Float32Array(9),
        radii: new Float32Array(2), // need 3
        slicePosition: new Float32Array(3),
        tolerance: new Float32Array(3),
        ndim: 3,
        numPoints: 3,
      })
    ).rejects.toThrow(/radii too short/);
    // WASM never invoked when validation rejects (the only guarantee).
    expect(wasm.compute_nd_visibility_points).not.toHaveBeenCalled();
  });

  it('rejects ndim out of [1, 16] (boundary, P5)', async () => {
    const wasm = makeWasmStub();
    const ctx = makeCtx(wasm);
    await expect(
      computeNDVisibilityPoints(ctx, {
        positions: new Float32Array(0),
        radii: new Float32Array(0),
        slicePosition: new Float32Array(20),
        tolerance: new Float32Array(20),
        ndim: 17,
        numPoints: 0,
      })
    ).rejects.toThrow(/ndim=17 out of range/);
  });

  it('rejects slicePosition shorter than ndim (boundary, P5)', async () => {
    const wasm = makeWasmStub();
    const ctx = makeCtx(wasm);
    await expect(
      computeNDVisibilityPoints(ctx, {
        positions: new Float32Array(9),
        radii: new Float32Array(3),
        slicePosition: new Float32Array(2), // ndim = 3
        tolerance: new Float32Array(3),
        ndim: 3,
        numPoints: 3,
      })
    ).rejects.toThrow(/slicePosition too short/);
  });
});

describe('computeNDVisibilityLines — direct unit (G1)', () => {
  it('throws NOT_INITIALIZED_MSG when ctx.wasm is null (P5)', async () => {
    const ctx: WasmCtx = { wasm: null, visibilityMaskBuffer: null };
    await expect(
      computeNDVisibilityLines(ctx, {
        vertices: new Float32Array(6),
        segments: new Uint32Array([0, 1]),
        widths: new Float32Array([1, 1]),
        slicePosition: new Float32Array(3),
        tolerance: new Float32Array(3),
        ndim: 3,
        numSegments: 1,
      })
    ).rejects.toThrow(NOT_INITIALIZED_MSG);
  });

  it('happy path: invokes WASM with caller-supplied buffers + returns visibleSegmentCount (P2)', async () => {
    const onLines = vi.fn();
    const wasm = makeWasmStub({ linesReturn: 1, onLines });
    const ctx = makeCtx(wasm);

    const vertices = new Float32Array(6);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([1, 1]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([1, 1, 1]);

    const result = await computeNDVisibilityLines(ctx, {
      vertices,
      segments,
      widths,
      slicePosition,
      tolerance,
      ndim: 3,
      numSegments: 1,
    });

    expect(wasm.compute_nd_visibility_lines).toHaveBeenCalledTimes(1);
    // Argument shape: vertices, segments, widths, slicePosition, tolerance, ndim, numSegments, buf.
    const args = onLines.mock.calls[0];
    expect(args[0]).toBe(vertices);
    expect(args[1]).toBe(segments);
    expect(args[2]).toBe(widths);
    expect(args[3]).toBe(slicePosition);
    expect(args[4]).toBe(tolerance);
    expect(args[5]).toBe(3);
    expect(args[6]).toBe(1);
    expect(args[7]).toBeInstanceOf(Uint8Array);
    expect(result.visibilityMask.length).toBe(1);
    expect(result.visibleCount).toBe(1);
  });

  it('numSegments=0 returns zero-length mask without touching buffer references (P5)', async () => {
    const wasm = makeWasmStub({ linesReturn: 0 });
    const ctx = makeCtx(wasm);
    // Pre-condition: no pooled buffer.
    expect(ctx.visibilityMaskBuffer).toBeNull();

    const result = await computeNDVisibilityLines(ctx, {
      vertices: new Float32Array(3),
      segments: new Uint32Array(0),
      widths: new Float32Array(0),
      slicePosition: new Float32Array(3),
      tolerance: new Float32Array(3),
      ndim: 3,
      numSegments: 0,
    });

    expect(result.visibleCount).toBe(0);
    expect(result.visibilityMask.length).toBe(0);
    // workers.md C5 fix: prior version pinned `visibilityMaskBuffer != null`,
    // which documents an implementation accident (the `!buf` branch fires
    // unconditionally even at numSegments=0). If the source is later
    // optimised to skip allocation when numSegments===0, this assertion
    // would fail for a non-bug. The behavioural contract is just "no
    // crash, return zero-length mask" — that's what's asserted above.
  });

  it('grows pooled buffer to ≥ numSegments using 1.5x policy (symmetry with points) (P8)', async () => {
    const wasm = makeWasmStub();
    const ctx = makeCtx(wasm);
    ctx.visibilityMaskBuffer = new Uint8Array(2);

    await computeNDVisibilityLines(ctx, {
      vertices: new Float32Array(30),
      segments: new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      widths: new Float32Array(10),
      slicePosition: new Float32Array(3),
      tolerance: new Float32Array(3),
      ndim: 3,
      numSegments: 5,
    });

    expect(ctx.visibilityMaskBuffer!.length).toBeGreaterThanOrEqual(5);
    // ceil(5 * 1.5) = 8 — symmetric with points pool policy.
    expect(ctx.visibilityMaskBuffer!.length).toBe(8);
  });

  it('rejects segments referencing past end of vertices (boundary, P5)', async () => {
    const wasm = makeWasmStub();
    const ctx = makeCtx(wasm);
    // segments[3] = 8 but vertices only has 5 vertices (15 floats).
    await expect(
      computeNDVisibilityLines(ctx, {
        vertices: new Float32Array(15),
        segments: new Uint32Array([0, 1, 2, 8]),
        widths: new Float32Array(9),
        slicePosition: new Float32Array(3),
        tolerance: new Float32Array(3),
        ndim: 3,
        numSegments: 2,
      })
    ).rejects.toThrow(/positions too short for max segment vertex 8/);
    expect(wasm.compute_nd_visibility_lines).not.toHaveBeenCalled();
  });

  it('rejects widths shorter than max-vertex+1 (boundary, P5)', async () => {
    const wasm = makeWasmStub();
    const ctx = makeCtx(wasm);
    await expect(
      computeNDVisibilityLines(ctx, {
        vertices: new Float32Array(30),
        segments: new Uint32Array([0, 1, 2, 3]),
        widths: new Float32Array(2), // need at least 4
        slicePosition: new Float32Array(3),
        tolerance: new Float32Array(3),
        ndim: 3,
        numSegments: 2,
      })
    ).rejects.toThrow(/widths too short/);
  });

  it('rejects ndim out of [1, 16] (boundary, P5)', async () => {
    const wasm = makeWasmStub();
    const ctx = makeCtx(wasm);
    await expect(
      computeNDVisibilityLines(ctx, {
        vertices: new Float32Array(0),
        segments: new Uint32Array(0),
        widths: new Float32Array(0),
        slicePosition: new Float32Array(20),
        tolerance: new Float32Array(20),
        ndim: 17,
        numSegments: 0,
      })
    ).rejects.toThrow(/ndim=17 out of range/);
  });

  it('rejects slicePosition shorter than ndim (boundary, P5)', async () => {
    const wasm = makeWasmStub();
    const ctx = makeCtx(wasm);
    await expect(
      computeNDVisibilityLines(ctx, {
        vertices: new Float32Array(9),
        segments: new Uint32Array([0, 1]),
        widths: new Float32Array([1, 1]),
        slicePosition: new Float32Array(2), // ndim = 3
        tolerance: new Float32Array(3),
        ndim: 3,
        numSegments: 1,
      })
    ).rejects.toThrow(/slicePosition.*too short/);
  });
});

describe('computeNDVisibilityGSplats — direct unit (G1)', () => {
  it('throws NOT_INITIALIZED_MSG when ctx.wasm is null (P5)', async () => {
    const ctx: WasmCtx = { wasm: null, visibilityMaskBuffer: null };
    await expect(
      computeNDVisibilityGSplats(ctx, {
        centers: new Float32Array(3),
        choleskyFactors: new Float32Array(6),
        slicePosition: new Float32Array(3),
        tolerance: new Float32Array(3),
        ndim: 3,
        numSplats: 1,
      })
    ).rejects.toThrow(NOT_INITIALIZED_MSG);
  });

  it('happy path: invokes WASM with caller-supplied buffers + returns visibleSplatCount (P2)', async () => {
    const onGSplats = vi.fn();
    const wasm = makeWasmStub({ gsplatsReturn: 2, onGSplats });
    const ctx = makeCtx(wasm);

    const splatCount = 3;
    const ndim = 3;
    const k = (ndim * (ndim + 1)) / 2; // 6 for 3D
    const centers = new Float32Array(splatCount * ndim);
    const choleskyFactors = new Float32Array(splatCount * k);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([1, 1, 1]);

    const result = await computeNDVisibilityGSplats(ctx, {
      centers,
      choleskyFactors,
      slicePosition,
      tolerance,
      ndim,
      numSplats: splatCount,
    });

    expect(wasm.compute_nd_visibility_gsplats).toHaveBeenCalledTimes(1);
    // Argument shape: centers, choleskyFactors, slicePosition, tolerance, ndim, numSplats, buf.
    const args = onGSplats.mock.calls[0];
    expect(args[0]).toBe(centers);
    expect(args[1]).toBe(choleskyFactors);
    expect(args[2]).toBe(slicePosition);
    expect(args[3]).toBe(tolerance);
    expect(args[4]).toBe(ndim);
    expect(args[5]).toBe(splatCount);
    expect(args[6]).toBeInstanceOf(Uint8Array);
    expect(result.visibilityMask.length).toBe(splatCount);
    expect(result.visibleCount).toBe(2);
  });

  it('rejects choleskyFactors shorter than packed-lower-triangular size (boundary, P5)', async () => {
    const wasm = makeWasmStub();
    const ctx = makeCtx(wasm);
    // ndim=3 → packed lower = 3*4/2 = 6 per splat; 5 splats → 30 entries.
    await expect(
      computeNDVisibilityGSplats(ctx, {
        centers: new Float32Array(15),
        choleskyFactors: new Float32Array(20), // need 30
        slicePosition: new Float32Array(3),
        tolerance: new Float32Array(3),
        ndim: 3,
        numSplats: 5,
      })
    ).rejects.toThrow(/choleskyFactors too short/);
    expect(wasm.compute_nd_visibility_gsplats).not.toHaveBeenCalled();
  });

  it('rejects ndim out of [1, 16] (boundary, P5)', async () => {
    const wasm = makeWasmStub();
    const ctx = makeCtx(wasm);
    await expect(
      computeNDVisibilityGSplats(ctx, {
        centers: new Float32Array(0),
        choleskyFactors: new Float32Array(0),
        slicePosition: new Float32Array(20),
        tolerance: new Float32Array(20),
        ndim: 17,
        numSplats: 0,
      })
    ).rejects.toThrow(/ndim=17 out of range/);
  });

  it('grows pooled buffer using 1.5x policy (symmetry with points/lines) (P8)', async () => {
    const wasm = makeWasmStub();
    const ctx = makeCtx(wasm);
    ctx.visibilityMaskBuffer = new Uint8Array(1);

    const splatCount = 4;
    const ndim = 3;
    const k = (ndim * (ndim + 1)) / 2;

    await computeNDVisibilityGSplats(ctx, {
      centers: new Float32Array(splatCount * ndim),
      choleskyFactors: new Float32Array(splatCount * k),
      slicePosition: new Float32Array(3),
      tolerance: new Float32Array(3),
      ndim,
      numSplats: splatCount,
    });

    expect(ctx.visibilityMaskBuffer!.length).toBeGreaterThanOrEqual(splatCount);
    // ceil(4 * 1.5) = 6 — same growth policy as points and lines (P8).
    expect(ctx.visibilityMaskBuffer!.length).toBe(6);
  });
});

describe('three-geometry symmetry across visibility helpers (P8)', () => {
  // The three helpers share an identical buffer-pool contract:
  //   - reuse `ctx.visibilityMaskBuffer` if it has capacity
  //   - otherwise allocate `Math.ceil(numItems * 1.5)`
  //   - returned mask is `buf.subarray(0, numItems)` (zero-copy view)
  //
  // If a future refactor breaks any one of those properties on any one
  // geometry, this test pins the symmetry: the same input shape must
  // produce the same pool growth & subarray contract on all three.

  it('all three helpers grow the pooled buffer to ceil(N * 1.5) on first call', async () => {
    const N = 7;
    const ndim = 3;
    const k = (ndim * (ndim + 1)) / 2;

    // Points.
    const ctxP = makeCtx(makeWasmStub());
    await computeNDVisibilityPoints(ctxP, {
      positions: new Float32Array(N * ndim),
      radii: new Float32Array(N),
      slicePosition: new Float32Array(ndim),
      tolerance: new Float32Array(ndim),
      ndim,
      numPoints: N,
    });

    // Lines (N segments × 2 vertices = 14, need 14 widths/vertices).
    const ctxL = makeCtx(makeWasmStub());
    const segs = new Uint32Array(N * 2);
    for (let i = 0; i < N * 2; i++) segs[i] = i;
    await computeNDVisibilityLines(ctxL, {
      vertices: new Float32Array(N * 2 * ndim),
      segments: segs,
      widths: new Float32Array(N * 2),
      slicePosition: new Float32Array(ndim),
      tolerance: new Float32Array(ndim),
      ndim,
      numSegments: N,
    });

    // GSplats.
    const ctxG = makeCtx(makeWasmStub());
    await computeNDVisibilityGSplats(ctxG, {
      centers: new Float32Array(N * ndim),
      choleskyFactors: new Float32Array(N * k),
      slicePosition: new Float32Array(ndim),
      tolerance: new Float32Array(ndim),
      ndim,
      numSplats: N,
    });

    const expected = Math.ceil(N * 1.5); // 11
    expect(ctxP.visibilityMaskBuffer!.length).toBe(expected);
    expect(ctxL.visibilityMaskBuffer!.length).toBe(expected);
    expect(ctxG.visibilityMaskBuffer!.length).toBe(expected);
  });

  it('all three helpers return a visibilityMask of length === N (subarray contract)', async () => {
    const N = 5;
    const ndim = 3;
    const k = (ndim * (ndim + 1)) / 2;

    const ctxP = makeCtx(makeWasmStub());
    const resP = await computeNDVisibilityPoints(ctxP, {
      positions: new Float32Array(N * ndim),
      radii: new Float32Array(N),
      slicePosition: new Float32Array(ndim),
      tolerance: new Float32Array(ndim),
      ndim,
      numPoints: N,
    });

    const ctxL = makeCtx(makeWasmStub());
    const segs = new Uint32Array(N * 2);
    for (let i = 0; i < N * 2; i++) segs[i] = i;
    const resL = await computeNDVisibilityLines(ctxL, {
      vertices: new Float32Array(N * 2 * ndim),
      segments: segs,
      widths: new Float32Array(N * 2),
      slicePosition: new Float32Array(ndim),
      tolerance: new Float32Array(ndim),
      ndim,
      numSegments: N,
    });

    const ctxG = makeCtx(makeWasmStub());
    const resG = await computeNDVisibilityGSplats(ctxG, {
      centers: new Float32Array(N * ndim),
      choleskyFactors: new Float32Array(N * k),
      slicePosition: new Float32Array(ndim),
      tolerance: new Float32Array(ndim),
      ndim,
      numSplats: N,
    });

    expect(resP.visibilityMask.length).toBe(N);
    expect(resL.visibilityMask.length).toBe(N);
    expect(resG.visibilityMask.length).toBe(N);
  });
});
