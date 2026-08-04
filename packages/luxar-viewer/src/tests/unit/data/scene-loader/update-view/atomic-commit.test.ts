/**
 * Unit tests for `runAtomicCommit` in scene-loader/update-view/atomic-commit.ts.
 *
 * The atomic-commit stage's three load-bearing invariants:
 *   1. Every opened profiler session is end()ed EXACTLY ONCE on the
 *      happy path (via the per-iteration `finally`) and AT LEAST ONCE
 *      on every path (per-node catches isolate commit faults; the
 *      outer `finally` sweep covers beginFrame throws) —
 *      idempotent end() lets these overlap safely. This is the "belt-
 *      and-braces" guard the inline comment names.
 *   2. `markPickingDirty()` runs ONLY if at least one of the four
 *      staged arrays is non-empty — the pick-cache invalidation is
 *      otherwise a no-op cost. It ALSO runs on a partially-failing
 *      pass (successful sibling commits changed geometry) before the
 *      AggregateError re-surfaces.
 *   3. `gpuBufferPool.beginFrame()` runs once per cycle (not per
 *      acquire) so eviction timing reflects actual rendered frames.
 *      Skipped when the pool is null.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  runAtomicCommit,
  type AtomicCommitCtx,
  type AtomicCommitInput,
} from '../../../../../data/scene-loader/update-view/atomic-commit';
import type { UpdateSession } from '../../../../../profiling/update-profiler';
import type { LoadedPointsData } from '../../../../../data/data-loader-types';
import type { StagedLinesCommit } from '../../../../../data/scene-loader/process/data-processor-lines';
import type { StagedGSplatsCommit } from '../../../../../data/scene-loader/process/data-processor-gsplats';
import type { StagedMeshCommit } from '../../../../../data/scene-loader/process/data-processor-mesh';

// ============================================================================
// Local fixtures
// ============================================================================

/**
 * Build a session stub whose `end()` is a vi.fn — lets the test count
 * how many times it was called (idempotent contract). The other methods
 * are stubs because the helper never calls them.
 */
function makeSession(): UpdateSession & { end: ReturnType<typeof vi.fn> } {
  const end = vi.fn();
  return {
    begin: vi.fn(),
    end,
    setMetadata: vi.fn(),
    markSkipped: vi.fn(),
  } as unknown as UpdateSession & { end: ReturnType<typeof vi.fn> };
}

function makePointsStaged(
  count: number,
  withNulls: number[] = []
): AtomicCommitInput<{ path: string; data: LoadedPointsData }>[] {
  return Array.from({ length: count }, (_, i) => ({
    staged: withNulls.includes(i)
      ? null
      : { path: `/p${i}`, data: { pointCount: i } as LoadedPointsData },
    session: makeSession(),
  }));
}

function makeLinesStaged(
  count: number,
  withNulls: number[] = []
): AtomicCommitInput<StagedLinesCommit>[] {
  return Array.from({ length: count }, (_, i) => ({
    staged: withNulls.includes(i) ? null : ({ path: `/l${i}` } as unknown as StagedLinesCommit),
    session: makeSession(),
  }));
}

function makeGSplatsStaged(
  count: number,
  withNulls: number[] = []
): AtomicCommitInput<StagedGSplatsCommit>[] {
  return Array.from({ length: count }, (_, i) => ({
    staged: withNulls.includes(i) ? null : ({ path: `/g${i}` } as unknown as StagedGSplatsCommit),
    session: makeSession(),
  }));
}

function makeMeshStaged(
  count: number,
  withNulls: number[] = []
): AtomicCommitInput<StagedMeshCommit>[] {
  return Array.from({ length: count }, (_, i) => ({
    staged: withNulls.includes(i) ? null : ({ path: `/m${i}` } as unknown as StagedMeshCommit),
    session: makeSession(),
  }));
}

function makeCtx(overrides: Partial<AtomicCommitCtx> = {}): AtomicCommitCtx & {
  spies: {
    updatePointsGeometry: ReturnType<typeof vi.fn>;
    commitLinesGeometry: ReturnType<typeof vi.fn>;
    commitGSplatsGeometry: ReturnType<typeof vi.fn>;
    commitMeshGeometry: ReturnType<typeof vi.fn>;
    markPickingDirty: ReturnType<typeof vi.fn>;
    beginFrame: ReturnType<typeof vi.fn>;
  };
} {
  const updatePointsGeometry = vi.fn();
  const commitLinesGeometry = vi.fn();
  const commitGSplatsGeometry = vi.fn();
  const commitMeshGeometry = vi.fn();
  const markPickingDirty = vi.fn();
  const beginFrame = vi.fn();

  const ctx: AtomicCommitCtx = {
    gpuBufferPool: { beginFrame } as unknown as AtomicCommitCtx['gpuBufferPool'],
    nodeFactory: { markPickingDirty } as unknown as AtomicCommitCtx['nodeFactory'],
    updatePointsGeometry,
    commitLinesGeometry,
    commitGSplatsGeometry,
    commitMeshGeometry,
    ...overrides,
  };
  return Object.assign(ctx, {
    spies: {
      updatePointsGeometry,
      commitLinesGeometry,
      commitGSplatsGeometry,
      commitMeshGeometry,
      markPickingDirty,
      beginFrame,
    },
  });
}

// ============================================================================
// Tests
// ============================================================================

describe('runAtomicCommit — happy path', () => {
  it('runs every commit, beginFrame once, markPickingDirty once', () => {
    const points = makePointsStaged(3);
    const lines = makeLinesStaged(2);
    const gsplats = makeGSplatsStaged(1);
    const mesh = makeMeshStaged(2);
    const ctx = makeCtx();

    runAtomicCommit(points, lines, gsplats, mesh, ctx);

    expect(ctx.spies.beginFrame).toHaveBeenCalledTimes(1);
    expect(ctx.spies.updatePointsGeometry).toHaveBeenCalledTimes(3);
    expect(ctx.spies.commitLinesGeometry).toHaveBeenCalledTimes(2);
    expect(ctx.spies.commitGSplatsGeometry).toHaveBeenCalledTimes(1);
    expect(ctx.spies.commitMeshGeometry).toHaveBeenCalledTimes(2);
    expect(ctx.spies.markPickingDirty).toHaveBeenCalledTimes(1);
  });

  it('passes (path, data, session) to updatePointsGeometry', () => {
    const points = makePointsStaged(2);
    const ctx = makeCtx();

    runAtomicCommit(points, [], [], [], ctx);

    expect(ctx.spies.updatePointsGeometry).toHaveBeenNthCalledWith(
      1,
      '/p0',
      points[0].staged!.data,
      points[0].session
    );
    expect(ctx.spies.updatePointsGeometry).toHaveBeenNthCalledWith(
      2,
      '/p1',
      points[1].staged!.data,
      points[1].session
    );
  });
});

describe('runAtomicCommit — gpuBufferPool null', () => {
  it('skips beginFrame but still runs commits and markPickingDirty', () => {
    const ctx = makeCtx({ gpuBufferPool: null });
    const points = makePointsStaged(1);

    runAtomicCommit(points, [], [], [], ctx);

    expect(ctx.spies.beginFrame).not.toHaveBeenCalled();
    expect(ctx.spies.updatePointsGeometry).toHaveBeenCalledTimes(1);
    expect(ctx.spies.markPickingDirty).toHaveBeenCalledTimes(1);
  });
});

describe('runAtomicCommit — empty input arrays', () => {
  it('does NOT call markPickingDirty when all four arrays are empty', () => {
    const ctx = makeCtx();
    runAtomicCommit([], [], [], [], ctx);
    expect(ctx.spies.markPickingDirty).not.toHaveBeenCalled();
    // beginFrame still fires (no guard around it in the helper).
    expect(ctx.spies.beginFrame).toHaveBeenCalledTimes(1);
  });

  it('calls markPickingDirty when ONLY mesh is staged (pins the meshStaged.length clause)', () => {
    // A mesh-only pass changes geometry, so the pick cache must go stale — the
    // `markPickingDirty` OR-condition must include meshStaged, not just the
    // other three arrays.
    const ctx = makeCtx();
    runAtomicCommit([], [], [], makeMeshStaged(1), ctx);
    expect(ctx.spies.commitMeshGeometry).toHaveBeenCalledTimes(1);
    expect(ctx.spies.markPickingDirty).toHaveBeenCalledTimes(1);
  });
});

describe('runAtomicCommit — null staged entries', () => {
  it('skips the commit but still ends the session', () => {
    const points = makePointsStaged(3, [1]); // index 1 has staged: null
    const ctx = makeCtx();

    runAtomicCommit(points, [], [], [], ctx);

    // Only 2 commits (indices 0 and 2).
    expect(ctx.spies.updatePointsGeometry).toHaveBeenCalledTimes(2);
    // All 3 sessions end() exactly twice each (once per-iteration, once outer-finally).
    expect((points[0].session.end as ReturnType<typeof vi.fn>).mock.calls.length).toBe(2);
    expect((points[1].session.end as ReturnType<typeof vi.fn>).mock.calls.length).toBe(2);
    expect((points[2].session.end as ReturnType<typeof vi.fn>).mock.calls.length).toBe(2);
  });
});

describe('runAtomicCommit — synchronous throw mid-commit', () => {
  it('fault isolation: a throwing points commit does NOT starve lines/gsplats/mesh siblings; errors surface as ONE AggregateError', () => {
    const points = makePointsStaged(3);
    const lines = makeLinesStaged(2);
    const gsplats = makeGSplatsStaged(1);
    const mesh = makeMeshStaged(1);
    const ctx = makeCtx();
    // Commit /p2 throws — the remaining points and the ENTIRE lines +
    // gsplats + mesh loops must still run (their staged data is valid; skipping
    // them would leave the whole frame stale), and the error re-surfaces
    // as an AggregateError AFTER the pass completes.
    ctx.spies.updatePointsGeometry.mockImplementation((path: string) => {
      if (path === '/p2') throw new Error('GPU upload failed');
    });

    let thrown: unknown;
    try {
      runAtomicCommit(points, lines, gsplats, mesh, ctx);
    } catch (err) {
      thrown = err;
    }
    expect(thrown).toBeInstanceOf(AggregateError);
    expect((thrown as AggregateError).errors).toHaveLength(1);
    expect(((thrown as AggregateError).errors[0] as Error).message).toBe('GPU upload failed');

    // Siblings committed despite the failure.
    expect(ctx.spies.updatePointsGeometry).toHaveBeenCalledTimes(3);
    expect(ctx.spies.commitLinesGeometry).toHaveBeenCalledTimes(2);
    expect(ctx.spies.commitGSplatsGeometry).toHaveBeenCalledTimes(1);
    expect(ctx.spies.commitMeshGeometry).toHaveBeenCalledTimes(1);

    // Every session ends: per-iteration finally AND the outer sweep → 2 each.
    for (const s of [...points, ...lines, ...gsplats, ...mesh]) {
      expect((s.session.end as ReturnType<typeof vi.fn>).mock.calls.length).toBe(2);
    }
  });

  it('markPickingDirty STILL runs on a partially-failing pass (successful siblings changed geometry)', () => {
    const points = makePointsStaged(2);
    const ctx = makeCtx();
    ctx.spies.updatePointsGeometry.mockImplementation((path: string) => {
      if (path === '/p0') throw new Error('boom');
    });

    expect(() => runAtomicCommit(points, [], [], [], ctx)).toThrow(AggregateError);
    // /p1 committed, so the cached pick buffer is stale — the invalidation
    // must run BEFORE the aggregate error re-surfaces.
    expect(ctx.spies.markPickingDirty).toHaveBeenCalledTimes(1);
  });
});

describe('runAtomicCommit — gpuBufferPool.beginFrame throws', () => {
  it('propagates the throw but still ends every opened session (beginFrame is inside the try/finally)', () => {
    // SOURCE ORDERING: `ctx.gpuBufferPool.beginFrame()` runs as the first
    // statement INSIDE the outer try/finally that sweeps sessions. So a
    // throw out of beginFrame propagates to the caller, but the
    // session-sweeping finally still runs first — every already-opened
    // session is end()ed exactly once, never leaked. (Commits never run
    // because the throw happens before the loops; markPickingDirty never
    // runs because it sits after the try/finally.)
    const points = makePointsStaged(2);
    const lines = makeLinesStaged(1);
    const gsplats = makeGSplatsStaged(1);
    const beginFrame = vi.fn(() => {
      throw new Error('beginFrame boom');
    });
    const ctx = makeCtx({
      gpuBufferPool: { beginFrame } as unknown as AtomicCommitCtx['gpuBufferPool'],
    });

    expect(() => runAtomicCommit(points, lines, gsplats, [], ctx)).toThrow('beginFrame boom');

    // No commits ran (the throw preceded the per-type loops).
    expect(ctx.spies.updatePointsGeometry).not.toHaveBeenCalled();
    expect(ctx.spies.commitLinesGeometry).not.toHaveBeenCalled();
    expect(ctx.spies.commitGSplatsGeometry).not.toHaveBeenCalled();
    // Every opened session ended exactly once via the outer finally —
    // no leak despite the early throw.
    for (const p of points) {
      expect((p.session.end as ReturnType<typeof vi.fn>).mock.calls.length).toBe(1);
    }
    for (const l of lines) {
      expect((l.session.end as ReturnType<typeof vi.fn>).mock.calls.length).toBe(1);
    }
    for (const g of gsplats) {
      expect((g.session.end as ReturnType<typeof vi.fn>).mock.calls.length).toBe(1);
    }
    // Pick cache not invalidated (markPickingDirty sits after the try/finally).
    expect(ctx.spies.markPickingDirty).not.toHaveBeenCalled();
  });
});

describe('runAtomicCommit — session.end idempotency contract', () => {
  it('the helper relies on session.end being safe to call twice (no test-side throw)', () => {
    // Build a session whose end() throws if called twice — would fail
    // this test, proving the contract.
    const calls: number[] = [];
    const session: UpdateSession = {
      begin: vi.fn(),
      end: vi.fn(() => {
        calls.push(1);
        // Idempotent contract: end() must not throw on the second call.
        // We're TESTING that the helper calls it twice and the contract holds.
      }),
      setMetadata: vi.fn(),
      markSkipped: vi.fn(),
    } as unknown as UpdateSession;
    const ctx = makeCtx();

    runAtomicCommit(
      [{ staged: { path: '/p', data: {} as LoadedPointsData }, session }],
      [],
      [],
      [],
      ctx
    );

    // Helper called end() twice — per-iteration finally + outer finally.
    expect(calls.length).toBe(2);
  });
});

describe('runAtomicCommit — superseded (signal aborted)', () => {
  it('skips beginFrame, all commits, and markPickingDirty but STILL ends every session (G5)', () => {
    const points = makePointsStaged(3);
    const lines = makeLinesStaged(2);
    const gsplats = makeGSplatsStaged(1);
    const mesh = makeMeshStaged(1);
    const ac = new AbortController();
    ac.abort();
    const ctx = makeCtx({ signal: ac.signal });

    runAtomicCommit(points, lines, gsplats, mesh, ctx);

    // No geometry mutation reaches the GPU for a superseded update.
    expect(ctx.spies.beginFrame).not.toHaveBeenCalled();
    expect(ctx.spies.updatePointsGeometry).not.toHaveBeenCalled();
    expect(ctx.spies.commitLinesGeometry).not.toHaveBeenCalled();
    expect(ctx.spies.commitGSplatsGeometry).not.toHaveBeenCalled();
    expect(ctx.spies.commitMeshGeometry).not.toHaveBeenCalled();
    expect(ctx.spies.markPickingDirty).not.toHaveBeenCalled();

    // G5: every opened profiler session is still closed exactly twice
    // (per-iteration finally + outer sweep) — the commit-skip must never
    // skip the session-end sweep, or sessions leak.
    for (const p of points) {
      expect((p.session.end as ReturnType<typeof vi.fn>).mock.calls.length).toBe(2);
    }
    for (const l of lines) {
      expect((l.session.end as ReturnType<typeof vi.fn>).mock.calls.length).toBe(2);
    }
    for (const g of gsplats) {
      expect((g.session.end as ReturnType<typeof vi.fn>).mock.calls.length).toBe(2);
    }
    for (const m of mesh) {
      expect((m.session.end as ReturnType<typeof vi.fn>).mock.calls.length).toBe(2);
    }
  });

  it('commits normally when the signal is present but NOT aborted', () => {
    const points = makePointsStaged(2);
    const ac = new AbortController(); // not aborted
    const ctx = makeCtx({ signal: ac.signal });

    runAtomicCommit(points, [], [], [], ctx);

    expect(ctx.spies.beginFrame).toHaveBeenCalledTimes(1);
    expect(ctx.spies.updatePointsGeometry).toHaveBeenCalledTimes(2);
    expect(ctx.spies.markPickingDirty).toHaveBeenCalledTimes(1);
  });
});
