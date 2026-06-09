/**
 * Atomic commit stage of `updateView`.
 *
 * After Stage 1 (parallel async load + process) finishes, all three
 * per-type staged-commit arrays are passed in here. The body runs as
 * a single synchronous block — JS is single-threaded so no
 * requestAnimationFrame can fire during it, which means every mesh
 * that participates updates in the SAME rendered frame. Without this
 * atomicity, dimension animation would flicker between partially-
 * updated and fully-updated frames.
 *
 * The commit step is wrapped in `try/finally` per-iteration AND
 * outer `try/finally` belt-and-braces sweep so every opened profiler
 * session ends exactly once even if a commit throws synchronously.
 * `SessionImpl.end()` is idempotent.
 */

import type { UpdateSession } from '../../../profiling/update-profiler';
import type { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { NodeFactory } from '../../../rendering/node-factory';
import type { LoadedPointsData } from '../../data-loader-types';
import type { StagedLinesCommit } from '../process/data-processor-lines';
import type { StagedGSplatsCommit } from '../process/data-processor-gsplats';

export interface AtomicCommitInput<TStaged> {
  staged: TStaged | null;
  session: UpdateSession;
}

export interface AtomicCommitCtx {
  /** Optional GPU buffer pool — frame counter is bumped once per cycle. */
  gpuBufferPool: GPUBufferPool | null;
  /** Pickable-state invalidator (cached pick buffer goes stale on geometry change). */
  nodeFactory: NodeFactory;
  /**
   * Per-update abort signal. When `aborted`, this update was superseded:
   * SKIP every geometry mutation (beginFrame / commits / markPickingDirty)
   * so no stale or partial frame reaches the GPU — but STILL end every
   * profiler session below (the sessions were opened in `runLoaderUpdates`
   * and must be closed exactly once regardless of commit). See Guards G5/G6.
   */
  signal?: AbortSignal;
  /** Per-type commit callbacks routed through the orchestrator's delegates. */
  updatePointsGeometry(path: string, data: LoadedPointsData, session?: UpdateSession): void;
  commitLinesGeometry(staged: StagedLinesCommit, session?: UpdateSession): void;
  commitGSplatsGeometry(staged: StagedGSplatsCommit, session?: UpdateSession): void;
}

/**
 * Run the synchronous commit-and-end-sessions stage. After this
 * returns, every mesh that participated in the update is on the GPU
 * in its post-update form. Invalidates the picking cache if any commit
 * happened.
 */
export function runAtomicCommit(
  pointsStaged: AtomicCommitInput<{ path: string; data: LoadedPointsData }>[],
  linesStaged: AtomicCommitInput<StagedLinesCommit>[],
  gsplatsStaged: AtomicCommitInput<StagedGSplatsCommit>[],
  ctx: AtomicCommitCtx
): void {
  // Commits run inside each per-node session so the GPU-upload step
  // ("Update Buffers") shows up under Points/Lines/GSplats in the
  // Performance tab. Always end the session afterwards — including
  // the staged === null case (loader failed or marked skipped) so
  // every opened session is closed exactly once.
  //
  // Belt-and-braces: if a commit (or beginFrame) throws synchronously,
  // the remaining iterations and the later geometry-type loops never
  // run, leaving their sessions un-ended. The outer `finally` sweeps
  // every staged session afterwards. `SessionImpl.end()` is idempotent
  // (no-ops on already-ended sessions), so this is safe to overlay on
  // the per-iteration end() calls that record accurate per-node timings
  // on the happy path. beginFrame() is inside the try so a future
  // throwing implementation can't leak the already-opened sessions.
  // Superseded mid-flight: skip all geometry mutations, but the session-end
  // sweep below MUST still run (G5/G6). `if (staged && !aborted)` keeps the
  // skip all-or-nothing across the three geometry types.
  const aborted = ctx.signal?.aborted ?? false;

  try {
    // Advance GPU buffer pool frame counter once per update cycle
    // (not per-acquire) so eviction timing reflects actual frames.
    if (ctx.gpuBufferPool && !aborted) {
      ctx.gpuBufferPool.beginFrame();
    }

    for (const { staged, session } of pointsStaged) {
      try {
        if (staged && !aborted) ctx.updatePointsGeometry(staged.path, staged.data, session);
      } finally {
        session.end();
      }
    }
    for (const { staged, session } of linesStaged) {
      try {
        if (staged && !aborted) ctx.commitLinesGeometry(staged, session);
      } finally {
        session.end();
      }
    }
    for (const { staged, session } of gsplatsStaged) {
      try {
        if (staged && !aborted) ctx.commitGSplatsGeometry(staged, session);
      } finally {
        session.end();
      }
    }
  } finally {
    for (const { session } of pointsStaged) session.end();
    for (const { session } of linesStaged) session.end();
    for (const { session } of gsplatsStaged) session.end();
  }

  // Invalidate cached pick buffer after geometry changes (skip when aborted —
  // nothing was committed, so the pick buffer is still valid for the prior frame).
  if (!aborted && (pointsStaged.length > 0 || linesStaged.length > 0 || gsplatsStaged.length > 0)) {
    ctx.nodeFactory.markPickingDirty();
  }
}
