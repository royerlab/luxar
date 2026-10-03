# `updateView` orchestration helpers

Helpers extracted from `data/scene-loader.ts::updateView`. The orchestrator
itself stays in `data/scene-loader.ts`; this folder owns per-type ctx
construction (entry), the atomic GPU commit (Stage 2), and the serialization
of passes, refinement runs and retries (`PassScheduler`).

## Files

| File                   | Role                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build-update-ctxs.ts` | Constructs the per-type handler ctx objects that feed `data/{points,lines,gsplats,mesh}/handler.ts::loadAndStage`. Centralises the shared fields (rootGroup, viewStateQueue, clearFailure, currentVersion, deriveNodeViewState) plus the type-specific bits (`updateVersion` for Lines/GSplats, `extendedToleranceCache` shared between Points + GSplats).                                                           |
| `atomic-commit.ts`     | Runs the synchronous Stage 2 commit. Takes the per-type staged-commit arrays from Stage 1 and routes them through the per-type commit callbacks, bumps the GPU buffer pool commit counter, and invalidates the picking cache. The body is a single synchronous block so every participating mesh updates in the SAME rendered frame. A failing node commit is rolled back and rethrown after its siblings committed. |
| `pass-scheduler.ts`    | `PassScheduler` — owns the serialization lock and everything that decides what runs next: the pending slot's lifecycle, request generations and their waiters (#2943), the targeted-resync stash, the B5 drag commit guarantee, refinement start / cancellation hand-off / release, and retry pre-emption of a refinement drain (A10). Explicit phases: `idle`, `pass`, `yielding`, `refining`, `retrying`.          |

## Public surface

None of these are re-exported from `data/` — only `data/scene-loader.ts`
consumes them (`DRAG_COMMIT_*` are re-exported from it).

- `buildUpdateCtxs(input: UpdateCtxsInput): { pointsCtx, linesCtx, gsplatsCtx, meshCtx }`
- `runAtomicCommit(pointsStaged, linesStaged, gsplatsStaged, meshStaged, ctx): void`
- `new PassScheduler(host: PassSchedulerHost)` — `request` / `beginPass` /
  `endPass` around a view pass; `startRefinement`, `beginRefinement`,
  `handOff`, `releaseRefinementLock`, `endRefinement` around a refinement run;
  `acquireForRetry` / `releaseRetry` around a retry; `kickRefinementIfIdle`,
  `releaseAndDrain`, `drainPending`, `resolveWaiters`, `dispose`.

## Invariants

- **Atomic commit is single-frame.** JS is single-threaded, so the
  synchronous body of `runAtomicCommit` cannot be interrupted by a
  `requestAnimationFrame` callback. Without this atomicity, dimension
  animation would flicker between partially-updated and fully-updated
  frames. Any per-type commit added later MUST stay inside the same
  synchronous block.
- **Session.end() is idempotent and swept twice.** `runAtomicCommit`
  closes every `UpdateSession` exactly once on the happy path (per-
  iteration `finally`) AND once more via an outer `try/finally`
  belt-and-braces sweep. If a commit throws synchronously, the later
  iterations and geometry-type loops never run; the outer sweep
  guarantees every opened profiler session still ends. The "ended
  twice" case is a no-op by `SessionImpl.end()` contract.
- **Picking cache invalidation is gated.** `nodeFactory.markPickingDirty()`
  fires only if at least one staged-commit array was non-empty —
  otherwise an update cycle that staged nothing (all loaders skipped
  or failed) would needlessly bust the pick buffer.
- **After a pass, pending wins over refinement.** A view queued during a
  pass runs next (after one frame); refinement starts only when nothing is
  queued, and a view queued during refinement cancels it.
- **The lock is held across the frame yield and the refinement drain**, so a
  view arriving then is queued rather than racing a concurrent pass. The
  state stays in the pending slot until the frame fires and is taken then, so
  the NEWEST state runs (A8). With no pass running, a request neither aborts
  nor holds anything: only a running view pass can be owed a B5 commit.
- **One release path.** Every place that gives the lock up with possibly a
  state queued behind it — the refinement run's final release and every
  orchestrator-failure recovery — goes through `releaseAndDrain`: drain the
  pending state into a fresh pass, else settle the waiters. A drained state's
  pass settles them at its own commit; settling here too would release the
  pacing gate before the view it asked for landed.
- **Frame scheduling is hidden-tab-proof.** The yield goes through
  `utils/schedule-frame.ts` (not bare `requestAnimationFrame`): rAF while
  the tab is visible (with a shadow-timer fallback covering a tab hidden
  after scheduling), `setTimeout(0)` when already hidden (rAF is suspended
  there and a queued view-state would otherwise stall until foregrounded),
  and synchronous re-entry when rAF is undefined (Vitest, Worker contexts).

## See also

- `../../scene-loader.ts` — the orchestrator that composes these helpers.
- `../view-state/view-state-queue.ts` — the pending slot the scheduler drives.
- `../progressive/refinement.ts` — the per-frame refinement loop whose
  cancellation check hands the lock to `PassScheduler.handOff`.
- `../process/` — Stage 1 producers of `StagedLinesCommit` /
  `StagedGSplatsCommit` (Points use `LoadedPointsData` directly).
- `../commit/commit-{points,lines,gsplats,mesh}-geometry.ts` —
  Stage 2 per-geometry commit bodies invoked through the ctx
  callbacks.
- `../../../profiling/update-profiler.ts` — `UpdateSession` /
  `SessionImpl.end()` idempotency contract.
