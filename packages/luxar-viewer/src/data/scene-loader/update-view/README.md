# `updateView` orchestration helpers

Helpers extracted from `data/scene-loader.ts::updateView`. The orchestrator
itself stays in `data/scene-loader.ts`; this folder owns the three
non-trivial sub-steps that would otherwise bloat the method: per-type
ctx construction (entry), atomic GPU commit (Stage 2), and the
`finally`-phase dispatch of whatever runs next.

## Files

| File                   | Role                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `build-update-ctxs.ts` | Constructs the three per-type handler ctx objects (`PointsHandlerCtx`, `LinesHandlerCtx`, `GSplatsHandlerCtx`) that feed `data/{points,lines,gsplats}/handler.ts::loadAndStage`. Centralises the shared fields (rootGroup, viewStateQueue, clearFailure, currentVersion, deriveNodeViewState) plus the type-specific bits (`updateVersion` for Lines/GSplats, `extendedToleranceCache` shared between Points + GSplats). |
| `atomic-commit.ts`     | Runs the synchronous Stage 2 commit. Takes the three per-type staged-commit arrays from Stage 1 and routes them through `updatePointsGeometry` / `commitLinesGeometry` / `commitGSplatsGeometry`, bumps the GPU buffer pool frame counter, and invalidates the picking cache. The body is a single synchronous block so every participating mesh updates in the SAME rendered frame.                                     |
| `queue-next.ts`        | `finally`-phase dispatcher. Inspects the `ViewStateQueue` and progressive-GSplats loaders to decide between three outcomes: rAF-yield then re-enter `updateView(pending)`, kick GSplats LOD refinement, or release the `_updateInProgress` lock. The lock-keep semantics in the first two branches are load-bearing for slider/animation correctness.                                                                    |

## Public surface

Each file exports exactly one function (plus its ctx interface). None
of these are re-exported from `data/` — only `data/scene-loader.ts`
consumes them.

- `buildUpdateCtxs(input: UpdateCtxsInput): { pointsCtx, linesCtx, gsplatsCtx }`
- `runAtomicCommit(pointsStaged, linesStaged, gsplatsStaged, ctx): void`
- `queueNext(ctx: QueueNextCtx): void`

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
- **`queueNext` branch order matters.** Pending view-state takes
  priority over GSplats LOD refinement: a slider event mid-refinement
  must cancel the refinement and start the new update. `queueNext`
  encodes this by checking `takePending()` first; the refinement
  branch only runs when no pending state exists.
- **Lock-keep during rAF yield and refinement.** Branches 1 (pending)
  and 2 (refinement) keep `_updateInProgress = true` so slider events
  fired during the yield/refinement window queue as `_pendingViewState`
  rather than racing into a concurrent `updateView` call. Branch 3
  (idle) is the only one that releases the lock.
- **Frame scheduling is hidden-tab-proof.** Branch 1 yields through
  `utils/schedule-frame.ts` (not bare `requestAnimationFrame`): rAF while
  the tab is visible (with a shadow-timer fallback covering a tab hidden
  after scheduling), `setTimeout(0)` when already hidden (rAF is suspended
  there and a queued view-state would otherwise stall until foregrounded),
  and synchronous re-entry when rAF is undefined (Vitest, Worker contexts).

## See also

- `../../scene-loader.ts` — the orchestrator that composes these three
  helpers; their ctx interfaces are shaped to match its private state.
- `../view-state/view-state-queue.ts` — `takePending()` source consumed by
  `queueNext`.
- `../process/` — Stage 1 producers of `StagedLinesCommit` /
  `StagedGSplatsCommit` (Points use `LoadedPointsData` directly).
- `../commit/commit-{points,lines,gsplats}-geometry.ts` —
  Stage 2 per-geometry commit bodies invoked through the ctx
  callbacks.
- `../../points/handler.ts`, `../../lines/handler.ts`,
  `../../gsplats/handler.ts` — `loadAndStage` consumers of the
  per-type ctx objects built here.
- `../../../profiling/update-profiler.ts` — `UpdateSession` /
  `SessionImpl.end()` idempotency contract.
