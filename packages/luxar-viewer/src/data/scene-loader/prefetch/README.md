# Prefetch

Background **t+1 slice prefetch** for dimension playback: while a dimension is
playing, the idle window between a tick's commit and the next tick (refinement
is suppressed under a frame budget) is used to build the NEXT timepoint's
decoded ladder into the shared **SliceCache**, so the next real tick's
`restoreLadder` hits instantly and its whole frame budget goes to _deepening_
the prefix instead of rebuilding it.

## File Structure

```
prefetch/
└── slice-prefetcher.ts   # SlicePrefetcher — shadow-loader t+1 warm pass
```

## How it works

1. **Prediction** — `input/.../setup.ts::updateAllNDNodes` fires after the
   awaited foreground pass (which resolves at COMMIT — the pass-waiter
   contract). It peeks the next value per playing dimension via
   `DimensionAnimationManager.peekNextValue` (pure
   `scene/animation/advance-value.ts`), so the prediction is loop/bounce/
   backward aware — including the loop wrap (t=max → t=min) that the
   linear-extrapolation chunk prefetch cannot predict.
2. **Routing** — `zarr-loader.prefetchSceneForDimensions` (mirror of
   `updateSceneForDimensions`, fire-and-forget) →
   `SceneLoader.prefetchSlice(viewState, budgetMs)`, which merges onto a
   **copy** of the persistent view state (a prefetch must never move the real
   view).
3. **Shadow loaders** — `SlicePrefetcher` lazily builds a second loader per
   registered node from the same `loader-factory` helpers (plain or
   progressive by `n_additive_sublods`). Foreground loader instances cannot be
   reused: they hold a reused accumulator, `_activeSignal`, and progressive
   ladder state, and `SceneLoader.updateView` is single-flight. Shadows share
   **nothing mutable** with the foreground — the S-cache is the only handoff.
   They are deliberately NOT monitor-connected (no metric double-counting).
4. **Budget + abort** — every shadow pass carries `frameBudgetMs` (which also
   makes the progressive loaders store _prefix_ ladders — the handoff would
   silently fail without it) and a per-pass `AbortSignal`.
   `SceneLoader.updateView` calls `abortInFlight()` at its very top, so the
   foreground always preempts. `releaseShadows()` frees the shadow
   accumulators when playback ends (setup.ts's non-playing branch).

## See Also

- [`../../loaders/progressive/slice-cache-helper.ts`](../../loaders/progressive/slice-cache-helper.ts)
  — the shared restore/store contract (upgrade-if-longer protects deeper
  foreground entries from shallow shadow prefixes).
- [`../../../cache/slice-cache.ts`](../../../cache/slice-cache.ts) — the
  S-cache itself, incl. scan-resistant (MRU-victim) eviction during playback.
- [`../view-state/derive-node-view-state.ts`](../view-state/derive-node-view-state.ts)
  — the same per-node derivation the handlers apply.
