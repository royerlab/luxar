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

1. **Prediction** — `scene/dimension-loading.ts::updateAllNDNodes` fires after the
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
   silently fail without it; the ladder is stored once, at the end of the
   pass) and a per-batch `AbortSignal`. The batch deliberately PERSISTS across
   foreground ticks — a cold level outlives one frame — so a new `prefetch()`
   is a no-op while a batch runs; only its stall guard, `releaseShadows()`
   (playback end, setup.ts's non-playing branch) and `dispose()` abort it.
   `prefetchTargets(viewState, …, targets)` instead JOINS the running batch
   for the loaders at/under `targets`: the partition parts `prefetchSlice`
   activates for the predicted slice (B4) register only after the batch
   enumerated its nodes, so they would otherwise be warmed a tick late. One
   shadow pass per node runs at a time, whichever call queued it.
5. **In-flight adoption** — each shadow pass registers an in-flight store for
   its S-cache key (`beginShadowStore`); a foreground pass for the same key
   awaits it (`awaitShadowStore`: bounded, and rejects on the foreground's own
   abort) and then restores the shadow's ladder, instead of redoing the
   dequant/assembly for the same slice.
6. **Pins** — shadow stores are pinned until a consuming read. The prefetcher
   records every key its shadows may have pinned and unpins them all in
   `releaseShadows()` / `dispose()`, so no pin outlives playback.

## See Also

- [`../../loaders/progressive/slice-cache-helper.ts`](../../loaders/progressive/slice-cache-helper.ts)
  — the shared restore/store contract (upgrade-if-longer protects deeper
  foreground entries from shallow shadow prefixes).
- [`../../../cache/slice-cache.ts`](../../../cache/slice-cache.ts) — the
  S-cache itself, incl. scan-resistant (MRU-victim) eviction during playback.
- [`../view-state/derive-node-view-state.ts`](../view-state/derive-node-view-state.ts)
  — the same per-node derivation the handlers apply.
