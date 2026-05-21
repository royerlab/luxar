# Scene animation

Two animation drivers used by the viewer: the render-loop controller
that ticks every frame, and the dimension-animation manager that
auto-scrubs an nD slider through its range. They are independent
concerns wired together via `addPerFrameCallback` — the dimension
manager registers itself as one callback among many on the shared
render loop.

The wider scene orchestration lives in `../scene-manager.ts`; this
folder contains only the loop and the dimension scrubber.

## Files

| File                            | Role                                                                                                                                                                                                                                                                                                                       |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `animation-controller.ts`       | `requestAnimationFrame`-driven render loop. Updates `ControlsManager`, runs registered per-frame callbacks, then renders through `PostProcessingManager`. Emits `frame-start` / `frame-end` on the event bus (for the PerformanceMonitor panel) and auto-pauses after `config.animation.idleTimeoutMs` of inactivity unless something continuous is active. |
| `dimension-animation-manager.ts` | Per-dimension FPS-throttled scrubber with `once` / `loop` / `bounce` modes. Mutates `SceneDimsManager` state and awaits `waitForUpdate()` so animation never advances faster than data loading. Extends `THREE.EventDispatcher` — emits `play`, `pause`, `complete`, `directionChange`, `speedChange`, `loopModeChange`, `fpsWarning`.                                            |

## Public surface

**`AnimationController`**

- `startAnimation()` / `stopAnimation()` — start or pause the rAF loop. `startAnimation` is also the event-handler wired to controls, canvas input, and dimension changes; calling it on every interaction resets the idle timer.
- `addPerFrameCallback(id, fn, { continuous? })` / `removePerFrameCallback(id)` / `hasPerFrameCallback(id)` — register named callbacks executed after `controls.update()` but before `postProcessing.render()`. `continuous: true` keeps the loop alive past the idle timeout (used by dimension animation and turntable recording); the default `false` is on-demand (e.g. dynamic clipping, scale bar).
- `setAdaptiveDPRManager(manager)` — opt-in DPR feedback: the loop calls `recordFrame(now)` each frame so the manager can downshift pixel ratio under load.
- `setContextLostPredicate(predicate)` — injected by `SceneManager` to suppress GPU work while the WebGL context is lost; controls and callbacks still tick so input stays responsive.
- `get isActive` — true while the loop is running.
- `dispose()` — stops the loop and clears all per-frame callbacks. Not reusable after dispose.

**`DimensionAnimationManager`**

- `play(dimIndex, options?)` / `pause(dimIndex)` / `togglePlay(dimIndex, options?)` / `stop(dimIndex)` — control playback for one dimension. `stop` also removes the state entry; `pause` keeps it. `play` lazily calls `ensureRegistered()`, which installs the `'dimension-animation'` callback on the shared `AnimationController` with `continuous: true` and starts the loop.
- `setTargetFPS(dimIndex, fps)` / `increaseSpeed(dimIndex)` / `decreaseSpeed(dimIndex)` — set or step through the FPS presets (`config.dimensionAnimation.presets.fps`, default `[1, 2, 5, 10, 15, 30, 60]`); values clamp to `[customMin, customMax]`. The speed-step helpers snap to the nearest preset and fall back to ±10 % beyond the range.
- `setLoopMode(dimIndex, mode)` — `'once' | 'loop' | 'bounce'`. `setTargetFPS` and `setLoopMode` create a paused state entry if none exists, so the UI can pre-configure a dimension before the user hits play.
- `isAnimating(dimIndex)` / `getState(dimIndex)` — query playback flag and the full `DimensionAnimationState` (target/actual FPS, frame counters, direction).
- `dispose()` — pauses every dimension, clears state, removes the per-frame callback, and unsubscribes the `SceneDimsManager` listener.

`PlayOptions` (`targetFPS?`, `loopMode?`, `direction?`) is exported for
callers that build the options object dynamically; defaults come from
`config.dimensionAnimation.defaults`.

## Invariants

- **One render loop, many callbacks.** `AnimationController` is a
  singleton-style driver inside the app — multiple subsystems share it
  through `addPerFrameCallback` with unique IDs. Removing a callback
  by ID must not affect the others, so the storage is a `Map`, not an
  array.
- **Continuous vs on-demand callbacks.** Only callbacks registered
  with `continuous: true` count toward `shouldContinueAnimating()`.
  Auto-rotate, post-processing effects that self-declare via
  `needsContinuousAnimation()`, and continuous callbacks together
  decide whether idle timeout fires.
- **No GPU work while context is lost.** When the injected
  `isContextLost` predicate returns true, the loop emits
  `frame-end` and returns before `postProcessing.render()`. Controls
  and per-frame callbacks still execute, so UI input is unaffected
  during the brief loss window.
- **Animation never outruns data.** `DimensionAnimationManager`
  tracks `pendingUpdates` per dimension. After
  `setDimensionValue(...)` it awaits `sceneDimsManager.waitForUpdate()`
  before clearing the pending flag; until the flag clears, that
  dimension's frame work is skipped. Independent dimensions animate
  in parallel because the flag is per-index.
- **FPS warning is informational only.** Below
  `feedbackThreshold * targetFPS` the manager emits `fpsWarning` and
  optionally logs, but never throttles or stops the animation —
  callers decide what to do with the signal.
- **Boundary semantics.** `handleBoundary` clamps to `[min, max]`,
  not past them — `once` clamps and stops, `loop` wraps to the
  opposite end, `bounce` clamps and flips `state.direction`. The
  discrete-vs-continuous distinction lives in `calculateNextValue`:
  discrete dims advance by `metadata.step`, continuous dims by
  `range / continuousTraverseSeconds` scaled to the current target
  FPS.

## Events

`DimensionAnimationManager` extends `THREE.EventDispatcher` with the
`DimensionAnimationEvents` map from `src/types/animation.ts`:

| Event             | Payload                                                  | When                                          |
| ----------------- | -------------------------------------------------------- | --------------------------------------------- |
| `play`            | `{ dimIndex }`                                           | `play()` transitions a paused dim to playing  |
| `pause`           | `{ dimIndex }`                                           | `pause()` transitions a playing dim           |
| `complete`        | `{ dimIndex }`                                           | `once` mode hit the far boundary              |
| `directionChange` | `{ dimIndex, direction }`                                | `bounce` mode flipped at a boundary           |
| `speedChange`     | `{ dimIndex, fps }`                                      | `setTargetFPS` applied (after clamp)          |
| `loopModeChange`  | `{ dimIndex, loopMode }`                                 | `setLoopMode` applied                         |
| `fpsWarning`      | `{ dimIndex, targetFPS, actualFPS }`                     | Measured FPS fell below `feedbackThreshold`   |

`AnimationController` does not extend `EventDispatcher`; it publishes
`frame-start` and `frame-end` on `utils/cross-layer/event-bus` so the
`PerformanceMonitor` UI panel can subscribe without
animation-controller importing UI code.

## See also

- `../scene-manager.ts` — instantiates both classes and wires them
  into controls, post-processing, resize, and WebGL-context-loss
  recovery.
- `../scene-dims-manager.ts` — the dimension state that
  `DimensionAnimationManager` mutates via `setDimensionValue` and
  observes via `addListener` / `waitForUpdate`.
- `../../rendering/adaptive-dpr-manager.ts` — receives
  `recordFrame(now)` calls each frame when wired through
  `setAdaptiveDPRManager`.
- `../../config/sections/animation` and
  `../../config/sections/dimension-animation` — idle timeout, FPS
  presets, min-frame time, continuous-traverse seconds, feedback
  threshold.
- `../../types/animation.ts` — `DimensionAnimationState`,
  `LoopMode`, `AnimationDirection`, `DimensionAnimationEvents`.
- `../../utils/cross-layer/event-bus.ts` — `frame-start` / `frame-end` topics.
