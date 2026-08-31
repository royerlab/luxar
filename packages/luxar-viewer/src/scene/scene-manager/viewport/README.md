# Viewport sizing

Canvas-viewport helpers owned by `SceneManager`: pixel-ratio (DPR)
policy and the rAF-coalesced window-resize loop. Together they keep
the renderer, camera aspect, post-processing chain, and DPR-dependent
material/effect uniforms consistent across window resizes,
monitor-DPI changes, and adaptive/manual DPR overrides.

`SceneManager` instantiates one `ResizeOrchestrator` for its lifetime
and delegates `updateSize()` to it; the DPR policy is a set of pure
functions called from both the orchestrator and from
`SceneManager.setAdaptivePixelRatio` / `AdaptiveDPRManager`.

## Files

| File                     | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dpr-policy.ts`          | DPR functions, stateless apart from the pixel-ratio CAP they consult (owned by `rendering/pixel-ratio-cap.ts`; high DPR is opt-in, so the CEILING is `min(window.devicePixelRatio, cap)` and usually 1.0). `getActivePixelRatio(override)` resolves the currently-applied DPR — the override or, when it is `null`, the live ceiling — clamped to that ceiling either way, which is what makes this the seam the cap is enforced at. `computePixelRatioOverride(dpr)` clamps to the ceiling, snaps to it within 0.01 (returns `override: null` so future monitor-DPI changes keep tracking it) and rejects non-finite / non-positive inputs. `getNormalizedDPRScale(activeDPR)` returns `activeDPR / ceiling` for perceptually-constant DPR-dependent effects — against the ceiling, not the display, so a capped session does not read as permanently degraded. `syncPostProcessingDPRScale(pp, override)` pushes that scale into the post-processing pipeline, where detector-noise grain is its one consumer. |
| `resize-orchestrator.ts` | `ResizeOrchestrator` class owning the rAF-coalescing window-resize loop. `scheduleResize(getCtx)` reads `window.innerWidth`/`innerHeight` synchronously, cancels any in-flight rAF, and schedules a single `doResize` (a burst of `ResizeObserver` fires becomes one renderer resize). `resizeNow(w, h, ctx)` applies a resize immediately for the adaptive-DPR path which already runs at frame-level granularity. `resizeLocked` suppresses scheduling while a recording is in progress (resolution must stay locked). `dispose()` cancels in-flight rAF and clears pending state.                                                                                                                                                                                                                                                                                                                                                                                                                             |

## ResizeCtx contract

`ResizeOrchestrator` is constructed once but reads its renderer /
camera / post-processing references on every resize via a caller-
supplied `ResizeCtx`. The fields change over a `SceneManager`'s
lifetime (camera type swap, post-processing late init), so threading
them through `getCtx()` keeps the orchestrator in sync without
re-construction.

```typescript
interface ResizeCtx {
  readonly renderer: Renderer; // WebGL or WebGPU
  readonly camera: LuxarCamera | null; // null pre-init
  readonly postProcessing: PostProcessingManager | null;
  readonly pixelRatioOverride: number | null; // current adaptive/manual DPR
  /** Refresh world-space point sizing uniforms after each resize. */
  updateMaterialsForCurrentCamera(): void;
}
```

## Resize pipeline

For each resize (rAF-coalesced or immediate), `doResize` runs:

1. Log the new dimensions (fullscreen vs windowed).
2. If `camera` exists, call `updateCameraAspect(camera, w, h)`.
3. Call `renderer.setPixelRatio(getActivePixelRatio(override))` — when
   an adaptive/manual DPR override is active it is preserved across
   ordinary window resizes; when no override is active, native
   `devicePixelRatio` changes (e.g. dragging between monitors) are
   tracked automatically.
4. If `postProcessing` is initialised, delegate to
   `postProcessing.resize(w, h)` (which calls `renderer.setSize` and
   `composer.setSize` internally) and then
   `syncPostProcessingDPRScale` so DPR-dependent effects refresh.
   Otherwise fall back to direct `renderer.setPixelRatio` +
   `renderer.setSize` (early-init path before PostProcessing exists).
5. If `camera` exists, call `ctx.updateMaterialsForCurrentCamera()` to
   refresh per-frame uniforms (e.g. world-space point sizing).

## Invariants

- **DPR policy is stateless.** All four functions in `dpr-policy.ts`
  take their inputs explicitly (override value, postProcessing
  reference) so they can be unit-tested without a `SceneManager`.
- **Snap-to-native clears the override.** `computePixelRatioOverride`
  returns `override: null` when the requested DPR is within `0.01` of
  `window.devicePixelRatio`, so future monitor-DPI changes continue to
  track natively rather than getting pinned to the last explicit value.
- **rAF coalescing for window resize, immediate for adaptive DPR.**
  `scheduleResize` exists because `ResizeObserver` / browser resize
  events can fire in bursts; `resizeNow` exists because the adaptive-
  DPR controller already runs at frame granularity and would just
  double-buffer its own work through another rAF.
- **`resizeLocked` is honoured by `scheduleResize` only.** A recording
  in progress must keep its resolution stable, but `resizeNow` is
  reserved for the controller-driven adaptive-DPR path which itself
  knows when not to fire — locking is enforced at the scheduling
  boundary, not the apply boundary.
- **PostProcessing owns renderer + composer sizing when present.** The
  direct `renderer.setSize` path is reserved for the brief window
  before `PostProcessingManager` is constructed. Once it exists, all
  size writes flow through it.
- **`getNormalizedDPRScale` is relative, not absolute.** It returns
  `activeDPR / nativeDPR`, not `activeDPR` itself, because effects
  like grain density need to stay perceptually constant across DPR
  overrides — what matters is the ratio to native, not the raw value.

## See also

- `../../scene-manager.ts` — the `SceneManager` owns the orchestrator
  instance, supplies the `getCtx()` callback, and routes
  `updateSize()` through it.
- `../../../rendering/post-processing/post-processing-manager.ts` —
  `PostProcessingManager.resize` and `setDPRScale`, the two sinks for
  the resize pipeline.
- `../../../utils/camera-utils.ts` — `updateCameraAspect`, called from
  `doResize` for both perspective and ortho `LuxarCamera` variants.
- `../../../rendering/renderer-capabilities.ts` — the `Renderer`
  abstraction (WebGL or WebGPU) used by `ResizeCtx`.
- `../../animation/animation-controller.ts` and the `AdaptiveDPR`
  manager — call `setAdaptivePixelRatio` on `SceneManager`, which
  composes `computePixelRatioOverride` + `resizeNow` from this folder.
