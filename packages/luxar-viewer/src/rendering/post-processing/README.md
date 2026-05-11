# Post-processing

WebGL post-processing pipeline for the viewer: tone mapping, bloom,
ambient occlusion, anti-aliasing (FXAA / SMAA / MSAA / SSAA), depth of
field, detector noise, vignette, and chromatic lens distortion.

The implementation builds on
[`postprocessing`](https://pmndrs.github.io/postprocessing/) (pmndrs)
and adds Luxar-specific effects, lifecycle discipline, and HDR-aware
state capture/restore.

## Architecture

```
┌─────────────────┐    ┌──────────────────────────────────────┐
│  PostProcessing │───▶│  EffectComposer                      │
│  Manager        │    │  ├─ RenderPass (scene → texture)     │
│  (public API)   │    │  ├─ EffectPass(es) (effect chain)    │
└─────────────────┘    │  └─ SMAAPass / FXAAPass / output     │
                       └──────────────────────────────────────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │  Canvas / HDR  │
                              └────────────────┘
```

`PostProcessingManager` is the single owner of the composer + effect
graph. Every other module (`scene/`, `ui/rendering-controls/`,
`ui/recording-panel`) routes through its public API and never touches
the composer directly.

## Key modules

| File | Role |
|------|------|
| `post-processing-manager.ts` | Lifecycle owner; composer creation, effect graph, durable state capture/apply, deferred rebuild, dispose, context-restore |
| `effect-orchestrator.ts` | Pure helper that decides which effect goes into which pass (Pass A pre-tonemap, Pass B post-tonemap) |
| `effect-disposal.ts` | `safeDisposeEffect`, `safeRemoveAndDisposePass` — try/catch wrappers for pmndrs's event-driven disposal |
| `context-recovery.ts` | Capture / re-apply durable settings across WebGL context loss |
| `hdr-capture.ts`, `hdr-pixel-utils.ts` | Render-target readback for video/image recording |
| `bloom-handler.ts` | Bloom levels / radius / threshold / strength |
| `antialiasing-handler.ts` | FXAA / SMAA / MSAA / SSAA configuration |
| `tone-mapping-handler.ts`, `luxar-tone-mapping-effect.ts`, `tone-mapping-mode-names.ts` | HDR tone-mapping pipeline (Neutral / ACES / AGX / Linear / etc.) |
| `chromatic-lens-distortion-effect.ts` | Lens distortion + chromatic dispersion |
| `detector-noise-effect.ts` | Physics-based detector noise (Poisson + Gaussian + FPN) |
| `robust-vignette-effect.ts` | Vignette with darkness + offset controls |
| `visual-effects-handler.ts` | DOF focus / strength + DPR-scaled noise math |
| `postprocessing-types.ts` | Shared effect-state types + clamps + numeric guards |
| `render-target-sizing.ts` | DPR/MSAA-aware render-target sizing |

## Public surface

The entry point is the [`PostProcessingManager`](./post-processing-manager.ts)
class. It exposes setters for every effect, a quality-preset switcher
(`setQualityPreset('low' | 'medium' | 'high' | 'ultra')`), and the
deferred-rebuild contract (see below).

```typescript
import { PostProcessingManager } from '@/rendering';

const pp = new PostProcessingManager(renderer, scene, camera, dpr);
pp.setBloomEnabled(true);
pp.updateBloomSettings(1.5, 0.5);
pp.setQualityPreset('high'); // batches its sub-setters internally
```

## Deferred-rebuild contract

The effect graph is **expensive to rebuild** (effect construction,
shader compilation, render-target allocation). Bulk setting updates
should batch through the deferred-rebuild API so the composer rebuilds
once at the end of the batch.

The recommended form is the closure helper:

```typescript
pp.withDeferredRebuild(() => {
  pp.updateBloomSettings(1.5, 0.5);
  pp.setVignetteEnabled(true);
  pp.setChromaticLensDistortionEnabled(true, -0.05, -0.05);
});
```

`withDeferredRebuild` wraps a `start/end` pair in `try/finally`, so a
thrown sub-setter cannot strand the depth counter above zero (which
would silently disable all future rebuilds — see SPECIFICATIONS for
the depth-counter semantics).

The lower-level `startDeferRebuild()` / `endDeferRebuild()` calls
remain available and are nestable (depth-counter), but should be
used only when a closure boundary is inconvenient.

## Context-restore protocol

WebGL contexts can be lost on tab switch, GPU driver crash, or
deliberate `WEBGL_lose_context.loseContext()`. The manager preserves
its **identity** across restore so cached references in `PickingSystem`,
`AnimationController`, and `RenderingControls` stay valid.

1. `captureDurableState()` snapshots every user-facing setting to a
   plain object (exposure / bloom / DOF / vignette / detector noise /
   chromatic lens / etc.).
2. `disposeTransientResources()` tears down the composer + passes +
   effects.
3. `initializeTransientResources()` rebuilds them fresh.
4. `applyDurableState()` reinstates the snapshot.

The E2E test `tests/e2e/context-restore.spec.ts` asserts both identity
preservation and a non-default exposure round-tripping across restore.

## Lifecycle

`dispose()` is **idempotent** (`this.disposed` guard). It walks the
effect list with `safeDisposeEffect` per effect (try/catch wrapped),
removes passes from the composer with `safeRemoveAndDisposePass`, and
finally disposes the composer itself.

Per-effect `safe*` wrappers exist because pmndrs's event-driven
disposal can throw on a partially-initialized effect (e.g. when a
prior construction failed mid-way). A throw in one effect must not
prevent the rest from being cleaned up.

## Troubleshooting

- **Effects don't update after a bulk change**: a `startDeferRebuild`
  somewhere never balanced with `endDeferRebuild` (depth stuck > 0).
  Switch to `withDeferredRebuild(fn)` and the counter unwinds on
  throw.
- **Black screen after settings change**: an effect construction
  failed; check the console for the `safeDisposeEffect` warning that
  identifies which effect.
- **Wrong colors after tone-mapping mode swap**: the manager wires
  shader-output-mode + tone-mapping-mode together; mode swaps must go
  through `setToneMapping(mode)`, not direct uniform writes.
- **Lost settings after tab restore**: durable-state capture/apply
  covers only documented settings. Custom uniforms you set externally
  must be re-applied by your own context-restore hook.

See `SPECIFICATIONS.md` for the algorithms behind tone mapping, bloom
mipmap math, anti-aliasing trade-offs, and the deferred-rebuild
depth-counter semantics.
