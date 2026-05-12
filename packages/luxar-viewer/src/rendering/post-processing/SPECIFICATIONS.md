# Post-processing Specifications

Algorithms, data structures, and invariants for the
`rendering/post-processing/` subsystem. The
[README](./README.md) covers usage; this document covers the
machinery.

## Pipeline architecture

```
RenderPass (scene → texture in linear HDR space)
       │
       ▼
EffectPass A (pre-tonemap)  → bloom, AO, vignette,
                              chromatic lens, detector noise, DOF
       │
       ▼
EffectPass B (post-tonemap) → tone mapping, FXAA
       │
       ▼
SMAAPass / MSAA-aware target / SSAA-resolved output
       │
       ▼
Canvas / HDR capture target
```

### Why two effect passes

pmndrs effects compose by chaining fragment passes through a single
ping-pong target. Bloom + AO + vignette need to see **linear HDR**
input; tone mapping converts to a display-ready signal. Splitting
into Pass A (HDR) and Pass B (LDR) keeps the math straight: tone
mapping happens once at a defined boundary, not implicitly mixed with
spatial filters.

`effect-orchestrator.ts` decides which effect goes into which pass.
Bloom/AO/vignette/etc. always go to Pass A; tone mapping is always
the first effect in Pass B; FXAA is always the last effect in Pass B
(after tonemap, before display).

## Tone mapping

Implemented in `luxar-tone-mapping-effect.ts`. Supported modes:

| Mode         | Curve                         | Use                                            |
| ------------ | ----------------------------- | ---------------------------------------------- |
| `Linear`     | identity, clipped to `[0, 1]` | reference / debugging                          |
| `Neutral`    | THREE's neutral curve         | scientific data (hue-preserving) — **default** |
| `ACESFilmic` | ACES filmic                   | cinematic, increases contrast                  |
| `AgX`        | AgX tone curve                | photographic, well-behaved highlights          |
| `Reinhard`   | x / (1 + x)                   | conservative, low-contrast                     |
| `Cineon`     | logarithmic film curve        | filmic / wide gamut                            |

Each mode has a `shaderOutputMode` hint (`alpha-weighted` /
`rgb-contribution` / `opaque`) consumed by the underlying material
shaders so RGB composition stays correct (e.g. `max` blending requires
premultiplied RGB; tonemap RGB needs unweighted).

The exposure uniform is applied **before** the tone curve:
`tonemapped = curve(exposure × radiance + offset)^gamma`. `offset` and
`gamma` are LDR adjustments applied post-curve.

## Bloom

Bloom uses pmndrs's `BloomEffect`. The `levels` parameter (1-12)
controls the number of mipmap levels of the bright-pass texture; more
levels = larger / smoother glow, at the cost of one extra blur pass
per level. Defaults: `low=3`, `medium=6`, `high=8`, `ultra=10`.

`threshold` is the luminance above which a pixel contributes to bloom.
`strength` is the additive intensity at composite time. `radius`
controls the Kawase-style upsample radius.

Bloom levels swap **in-place** when changed (the old bloom effect is
disposed AFTER the new one is constructed and wired, to avoid a brief
use-after-free window — see `effect-disposal.ts`).

## Anti-aliasing

Four modes, applied at different pipeline stages:

| Mode     | Stage                                 | Cost                          | Quality                                               |
| -------- | ------------------------------------- | ----------------------------- | ----------------------------------------------------- |
| **FXAA** | Pass B (post-tonemap)                 | very cheap                    | medium; blurs subpixel detail                         |
| **SMAA** | Dedicated SMAAPass after Pass B       | low-medium                    | very good edges; preset-tuned (LOW/MEDIUM/HIGH/ULTRA) |
| **MSAA** | Hardware multisample on render target | medium; GPU-feature dependent | great geometry; can't AA shader-introduced edges      |
| **SSAA** | Supersample target sized × multiplier | very high (squared cost)      | best; brute-force                                     |

These are independent — multiple can stack (e.g. MSAA for geometry +
FXAA for shader edges), at the cost of compounding GPU time.

SMAA settings (`smaaThreshold`, `smaaSearchSteps`) tune detection
sensitivity and pattern search depth. `updateSMAASettings('PRESET')`
maps a string preset to those parameters.

## Deferred rebuild (depth-counter semantics)

Effect-pass rebuilds are expensive: pass reconstruction, effect
re-attachment, shader recompilation, render-target re-sizing. Bulk
setting updates batch through:

```typescript
this.deferRebuildDepth = 0;  // counter, not a boolean
private get deferRebuild(): boolean { return this.deferRebuildDepth > 0; }
```

- `startDeferRebuild()` increments depth.
- `endDeferRebuild()` decrements depth; rebuilds only when depth
  reaches 0.
- Nested `start/end` pairs are supported and only the outermost
  `end` triggers the rebuild.

### Why a counter and not a boolean

A boolean `deferRebuild = true/false` flag has a fragile failure
mode: if a setter throws between `start` and `end`, the boolean
stays `true` until the next `end` runs — which never happens unless
the caller has a try/finally. The depth counter fails the same way
under the same bug, but `withDeferredRebuild(fn)` (a closure helper)
wraps `start/end` in `try/finally` so the depth always unwinds.
**Always prefer `withDeferredRebuild`** at call sites; the
lower-level start/end remain only for cases where a closure boundary
is inconvenient.

## Context-restore protocol

WebGL contexts can be lost — tab switch, driver crash, deliberate
`WEBGL_lose_context.loseContext()`. The manager preserves its
**identity** across restore so cached references in consumer modules
(`PickingSystem`, `AnimationController`, `RenderingControls`,
`RecordingPanel`) stay valid.

Sequence:

1. `captureDurableState()` snapshots every user-facing setting into a
   plain serializable object: `{ exposure, offset, gamma, bloom: {…},
dof: {…}, vignette: {…}, detectorNoise: {…}, chromaticLens: {…},
aa: {…}, toneMapping: {…} }`.
2. `disposeTransientResources()` tears down composer + passes +
   effects through the `safe*` helpers.
3. `initializeTransientResources()` rebuilds composer + render pass +
   empty effect graph.
4. `applyDurableState(snapshot)` reinstates every captured setting
   (which re-creates effects as needed, via the same setters that
   handle first-time creation).

The E2E lock-in lives in `tests/e2e/context-restore.spec.ts`. Unit
tests in `tests/unit/rendering/post-processing/context-recovery.test.ts`
exercise capture/apply on plain objects without a WebGL context.

## Disposal invariants

`dispose()` MUST be idempotent. Guard with `this.disposed = false`
field, return early on second call. Order:

1. Effect-level `safeDisposeEffect` on every owned effect (try/catch
   per call). pmndrs's event-driven disposal can throw on partially-
   constructed effects.
2. `safeRemoveAndDisposePass` for each pass.
3. `this.composer.dispose()` wrapped in a try/catch so a thrown
   composer dispose (rare; possible on mid-teardown context loss)
   doesn't strand the remaining cleanup.
4. Null out instance fields.

## Render-target sizing

`render-target-sizing.ts` computes the effective render-target size:
`drawingBuffer = canvasSize × DPR × ssaaMultiplier`. MSAA samples are
applied as a property of the GL render target itself (not size-
multiplying), so MSAA × SSAA stacks multiplicatively in cost but
additively in dimensions.

DPR is **clamped** through `adaptive-dpr-manager.ts` so a 4K monitor
under DPR=2 (effectively 5K render) doesn't blow GPU memory.

## HDR capture

`hdr-capture.ts` reads back the pre-tonemap HDR texture for video /
image recording. The capture path bypasses tone-mapping pass so
external tools (e.g. ffmpeg with HDR10 metadata) can apply their own
curve.

`hdr-pixel-utils.ts` provides the float-to-half-float packing for
EXR-style file formats.

## Performance targets

- 1M points @ 1080p: ≤ 5ms base render pass; effects add ~1-3ms
  each on integrated GPUs, ~0.5-1ms on dedicated.
- Bloom at 8 levels: ~1.5ms @ 1080p on dedicated GPU.
- SMAA HIGH: ~1ms @ 1080p.
- SSAA × 2: ~3-4× base cost; reserve for stills/screenshots.

These are guidance; profile on the target GPU before locking in
quality presets.
