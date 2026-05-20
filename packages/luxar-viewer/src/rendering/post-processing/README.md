# Post-processing

Post-processing pipeline for the viewer. Built from a custom
mega-shader: tone mapping, bloom, anti-aliasing (FXAA / MSAA / SSAA),
detector noise, vignette, and chromatic lens distortion.

`PostProcessingManager` is the public API class at
`rendering/post-processing-manager.ts`. This folder contains its private
helpers: bloom chain, FXAA pass, fullscreen geometry, mega-shader,
HDR-capture helpers, and the focused modules under
`post-processing-manager/` for resource lifecycle, settings, pipeline,
and capture.

The pipeline fuses all per-pixel effects into a single fullscreen
fragment shader. Net effect: fewer fullscreen passes per frame, no
third-party dependency, and matching GLSL/TSL implementations for the
WebGL2 and WebGPU paths.

## Architecture

```
                         ┌────────────────────────┐
                         │  PostProcessingManager │
                         │  (public API, owns     │
                         │   lifecycle + state)   │
                         └─────────┬──────────────┘
                                   │
        ┌──────────────────────────┼────────────────────────┐
        ▼                          ▼                        ▼
┌────────────────┐        ┌────────────────┐        ┌──────────────┐
│  scene →       │        │  bloom pyramid │        │  MegaShader  │
│  hdrTarget     │        │  (BloomChain)  │        │  fullscreen  │
│  (HalfFloat,   │        │  threshold ↓   │        │  pass — all  │
│   optional     │        │  ↓ downsample  │        │  per-pixel   │
│   MSAA)        │        │  ↑ tent upsamp.│        │  effects     │
└────────┬───────┘        └────────┬───────┘        └──────┬───────┘
         │                         │                       │
         └─────────────────────────┴───────────────────────┘
                                   │
                                   ▼
                       ┌────────────────────┐
                       │  optional FXAA     │
                       │  (FxaaPass)        │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │  canvas / EXR      │
                       └────────────────────┘
```

The mega-shader fuses these steps in one fragment pass:

1. Chromatic lens distortion (per-channel sample at distorted UVs)
2. Additive bloom mix from the bloom texture
3. Detector noise (procedural per-pixel)
4. EOG — Exposure / Offset / Gamma
5. Tone mapping (THREE's `<tonemapping_pars_fragment>` chunk;
   Linear / Reinhard / Cineon / ACES / AgX / Neutral)
6. Vignette (multiplicative)
7. sRGB encoding (skipped when capturing for EXR)

Two `#define`-gated shortcuts cover the EXR/HDR-capture paths
(`LUXAR_CAPTURE_RAW_HDR` for pre-tone linear HDR,
`LUXAR_CAPTURE_LINEAR_LDR` for post-tone-mapped linear LDR).

## Module map

| File                         | Role                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `post-processing-manager.ts` | Public API: setters, lifecycle, capture paths, context-restore               |
| `mega-shader.glsl.ts`        | Fused fragment shader (vertex is a trivial fullscreen triangle)              |
| `mega-shader-material.ts`    | `ShaderMaterial` wrapper — uniform layout, `#define` toggles for each effect |
| `bloom-chain.ts`             | Threshold + downsample/upsample pyramid producing the bloom texture          |
| `fxaa-pass.ts`               | Inline FXAA on the LDR ldrTarget → backbuffer                                |
| `hdr-capture.ts`             | One helper — the EXR log-line formatter                                      |
| `hdr-pixel-utils.ts`         | HalfFloat ↔ Float32 conversion + vertical flip for `ImageData`               |
| `render-target-sizing.ts`    | DPR/SSAA-aware physical-pixel size helper                                    |

## Public surface

```typescript
import { PostProcessingManager } from '@/rendering';

const pp = new PostProcessingManager(renderer, scene, camera, { width, height }, () =>
  sceneManager.updateMaterialsForCurrentCamera()
);

pp.setBloomEnabled(true);
pp.updateBloomSettings(1.5, 0.5);
pp.setToneMapping(THREE.AgXToneMapping);
pp.updateExposure(0.5);
pp.setVignetteEnabled(true, 0.5, 0.6);
pp.setMSAAEnabled(true);
pp.setMSAASamples(4);
```

The optional `onResize` callback runs after every render-target
reallocation (resize, SSAA toggle, MSAA toggle, DPR change). The host
wires it to `SceneManager.updateMaterialsForCurrentCamera()` so the
scene materials' cached `pointSizeFactor` / `uResolution` uniforms
follow the new drawing-buffer dimensions.

### Capture paths

```typescript
// EXR with linear HDR (default — bloom kept, EOG / tone / vignette /
// detector noise / lens distortion bypassed):
const exr = await pp.captureHDRAsEXR();

// EXR with full pipeline but linear (post-tone-mapped, pre-sRGB):
const exr2 = await pp.captureHDRAsEXR({ mode: 'visible-ldr' });

// Raw scene HDR with no bloom and no mega-shader:
const raw = await pp.captureHDRAsEXR({ mode: 'raw-scene-hdr' });

// Display-ready ImageData (full pipeline, sRGB-encoded, FXAA if on):
const img = pp.renderToImageData();
```

### Deferred rebuild

```typescript
pp.withDeferredRebuild(() => {
  pp.updateBloomSettings(1.5, 0.5);
  pp.setVignetteEnabled(true);
  pp.setChromaticLensDistortionEnabled(true, -0.05, -0.05);
});
```

This API remains available for callers that batch setting changes. In
the mega-shader pipeline individual setters are cheap, so the
deferred-rebuild path is effectively a no-op pass-through. The
`try/finally` in `withDeferredRebuild` still protects the depth counter
against sub-setter throws.

## Context-restore protocol

WebGL contexts can be lost on tab switch, GPU driver crash, or
`WEBGL_lose_context.loseContext()`. The manager preserves its
**identity** across restore so cached references in `PickingSystem`,
`AnimationController`, and `RenderingControls` stay valid.

1. Snapshot every user-facing uniform / define / toggle from the
   current `MegaShaderMaterial` + bloom-chain state.
2. `disposeTransientResources()` tears down all GPU resources.
3. `initializeTransientResources()` rebuilds them fresh at the current
   physical size.
4. Re-apply the snapshot through the regular setter API.

The detector-noise wall-clock timestamp is also reset so the first
post-restore frame doesn't see a multi-second `dt` jump.

The E2E test `tests/e2e/context-restore.spec.ts` asserts both identity
preservation and a non-default exposure round-tripping across restore.

## Unsupported effects

Three effects are intentionally unsupported because they do not fit the
current renderer model:

- **SMAA** — 3-pass edge-detect → weight → blend; FXAA remains as the
  inline AA option.
- **Depth of Field** — needs depth-aware multi-pass blur.
- **Ambient Occlusion** — needs surface normals, which point / gsplat /
  line geometry do not provide.

There are no `RenderingSettings`, viewer-config, or UI fields for these
effects.

## Troubleshooting

- **Black canvas, console "redefinition" GLSL error** — a custom
  ShaderMaterial got `toneMapped = true` (THREE's default), so THREE
  injected `<tonemapping_pars_fragment>` on top of our explicit
  include. Every PP material must set `toneMapped: false`.
- **Brightness changes with DPR** — render targets allocated in
  logical pixels instead of physical. They MUST be sized as
  `effectiveSize × renderer.getPixelRatio()` (see `getPhysicalSize`)
  so they match what materials read from `getDrawingBufferSize`.
- **Point/line sizes feel off after toggling SSAA/MSAA** — the
  `onResize` callback isn't wired or the manager's caller forgot to
  pass it. Re-check the constructor call site.
- **Detector noise jumps after context restore** — the
  `_previousRenderTimestamp` field needs to be cleared in
  `rebuildAfterContextRestore()` (it is).

See `../README.md` for the rendering-pipeline overview and how the
post-processing stage fits into it.
