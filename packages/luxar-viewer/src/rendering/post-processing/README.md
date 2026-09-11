# Post-processing

Post-processing pipeline for the viewer. Built from a custom
mega-shader: tone mapping, bloom, anti-aliasing (FXAA / MSAA / SSAA),
detector noise, vignette, and chromatic lens distortion.

`PostProcessingManager` is the public-API class at
`rendering/post-processing/post-processing-manager.ts`. This folder
contains its private helpers grouped by effect (`bloom/`, `fxaa/`,
`mega/`) plus shared infrastructure (`fullscreen/`, `hdr/`) and the
orchestrator-focused modules under `post-processing-manager/` for
resource lifecycle, settings, pipeline, and capture.

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

```
post-processing/
├── post-processing-manager.ts      # Public-API orchestrator class
├── post-processing-manager/        # Focused helpers behind the orchestrator
│   ├── capture.ts                  #   three capture paths + EXR encoding
│   ├── pipeline.ts                 #   per-frame composition (scene → bloom → mega → fxaa)
│   ├── resource-lifecycle.ts       #   sizing + GPU resource allocate/dispose
│   └── settings.ts                 #   user-toggle setter logic + validation
├── mega/                           # Fused tonemap / distortion / noise / vignette fragment
│   ├── material.ts                 #   ShaderMaterial wrapper (#define toggles)
│   ├── material-tsl.ts             #   WebGPU NodeMaterial counterpart
│   ├── shader.glsl.ts              #   GLSL3 vertex + fragment + MEGA_SOURCE
│   └── shader.tsl.ts               #   TSL/WebGPU factory + LuxarToneMappingMode
├── bloom/                          # Threshold + downsample/upsample pyramid
│   ├── chain.ts                    #   BloomChain class
│   ├── shaders.ts                  #   GLSL3 sources + three ShaderSource records
│   └── bloom.tsl.ts                #   TSL factories
├── fxaa/                           # Single-pass FXAA Quality
│   ├── pass.ts                     #   FxaaPass class
│   ├── shaders.ts                  #   GLSL3 source + FXAA_SOURCE
│   └── fxaa.tsl.ts                 #   TSL/WebGPU factory
├── fullscreen/                     # Shared fullscreen-rendering plumbing
│   ├── geometry.ts                 #   caps-aware fullscreen triangle
│   └── pass.ts                     #   FullscreenPass (scene+camera+mesh triplet)
├── hdr/                            # HDR readback + EXR-log
│   ├── pixel-utils.ts              #   unified WebGL2/WebGPU readPixelsCompactAsync
│   └── capture.ts                  #   formatHDRExrLogLine
└── render-target-sizing.ts         # SSAA/DPR allocation and framebuffer-limit clamping
```

The orchestrator lives at `post-processing/post-processing-manager.ts`
(file) next to its `post-processing-manager/` (folder) of private helpers
— P2 layout. Each per-effect subfolder is self-contained; intra-folder
imports never cross effect boundaries. Shared infrastructure
(`fullscreen/`, `hdr/`) sits in its own concern-named folder.

## Public surface

```typescript
import { PostProcessingManager } from '@/rendering';
import { createRendererCapabilities } from '@/rendering/renderer-capabilities';

const capabilities = createRendererCapabilities(renderer);
const pp = new PostProcessingManager(renderer, capabilities, scene, camera, { width, height }, () =>
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

SSAA multipliers above 1x suspend the configured MSAA samples to avoid
allocating redundant multisample renderbuffers at the supersampled size.

The optional `onResize` callback runs after every render-target
reallocation (resize, SSAA toggle, MSAA toggle, DPR change) and receives
the logical display size. The SceneManager factory restores the camera
projection from that size before refreshing the scene materials' cached
`pointSizeFactor` / `uResolution` uniforms, so SSAA's enlarged drawing
buffer cannot leak into the orbit projection.

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

The E2E test `src/tests/e2e/error-recovery.spec.ts` and the unit tests
`src/tests/unit/rendering/post-processing-manager-lifecycle.test.ts` and
`src/tests/unit/scene/scene-manager/render-pipeline/webgl-context-recovery.test.ts`
exercise this path, asserting identity preservation and uniform
round-tripping across restore.

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
  logical pixels instead of the shared physical allocation. They MUST use
  `getPhysicalSize`, which applies DPR rounding and framebuffer-limit clamping,
  so they match the canvas and what materials read from `getDrawingBufferSize`.
- **Point/line sizes feel off after toggling SSAA/MSAA** — the
  `onResize` callback isn't wired or the manager's caller forgot to
  pass it. Re-check the constructor call site.
- **Detector noise jumps after context restore** — the
  `_previousRenderTimestamp` field needs to be cleared in
  `rebuildAfterContextRestore()` (it is).

## Dependencies

- Internal: `rendering/materials/_shared` (shader-source / buildMaterial),
  `rendering/renderer-capabilities`, `utils/log`, `utils/clamp`.
- External: `three`, `three/tsl`, `three/webgpu`,
  `three/examples/jsm/exporters/EXRExporter.js`.

See `../README.md` for the rendering-pipeline overview and how the
post-processing stage fits into it.
