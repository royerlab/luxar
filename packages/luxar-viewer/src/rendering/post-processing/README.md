# Post-processing

WebGL post-processing pipeline for the viewer. Built from a custom
mega-shader: tone mapping, bloom, anti-aliasing (FXAA / MSAA / SSAA),
detector noise, vignette, and chromatic lens distortion.

This module previously sat on top of `pmndrs/postprocessing` and its
`EffectComposer`. That was replaced with a hand-written three-stage
pipeline that fuses all per-pixel effects into a single fullscreen
fragment shader. Net effect: fewer fullscreen passes per frame, no
third-party dependency, easier path to WebGPU/TSL later.

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

This API is kept for source-compatibility with the old pmndrs era. In
the mega-shader pipeline individual setters are cheap, so the
deferred-rebuild path is effectively a no-op pass-through. The
`try/finally` in `withDeferredRebuild` still protects the depth
counter against sub-setter throws.

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

## Dropped features

The mega-shader refactor explicitly removed three effects that don't
fit a single-pass model:

- **SMAA** — 3-pass edge-detect → weight → blend; can't fuse cleanly.
  FXAA remains as the inline AA option.
- **Depth of Field** — needs depth-aware multi-pass blur.
- **Ambient Occlusion** — needs surface normals which point / gsplat /
  line geometry don't provide. The previous SSAO output was always
  degenerate for our scenes.

The corresponding `RenderingSettings` fields, viewer-config keys, and
UI controls were removed in the same change.

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

See `SPECIFICATIONS.md` for the per-effect math and the wider
operation-ordering invariants.
