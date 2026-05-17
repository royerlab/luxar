# Post-processing Specifications

Algorithms, data structures, and invariants for the
`rendering/post-processing/` subsystem. The
[README](./README.md) covers usage and the module map; this document
covers the machinery.

## Pipeline architecture

```
scene (linear HDR materials)
       │
       ▼
scene render → hdrTarget (HalfFloat, optional MSAA)
       │
       ├─────────────────────────────────────────────┐
       │                                             │
       ▼                                             │
BloomChain (only when bloom is enabled)              │
   threshold + 2× downsample → mip[0]                │
   downsample chain          mip[i]   → mip[i+1]     │
   upsample chain   mip[i+1] → mip[i] (additive)     │
   output = mip[0] = bloom texture                   │
       │                                             │
       │   ┌─────────────────────────────────────────┘
       │   │
       ▼   ▼
MegaShader fullscreen pass — fragment shader fuses:
   1. lens distortion (per-channel sample @ distorted UVs)
   2. bloom mix       (sample bloom texture, additive)
   3. detector noise  (procedural per-pixel)
   4. EOG             (exposure / offset / gamma)
   5. tone mapping    (Linear / Reinhard / Cineon / ACES / AgX / Neutral)
   6. vignette        (multiplicative)
   7. sRGB encoding   (skipped for capture paths)
       │
       ▼
ldrTarget (only allocated when FXAA enabled OR for capture)
       │
       ▼
FxaaPass (only when fxaaEnabled)
       │
       ▼
canvas backbuffer
```

### Why a fused mega-shader

The old pmndrs pipeline ran one full-screen pass per effect, with a
ping-pong target pair between them. For the per-pixel effects we
care about (chromatic lens distortion, detector noise, EOG, tone
mapping, vignette, sRGB encode) there's no inter-pixel dependency —
they read one pixel, write one pixel. Folding them into a single
fragment shader means the GPU does one rasterization, one set of
texture binds, one program switch, and reuses the linear-HDR sample
all the way through.

Bloom is kept as a separate pre-pass because it needs neighbor reads
(threshold + multi-tap downsample / upsample). FXAA is kept as a
separate post-pass because its edge detection needs neighbor reads
of the already-tone-mapped LDR output.

## Operation ordering

The order of operations in the mega-shader mirrors the canonical
order the old pmndrs `EffectComposer` ran through:

```
ChromaticLensDistortion → Bloom (additive) → DetectorNoise →
ToneMapping (with EOG) → Vignette → AA → sRGB encode
```

Critically, chromatic distortion samples the scene **after** bloom
was additively blended in. In the fused shader this is preserved by
having `sampleHdrPlusBloom(uv)` return `texture(uHdrScene, uv) +
texture(uBloomTexture, uv) * uBloomIntensity` — so a per-channel
distorted sample picks up bloom at the same chromatically-aberrated
UV, exactly as the old chain produced.

## Y-orientation contract

WebGL2 and WebGPU disagree about the framebuffer's memory layout. In
WebGL2 framebuffer row 0 is the **bottom** of the viewport; in real
WebGPU it is the **top**. Three.js's `WebGPURenderer` normalises Y on
both its native-WebGPU backend and its WebGL2 compat backend (used
by `forceWebGL: true` and the natural fallback) so it always presents
a top-down framebuffer to user code. The legacy `THREE.WebGLRenderer`
GLSL path is the only renderer that exposes a bottom-up framebuffer.
`caps.framebufferYDown` is the canonical discriminator — `true` for
every `WebGPURenderer`, `false` only for `WebGLRenderer`.

The viewer canonicalises Y handling in two — and only two — seams:

1. **`createFullscreenTriangleGeometry(caps)`** emits a UV attribute
   that compensates the active backend so that a passthrough sample
   `texture(src, uv)` at a fragment under NDC (-1, -1) reads the
   _bottom-left_ texel of the source target on both backends. The
   five fullscreen passes (mega-shader, bloom threshold / downsample
   / upsample, FXAA) consume this attribute through the shared
   `FullscreenPass` class. GLSL3 vertex shaders read `vUv = uv;` and
   TSL fragment factories read `uv()` — both routes land on the same
   caps-aware attribute, so neither path has a backend branch in its
   body.

2. **`readPixelsCompactAsync(renderer, caps, opts)`** is the only
   render-target readback API in the viewer. It accepts `(x, y)` in
   canonical **top-down** coordinates (row 0 = top of the source
   target) and returns pixels in the same top-down row order — on
   both backends. Under WebGL2 (and WebGPURenderer's WebGL2 fallback),
   the primitive translates the input `y` to `gl.readPixels`'
   bottom-up framebuffer convention internally, and inverts the
   output rows on the way back to canonical top-down. The four
   production readback paths — screenshot (`renderToImageData`), HDR
   HalfFloat capture, HDR Float32 capture, and the picking 5×5 voter
   — all consume the primitive and see identical input/output row
   order regardless of backend. WebGPU's 256-byte `bytesPerRow`
   padding is compacted inside the primitive too, so callers never
   see it.

**Exception:** `captureHDRAsEXR` passes `flipY: true` to
`readPixelsCompactAsync` so the exported EXR keeps its
scene-space bottom-up convention. External tooling (Nuke, Houdini,
oiiotool) expects this orientation; matching it here means an EXR
exported under WebGPU is byte-identical to one exported under
WebGL2.

After this contract, no module outside `fullscreen-geometry.ts`,
`hdr-pixel-utils.ts`, and `renderer-capabilities.ts` branches on the
renderer backend for Y orientation. New post-processing or capture
code should plug into one of those two seams; never re-derive Y
handling at the call site.

## Capture modes

| Mode                   | Bypasses                            | Output                    | Used by               |
| ---------------------- | ----------------------------------- | ------------------------- | --------------------- |
| `hdr-effects-pre-tone` | EOG, tone mapping, vignette, noise, | **Linear HDR** with bloom | EXR export, recording |
| (default)              | chromatic distortion, sRGB encoding |                           |                       |
| `visible-ldr`          | sRGB encoding only                  | **Linear LDR**            | Composer parity       |
| `raw-scene-hdr`        | Mega-shader entirely (no bloom)     | Pure scene HDR            | Diagnostics           |

The bypasses are implemented as `#define`-gated shortcuts in the
shader (`LUXAR_CAPTURE_RAW_HDR`, `LUXAR_CAPTURE_LINEAR_LDR`). Toggling
a define triggers `material.needsUpdate = true`, but THREE.js caches
compiled programs by define-set so repeated capture calls reuse the
cached programs.

## Render-target sizing

Two unit systems:

- **Logical** (CSS) pixels — what `renderer.setSize(width, height)`
  takes. THREE multiplies by `pixelRatio` to derive the canvas
  backbuffer.
- **Physical** pixels — `logical × pixelRatio`. This is what
  `getDrawingBufferSize()` reports to scene materials, what the
  canvas backbuffer is, and what our render targets MUST match.

`getPhysicalSize()` = `effectiveSize × renderer.getPixelRatio()` where
`effectiveSize` = `renderSize × ssaaMultiplier` (when SSAA enabled).
All targets — `hdrTarget`, `ldrTarget`, the bloom mip pyramid, the
FXAA pass — are allocated at this physical size. Mismatching this
(e.g. allocating in logical pixels at DPR > 1) causes the materials'
`gl_PointSize` math to overshoot the actual framebuffer and the
scene to appear noticeably brighter than at DPR 1.

Whenever the size changes (resize / SSAA toggle / MSAA toggle / DPR
change) the manager calls the optional `onResize` callback. The host
wires that to `SceneManager.updateMaterialsForCurrentCamera()` so the
scene materials pick up the new drawing-buffer dimensions on the same
frame.

## Tone-mapping enum

The internal define `LUXAR_TONE_MAPPING_MODE` uses Luxar-internal IDs
(1..6), not THREE's enum (which has gaps for `Custom` and was
renumbered across r-bumps). The mapping is in
`mega-shader-material.ts:toneMappingModeDefine()`:

| THREE constant          | Luxar mode | GLSL function           |
| ----------------------- | ---------- | ----------------------- |
| `NoToneMapping`         | 1 (alias)  | `LinearToneMapping`     |
| `LinearToneMapping`     | 1          | `LinearToneMapping`     |
| `ReinhardToneMapping`   | 2          | `ReinhardToneMapping`   |
| `CineonToneMapping`     | 3          | `CineonToneMapping`     |
| `ACESFilmicToneMapping` | 4          | `ACESFilmicToneMapping` |
| `AgXToneMapping`        | 5          | `AgXToneMapping`        |
| `NeutralToneMapping`    | 6          | `NeutralToneMapping`    |

`NoToneMapping` is deliberately aliased to `LinearToneMapping` (which
saturates / clamps to [0,1]) — the old `THREE.NoToneMapping →
pmndrs.LINEAR` mapping had the same effect.

The functions come from THREE's `<tonemapping_pars_fragment>` chunk.
The chunk declares `uniform float toneMappingExposure`, which we
provide via `material.uniforms` and pin to `1.0` (because our own
`uExposure` already pre-multiplies before the tone-mapping call).

**Critical**: every post-processing material sets `toneMapped: false`.
Without it, THREE auto-injects the tone-mapping chunk again on top of
our explicit `#include`, producing a `toneMappingExposure:
redefinition` GLSL compile error and a black canvas.

## Bloom chain

The pyramid has 1..12 mip levels (UI-tunable). Mip[0] is half the
physical-pixel resolution — bloom is a soft glow and full-res doesn't
visibly improve quality while doubling memory.

Threshold uses **Rec.709 relative luma** `dot(c, vec3(0.2126,
0.7152, 0.0722))` — same shape as the old pmndrs `LuminanceMaterial`.
A previous iteration used `max(r, g, b)` which overstated saturated
single-channel pixels (pure red would bloom even at low intensity);
fixed.

Downsample is a 2×2 box filter. Upsample is a 4-tap tent filter
blended additively (via `THREE.AdditiveBlending` on the material).
The autoclear state is briefly flipped to `false` during the upsample
chain so each pass accumulates onto the previous larger mip; the
chain saves and restores `renderer.autoClear` around its work.

`setLevels(n, canvasSize?)` accepts an explicit canvas size so a
mid-resize-debounce quality-preset change doesn't re-allocate the
pyramid at a stale `mip[0].width × 2`. The manager passes
`getPhysicalSize()` from the host side.

## Detector noise

Three components, summed at the per-pixel level:

1. **Shot noise (Poisson)** — Anscombe transform stabilizes variance,
   add a unit-variance Gaussian, inverse-transform back. Smoothstep
   weighting in the [0, 0.01] intensity range fades the Anscombe-3/8
   bias to zero in pure-black pixels (otherwise vignette-darkened
   areas would brighten).
2. **Readout noise (Gaussian, temporal)** — clamped logistic
   approximation (scale 0.5513 for ~unit variance after the [-4, 4]
   clamp). Fed from a per-pixel/per-frame Bob Jenkins hash.
3. **Fixed pattern noise (Gaussian, static)** — same Gaussian
   approximation, but the seed depends only on UV — pattern is stable
   across frames.

Time advances via wall-clock `dt` measured between successive
`render()` calls. Render-duration around the pipeline used to be the
source and animated noise ~4× slower at 60 FPS because rendering
takes well under 16 ms — fixed.

The `_previousRenderTimestamp` field is reset to 0 in
`rebuildAfterContextRestore()` so the first post-restore frame
doesn't see a multi-second `dt` jump.

## Chromatic lens distortion

Brown-Conrady radial distortion plus a 3-component camera intrinsic
matrix. Per-channel sample at distorted UVs with a soft border mask
suppresses out-of-bounds bleed. The dispersion parameter scales the
distortion coefficient per channel — blue gets more distortion than
red, matching the physical Abbe behavior of optical glass.

Bloom is sampled at the SAME distorted UVs per channel — see
"Operation ordering" above.

`getLensDistortionParams()` returns cloned `Vector2` values so the
picking system (which uses these to back-project mouse coords through
the distortion) cannot accidentally mutate the shader's live uniforms.

## sRGB encoding

Linear → sRGB (Rec.709 transfer function) is applied at the END of
the mega-shader. Required for both write targets (`ldrTarget` is
sampled by FXAA, then written to backbuffer with `outputColorSpace =
SRGBColorSpace`). FXAA reads the already-sRGB-encoded ldrTarget and
passes it through unchanged; its luma-based edge detection works on
sRGB inputs (in fact that's closer to how FXAA was originally tuned
than the linear-LDR input the old composer fed it).

The encoding is conditionally skipped under
`LUXAR_CAPTURE_LINEAR_LDR` so the visible-ldr EXR capture matches
the old composer's HalfFloat ping-pong contents (linear LDR).

## Deferred-rebuild contract

The mega-shader-era pipeline has cheap rebuilds (each effect toggle
is a `#define` change + lazy program recompile, which THREE caches
by define-set). The `withDeferredRebuild` / `start/endDeferRebuild`
API is preserved for source-compatibility with cinematic-mode
batching, but it's effectively a no-op pass-through. The depth
counter still exists so a thrown sub-setter doesn't strand the
manager in a half-built state.

## Context-restore

WebGL contexts can be lost. The manager preserves its **identity**
across restore so cached references in `PickingSystem`,
`AnimationController`, and `RenderingControls` stay valid.

1. `rebuildAfterContextRestore` first resets `_previousRenderTimestamp`
   to 0 (otherwise detector noise jumps).
2. Captures every user-facing uniform / define / toggle from the
   current `MegaShaderMaterial` + bloom-chain state.
3. `disposeTransientResources()` tears down all GPU resources.
4. `initializeTransientResources()` rebuilds at the current physical
   size.
5. Restores the snapshot via the regular setter API.

The E2E test `tests/e2e/context-restore.spec.ts` asserts both
identity preservation and a non-default exposure round-tripping.

## Renderer state save/restore

`runPipeline()` is defensive: it saves the renderer's current
`renderTarget` and `autoClear` flag on entry and restores them in a
`finally` block. This protects future callers (picking, offscreen
probe) that might invoke `runPipeline` while another target is bound.

`renderToImageData()` follows up with an explicit
`renderer.setRenderTarget(null)` before `gl.readPixels` so the
backbuffer is bound regardless of what the prior caller had set.

## Lifecycle

`dispose()` is **idempotent** (`this.disposed` guard). It disposes:

- `MegaShaderMaterial` (frees uniforms + program)
- `BloomChain` (all mip targets + threshold/downsample/upsample
  materials + the shared fullscreen mesh)
- `FxaaPass` (material + mesh)
- `hdrTarget` and `ldrTarget`

There are no pmndrs-era event-driven disposal traps to work around;
every owned resource has a direct, idempotent `dispose()`.
