# Mega-shader post-processing — design doc (historical)

**Status:** ✅ SHIPPED (2026-05-12). The mega-shader is the active post-processing pipeline; it replaced `pmndrs/postprocessing` and runs under both the WebGL2 (`MegaShaderMaterial`, `mega-shader.glsl.ts`) and WebGPU (`MegaShaderTSLMaterial`, `mega.tsl.ts`) backends.

This document is retained as the **architectural reference** that motivated the implementation. The "Confirmed scope decisions" below are the ones that actually shipped; the open questions in later sections have all been resolved (search `SPECIFICATIONS.md` and `post-processing-manager.ts` for current behaviour).

**Scope:** Phase 0 of the WebGPU migration plan. Decoupled the viewer from `pmndrs/postprocessing` while staying on WebGL. Strict improvement that paid off on WebGL2 and shortened the WebGPU port.

**Confirmed scope decisions (from user):**

- Drop SMAA (keep FXAA as final pass)
- Drop SSAO entirely
- Drop DoF entirely
- Drop Reinhard2 Adaptive (auto-luminance reduction). Since the public
  THREE-enum API never reached Reinhard2 fixed either (the mapping
  `THREE.ReinhardToneMapping → pmndrs.REINHARD` skipped the
  Reinhard2 sub-class entirely), this effectively drops Reinhard2
  altogether. Net result: tone-mapping modes collapse to **exactly
  THREE's own enum**, no custom modes.
- Drop Uncharted2 tone mapping
- HDR-EXR capture stays post-tone-mapping (preserve current behavior;
  reads from the mega-shader's LDR output, not the HDR scene target)
- Work on current branch `chore/threejs-r184-migration`

---

## Why this design

`pmndrs/postprocessing`'s per-effect model runs each effect as a
separate fullscreen pass through `EffectComposer`. For Luxar's current
chain, that's ~10+ fullscreen passes per frame:

```
scene → HDR target → Bloom (3-5 passes) → Pass A (ChromaticLens) →
Pass B (DetectorNoise + ToneMapping + Vignette) → FXAA → backbuffer
```

(Pass A and B split because Bloom is "convolution" and ChromaticLens
is "UV-transform", and pmndrs refuses to share them — see
`effect-orchestrator.ts:23-39`.)

Most of these are per-pixel operations that **don't depend on
neighbors**. They can collapse into one fullscreen pass with zero
quality loss. The only multi-pass operations are bloom (needs
downsample/blur/upsample) and FXAA (needs neighbor reads to detect
edges). Everything else folds into a single mega-shader.

Target reduction: **~10+ passes → 4-6 passes** per frame on a typical
composition. Plus zero dependency on a third-party post-processing
framework.

---

## Final architecture

```
SceneRenderer ──→ HDR HalfFloat target  ┐
                                         │
                                         ├──→ BloomChain  ──→ bloomTexture (HDR HalfFloat)
                                         │   (3-5 passes)
                                         │
                                         └──→ MegaShader (one fullscreen pass)
                                                │
                                                ├ uniform: uHdrScene
                                                ├ uniform: uBloomTexture (optional)
                                                ├ uniform: vec2/float for each effect
                                                │
                                                ├ samples uHdrScene at distorted UVs (1-3x)
                                                ├ adds bloom * intensity
                                                ├ adds detector noise (procedural)
                                                ├ applies EOG (exp/offset/gamma)
                                                ├ applies tone mapping (mode-switched)
                                                ├ applies vignette multiplicative darkening
                                                │
                                                ↓
                                          LDR intermediate target
                                                │
                                                ↓
                                          FXAA pass (optional)
                                                │
                                                ↓
                                          backbuffer

(if FXAA disabled: MegaShader writes directly to backbuffer)
```

### Pass count comparison

| Scene state                                      | Today (pmndrs) | After (mega-shader) |
| ------------------------------------------------ | -------------- | ------------------- |
| Bloom + tone + vignette + FXAA (typical)         | ~9             | 5–7                 |
| All effects on (chromatic + noise + bloom + ...) | ~12            | 5–7                 |
| Bare scene (no post FX)                          | 2              | 2                   |
| Tone + EOG only (minimum)                        | 3              | 2                   |

---

## Mega-shader operation order

Matches the current pmndrs effect chain order (per
`effect-orchestrator.ts:82-114`, "buildOrderedEffects"). Visual output
should be near-identical to today modulo subpixel rounding from
combining passes:

1. **Chromatic lens distortion** → re-samples HDR scene at distorted
   UVs (3 reads if dispersion > 0, otherwise 1). This is the only
   operation that affects sampling, so it goes first.
2. **Bloom mix** → add bloom texture × intensity to scene color (single
   read at center UV; bloom is already a soft glow so distorting it
   would look weird).
3. **Detector noise** → procedural per-pixel. Hash-based PRNG +
   Anscombe Poisson approx + Gaussian readout + fixed-pattern offset.
4. **EOG** → `color *= 2^uExposure; color = max(color + uOffset, 0);
color = pow(color, 1/uGamma)`.
5. **Tone mapping** → switch on `TONE_MAPPING_MODE` define. Linear /
   Reinhard / Cineon / ACES Filmic / **AgX (default)** / Neutral /
   Reinhard2 / Uncharted2.
6. **Vignette** → multiplicative darkening based on radial distance.

After mega-shader the result is LDR sRGB-ready, with alpha forced to 1.

### Dropped from operation 5 (tone mapping)

The current `LuxarToneMappingEffect` supports an "adaptive luminance"
mode (`REINHARD2_ADAPTIVE`, TONE_MAPPING_MODE == 3) that requires a
separate `LuminancePass` + `AdaptiveLuminancePass` to compute scene
average luminance via mipmap reduction. This **cannot fit in a single
pass** (requires reduction across the image).

**Proposal:** drop just the adaptive sub-mode. Keep Reinhard2 (mode 2)
with fixed `averageLuminance`. Adaptive auto-exposure is a niche
feature and the user can use the `uExposure` uniform manually.
**Confirm with user before implementing.**

---

## New files

### `src/rendering/post-processing/mega-shader-material.ts` (~250 LOC)

```ts
export interface MegaShaderConfig {
  // Same shape as the union of LuxarToneMappingConfig +
  // ChromaticLensDistortionConfig + DetectorNoiseConfig + VignetteConfig
  // + bloom intensity. Each section optional; defaults from config/index.ts.
}

export class MegaShaderMaterial extends THREE.ShaderMaterial {
  constructor(cfg: MegaShaderConfig = {}) {
    super({
      uniforms: {
        /* see uniform list below */
      },
      vertexShader: MEGA_VERTEX, // trivial fullscreen triangle
      fragmentShader: MEGA_FRAGMENT,
      defines: {
        TONE_MAPPING_MODE: '5', // AgX default
        // USE_LENS_DISTORTION, USE_BLOOM, USE_DETECTOR_NOISE,
        // USE_VIGNETTE — gated dynamically via setters below
      },
      glslVersion: THREE.GLSL3,
    });
  }

  // === Tone mapping + EOG ===
  setToneMapping(mode: THREE.ToneMapping): void;
  setExposure(v: number): void;
  setGlobalOffset(v: number): void;
  setGlobalGamma(v: number): void;
  setReinhard2Params(white: number, mid: number, avgLum: number): void;

  // === Chromatic lens distortion ===
  setLensDistortion(p: {
    distortion: THREE.Vector2;
    principalPoint?: THREE.Vector2;
    focalLength?: THREE.Vector2;
    skew?: number;
    dispersion?: number;
  }): void;
  toggleLensDistortion(enabled: boolean): void;

  // === Detector noise ===
  setDetectorNoise(p: { readoutSigma: number; photonGain: number; fpnSigma: number }): void;
  toggleDetectorNoise(enabled: boolean): void;
  advanceTime(deltaTime: number): void;

  // === Vignette ===
  setVignette(darkness: number, offset: number): void;
  toggleVignette(enabled: boolean): void;

  // === Bloom ===
  setBloom(intensity: number, texture: THREE.Texture | null): void;
  toggleBloom(enabled: boolean): void;

  // === Common ===
  setResolution(w: number, h: number): void;

  dispose(): void;
}
```

Each `toggle*` setter mutates `this.defines` and sets
`this.needsUpdate = true`. Recompiles are rare (only on
enable/disable, not on value tweaks).

### `src/rendering/post-processing/mega-shader.glsl.ts` (~350 LOC)

Exports the vertex + fragment shader strings. Structured with one
`#ifdef` block per optional effect:

```glsl
// Fragment shader skeleton

precision highp float;
in vec2 vUv;

uniform sampler2D uHdrScene;
uniform vec2 uResolution;

#ifdef USE_BLOOM
uniform sampler2D uBloomTexture;
uniform float uBloomIntensity;
#endif

#ifdef USE_LENS_DISTORTION
uniform vec2 uDistortion;
uniform vec2 uPrincipalPoint;
uniform vec2 uFocalLength;
uniform float uSkew;
uniform float uDispersion;

vec2 applyDistortion(vec2 uv, vec2 k) { /* Brown-Conrady from current code */ }
float border(vec2 uv) { /* current implementation */ }
#endif

#ifdef USE_DETECTOR_NOISE
uniform float uTime;
uniform float uReadoutSigma;
uniform float uPhotonGain;
uniform float uFpnSigma;

// All current helpers verbatim from detector-noise-effect.ts:
uint bobJenkinsHash(uint a) { ... }
uint rnguint2(vec2 x) { ... }
uint rnguint3(vec3 x) { ... }
float rngfloat2(vec2 x) { ... }
float rngfloat3(vec3 x) { ... }
float clampedLogistic(float u) { ... }
vec3 normal3_temporal(vec3 s) { ... }
vec3 normal3_fixed(vec2 s) { ... }
float anscombeForward(float x) { ... }
float anscombeInverse(float y) { ... }
vec3 poissonNoise(vec3 seed, vec3 lambda) { ... }
#endif

uniform float uExposure;
uniform float uGlobalOffset;
uniform float uGlobalGamma;

// THREE.js built-in tone mapping chunk — gives us Linear/Reinhard/
// Cineon/ACES/AgX/Neutral as functions taking vec3 → vec3.
#include <tonemapping_pars_fragment>

#ifdef USE_REINHARD2
uniform float uWhitePoint;
uniform float uMiddleGrey;
uniform float uAverageLuminance;
vec3 reinhard2(vec3 color) { /* from luxar-tone-mapping-effect.ts */ }
#endif

#ifdef USE_UNCHARTED2
uniform float uWhitePoint;
vec3 uncharted2(vec3 color) { /* from luxar-tone-mapping-effect.ts */ }
#endif

#ifdef USE_VIGNETTE
uniform float uVignetteDarkness;
uniform float uVignetteOffset;
#endif

out vec4 fragColor;

void main() {
  vec2 uv = vUv;
  vec3 color;

  // (1) Lens distortion → samples HDR scene at distorted UVs
#ifdef USE_LENS_DISTORTION
  vec2 distR = uDistortion * (1.0 - uDispersion);
  vec2 distG = uDistortion;
  vec2 distB = uDistortion * (1.0 + uDispersion);
  vec2 uvR = applyDistortion(uv, distR);
  vec2 uvG = applyDistortion(uv, distG);
  vec2 uvB = applyDistortion(uv, distB);
  float r = texture(uHdrScene, uvR).r * border(uvR);
  float g = texture(uHdrScene, uvG).g * border(uvG);
  float b = texture(uHdrScene, uvB).b * border(uvB);
  color = vec3(r, g, b);
#else
  color = texture(uHdrScene, uv).rgb;
#endif

  // (2) Bloom mix
#ifdef USE_BLOOM
  vec3 bloom = texture(uBloomTexture, uv).rgb;
  color += bloom * uBloomIntensity;
#endif

  // (3) Detector noise
#ifdef USE_DETECTOR_NOISE
  // ... apply Anscombe Poisson + readout + FPN exactly as current code ...
#endif

  // (4) EOG — Exposure / Offset / Gamma
  color *= exp2(uExposure);
  color = max(color + vec3(uGlobalOffset), vec3(0.0));
  color = pow(color, vec3(1.0 / uGlobalGamma));

  // (5) Tone mapping — TONE_MAPPING_MODE values match THREE's enum exactly
#if TONE_MAPPING_MODE == 0
  // THREE.NoToneMapping — pass-through
#elif TONE_MAPPING_MODE == 1
  color = LinearToneMapping(color);          // THREE.LinearToneMapping
#elif TONE_MAPPING_MODE == 2
  color = ReinhardToneMapping(color);        // THREE.ReinhardToneMapping
#elif TONE_MAPPING_MODE == 3
  color = CineonToneMapping(color);          // THREE.CineonToneMapping
#elif TONE_MAPPING_MODE == 4
  color = ACESFilmicToneMapping(color);      // THREE.ACESFilmicToneMapping
#elif TONE_MAPPING_MODE == 5
  color = AgXToneMapping(color);             // THREE.AgXToneMapping (DEFAULT)
#elif TONE_MAPPING_MODE == 6
  color = NeutralToneMapping(color);         // THREE.NeutralToneMapping
#endif

  // (6) Vignette
#ifdef USE_VIGNETTE
  vec2 vcoord = (uv - 0.5) / uVignetteOffset;
  float vdist2 = dot(vcoord, vcoord);
  float vfactor = 1.0 - smoothstep(0.0, 1.5, vdist2) * uVignetteDarkness;
  color *= vfactor;
#endif

  fragColor = vec4(color, 1.0);
}
```

### `src/rendering/post-processing/bloom-chain.ts` (~200 LOC)

Standalone bloom pre-pass replacing `pmndrs` BloomEffect. Pyramid
downsample + dual-filter upsample.

```ts
export interface BloomChainConfig {
  levels: number; // 1-8, default 5
  threshold: number; // default 0.85
  smoothing: number; // default 0.01 (for soft threshold)
  radius: number; // default 0.85 — mixed in upsample blend
}

export class BloomChain {
  constructor(renderer: THREE.WebGLRenderer, cfg: BloomChainConfig);

  /** Returns the bloom texture (will reuse internally allocated target). */
  render(sceneHdrTarget: THREE.WebGLRenderTarget): THREE.Texture;

  setLevels(n: number): void;
  setRadius(r: number): void;
  setThreshold(t: number): void;
  setSize(w: number, h: number): void;
  dispose(): void;
}
```

Implementation:

- Initial pass: threshold-soft + downsample by 2x → mip[0]
- Downsample chain: each mip[i] → mip[i+1] at half resolution
- Upsample chain: blend mip[i+1] back onto mip[i] with bilateral-ish
  filtering (cheap dual filter — see Kawase Light Streaks)
- Final output: mip[0] (at full res / 2)

This is essentially a custom mipmap-blur in TS, ~150 LOC of shaders.
The shaders are simple compared to mega-shader; the complexity is in
the JS orchestration.

### `src/rendering/post-processing/fxaa-pass.ts` (~80 LOC)

Standard FXAA implementation. Single pass, samples LDR target,
writes to backbuffer.

```ts
export class FxaaPass {
  constructor(renderer: THREE.WebGLRenderer);
  render(ldrTexture: THREE.Texture, target: THREE.WebGLRenderTarget | null): void;
  setSize(w: number, h: number): void;
  dispose(): void;
}
```

FXAA shader is well-known boilerplate (~40 LOC). We can use Three's
own `FXAAShader` from `three/examples/jsm/shaders/FXAAShader.js` and
wrap it.

---

## Files modified

### `src/rendering/post-processing/post-processing-manager.ts`

Heavy rewrite. Most of the EffectComposer plumbing and Pass A/Pass B
splitting goes away. The public API surface stays the same except for
the dropped methods.

```ts
// REPLACE:
//   private composer: EffectComposer
//   private bloomEffect, toneMappingEffect, detectorNoiseEffect, ...
//   private rebuildEffectPass() { ... 200 LOC ... }
//   render() { composer.render() }
//
// WITH:
//   private hdrTarget: THREE.WebGLRenderTarget  // HalfFloat
//   private ldrTarget: THREE.WebGLRenderTarget  // RGBA8 (only when FXAA enabled)
//   private megaShader: MegaShaderMaterial
//   private megaQuad: THREE.Mesh                // fullscreen triangle
//   private bloomChain: BloomChain | null
//   private fxaaPass: FxaaPass | null
//
//   render() {
//     renderer.setRenderTarget(this.hdrTarget); renderer.render(scene, camera);
//     const bloomTex = this.bloomChain?.render(this.hdrTarget) ?? null;
//     this.megaShader.setBloom(bloomIntensity, bloomTex);
//     if (this.fxaaPass) {
//       renderer.setRenderTarget(this.ldrTarget);
//       renderer.render(this.megaQuad, this.megaCamera);
//       this.fxaaPass.render(this.ldrTarget.texture, null);  // null = backbuffer
//     } else {
//       renderer.setRenderTarget(null);
//       renderer.render(this.megaQuad, this.megaCamera);
//     }
//   }
```

Public methods to **delete** (matching the SMAA/SSAO/DoF drop):

- `setSMAAEnabled`, `updateSMAASettings`, `setDOF`, `updateDOF`,
  `setAOEnabled`, `setAOQuality`

Public methods to **keep with identical signatures**:

- `setBloomEnabled`, `updateBloomSettings`, `setBloomLevels`,
  `getBloomLevels`, `isBloomEnabled`
- `setToneMapping`, `getToneMapping`
- `updateExposure`, `updateGlobalOffset`, `updateGlobalGamma`
- `setFXAAEnabled`, `isFXAAEnabled`
- `setVignetteEnabled`
- `setMSAAEnabled`, `setMSAASamples`, etc. (canvas-level, not effects)
- `setSSAAEnabled`, `setSSAAMultiplier`, `setDPRScale` (target-sizing)
- `setQualityPreset` (will drop the SMAA/SSAO/DoF tweaks; simpler)
- `needsContinuousAnimation` (driven by detector noise)
- `render`, `resize`, `dispose`, `rebuildAfterContextRestore`

### `src/rendering/post-processing/context-recovery.ts`

State capture/restore — drop the DOF/SMAA/SSAO entries from the
durable state shape. Otherwise same shape.

### `src/rendering/post-processing/hdr-capture.ts`

EXR export reads from `hdrTarget` (the scene render target, before
post-processing) rather than the composer's result buffer. Cleaner —
the export now captures actual HDR values, not post-processed LDR.

### `src/rendering/post-processing/tone-mapping-handler.ts`,

### `src/rendering/post-processing/tone-mapping-mode-names.ts`

Stay; their pure helpers are still useful. Tone-mapping-handler will
talk to `MegaShaderMaterial` instead of `LuxarToneMappingEffect`.

### `package.json`

```diff
   "dependencies": {
     "comlink": "^4.4.2",
     "fflate": "^0.8.2",
     "mediabunny": "^1.44.2",
-    "postprocessing": "^6.39.1",
     "zarrita": "^0.7.3"
   },
```

### `src/ui/rendering-controls.ts`

Remove the SMAA preset selector, the DoF toggle/sliders, and the SSAO
toggle/quality selector. Keep everything else.

---

## Files deleted

- `luxar-tone-mapping-effect.ts` — folded into mega-shader
- `detector-noise-effect.ts` — folded into mega-shader
- `robust-vignette-effect.ts` — folded into mega-shader
- `chromatic-lens-distortion-effect.ts` — folded into mega-shader
- `bloom-handler.ts` — replaced by `bloom-chain.ts`
- `antialiasing-handler.ts` — SMAA gone; FXAA is just one boolean
- `effect-orchestrator.ts` — no chain to orchestrate
- `effect-disposal.ts` — no pmndrs Effects to dispose
- `postprocessing-types.ts` — no pmndrs type guards
- `visual-effects-handler.ts` — utilities mostly fold into mega-shader-material

Plus their `*.test.ts` partners under `src/tests/unit/rendering/`.

---

## Test strategy

### Unit

- `mega-shader-material.test.ts` — uniform plumbing, define toggles, mode switching
- `bloom-chain.test.ts` — level adjustment, render target sizing, dispose
- `fxaa-pass.test.ts` — enable/disable, size
- Keep `tone-mapping-handler.test.ts` (still relevant)
- **Delete** unit tests for removed effects (SMAA/SSAO/DoF + the 4 vendored ones)

### E2E

- Re-baseline `visual-regression.spec.ts`, `theme-visual-regression.spec.ts`,
  `post-processing-pipeline.spec.ts`, `blending-modes.spec.ts`
- Diff the re-baselines against current — expect subpixel differences
  (~< 5% pixel diff). Large diffs mean a math bug.
- Skip the SMAA/SSAO/DoF-specific specs (delete or mark obsolete)

### Performance

- Re-run benchmarks; compare fullscreen-pass count and frame time
  against `performance-baselines.json`
- Expected: ~30-50% fewer fullscreen passes, ~10-20% frame-time
  improvement on heavily-composed scenes, neutral on bare scenes

---

## Risks and open questions

1. **Reinhard2 adaptive (mode 3) — drop entirely?**
   - Current code supports REINHARD2_ADAPTIVE via separate luminance
     passes. Cannot fold into single mega-shader.
   - **Recommend dropping**, keeping fixed Reinhard2.
   - Confirm or push back.
2. **Uncharted2 tone mapping — keep?**
   - Custom mode beyond Three's built-ins. Probably-rarely-used. Easy
     to keep though (one extra GLSL function). Default: keep.
3. **HDR capture timing.**
   - Current code can capture either the input or the result buffer.
     Will move to capturing `hdrTarget` (pre-mega-shader). Need to
     confirm no consumers rely on capturing post-tone-mapped output.
4. **Bloom radius / kernel size.**
   - pmndrs uses `KernelSize.LARGE` (configurable). My BloomChain will
     fix a default kernel size (probably 9-tap) and expose the radius
     as a uniform in the upsample pass. Slightly different visual
     character than pmndrs's Kawase implementation. Plan: tune to
     match pmndrs reference visually before re-baselining.
5. **Visual-regression baseline churn.**
   - Re-baselining is unavoidable. We'll see hundreds of snapshot
     diffs. Need a careful diff review per spec class, not blanket
     accept.
6. **Disposal correctness in context loss.**
   - WebGL context-loss recovery currently leans on pmndrs's disposal
     semantics. My new classes need their own context-loss path.
     Already a story in `context-recovery.ts`; just need to wire the
     new objects in.

---

## Implementation order

If/when this design is signed off:

1. **Bloom chain** (smallest, isolated; can swap in alone).
2. **FXAA pass** (also isolated, well-known shader).
3. **Mega-shader material + GLSL** (the centerpiece).
4. **Post-processing manager rewrite** (wires 1-3 together; drops
   pmndrs imports incrementally).
5. **Delete SMAA/SSAO/DoF handler files + UI controls**.
6. **Drop `postprocessing` from package.json**, fix remaining imports.
7. **Unit-test sweep** — write new, delete old.
8. **Manual smoke** with `pnpm dev`, compare visual to a checkpoint
   commit at each effect toggle.
9. **E2E baselines re-capture**; per-spec review of diffs.
10. **Performance baseline re-run**.

Estimated: 1.5-2.5 weeks of focused work.
