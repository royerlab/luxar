# Luxar Rendering Package

> Dual-stack WebGL2/WebGPU rendering pipeline (GLSL `ShaderMaterial` default, TSL `NodeMaterial` opt-in via `?renderer=webgpu`) with a custom mega-shader for high-quality nD scientific visualization

## Overview

The Luxar Rendering package provides a high-performance rendering pipeline built on Three.js r184. Each of the 9 production shaders (3 geometry visual + 3 geometry picking + mega-shader + FXAA + bloom-threshold) ships as a `ShaderSource` pair: a `WebGLRenderer`-targeted GLSL3 string and a `WebGPURenderer`-targeted TSL factory. `MaterialManager` dispatches on `RendererCapabilities.apiSurface` so the same scene graph renders identically through either backend. Post-processing runs through a hand-written **mega-shader** that fuses all per-pixel effects into a single fullscreen fragment pass — bloom is a separate pre-pass (needs neighbor reads) and FXAA is a separate post-pass (edge detection on the LDR output).

The default backend is `THREE.WebGLRenderer` (GLSL `ShaderMaterial`). `WebGPURenderer` (TSL `NodeMaterial`) is selectable via `?renderer=webgpu` or `VITE_LUXAR_USE_WEBGPU=1`; it falls back to its internal WebGL2 backend when no WebGPU adapter is available. Every TSL shader is validated against its GLSL counterpart through `tsl-shader-parity.spec.ts`. For diagnostics, `?renderer=webgpu&webgpu-force-webgl` constructs `WebGPURenderer({ forceWebGL: true })`: Luxar still uses TSL `NodeMaterial` shaders and the WebGPURenderer API surface, but Three.js routes rendering through its internal WebGL2 backend instead of native WebGPU.

### Key Features

- **HDR Rendering Pipeline**: 16-bit float (HalfFloat) buffers for true HDR support
- **Mega-shader post-processing**: One fused fragment pass for bloom mix, detector noise, EOG, tone mapping, vignette, chromatic lens distortion, and sRGB encoding
- **Custom Shader System**: Optimized shaders for Points, Lines, and GSplats
- **Line Rendering**: Instanced quad geometry for thick lines with seamless joints
- **Material Management**: Efficient caching and reuse for Points, Lines, and GSplats
- **World-Space Point Sizing**: Physically accurate scaling
- **Anti-Aliasing Options**: FXAA, MSAA, and SSAA support

### Package Architecture

```
rendering/
├── material-manager.ts                 # Material creation, caching, and global camera updates
├── node-factory.ts                     # Scene-node factories for Points / Lines / GSplats
├── gpu-buffer-pool.ts                  # Geometry reuse with count and byte-budget eviction
├── gpu-byte-budget.ts                  # Single adaptive VRAM budget (pool + LOD registry share it)
├── adaptive-dpr-manager.ts             # Adaptive resolution
├── colormap-textures.ts                # Built-in/custom DataTexture creation and cache disposal
├── colormap-data.ts                    # Built-in colormap lookup tables (auto-generated)
├── renderer-capabilities.ts            # WebGL2 vs WebGPU capability detection
├── blending-state.ts                   # THREE blending state for every Luxar mode
├── material-colormap-helpers.ts        # Shared scalar-colormap guards and uniform helpers
├── material-sync-helpers.ts            # Geometry-commit material sync helpers
├── line-geometry.ts                    # Instanced line mesh creation/update helpers
├── gsplat-geometry.ts                  # Instanced GSplat mesh creation/update helpers
├── point-geometry.ts                   # Point unit-quad base geometry
├── interleaved-attributes.ts           # InterleavedBufferAttribute helpers
│
├── materials/                          # Per-geometry material and shader stacks
│   ├── point/   { material-glsl, material-tsl, shader-glsl, shader-tsl }
│   ├── line/    { material-glsl, material-tsl, shader-glsl, shader-tsl }
│   ├── gsplat/  { material-glsl, material-tsl, shader-glsl, shader-tsl, math }
│   └── _shared/ { camera-aware-material, colormap-aware-material, camera-uniforms,
│                  uniform-helpers, material-builder, tsl-helpers, glsl-lib, shader-source }
│
├── material-manager/                   # MaterialManager helper modules
│   ├── factories.ts                    # VISUAL/PICKING/MEGA_SHADER_FACTORIES + cache-key fns
│   ├── lru-cache.ts                    # Generic lruGet / lruSet
│   ├── lifecycle.ts                    # subscribeToDispose + removeFromRegistries + SOFT_DISPOSE_FLAG
│   └── stats.ts                        # getCacheStats snapshot
│
├── node-factory/                       # NodeFactory helper modules
│   ├── validation.ts                   # validateLoadedPointsData / ColorMode / TransformFormat
│   ├── transforms.ts                   # applyTransform
│   ├── create-points-node.ts           # createPointsGeometry + createPointsMaterial
│   ├── create-lines-node.ts            # createLinesNode + createEmptyLinesNode
│   └── create-gsplats-node.ts          # createGSplatsNode + createEmptyGSplatsNode
│
├── post-processing/                    # HDR post-processing pipeline
│   ├── post-processing-manager.ts      # Public API — HDR pipeline orchestrator
│   ├── bloom/                          # chain / shaders / bloom.tsl
│   ├── fxaa/                           # pass / shaders / fxaa.tsl
│   ├── fullscreen/                     # pass / geometry
│   ├── mega/                           # material / material-tsl / shader.glsl / shader.tsl
│   ├── hdr/                            # pixel-utils / capture
│   ├── render-target-sizing.ts
│   └── post-processing-manager/        # PostProcessingManager helper modules
│       ├── resource-lifecycle.ts       # buildTransientResources / disposeTransientResources / sizing
│       ├── settings.ts                 # bloom / msaa / vignette / chromatic-lens setters
│       ├── pipeline.ts                 # runPipeline (scene → HDR → bloom → mega → FXAA)
│       └── capture.ts                  # captureHDRPixels / captureHDRAsEXR / renderToImageData
│
├── picking/                            # GPU picking materials + orchestration
│   ├── index.ts                        # Picking barrel
│   ├── picking-system.ts               # Orchestrator
│   ├── PICKING_DESIGN.md               # Backend readback strategy + 1-frame-latency rationale
│   ├── point/    { material, material-tsl, shaders (GLSL), pick.tsl (TSL) }
│   ├── line/     { material, material-tsl, shaders (GLSL), pick.tsl (TSL) }
│   ├── gsplat/   { material, material-tsl, shaders (GLSL), pick.tsl (TSL) }
│   └── picking-system/                 # PickingSystem helper modules
│       ├── registration.ts             # disposePickMaterial / unregisterAllPickMaterials / PickNodeEntry
│       ├── ray-aabb.ts                 # rayHitsAnyNode / getOrComputeWorldBox / invalidateBoxCache
│       ├── pick-render.ts              # voteWinner (brightness-weighted majority over 5×5)
│       ├── lens-distortion.ts          # applyLensDistortion (sync screen-space picking with mega-shader)
│       ├── settle-loop.ts              # HOVER_SETTLE_MS + evaluateSettle decision logic
│       └── settle-scheduler.ts         # Debounced settle-pass scheduling
│
├── gpu-buffer-pool/                    # Per-type adapters + eviction
│   ├── {points,lines,gsplats}-adapter.ts
│   ├── attribute-codec.ts / eviction-policy.ts / pool-stats.ts
│   ├── capacity.ts                     # chooseCapacity (1.5× growth, min-instance floor)
│   ├── geometry-bytes.ts               # estimateGeometryBytes + cached size invalidation
│   └── byte-budget-evictor.ts          # Cross-type byte-budget enforcement
│
├── shaders/                            # Barrel only — re-exports GLSL constants from materials/<kind>/shader-glsl.ts
│   └── index.ts                        # Keeps the tsl-shader-parity e2e harness's import path stable
│
├── index.ts                            # Public-API barrel
└── README.md                           # This documentation
```

---

## Getting Started

### Step 1: Initialize Post-Processing

```typescript
import { PostProcessingManager } from './rendering/post-processing/post-processing-manager';
import { createRendererCapabilities } from './rendering/renderer-capabilities';

const capabilities = createRendererCapabilities(renderer);
const postProcessing = new PostProcessingManager(renderer, capabilities, scene, camera, {
  width: window.innerWidth,
  height: window.innerHeight,
});

// In your render loop
function animate() {
  requestAnimationFrame(animate);
  postProcessing.render(); // Renders scene with all effects
}
```

### Step 2: Enable Bloom (HDR Glow)

```typescript
postProcessing.updateBloomSettings(
  0.3, // strength: how intense the glow (0-1)
  0.85, // radius: how far it spreads (0-1)
  0.01 // threshold: HDR values above this glow
);
```

### Step 3: Choose Tone Mapping

```typescript
// AgX: Neutral, film-like (recommended)
postProcessing.setToneMapping(THREE.AgXToneMapping);

// Or try others:
// THREE.ACESFilmicToneMapping - Cinematic with warm tones
// THREE.ReinhardToneMapping  - Classic, simple
// THREE.LinearToneMapping    - No tone mapping
```

### Step 4: Add Anti-Aliasing

```typescript
// FXAA: Single-pass post-process AA (recommended for real-time)
postProcessing.setFXAAEnabled(true);

// OR MSAA: Hardware-accelerated multisample (sharp, but additive blending caveat)
postProcessing.setMSAAEnabled(true);
postProcessing.setMSAASamples(4);
```

**You're done!** Your scene now has professional HDR rendering with bloom, tone mapping, and anti-aliasing.

---

## Components

### 1. PostProcessing Manager (mega-shader)

The `PostProcessingManager` runs the mega-shader pipeline: a custom fragment shader that fuses tone mapping, EOG, vignette, detector noise, chromatic lens distortion, and sRGB encoding into a single fullscreen pass. Bloom is a separate pre-pass (neighbor reads). FXAA is a separate post-pass (edge detection on the LDR output). See `post-processing/README.md` for the full pipeline.

**Key Advantages:**

- One fused fullscreen pass for per-pixel effects: fewer rasterizations, fewer texture binds, no ping-pong target pair
- No third-party post-processing dependency
- A single GLSL/TSL shader pair (`mega/shader.glsl.ts` + `mega/shader.tsl.ts`) backs both the WebGL2 and WebGPU backends

**Core API:**

```typescript
// Initialize with HDR support
const postProcessing = new PostProcessingManager(renderer, capabilities, scene, camera, {
  width,
  height,
});

// Configure bloom
postProcessing.updateBloomSettings(/* strength */ 0.3, /* radius */ 0.85, /* threshold */ 0.01);

// Set tone mapping
postProcessing.setToneMapping(THREE.ACESFilmicToneMapping);

// Enable effects
postProcessing.setVignetteEnabled(true, 0.5, 0.5);
postProcessing.setChromaticLensDistortionEnabled(true, -0.05, -0.05);
```

### 2. Point Material

Advanced shader material for points rendering with custom vertex and fragment shaders.

**Vertex Shader Features:**

- World-space sizing with correct angular calculation
- Sharpness compensation using mathematical model
- FOV-independent sizing
- Automatic viewport adaptation

**Fragment Shader Features:**

- Power-based falloff for smooth edges
- Per-node GOG (Gain-Offset-Gamma) color adjustment: `color * intensity + offset; clip; pow(color, 1/gamma)`
- Per-point sharpness control
- Optimized with pre-computed uniforms

### 3. Line Material

Specialized shader material for thick lines using instanced quad geometry.

**Key Features:**

- **Super-Gaussian Kernel**: shifted-truncated super-Gaussian perpendicular falloff `max(exp(−K·p^β) − C, 0)/(1 − C)` (K=ln(100)≈4.605, C=0.01); sharpness is a normalised `[0, 1]` knob mapping to `β = 2^(6s − 2)` (s=0.5 → β=2 Gaussian, the default)
- **Seamless Joints**: Cap factor calculation ensures correct additive blending at joints
- **World-Space Width**: Lines have consistent thickness regardless of distance
- **nD Clipping**: Clipped endpoints use full intensity for correct visual appearance
- **Per-Vertex Attributes**: Color, width, and sharpness interpolate along segments
- **Aspect-Ratio Correct**: Perpendicular direction computed in pixel space for correct line width
- **Anti-Aliasing for Thin Lines**: Minimum pixel width (1.5px) prevents sub-pixel rendering gaps; intensity scaling preserves visual weight of thin lines; smooth edge falloff using smoothstep

**Architecture Note:** Lines use `THREE.Mesh` with `InstancedBufferGeometry` (not `THREE.InstancedMesh`). All per-instance attributes are packed into a single `InstancedInterleavedBuffer` so the geometry reports one vertex buffer slot — this both fits within WebGL2's 16-attribute-location limit and stays under WebGPU's `maxVertexBuffers` ceiling on compat-mode adapters (Luxar requests the adapter's higher real limits at init; see `scene-manager.ts::setupWebGPURenderer`).

### 4. GSplat Material

Specialized shader material for volumetric Gaussian splatting with nD slicing support.

**Key Features:**

- **Oriented Anisotropic Gaussian**: Full 3D covariance via Cholesky factors
- **Perspective Projection**: Projects 3D covariance to 2D screen space using Jacobian
- **Standard Gaussian Falloff**: `exp(-½ · r²)` for physically correct rendering
- **Sum/Max Projection Modes**: Ray integration for additive, peak value for max blending
- **Two-Stage Near-Plane Culling**: Fixed threshold (1 cycle) + adaptive threshold (6 cycles) for large splats
- **GPU Optimizations**: Flat interpolation, reciprocal precomputation, early discard at 3σ

**Culling Strategy:**

- **Behind-camera rejection**: Prevents rendering splats behind the viewer
- **Near-plane culling**: Two-stage approach prevents overdraw from splats very close to camera
  - Stage 1: Fixed threshold (z < 0.1) culls most splats instantly
  - Stage 2: Adaptive threshold (z < sigma × truncate) only for large splats (sigma > 0.1)
  - Prevents white-screen artifacts when navigating inside datasets

**Architecture Note:** GSplats use `THREE.Mesh` with `InstancedBufferGeometry` for instanced quad rendering, similar to line material approach.

### 5. Material Manager

Singleton manager for efficient material creation and caching across Points, Lines, and GSplats. Runtime blending changes use `blending-state.ts` so UI updates apply the same complete THREE.js state as material creation.

```typescript
// Get cached point material
const pointMaterial = materialManager.getPointMaterial({
  blendingMode: 'additive',
  opacity: 1.0,
  gamma: 1.0,
});

// Get cached line material
const lineMaterial = materialManager.getLineMaterial({
  blendingMode: 'additive',
  opacity: 1.0,
  intensity: 1.0,
  offset: 0.0,
});

// Update global parameters (updates both point and line materials)
materialManager.updateCameraParams(fov, resolution);
```

#### Material Lifecycle and Memory Management

**Automatic Disposal**: Materials are automatically registered with the MaterialManager when created and unregistered when disposed. This prevents memory leaks.

```typescript
// Materials are cached and reused automatically
const material1 = materialManager.getPointMaterial({ opacity: 1.0 });
const material2 = materialManager.getPointMaterial({ opacity: 1.0 }); // Returns same instance

// When disposing geometry/points, material is automatically handled
points.geometry.dispose(); // Frees GPU buffers
// Material manager keeps material alive if other objects use it
```

**Global Updates**: When camera settings change, MaterialManager automatically updates ALL registered materials - no manual scene traversal needed. Global exposure/offset/gamma are handled inside the mega-shader post-processing pass, not per-material.

```typescript
// Updates all materials in the scene automatically
materialManager.updateCameraParams(newFov, newResolution);
```

**Memory Leak Prevention**: Always dispose geometries and points when done. The material system handles cleanup automatically.

**Key Points**:

- Materials are cached by properties (opacity, gamma, intensity, offset, blending mode)
- Global uniform updates affect all materials simultaneously
- Disposal is automatic - no manual material cleanup needed
- Thread-safe caching prevents duplicate material creation

### 6. GPU Buffer Pool

The `GPUBufferPool` manages geometry reuse for Points, Lines, and GSplats, eliminating per-frame GPU allocations.

**Key Features:**

- Size-based bucketing: reuses geometries when size and type match (0ms GPU allocation)
- In-place attribute updates via `TypedArray.set()`
- Count-based and byte-budget eviction (`gpuPoolMaxBytes`, `gpuPoolEvictBatchSize`)
- Multi-type support: Points, Lines, and GSplats, including optional scalar attributes for colormaps

### 7. Adaptive DPR Manager

The `AdaptiveDPRManager` dynamically adjusts device pixel ratio based on real-time FPS, trading resolution for frame rate when needed. It is a facade over pure, timestamp-driven modules in `adaptive-dpr/` (FPS tracker, refresh-rate estimator, hysteresis tracker, probe controller, bounds ledger — see that folder's README).

**Control loop:**

- Samples FPS using a 1-second sliding window, evaluated every 500 ms; a frame gap > `gapResetMs` (stall, idle-resume) resets the window so dead time never reads as low FPS
- Thresholds are RELATIVE to the display's estimated achievable rAF rate: scale down below `scaleDownFpsRatio × cap`, count toward scale-up above `scaleUpFpsRatio × cap` (works unchanged on 30/60/120/144 Hz; the estimator holds a high-water mark, lower-bounded by `refreshRateFallback` until genuine rAF throttling is detected)
- Scale-up fires after `hysteresisSeconds` of sustained high FPS, with a small mid-band grace so isolated dropped-frame samples don't restart the wait
- The DPR walks multiplicatively below the LIVE `window.devicePixelRatio` (re-read on every evaluation and public read; a monitor/zoom change rebases all learned state and clamps an engaged override — no supersampling on a lower-DPI display) and stops strictly above `max(minDPR, learned floor)`

**U-shape probe and learned bounds:**

- Every scale-down is probe-verified: if FPS did not improve ≥ `probeImprovement` on a clean sample (`probeMinSamples`, half-window span, not load-suppressed), it reverts and the probed DPR becomes a floor; unclean probes void as inconclusive (nothing learned)
- Repeated identical rejections back off exponentially (`floorTtlMs` × `backoffMultiplier`^n, capped) — no eternal probe/blur cycle on scenes DPR reduction can't help
- On HiDPI displays, the operating ceiling demotes from native to exactly 1.0 for the session (TTL-decayed, backed off on re-demotion) on either kind of evidence: repeated "punished ascents" (a scale-up above 1.0 followed by an FPS collapse), or sustained sub-throttle DISTRESS — FPS below what any real display mode can produce (< ~22 fps) for a sustained period, which the estimator reports instead of ever latching its throttle verdict down there (pre-fix that latch collapsed the cap onto the loaded FPS and parked the DPR at native on exactly the scenes that needed help)
- `notifyContentChanged()` (dataset/layer/LOD changes) pulls learned-bound expiries forward and resets backoff streaks; `setLoadActivityPredicate()` suppresses all learning while data loads

**Idle/pause integration:**

- `notifyPaused()` (from `stopAnimation`) clears session state only — learned bounds survive
- At idle-pause the resting frame is restored to full native sharpness (`prepareIdleFrame()` + one direct render); `notifyResumed()` snaps back to the remembered operating DPR in one step
- Frames are not recorded while the rendering context is lost. Known limitation: under `?renderer=webgpu`, device loss is handled via a pipeline latch (unrecoverable this release), and WebGPU async pipeline-compile jank inside probe windows is not specially detected — the probe backoff bounds the damage
- `?dpr=<value>` pins a fixed pixel ratio and locks adaptation off for the session (deterministic E2E/visual runs)

### 8. Colormap Data

`colormap-data.ts` contains the auto-generated lookup tables (LUTs) for all built-in colormaps (e.g., viridis, magma, turbo). Each LUT is a flat Uint8 array of RGBA values.

### 9. Colormap Textures

`colormap-textures.ts` manages creation and caching of `THREE.DataTexture` instances from built-in and custom colormap LUTs. Built-in textures live for the app lifetime; custom LUT textures are bounded and can be disposed on dataset unload.

### 10. Global EOG (Exposure-Offset-Gamma)

Applied inside the mega-shader fragment before the tone-mapping operator, in a single fullscreen pass:

**EOG Uniforms:**

- `uExposure`: Log2 stops (`color * 2^exposure`)
- `uGlobalOffset`: Additive shift (`color + offset`)
- `uGlobalGamma`: Power curve (`pow(color, 1/gamma)`)

---

## Effects Library

### Core Effects

#### Bloom

HDR bloom via the separate `BloomChain` pre-pass:

- Rec.709 luma threshold with soft knee
- Configurable intensity, radius, and mip-count (1-12)
- Output texture sampled and additively mixed by the mega-shader

#### Tone Mapping

Multiple tone mapping operators (all run inside the mega-shader via THREE's `<tonemapping_pars_fragment>` chunk):

- Neutral (default) - Minimal color shift, preserves hue fidelity for scientific data
- ACES Filmic - Industry standard cinematic look (used in cinematic mode)
- AgX - Modern alternative
- Reinhard / Cineon - Classic operators
- Linear - Clamp/saturate to [0, 1]

### Anti-Aliasing

#### FXAA

Fast Approximate Anti-Aliasing:

- Very fast performance
- Good quality for most cases
- Single-pass implementation
- **Recommended for general use**
- Works perfectly with additive blending

#### MSAA

Multisample Anti-Aliasing:

- Hardware-accelerated
- Sample counts: 2x, 4x, 8x
- **⚠️ WARNING**: Incompatible with additive blending
- Causes brightness multiplication artifacts with points
- Only use with normal blending mode

#### SSAA

Super-Sample Anti-Aliasing:

- Renders at higher resolution (1.5x, 2x, 3x, 4x)
- Best possible quality
- **Heavy performance cost**
- Recommended only for screenshots or high-end GPUs

### Cinematic Effects

#### Vignette

Screen edge darkening (multiplicative stage in the mega-shader):

- Adjustable darkness
- Configurable offset
- No alpha-overflow artifacts (mega-shader forces alpha = 1.0 at the final write)

#### Chromatic Lens Distortion

Physically accurate lens distortion with wavelength-dependent chromatic aberration (per-channel sampling stage in the mega-shader):

- **Wavelength-dependent distortion**: Blue refracts more than red (optical dispersion)
- **Realistic chromatic fringing**: Follows lens geometry (stronger at edges)
- Full camera model: Distortion, principal point, focal length, skew
- Barrel/pincushion distortion for wide angle/telephoto simulation

#### Detector Noise (Physics-Based)

Realistic camera/detector noise simulation for scientific imaging:

- **Shot Noise (Poisson)**: Signal-dependent noise from photon statistics
- **Readout Noise (Gaussian, temporal)**: Signal-independent electronic noise, varies per frame
- **Fixed Pattern Noise (Gaussian, static)**: Per-pixel offset from detector non-uniformities
- Configurable photon gain for low-light simulation
- Uses efficient GPU approximations (Bob Jenkins hash, clamped logistic, Anscombe transform)
- **Recommended for scientific visualization aesthetics**

**Physics model:**

```
I_observed = Poisson(I_true / gain) × gain + Gaussian_temporal(0, σ_read²) + FPN(pixel)
```

```typescript
// Enable physics-based detector noise
postProcessing.setDetectorNoiseEnabled(
  true, // enabled
  0.01, // readoutSigma: Temporal readout noise (0-0.1)
  0.01, // photonGain: Shot noise visibility (0.0001-0.1)
  0.005 // fpnSigma: Fixed pattern noise (0-0.05)
);

// Update parameters dynamically
postProcessing.updateDetectorNoiseSettings({
  photonGain: 0.05, // Simulate low-light conditions
  fpnSigma: 0.01, // Add more fixed pattern noise
});
```

---

## Anti-Aliasing Recommendations

### For Best Results with Point Clouds

1. **General Use**: Enable **FXAA** - fast and effective
2. **Maximum Quality**: Enable **SSAA** at 2x (heavy performance cost)
3. **Avoid MSAA with additive blending** (used by Points / GSplats)

### Troubleshooting

#### SSAA Issues

- If viewport appears cropped, restart the viewer
- Performance impact scales quadratically with multiplier

#### MSAA Not Working

- Check console for GPU support warnings
- Under WebGL2, MSAA requires float-buffer extensions on the active context; check `RendererCapabilities.maxMSAASamples > 0`
- Under WebGPU, MSAA is native (no extension required); the same `maxMSAASamples` query reports the adapter's supported sample counts
- Will not show effect with additive blending
- Try switching to normal blending mode to verify

#### Performance Tips

- Start with FXAA for best performance
- SSAA should only be used for final renders
- Monitor FPS when enabling AA effects

---

## Pipeline Architecture

### Rendering Flow

```
Scene Geometry
    ↓
Custom Point / Line / GSplat Shaders (HDR colors)
    ↓
HDR Render Target (HalfFloatType, optional MSAA + SSAA)
    ↓
Bloom pre-pass (if enabled): threshold + mip pyramid → bloom texture
    ↓
Mega-shader fullscreen pass: fuses
   ChromaticLensDistortion → Bloom mix (additive) → DetectorNoise →
   EOG → ToneMapping → Vignette → sRGB encode
    ↓
Optional FXAA post-pass on the tone-mapped LDR output
    ↓
Canvas backbuffer
```

### Performance Optimizations

1. **Single fused pass** for per-pixel effects (one rasterization, one set of binds)
2. **Cached programs**: Toggle defines trigger lazy recompile, then the program is cached by define-set
3. **Material caching**: Reuse materials with same properties
4. **Selective AA**: Choose AA method based on performance

---

## Configuration

### HDR Configuration

```typescript
// Renderer setup for HDR (the PostProcessingManager pins these)
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;

// HDR render target (managed by PostProcessingManager)
new THREE.WebGLRenderTarget(width, height, {
  type: THREE.HalfFloatType,
});
```

---

## Usage Examples

### Basic Setup

```typescript
import { PostProcessingManager } from './rendering/post-processing/post-processing-manager';
import { materialManager } from './rendering/material-manager';

// Initialize post-processing
const postProcessing = new PostProcessingManager(renderer, capabilities, scene, camera, {
  width: canvas.width,
  height: canvas.height,
});

// Create point material
const material = materialManager.getPointMaterial({
  blendingMode: 'additive',
  opacity: 1.0,
  gamma: 1.0,
});
```

### Configuring Effects

```typescript
// Enable multiple effects
postProcessing.updateBloomSettings(0.3, 1.0, 0.01);
postProcessing.setToneMapping(THREE.ACESFilmicToneMapping);
postProcessing.setFXAAEnabled(true);

// Add cinematic effects
postProcessing.setVignetteEnabled(true, 0.5, 0.5);
postProcessing.setChromaticLensDistortionEnabled(true, -0.05, -0.05, 0.03);
```

### Render Loop

```typescript
function animate() {
  requestAnimationFrame(animate);

  // Update controls
  controls.update();

  // Render with post-processing
  postProcessing.render();
}
```

---

## Performance Guidelines

### Optimization Strategies

1. **Start Simple**: Begin with bloom and tone mapping only
2. **Add Selectively**: Enable effects based on performance budget
3. **Profile First**: Measure impact before adding effects
4. **Use Quality Presets**: Match quality to hardware capability
5. **Monitor FPS**: Disable effects if FPS drops below target

### Performance Impact (1M points, 1080p, indicative)

| Effect                                        | Performance Cost |
| --------------------------------------------- | ---------------- |
| Base Rendering                                | ~5ms             |
| Bloom (BloomChain)                            | ~2ms             |
| Mega-shader fused pass (everything per-pixel) | ~0.5-1ms         |
| FXAA                                          | ~0.5ms           |

---

## Troubleshooting

### Common Issues

**Problem: Black screen after enabling effects**

- Check browser console for WebGL/WebGPU errors (the message prefix tells you which backend is active)
- Verify HDR buffer support via the backend-agnostic `RendererCapabilities.hdr.floatTextures` (don't probe `renderer.capabilities.isWebGL2` — that's WebGL-only and silently undefined under WebGPU)
- Try disabling effects one by one to isolate the issue
- Verify tone mapping mode is set (required for HDR pipeline)
- See `post-processing/README.md` Troubleshooting for known causes (e.g. missing `toneMapped: false` on a custom material)

**Problem: Poor performance with all effects**

- Disable bloom first (~2 ms at 1080p, plus mip allocations)
- Reduce bloom mipmap levels: `setBloomLevels(3)` instead of default 8
- Disable detector noise if not needed
- Lower SSAA multiplier or disable: `setSSAAEnabled(false)`

**Problem: Colors look wrong**

- Verify tone mapping operator: try 'AgX' or 'ACES Filmic' instead of 'Reinhard'
- Check exposure value in HDR controls
- Ensure proper color space: `renderer.outputColorSpace = THREE.SRGBColorSpace`
- Verify bloom threshold isn't too low (washing out colors)
- Check gamma correction in materials (should be 1.0 for linear workflow)

**Problem: Effects not visible**

- Check effect enabled state in debug console
- Verify threshold values (bloom threshold too high will gate everything)
- Confirm `LUXAR_TONE_MAPPING_MODE` matches the THREE constant you set
- Confirm the mega-shader is the bound material (capture-mode defines bypass downstream stages)

**Problem: Thin lines have aliasing/gaps**

- Line material automatically handles this with 1.5px minimum width
- Ensure anti-aliasing is enabled (FXAA or SSAA)
- Check that line widths are properly set (not zero or NaN)
- For very thin lines, increase width slightly or use higher SSAA

**Problem: Bright artifacts in dark areas (additive blending)**

- The mega-shader forces fragColor.a = 1.0 at the final write, so this is unlikely now. If you see it, check that no upstream custom material is propagating NaN/Inf into the HDR target.

**Problem: nD slicing shows no points**

- Points with zero effective radius are filtered in fragment shader
- Navigate to a different slice where points intersect the hyperplane
- Check dimension ranges and current slider positions
- Verify the dataset has points in the current nD region

**Problem: Material cache thrashing (many materials created)**

- Materials use integer bucketing to group similar values
- Small variations (e.g., opacity 0.999 vs 1.0) create separate materials
- Use consistent values: prefer 1.0, 0.5, 0.25 instead of arbitrary floats
- Check cache statistics: `materialManager.getCacheStats()`

**Problem: Out of memory with large datasets**

- Enable chunked loading in data loader
- Reduce bloom mipmap levels
- Use lower SSAA multiplier
- Consider using lower encoding mode (AGGRESSIVE)

---

## API Reference

### PostProcessingManager

| Method                                                       | Description                                                                                                                                                                                                                       |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `render()`                                                   | Execute rendering pipeline                                                                                                                                                                                                        |
| `setBloomEnabled(enabled, strength?, radius?, threshold?)`   | Enable/disable bloom (and update settings)                                                                                                                                                                                        |
| `updateBloomSettings(strength?, radius?, threshold?)`        | Update bloom settings                                                                                                                                                                                                             |
| `setBloomLevels(levels)`                                     | Set bloom mip pyramid depth (1-12)                                                                                                                                                                                                |
| `setToneMapping(mode)`                                       | Set tone mapping operator (THREE.ToneMapping)                                                                                                                                                                                     |
| `updateExposure(value)` / `getExposure()`                    | EOG exposure (log2 stops)                                                                                                                                                                                                         |
| `updateGlobalOffset(value)` / `updateGlobalGamma(value)`     | EOG offset and gamma                                                                                                                                                                                                              |
| `setFXAAEnabled(enabled)`                                    | Toggle FXAA post-pass                                                                                                                                                                                                             |
| `setMSAAEnabled(enabled)` / `setMSAASamples(n)`              | Toggle MSAA on the HDR target / set sample count                                                                                                                                                                                  |
| `setSSAAEnabled(enabled)` / `setSSAAMultiplier(value)`       | Toggle SSAA / set supersampling factor                                                                                                                                                                                            |
| `setDetectorNoiseEnabled(enabled, sigma?, gain?, fpnSigma?)` | Configure physics-based detector noise                                                                                                                                                                                            |
| `updateDetectorNoiseSettings(params)`                        | Update detector noise parameters                                                                                                                                                                                                  |
| `setVignetteEnabled(enabled, darkness?, offset?)`            | Configure vignette                                                                                                                                                                                                                |
| `setChromaticLensDistortionEnabled(enabled, ...params)`      | Configure chromatic lens distortion                                                                                                                                                                                               |
| `updateChromaticLensDistortion(params)`                      | Update chromatic lens distortion params                                                                                                                                                                                           |
| `getLensDistortionParams()`                                  | Read distortion uniforms (cloned, for picking)                                                                                                                                                                                    |
| `captureHDRPixels(mode?)` / `captureHDRAsEXR(opts?)`         | Read HDR/LDR pixels for EXR export                                                                                                                                                                                                |
| `renderToImageData()`                                        | Render once and read back as ImageData (sRGB)                                                                                                                                                                                     |
| `rebuildAfterContextRestore()`                               | Rebuild GPU resources after a WebGL2 `webglcontextrestored` event. WebGPU device loss uses a different model (`device.lost` promise) and is currently treated as unrecoverable — see `scene-manager.ts::setupContextLossHandling` |
| `setDPRScale(value)`                                         | Apply an adaptive DPR scale                                                                                                                                                                                                       |
| `startDeferRebuild()` / `endDeferRebuild()`                  | Defer rebuilds during bulk changes (no-op in mega-shader pipeline)                                                                                                                                                                |
| `dispose()`                                                  | Clean up resources                                                                                                                                                                                                                |

---

## Future Enhancements

- Temporal Anti-Aliasing (TAA)
- Screen Space Reflections (SSR)
- Motion Blur
- Lens Flare
- God Rays
- Color Grading with LUTs
- Custom Effect API

---

## Contributing

When extending the rendering system:

1. **Add per-pixel effects to the mega-shader** rather than as a new full-screen pass — see `post-processing/mega/shader.glsl.ts` and `post-processing/mega/material.ts`
2. **Bracket new effects with `#ifdef USE_*` defines** so disabled effects compile out entirely
3. **Test performance** across different hardware
4. **Document settings** and performance impact
5. **Maintain HDR pipeline** integrity (custom materials must set `toneMapped: false`)

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

_For implementation details, see the source files in this directory._
