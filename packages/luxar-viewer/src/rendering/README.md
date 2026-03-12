# Luxar Rendering Package

> Advanced WebGL rendering pipeline using pmndrs/postprocessing for high-quality points visualization

## Overview

The Luxar Rendering package provides a modern, high-performance rendering pipeline powered by the pmndrs/postprocessing library. It delivers professional-grade visual effects with optimized performance for large-scale points visualization.

### Key Features

- **HDR Rendering Pipeline**: 16-bit float buffers for true HDR support
- **Modern Post-Processing**: Powered by pmndrs/postprocessing
- **Professional Effects**: Bloom, SSAO, DOF, tone mapping, and more
- **Custom Shader System**: Optimized shaders for points and lines
- **Line Rendering**: Instanced quad geometry for thick lines with seamless joints
- **Material Management**: Efficient caching and reuse for points and lines
- **World-Space Point Sizing**: Physically accurate scaling
- **Multiple Anti-Aliasing Options**: FXAA and SMAA support

### Package Architecture

```
rendering/
├── post-processing-manager.ts          # Post-processing pipeline using pmndrs
├── point-material.ts                   # Custom points shaders (with per-node GOG)
├── line-material.ts                    # Instanced line rendering with semicircle kernel
├── gsplat-material.ts                  # Gaussian splatting with volumetric rendering
├── material-manager.ts                 # Material creation and caching (points + lines + gsplats)
├── luxar-tone-mapping-effect.ts        # Vendored tone mapping with EOG (exposure-offset-gamma)
├── chromatic-lens-distortion-effect.ts # Physically accurate lens distortion + chromatic aberration
├── robust-vignette-effect.ts           # Custom vignette for additive blending
├── detector-noise-effect.ts            # Physics-based detector noise
├── gpu-buffer-pool.ts                  # Geometry reuse with size-based bucketing and LRU eviction
├── adaptive-dpr-manager.ts             # Dynamic resolution scaling based on real-time FPS
├── postprocessing-types.ts             # Type utilities and depth mapper
├── SPECIFICATIONS.md                   # Technical specification
└── README.md                           # This documentation
```

---

## Getting Started

### Step 1: Initialize Post-Processing

```typescript
import { PostProcessingManager } from './rendering/post-processing-manager';

const postProcessing = new PostProcessingManager(renderer, scene, camera, {
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
postProcessing.setToneMapping('AgX');

// Or try others:
// 'ACES' - Cinematic with warm tones
// 'Reinhard' - Classic, simple
// 'Linear' - No tone mapping
```

### Step 4: Add Anti-Aliasing

```typescript
// SMAA: Best quality (recommended for static scenes)
postProcessing.setSMAAEnabled(true);

// OR FXAA: Faster (better for real-time interaction)
postProcessing.setFXAAEnabled(true);
```

### Step 5: Use Quality Presets (Optional)

```typescript
// Quick setup for different performance targets
postProcessing.setQualityPreset('high'); // All effects, high settings
postProcessing.setQualityPreset('medium'); // Balanced
postProcessing.setQualityPreset('low'); // Performance priority
```

**You're done!** Your scene now has professional HDR rendering with bloom, tone mapping, and anti-aliasing.

---

## Components

### 1. PostProcessing Manager (pmndrs)

The `PostProcessingManager` leverages the pmndrs/postprocessing library for state-of-the-art visual effects.

**Key Advantages:**

- Single-pass effect composition for optimal performance
- Automatic effect merging to minimize draw calls
- Professional-grade effects out of the box
- Active community and regular updates

**Core Effects:**

```typescript
// Initialize with HDR support
const postProcessing = new PostProcessingManager(
  renderer,
  scene,
  camera,
  { width, height }
);

// Configure bloom
postProcessing.updateBloomSettings(
  strength: 0.3,
  radius: 0.85,
  threshold: 0.01
);

// Set tone mapping
postProcessing.setToneMapping(THREE.ACESFilmicToneMapping);

// Enable advanced effects
postProcessing.setAOEnabled(true, 'medium');
postProcessing.setVignetteEnabled(true, 0.5, 0.5);
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

- **Semicircle Kernel Model**: Parabolic intensity falloff (1 - p²)^sharpness
- **Seamless Joints**: Cap factor calculation ensures correct additive blending at joints
- **World-Space Width**: Lines have consistent thickness regardless of distance
- **nD Clipping**: Clipped endpoints use full intensity for correct visual appearance
- **Per-Vertex Attributes**: Color, width, and sharpness interpolate along segments
- **Aspect-Ratio Correct**: Perpendicular direction computed in pixel space for correct line width
- **Anti-Aliasing for Thin Lines**: Minimum pixel width (1.5px) prevents sub-pixel rendering gaps; intensity scaling preserves visual weight of thin lines; smooth edge falloff using smoothstep

**Architecture Note:** Lines use `THREE.Mesh` with `InstancedBufferGeometry` (not `THREE.InstancedMesh`) to avoid exceeding WebGL's 16 attribute location limit.

### 4. GSplat Material

Specialized shader material for volumetric Gaussian splatting with nD slicing support.

**Key Features:**

- **Oriented Anisotropic Gaussian**: Full 3D covariance via Cholesky factors
- **Perspective Projection**: Projects 3D covariance to 2D screen space using Jacobian
- **Generalized Gaussian Falloff**: `exp(-½ · r^sharpness)` for artistic control
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

Singleton manager for efficient material creation and caching (supports both points and lines).

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

**Global Updates**: When camera settings change, MaterialManager automatically updates ALL registered materials - no manual scene traversal needed. Global exposure/offset/gamma are handled by the LuxarToneMappingEffect post-processing pass, not per-material.

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

- Size-based bucketing: reuses geometries when size AND type match (0ms GPU allocation)
- In-place attribute updates via `TypedArray.set()`
- LRU eviction after 300 frames of non-use
- Multi-type support: Points (Float32), Lines (Float32 + Uint8), GSplats (Float32)

### 7. Adaptive DPR Manager

The `AdaptiveDPRManager` dynamically adjusts device pixel ratio based on real-time FPS, trading resolution for frame rate when needed.

**Algorithm:**

- Samples FPS using a 1-second sliding window, evaluated every 500ms
- Scales DPR down when FPS drops below `minFPS`
- Scales DPR up when FPS exceeds `maxFPS` for `hysteresisSeconds`
- DPR clamped between `minDPR` and `window.devicePixelRatio`

### 8. Luxar Tone Mapping Effect

Vendored from pmndrs/postprocessing with injected Exposure-Offset-Gamma (EOG) uniforms applied in a single shader pass before tone mapping. Zero extra bandwidth cost.

**EOG Uniforms:**

- `exposure`: Log2 stops (`color * 2^exposure`)
- `global_offset`: Additive shift (`color + offset`)
- `global_gamma`: Power curve (`pow(color, 1/gamma)`)

---

## Effects Library

### Core Effects

#### Bloom

Professional bloom effect with HDR support:

- Luminance threshold for selective blooming
- Configurable intensity and radius
- Mipmap blur with adjustable levels (1-12)
- Multiple kernel sizes
- Performance/quality tradeoff via mipmap levels

#### Tone Mapping

Multiple tone mapping operators:

- ACES Filmic (default) - Industry standard
- AgX - Modern alternative
- Reinhard - Classic operator
- Linear - No tone mapping
- Neutral - Balanced look

#### Ambient Occlusion (SSAO)

Screen-space ambient occlusion for depth:

- Multiple quality levels
- Configurable radius and intensity
- Luminance-based influence
- Minimal performance impact

### Anti-Aliasing

#### FXAA

Fast Approximate Anti-Aliasing:

- Very fast performance
- Good quality for most cases
- Single-pass implementation
- **Recommended for general use**
- Works perfectly with additive blending

#### SMAA

Subpixel Morphological Anti-Aliasing:

- Superior edge detection
- Multiple quality presets (LOW, MEDIUM, HIGH, ULTRA)
- Better quality than FXAA
- Moderate performance impact
- **Best quality/performance balance**
- Compatible with additive blending

#### MSAA

Multisample Anti-Aliasing:

- Hardware-accelerated
- Sample counts: 2x, 4x, 8x
- **⚠️ WARNING**: Incompatible with additive blending
- Causes brightness multiplication artifacts with points
- Only use with normal blending mode
- Automatically validates GPU support

#### SSAA

Super-Sample Anti-Aliasing:

- Renders at higher resolution (1.5x, 2x, 3x, 4x)
- Best possible quality
- **Heavy performance cost**
- Recommended only for screenshots or high-end GPUs
- Properly manages renderer and composer sizes

### Cinematic Effects

#### Depth of Field

Realistic camera focus simulation:

- Configurable focus distance
- Bokeh scale adjustment
- Performance-optimized

#### Vignette

Screen edge darkening:

- Adjustable darkness
- Configurable offset
- Minimal performance cost
- **Custom Implementation**: Uses `RobustVignetteEffect` to prevent alpha overflow artifacts with additive blending and Float16 HDR buffers

#### Chromatic Lens Distortion

Physically accurate lens distortion with wavelength-dependent chromatic aberration:

- **Combined effect**: Replaces separate lens distortion + chromatic aberration
- **Wavelength-dependent distortion**: Blue refracts more than red (optical dispersion)
- **Realistic chromatic fringing**: Follows lens geometry (stronger at edges)
- Full camera model: Distortion, principal point, focal length, skew
- Barrel/pincushion distortion for wide angle/telephoto simulation
- **More efficient**: 3 texture samples in single pass vs separate effects
- **Custom Implementation**: See `chromatic-lens-distortion-effect.ts`

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
2. **Quality Priority**: Enable **SMAA** with HIGH preset
3. **Maximum Quality**: Enable **SSAA** at 2x (heavy performance cost)
4. **Avoid MSAA**: Due to additive blending incompatibility

### Troubleshooting

#### SSAA Issues

- If viewport appears cropped, restart the viewer
- Performance impact scales quadratically with multiplier

#### MSAA Not Working

- Check console for GPU support warnings
- MSAA requires WebGL2 with float buffer extensions
- Will not show effect with additive blending
- Try switching to normal blending mode to verify

#### Performance Tips

- Start with FXAA for best performance
- SMAA provides best quality/performance ratio
- SSAA should only be used for final renders
- Monitor FPS when enabling AA effects

---

## Pipeline Architecture

### Rendering Flow

```
Scene Geometry
    ↓
Custom Point Shaders (HDR colors)
    ↓
HDR Render Target (HalfFloatType)
    ↓
Dynamic Pass Assignment Algorithm:

1. Process effects in order: Bloom → DOF → AO → Vignette → ChromaticLensDistortion → DetectorNoise → ToneMapping → AA
2. Add effects sequentially to Pass A until incompatibility detected
3. When incompatibility found, switch to Pass B for that effect and ALL remaining effects
4. Pass A (if exists) → Pass B (if exists) → Final Output

Example Scenarios:
┌─ No Incompatibilities ─┐     ┌─ Incompatibilities Detected ─┐
│ Pass A:                │     │ Pass A:                       │
│ ├── Bloom              │     │ ├── Bloom                     │
│ ├── Vignette           │     │ └── Vignette                  │
│ ├── Noise              │     │                               │
│ └── Tone Mapping       │     │ Pass B:                       │
│                        │     │ ├── Chromatic Lens Dist (UV) │
│ Final Output           │     │ ├── Noise                     │
└────────────────────────┘     │ └── Tone Mapping             │
                               │                               │
                               │ Pass A → Pass B → Final       │
                               └───────────────────────────────┘
```

### Dynamic Pass System

The renderer automatically handles effect incompatibilities using a **sequential pass assignment algorithm**:

1. **Sequential Processing**: Effects are processed in their correct visual order
2. **Simplified Pipeline**: ChromaticLensDistortion combines UV transformation with chromatic effect, eliminating incompatibility issues
3. **Pass Switching**: Upon incompatibility, all remaining effects (including the incompatible one) are moved to Pass B
4. **Final Pass Logic**: Pass B always contains tone mapping when it exists, ensuring proper HDR→LDR conversion
5. **Single vs Dual Pass**: If no incompatibilities exist, only Pass A is used; otherwise Pass A feeds into Pass B

### Performance Optimizations

1. **Effect Merging**: Multiple effects rendered in single pass
2. **Smart Rebuilding**: Effect pass only rebuilt when necessary
3. **Material Caching**: Reuse materials with same properties
4. **Selective AA**: Choose AA method based on performance
5. **Quality Presets**: Easy performance/quality tradeoffs

---

## Configuration

### Quality Presets

```typescript
// Low quality (60+ FPS target)
{
  bloom: { kernelSize: KernelSize.SMALL },
  ao: null,  // Disabled
  aa: 'FXAA'
}

// Medium quality (30+ FPS target)
{
  bloom: { kernelSize: KernelSize.MEDIUM },
  ao: { quality: 'low' },
  aa: 'SMAA'
}

// High quality (best visuals)
{
  bloom: { kernelSize: KernelSize.LARGE },
  ao: { quality: 'high' },
  aa: 'SMAA'
}
```

### HDR Configuration

```typescript
// Renderer setup for HDR
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;

// Composer with HDR buffers
new EffectComposer(renderer, {
  frameBufferType: THREE.HalfFloatType,
});
```

---

## Usage Examples

### Basic Setup

```typescript
import { PostProcessingManager } from './rendering/post-processing-manager';
import { materialManager } from './rendering/material-manager';

// Initialize post-processing
const postProcessing = new PostProcessingManager(renderer, scene, camera, {
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
postProcessing.setAOEnabled(true, 'medium');
postProcessing.setFXAAEnabled(true);

// Add cinematic effects
postProcessing.setDOF(true, 10.0, 0.5);
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

### Performance Impact (1M points, 1080p)

| Effect         | Performance Cost |
| -------------- | ---------------- |
| Base Rendering | ~5ms             |
| Bloom          | ~2ms             |
| Tone Mapping   | ~0.5ms           |
| FXAA           | ~0.5ms           |
| SMAA           | ~1-2ms           |
| SSAO (medium)  | ~2-3ms           |
| DOF            | ~2-3ms           |
| Vignette       | <0.1ms           |

---

## Troubleshooting

### Common Issues

**Problem: Black screen after enabling effects**

- Check browser console for WebGL errors
- Verify HDR buffer support: `renderer.capabilities.isWebGL2`
- Try disabling effects one by one to isolate the issue
- Check if depth buffer is available (required for DOF, SSAO)
- Verify tone mapping is enabled (required for HDR pipeline)

**Problem: Poor performance with all effects**

- Use quality presets: `setQualityPreset('medium')` or `'low'`
- Disable SSAO first (highest cost: ~2-3ms)
- Use FXAA instead of SMAA (FXAA < 0.5ms, SMAA ~1-2ms)
- Reduce bloom mipmap levels: `setBloomLevels(3)` instead of default 8
- Disable detector noise if not needed
- Lower SSAA multiplier or disable: `setSSAAEnabled(false)`
- Monitor with: `getPerformanceMetrics()` to identify bottlenecks

**Problem: Colors look wrong**

- Verify tone mapping operator: try 'AgX' or 'ACES Filmic' instead of 'Reinhard'
- Check exposure value in HDR controls
- Ensure proper color space: `renderer.outputColorSpace = THREE.SRGBColorSpace`
- Verify bloom threshold isn't too low (washing out colors)
- Check gamma correction in materials (should be 1.0 for linear workflow)

**Problem: Effects not visible**

- Check effect enabled state in debug console
- Verify threshold values (bloom threshold too high, AO intensity too low)
- Ensure proper effect order (tone mapping must be last)
- Check if effect is in compatible pass (see incompatibility warnings)
- Verify camera near/far planes for depth-dependent effects (DOF, SSAO)

**Problem: Thin lines have aliasing/gaps**

- Line material automatically handles this with 1.5px minimum width
- Ensure anti-aliasing is enabled (SMAA or FXAA)
- Check that line widths are properly set (not zero or NaN)
- For very thin lines, increase width slightly or use higher SSAA

**Problem: Bright artifacts in dark areas (additive blending)**

- This is caused by alpha overflow in Float16 buffers
- Solution: RobustVignetteEffect is automatically used (prevents this issue)
- If you see this with custom effects, ensure alpha is clamped to 1.0

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
- Disable SSAO (requires additional buffers)
- Use lower SSAA multiplier
- Consider using lower encoding mode (AGGRESSIVE)

---

## API Reference

### PostProcessingManager

| Method                                                    | Description                                          |
| --------------------------------------------------------- | ---------------------------------------------------- |
| `render()`                                                | Execute rendering pipeline                           |
| `updateBloomSettings(strength, radius, threshold)`        | Configure bloom                                      |
| `setToneMapping(type)`                                    | Set tone mapping operator                            |
| `setDetectorNoiseEnabled(enabled, sigma, gain, fpnSigma)` | Configure physics-based detector noise               |
| `updateDetectorNoiseSettings(params)`                     | Update detector noise parameters                     |
| `setFXAAEnabled(enabled)`                                 | Toggle FXAA                                          |
| `setSMAAEnabled(enabled)`                                 | Toggle SMAA                                          |
| `setAOEnabled(enabled, quality)`                          | Configure ambient occlusion                          |
| `setDOF(enabled, focus, strength)`                        | Configure depth of field                             |
| `setVignetteEnabled(enabled, darkness, offset)`           | Configure vignette                                   |
| `setChromaticLensDistortionEnabled(enabled, ...params)`   | Configure chromatic lens distortion                  |
| `updateChromaticLensDistortion(params)`                   | Update chromatic lens distortion params              |
| `setQualityPreset(preset)`                                | Set quality preset: 'low', 'medium', 'high', 'ultra' |
| `setBloomLevels(levels)`                                  | Set bloom mipmap levels (1-12)                       |
| `setSSAAEnabled(enabled)`                                 | Toggle SSAA                                          |
| `setSSAAMultiplier(multiplier)`                           | Set SSAA multiplier (1.5-4.0)                        |
| `startDeferRebuild()` / `endDeferRebuild()`               | Defer rebuilds during bulk changes                   |
| `getPerformanceMetrics()`                                 | Get FPS, frame time, memory usage                    |
| `needsContinuousAnimation()`                              | Check if effects need animation                      |
| `resize(width, height)`                                   | Update render size                                   |
| `dispose()`                                               | Clean up resources                                   |

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

1. **Use pmndrs effects** when available
2. **Create custom effects** following pmndrs patterns
3. **Test performance** across different hardware
4. **Document settings** and performance impact
5. **Maintain HDR pipeline** integrity

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

_For implementation details, see the source files in this directory._
