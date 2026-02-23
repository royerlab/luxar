# luxar-viewer.rendering - Technical Specification

**Version**: 1.3.7
**Last Updated**: 2025-12-17

## Purpose

The `luxar-viewer.rendering` package provides advanced WebGL rendering capabilities including HDR post-processing pipeline, custom point and line materials with world-space sizing, and material management for data visualization.

**Core Responsibility**: Deliver professional-grade visual effects through pmndrs/postprocessing library integration, custom shaders for physically accurate point and line rendering, and efficient material caching.

**Related Specifications**:

- `luxar-viewer.scene` - Scene management integration (see `../scene/SPECIFICATIONS.md`)

---

## Table of Contents

1. [HDR Rendering Pipeline](#hdr-rendering-pipeline)
2. [Point Material System](#point-material-system)
3. [World-Space Point Sizing](#world-space-point-sizing)
4. [Post-Processing Effects](#post-processing-effects)
5. [Material Management](#material-management)
6. [Anti-Aliasing](#anti-aliasing)
7. [Line Material System](#line-material-system)
8. [GSplat Material System](#gsplat-material-system)
9. [Adaptive Resolution System](#adaptive-resolution-system)

---

## 1. HDR Rendering Pipeline

### 1.1 Render Target Configuration

**HDR Support**: Use `THREE.HalfFloatType` (16-bit float) render targets to support HDR color values > 1.0.

**Setup**:

```typescript
const composer = new EffectComposer(renderer, {
  frameBufferType: THREE.HalfFloatType, // 16-bit float for HDR
  multisampling: 0, // MSAA incompatible with additive blending
});
```

**Color Space Pipeline**:

```
Scene Rendering → HDR Buffer (LinearSRGB) → Effects → Tone Mapping → Output (SRGB)
```

### 1.2 Tone Mapping

**Purpose**: Map HDR colors (unbounded) to LDR display range [0, 1].

**Tone Mapping Operators**:

| Operator        | Description                 | Characteristics                 |
| --------------- | --------------------------- | ------------------------------- |
| **ACES Filmic** | Industry standard (default) | Smooth highlights, natural look |
| **AgX**         | Modern alternative          | Balanced, film-like             |
| **Reinhard**    | Classic operator            | Simple, local adaptation        |
| **Linear**      | No mapping                  | Raw HDR (clips >1)              |
| **Neutral**     | Balanced                    | Minimal color shift             |

**Implementation** (via pmndrs/postprocessing):

```typescript
// Tone mapping applied as final pass
const toneMappingEffect = new ToneMappingEffect({
  mode: ToneMappingMode.ACES_FILMIC,
  resolution: 256,
  adaptive: false,
});
```

### 1.3 Effect Composition Strategy

**Dynamic Pass Assignment Algorithm**:

```typescript
function buildEffectPasses(enabledEffects: Effect[]): Pass[] {
  const passes: Pass[] = [];

  // Effects in visual order
  const orderedEffects = [
    bloom,
    dof,
    ao,
    vignette,
    chromaticLensDistortion, // Combined effect (replaces old chromaticAberration + lensDistortion)
    detectorNoise,
    toneMapping,
    aa,
  ].filter((e) => e && e.enabled);

  // Note: ChromaticLensDistortionEffect combines wavelength-dependent lens distortion
  // with chromatic aberration in a single UV transformation pass. This is more efficient
  // and physically accurate than separate effects.

  // Detect incompatibilities (currently none with simplified effect set)
  let switchToPassB = false;
  const passA: Effect[] = [];
  const passB: Effect[] = [];

  for (const effect of orderedEffects) {
    // UV transformation effects would be incompatible with convolution effects
    // Currently we use ChromaticLensDistortion (UV-based) instead of separate effects
    if (switchToPassB) {
      passB.push(effect);
    } else {
      passA.push(effect);
    }
  }

  // Create passes
  if (passA.length > 0) {
    passes.push(new EffectPass(camera, ...passA));
  }
  if (passB.length > 0) {
    passes.push(new EffectPass(camera, ...passB));
  }

  return passes;
}
```

**Invariant**: Tone mapping always in final pass to ensure proper HDR→LDR conversion.

---

## 2. Point Material System

### 2.1 Custom Point Material

**Base**: `THREE.ShaderMaterial` with custom vertex and fragment shaders

**Attributes**:

- `position`: vec3 - Point center in object space
- `color`: vec3 - RGB color (or HDR)
- `radius`: float - World-space radius
- `sharpness`: float - Edge falloff power

**Uniforms**:

- `uFOV`: float - Camera field of view (radians)
- `uResolution`: vec2 - Framebuffer resolution [width, height]
- `uHDRMultiplier`: float - HDR boost factor (typ. 16.0)
- `uOpacity`: float - Global opacity multiplier
- `uGamma`: float - Gamma correction factor

### 2.2 Vertex Shader

**Purpose**: Transform points to screen space and calculate world-space point sizes.

**Algorithm**:

```glsl
// Vertex Shader
attribute vec3 position;     // Point center (world space)
attribute vec3 color;         // Point color
attribute float radius;       // World-space radius
attribute float sharpness;    // Edge falloff

uniform float uFOV;           // Camera FOV (radians)
uniform vec2 uResolution;     // [width, height]

varying vec3 vColor;
varying float vSharpness;

void main() {
    // Transform to clip space
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    // Calculate world-space point size
    float distance = length(mvPosition.xyz);

    // Direct formula (equivalent to angular calculation for small angles):
    // pixelSize = 2 * radius * resolution.y / (distance * tan(fov/2))
    float basePointSize = 2.0 * radius * uResolution.y / (distance * tan(uFOV * 0.5));

    // Sharpness compensation (linear approximation):
    // compensation = 1.0 + (sharpness - 1.0) * 0.15
    // This gives: s=1→1.0, s=2→1.15, s=4→1.45, s=8→2.05
    float sharpnessCompensation = 1.0 + (sharpness - 1.0) * 0.15;
    float pixelSize = basePointSize * sharpnessCompensation;

    gl_PointSize = max(1.0, min(pixelSize, uResolution.y * 0.5));

    // Pass attributes to fragment shader
    vColor = color;
    vSharpness = sharpness;
}
```

**Critical Details**:

1. **Angular size formula**: `θ = 2 * atan(r / d)` ensures correct perspective scaling
2. **Sharpness compensation**: Enlarges point so that ~85% of energy is within nominal radius
3. **FOV-independent**: Point size calculated from angular extent, not FOV

### 2.3 Fragment Shader

**Purpose**: Render points with smooth falloff and HDR color support.

**Algorithm**:

```glsl
// Fragment Shader
varying vec3 vColor;
varying float vSharpness;

uniform float uHDRMultiplier;
uniform float uOpacity;
uniform float uGamma;

void main() {
    // Distance from point center (in [0, 1])
    vec2 cxy = 2.0 * gl_PointCoord - 1.0;
    float r = length(cxy);

    // Discard fragments outside point
    if (r > 1.0) discard;

    // Power-based falloff: intensity = (1 - r^sharpness)
    float intensity = pow(1.0 - r, vSharpness);

    // Apply HDR multiplier for bloom
    vec3 color = vColor * uHDRMultiplier;

    // Apply gamma correction
    color = pow(color, vec3(1.0 / uGamma));

    // Apply intensity falloff
    color *= intensity;

    // Output with opacity
    gl_FragColor = vec4(color, intensity * uOpacity);
}
```

**Blending Configuration**:

```typescript
material.blending = THREE.AdditiveBlending; // For overlapping points
material.transparent = true;
material.depthWrite = false; // Allow proper blending
material.depthTest = true; // Respect depth buffer
```

---

## 3. World-Space Point Sizing

### 3.1 Mathematical Foundation

**Goal**: Two points with radius `r` separated by distance `2r` should visually touch.

**Approach**: Calculate point size based on **angular extent** rather than screen-space pixels.

### 3.2 Point Size Calculation

**Direct Formula** (optimized implementation):

```
pixelSize = 2 × radius × resolution.y / (distance × tan(FOV / 2))
```

**Derivation**: For perspective projection, a world-space radius `r` at distance `d` subtends an angle:

```
θ = 2 × arctan(r / d)     // Angular diameter

For small angles: arctan(x) ≈ x
θ ≈ 2r / d

Vertical FOV φ = 2 × arctan(h / (2d)) where h = image plane height
For small angles: φ ≈ h / d

Therefore: tan(φ/2) ≈ h / (2d)

Pixel size = (θ / φ) × resolution.y
           = (2r/d) / (h/d) × resolution.y
           = 2r × resolution.y / h
           = 2r × resolution.y / (d × tan(φ/2))
```

**Rationale**: This direct formula avoids expensive `atan()` calls in the shader while maintaining physical accuracy.

### 3.3 Sharpness Compensation

**Problem**: Soft-edged points appear smaller than their nominal radius.

**Solution**: Enlarge the point to account for falloff at edges.

**Model** (linear approximation for GPU efficiency):

```
compensation = 1.0 + (sharpness - 1.0) × 0.15
```

**Effect**:

- s=1: compensation = 1.0 (no enlargement for sharp points)
- s=2: compensation = 1.15 (15% enlargement)
- s=4: compensation = 1.45 (45% enlargement)
- s=8: compensation = 2.05 (105% enlargement)

**Rationale**: Linear approximation avoids expensive `pow()` operations while providing visually acceptable results. The constant 0.15 is empirically tuned for good appearance.

### 3.4 Resolution Handling

**Critical**: Use **actual framebuffer resolution**, not canvas resolution.

```typescript
// Correct: Account for device pixel ratio
const actualHeight = canvas.height; // Framebuffer pixels
const actualWidth = canvas.width;

material.uniforms.uResolution.value.set(actualWidth, actualHeight);

// Incorrect: Canvas CSS pixels (doesn't match framebuffer)
// const cssHeight = canvas.clientHeight
```

**Rationale**: `gl_PointSize` is specified in framebuffer pixels, which includes `devicePixelRatio` scaling.

---

## 4. Post-Processing Effects

### 4.1 Bloom Effect

**Purpose**: Simulate light scattering for bright objects (HDR colors > 1.0).

**Algorithm** (via pmndrs/postprocessing):

```typescript
const bloomEffect = new BloomEffect({
  intensity: 0.5, // Bloom strength
  luminanceThreshold: 0.01, // Brightness threshold
  luminanceSmoothing: 0.9, // Threshold smoothing
  mipmapBlur: true, // Use mipmap blur (better quality)
  levels: 8, // Mipmap levels (1-12)
});

// Separate blur pass for radius control
bloomEffect.mipmapBlurPass.radius = 0.6;
```

**Key Parameters**:

- **Threshold**: Only colors with luminance > threshold bloom
- **Intensity**: Bloom contribution to final image
- **Radius**: Blur extent (larger = more spread)
- **Levels**: More levels = smoother bloom, higher cost

### 4.2 Ambient Occlusion (SSAO)

**Purpose**: Screen-space ambient occlusion for depth perception.

**Algorithm**:

```typescript
const ssaoEffect = new SSAOEffect(camera, normalBuffer, {
  samples: 16, // Sample count (higher = better quality)
  radius: 0.1, // Occlusion radius
  intensity: 1.0, // Effect strength
  luminanceInfluence: 0.7, // How much lighting affects AO
});
```

**Quality Levels**:

- Low: samples=8, radius=0.05
- Medium: samples=16, radius=0.1
- High: samples=32, radius=0.15

### 4.3 Depth of Field (DOF)

**Purpose**: Simulate camera focus with bokeh blur.

```typescript
const dofEffect = new DepthOfFieldEffect(camera, {
  focusDistance: 10.0, // Focus plane distance
  focalLength: 0.05, // Lens focal length
  bokehScale: 2.0, // Bokeh blur size
});
```

### 4.4 Detector Noise Effect (Physics-Based)

**Purpose**: Simulate realistic camera/detector noise for scientific imaging aesthetics.

**Physics Model**: Combined shot noise (Poisson) + readout noise (Gaussian, temporal) + fixed pattern noise (Gaussian, static)

```
I_observed = Poisson(I_true / gain) × gain + Gaussian_temporal(0, σ_read²) + FPN(pixel)
```

**Three Noise Components**:

1. **Shot Noise (Poisson)**: Signal-dependent noise from photon statistics
2. **Readout Noise (Gaussian, temporal)**: Signal-independent electronic noise, varies per frame
3. **Fixed Pattern Noise (Gaussian, static)**: Per-pixel offset from detector non-uniformities (dark current, gain variations)

**Algorithm Components**:

1. **Bob Jenkins Hash**: Fast deterministic PRNG from pixel position + time
2. **Clamped Logistic Distribution**: Efficient Gaussian approximation (faster than Box-Muller)
3. **Anscombe Transform**: Variance-stabilizing for Poisson approximation

**Implementation**:

```typescript
const detectorNoiseEffect = new DetectorNoiseEffect({
  readoutSigma: 0.002, // Temporal readout noise sigma (0-0.1)
  photonGain: 0.002, // Controls shot noise visibility (0.0001-0.1)
  fpnSigma: 0.001, // Fixed pattern noise sigma (0-0.05)
});
```

**Default Values** (as of v1.3.7):

| Parameter    | Default | Range        |
| ------------ | ------- | ------------ |
| readoutSigma | 0.002   | 0 - 0.1      |
| photonGain   | 0.002   | 0.0001 - 0.1 |
| fpnSigma     | 0.001   | 0 - 0.05     |

**Mathematical Details**:

**Clamped Logistic** (Gaussian approximation):

```glsl
// Faster than Box-Muller (no sqrt, no trig)
float clampedLogistic(float u) {
  float f = clamp(u, 0.0001, 0.9999);
  return clamp(log(f / (1.0 - f)), -4.0, 4.0) * 0.5513;
}
```

**Anscombe Transform** (Poisson→Gaussian):

```glsl
// Forward: Y = 2 × sqrt(X + 3/8) → ~N(2√λ, 1)
float anscombeForward(float x) { return 2.0 * sqrt(x + 0.375); }

// Inverse: X = (Y/2)² - 3/8
float anscombeInverse(float y) { return (y * 0.5)² - 0.375; }
```

**Fixed Pattern Noise** (static per-pixel):

```glsl
// Uses only UV coordinates (no time), so pattern is constant across frames
vec3 normal3_fixed(vec2 seed) {
  return vec3(
    clampedLogistic(rngfloat2(seed)),
    clampedLogistic(rngfloat2(seed + vec2(13.37, 7.31))),
    clampedLogistic(rngfloat2(seed + vec2(31.17, 41.23)))
  );
}
```

**Use Cases**:

| Scenario                | readoutSigma | photonGain | fpnSigma |
| ----------------------- | ------------ | ---------- | -------- |
| Bright field microscopy | 0.005        | 0.001      | 0.002    |
| Low-light fluorescence  | 0.01         | 0.05       | 0.005    |
| Single-molecule imaging | 0.02         | 0.1        | 0.01     |
| Old/uncooled detector   | 0.05         | 0.01       | 0.03     |
| Cinematic film look     | 0.015        | 0.008      | 0.003    |

**DPR-Based Noise Scaling** (as of v1.3.7):

When rendering at reduced resolution (DPR < 1.0) for performance, noise parameters are automatically scaled to maintain perceptual consistency.

**Problem**: At lower DPR, each rendered pixel represents multiple screen pixels. When the image is upscaled, the noise becomes visually coarser/grainier than intended.

**Solution**: Scale noise parameters to maintain equivalent perceived noise:

```typescript
// Mathematical basis:
// When DPR < 1, each rendered pixel covers 1/DPR² screen pixels
// Averaging N noisy samples reduces σ by √N
// So for Gaussian noise: σ_effective = σ × DPR
//
// For shot noise (Poisson via Anscombe), the output σ ∝ √photonGain
// To scale σ by DPR: photonGain_effective = photonGain × DPR²

private applyScaledNoiseSettings(): void {
  const scale = this.currentDPRScale; // e.g., 0.5 for 50% resolution

  // Gaussian noise: σ scales linearly with DPR
  this.detectorNoiseEffect.readoutSigma = this.baseNoiseSettings.readoutSigma * scale;
  this.detectorNoiseEffect.fpnSigma = this.baseNoiseSettings.fpnSigma * scale;

  // Shot noise: σ ∝ √photonGain, so photonGain scales by DPR²
  this.detectorNoiseEffect.photonGain = this.baseNoiseSettings.photonGain * scale * scale;
}

setDPRScale(dpr: number): void {
  const scale = Math.max(0.25, Math.min(1.0, dpr));
  if (Math.abs(scale - this.currentDPRScale) < 0.01) return;

  this.currentDPRScale = scale;
  this.applyScaledNoiseSettings();
}
```

**Behavior** (base values: readoutSigma=0.002, photonGain=0.002, fpnSigma=0.001):

| DPR  | Effective readoutSigma | Effective photonGain | Effective fpnSigma | Noise σ Reduction |
| ---- | ---------------------- | -------------------- | ------------------ | ----------------- |
| 1.0  | 0.002 (×1.0)           | 0.002 (×1.0)         | 0.001 (×1.0)       | 100%              |
| 0.75 | 0.0015 (×0.75)         | 0.001125 (×0.5625)   | 0.00075 (×0.75)    | 75%               |
| 0.5  | 0.001 (×0.5)           | 0.0005 (×0.25)       | 0.0005 (×0.5)      | 50%               |
| 0.25 | 0.0005 (×0.25)         | 0.000125 (×0.0625)   | 0.00025 (×0.25)    | 25%               |

**Integration**: Called from `SceneManager.setAdaptivePixelRatio()` when DPR changes:

```typescript
// In scene-manager.ts
if (this.postProcessing) {
  const nativeDPR = window.devicePixelRatio;
  const normalizedDPR = dpr / nativeDPR;
  this.postProcessing.setDPRScale(normalizedDPR);
}
```

### 4.5 ChromaticLensDistortionEffect (Custom Implementation)

**Purpose**: Physically accurate lens distortion with wavelength-dependent chromatic aberration.

**Physical Basis**:

- Refractive index varies with wavelength (Abbe dispersion)
- Shorter wavelengths (blue ~450nm) refract more than longer wavelengths (red ~650nm)
- This causes different focal lengths and distortion amounts per color channel
- Color fringing naturally follows lens geometry (stronger where distortion is greater)

**Advantages over Separate Effects**:

- ✅ More efficient: 3 texture samples vs separate passes for lens distortion + chromatic aberration
- ✅ More realistic: Chromatic fringing follows radial distortion pattern (stronger at edges)
- ✅ Unified control: Single dispersion parameter controls chromatic effect strength
- ✅ No pass incompatibility: Single UV transformation combines both effects

**Mathematical Model**: Brown-Conrady distortion with wavelength-dependent coefficients

```glsl
// Apply different distortion per color channel
vec2 distortionR = distortion * (1.0 - dispersion);  // Red: least distortion
vec2 distortionG = distortion;                        // Green: reference
vec2 distortionB = distortion * (1.0 + dispersion);  // Blue: most distortion

// Radial distortion: r' = r * (1 + k * r²)
vec2 xn = 2.0 * (uv - 0.5);  // Normalize to [-1, 1]
float r2 = dot(xn, xn);
vec2 xDistorted_R = (1.0 + distortionR * r2) * xn;
vec2 xDistorted_G = (1.0 + distortionG * r2) * xn;
vec2 xDistorted_B = (1.0 + distortionB * r2) * xn;

// Apply camera intrinsic matrix K:
// | fx   s*fx  cx |
// | 0    fy    cy |
// | 0    0     1  |
mat3 kk = mat3(
  vec3(focalLength.x, 0.0, 0.0),
  vec3(skew * focalLength.x, focalLength.y, 0.0),
  vec3(principalPoint.x, principalPoint.y, 1.0)
);

// Sample each channel independently
float r = texture2D(inputBuffer, (kk * vec3(xDistorted_R, 1.0)).xy * 0.5 + 0.5).r;
float g = texture2D(inputBuffer, (kk * vec3(xDistorted_G, 1.0)).xy * 0.5 + 0.5).g;
float b = texture2D(inputBuffer, (kk * vec3(xDistorted_B, 1.0)).xy * 0.5 + 0.5).b;
```

**Implementation**:

```typescript
import { ChromaticLensDistortionEffect } from '../rendering/chromatic-lens-distortion-effect';

const chromaticLensEffect = new ChromaticLensDistortionEffect({
  distortion: new THREE.Vector2(-0.05, -0.05), // Barrel distortion (wide angle)
  dispersion: 0.03, // Subtle chromatic aberration
  principalPoint: new THREE.Vector2(0, 0), // Centered
  focalLength: new THREE.Vector2(1, 1), // Normal focal length
  skew: 0, // No skew
});
```

**Parameters**:

| Parameter      | Type    | Range       | Default | Description                                    |
| -------------- | ------- | ----------- | ------- | ---------------------------------------------- |
| distortion     | Vector2 | [-1, 1]     | (0, 0)  | Radial distortion (- = barrel, + = pincushion) |
| dispersion     | float   | [0, 0.5]    | 0.0     | Chromatic dispersion strength                  |
| principalPoint | Vector2 | [-1, 1]     | (0, 0)  | Optical center offset                          |
| focalLength    | Vector2 | [0.1, 3]    | (1, 1)  | Focal length scale (< 1 = wide, > 1 = tele)    |
| skew           | float   | [-0.1, 0.1] | 0       | Pixel skew correction (radians)                |

**Realistic Dispersion Values** (FOV Preset Defaults):

| Lens Type     | FOV | Distortion | Dispersion | Rationale                           |
| ------------- | --- | ---------- | ---------- | ----------------------------------- |
| 28mm Wide     | 75° | -0.07      | 0.05       | Wide angle = high light bending     |
| 35mm          | 63° | -0.05      | 0.035      | Moderate wide angle                 |
| 50mm Normal   | 47° | 0.0        | 0.02       | Normal lens = minimal chromatic     |
| 85mm Portrait | 29° | 0.05       | 0.025      | Telephoto = less bending            |
| 135mm Tele    | 18° | 0.07       | 0.03       | Strong telephoto with edge fringing |

**Performance**: ~3ms per frame at 1080p (3 texture samples)

**Visual Characteristics**:

- Color fringing scales with radial distance (stronger at edges)
- Barrel distortion: Red-Cyan fringing (red outside, blue inside)
- Pincushion distortion: Cyan-Red fringing (blue outside, red inside)
- Dispersion = 0.0: Pure lens distortion (no chromatic effect)

### 4.6 RobustVignetteEffect (Custom Implementation)

**Purpose**: Screen edge darkening effect that handles additive blending artifacts.

**Problem Solved**: The standard pmndrs `VignetteEffect` preserves alpha channel values, which causes rendering artifacts when:

- Using additive blending (`THREE.AdditiveBlending`) with many overlapping points/lines
- Alpha values accumulate and can exceed 1.0 or even overflow to `Infinity` in Float16 HDR buffers
- Subsequent effects receive problematic alpha values causing brightness artifacts

**Solution**: `RobustVignetteEffect` forces output alpha to 1.0, preventing alpha overflow artifacts while maintaining identical visual output.

**Implementation**:

```typescript
// Drop-in replacement for VignetteEffect
import { RobustVignetteEffect } from '../rendering/robust-vignette-effect';

const vignetteEffect = new RobustVignetteEffect({
  darkness: 0.5, // Edge darkening amount [0, 1]
  offset: 0.5, // Vignette start distance [0, 1]
});
```

**Key Differences from Standard VignetteEffect**:

- ✅ Identical parameters and visual output
- ✅ Compatible with additive blending
- ✅ Prevents alpha overflow in Float16 HDR buffers
- ✅ No brightness artifacts in dark areas (e.g., vignette edges)

**Technical Detail**: The fix is simple but critical - the fragment shader's final line sets `gl_FragColor.a = 1.0;` instead of preserving the input alpha. This prevents accumulated alpha from additive blending from propagating through the effect pipeline.

**Use Case**: Always use `RobustVignetteEffect` instead of pmndrs `VignetteEffect` when:

- Using additive blending for points or lines
- Using Float16 HDR render targets
- Rendering many overlapping transparent objects

### 4.6 PerspectiveDepthMapper (Utility Class)

**Purpose**: Converts between different depth representations for effects that need depth information (e.g., DOF, SSAO).

**Problem**: THREE.js stores depth in various formats:

- **View-space depth**: Linear distance from camera in world units (negative Z in view space)
- **NDC depth**: Non-linear depth in [0, 1] stored in depth buffer
- **Camera-relative depth**: Distance from camera origin
- **Normalized linear depth**: Linear depth normalized to [near, far] range

Effects like DOF need specific depth representations, requiring conversions between these formats.

**Implementation**:

```typescript
import { PerspectiveDepthMapper } from '../rendering/postprocessing-types';

const mapper = new PerspectiveDepthMapper(camera);

// Four conversion methods:
const viewZ = mapper.getViewZ(ndc); // NDC → view-space Z
const ndcDepth = mapper.getNDC(viewZ); // view-space Z → NDC
const linearDepth = mapper.getLinear(ndc); // NDC → normalized linear
const orthoDepth = mapper.getOrtho(viewZ); // view-space Z → orthographic
```

**Conversion Formulas**:

1. **NDC to View-Space Z**: `getViewZ(ndc)`

   ```glsl
   viewZ = (near * far) / (far - ndc * (far - near))
   ```

   Converts non-linear depth buffer value to linear view-space distance.

2. **View-Space Z to NDC**: `getNDC(viewZ)`

   ```glsl
   ndc = (far * (viewZ - near)) / (viewZ * (far - near))
   ```

   Inverse of the above transformation.

3. **NDC to Normalized Linear**: `getLinear(ndc)`

   ```glsl
   linear = (viewZ - near) / (far - near)
   ```

   Linear depth in [0, 1] where 0 = near plane, 1 = far plane.

4. **View-Space Z to Orthographic**: `getOrtho(viewZ)`
   ```glsl
   ortho = (viewZ + (far + near) / 2) / (far - near)
   ```
   Used for orthographic projections (less common).

**Usage with DOF Effect**:

```typescript
const mapper = new PerspectiveDepthMapper(camera);
const dofEffect = new DepthOfFieldEffect(camera, {
  focusDistance: mapper.getViewZ(0.5), // Focus at middle depth
  // ...other params
});
```

**Key Properties**:

- `near`: Camera near plane distance
- `far`: Camera far plane distance
- All methods handle perspective projection math correctly
- Thread-safe (pure functions based on camera parameters)

**When to Use**:

- Setting DOF focus distances from screen-space depth
- Converting depth buffer values for custom shaders
- Debugging depth-based effects
- Implementing custom depth-dependent effects

---

## 5. Material Management

### 5.1 Material Caching

**Purpose**: Reuse materials with identical properties to reduce memory and shader compilations.

**Cache Key**:

```typescript
function getMaterialCacheKey(config: MaterialConfig): string {
  return `${config.blendingMode}_${config.opacity}_${config.gamma}`;
}
```

**Cache Algorithm**:

```typescript
class MaterialManager {
  private cache = new Map<string, THREE.ShaderMaterial>();

  getPointMaterial(config: MaterialConfig): THREE.ShaderMaterial {
    const key = getMaterialCacheKey(config);

    if (this.cache.has(key)) {
      return this.cache.get(key)!;
    }

    const material = createPointMaterial(config);
    this.cache.set(key, material);
    return material;
  }

  updateGlobalParams(fov: number, resolution: [number, number]): void {
    // Update all cached materials
    for (const material of this.cache.values()) {
      material.uniforms.uFOV.value = fov;
      material.uniforms.uResolution.value.set(resolution[0], resolution[1]);
    }
  }

  dispose(): void {
    for (const material of this.cache.values()) {
      material.dispose();
    }
    this.cache.clear();
  }
}
```

### 5.2 Dynamic Parameter Updates

**Camera Changes**:

```typescript
// When camera FOV changes
materialManager.updateGlobalParams(
  (camera.fov * Math.PI) / 180, // Convert to radians
  [canvas.width, canvas.height]
);
```

**Resolution Changes**:

```typescript
// On window resize
materialManager.updateGlobalParams(
  (camera.fov * Math.PI) / 180,
  [canvas.width, canvas.height] // New resolution
);
```

**HDR Multiplier Initialization**:

When HDR multiplier is updated via `updateHDRMultiplier()`, the MaterialManager stores the current value. New materials created after an HDR update automatically receive the current HDR multiplier value, ensuring consistent brightness across all materials regardless of creation order.

```typescript
// MaterialManager stores current HDR value
private currentHDRMultiplier: number = config.renderingControls.defaults.hdrMultiplier;

updateHDRMultiplier(value: number): void {
  this.currentHDRMultiplier = value;
  // Update all existing materials
  for (const material of this.registeredMaterials) {
    material.uniforms.hdrMultiplier.value = value;
  }
}

getPointMaterial(config: MaterialConfig): PointMaterial {
  // New materials get current HDR value, not config default
  const material = new PointMaterial(config);
  material.uniforms.hdrMultiplier.value = this.currentHDRMultiplier;
  return material;
}
```

---

## 6. Anti-Aliasing

### 6.1 FXAA (Fast Approximate)

**Characteristics**:

- Very fast (< 0.5ms overhead)
- Good quality for most cases
- Single-pass implementation
- Compatible with additive blending

**Setup**:

```typescript
const fxaaEffect = new FXAAEffect();
effectPass.addEffect(fxaaEffect);
```

### 6.2 SMAA (Subpixel Morphological)

**Characteristics**:

- Superior edge detection
- Multiple quality presets (LOW, MEDIUM, HIGH, ULTRA)
- Better quality than FXAA
- Moderate performance impact (1-2ms)
- Compatible with additive blending

**UI Behavior**: The rendering controls UI exposes SMAA as a simple on/off toggle using the HIGH preset. The pmndrs/postprocessing SMAAEffect only supports preset-based configuration, so custom threshold/searchSteps values are not available.

**Setup**:

```typescript
const smaaEffect = new SMAAEffect({
  preset: SMAAPreset.HIGH,
});
effectPass.addEffect(smaaEffect);
```

### 6.3 MSAA (Multisample) - NOT RECOMMENDED

**Problem**: MSAA is **incompatible with additive blending** for points.

**Reason**: MSAA samples are averaged **before** blending, causing incorrect brightness multiplication.

**Example**:

```
Without MSAA: Point A + Point B = 1.0 + 1.0 = 2.0 (correct)
With MSAA:    Avg(1.0, 1.0) + Avg(1.0, 1.0) = 1.0 + 1.0 = 2.0 per sample
              But blending happens per sample, causing 4x brightness
```

**Recommendation**: Use SMAA or FXAA instead.

### 6.4 SSAA (Super-Sample)

**Characteristics**:

- Renders at higher resolution (1.5x, 2x, 3x, 4x)
- Best possible quality
- Heavy performance cost (scales quadratically)
- Only for screenshots or high-end GPUs

**Setup**:

```typescript
// Render at 2x resolution internally
renderer.setSize(width * 2, height * 2, false);
composer.setSize(width * 2, height * 2);

// Display at normal resolution (downsampling provides AA)
canvas.style.width = `${width}px`;
canvas.style.height = `${height}px`;
```

---

## Data Structures

### MaterialConfig

```typescript
interface MaterialConfig {
  blendingMode: 'additive' | 'normal' | 'max';
  opacity: number; // 0.0 to 1.0
  gamma: number; // Typically 1.0 (no correction)
}
```

### PointMaterialUniforms

```typescript
interface PointMaterialUniforms {
  uFOV: { value: number }; // Radians
  uResolution: { value: THREE.Vector2 }; // [width, height]
  uHDRMultiplier: { value: number }; // Typically 16.0
  uOpacity: { value: number }; // 0.0 to 1.0
  uGamma: { value: number }; // Gamma correction
}
```

### PostProcessingConfig

```typescript
interface PostProcessingConfig {
  bloom: {
    enabled: boolean;
    intensity: number;
    threshold: number;
    radius: number;
    levels: number;
  };
  toneMapping: {
    mode: 'ACES' | 'AgX' | 'Reinhard' | 'Linear' | 'Neutral';
  };
  aa: {
    method: 'none' | 'FXAA' | 'SMAA' | 'SSAA';
    smaaQuality?: 'LOW' | 'MEDIUM' | 'HIGH' | 'ULTRA';
    ssaaMultiplier?: 1.5 | 2 | 3 | 4;
  };
  dof: {
    enabled: boolean;
    focusDistance: number;
    bokehScale: number;
  };
  // ... other effects
}
```

### 5.3 Material Lifecycle & Memory Management

**Memory Leak Prevention**: Disposed materials MUST be unregistered from MaterialManager to prevent accumulation in global update lists.

**Problem Without Unregistration**:

```typescript
// Long-running app creates and disposes many materials
for (let i = 0; i < 1000; i++) {
  const material = materialManager.getPointMaterial(config);
  // ... use material ...
  material.dispose(); // WITHOUT unregister()
  // Material remains in MaterialManager.registeredMaterials Set
  // updateCameraParams() still iterates over disposed materials
}
// Result: Memory leak + performance degradation
```

**Solution**:

```typescript
// MaterialManager.unregister() method
unregister(material: THREE.Material): void {
  this.registeredMaterials.delete(material);

  // Also remove from cache if it's a point material
  if (material instanceof PointMaterial) {
    for (const [key, cachedMaterial] of this.pointMaterialCache.entries()) {
      if (cachedMaterial === material) {
        this.pointMaterialCache.delete(key);
        break;
      }
    }
  }
}

// PointMaterial.dispose() override
dispose(): void {
  // Unregister from material manager to prevent memory leaks
  materialManager.unregister(this);

  // Call parent dispose to free GPU resources
  super.dispose();
}
```

**When to Unregister**:

- Material.dispose() called (automatic via override)
- Scene cleared (dispose all objects)
- Dataset switched (remove old materials)

**Impact**: Prevents memory leaks in long-running applications and maintains performance of global material updates.

---

## 7. Line Material System

### 7.1 Mathematical Foundation: Semicircle Kernel Convolution

**Problem**: Lines with additive blending must render seamlessly at joints where two segments meet (a-b-c). A naive approach causes double-brightness at junction points.

**Solution**: Model line intensity as the convolution of a radial kernel with the line path. When two segments share an endpoint, the convolutions naturally combine without double-counting due to associativity.

**Semicircle Kernel Choice**:

```
f(r) = √(1 - (r/R)²)  for r < R, 0 otherwise
```

This kernel is chosen because:

1. Convolution with a line path yields a **parabolic profile** (cheap to compute)
2. The integral at endpoints is exactly **half** the body value
3. Two adjacent endpoints sum to full intensity (correct joint rendering)

**Derivation**:

For a perpendicular distance `p` from the line centerline:

```
I(p) = ∫ f(|x - path(t)|) dt

For a straight segment and semicircle kernel:
I(p) = ∫_{-√(R²-p²)}^{+√(R²-p²)} √(1 - (p² + u²)/R²) du

Result: I(p) = (π/2) × (1 - p²/R²)

Normalized: I(p) / I(0) = 1 - p²/R²
```

**Body Profile**: `intensity = 1 - (p/R)²` where p is perpendicular distance, R is line half-width

**Endpoint Behavior**: At segment endpoints, only half the integration range contributes:

- Body integral: full parabola region
- Endpoint integral: half-disk → exactly half the body intensity

### 7.2 Rendering Approach Decision

**Three options considered**:

| Approach                                   | Pros                        | Cons                                   |
| ------------------------------------------ | --------------------------- | -------------------------------------- |
| **THREE.LineSegments + LineBasicMaterial** | Simple, native THREE.js     | No width control in WebGL (always 1px) |
| **Instanced quads with custom shaders**    | Variable width, cap control | Medium complexity                      |
| **Mesh-based tubes/ribbons**               | Full control, proper 3D     | Expensive for many segments            |

**Recommendation**: Instanced quads with `THREE.Mesh` using `InstancedBufferGeometry`.

**Rationale**:

- WebGL `lineWidth` is deprecated and ignored on most hardware (always 1px)
- Instanced quads allow variable width AND cap factor control for proper joints
- Modern approach, performant for millions of segments

### 7.3 Custom Line Material

**Base**: `THREE.ShaderMaterial` with custom vertex and fragment shaders for instanced quad segments

**Attributes** (per instance/segment):

- `aStartPos`: vec3 - Segment start position (3D display space)
- `aEndPos`: vec3 - Segment end position (3D display space)
- `aStartColor`: vec3 - RGB color at start vertex (HDR)
- `aEndColor`: vec3 - RGB color at end vertex (HDR)
- `aStartWidth`: float - Start vertex half-width (world units)
- `aEndWidth`: float - End vertex half-width (world units)
- `aStartSharpness`: float - Start vertex sharpness
- `aEndSharpness`: float - End vertex sharpness
- `aSegmentLength`: float - World-space length of segment (for cap calculation)
- `aStartClipped`: float - 1.0 if start was clipped by nD slicing
- `aEndClipped`: float - 1.0 if end was clipped by nD slicing

**Per-vertex attribute** (quad corners):

- `aQuadCorner`: vec2 - Corner position: (-1,-1), (1,-1), (-1,1), (1,1)

**Uniforms**:

- `uFOV`: float - Camera field of view (radians)
- `uResolution`: vec2 - Framebuffer resolution [width, height]
- `uHDRMultiplier`: float - HDR boost factor (typ. 16.0)
- `uOpacity`: float - Global opacity multiplier

### 7.4 Line Vertex Shader

**Purpose**: Expand quad vertices to form thick line segment in screen space, compute cap factor for endpoints.

```glsl
// Instanced Thick Line Vertex Shader with Cap Factor
attribute vec3 aStartPos;
attribute vec3 aEndPos;
attribute vec3 aStartColor;
attribute vec3 aEndColor;
attribute float aStartWidth;
attribute float aEndWidth;
attribute float aStartSharpness;
attribute float aEndSharpness;
attribute float aSegmentLength;
attribute float aStartClipped;   // 1.0 if start was clipped by nD slicing
attribute float aEndClipped;     // 1.0 if end was clipped by nD slicing

// Per-vertex within quad
attribute vec2 aQuadCorner; // x: -1 (start) or +1 (end), y: -1 or +1 (perp direction)

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec2 uResolution;
uniform float uFOV;

varying vec3 vColor;
varying float vSharpness;
varying float vPerpNorm;   // Signed: -1 at bottom edge, +1 at top edge (GPU interpolates)
varying float vCapFactor;  // 0.5 at endpoints, 1.0 in body (or 1.0 if clipped)
varying float vPixelWidth; // Line width in pixels (for anti-aliasing)

void main() {
    // Determine position along segment: t=0 at start, t=1 at end
    float t = aQuadCorner.x > 0.0 ? 1.0 : 0.0;

    // Interpolate all per-vertex attributes along segment
    vec3 worldPos = mix(aStartPos, aEndPos, t);
    vColor = mix(aStartColor, aEndColor, t);
    float width = mix(aStartWidth, aEndWidth, t);
    vSharpness = mix(aStartSharpness, aEndSharpness, t);

    // Project to clip space
    vec4 clipStart = projectionMatrix * modelViewMatrix * vec4(aStartPos, 1.0);
    vec4 clipEnd = projectionMatrix * modelViewMatrix * vec4(aEndPos, 1.0);
    vec4 clipPos = projectionMatrix * modelViewMatrix * vec4(worldPos, 1.0);

    // Convert clip-space endpoints to pixel coordinates for correct aspect ratio handling
    vec2 ndcStart = clipStart.xy / clipStart.w;
    vec2 ndcEnd = clipEnd.xy / clipEnd.w;
    vec2 pixelStart = (ndcStart * 0.5 + 0.5) * uResolution;
    vec2 pixelEnd = (ndcEnd * 0.5 + 0.5) * uResolution;

    // Compute line direction and perpendicular in pixel space (aspect-ratio correct)
    vec2 pixelDir = pixelEnd - pixelStart;
    float pixelLen = length(pixelDir);
    vec2 lineDir = pixelLen > 0.0001 ? pixelDir / pixelLen : vec2(1.0, 0.0);
    vec2 perpendicular = vec2(-lineDir.y, lineDir.x);  // Unit vector in pixel space

    // Calculate pixel width from world width (perspective-correct)
    float dist = length((modelViewMatrix * vec4(worldPos, 1.0)).xyz);
    float rawPixelWidth = width * uResolution.y / (dist * tan(uFOV * 0.5));

    // Enforce minimum pixel width to prevent sub-pixel rendering artifacts
    float minPixelWidth = 1.5;
    float pixelWidth = max(rawPixelWidth, minPixelWidth);

    // Pass raw pixel width to fragment shader for intensity scaling
    vPixelWidth = rawPixelWidth;

    // Perpendicular position: -1 at bottom edge, +1 at top edge
    vPerpNorm = aQuadCorner.y;

    // Expand quad by perpendicular offset in pixel space, then convert to clip space
    vec2 pixelOffset = perpendicular * aQuadCorner.y * pixelWidth;
    vec2 ndcOffset = pixelOffset / uResolution * 2.0;
    clipPos.xy += ndcOffset * clipPos.w;

    gl_Position = clipPos;

    // --- Cap Factor Calculation with Clipping Awareness ---
    // Normal: 0.5 at true endpoints, 1.0 in body
    // Clipped endpoints: force 1.0 (the "real" endpoint is outside the slice)
    float distFromStart = t * aSegmentLength;
    float distFromEnd = (1.0 - t) * aSegmentLength;

    // Base cap factor from distance to nearest endpoint
    float distToNearest = min(distFromStart, distFromEnd);
    float baseCap = (distToNearest >= width) ? 1.0 : 0.5 + 0.5 * (distToNearest / width);

    // Override if the nearest endpoint was clipped
    float nearestIsStart = step(distFromEnd, distFromStart);  // 1 if closer to start
    float nearestClipped = mix(aEndClipped, aStartClipped, nearestIsStart);

    // If nearest endpoint was clipped, use full intensity (1.0)
    vCapFactor = mix(baseCap, 1.0, nearestClipped);
}
```

### 7.5 Line Fragment Shader

**Purpose**: Render line with parabolic intensity profile, cap factor for seamless joints, and edge anti-aliasing.

```glsl
// Line Fragment Shader with Semicircle Kernel Convolution Profile and Edge AA
uniform float uHDRMultiplier;
uniform float uOpacity;

varying vec3 vColor;
varying float vSharpness;        // Per-vertex sharpness (interpolated from vertex shader)
varying float vPerpNorm;         // Interpolated: 0 at centerline, ±1 at edges
varying float vCapFactor;        // 0.5 at endpoints, 1.0 in body
varying float vPixelWidth;       // Raw line width in pixels (before minimum clamping)

void main() {
    // Compute distance from centerline (0 to 1)
    float p = abs(vPerpNorm);

    // Discard pixels outside the line width
    if (p >= 1.0) discard;

    // Parabolic falloff from semicircle kernel convolution
    // Base profile: (1 - p²) where p = distance from centerline
    // With per-vertex sharpness: (1 - p²)^sharpness
    float p2 = p * p;
    float perpFalloff = pow(1.0 - p2, vSharpness);

    // Anti-aliasing: smooth falloff at edges
    // The AA region is ~1 pixel wide in the rendered quad
    float minPixelWidth = 1.5;
    float renderedWidth = max(vPixelWidth, minPixelWidth);
    float aaWidth = 1.0 / renderedWidth;
    float edgeAA = 1.0 - smoothstep(1.0 - aaWidth, 1.0, p);

    // Intensity scaling for sub-pixel lines
    // When a line is rendered wider than intended, reduce intensity proportionally
    float widthScale = min(vPixelWidth / minPixelWidth, 1.0);

    // Apply cap factor for correct joint intensity
    float intensity = vCapFactor * perpFalloff * edgeAA * widthScale;

    // HDR output (gamma correction in post-processing)
    vec3 finalColor = vColor * intensity * uHDRMultiplier;

    gl_FragColor = vec4(finalColor, intensity * uOpacity);
}
```

### 7.5a Anti-Aliasing for Thin Lines

**Problem**: When lines are very thin (sub-pixel or only a few pixels wide), two issues cause severe aliasing:

1. **Rasterization gaps**: Sub-pixel quads may not cover every pixel along the line
2. **Hard edge discard**: The `discard` at `p >= 1.0` creates jagged edges

**Solution**: A two-part approach:

**Part 1: Minimum Pixel Width**

Enforce a minimum rendered width of 1.5 pixels to ensure continuous rasterization:

```glsl
// In vertex shader
float minPixelWidth = 1.5;
float pixelWidth = max(rawPixelWidth, minPixelWidth);
vPixelWidth = rawPixelWidth;  // Pass raw width to fragment shader
```

**Part 2: Intensity Scaling + Edge Smoothing**

In the fragment shader, compensate for the wider rendering and smooth edges:

```glsl
float minPixelWidth = 1.5;
float renderedWidth = max(vPixelWidth, minPixelWidth);

// Edge anti-aliasing: ~1 pixel smooth region
float aaWidth = 1.0 / renderedWidth;
float edgeAA = 1.0 - smoothstep(1.0 - aaWidth, 1.0, p);

// Intensity scaling: reduce brightness for lines rendered wider than intended
float widthScale = min(vPixelWidth / minPixelWidth, 1.0);

float intensity = vCapFactor * perpFalloff * edgeAA * widthScale;
```

**Behavior**:

- **Wide lines (10+ pixels)**: Rendered at actual width, full intensity, small AA region
- **Thin lines (1.5+ pixels)**: Rendered at actual width, smooth edges
- **Sub-pixel lines (<1.5 pixels)**: Rendered at 1.5px but with reduced intensity proportional to actual width

**Visual Effect**:

- Eliminates "dashed/stippled" appearance from rasterization gaps
- Smooth edges instead of jagged discard boundaries
- Sub-pixel lines appear as expected thin/faint lines rather than broken segments

### 7.6 Cap Factor Visualization

```
Segment a-b:           Segment b-c:
  a ═══════════ b        b ═══════════ c

Cap factor along segment:

    0.5 → 1.0         1.0          1.0 → 0.5
    |-------|---------------------|-------|
    a      a+w       body        c-w      c
         (ramp)                 (ramp)

At joint b:
  - Segment a-b contributes 0.5 at endpoint b
  - Segment b-c contributes 0.5 at endpoint b
  - Total: 0.5 + 0.5 = 1.0 ✓ (correct with additive blending)
```

### 7.7 Line Width Calculation

**World-Space Width**: Line widths are specified in scene units (same as point radii).

**Pixel Half-Width Formula**:

```
pixelHalfWidth = width × resolution.y / (distance × tan(FOV / 2))
```

This is half the point size formula (since lines have half-width, points have full radius).

**Consistency**: A line with width `w` connecting two points of radius `w` will perfectly join them.

### 7.8 Sharpness Parameter

**Purpose**: Artistic control over edge falloff, independent of the mathematical model.

**Default**: `sharpness = 1.0` (pure parabolic profile from semicircle kernel)

**Effect of sharpness values**:

- `s = 1.0`: Parabolic profile `(1 - p²)`
- `s = 2.0`: Sharper falloff `(1 - p²)²`
- `s = 0.5`: Softer falloff `√(1 - p²)` (approaches semicircle itself)

**Note**: Sharpness does NOT affect the cap factor math - it only modifies the perpendicular falloff for artistic purposes.

### 7.9 Blending Configuration

Lines use the same additive blending as points:

```typescript
material.blending = THREE.AdditiveBlending;
material.transparent = true;
material.depthWrite = false;
material.depthTest = true;
```

**Critical**: The cap factor model is designed specifically for additive blending. Using different blend modes will produce incorrect joint intensities.

### 7.10 Color Interpolation

Colors are linearly interpolated along each segment:

```glsl
// In vertex shader
float t = aQuadCorner.x > 0.0 ? 1.0 : 0.0;
vec3 color = mix(aStartColor, aEndColor, t);
```

**Smooth gradients**: Color changes smoothly from start to end of each segment.

### 7.11 Geometry Setup

**Instanced Quad Geometry**:

```typescript
// Base quad geometry (2 triangles, 4 vertices)
const quadGeometry = new THREE.BufferGeometry();

// Quad corners: two triangles covering [-1,1] x [-1,1]
const corners = new Float32Array([
  -1,
  -1, // bottom-left (start, bottom)
  1,
  -1, // bottom-right (end, bottom)
  -1,
  1, // top-left (start, top)
  1,
  1, // top-right (end, top)
]);
quadGeometry.setAttribute('aQuadCorner', new THREE.BufferAttribute(corners, 2));

// Indices for two triangles
quadGeometry.setIndex([0, 1, 2, 2, 1, 3]);

// Instanced attributes (per segment)
const instancedGeometry = new THREE.InstancedBufferGeometry();
instancedGeometry.copy(quadGeometry);

// Set up instance attributes from loaded line data
instancedGeometry.setAttribute('aStartPos', new THREE.InstancedBufferAttribute(startPositions, 3));
instancedGeometry.setAttribute('aEndPos', new THREE.InstancedBufferAttribute(endPositions, 3));
// ... etc for colors, width, sharpness, segmentLength
```

### 7.12 Performance Considerations

**Segment Count Limits**:

| Segments | Performance          |
| -------- | -------------------- |
| < 100K   | Smooth (60 fps)      |
| 100K-1M  | Good (30-60 fps)     |
| 1M-10M   | Moderate (10-30 fps) |
| > 10M    | May require LOD      |

**Optimizations**:

1. **Instanced rendering**: One draw call for all visible segments
2. **Spatial chunking**: Load only visible segment chunks (see data/SPECIFICATIONS.md Section 7)
3. **Frustum culling**: Spatial index query excludes off-screen segments
4. **Instance buffer updates**: Only update GPU buffers when visible segments change

### 7.13 LineMaterialConfig

```typescript
interface LineMaterialConfig {
  blendingMode: 'additive' | 'normal' | 'max';
  opacity: number; // 0.0 to 1.0
  hdrMultiplier: number; // Typically 16.0
}
```

**Note**: Sharpness is per-vertex (aStartSharpness/aEndSharpness), not a global config parameter.

### 7.14 LineMaterialUniforms

```typescript
interface LineMaterialUniforms {
  uFOV: { value: number }; // Radians
  uResolution: { value: THREE.Vector2 }; // [width, height]
  uHDRMultiplier: { value: number }; // Typically 16.0
  uOpacity: { value: number }; // 0.0 to 1.0
}
```

**Note**: No gamma or sharpness uniforms - gamma correction handled in post-processing pipeline (same as points), and sharpness is per-vertex (aStartSharpness/aEndSharpness).

---

## 8. GSplat Material System

### 8.1 Overview

The GSplat Material implements volumetric Gaussian splatting for rendering oriented, anisotropic 3D Gaussian density functions with nD slicing support.

**Key Features:**
- Full 3D covariance representation via packed Cholesky factors
- Perspective-correct projection of 3D covariance to 2D screen space
- Generalized Gaussian falloff: `exp(-½ · r^sharpness)`
- Sum and max projection modes with proper ray integration
- Two-stage near-plane culling for performance

### 8.2 Mathematical Foundation

**GSplat Density Function:**

```
G(x) = a · exp(-½ · ‖L⁻¹(x - μ)‖^s)
```

Where:
- `a` = amplitude (intensity, attenuated by hidden nD dimensions)
- `μ` = center position (3D after nD slicing)
- `L` = Cholesky factor of covariance (Σ = L·Lᵀ)
- `s` = sharpness (2.0 = standard Gaussian)
- `‖L⁻¹(x - μ)‖` = Mahalanobis distance

**Covariance Representation:**

Covariance stored as packed Cholesky factors (lower triangular):
```
3D Cholesky: [L00, L10, L11, L20, L21, L22] → 6 elements
Packed into three vec2 attributes for GPU efficiency
```

### 8.3 Vertex Shader Algorithm

**Purpose**: Transform splat center and covariance to screen space, expand oriented quad.

**Key Steps:**

1. **Transform to Camera Space**
   ```glsl
   vec4 centerCam4 = modelViewMatrix * vec4(aCenter, 1.0);
   mat3 R = mat3(modelViewMatrix);
   mat3 L_cam = R * L3D;  // Rotate Cholesky to camera space
   mat3 Sigma_cam = L_cam * transpose(L_cam);
   ```

2. **Two-Stage Near-Plane Culling**
   ```glsl
   // Stage 1: Fixed threshold (fast path: 1 cycle)
   if (-centerCam.z < 0.1) {
       gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
       return;
   }

   // Stage 2: Adaptive threshold for large splats (slow path: ~6 cycles)
   float sigmaTraceSq = Sigma_cam[0][0] + Sigma_cam[1][1] + Sigma_cam[2][2];
   if (sigmaTraceSq > 0.01) {  // Only if sigma > 0.1
       float sigmaTrace = sqrt(sigmaTraceSq);
       if (-centerCam.z < sigmaTrace * uTruncate) {
           gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
           return;
       }
   }
   ```

3. **Perspective Jacobian Projection**
   ```glsl
   // Jacobian J = ∂(screen)/∂(camera)
   mat3x2 J;
   J[0] = vec2(uFx * invZ, 0.0);
   J[1] = vec2(0.0, uFy * invZ);
   J[2] = vec2(uFx * centerCam.x * invZ2, uFy * centerCam.y * invZ2);

   // Project: Σ_2D = J · Σ_cam · Jᵀ
   ```

4. **Eigenvalue Decomposition & Quad Expansion**
   ```glsl
   // Compute eigenvalues for extent
   float trace = Sigma2D[0][0] + Sigma2D[1][1];
   float det = Sigma2D[0][0] * Sigma2D[1][1] - Sigma2D[0][1] * Sigma2D[1][0];
   float lambda1 = 0.5 * (trace + sqrt(trace² - 4*det));
   float lambda2 = 0.5 * (trace - sqrt(trace² - 4*det));

   // Eigenvector for orientation
   vec2 majorAxis = normalize(vec2(lambda1 - Sigma2D[1][1], Sigma2D[0][1]));

   // Quad extents
   float extent1 = uTruncate * sqrt(lambda1);
   float extent2 = uTruncate * sqrt(lambda2);
   ```

5. **Amplitude Calculation (Mode-Dependent)**
   ```glsl
   // Sum projection: integrate Gaussian along ray
   if (uProjectionMode == 0) {
       vec3 rayDir = normalize(centerCam);
       float sigmaRay = sqrt(dot(rayDir, Sigma_cam * rayDir));
       float c_s = sharpnessIntegralFactor(aSharpness);
       vAmplitude2D = aAmplitude * sigmaRay * c_s;
   }
   // Max projection: peak value (no integration)
   else {
       vAmplitude2D = aAmplitude;
   }
   ```

### 8.4 Fragment Shader Algorithm

**Purpose**: Evaluate generalized Gaussian falloff using Mahalanobis distance.

**Key Steps:**

1. **Mahalanobis Distance via Forward Substitution**
   ```glsl
   vec2 d = gl_FragCoord.xy - vCenterScreen;
   float y0 = d.x * vL2D.x;  // d.x * invL00
   float y1 = (d.y - vL2D.y * y0) * vL2D.z;  // (d.y - L10*y0) * invL11
   float mahalSq = y0*y0 + y1*y1;
   ```

2. **Early Discard at 3σ**
   ```glsl
   if (mahalSq > 9.0) discard;  // exp(-4.5) ≈ 0.011 (negligible)
   ```

3. **Generalized Gaussian Falloff**
   ```glsl
   // Standard Gaussian (s=2.0)
   intensity = vAmplitude2D * exp(-0.5 * mahalSq);

   // Sum projection with s≠2 (non-separable correction)
   float gauss_2d = exp(-0.5 * mahalSq);
   float correction = correctionFactor(r_2D, vSharpness, vAspectRatio);
   intensity = vAmplitude2D * gauss_2d * correction;

   // Max projection with s≠2
   float rToTheS = pow(mahalSq, vSharpness * 0.5);
   intensity = vAmplitude2D * exp(-0.5 * rToTheS);
   ```

### 8.5 Near-Plane Culling Strategy

**Problem**: When camera is very close to splats or inside datasets:
- Splats behind camera still render (if not culled)
- Large splats project to thousands of pixels (overdraw)
- Result: White screen, performance degradation

**Solution**: Two-stage culling with performance-cost tradeoff

**Stage 1 - Fixed Threshold (1 cycle cost)**:
- Cull if `z < 0.1`
- Catches ~95% of normal-sized splats
- Single comparison, very fast

**Stage 2 - Adaptive Threshold (~6 cycles cost)**:
- Only runs if `sigma > 0.1` (trace of covariance > 0.01)
- Computes splat extent: `sigmaTrace = sqrt(Σ_cam[0][0] + Σ_cam[1][1] + Σ_cam[2][2])`
- Cull if `z < sigmaTrace * uTruncate`
- Prevents large splats from projecting beyond screen bounds

**Performance Benefit**: Early exit before expensive covariance projection and quad expansion saves thousands of fragment shader invocations.

### 8.6 Blending Configuration

```typescript
// For additive/luminous modes
this.blendEquation = THREE.AddEquation;
this.blendSrc = THREE.OneFactor;  // NOT SrcAlpha!
this.blendDst = THREE.OneFactor;

// For max mode
this.blendEquation = THREE.MaxEquation;
```

**Critical**: Using `SrcAlpha` would square intensities (incorrect). `OneFactor` gives correct linear sum: `final = src + dst`.

### 8.7 GPU Optimizations

| Optimization | Implementation | Benefit |
|--------------|----------------|---------|
| `flat` interpolation | All per-instance varyings | Skips GPU interpolator hardware |
| Reciprocal precomputation | `vL2D = [1/L00, L10, 1/L11]` | DIV→MUL in fragment shader |
| Early discard at 3σ | Before `pow()/exp()` | Avoids expensive math for edges |
| Sharpness=2.0 specialization | Skip `pow()` in both shaders | Common case fast path |
| Two-stage culling | Fixed + adaptive thresholds | Minimal cost for common case |
| `mediump` for colors | Fragment shader precision | Reduces register pressure |

### 8.8 Data Structures

**GSplatMaterialConfig**:
```typescript
interface GSplatMaterialConfig {
  opacity?: number;               // 0.0 to 1.0
  hdrMultiplier?: number;         // Typically 16.0
  truncationRadius?: number;      // In sigmas (default 3.0)
  blendingMode?: 'additive' | 'normal' | 'max' | 'opaque' | 'luminous';
  transparent?: boolean;
  depthTest?: boolean;
}
```

**GSplatMaterialUniforms**:
```typescript
interface GSplatMaterialUniforms {
  uResolution: { value: THREE.Vector2 };     // [width, height]
  uFx: { value: number };                    // Focal length X (pixels)
  uFy: { value: number };                    // Focal length Y (pixels)
  uTruncate: { value: number };              // Truncation radius (sigmas)
  uHDRMultiplier: { value: number };         // HDR intensity boost
  uOpacity: { value: number };               // Global opacity
  uProjectionMode: { value: number };        // 0=sum, 1=max
}
```

**Per-Instance Attributes**:
```typescript
- aCenter: vec3           // 3D center (after nD slicing)
- aCholesky01: vec2       // [L00, L10]
- aCholesky23: vec2       // [L11, L20]
- aCholesky45: vec2       // [L21, L22]
- aAmplitude: float       // Attenuated by hidden dims
- aSharpness: float       // Generalized Gaussian exponent
- aColor: vec3            // RGB color
```

### 8.9 Instanced Geometry Setup

GSplats use `THREE.Mesh` with `InstancedBufferGeometry` (same pattern as lines):

```typescript
const geometry = new THREE.InstancedBufferGeometry();
geometry.setAttribute('aQuadCorner', baseGeometry.getAttribute('aQuadCorner'));
geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(centers, 3));
geometry.setAttribute('aCholesky01', new THREE.InstancedBufferAttribute(cholesky01, 2));
geometry.setAttribute('aCholesky23', new THREE.InstancedBufferAttribute(cholesky23, 2));
geometry.setAttribute('aCholesky45', new THREE.InstancedBufferAttribute(cholesky45, 2));
geometry.setAttribute('aAmplitude', new THREE.InstancedBufferAttribute(amplitudes, 1));
geometry.setAttribute('aSharpness', new THREE.InstancedBufferAttribute(sharpness, 1));
geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(colors, 3));
geometry.instanceCount = splatCount;

const mesh = new THREE.Mesh(geometry, gsplatMaterial);
mesh.frustumCulled = true;
```

**Note**: Uses `THREE.Mesh` (not `THREE.InstancedMesh`) to avoid the 16 attribute location limit.

### 8.10 Performance Considerations

**Splat Count Limits**:

| Splats | Performance          |
|--------|----------------------|
| < 100K | Smooth (60 fps)      |
| 100K-1M | Good (30-60 fps)    |
| 1M-10M | Moderate (10-30 fps) |
| > 10M  | May require LOD      |

**Optimizations**:
1. **Two-stage culling**: Prevents rendering splats too close to camera
2. **Spatial chunking**: Load only visible splat chunks
3. **Frustum culling**: Bounding box-based visibility
4. **Instance buffer updates**: Only update GPU buffers when visible splats change

---

## 9. Adaptive Resolution System

### 9.1 Purpose

The AdaptiveDPRManager dynamically adjusts the device pixel ratio (DPR) based on real-time FPS to maintain smooth rendering performance. When frame rates drop below threshold, resolution is reduced; when performance improves, resolution is restored.

### 9.2 Algorithm

**Core Loop** (evaluated every 500ms):

```
recordFrame(timestamp)
  ↓
Add timestamp to sliding window (1 second)
  ↓
Calculate FPS from frame count / time span
  ↓
[FPS < minFPS?] → Scale down immediately
  ↓
[FPS > maxFPS for hysteresis period?] → Scale up
  ↓
Notify callback (for UI indicators)
```

**Hysteresis**: Scale up requires sustained high FPS (configurable, default 2 seconds) to prevent rapid toggling.

### 9.3 Public API

```typescript
class AdaptiveDPRManager {
  constructor(customConfig?: Partial<AdaptiveDPRConfig>);

  // Renderer integration
  setRenderer(renderer: DPRRenderer): void;
  recordFrame(timestamp: number): void;

  // Enable/disable
  setEnabled(enabled: boolean): void;
  isActive(): boolean;

  // Manual control (only when adaptive is disabled)
  setManualDPR(dpr: number): void;

  // State queries
  getCurrentDPR(): number;
  getNativeDPR(): number;
  getCurrentFPS(): number;
  getIsReducedResolution(): boolean;
  getState(): AdaptiveDPRState;

  // Callbacks
  setOnDPRChangeCallback(callback: DPRChangeCallback | null): void;

  dispose(): void;
}

interface AdaptiveDPRState {
  enabled: boolean;
  currentDPR: number;
  currentFPS: number;
  isReducedResolution: boolean;
  nativeDPR: number;
}

type DPRChangeCallback = (dpr: number, isReducedResolution: boolean) => void;
```

### 9.4 Configuration

```typescript
interface AdaptiveDPRConfig {
  enabled: boolean; // Default: true
  minFPS: number; // Trigger scale down (default: 25)
  maxFPS: number; // Trigger scale up (default: 55)
  minDPR: number; // Floor value (default: 0.5)
  scaleDownFactor: number; // DPR reduction multiplier (default: 0.8)
  scaleUpFactor: number; // DPR increase multiplier (default: 1.1)
  evaluationIntervalMs: number; // Check interval (default: 500)
  hysteresisSeconds: number; // Sustain before scale up (default: 2)
}
```

### 9.5 Manual DPR Control

When adaptive mode is disabled, users can manually set the DPR:

```typescript
manager.setEnabled(false); // Disable adaptive mode
manager.setManualDPR(0.75); // Set 75% of native resolution

// DPR is clamped to [0.25, nativeDPR]
// Logs warning if called while adaptive mode is enabled
```

**Use Case**: Testing performance at specific resolutions, or deliberately reducing quality for presentations.

### 9.6 Integration with Scene Manager

```typescript
// In SceneManager
setAdaptivePixelRatio(dpr: number): void {
  this.renderer.setPixelRatio(dpr);
  const w = this.canvas.clientWidth;
  const h = this.canvas.clientHeight;
  this.renderer.setSize(w, h, false);

  // Update post-processing resolution
  if (this.postProcessing) {
    this.postProcessing.resize(w, h);
    // Scale noise parameters for perceptual consistency
    const normalizedDPR = dpr / window.devicePixelRatio;
    this.postProcessing.setDPRScale(normalizedDPR);
  }
}
```

### 9.7 Reduced Resolution Mode Detection

Resolution is considered "reduced" when DPR is below 95% of native:

```typescript
this.isReducedResolution = newDPR < this.nativeDPR * 0.95;
```

This status is passed to UI components (ResolutionIndicator) via the callback.

---

## Changelog

- **v1.3.8** (2025-12-28): Terminology fix - "Low Power Mode" → "Reduced Resolution"
  - **RENAMED**: `isLowPowerMode` → `isReducedResolution` throughout codebase
  - **RENAMED**: `getIsLowPowerMode()` → `getIsReducedResolution()`
  - **RENAMED**: Section 8.7 "Low Power Mode Detection" → "Reduced Resolution Mode Detection"
  - **RENAMED**: Callback parameter from `isLowPowerMode` to `isReducedResolution`
  - **RATIONALE**: "Low Power Mode" was misleading (suggests battery saving, not resolution scaling)

- **v1.3.7** (2025-12-17): Adaptive resolution system and DPR-based noise scaling
  - **ADDED**: Section 8 "Adaptive Resolution System" documenting AdaptiveDPRManager
    - Dynamic DPR adjustment based on real-time FPS
    - Hysteresis algorithm to prevent rapid toggling
    - Manual DPR control when adaptive mode is disabled
    - Reduced resolution mode detection (DPR < 95% native)
    - Integration with SceneManager and PostProcessingManager
  - **ADDED**: DPR-based noise scaling for perceptual consistency at reduced resolutions
    - Noise parameters (readoutSigma, photonGain, fpnSigma) scale linearly with DPR
    - Maintains visual appearance when adaptive resolution reduces rendering quality
    - New `setDPRScale(dpr)` method on PostProcessingManager
  - **ADDED**: `getNativeDPR()` and `setManualDPR(dpr)` methods on AdaptiveDPRManager
  - **CHANGED**: Updated detector noise default values
    - readoutSigma: 0.01 → 0.002
    - photonGain: 0.01 → 0.002
    - fpnSigma: 0.005 → 0.001
  - **ADDED**: `baseNoiseSettings` and `currentDPRScale` state variables for tracking
  - **ADDED**: `applyScaledNoiseSettings()` method for applying DPR-adjusted values
  - **DOCUMENTED**: Mathematical basis for DPR-proportional noise scaling

- **v1.3.6** (2025-12-11): Line rendering anti-aliasing and aspect ratio fix
  - **BUGFIX**: Fixed aspect ratio issue in line perpendicular calculation
    - Now computes line direction in pixel space instead of NDC space
    - Ensures correct line width regardless of screen aspect ratio
  - **ADDED**: Minimum pixel width (1.5px) to prevent sub-pixel rasterization gaps
  - **ADDED**: Intensity scaling for sub-pixel lines (preserves visual weight)
  - **ADDED**: Section 7.5a "Edge Anti-Aliasing for Thin Lines"
  - **ADDED**: `vPixelWidth` varying passed from vertex to fragment shader
  - **IMPROVED**: Smooth edge falloff using `smoothstep` for thin lines
  - **FIXED**: Eliminated "dashed/stippled" aliasing artifacts on thin detector geometry lines

- **v1.3.5** (2025-12-09): Vignette artifacts fix and detector noise improvements
  - **BUGFIX**: Fixed vignette artifacts with additive blending (alpha overflow)
    - Added `RobustVignetteEffect` to replace pmndrs `VignetteEffect`
    - Root cause: Additive blending accumulates alpha, which can overflow to Infinity in Float16
    - Fix: Force alpha to 1.0 in vignette output (screen-space effects should always be opaque)
    - See: `robust-vignette-effect.ts`
  - **BUGFIX**: Fixed detector noise time overflow after long sessions
    - Use `mod(time, 1000.0)` to prevent precision loss
  - **BUGFIX**: Fixed detector noise "burning" in dark areas
    - Anscombe transform bias (3/8) was brightening near-black pixels
    - Added smoothstep blend to reduce shot noise influence for dark pixels
  - **IMPROVEMENT**: Added `safeDisposeEffect()` helper for proper effect cleanup
  - **IMPROVEMENT**: Added state preservation in `recreateComposer()` for vignette and detector noise

- **v1.3.4** (2025-12-09): Per-vertex attribute cleanup
  - **FIXED**: Removed unused `sharpness` from `LineMaterialConfig` (sharpness is per-vertex)
  - **FIXED**: Removed unused `uSharpness` from `LineMaterialUniforms` (sharpness is per-vertex)
  - Added clarifying notes about per-vertex sharpness via aStartSharpness/aEndSharpness
  - Synced with data spec v1.2.2

- **v1.3.3** (2025-12-09): nD slicing with endpoint clipping
  - **ADDED**: aStartClipped, aEndClipped attributes for clipping awareness
  - Cap factor now respects clipped endpoints (force 1.0 for clipped ends)
  - Clipped endpoints don't contribute to joint intensity reduction
  - Updated vertex shader with clipping-aware cap factor calculation

- **v1.3.2** (2025-12-09): Critical shader bug fixes and per-vertex attributes
  - **BUGFIX**: Fixed vPerpNorm always being 1.0 (was using abs() on -1/+1 values)
    - Now passes signed value to fragment shader, computes abs() there
    - GPU interpolation gives 0 at centerline, ±1 at edges
  - **CHANGE**: Per-vertex sharpness instead of global uniform
    - Added aStartSharpness, aEndSharpness instanced attributes
    - Sharpness interpolates along segment like color and width
  - **CHANGE**: Variable width per segment (aStartWidth, aEndWidth)
  - Removed uSharpness uniform (now per-vertex varying)
  - Updated all attribute names for consistency

- **v1.3.1** (2025-12-09): Semicircle kernel convolution model for lines
  - **MAJOR**: Added mathematical foundation for seamless joint rendering
  - **MODEL**: Semicircle kernel f(r) = √(1-(r/R)²) produces parabolic body profile
  - **JOINTS**: Cap factor (0.5 at endpoints → 1.0 in body) enables correct additive blending
  - **CRITICAL**: Two adjacent segments sum to full intensity at shared endpoint
  - Derived closed-form convolution result for efficient GPU computation
  - Added cap factor calculation in vertex shader
  - Updated fragment shader with parabolic falloff: `(1 - p²)^sharpness`
  - Added cap factor visualization diagram
  - Documented sharpness as artistic control independent of mathematical model

- **v1.3.0** (2025-12-09): Line Material System
  - **ADDED**: Section 7 - Line Material System
  - Documented rendering approach decision (instanced quads with THREE.Mesh + InstancedBufferGeometry)
  - Added instanced quad geometry approach for thick lines
  - Added line vertex and fragment shaders
  - Added world-space width calculation (same formula as points)
  - Documented color interpolation along segments
  - Added performance considerations and optimization strategies
  - Added `LineMaterialConfig` and `LineMaterialUniforms` data structures

- **v1.2.0** (2025-12-08): DetectorNoiseEffect simplified and enhanced
  - Added Fixed Pattern Noise (FPN) - static per-pixel offset from detector non-uniformities
  - Removed `intensity` parameter (not physics-based, was just a blend factor)
  - Removed `animated` parameter (real detector noise always has temporal components)
  - Retired old `NoiseEffect` (pmndrs film grain) in favor of physics-based model
  - Three-component physics model: Shot (Poisson) + Readout (Gaussian temporal) + FPN (Gaussian static)
  - New API: `setDetectorNoiseEnabled(enabled, readoutSigma?, photonGain?, fpnSigma?)`
  - New config properties: `detectorNoiseReadoutSigma`, `detectorNoisePhotonGain`, `detectorNoiseFpnSigma`
  - See: `detector-noise-effect.ts`, `post-processing-manager.ts`

- **v1.1.0** (2025-12-08): Physics-based detector noise effect
  - Added `DetectorNoiseEffect` class for realistic camera/detector noise simulation
  - Implements combined Poisson (shot noise) + Gaussian (readout noise) model
  - Uses Bob Jenkins hash for fast deterministic PRNG
  - Uses clamped logistic distribution for efficient Gaussian approximation
  - Uses Anscombe transform for Poisson approximation
  - Integrated into PostProcessingManager via `setDetectorNoiseEnabled()`
  - See: `detector-noise-effect.ts`, `post-processing-manager.ts`

- **v1.0.1** (2025-12-08): Material lifecycle improvements
  - Added `MaterialManager.unregister()` method to remove materials from global update lists
  - Added `PointMaterial.dispose()` override to automatically unregister on disposal
  - Fixed memory leak where disposed materials accumulated in MaterialManager registry
  - Long-running apps no longer accumulate dead materials
  - See: `material-manager.ts:132-149`, `point-material.ts:241-247`

- **v1.0.0** (2025-01-30): Initial specification
  - HDR rendering pipeline with 16-bit float buffers
  - Custom point materials with world-space sizing
  - Angular size calculation for physically accurate scaling
  - Sharpness compensation mathematical model
  - Post-processing effects via pmndrs/postprocessing
  - Material caching system
  - Anti-aliasing recommendations (FXAA, SMAA preferred; MSAA incompatible)
  - Dynamic pass assignment for effect composition
