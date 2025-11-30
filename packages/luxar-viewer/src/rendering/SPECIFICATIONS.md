# luxar-viewer.rendering - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2025-01-30

## Purpose

The `luxar-viewer.rendering` package provides advanced WebGL rendering capabilities including HDR post-processing pipeline, custom point materials with world-space sizing, and material management for point cloud visualization.

**Core Responsibility**: Deliver professional-grade visual effects through pmndrs/postprocessing library integration, custom shaders for physically accurate point rendering, and efficient material caching.

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

---

## 1. HDR Rendering Pipeline

### 1.1 Render Target Configuration

**HDR Support**: Use `THREE.HalfFloatType` (16-bit float) render targets to support HDR color values > 1.0.

**Setup**:

```typescript
const composer = new EffectComposer(renderer, {
    frameBufferType: THREE.HalfFloatType,  // 16-bit float for HDR
    multisampling: 0  // MSAA incompatible with additive blending
})
```

**Color Space Pipeline**:

```
Scene Rendering → HDR Buffer (LinearSRGB) → Effects → Tone Mapping → Output (SRGB)
```

### 1.2 Tone Mapping

**Purpose**: Map HDR colors (unbounded) to LDR display range [0, 1].

**Tone Mapping Operators**:

| Operator | Description | Characteristics |
|----------|-------------|-----------------|
| **ACES Filmic** | Industry standard (default) | Smooth highlights, natural look |
| **AgX** | Modern alternative | Balanced, film-like |
| **Reinhard** | Classic operator | Simple, local adaptation |
| **Linear** | No mapping | Raw HDR (clips >1) |
| **Neutral** | Balanced | Minimal color shift |

**Implementation** (via pmndrs/postprocessing):

```typescript
// Tone mapping applied as final pass
const toneMappingEffect = new ToneMappingEffect({
    mode: ToneMappingMode.ACES_FILMIC,
    resolution: 256,
    adaptive: false
})
```

### 1.3 Effect Composition Strategy

**Dynamic Pass Assignment Algorithm**:

```typescript
function buildEffectPasses(enabledEffects: Effect[]): Pass[] {
    const passes: Pass[] = []

    // Effects in visual order
    const orderedEffects = [
        bloom, dof, ao, vignette,
        chromaticAberration, lensDistortion,
        noise, toneMapping, aa
    ].filter(e => e && e.enabled)

    // Detect incompatibilities
    let switchToPassB = false
    const passA: Effect[] = []
    const passB: Effect[] = []

    for (const effect of orderedEffects) {
        // UV transformation effects (lens distortion) incompatible with
        // convolution effects (chromatic aberration) in same pass
        if (isUVTransform(effect) && passA.some(isConvolution)) {
            switchToPassB = true
        }
        if (isConvolution(effect) && passA.some(isUVTransform)) {
            switchToPassB = true
        }

        if (switchToPassB) {
            passB.push(effect)
        } else {
            passA.push(effect)
        }
    }

    // Create passes
    if (passA.length > 0) {
        passes.push(new EffectPass(camera, ...passA))
    }
    if (passB.length > 0) {
        passes.push(new EffectPass(camera, ...passB))
    }

    return passes
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
material.blending = THREE.AdditiveBlending  // For overlapping points
material.transparent = true
material.depthWrite = false  // Allow proper blending
material.depthTest = true    // Respect depth buffer
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
const actualHeight = canvas.height  // Framebuffer pixels
const actualWidth = canvas.width

material.uniforms.uResolution.value.set(actualWidth, actualHeight)

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
    intensity: 0.5,           // Bloom strength
    luminanceThreshold: 0.01, // Brightness threshold
    luminanceSmoothing: 0.9,  // Threshold smoothing
    mipmapBlur: true,         // Use mipmap blur (better quality)
    levels: 8                 // Mipmap levels (1-12)
})

// Separate blur pass for radius control
bloomEffect.mipmapBlurPass.radius = 0.6
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
    samples: 16,              // Sample count (higher = better quality)
    radius: 0.1,              // Occlusion radius
    intensity: 1.0,           // Effect strength
    luminanceInfluence: 0.7   // How much lighting affects AO
})
```

**Quality Levels**:
- Low: samples=8, radius=0.05
- Medium: samples=16, radius=0.1
- High: samples=32, radius=0.15

### 4.3 Depth of Field (DOF)

**Purpose**: Simulate camera focus with bokeh blur.

```typescript
const dofEffect = new DepthOfFieldEffect(camera, {
    focusDistance: 10.0,      // Focus plane distance
    focalLength: 0.05,        // Lens focal length
    bokehScale: 2.0           // Bokeh blur size
})
```

### 4.4 Noise Effect

**Purpose**: Add film grain or TV static aesthetic.

```typescript
const noiseEffect = new NoiseEffect({
    premultiply: false,       // Film grain mode vs TV static
    blendFunction: BlendFunction.SCREEN  // Additive blending
})

noiseEffect.blendMode.opacity.value = 0.05  // Subtle grain
```

---

## 5. Material Management

### 5.1 Material Caching

**Purpose**: Reuse materials with identical properties to reduce memory and shader compilations.

**Cache Key**:

```typescript
function getMaterialCacheKey(config: MaterialConfig): string {
    return `${config.blendingMode}_${config.opacity}_${config.gamma}`
}
```

**Cache Algorithm**:

```typescript
class MaterialManager {
    private cache = new Map<string, THREE.ShaderMaterial>()

    getPointMaterial(config: MaterialConfig): THREE.ShaderMaterial {
        const key = getMaterialCacheKey(config)

        if (this.cache.has(key)) {
            return this.cache.get(key)!
        }

        const material = createPointMaterial(config)
        this.cache.set(key, material)
        return material
    }

    updateGlobalParams(fov: number, resolution: [number, number]): void {
        // Update all cached materials
        for (const material of this.cache.values()) {
            material.uniforms.uFOV.value = fov
            material.uniforms.uResolution.value.set(resolution[0], resolution[1])
        }
    }

    dispose(): void {
        for (const material of this.cache.values()) {
            material.dispose()
        }
        this.cache.clear()
    }
}
```

### 5.2 Dynamic Parameter Updates

**Camera Changes**:

```typescript
// When camera FOV changes
materialManager.updateGlobalParams(
    camera.fov * Math.PI / 180,  // Convert to radians
    [canvas.width, canvas.height]
)
```

**Resolution Changes**:

```typescript
// On window resize
materialManager.updateGlobalParams(
    camera.fov * Math.PI / 180,
    [canvas.width, canvas.height]  // New resolution
)
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
const fxaaEffect = new FXAAEffect()
effectPass.addEffect(fxaaEffect)
```

### 6.2 SMAA (Subpixel Morphological)

**Characteristics**:
- Superior edge detection
- Multiple quality presets (LOW, MEDIUM, HIGH, ULTRA)
- Better quality than FXAA
- Moderate performance impact (1-2ms)
- Compatible with additive blending

**Setup**:

```typescript
const smaaEffect = new SMAAEffect({
    preset: SMAAPreset.HIGH
})
effectPass.addEffect(smaaEffect)
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
renderer.setSize(width * 2, height * 2, false)
composer.setSize(width * 2, height * 2)

// Display at normal resolution (downsampling provides AA)
canvas.style.width = `${width}px`
canvas.style.height = `${height}px`
```

---

## Data Structures

### MaterialConfig

```typescript
interface MaterialConfig {
    blendingMode: 'additive' | 'normal'
    opacity: number      // 0.0 to 1.0
    gamma: number        // Typically 1.0 (no correction)
}
```

### PointMaterialUniforms

```typescript
interface PointMaterialUniforms {
    uFOV: { value: number }              // Radians
    uResolution: { value: THREE.Vector2 } // [width, height]
    uHDRMultiplier: { value: number }    // Typically 16.0
    uOpacity: { value: number }          // 0.0 to 1.0
    uGamma: { value: number }            // Gamma correction
}
```

### PostProcessingConfig

```typescript
interface PostProcessingConfig {
    bloom: {
        enabled: boolean
        intensity: number
        threshold: number
        radius: number
        levels: number
    }
    toneMapping: {
        mode: 'ACES' | 'AgX' | 'Reinhard' | 'Linear' | 'Neutral'
    }
    aa: {
        method: 'none' | 'FXAA' | 'SMAA' | 'SSAA'
        smaaQuality?: 'LOW' | 'MEDIUM' | 'HIGH' | 'ULTRA'
        ssaaMultiplier?: 1.5 | 2 | 3 | 4
    }
    dof: {
        enabled: boolean
        focusDistance: number
        bokehScale: number
    }
    // ... other effects
}
```

---

## Changelog

- **v1.0.0** (2025-01-30): Initial specification
  - HDR rendering pipeline with 16-bit float buffers
  - Custom point materials with world-space sizing
  - Angular size calculation for physically accurate scaling
  - Sharpness compensation mathematical model
  - Post-processing effects via pmndrs/postprocessing
  - Material caching system
  - Anti-aliasing recommendations (FXAA, SMAA preferred; MSAA incompatible)
  - Dynamic pass assignment for effect composition
