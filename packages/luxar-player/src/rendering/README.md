# Luxar Rendering Package

> Advanced WebGL rendering pipeline for high-quality point cloud visualization

## Overview

The Luxar Rendering package provides a sophisticated rendering pipeline optimized for large-scale point cloud visualization with support for HDR rendering, post-processing effects, and dynamic material management. It leverages THREE.js and custom WebGL shaders to achieve high-performance, visually stunning results.

### Key Features

- **HDR Rendering Pipeline**: Full HDR color support with bloom effects
- **Custom Shader System**: Optimized shaders for point cloud rendering
- **Post-Processing Effects**: Bloom, tone mapping, FXAA, SMAA, DOF
- **Material Management**: Dynamic material creation and caching
- **World-Space Point Sizing**: Physically accurate point scaling
- **Additive Blending**: Beautiful overlapping point effects

### Package Architecture

```
rendering/
├── post-processing.ts    # Post-processing pipeline manager
├── shader-manager.ts     # Custom WebGL shader programs
├── material-manager.ts   # Material creation and caching
└── README.md            # This documentation
```

---

## Components

### 1. Post-Processing Manager

The `PostProcessingManager` orchestrates all post-processing effects in the rendering pipeline.

**Features:**

- HDR render targets with 16-bit float precision
- Configurable bloom with threshold, strength, and radius
- Multiple tone mapping operators (ACES, Reinhard, Linear, etc.)
- Anti-aliasing (FXAA, SMAA, MSAA, SSAA)
- Depth of field effects
- Chromatic aberration

**Usage:**

```typescript
const postProcessing = new PostProcessingManager(
  renderer,
  scene,
  camera,
  { width, height }
);

// Configure bloom
postProcessing.updateBloomSettings(
  strength: 0.25,
  radius: 1.0,
  threshold: 0.01
);

// Set tone mapping
postProcessing.setToneMapping(THREE.ACESFilmicToneMapping);

// Enable anti-aliasing
postProcessing.setFXAAEnabled(true);
```

### 2. Shader Manager

The `ShaderManager` provides optimized WebGL shaders for point cloud rendering.

**Custom Shaders:**

#### Vertex Shader Features:

- World-space point sizing based on camera distance
- Angular size calculation for perspective consistency
- FOV-independent sizing
- Depth-based scaling

#### Fragment Shader Features:

- Gaussian falloff for smooth point edges
- HDR color multiplication
- Alpha blending with configurable intensity
- Depth testing and writing

**Configuration:**

```typescript
export const SHADER_CONFIG = {
  points: {
    hdrMultiplier: 16.0, // HDR intensity boost
    baseAlpha: 0.01, // Base transparency
    falloffSteepness: 20.0, // Gaussian edge softness
    depthTest: true, // Enable depth testing
    depthWrite: false, // Disable for additive blending
  },
};
```

### 3. Material Manager

The `MaterialManager` handles creation and caching of THREE.js materials.

**Features:**

- Material caching by configuration key
- Dynamic uniform updates
- Memory-efficient material reuse
- Automatic disposal management

**Material Types:**

```typescript
// Point cloud material with custom shaders
const material = materialManager.getPointMaterial({
  vertexShader: customVertexShader,
  fragmentShader: customFragmentShader,
  uniforms: {
    hdrMultiplier: { value: 16.0 },
    baseAlpha: { value: 0.01 },
    falloffSteepness: { value: 20.0 },
  },
  transparent: true,
  blending: THREE.AdditiveBlending,
  depthTest: true,
  depthWrite: false,
});
```

---

## Rendering Pipeline

### Pipeline Stages

```
Scene Geometry
    ↓
Custom Vertex Shader (world-space sizing)
    ↓
Rasterization
    ↓
Custom Fragment Shader (HDR colors)
    ↓
HDR Render Target (16-bit float)
    ↓
Bloom Pass (extract bright pixels)
    ↓
Tone Mapping (HDR to LDR)
    ↓
Anti-Aliasing (FXAA/SMAA)
    ↓
Final Output
```

### HDR Color Pipeline

1. **Input**: Colors as Float32Array (0.0 to unlimited)
2. **Shader**: Multiply by hdrMultiplier (typically 16.0)
3. **Render Target**: Store in HalfFloatType buffer
4. **Bloom**: Extract pixels above threshold
5. **Tone Map**: Convert to displayable range
6. **Output**: Final 8-bit sRGB for display

---

## Configuration

### Render Settings

```typescript
const renderSettings = {
  // HDR Configuration
  hdrEnabled: true,
  hdrMultiplier: 16.0,

  // Bloom Settings
  bloomThreshold: 0.01, // Minimum brightness for bloom
  bloomStrength: 0.25, // Bloom intensity
  bloomRadius: 1.0, // Blur radius

  // Anti-Aliasing
  fxaaEnabled: false, // Fast approximate AA
  smaaEnabled: false, // Enhanced subpixel AA
  msaaEnabled: false, // Multi-sample AA
  ssaaEnabled: false, // Super-sample AA

  // Tone Mapping
  toneMapping: 'ACES', // Options: None, Linear, Reinhard, ACES, etc.
  exposure: 1.0, // Exposure adjustment

  // Advanced Effects
  dofEnabled: false, // Depth of field
  dofFocus: 10.0, // Focus distance
  dofStrength: 0.5, // Blur strength
};
```

### Performance Tuning

```typescript
// Quality presets
const qualityPresets = {
  low: {
    bloomResolutionScale: 8,
    fxaaEnabled: true,
    msaaSamples: 0,
  },
  medium: {
    bloomResolutionScale: 4,
    smaaEnabled: true,
    msaaSamples: 2,
  },
  high: {
    bloomResolutionScale: 2,
    smaaEnabled: true,
    msaaSamples: 4,
    ssaaMultiplier: 1.5,
  },
};
```

---

## World-Space Point Sizing

The rendering system implements physically accurate point sizing:

### Mathematical Model

```glsl
// Calculate angular size of point
float distance = length(mvPosition.xyz);
float angularSize = 2.0 * atan(radius / distance);

// Convert to pixels
float fov = radians(60.0);  // Camera FOV
gl_PointSize = angularSize * resolution.y / fov;
```

### Key Properties

- Points maintain consistent physical size
- Two points of radius r at distance 2r will just touch
- Size is independent of FOV changes
- Proper perspective scaling with distance

---

## Anti-Aliasing Techniques

### FXAA (Fast Approximate Anti-Aliasing)

- **Performance**: Very fast
- **Quality**: Good for most cases
- **Compatibility**: Works everywhere
- **Best for**: Real-time interaction

### SMAA (Enhanced Subpixel Morphological AA)

- **Performance**: Moderate
- **Quality**: Excellent edge detection
- **Compatibility**: Good
- **Best for**: High-quality captures

### MSAA (Multi-Sample Anti-Aliasing)

- **Performance**: GPU-intensive
- **Quality**: Hardware-accelerated
- **Compatibility**: Limited with additive blending
- **Best for**: Opaque geometry

### SSAA (Super-Sample Anti-Aliasing)

- **Performance**: Very expensive
- **Quality**: Best possible
- **Compatibility**: Universal
- **Best for**: Final renders

---

## Usage Examples

### Basic Setup

```typescript
import { PostProcessingManager } from './rendering/post-processing';
import { MaterialManager } from './rendering/material-manager';
import { ShaderManager } from './rendering/shader-manager';

// Initialize rendering pipeline
const postProcessing = new PostProcessingManager(renderer, scene, camera, size);

const materialManager = new MaterialManager();
const shaderManager = new ShaderManager();
```

### Creating Point Cloud Material

```typescript
// Get optimized point material
const material = materialManager.getPointMaterial({
  vertexShader: shaderManager.getVertexShader(),
  fragmentShader: shaderManager.getFragmentShader(),
  uniforms: shaderManager.getUniforms(),
  transparent: true,
  blending: THREE.AdditiveBlending,
});

// Apply to point cloud
const points = new THREE.Points(geometry, material);
scene.add(points);
```

### Configuring Post-Processing

```typescript
// Enable HDR bloom
postProcessing.updateBloomSettings(0.3, 1.2, 0.0);

// Set tone mapping
postProcessing.setToneMapping(THREE.ACESFilmicToneMapping);
postProcessing.setExposure(1.2);

// Enable anti-aliasing
postProcessing.setFXAAEnabled(true);

// Add depth of field
postProcessing.setDOF(true, 15.0, 0.5);
```

### Render Loop

```typescript
function animate() {
  requestAnimationFrame(animate);

  // Update any animations
  controls.update();

  // Render with post-processing
  postProcessing.render();
}
```

---

## Performance Considerations

### Optimization Strategies

1. **Resolution Scaling**: Reduce bloom resolution for better performance
2. **Conditional Effects**: Enable effects only when needed
3. **Material Reuse**: Cache and reuse materials
4. **Depth Sorting**: Disable for additive blending
5. **LOD System**: Reduce point count at distance

### Benchmarks

Typical performance with 1M points:

- Base rendering: ~5ms
- Bloom pass: ~2-3ms
- FXAA: ~0.5ms
- SMAA: ~1-2ms
- Total frame time: ~8-10ms (100+ FPS)

### Memory Usage

- HDR render target: ~32MB at 1080p
- Bloom buffers: ~8MB total
- Material cache: ~1MB
- Shader programs: ~100KB

---

## Troubleshooting

### Common Issues

**Problem: Points appear blocky**

- Solution: Increase `falloffSteepness` in shader config

**Problem: Bloom too intense**

- Solution: Reduce `bloomStrength` or increase `bloomThreshold`

**Problem: Colors look washed out**

- Solution: Adjust tone mapping and exposure settings

**Problem: Performance drops with MSAA**

- Solution: Use FXAA or SMAA instead with additive blending

**Problem: Points disappear at distance**

- Solution: Check far clipping plane and point size calculations

---

## API Reference

### PostProcessingManager

| Method                                             | Description                |
| -------------------------------------------------- | -------------------------- |
| `render()`                                         | Execute rendering pipeline |
| `updateBloomSettings(strength, radius, threshold)` | Configure bloom            |
| `setToneMapping(type)`                             | Set tone mapping operator  |
| `setExposure(value)`                               | Adjust exposure            |
| `setFXAAEnabled(enabled)`                          | Toggle FXAA                |
| `setSMAAEnabled(enabled)`                          | Toggle SMAA                |
| `setMSAAEnabled(enabled)`                          | Toggle MSAA                |
| `setSSAAEnabled(enabled)`                          | Toggle SSAA                |
| `setDOF(enabled, focus, strength)`                 | Configure depth of field   |
| `resize(width, height)`                            | Update render size         |
| `dispose()`                                        | Clean up resources         |

### MaterialManager

| Method                          | Description                |
| ------------------------------- | -------------------------- |
| `getPointMaterial(config)`      | Get/create point material  |
| `updateMaterial(key, uniforms)` | Update material uniforms   |
| `disposeMaterial(key)`          | Remove material from cache |
| `clear()`                       | Clear all cached materials |

### ShaderManager

| Method                 | Description                 |
| ---------------------- | --------------------------- |
| `getVertexShader()`    | Get point vertex shader     |
| `getFragmentShader()`  | Get point fragment shader   |
| `getUniforms()`        | Get shader uniforms         |
| `updateConfig(config)` | Update shader configuration |

---

## Contributing

When extending the rendering system:

1. **Maintain HDR pipeline** - Preserve float precision
2. **Test across GPUs** - Ensure compatibility
3. **Profile performance** - Monitor frame times
4. **Document shaders** - Explain mathematical models
5. **Cache aggressively** - Reuse materials and buffers

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

_For implementation details, see the source files in this directory._
