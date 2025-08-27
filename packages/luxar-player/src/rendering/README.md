# Luxar Rendering Package

> Advanced WebGL rendering pipeline using pmndrs/postprocessing for high-quality point cloud visualization

## Overview

The Luxar Rendering package provides a modern, high-performance rendering pipeline powered by the pmndrs/postprocessing library. It delivers professional-grade visual effects with optimized performance for large-scale point cloud visualization.

### Key Features

- **HDR Rendering Pipeline**: 16-bit float buffers for true HDR support
- **Modern Post-Processing**: Powered by pmndrs/postprocessing
- **Professional Effects**: Bloom, SSAO, DOF, tone mapping, and more
- **Custom Shader System**: Optimized shaders for point clouds
- **Material Management**: Efficient caching and reuse
- **World-Space Point Sizing**: Physically accurate scaling
- **Multiple Anti-Aliasing Options**: FXAA and SMAA support

### Package Architecture

```
rendering/
├── postprocessing-manager.ts  # Post-processing pipeline using pmndrs
├── point-material.ts          # Custom point cloud shaders
├── material-manager.ts        # Material creation and caching
└── README.md                 # This documentation
```

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

Advanced shader material for point cloud rendering with custom vertex and fragment shaders.

**Vertex Shader Features:**

- World-space sizing with correct angular calculation
- Sharpness compensation using mathematical model
- FOV-independent sizing
- Automatic viewport adaptation

**Fragment Shader Features:**

- Power-based falloff for smooth edges
- HDR color support with multiplier
- Per-point sharpness control
- Optimized with pre-computed uniforms

### 3. Material Manager

Singleton manager for efficient material creation and caching.

```typescript
// Get cached material
const material = materialManager.getPointMaterial({
  blendingMode: 'additive',
  opacity: 1.0,
  gamma: 1.0,
});

// Update global parameters
materialManager.updateCameraParams(fov, resolution);
materialManager.updateHDRMultiplier(16.0);
```

---

## Effects Library

### Core Effects

#### Bloom

Professional bloom effect with HDR support:

- Luminance threshold for selective blooming
- Configurable intensity and radius
- Mipmap blur for performance
- Multiple kernel sizes

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

#### SMAA

Subpixel Morphological Anti-Aliasing:

- Superior edge detection
- Multiple quality presets
- Better quality than FXAA
- Moderate performance impact

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

#### Chromatic Aberration

Lens color fringing effect:

- RGB channel separation
- Configurable strength
- Cinematic look

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
Effect Composition (single pass)
    ├── Bloom (HDR space)
    ├── SSAO
    ├── DOF
    ├── Tone Mapping (HDR→LDR)
    ├── Vignette
    └── Chromatic Aberration
    ↓
Anti-Aliasing (FXAA/SMAA)
    ↓
Final Output (sRGB)
```

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
import { PostProcessingManager } from './rendering/postprocessing-manager';
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
postProcessing.setChromaticAberration(true, 0.15);
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

## Migration from Custom System

The rendering system has been migrated from a custom post-processing implementation to the pmndrs/postprocessing library. Key benefits:

1. **Better Performance**: Single-pass effect composition
2. **More Effects**: Access to professional-grade effects
3. **Active Maintenance**: Regular updates and bug fixes
4. **Community Support**: Large user base and documentation
5. **Future-Proof**: Industry-standard implementation

### Breaking Changes

- MSAA no longer supported (use SMAA instead)
- SSAA temporarily unavailable (coming soon)
- Some method signatures changed
- Effect parameters may differ slightly

---

## Troubleshooting

### Common Issues

**Problem: Black screen after enabling effects**

- Check console for WebGL errors
- Verify HDR buffer support
- Try disabling effects one by one

**Problem: Poor performance with all effects**

- Reduce effect quality settings
- Disable SSAO first (highest cost)
- Use FXAA instead of SMAA
- Lower bloom resolution

**Problem: Colors look wrong**

- Verify tone mapping settings
- Check HDR multiplier value
- Ensure proper color space (sRGB)

**Problem: Effects not visible**

- Check effect enabled state
- Verify threshold values
- Ensure proper effect order

---

## API Reference

### PostProcessingManager

| Method                                             | Description                    |
| -------------------------------------------------- | ------------------------------ |
| `render()`                                         | Execute rendering pipeline     |
| `updateBloomSettings(strength, radius, threshold)` | Configure bloom                |
| `setToneMapping(type)`                             | Set tone mapping operator      |
| `updateExposure(value)`                            | Adjust exposure                |
| `setFXAAEnabled(enabled)`                          | Toggle FXAA                    |
| `setSMAAEnabled(enabled)`                          | Toggle SMAA                    |
| `setAOEnabled(enabled, quality)`                   | Configure ambient occlusion    |
| `setDOF(enabled, focus, strength)`                 | Configure depth of field       |
| `setVignetteEnabled(enabled, darkness, offset)`    | Configure vignette             |
| `setChromaticAberration(enabled, strength)`        | Configure chromatic aberration |
| `resize(width, height)`                            | Update render size             |
| `dispose()`                                        | Clean up resources             |

---

## Future Enhancements

- Temporal Anti-Aliasing (TAA)
- Screen Space Reflections (SSR)
- Motion Blur
- Lens Flare
- God Rays
- Color Grading with LUTs
- SSAA Re-implementation
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
