# HDR Color Specification for Luxar

## Overview

Luxar supports High Dynamic Range (HDR) colors for enhanced visual fidelity in 3D scenes. This document specifies the HDR color system, its ranges, best practices, and implementation details.

## Color Formats

### Supported Data Types

1. **Float32 (Recommended for HDR)**
   - Range: [0.0, ∞) theoretically
   - Practical range: [0.0, 10.0]
   - Storage: 4 bytes per channel
   - Precision: ~7 significant digits

2. **Uint8 (Legacy/SDR)**
   - Range: [0, 255]
   - Mapped to [0.0, 1.0] when converted to float
   - Storage: 1 byte per channel
   - Limited to Standard Dynamic Range (SDR)

## HDR Color Ranges

### Standard Dynamic Range (SDR)
- **Range**: [0.0, 1.0]
- **Usage**: Traditional 8-bit displays
- **Gamma**: Typically 2.2 (sRGB)

### High Dynamic Range (HDR)
- **Minimum**: 0.0 (complete darkness)
- **SDR Maximum**: 1.0 (white on SDR displays)
- **Typical HDR Maximum**: 10.0 (10× brighter than SDR white)
- **Theoretical Maximum**: No hard limit (float32 can represent very large values)

### Recommended Value Ranges

| Intensity Level | Value Range | Description |
|----------------|-------------|-------------|
| Dark shadows | 0.0 - 0.1 | Very dark areas, near black |
| Shadows | 0.1 - 0.3 | Dark areas with some detail |
| Midtones | 0.3 - 0.7 | Normal exposure range |
| Highlights | 0.7 - 1.0 | Bright areas (SDR range) |
| HDR Highlights | 1.0 - 3.0 | Bright lights, reflections |
| HDR Specular | 3.0 - 10.0 | Sun, fire, bright emissive |
| Extreme HDR | > 10.0 | Special effects, not recommended |

## Color Space

### Linear Color Space
- Luxar stores colors in **linear color space**
- No gamma encoding applied to stored values
- Gamma correction applied during rendering

### Gamma Correction
- **Range**: 0.2 to 2.0
- **Default**: 1.0 (no correction)
- **Typical Display Gamma**: 2.2 (sRGB standard)
- Applied per-node for artistic control

## Implementation Details

### Color Broadcasting
When a single color is specified for multiple points:
```python
# Single RGB color tuple/list
color = (1.5, 0.8, 0.3)  # HDR orange
# Broadcast to all points automatically

# Per-point colors
colors = np.array([
    [1.0, 0.5, 0.2],  # Point 1: SDR orange
    [2.0, 1.0, 0.4],  # Point 2: HDR orange  
    [0.5, 0.5, 1.5],  # Point 3: HDR blue
], dtype=np.float32)
```

### HDR Detection
The viewer automatically detects HDR content when:
- Any color channel value > 1.0
- Colors stored as float32

### Tone Mapping
For display on SDR monitors, HDR colors are tone-mapped:
1. **Exposure adjustment**: Scale by exposure value
2. **Gamma correction**: Apply node-specific gamma
3. **Clipping**: Clamp to [0, 1] for display

## Best Practices

### DO:
- Use float32 for HDR colors
- Keep most colors in [0.0, 3.0] range
- Use values > 1.0 for emissive/bright objects
- Document HDR usage in scene metadata
- Test on both HDR and SDR displays

### DON'T:
- Use values > 10.0 unless necessary
- Mix uint8 and float32 in same scene
- Assume all displays support HDR
- Forget gamma correction for SDR output

## Examples

### Emissive Materials
```python
# Glowing object
emissive_color = np.array([5.0, 3.0, 0.5], dtype=np.float32)  # Bright yellow glow

# Fire effect
fire_colors = np.array([
    [8.0, 2.0, 0.3],  # Bright orange core
    [4.0, 1.0, 0.2],  # Orange flames
    [2.0, 0.5, 0.1],  # Darker edges
], dtype=np.float32)
```

### Natural Lighting
```python
# Sunlight
sun_color = np.array([6.0, 5.5, 4.0], dtype=np.float32)

# Sky
sky_color = np.array([0.5, 0.7, 1.2], dtype=np.float32)  # Slightly HDR blue

# Shadows
shadow_color = np.array([0.1, 0.1, 0.15], dtype=np.float32)
```

## Validation

### Color Validation Rules
1. **Type**: Must be float32 for HDR
2. **Shape**: (N, 3) for N points or (3,) for single color
3. **Range**: No negative values
4. **Warning**: Values > 10.0 trigger warning

### Validation Functions
```python
from luxar.constants import COLOR_HDR_TYPICAL_MAX

def validate_hdr_colors(colors):
    # Check for negative values
    if np.any(colors < 0):
        raise ValueError("Colors cannot be negative")
    
    # Warn for extreme HDR
    if np.any(colors > COLOR_HDR_TYPICAL_MAX):
        warnings.warn(f"HDR colors exceed typical maximum of {COLOR_HDR_TYPICAL_MAX}")
    
    return colors
```

## Migration Guide

### From Uint8 to Float32
```python
# Old (uint8)
colors_uint8 = np.array([[255, 128, 64]], dtype=np.uint8)

# New (float32 HDR)
colors_float32 = colors_uint8.astype(np.float32) / 255.0
# Or with HDR enhancement
colors_hdr = colors_float32 * 2.0  # Make 2× brighter
```

### From Legacy to HDR
1. Convert uint8 arrays to float32
2. Divide by 255 to get [0, 1] range
3. Optionally multiply by HDR factor
4. Apply artistic adjustments

## Technical Limitations

### WebGL/Browser Limitations
- Canvas limited to 8-bit output (no true HDR display)
- HDR processing happens in shaders
- Tone mapping required for display

### File Size Considerations
- Float32: 12 bytes per point (3 channels × 4 bytes)
- Uint8: 3 bytes per point (3 channels × 1 byte)
- HDR uses 4× more storage than SDR

### Performance Impact
- Float32 textures may be slower on older GPUs
- Additional tone mapping overhead
- Higher memory bandwidth requirements

## Future Enhancements

### Planned Features
- [ ] OpenEXR export for HDR images
- [ ] Automatic exposure adjustment
- [ ] Advanced tone mapping operators
- [ ] HDR display detection
- [ ] Color space conversions (sRGB, Rec.2020, etc.)

### Under Consideration
- Support for Rec.2100 PQ/HLG
- 16-bit half-float option
- Spectral rendering
- Advanced color grading tools