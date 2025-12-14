# HDR Color Guide for Luxar

**Version**: 1.0
**Last Updated**: 2025-12-13

---

## Overview

Luxar supports High Dynamic Range (HDR) colors, allowing you to create visualizations with colors brighter than standard white. This guide covers everything you need to know about using HDR colors effectively.

---

## What is HDR in Luxar?

**Standard Dynamic Range (SDR)**: Colors in range [0.0, 1.0]
- 0.0 = black
- 1.0 = white
- Traditional 8-bit displays

**High Dynamic Range (HDR)**: Colors can exceed 1.0
- Values > 1.0 represent colors brighter than white
- Simulates emissive materials, bright lights, sun
- Requires float32 data type

---

## Color Value Ranges

| Intensity Level | Value Range | Description | Use Cases |
|----------------|-------------|-------------|-----------|
| Dark shadows | 0.0 - 0.1 | Very dark areas, near black | Background, deep space |
| Shadows | 0.1 - 0.3 | Dark areas with detail | Shadow regions |
| Midtones | 0.3 - 0.7 | Normal exposure | Most scene content |
| Highlights | 0.7 - 1.0 | Bright areas (SDR) | Lit surfaces |
| **HDR Highlights** | 1.0 - 3.0 | Bright lights, reflections | Light sources, reflections |
| **HDR Specular** | 3.0 - 10.0 | Very bright emissive | Sun, fire, lasers |
| **Extreme HDR** | > 10.0 | Special effects | Usually not recommended |

---

## Using HDR Colors in Python

### Basic Usage

```python
import numpy as np
from luxar import Scene

# Create a scene
scene = Scene(name="hdr_demo", store_path="hdr_example.zarr")

# HDR colors (values > 1.0)
emissive_colors = np.array([
    [5.0, 3.0, 0.5],  # Bright yellow glow
    [2.0, 1.5, 1.0],  # Moderate HDR
    [0.5, 0.7, 1.2],  # Slightly bright blue
], dtype=np.float32)

# IMPORTANT: Specify color_mode for float32 colors
scene.add_points(
    name="emissive",
    positions=positions,
    colors=emissive_colors,
    color_mode="hdr",  # Required for float32 colors since v0.4
    radii=radii
)
```

### Color Mode Parameter

As of Luxar v0.4, you **must** specify `color_mode` when using float32 colors:

```python
# For HDR colors (float32, values can exceed 1.0)
color_mode="hdr"

# For SDR colors (float32, but values in [0, 1])
color_mode="sdr"

# Auto-detect is NO LONGER SUPPORTED
```

### Example Scenes

**Glowing Objects:**
```python
# Fire effect with varying intensity
fire_colors = np.array([
    [8.0, 2.0, 0.3],  # Bright core
    [4.0, 1.0, 0.2],  # Flames
    [2.0, 0.5, 0.1],  # Edges
], dtype=np.float32)

scene.add_points("fire", positions, colors=fire_colors, color_mode="hdr")
```

**Natural Lighting:**
```python
# Sunlight (very bright)
sun_color = np.array([6.0, 5.5, 4.0], dtype=np.float32)

# Sky (slightly HDR)
sky_color = np.array([0.5, 0.7, 1.2], dtype=np.float32)

# Shadows (subdued)
shadow_color = np.array([0.1, 0.1, 0.15], dtype=np.float32)
```

---

## Browser HDR Support

### Do I Have an HDR Display?

Run this in your browser console:

```javascript
// Quick HDR capability check
console.log('P3 Wide Gamut:', matchMedia('(color-gamut: p3)').matches);
console.log('Rec2020 Gamut:', matchMedia('(color-gamut: rec2020)').matches);
console.log('HDR Display:', matchMedia('(dynamic-range: high)').matches);
console.log('10-bit Color:', matchMedia('(color: 48)').matches);
```

### Enabling HDR in Chrome

**Chrome Flags** (for testing):
1. Navigate to `chrome://flags/#enable-experimental-web-platform-features`
2. Set to **Enabled**
3. Navigate to `chrome://flags/#force-color-profile`
4. Select appropriate profile:
   - **Display P3** for macOS/iOS
   - **Rec2020** for HDR10 displays
   - **sRGB** for standard displays
5. Restart Chrome

**OS Configuration:**

**Windows 10/11:**
- Settings → System → Display → Enable "Use HDR"
- Adjust HDR/SDR brightness balance

**macOS:**
- System Settings → Displays → Select HDR profile if available

**Linux:**
- HDR support varies by distribution
- Check: `xrandr --props | grep HDR`

---

## Verifying HDR in Luxar

### Console Test Function

Paste this into the Luxar viewer console (`Ctrl+L`):

```javascript
// Comprehensive HDR verification
(function testHDR() {
  console.log('=== Luxar HDR Display Test ===');

  // Check browser capabilities
  console.log('\nDisplay Capabilities:');
  console.log('  P3 Gamut:', matchMedia('(color-gamut: p3)').matches);
  console.log('  Rec2020 Gamut:', matchMedia('(color-gamut: rec2020)').matches);
  console.log('  HDR:', matchMedia('(dynamic-range: high)').matches);
  console.log('  10-bit+:', matchMedia('(color: 48)').matches);

  // Check WebGL context
  const canvas = document.querySelector('canvas');
  if (!canvas) {
    console.log('❌ No canvas found');
    return;
  }

  const gl = canvas.getContext('webgl2');
  if (!gl) {
    console.log('❌ No WebGL2 context');
    return;
  }

  console.log('\nWebGL Configuration:');
  console.log('  Float Textures:', !!gl.getExtension('EXT_color_buffer_float'));
  console.log('  Color Depth: R' + gl.getParameter(gl.RED_BITS) +
              ' G' + gl.getParameter(gl.GREEN_BITS) +
              ' B' + gl.getParameter(gl.BLUE_BITS));
  console.log('  GPU:', gl.getParameter(gl.RENDERER));

  // Check Luxar state
  if (window.__luxarDebug) {
    const state = window.__luxarDebug.getState();
    console.log('\nLuxar HDR Settings:');
    console.log('  Tone Mapping:', state.rendering?.toneMapping || 'Unknown');
    console.log('  Exposure:', state.rendering?.exposure || 'Unknown');
  }

  console.log('\n✅ Test complete');
})();
```

---

## Troubleshooting

### HDR Not Working?

**Check Display Hardware:**
- Confirm display supports HDR10 or Dolby Vision
- Verify cable is HDMI 2.0+ or DisplayPort 1.4+
- Test with HDR video to confirm display works

**Check OS Settings:**
- HDR must be enabled at system level
- Check display settings/preferences
- Some OS require reboot after enabling HDR

**Check Browser:**
- Chrome 94+ required for WebGL HDR
- Hardware acceleration must be enabled: `chrome://gpu`
- Update GPU drivers to latest version

**Check WebGL Support:**
```javascript
// Must return true for HDR
const canvas = document.querySelector('canvas');
const gl = canvas.getContext('webgl2');
console.log('Float support:', !!gl.getExtension('EXT_color_buffer_float'));
```

### Common Issues

**"HDR looks washed out"**
- Adjust HDR/SDR brightness in OS settings
- Try different tone mapping exposure values in viewer
- Check display calibration

**"No visible difference with HDR"**
- Ensure your dataset uses values > 1.0
- Check that `color_mode="hdr"` is set in Python
- Verify display is actually in HDR mode (not SDR)

**"Colors look wrong"**
- Calibrate display with color profile
- Check Chrome color management: `chrome://flags/#force-color-profile`
- Try different tone mapping operators

**"Performance issues"**
- HDR uses 4× more memory (float32 vs uint8)
- Reduce render resolution if needed
- Disable expensive post-processing effects

---

## Current Luxar HDR Implementation

**What Works:**
- ✅ Float32 color buffers for HDR values
- ✅ 16-bit float (HalfFloatType) render targets
- ✅ HDR intensity multipliers (configurable)
- ✅ ACES Filmic tone mapping (default)
- ✅ Multiple tone mapping operators (ACES, AgX, Reinhard, Linear, Neutral)
- ✅ Automatic HDR capability detection on startup
- ✅ Per-node gamma correction
- ✅ Additive blending for bright emissive materials

**Limitations:**
- Canvas output limited to 8-bit in most browsers
- True 10-bit output requires HDR-capable display + browser support
- HDR processing happens in shaders (internal), output may be SDR
- Performance impact on older GPUs

---

## Best Practices

### DO:
- Use `dtype=np.float32` for HDR colors
- Specify `color_mode="hdr"` when adding points/lines
- Keep most colors in [0.0, 3.0] range
- Reserve values > 3.0 for very bright emissive objects
- Use ACES tone mapping for HDR content
- Test on both HDR and SDR displays

### DON'T:
- Use values > 10.0 unless necessary (reduces dynamic range)
- Mix uint8 and float32 colors in same scene
- Assume all users have HDR displays
- Forget to set `color_mode` parameter
- Use negative color values (physically meaningless)

---

## Validation

Luxar validates HDR colors:
- **Negative values**: Rejected with `ValueError`
- **Values > 10.0**: Warning issued (still allowed)
- **NaN/Inf values**: Rejected
- **Wrong dtype**: Warning if using uint8 for values > 1.0

---

## Performance Impact

**Memory Usage:**
- Float32: 12 bytes per point (3 channels × 4 bytes)
- Uint8: 3 bytes per point (3 channels × 1 byte)
- **4× increase** for HDR

**GPU Performance:**
- Float textures: ~10-20% slower on older GPUs
- Tone mapping overhead: ~1-2ms per frame
- Overall impact: Usually negligible for modern GPUs

**Recommendations:**
- Use HDR where it adds value (emissive, bright objects)
- Consider uint8 for purely decorative elements
- Profile performance with your specific dataset

---

## Related Documentation

- **Format Specification**: `LUXAR_ZARR_FORMAT.md` - HDR data format details (in same directory)
- **Encoding Specs**: `packages/luxar/src/luxar/encoding/SPECIFICATIONS.md` - Color encoding modes
- **Rendering Specs**: `packages/luxar-viewer/src/rendering/SPECIFICATIONS.md` - HDR rendering pipeline

---

## References

- Luxar HDR detection: `packages/luxar-viewer/src/utils/hdr-detection.ts`
- Python constants: `packages/luxar/src/luxar/typing_utils/constants.py` (`COLOR_HDR_TYPICAL_MAX = 10.0`)
- HDR tests: `packages/luxar/src/luxar/core/tests/test_hdr_colors.py`
