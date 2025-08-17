# HDR Verification Guide for Chrome

## Summary

Chrome **CAN** render Three.js graphics in 10-bit HDR, but requires:
1. HDR-capable display hardware
2. OS-level HDR enabled
3. Proper WebGL context configuration
4. Float texture support in GPU

## Quick Verification in Luxar

### 1. Check HDR Support
Open Luxar viewer and check the browser console. You'll see:
```
🎨 HDR Display Capabilities
✅ P3 Wide Gamut        (if supported)
✅ Rec2020 Gamut        (if supported)
✅ High Dynamic Range   (if supported)
✅ 10-bit+ Deep Color   (if supported)
✅ Float Textures       (WebGL capability)
📊 Color Buffer Depth: R8 G8 B8
🎯 Recommended Color Space: display-p3
```

### 2. Visual HDR Test
Press **Shift+T** in the viewer to toggle HDR test pattern:
- Bottom band: SDR range (0-1)
- Middle band: Extended range (0-2)
- Top band: Full HDR range (0-10)

If HDR is working, you'll see:
- Smooth gradients without banding
- Brighter values in the top band
- No clipping in bright areas

### 3. Chrome DevTools Verification

#### Open Rendering Tab:
1. Press F12 to open DevTools
2. Three dots menu → More tools → Rendering
3. Check "Force color gamut" dropdown

If you see P3 or Rec2020 options, HDR is available.

#### Check WebGL Context:
```javascript
// Run in console while Luxar is open
const canvas = document.querySelector('canvas');
const gl = canvas.getContext('webgl2');
console.log('Color space:', gl.getContextAttributes().colorSpace);
console.log('Float textures:', !!gl.getExtension('EXT_color_buffer_float'));
console.log('Bits per channel:', gl.getParameter(gl.RED_BITS));
```

## Chrome Configuration for HDR

### 1. Enable Chrome Flags
Navigate to these URLs and enable:
```
chrome://flags/#enable-experimental-web-platform-features
chrome://flags/#force-color-profile
```

Set force-color-profile to:
- **Display P3** for Mac displays
- **Rec709** or **Rec2020** for HDR monitors

### 2. System Configuration

#### Windows 10/11:
1. Settings → System → Display
2. Enable "Use HDR"
3. Adjust "HDR/SDR brightness balance"

#### macOS:
1. System Settings → Displays
2. Select "High Dynamic Range" if available
3. Choose appropriate color profile (P3, Rec709)

#### Linux:
HDR support varies by distribution and desktop environment.
Check with: `xrandr --props | grep HDR`

## Verify HDR is Active

### Method 1: CSS Media Queries
```javascript
// Check in browser console
console.log('P3 Gamut:', matchMedia('(color-gamut: p3)').matches);
console.log('HDR:', matchMedia('(dynamic-range: high)').matches);
console.log('10-bit:', matchMedia('(color: 48)').matches);
```

### Method 2: Test HDR Video
Open an HDR video on YouTube and check if HDR badge appears.

### Method 3: HDR Test Sites
- https://www.uhd4k.net/hdr-test/
- https://webkit.org/blog-files/color-gamut/

## Luxar HDR Features

### Already Implemented:
✅ Float32 color buffers for HDR values
✅ 16-bit float render targets
✅ HDR intensity multipliers (up to 100x)
✅ ACES filmic tone mapping
✅ Automatic HDR detection
✅ P3/Rec2020 gamut detection
✅ HDR test pattern (Shift+T)

### How It Works:
1. **Detection**: On startup, Luxar detects display capabilities
2. **Configuration**: Automatically configures optimal color space
3. **Rendering**: Uses float buffers to preserve HDR values
4. **Tone Mapping**: ACES converts HDR to display range

### Rendering Pipeline:
```
Scene (HDR values) 
  → Float32 buffers 
  → 16-bit render targets
  → Post-processing (bloom, DOF)
  → Tone mapping (ACES)
  → Display color space
  → 10-bit output (if supported)
```

## Troubleshooting

### HDR Not Working?

1. **Check Hardware**:
   - Confirm display supports HDR10/Dolby Vision
   - Verify HDMI 2.0+ or DisplayPort 1.4+ cable

2. **Check OS Settings**:
   - HDR must be enabled at OS level
   - Brightness/contrast properly configured

3. **Check Browser**:
   - Chrome 94+ required
   - Hardware acceleration enabled
   - GPU drivers up to date

4. **Check WebGL**:
   ```javascript
   // Must return true for HDR
   const gl = document.querySelector('canvas').getContext('webgl2');
   console.log(!!gl.getExtension('EXT_color_buffer_float'));
   ```

### Common Issues:

**"HDR looks washed out"**
- Adjust HDR/SDR brightness balance in OS
- Increase tone mapping exposure in Luxar

**"No difference with HDR"**
- Check if content actually has HDR values
- Increase HDR intensity multiplier
- Verify display is in HDR mode

**"Colors look wrong"**
- Calibrate display color profile
- Check Chrome color management settings

## Performance Considerations

HDR rendering impact:
- 2x memory usage (float vs byte)
- 10-20% GPU performance cost
- Higher bandwidth requirements

Optimize by:
- Reducing render resolution
- Disabling unnecessary effects
- Using FXAA instead of MSAA

## Testing Your Display

Run this in Luxar's console to get a full report:
```javascript
// Comprehensive HDR test
(function testHDR() {
  const renderer = window.app?.sceneManager?.renderer;
  if (!renderer) {
    console.log('❌ Luxar not loaded');
    return;
  }
  
  const gl = renderer.getContext();
  const ext = gl.getExtension('EXT_color_buffer_float');
  
  console.log('=== HDR Display Test ===');
  console.log('Display:', {
    p3: matchMedia('(color-gamut: p3)').matches,
    rec2020: matchMedia('(color-gamut: rec2020)').matches,
    hdr: matchMedia('(dynamic-range: high)').matches,
    deepColor: matchMedia('(color: 48)').matches
  });
  
  console.log('WebGL:', {
    floatTextures: !!ext,
    colorDepth: gl.getParameter(gl.RED_BITS),
    renderer: gl.getParameter(gl.RENDERER)
  });
  
  console.log('Three.js:', {
    outputColorSpace: renderer.outputColorSpace,
    toneMapping: renderer.toneMapping,
    exposure: renderer.toneMappingExposure
  });
})();
```

## Conclusion

Chrome with Three.js can achieve true 10-bit HDR rendering when:
- Hardware supports it (display + GPU)
- OS has HDR enabled
- WebGL context supports float textures
- Content contains HDR values (>1.0)

Luxar automatically detects and configures optimal settings for your display!