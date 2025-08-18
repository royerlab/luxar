# HDR Setup Guide for Luxar Viewer

## Enabling 10-bit HDR in Chrome

### 1. Chrome Configuration

Navigate to Chrome flags and enable:
```
chrome://flags/#enable-experimental-web-platform-features
chrome://flags/#force-color-profile
```

Set color profile to:
- **Display P3** for Apple displays
- **sRGB** or **Rec2020** for HDR10 displays

### 2. WebGL2 Context Configuration

The viewer needs to request an HDR-capable context:

```javascript
// Request wide gamut P3 color space
const canvas = document.createElement('canvas');
const context = canvas.getContext('webgl2', {
  alpha: false,
  antialias: false,
  depth: true,
  stencil: false,
  powerPreference: 'high-performance',
  // Request wide color gamut
  colorSpace: 'display-p3',  // or 'rec2020' for HDR10
  // Request high bit depth
  desynchronized: true,
  preserveDrawingBuffer: false
});
```

### 3. Three.js HDR Configuration

```javascript
// Configure renderer for HDR
renderer.outputColorSpace = THREE.DisplayP3ColorSpace; // or THREE.Rec2020ColorSpace
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

// Use float textures for HDR render targets
const renderTarget = new THREE.WebGLRenderTarget(width, height, {
  type: THREE.HalfFloatType,  // 16-bit float for HDR
  format: THREE.RGBAFormat,
  colorSpace: THREE.LinearSRGBColorSpace,
  samples: 4  // MSAA if needed
});
```

## Verifying HDR is Working

### 1. Check Canvas Color Space

```javascript
// Add this debug function to verify HDR
function checkHDRSupport() {
  const gl = renderer.getContext();
  
  // Check color space
  const colorSpace = gl.getContextAttributes().colorSpace;
  console.log('Canvas color space:', colorSpace);
  
  // Check if we have float texture support
  const floatTextureSupport = gl.getExtension('EXT_color_buffer_float');
  console.log('Float texture support:', !!floatTextureSupport);
  
  // Check bit depth
  const redBits = gl.getParameter(gl.RED_BITS);
  const greenBits = gl.getParameter(gl.GREEN_BITS);
  const blueBits = gl.getParameter(gl.BLUE_BITS);
  console.log(`Color depth: R${redBits} G${greenBits} B${blueBits}`);
  
  // Check max texture samples
  const maxSamples = gl.getParameter(gl.MAX_SAMPLES);
  console.log('Max MSAA samples:', maxSamples);
  
  return {
    colorSpace,
    floatTextures: !!floatTextureSupport,
    bitsPerChannel: redBits,
    maxSamples
  };
}
```

### 2. Visual HDR Test Pattern

```javascript
// Create HDR test gradient that exceeds sRGB range
function createHDRTestPattern() {
  const geometry = new THREE.PlaneGeometry(10, 2);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      hdrMultiplier: { value: 10.0 }
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float hdrMultiplier;
      varying vec2 vUv;
      void main() {
        // Create gradient from 0 to 10x brightness
        float intensity = vUv.x * hdrMultiplier;
        
        // Red channel: standard range (0-1)
        // Green channel: extended range (0-5)  
        // Blue channel: HDR range (0-10)
        gl_FragColor = vec4(
          vUv.x,
          vUv.x * 5.0,
          intensity,
          1.0
        );
      }
    `
  });
  
  return new THREE.Mesh(geometry, material);
}
```

### 3. Chrome DevTools Verification

1. Open DevTools (F12)
2. Go to **Rendering** tab (three dots menu → More tools → Rendering)
3. Check these options:
   - **Emulate vision deficiencies**: Should show P3 gamut option
   - **Force color gamut**: Can force different color spaces

### 4. System-Level Verification

#### Windows:
```powershell
# Check HDR status
Get-WmiObject -Namespace root\wmi -Class WmiMonitorColorCharacteristics

# Or in Settings
Settings → System → Display → HDR
```

#### macOS:
```bash
# Check color profile
system_profiler SPDisplaysDataType | grep "Color"

# Or in System Settings
System Settings → Displays → Color Profile
```

#### Linux:
```bash
# Check color depth
xdpyinfo | grep "depth"
xrandr --verbose | grep "Brightness"
```

## Platform-Specific HDR Detection

```javascript
// Detect HDR capability
async function detectHDRSupport() {
  // Check for color gamut support
  const supportsP3 = window.matchMedia('(color-gamut: p3)').matches;
  const supportsRec2020 = window.matchMedia('(color-gamut: rec2020)').matches;
  
  // Check for high dynamic range
  const supportsHDR = window.matchMedia('(dynamic-range: high)').matches;
  
  // Check for bit depth
  const supports10Bit = window.matchMedia('(color: 48)').matches; // 48-bit = 16 bits per channel
  
  console.log('HDR Support Detection:');
  console.log('- P3 Gamut:', supportsP3);
  console.log('- Rec2020 Gamut:', supportsRec2020);
  console.log('- High Dynamic Range:', supportsHDR);
  console.log('- 10-bit+ Color:', supports10Bit);
  
  return {
    p3: supportsP3,
    rec2020: supportsRec2020,
    hdr: supportsHDR,
    tenBit: supports10Bit
  };
}
```

## Limitations and Considerations

1. **Browser Limitations:**
   - Not all browsers support HDR WebGL contexts
   - Safari has limited HDR support
   - Firefox HDR support is experimental

2. **Performance Impact:**
   - HDR rendering uses more GPU memory (2x for 16-bit float)
   - May impact performance on older GPUs

3. **Tone Mapping:**
   - Even with HDR display, tone mapping is often needed
   - ACES Filmic is recommended for HDR content

4. **Testing Without HDR Display:**
   - Use Chrome DevTools color gamut emulation
   - Check that values > 1.0 are preserved in shaders
   - Verify float render targets are working

## Current Luxar HDR Implementation

Luxar already supports HDR rendering:
- ✅ Float32 color buffers
- ✅ 16-bit float render targets
- ✅ HDR intensity multipliers
- ✅ ACES tone mapping
- ✅ Values can exceed 1.0

To fully enable 10-bit HDR output, we need to add color space configuration to the renderer initialization.