> **⚠️ Archived — historical document, not maintained.** Kept for design history; it reflects the project state as of its original date and may not match current code. Do not treat it as current guidance. See [the archive README](../README.md) for status labels and retention policy.

# SparkJS Technical Analysis

A comprehensive analysis of [SparkJS](https://github.com/sparkjsdev/spark), an advanced 3D Gaussian Splatting renderer for Three.js. This document extracts implementation tricks, optimizations, and architectural insights that could inform Luxar's nD Gaussian splat renderer development.

**Document Version**: 1.0
**Analysis Date**: December 2025
**SparkJS Version**: 0.1.10 (MIT License)

---

## Executive Summary

SparkJS is a production-quality 3DGS renderer with several notable innovations:

| Category | Key Insight |
|----------|-------------|
| **Memory** | 16 bytes/splat via octahedral quaternion + log scales |
| **Texture Storage** | 3D texture array (2048³) with bit-manipulation addressing |
| **GPU Compute** | Fragment shader as pseudo-compute (WebGL2 lacks compute shaders) |
| **Sorting** | GPU distance → Worker thread → WASM radix sort O(n) |
| **WASM** | Rust-compiled sorting + raycasting with persistent buffers |
| **Workers** | Zero-copy ArrayBuffer transfers, RPC-style communication |
| **Caching** | Version-based generator caching prevents regeneration |
| **Rendering** | Single instanced draw call for millions of splats |
| **Animation** | Stateless procedural animation via hash + time |
| **Shaders** | 2D covariance projection + eigenvalue decomposition |
| **Anti-Aliasing** | Gaussian blur convolution with alpha correction |
| **DoF** | Aperture-based depth of field simulation |
| **Quality** | 20+ configurable parameters for rendering/sorting |
| **WebXR** | Multi-eye averaging for stereoscopic rendering |
| **Skinning** | Dual quaternion skeletal animation (256 bones) |

**For Luxar nD**: The core patterns (accumulator pooling, worker sorting, instanced rendering, WASM acceleration) are directly applicable. However, the 3D-specific data format (quaternions, fixed 16 bytes) cannot be reused for nD Cholesky-parameterized splats.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Encoding & Compression](#data-encoding--compression)
3. [Rendering Pipeline](#rendering-pipeline) - Texture storage, GPU readback, covariance projection
4. [Sorting Algorithms](#sorting-algorithms) - Radix sort, worker integration
5. [WASM Acceleration](#wasm-acceleration) - Rust sorting, raycasting, memory management
6. [Shader Graph System (Dyno)](#shader-graph-system-dyno) - Type-safe shader generation
7. [GPU Memory Management](#gpu-memory-management)
8. [Performance Optimizations](#performance-optimizations) - 17 patterns including WebXR, supersampling
9. [Quality Parameters Reference](#quality-parameters-reference) - Complete parameter table
10. [File Format Support](#file-format-support) - PLY, SPZ, SPLAT, KSPLAT, PCSOGS
11. [Advanced Features](#advanced-features) - Dual quaternion skinning, SDF editing, animation
12. [Applicability to Luxar nD](#applicability-to-luxar-nd)

---

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                       SparkRenderer                              │
│  - Orchestrates entire pipeline                                  │
│  - Extends THREE.Mesh for scene integration                     │
│  - Manages multiple SplatAccumulators                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ SplatMesh     │ │ SplatMesh     │ │ SplatGenerator│
    │ (loaded file) │ │ (procedural)  │ │ (custom)      │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ SplatAccumulator │
                    │ - Aggregates all│
                    │   splats        │
                    │ - Single draw   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  SparkViewpoint │
                    │ - Sort by depth │
                    │ - GPU readback  │
                    │ - Worker sort   │
                    └─────────────────┘
```

### Key Architectural Decisions

1. **Single Draw Call**: All splats from all sources accumulated into one buffer and rendered in a single instanced draw call.

2. **Separation of Concerns**:
   - `SplatMesh`: High-level API for loading/manipulating splats
   - `PackedSplats`: Low-level packed data storage
   - `SplatGenerator`: Base class for procedural generation
   - `SparkViewpoint`: Independent sorting/rendering viewpoints

3. **Pooled Resources**: Accumulators are pooled and reference-counted to minimize allocations.

---

## Data Encoding & Compression

### PackedSplats Format (16 bytes/splat)

SparkJS uses an incredibly compact 16-byte-per-splat format:

```
Bytes 0-3:   RGBA (4 × uint8)
             - RGB: sRGB values 0-255
             - Alpha: opacity 0-255

Bytes 4-9:   Center XYZ (3 × float16 = 6 bytes)
             - 10 bits mantissa, ~0.1% resolution
             - Effective range: ±32K from origin

Bytes 10-11: Quaternion axis (2 × uint8)
             - Octahedral encoding of rotation axis
             - Compresses 3D unit vector to 2 bytes

Bytes 12-14: Scales XYZ (3 × uint8)
             - Logarithmic encoding: e^-12 to e^9
             - ~7% steps between discrete values
             - Zero encodes as e^-30 (essentially invisible)

Byte 15:     Quaternion angle (1 × uint8)
             - Rotation magnitude: 0 to π
```

**Key Insight**: This format is read-only optimized. The octahedral + log encoding makes random writes expensive but achieves excellent memory density.

### Encoding Tricks

#### Float16 Conversion

```typescript
// Efficient float32 → float16 conversion
export function toHalf(f: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, f);
  const bits = view.getUint32(0);

  const sign = (bits >> 16) & 0x8000;
  const exp = ((bits >> 23) & 0xff) - 127 + 15;
  const mantissa = (bits >> 13) & 0x3ff;

  if (exp <= 0) return sign; // Denormal → 0
  if (exp >= 31) return sign | 0x7c00; // Overflow → Inf
  return sign | (exp << 10) | mantissa;
}
```

#### Logarithmic Scale Encoding

```typescript
const LN_SCALE_MIN = -12.0;  // e^-12 ≈ 6e-6
const LN_SCALE_MAX = 9.0;    // e^9 ≈ 8103
const LN_SCALE_ZERO = -30.0; // Essentially zero

function encodeScale(scale: number): number {
  const ln = Math.log(scale);
  if (ln < LN_SCALE_ZERO) return 0;  // Too small
  // Map [LN_SCALE_MIN, LN_SCALE_MAX] → [1, 255]
  return Math.round(1 + 254 * (ln - LN_SCALE_MIN) / (LN_SCALE_MAX - LN_SCALE_MIN));
}
```

#### Octahedral Quaternion Encoding

This is a clever technique to encode a quaternion using only 3 bytes:

```typescript
// Encode rotation as: octahedral(axis) + angle
function encodeQuaternion(q: THREE.Quaternion): [number, number, number] {
  // Extract axis-angle from quaternion
  const angle = 2 * Math.acos(Math.min(1, Math.abs(q.w)));
  const sinHalf = Math.sin(angle / 2);

  let axis = new THREE.Vector3(q.x, q.y, q.z);
  if (sinHalf > 1e-6) axis.divideScalar(sinHalf);

  // Octahedral encode the axis (2 bytes)
  const [u, v] = octEncode(axis);  // Each in [0, 255]

  // Encode angle (1 byte): [0, π] → [0, 255]
  const angleByte = Math.round(255 * angle / Math.PI);

  return [u, v, angleByte];
}

// Octahedral encoding: unit sphere → unit square
function octEncode(n: THREE.Vector3): [number, number] {
  const sum = Math.abs(n.x) + Math.abs(n.y) + Math.abs(n.z);
  let u = n.x / sum;
  let v = n.y / sum;

  if (n.z < 0) {
    // Fold lower hemisphere
    const ou = u, ov = v;
    u = (1 - Math.abs(ov)) * (ou >= 0 ? 1 : -1);
    v = (1 - Math.abs(ou)) * (ov >= 0 ? 1 : -1);
  }

  // Map [-1, 1] → [0, 255]
  return [
    Math.round(127.5 + 127.5 * u),
    Math.round(127.5 + 127.5 * v)
  ];
}
```

### SPZ Format (Niantic Compressed)

SparkJS supports the SPZ format with several compression techniques:

```typescript
// SPZ v3: "Smallest Three" quaternion compression
// Store only 3 of 4 quaternion components (32 bits total)
function encodeQuaternionSmallestThree(q: THREE.Quaternion): number {
  const components = [q.x, q.y, q.z, q.w];

  // Find largest component
  let maxIdx = 0;
  let maxVal = Math.abs(components[0]);
  for (let i = 1; i < 4; i++) {
    if (Math.abs(components[i]) > maxVal) {
      maxIdx = i;
      maxVal = Math.abs(components[i]);
    }
  }

  // Encode: 2 bits for index, 10 bits × 3 for remaining components
  // Sign of largest is implicitly positive (negate quat if needed)
  let bits = maxIdx;  // 2 bits
  for (let i = 0, shift = 2; i < 4; i++) {
    if (i !== maxIdx) {
      // Map [-0.707, 0.707] → [0, 1023]
      const val = Math.round(511.5 + 511.5 * components[i] / 0.707107);
      bits |= (val & 0x3ff) << shift;
      shift += 10;
    }
  }
  return bits;
}
```

### Spherical Harmonics Encoding

```typescript
// SH1: 9 coefficients (3 per RGB) packed into 64 bits
function encodeSh1Rgb(coeffs: number[][]): [number, number] {
  // Pack 9 values as 7-bit signed integers
  let bits0 = 0, bits1 = 0;
  for (let i = 0; i < 9; i++) {
    const val = Math.round(63 + 63 * coeffs[i % 3][Math.floor(i / 3)]);
    const clamped = Math.max(0, Math.min(127, val));
    if (i < 4) bits0 |= clamped << (i * 7);
    else bits1 |= clamped << ((i - 4) * 7);
  }
  return [bits0, bits1];
}
```

---

## Rendering Pipeline

### Texture Storage Architecture

SparkJS uses a clever 3D texture array trick to store millions of splats efficiently:

```typescript
// Texture addressing: 33-bit virtual address space using 3D texture array
// Index → (x, y, layer) coordinates via bit manipulation

const SPLAT_TEX_WIDTH = 2048;   // 11 bits (0-2047)
const SPLAT_TEX_HEIGHT = 2048;  // 11 bits
const SPLAT_TEX_DEPTH = 2048;   // 11 bits (layers)

// Max addressable splats: 2048³ = 8.5 billion (theoretical)
// Practical limit: ~4M splats per layer × 2048 layers

// Index to texture coordinate conversion (in GLSL):
int layer = index >> 22;              // Top 11 bits
int y = (index >> 11) & 0x7ff;        // Middle 11 bits
int x = index & 0x7ff;                // Bottom 11 bits
uvec4 data = texelFetch(tex, ivec3(x, y, layer), 0);
```

#### Why Textures Instead of Vertex Buffers?

| Aspect | Vertex Buffer | Texture Storage |
|--------|---------------|-----------------|
| **Random Access** | Sequential only (vertex shader gets one vertex) | Any splat from any shader (`texelFetch`) |
| **Indirect Indexing** | Requires separate index buffer | Built-in via texture coordinates |
| **Update Granularity** | Must re-upload entire buffer | Partial texture updates possible |
| **Cross-Shader Access** | Vertex data unavailable in fragment shader | Textures accessible everywhere |
| **Sorting** | Requires reordering vertex data | Just reorder index array |

**The key insight**: With textures, sorting becomes O(n) index reordering instead of O(n) data movement. The splat data stays in place; only a 4-byte index per splat needs to be sorted and uploaded each frame.

```
Traditional approach (vertex buffers):
  Sort → Copy 16 bytes × N splats → Upload to GPU
  Memory bandwidth: 16N bytes/frame

SparkJS approach (texture + index buffer):
  Sort → Copy 4 bytes × N splats → Upload indices only
  Memory bandwidth: 4N bytes/frame (4× reduction)
```

**Key Texture Tricks**:

1. **RGBA32UI Format**: Each texel stores 4 × uint32 = 16 bytes = 1 splat
   ```typescript
   texture.format = THREE.RGBAIntegerFormat;
   texture.type = THREE.UnsignedIntType;
   texture.internalFormat = 'RGBA32UI';
   ```

2. **Nearest Filtering**: Essential for integer data (no interpolation)
   ```typescript
   texture.minFilter = THREE.NearestFilter;
   texture.magFilter = THREE.NearestFilter;
   ```

3. **Partial Layer Updates**: Only allocate height needed for last layer
   ```typescript
   function getTextureSize(numSplats: number): [width, height, depth] {
     const perLayer = SPLAT_TEX_WIDTH * SPLAT_TEX_HEIGHT;  // 4M per layer
     const layers = Math.ceil(numSplats / perLayer);
     const lastLayerSplats = numSplats % perLayer || perLayer;
     const lastLayerHeight = Math.ceil(lastLayerSplats / SPLAT_TEX_WIDTH);
     return [SPLAT_TEX_WIDTH, lastLayerHeight, layers];
   }
   ```

4. **Pseudo-Compute via Fragment Shader**: WebGL2 lacks compute shaders, so SparkJS renders a fullscreen quad to generate splat data
   ```glsl
   // Fragment shader as "compute shader"
   void main() {
     // Convert fragment coordinates to splat index
     int index = int(targetLayer << 22) +
                 int(uint(gl_FragCoord.y) << 11) +
                 int(gl_FragCoord.x);

     if (index < targetCount) {
       produceSplat(index);  // Write to uvec4 output
     } else {
       target = uvec4(0u);
     }
   }
   ```

### GPU Readback Strategy

The only portable WebGL2 readback format is RGBA8. SparkJS uses a multi-pass approach:

```typescript
class Readback {
  // WebGL2 constraint: can only read RGBA8 pixels
  private target: THREE.WebGLArrayRenderTarget;

  async renderReadback(renderer: THREE.WebGLRenderer): Promise<Uint8Array> {
    // 1. Save renderer state
    const savedXR = renderer.xr.enabled;
    const savedAutoClear = renderer.autoClear;
    renderer.xr.enabled = false;
    renderer.autoClear = false;

    // 2. Render in horizontal strips (handle large datasets)
    const results: Promise<void>[] = [];
    for (let layer = 0; layer < this.depth; layer++) {
      // Set scissor to current layer
      renderer.setScissor(0, 0, this.width, this.height);
      renderer.setScissorTest(true);

      // Render dyno program to target
      renderer.setRenderTarget(this.target, layer);
      renderer.render(this.scene, this.camera);

      // Async readback (non-blocking)
      results.push(
        renderer.readRenderTargetPixelsAsync(
          this.target, 0, 0, this.width, this.height, this.buffer
        )
      );
    }

    // 3. Wait for all layers
    await Promise.all(results);

    // 4. Restore state
    renderer.xr.enabled = savedXR;
    renderer.autoClear = savedAutoClear;

    return this.buffer;
  }
}
```

**Key insight**: `readRenderTargetPixelsAsync()` is non-blocking, allowing multiple layers to be queued without waiting.

### Frame Update Flow

```
1. SparkRenderer.onBeforeRender()
   │
   ├─► Compute deltaTime
   ├─► Update camera matrices
   ├─► Update uniforms (renderSize, near/far, quality params)
   │
   ▼
2. SparkRenderer.updateInternal()
   │
   ├─► Check accumulator availability (backpressure)
   ├─► compileScene() - collect all SplatGenerators
   ├─► frameUpdate() on each generator
   ├─► Version tracking (detect changes)
   │
   ├─► If update needed:
   │   ├─► Sort generators by version delta
   │   ├─► generateMapping() - assign splat ranges
   │   ├─► accumulator.generateSplats() - GPU generation
   │   └─► Update versions, release old accumulator
   │
   ▼
3. SparkViewpoint.autoPoll()
   │
   ├─► Check camera movement thresholds
   ├─► driveSort() if needed
   │   ├─► GPU readback of distances
   │   ├─► Worker thread radix sort
   │   └─► updateDisplay() with sorted indices
   │
   ▼
4. THREE.js render() → SparkRenderer draws
   │
   └─► Single instanced draw call with sorted ordering
```

### Lazy Generation with Caching

SparkJS uses aggressive caching to avoid redundant work:

```typescript
// SplatAccumulator.generateSplats()
generateSplats(mappings: GeneratorMapping[], modifier: SplatModifier) {
  for (const mapping of mappings) {
    const gen = mapping.generator;

    // Check if generator has changed
    const cached = this.mapping.get(gen.uuid);
    if (cached &&
        cached.version === gen.version &&
        cached.numSplats === gen.numSplats) {
      // Skip - use cached data
      continue;
    }

    // Generate new splats
    gen.generate(this.splats, mapping.offset, modifier);

    // Update cache
    this.mapping.set(gen.uuid, {
      version: gen.version,
      numSplats: gen.numSplats
    });
  }
}
```

### 2D Covariance Projection (Vertex Shader)

The vertex shader implements the full 3DGS projection pipeline:

```glsl
// 1. Build 3x3 rotation-scale matrix from quaternion and scales
mat3 RS = scaleQuaternionToMatrix(scales, quaternion);

// 2. Compute 3D covariance: Σ = RS × RSᵀ
mat3 cov3D = RS * transpose(RS);

// 3. Compute Jacobian of perspective projection
// J = ∂(screen) / ∂(view)
float fx = focalX;  // Focal length in pixels
float fy = focalY;
float z2 = viewCenter.z * viewCenter.z;
mat3 J = mat3(
  fx / viewCenter.z, 0.0, -fx * viewCenter.x / z2,
  0.0, fy / viewCenter.z, -fy * viewCenter.y / z2,
  0.0, 0.0, 0.0
);

// 4. Project 3D covariance to 2D: Σ₂D = Jᵀ × Σ₃D × J
mat3 cov2D = transpose(J) * cov3D * J;

// 5. Extract 2×2 XY components for eigendecomposition
float a = cov2D[0][0];  // σ²ₓ
float b = cov2D[0][1];  // σₓᵧ (covariance)
float d = cov2D[1][1];  // σ²ᵧ
```

### Eigenvalue Decomposition (Vertex Shader)

Efficient 2×2 eigenvalue computation for ellipse sizing:

```glsl
// Eigenvalues of 2×2 symmetric matrix [[a, b], [b, d]]
float trace = a + d;
float det = a * d - b * b;
float disc = sqrt(max(0.0, 0.25 * trace * trace - det));
float lambda1 = 0.5 * trace + disc;  // Larger eigenvalue
float lambda2 = 0.5 * trace - disc;  // Smaller eigenvalue

// Eigenvector for larger eigenvalue (principal axis)
vec2 v1;
if (abs(b) > 1e-6) {
  v1 = normalize(vec2(lambda1 - d, b));
} else {
  v1 = (a > d) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
}

// Ellipse semi-axes (3σ radius for 99.7% coverage)
float axis1 = 3.0 * sqrt(lambda1);
float axis2 = 3.0 * sqrt(lambda2);
```

### Anti-Aliasing via Blur Convolution

SparkJS applies Gaussian blur to prevent aliasing:

```glsl
// Pre-blur: add to covariance before projection
// (equivalent to convolving with larger Gaussian)
mat3 blurCov = cov3D;
blurCov[0][0] += preBlurAmount * preBlurAmount;
blurCov[1][1] += preBlurAmount * preBlurAmount;
blurCov[2][2] += preBlurAmount * preBlurAmount;

// Post-blur: scale 2D ellipse
float blurScale = 1.0 + blurAmount;
axis1 *= blurScale;
axis2 *= blurScale;

// Anti-aliasing intensity correction
// When blur increases area, reduce alpha proportionally
float originalDet = det;
float blurredDet = (a + blurAmount) * (d + blurAmount) - b * b;
float alphaCorrection = sqrt(originalDet / blurredDet);
rgba.a *= alphaCorrection;
```

### Depth of Field Simulation

```glsl
// Aperture-based blur based on defocus distance
float focalDist = focalDistance;
float aperture = apertureAngle;  // In radians

// Circle of confusion radius
float defocus = abs(viewCenter.z - focalDist);
float cocRadius = defocus * tan(aperture);

// Add CoC to splat variance
float cocVariance = cocRadius * cocRadius;
axis1 = sqrt(axis1 * axis1 + cocVariance);
axis2 = sqrt(axis2 * axis2 + cocVariance);
```

### Instanced Geometry

```typescript
// SplatGeometry extends THREE.InstancedBufferGeometry
class SplatGeometry {
  // Quad vertices: 4 corners at (-1,-1), (1,-1), (1,1), (-1,1)
  static QUAD_VERTICES = new Float32Array([
    -1, -1, 0,
     1, -1, 0,
     1,  1, 0,
    -1,  1, 0
  ]);

  // Two triangles per quad
  static QUAD_INDICES = new Uint16Array([0, 1, 2, 0, 2, 3]);

  constructor(maxInstances: number) {
    super();

    // Static vertex positions
    this.setAttribute('position',
      new THREE.BufferAttribute(SplatGeometry.QUAD_VERTICES, 3));
    this.setIndex(new THREE.BufferAttribute(SplatGeometry.QUAD_INDICES, 1));

    // Instance ordering (updated each frame)
    this.ordering = new Uint32Array(maxInstances);
    this.orderingAttr = new THREE.InstancedBufferAttribute(
      this.ordering, 1
    );
    this.orderingAttr.setUsage(THREE.DynamicDrawUsage);
    this.setAttribute('ordering', this.orderingAttr);
  }

  update(numSplats: number, sortedIndices: Uint32Array) {
    this.ordering.set(sortedIndices);
    this.orderingAttr.addUpdateRange(0, numSplats);
    this.orderingAttr.needsUpdate = true;
    this.instanceCount = numSplats;
  }
}
```

### Fragment Shader: Gaussian Falloff

```glsl
// Core alpha blending with exponential decay
void main() {
  // UV distance from splat center (passed from vertex shader)
  float z = dot(vUV, vUV);  // Squared distance

  // Discard fragments outside max standard deviations
  if (z > maxStdDev * maxStdDev) discard;

  // Gaussian falloff: α × exp(-0.5 × d²)
  // falloff parameter: 0 = flat shading, 1 = full Gaussian
  rgba.a *= mix(1.0, exp(-0.5 * z), falloff);

  // Alpha threshold culling
  if (rgba.a < minAlpha) discard;

  // Stochastic rendering: convert alpha to binary visibility
  if (stochastic) {
    // Hash-based pseudo-randomness for temporal coherence
    float hash = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    if (hash > rgba.a) discard;
    rgba.a = 1.0;  // Full opacity for surviving fragments
  }

  // Color space conversion
  if (encodeLinear) {
    rgba.rgb = srgbToLinear(rgba.rgb);
  }

  // Output: premultiplied or straight alpha
  #ifdef PREMULTIPLIED_ALPHA
    fragColor = vec4(rgba.rgb * rgba.a, rgba.a);
  #else
    fragColor = rgba;
  #endif
}
```

### 2D Gaussian Splatting Mode (2DGS)

SparkJS supports oriented 2D Gaussians (zero-thickness splats):

```typescript
// enable2DGS: Interprets splats with zero Z-scale as oriented 2D projections
class SparkRenderer {
  enable2DGS = false;  // Enable 2D Gaussian interpretation

  // In vertex shader, when enable2DGS && scales.z ≈ 0:
  // - Treat splat as a flat disk aligned to minimum scale axis
  // - Skip 3D covariance computation, use direct 2D ellipse
  // - Apply view-dependent orientation to face camera
}
```

---

## Sorting Algorithms

### Why Sorting Matters for Gaussian Splatting

Gaussian splats use **alpha blending** for semi-transparent rendering. Unlike opaque geometry where depth buffering handles occlusion, transparent objects must be rendered **back-to-front** for correct compositing:

```
Correct (back-to-front):   A over B over C = A + (1-αA)(B + (1-αB)C)
Wrong (any other order):   Visual artifacts, color bleeding, popping
```

**Challenge**: Re-sorting millions of splats every frame when the camera moves.

### Two-Tier Sorting Strategy

SparkJS uses a clever two-tier approach:

1. **GPU**: Compute distances via shader, readback to CPU
2. **CPU Worker**: Radix sort in background thread

#### Why This Split?

| Task | Best Processor | Reason |
|------|---------------|--------|
| Distance computation | GPU | Embarrassingly parallel, one op per splat |
| Sorting | CPU | GPUs lack efficient sorting primitives in WebGL2 |
| Main thread | Neither | Must stay free for UI responsiveness |

**Key insight**: WebGL2 has no compute shaders. GPU sorting would require multiple draw passes with ping-pong buffers—slower than CPU radix sort for this data size.

### Distance Computation (GPU)

```glsl
// Compute sort metric per splat
float computeSortMetric(vec3 center, vec3 viewPos, vec3 viewDir,
                        float depthBias, bool radial, bool cull360) {
  vec3 toCenter = center - viewPos;

  // Back-face culling (unless 360° mode)
  if (!cull360 && dot(toCenter, viewDir) < depthBias) {
    return 1e38;  // INFINITY → sort to end
  }

  if (radial) {
    // Euclidean distance (for 360° views)
    return length(toCenter);
  } else {
    // Z-depth (standard perspective)
    return dot(toCenter, viewDir) + depthBias;
  }
}
```

### Why Radix Sort?

| Algorithm | Time Complexity | 1M Splats | 10M Splats |
|-----------|-----------------|-----------|------------|
| Quick Sort | O(n log n) | ~20M comparisons | ~230M comparisons |
| Merge Sort | O(n log n) | ~20M comparisons | ~230M comparisons |
| **Radix Sort** | **O(n)** | **~1M passes** | **~10M passes** |

**Radix sort wins because**:
1. **No comparisons**: Distributes values into buckets by digit, not pairwise comparison
2. **Predictable memory access**: Sequential reads, bucket writes (cache-friendly)
3. **Fixed key size**: Depth values are 16 or 32 bits—known at compile time
4. **Stable sort**: Equal distances maintain original order (reduces flickering)

**The trade-off**: Radix sort requires O(k) extra space for buckets (k = key range). For 16-bit floats, that's 65,536 buckets × 4 bytes = 256KB—trivial for modern CPUs.

### Radix Sort (CPU Worker)

```typescript
// 16-bit radix sort for float16 distances
function radixSort16(
  distances: Uint16Array,
  indices: Uint32Array,
  numSplats: number
): Uint32Array {
  const BUCKETS = 0x7c01;  // Max float16 value + 1

  // Allocate persistent buffers (avoid GC)
  const counts = new Uint32Array(BUCKETS);
  const output = new Uint32Array(numSplats);

  // Count phase
  counts.fill(0);
  for (let i = 0; i < numSplats; i++) {
    const d = distances[i];
    if (d < BUCKETS) counts[d]++;
  }

  // Prefix sum (cumulative counts)
  let sum = 0;
  for (let i = BUCKETS - 1; i >= 0; i--) {
    const c = counts[i];
    counts[i] = sum;
    sum += c;
  }

  // Scatter phase (back-to-front for stability)
  for (let i = numSplats - 1; i >= 0; i--) {
    const d = distances[i];
    if (d < BUCKETS) {
      output[counts[d]++] = indices[i];
    }
  }

  return output;
}
```

### 32-bit Radix Sort

For higher precision, SparkJS uses two-pass 32-bit sorting:

```typescript
function radixSort32(distances: Float32Array, numSplats: number): Uint32Array {
  const RADIX = 65536;  // 16-bit buckets

  // Convert float to sortable uint32
  const keys = new Uint32Array(numSplats);
  for (let i = 0; i < numSplats; i++) {
    const bits = floatBitsToUint(distances[i]);
    // Handle negative floats: flip all bits if sign set
    keys[i] = (bits & 0x80000000) ? ~bits : bits ^ 0x80000000;
  }

  // Two-pass radix sort
  // Pass 1: Sort by low 16 bits
  const temp = radixPass(keys, indices, 0, RADIX);
  // Pass 2: Sort by high 16 bits
  return radixPass(keys, temp, 16, RADIX);
}
```

### Worker Thread Integration

```typescript
// Main thread: SplatWorker
class SplatWorker {
  private worker: Worker;
  private pending = new Map<number, Promise<any>>();
  private nextId = 0;

  async sort(distances: Float32Array): Promise<Uint32Array> {
    const id = this.nextId++;

    return new Promise((resolve) => {
      this.pending.set(id, resolve);

      // Transfer buffer ownership (zero-copy)
      this.worker.postMessage(
        { name: 'sort', id, args: [distances] },
        [distances.buffer]  // Transferable
      );
    });
  }
}

// Worker thread
self.onmessage = (e) => {
  const { name, id, args } = e.data;

  if (name === 'sort') {
    const result = radixSort32(args[0], args[0].length);
    self.postMessage({ id, result }, [result.buffer]);
  }
};
```

---

## WASM Acceleration

SparkJS uses Rust-compiled WebAssembly for performance-critical operations.

### Why WASM Instead of JavaScript?

| Aspect | JavaScript | WebAssembly |
|--------|------------|-------------|
| **Numeric operations** | Boxed numbers, type coercion | Native i32/f32, no overhead |
| **Memory layout** | Objects scattered in heap | Contiguous linear memory |
| **GC pauses** | Unpredictable stalls | No garbage collector |
| **SIMD** | Limited (experimental) | Full SIMD.js support |
| **Compilation** | JIT, warm-up time | AOT, instant full speed |

**Benchmark** (sorting 1M splats):
```
JavaScript radix sort:  ~45ms (with JIT warm-up: ~80ms first run)
WASM radix sort:        ~12ms (consistent every run)
Speedup:                3.7× faster, more predictable
```

**Critical insight**: For sorting, the tight inner loop iterates millions of times. JavaScript's type checks and potential deoptimizations add ~3 cycles per iteration. WASM's guaranteed types eliminate this overhead.

### Why Rust for WASM?

1. **Zero-cost abstractions**: Rust iterators compile to optimal loops
2. **No runtime**: Unlike Go/Java WASM, no GC or runtime embedded
3. **wasm-bindgen**: Seamless TypeScript interop with typed arrays
4. **Memory safety**: No buffer overflows even in hot paths

### WASM Module Structure

```
rust/spark-internal-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs      # Entry point, exports to JS
│   ├── sort.rs     # Radix sorting algorithms
│   └── raycast.rs  # Ray-splat intersection
```

### WASM-Exported Functions

```rust
// lib.rs - Functions exported to JavaScript

#[wasm_bindgen]
pub fn sort_splats(
    num_splats: u32,
    readback: Uint16Array,    // Float16 distances from GPU
    ordering: Uint32Array     // Output: sorted indices
) -> u32;  // Returns active splat count

#[wasm_bindgen]
pub fn sort32_splats(
    num_splats: u32,
    readback: Uint32Array,    // Float32 distances (as bits)
    ordering: Uint32Array
) -> u32;

#[wasm_bindgen]
pub fn raycast_splats(
    origin_x: f32, origin_y: f32, origin_z: f32,
    dir_x: f32, dir_y: f32, dir_z: f32,
    near: f32, far: f32,
    num_splats: u32,
    packed_splats: Uint32Array,
    raycast_ellipsoid: bool,
    ln_scale_min: f32, ln_scale_max: f32
) -> Float32Array;  // Returns hit distances
```

### WASM 16-bit Radix Sort (Rust)

Single-pass counting sort for float16 depth values:

```rust
pub fn sort_internal(buffers: &mut SortBuffers, num_splats: usize) -> Result<u32, String> {
    let SortBuffers { readback, ordering, buckets } = buffers;
    let readback = &readback[..num_splats];

    // 1. Initialize 31745 buckets (0x7c01 = max float16 + 1)
    buckets.clear();
    buckets.resize(DEPTH_SIZE_F16, 0);  // DEPTH_SIZE_F16 = 0x7c01

    // 2. Count phase - tally splats per distance bucket
    for &metric in readback.iter() {
        if (metric as u32) < DEPTH_INFINITY_F16 {
            buckets[metric as usize] += 1;
        }
    }

    // 3. Prefix sum (reverse for back-to-front ordering)
    let mut active_splats = 0;
    for count in buckets.iter_mut().rev().skip(1) {
        let new_total = active_splats + *count;
        *count = active_splats;
        active_splats = new_total;
    }

    // 4. Scatter phase - write sorted indices
    for (index, &metric) in readback.iter().enumerate() {
        if (metric as u32) < DEPTH_INFINITY_F16 {
            ordering[buckets[metric as usize] as usize] = index as u32;
            buckets[metric as usize] += 1;
        }
    }

    Ok(active_splats)
}
```

**Performance**: O(n) time, O(65536) space. Single pass through data.

### WASM 32-bit Radix Sort (Rust)

Two-pass radix sort using base 2^16:

```rust
pub fn sort32_internal(
    buffers: &mut Sort32Buffers,
    max_splats: usize,
    num_splats: usize,
) -> Result<u32, String> {
    buffers.ensure_size(max_splats);
    let Sort32Buffers { readback, ordering, buckets16lo, buckets16hi, scratch } = buffers;
    let keys = &readback[..num_splats];

    // 1. Count both low and high 16-bit buckets simultaneously
    buckets16lo.fill(0);
    buckets16hi.fill(0);
    for &key in keys.iter() {
        if key < DEPTH_INFINITY_F32 {
            let inv = !key;  // Bitwise NOT for descending order
            buckets16lo[(inv & 0xFFFF) as usize] += 1;
            buckets16hi[(inv >> 16) as usize] += 1;
        }
    }

    // 2. Prefix sum for low buckets
    let mut total: u32 = 0;
    for slot in buckets16lo.iter_mut() {
        let cnt = *slot;
        *slot = total;
        total = total.wrapping_add(cnt);
    }
    let active_splats = total;

    // 3. First pass: sort by low 16 bits into scratch
    for (i, &key) in keys.iter().enumerate() {
        if key < DEPTH_INFINITY_F32 {
            let inv = !key;
            let lo = (inv & 0xFFFF) as usize;
            scratch[buckets16lo[lo] as usize] = i as u32;
            buckets16lo[lo] += 1;
        }
    }

    // 4. Prefix sum for high buckets
    let mut sum: u32 = 0;
    for slot in buckets16hi.iter_mut() {
        let cnt = *slot;
        *slot = sum;
        sum = sum.wrapping_add(cnt);
    }

    // 5. Second pass: sort by high 16 bits into final ordering
    for &idx in scratch.iter().take(active_splats as usize) {
        let key = keys[idx as usize];
        let inv = !key;
        let hi = (inv >> 16) as usize;
        ordering[buckets16hi[hi] as usize] = idx;
        buckets16hi[hi] += 1;
    }

    Ok(active_splats)
}
```

**Key tricks**:
- `!key` (bitwise NOT) converts ascending to descending order
- Two separate bucket arrays avoid re-counting
- Scratch buffer for intermediate results

### WASM Raycasting (Rust)

```rust
// Ellipsoid ray intersection for accurate picking
fn raycast_ellipsoid(
    origin: [f32; 3],
    direction: [f32; 3],
    center: [f32; 3],
    scales: [f32; 3],
    quat: [f32; 4],
    near: f32,
    far: f32,
) -> Option<f32> {
    // 1. Transform ray to ellipsoid local space
    let local_origin = quat_rotate_inv(quat, sub(origin, center));
    let local_dir = quat_rotate_inv(quat, direction);

    // 2. Scale to unit sphere space
    let scaled_origin = [
        local_origin[0] / scales[0],
        local_origin[1] / scales[1],
        local_origin[2] / scales[2],
    ];
    let scaled_dir = [
        local_dir[0] / scales[0],
        local_dir[1] / scales[1],
        local_dir[2] / scales[2],
    ];

    // 3. Standard ray-sphere intersection (quadratic formula)
    let a = dot(scaled_dir, scaled_dir);
    let b = 2.0 * dot(scaled_origin, scaled_dir);
    let c = dot(scaled_origin, scaled_origin) - 1.0;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }

    let t = (-b - discriminant.sqrt()) / (2.0 * a);
    if t >= near && t <= far {
        Some(t)
    } else {
        None
    }
}

// Sphere approximation for faster but less accurate picking
fn raycast_sphere(
    origin: [f32; 3],
    direction: [f32; 3],
    center: [f32; 3],
    radius: f32,  // Average of scales
    near: f32,
    far: f32,
) -> Option<f32> {
    // Standard ray-sphere intersection
    let oc = sub(origin, center);
    let a = dot(direction, direction);
    let b = 2.0 * dot(oc, direction);
    let c = dot(oc, oc) - radius * radius;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }

    let t = (-b - discriminant.sqrt()) / (2.0 * a);
    if t >= near && t <= far {
        Some(t)
    } else {
        None
    }
}
```

### WASM Memory Management

```rust
// Thread-local persistent buffers to avoid allocation during hot path
thread_local! {
    static SORT_BUFFERS: RefCell<SortBuffers> = RefCell::new(SortBuffers::new());
    static SORT32_BUFFERS: RefCell<Sort32Buffers> = RefCell::new(Sort32Buffers::new());
}

impl SortBuffers {
    fn ensure_size(&mut self, max_splats: usize) {
        if self.ordering.len() < max_splats {
            // Grow with headroom to amortize allocations
            let new_size = (max_splats * 3 / 2).max(1024);
            self.ordering.resize(new_size, 0);
            self.readback.resize(new_size, 0);
        }
    }
}
```

### Web Worker Architecture

#### Why Offload to a Web Worker?

The browser's **main thread** handles:
- JavaScript execution
- DOM updates
- Event handling
- **Rendering** (requestAnimationFrame)

If sorting blocks the main thread for 50ms, the frame rate drops from 60fps to 20fps. Users perceive this as stuttering.

```
Without worker:
  Frame N: [Render 10ms][Sort 50ms][Render 10ms] = 70ms = 14fps ❌

With worker (async):
  Main:   [Render 10ms][Render 10ms][Render 10ms]...
  Worker:          [Sort 50ms]
  Result: Smooth 60fps, sort result applied when ready ✓
```

**Key pattern**: Sorting happens in parallel with rendering. The display uses the previous frame's sort order until the new one arrives.

SparkJS uses a dedicated Web Worker for CPU-intensive tasks:

```typescript
// worker.ts - Runs in separate thread
const handlers: Record<string, Function> = {
  // File parsing (runs on worker to not block main thread)
  unpackPly: (bytes: Uint8Array, encoding: SplatEncoding) => {
    const reader = new PlyReader(bytes);
    return reader.parseSplats(encoding);
  },

  decodeSpz: (bytes: Uint8Array) => {
    const reader = new SpzReader(bytes);
    return reader.decode();
  },

  // Sorting (WASM or JS fallback)
  sort16: (numSplats: number, readback: Uint16Array, ordering: Uint32Array) => {
    if (WASM_SPLAT_SORT) {
      return sort_splats(numSplats, readback, ordering);
    } else {
      return jsFallbackSort16(numSplats, readback, ordering);
    }
  },

  sort32: (numSplats: number, readback: Uint32Array, ordering: Uint32Array) => {
    if (WASM_SPLAT_SORT) {
      return sort32_splats(numSplats, readback, ordering);
    } else {
      return jsFallbackSort32(numSplats, readback, ordering);
    }
  },
};

self.onmessage = async (e: MessageEvent) => {
  const { name, id, args } = e.data;
  try {
    const result = await handlers[name](...args);
    // Transfer ownership of ArrayBuffers back (zero-copy)
    const transferables = getArrayBuffers(result);
    self.postMessage({ id, result }, transferables);
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
```

### Worker Communication Patterns

```typescript
// Main thread: RPC-style communication
class SplatWorker {
  private worker: Worker;
  private pending = new Map<number, { resolve: Function, reject: Function }>();
  private nextId = 0;

  constructor() {
    // Inline worker via bundler (Vite)
    this.worker = new BundledWorker();
    this.worker.onmessage = this.handleMessage.bind(this);
  }

  private handleMessage(e: MessageEvent) {
    const { id, result, error } = e.data;
    const pending = this.pending.get(id);
    if (pending) {
      this.pending.delete(id);
      if (error) {
        pending.reject(new Error(error));
      } else {
        pending.resolve(result);
      }
    }
  }

  // Generic RPC call
  async call<T>(name: string, ...args: any[]): Promise<T> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });

      // Extract transferable buffers for zero-copy transfer
      const transferables = getArrayBuffers(args);
      this.worker.postMessage({ name, id, args }, transferables);
    });
  }

  // Typed convenience methods
  async sort16(numSplats: number, readback: Uint16Array, ordering: Uint32Array): Promise<number> {
    return this.call('sort16', numSplats, readback, ordering);
  }

  async unpackPly(bytes: Uint8Array, encoding: SplatEncoding): Promise<PackedSplatsData> {
    return this.call('unpackPly', bytes, encoding);
  }
}

```

#### Why Zero-Copy Transfers?

Normally, `postMessage()` **copies** data between threads. For 4MB of splat indices (1M × 4 bytes), this adds ~2ms per transfer.

**Transferable objects** move ownership instead of copying:

```
Standard postMessage (copy):
  Main thread: [4MB buffer] → copy → [4MB buffer] → Worker
  Cost: ~2ms allocation + memcpy

Transferable postMessage (move):
  Main thread: [4MB buffer] → ownership transfer → Worker: [4MB buffer]
  Cost: ~0.01ms (just pointer update)
```

**Caveat**: After transfer, the original ArrayBuffer becomes **detached** (length = 0). The sender can no longer access it. This is intentional—it prevents data races.

```typescript
// Helper: Extract all ArrayBuffers for transfer
function getArrayBuffers(obj: any): ArrayBuffer[] {
  const buffers: ArrayBuffer[] = [];

  function recurse(value: any) {
    if (value?.buffer instanceof ArrayBuffer) {
      buffers.push(value.buffer);
    } else if (Array.isArray(value)) {
      value.forEach(recurse);
    } else if (value && typeof value === 'object') {
      Object.values(value).forEach(recurse);
    }
  }

  recurse(obj);
  return buffers;
}
```

---

## Shader Graph System (Dyno)

### Architecture

The Dyno system is a compile-time shader graph that transforms JavaScript node definitions into GLSL:

```
JavaScript Dyno Graph → Compilation → GLSL Shader Code
     (type-safe)          (code gen)     (runtime)
```

### Type System

```typescript
// GLSL types mapped to TypeScript
type DynoType =
  | 'bool' | 'bvec2' | 'bvec3' | 'bvec4'
  | 'int' | 'ivec2' | 'ivec3' | 'ivec4'
  | 'uint' | 'uvec2' | 'uvec3' | 'uvec4'
  | 'float' | 'vec2' | 'vec3' | 'vec4'
  | 'mat2' | 'mat3' | 'mat4'
  | 'sampler2D' | 'sampler3D' | 'samplerCube'
  | 'Gsplat' | 'TPackedSplats';  // Custom types

// Type-safe value wrapper
interface DynoVal<T extends DynoType> {
  readonly type: T;
  readonly name: string;  // GLSL variable name
}
```

### Node Definition Pattern

```typescript
// Base class for all Dyno operations
abstract class Dyno<InTypes, OutTypes> {
  abstract globals(comp: Compilation): string;
  abstract statements(comp: Compilation, inputs: InTypes, outputs: OutTypes): string;
}

// Example: Addition operation
class Add extends BinaryOp<'float', 'float', 'float'> {
  statements(comp: Compilation, inputs: {a: string, b: string}, outputs: {sum: string}) {
    return `${outputs.sum} = ${inputs.a} + ${inputs.b};`;
  }
}

// Usage
const a = dynoFloat(1.0);
const b = dynoFloat(2.0);
const sum = add(a, b);  // Returns DynoVal<'float'>
```

### Compilation Process

```typescript
class Compilation {
  private globals = new Set<string>();
  private statements: string[] = [];
  private uniforms = new Map<string, IUniform>();
  private counter = 0;

  // Generate unique variable names
  uniqueName(prefix: string): string {
    return `${prefix}_${this.counter++}`;
  }

  // Add global declarations (deduplicated)
  addGlobal(code: string) {
    this.globals.add(code);
  }

  // Add statements (in order)
  addStatement(code: string) {
    this.statements.push(code);
  }

  // Generate final shader
  compile(template: string): string {
    return template
      .replace('{{ GLOBALS }}', [...this.globals].join('\n'))
      .replace('{{ STATEMENTS }}', this.statements.join('\n'));
  }
}
```

### Gsplat Operations

```typescript
// Gsplat struct definition
const gsplatGlobal = `
struct Gsplat {
  vec3 center;
  uint flags;      // Bit 0: active
  vec3 scales;
  int index;
  vec4 quaternion; // xyzw
  vec4 rgba;       // rgb + alpha
};
`;

// Read packed splat from texture
class ReadPackedSplat extends Dyno<{index: 'int'}, {gsplat: 'Gsplat'}> {
  globals() {
    return gsplatGlobal + `
Gsplat unpackSplat(usampler2DArray tex, int index) {
  int layer = index >> 22;
  int y = (index >> 11) & 0x7ff;
  int x = index & 0x7ff;

  uvec4 data = texelFetch(tex, ivec3(x, y, layer), 0);

  Gsplat g;
  g.rgba = vec4(
    float((data.x >> 0) & 0xffu) / 255.0,
    float((data.x >> 8) & 0xffu) / 255.0,
    float((data.x >> 16) & 0xffu) / 255.0,
    float((data.x >> 24) & 0xffu) / 255.0
  );

  g.center = vec3(
    unpackHalf2x16(data.y).x,
    unpackHalf2x16(data.y).y,
    unpackHalf2x16((data.y >> 16) | (data.z << 16)).x
  );

  // ... quaternion and scales unpacking

  return g;
}`;
  }

  statements(comp, inputs, outputs) {
    return `${outputs.gsplat} = unpackSplat(packedTex, ${inputs.index});`;
  }
}
```

### Transform Pipeline

```typescript
// Gsplat transformation in object/world space
function transformGsplat(
  gsplat: DynoVal<'Gsplat'>,
  opts: {
    scale?: DynoVal<'float'>,
    rotate?: DynoVal<'vec4'>,  // quaternion
    translate?: DynoVal<'vec3'>,
    recolor?: DynoVal<'vec4'>
  }
): DynoVal<'Gsplat'> {
  // Each transformation is a Dyno node
  let result = gsplat;

  if (opts.scale) {
    result = dyno({
      inputs: { g: result, s: opts.scale },
      outputs: { out: 'Gsplat' },
      statements: ({ g, s }, { out }) => `
        ${out} = ${g};
        ${out}.center *= ${s};
        ${out}.scales *= ${s};
      `
    });
  }

  if (opts.rotate) {
    result = dyno({
      inputs: { g: result, q: opts.rotate },
      outputs: { out: 'Gsplat' },
      statements: ({ g, q }, { out }) => `
        ${out} = ${g};
        ${out}.center = quatVec(${q}, ${out}.center);
        ${out}.quaternion = quatQuat(${q}, ${out}.quaternion);
      `
    });
  }

  // ... translate, recolor similarly

  return result;
}
```

---

## GPU Memory Management

### Texture-Based Storage

```typescript
// Constants for texture dimensions
const SPLAT_TEX_WIDTH = 2048;   // 11 bits
const SPLAT_TEX_HEIGHT = 2048;  // 11 bits
const SPLAT_TEX_DEPTH = 2048;   // 11 bits (array layers)

// Max splats = 2048 * 2048 * 2048 ≈ 8.5 billion
// Practical limit depends on GPU memory

function getTextureSize(numSplats: number): [number, number, number] {
  const perLayer = SPLAT_TEX_WIDTH * SPLAT_TEX_HEIGHT;
  const layers = Math.ceil(numSplats / perLayer);
  const lastLayerSplats = numSplats % perLayer || perLayer;
  const lastLayerHeight = Math.ceil(lastLayerSplats / SPLAT_TEX_WIDTH);

  return [SPLAT_TEX_WIDTH, lastLayerHeight, layers];
}
```

### DataArrayTexture Upload

```typescript
class PackedSplats {
  private texture: THREE.DataArrayTexture | null = null;

  maybeUpdateSource() {
    if (!this.needsUpdate) return;

    const [width, height, depth] = getTextureSize(this.numSplats);

    // Create or resize texture
    if (!this.texture || this.texture.image.depth < depth) {
      this.texture?.dispose();

      this.texture = new THREE.DataArrayTexture(
        this.packedArray,
        width,
        height,
        depth
      );
      this.texture.format = THREE.RGBAIntegerFormat;
      this.texture.type = THREE.UnsignedIntType;
      this.texture.internalFormat = 'RGBA32UI';
      this.texture.minFilter = THREE.NearestFilter;
      this.texture.magFilter = THREE.NearestFilter;
      this.texture.needsUpdate = true;
    } else {
      // Partial update
      this.texture.image.data = this.packedArray;
      this.texture.needsUpdate = true;
    }

    this.needsUpdate = false;
  }
}
```

### Exponential Buffer Growth

```typescript
// PackedSplats.ensureSplats()
ensureSplats(minCapacity: number) {
  if (this.packedArray.length >= minCapacity * 4) return;

  // Grow by 1.5x to amortize allocations
  let newCapacity = Math.max(this.maxSplats, 64);
  while (newCapacity < minCapacity) {
    newCapacity = Math.ceil(newCapacity * 1.5);
  }

  // Allocate new array and copy
  const newArray = new Uint32Array(newCapacity * 4);
  newArray.set(this.packedArray);
  this.packedArray = newArray;
  this.maxSplats = newCapacity;
}
```

### GPU Readback

```typescript
class Readback {
  private target: THREE.WebGLArrayRenderTarget;

  async readback(renderer: THREE.WebGLRenderer): Promise<Uint8Array> {
    const gl = renderer.getContext();

    // Only portable format for WebGL2
    const format = gl.RGBA;
    const type = gl.UNSIGNED_BYTE;

    const buffer = new Uint8Array(this.width * this.height * 4);

    // Async readback (non-blocking)
    await renderer.readRenderTargetPixelsAsync(
      this.target,
      0, 0,
      this.width, this.height,
      buffer
    );

    return buffer;
  }
}
```

---

## Performance Optimizations

### Why Single Draw Call Matters (Instanced Rendering)

Every WebGL draw call has CPU overhead:

```
Per draw call cost:
  - State validation: ~5μs
  - Uniform uploads: ~2μs
  - Driver overhead: ~10μs
  Total: ~17μs per draw call

1M splats with individual draws: 1,000,000 × 17μs = 17 seconds ❌
1M splats with instancing:       1 × 17μs = 17μs ✓
```

**Instanced rendering** draws the same geometry (a quad) millions of times with one API call. The GPU parallelizes across instances automatically.

```typescript
// Without instancing (SLOW):
for (let i = 0; i < numSplats; i++) {
  gl.uniform3fv(centerLoc, splats[i].center);
  gl.drawArrays(gl.TRIANGLES, 0, 6);  // 6 vertices per quad
}

// With instancing (FAST):
gl.vertexAttribDivisor(orderingLoc, 1);  // Advance per instance
gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, numSplats);  // One call!
```

**Key insight**: The vertex shader receives `gl_InstanceID` (0 to numSplats-1), uses it to fetch splat data from the texture, and positions the quad accordingly.

---

### Performance Patterns Summary

The table below summarizes all performance optimization patterns used in SparkJS and their quantified benefits:

| Pattern | Problem Solved | Typical Speedup |
|---------|---------------|-----------------|
| **Texture Storage** | Sorting requires copying all data | 4× less bandwidth |
| **Radix Sort** | O(n log n) comparison sorts | O(n) linear time |
| **WASM Sorting** | JavaScript overhead in tight loops | 3.7× faster |
| **Web Worker** | Main thread blocking | 60fps vs 14fps |
| **Zero-Copy Transfer** | memcpy overhead | ~2ms saved/frame |
| **Instanced Rendering** | Per-splat draw call overhead | 1,000,000× less calls |
| **Object Pooling** | GC pause jank | Zero GC in hot path |
| **Version Caching** | Regenerating unchanged data | 10× for partial updates |
| **Persistent Allocations** | Worker GC delays | Predictable latency |
| **Threshold Updates** | Sorting every frame | ~10× less sorting |
| **Static Buffers** | Allocation in hot loops | 0 bytes/frame garbage |
| **Correspondence Detection** | Re-sorting unchanged layouts | 3600× when unchanged |
| **Scissor Updates** | Full texture re-render | 11× for incremental |
| **WeakMap Caching** | Memory leaks from cache | Auto-cleanup on GC |
| **Backpressure** | Worker queue buildup | Bounded memory |
| **Compiled Parsers** | Interpreted schema loops | 60M ops eliminated |

---

### Why Object Pooling Reduces GC Pauses

JavaScript's garbage collector (GC) runs unpredictably and can cause frame drops:

```
Without pooling:
  Frame 1: [Render][Alloc][Render][Alloc][Render]...
  Frame 50: [Render][===GC PAUSE 16ms===][Render] ← Frame drop!

Problem: Creating objects generates garbage. GC pauses are:
  - Unpredictable (might hit during animation)
  - Proportional to allocation rate
  - Can block main thread 5-20ms

With pooling:
  Startup: [Alloc][Alloc][Alloc]  ← All allocations upfront
  Frame 1-∞: [Render][Reuse][Render][Reuse]...

Result: Zero allocations in hot path = zero GC pauses
```

**Object pooling** pre-allocates objects and reuses them instead of creating new ones. SparkJS applies this pattern to:
- **Accumulators**: Expensive GPU buffer containers
- **Sort buffers**: Large typed arrays for radix sort
- **Temporary objects**: Vectors, quaternions, colors

---

### 1. Accumulator Pooling

```typescript
class SparkRenderer {
  private freeAccumulators: SplatAccumulator[] = [];
  private MAX_ACCUMULATORS = 5;

  maybeAllocAccumulator(): SplatAccumulator | null {
    // Reuse from pool if available
    if (this.freeAccumulators.length > 0) {
      const acc = this.freeAccumulators.pop()!;
      acc.refCount = 1;
      return acc;
    }

    // Check limit
    const total = this.activeAccumulators.length + this.freeAccumulators.length;
    if (total >= this.MAX_ACCUMULATORS) {
      return null;  // Backpressure
    }

    // Create new
    return new SplatAccumulator();
  }

  releaseAccumulator(acc: SplatAccumulator) {
    acc.refCount--;
    if (acc.refCount === 0) {
      this.freeAccumulators.push(acc);
    }
  }
}
```

### Why Version-Based Caching Prevents Redundant Work

Regenerating all splats every frame is wasteful when most data hasn't changed:

```
Scenario: 10 generators, only 1 changes per frame

Without versioning:
  Frame N: Regenerate all 10 → process 1M splats → 45ms

With versioning:
  Frame N: Check versions → regenerate 1 → process 100K splats → 4.5ms

Speedup: 10× for this common case
```

**Version numbers** act as lightweight dirty flags. Instead of deep-comparing data, SparkJS increments a counter when data changes and compares integers:

| Comparison Type | Cost | Reliability |
|----------------|------|-------------|
| Deep equality check | O(n) data size | 100% accurate |
| **Version number** | **O(1) integer compare** | **100% accurate** |
| Timestamp check | O(1) | Can miss rapid updates |

**Key insight**: The generator (source of truth) owns the version. Consumers cache the version they last processed. Mismatch = need to regenerate.

---

### 2. Version-Based Change Detection

```typescript
// Only regenerate changed generators
class SplatGenerator {
  version = 0;

  set needsUpdate(value: boolean) {
    if (value) this.version++;
  }
}

// In accumulator
const cached = this.cache.get(generator.uuid);
if (cached?.version === generator.version) {
  // Skip regeneration
  continue;
}
```

### 3. Generator Sorting for Correspondence

```typescript
// Sort generators by version delta to preserve unchanging splats at front
// This enables partial sort order reuse
function sortByVersionDelta(generators: SplatGenerator[], cache: Map) {
  return generators.sort((a, b) => {
    const deltaA = a.version - (cache.get(a.uuid)?.version ?? 0);
    const deltaB = b.version - (cache.get(b.uuid)?.version ?? 0);
    return deltaA - deltaB;  // Unchanged first
  });
}
```

### Why Persistent Allocations in Workers Matter

Even in web workers, GC can cause jank. Worker GC pauses delay message responses:

```
Without persistent allocations:
  Worker receives sort request
  → Allocate depthArray (16MB) ← GC pressure
  → Allocate buckets (32KB) ← GC pressure
  → Sort
  → Return
  → [Worker GC pause] ← Main thread waits for response!

With persistent allocations:
  Startup: Allocate once with headroom
  Frame N: Reuse existing arrays (zero allocation)
  Result: Predictable response times
```

**1.5× headroom**: Pre-allocating 50% extra capacity avoids reallocation when splat count grows slightly between frames.

---

### 4. Worker Persistent Allocations

```typescript
// worker.ts - Avoid GC during hot path
let depthArray16: Uint16Array | null = null;
let bucket16: Uint32Array | null = null;
let scratchSplats: Uint32Array | null = null;

function ensureAllocations(numSplats: number) {
  if (!depthArray16 || depthArray16.length < numSplats) {
    // Allocate with headroom
    const size = Math.ceil(numSplats * 1.5);
    depthArray16 = new Uint16Array(size);
    bucket16 = new Uint32Array(0x7c01);  // Float16 range
    scratchSplats = new Uint32Array(size);
  }
}
```

### Why Threshold-Based Updates Avoid Unnecessary Sorts

Sorting is expensive. But imperceptibly small camera movements don't change visible order:

```
Camera movement vs. visual impact:

  Movement: 0.001 units → Sort difference: ~0 splats change order
  Movement: 0.01 units  → Sort difference: ~0.1% splats change order
  Movement: 0.1 units   → Sort difference: ~5% splats change order
  Movement: 1.0 units   → Sort difference: ~30% splats change order

Sorting cost: 12ms for 1M splats (WASM)
Frame budget: 16.6ms for 60fps

Without thresholds:
  Every frame: Sort → 12ms → Only 4.6ms left for everything else!

With thresholds:
  Frames 1-9: No sort (camera moved < 0.01) → Full 16.6ms budget
  Frame 10: Sort (camera moved > 0.01) → 12ms
  Average: ~1.2ms per frame for sorting
```

**Position threshold** (sortDistance): Minimum camera translation before re-sorting
**Orientation threshold** (sortCoorient): Minimum rotation (as cos of angle) before re-sorting

The quaternion dot product gives `cos(angle/2)`, so `0.999` means ~2.5° rotation threshold.

---

### 5. Threshold-Based Auto-Update

```typescript
class SparkViewpoint {
  sortDistance = 0.01;    // Movement threshold
  sortCoorient = 0.999;   // Orientation threshold (cos of angle)

  shouldUpdate(camera: THREE.Camera): boolean {
    const currentPos = camera.position;
    const currentQuat = camera.quaternion;

    // Check position change
    if (currentPos.distanceTo(this.lastPos) > this.sortDistance) {
      return true;
    }

    // Check orientation change (dot product)
    if (this.lastQuat.dot(currentQuat) < this.sortCoorient) {
      return true;
    }

    return false;
  }
}
```

### 6. Stochastic Rendering Mode

For very high splat counts, SparkJS can use stochastic depth testing:

```typescript
// When stochastic=true:
// - Disable alpha blending
// - Enable depth test
// - Random splat ordering is acceptable
// - Much faster but lower quality

this.material.transparent = !this.stochastic;
this.material.depthWrite = this.stochastic;
```

### Why Compiled Parsers Are Faster Than Interpreted Loops

When parsing binary data, schema interpretation has overhead per field:

```
Interpreted parsing (per splat):
  for (field in schema) {           // Loop overhead
    type = schema.getType(field);   // Property lookup
    size = typeToSize[type];        // Hash lookup
    getter = typeToGetter[type];    // Hash lookup
    value = dataView[getter](pos);  // Indirect call
    result[field] = value;          // Dynamic property
    pos += size;
  }
  Overhead: ~6 operations per field × 10 fields × 1M splats = 60M extra ops

Compiled parsing (per splat):
  // Generated code with inlined constants:
  result.x = dataView.getFloat32(pos, true); pos += 4;
  result.y = dataView.getFloat32(pos, true); pos += 4;
  result.z = dataView.getFloat32(pos, true); pos += 4;
  // ... (no loops, no lookups, direct calls)
  Overhead: Near-zero (JIT-optimized straight-line code)
```

**Code generation** converts runtime schema interpretation into compile-time optimized code. The V8/SpiderMonkey JIT can then:
- Inline getters
- Eliminate bounds checks
- Use SIMD where applicable

---

### 7. Compiled Parser Functions

```typescript
// Generate optimized parser from PLY schema
function compileParser(elements: PlyElement[]): Function {
  let code = 'return function parse(dataView, offset, items) {\n';

  for (const elem of elements) {
    for (const [name, prop] of Object.entries(elem.properties)) {
      const getter = `dataView.get${typeToGetter[prop.type]}`;
      code += `  items.${name} = ${getter}(offset, true);\n`;
      code += `  offset += ${typeToSize[prop.type]};\n`;
    }
  }

  code += '  return offset;\n}';

  // Compile once, execute many times
  return new Function(code)();
}
```

### 8. Object Pooling with FreeList

```typescript
// Generic object pool with heterogeneous item validation
class FreeList<T, Args> {
  private items: T[] = [];
  private allocate: (args: Args) => T;
  private valid: (item: T, args: Args) => boolean;
  private dispose?: (item: T) => void;

  alloc(args: Args): T {
    // Try to reuse from pool
    while (this.items.length > 0) {
      const item = this.items.pop()!;
      // Validate item matches requirements (e.g., buffer size)
      if (this.valid(item, args)) {
        return item;
      }
      // Dispose if invalid (wrong size, etc.)
      this.dispose?.(item);
    }
    // Allocate fresh if pool empty
    return this.allocate(args);
  }

  free(item: T) {
    this.items.push(item);
  }
}

// Usage: Pool buffers of varying sizes
const bufferPool = new FreeList<Uint32Array, number>(
  (size) => new Uint32Array(size),
  (buffer, size) => buffer.length >= size,  // Reuse if big enough
  (buffer) => { /* allow GC */ }
);
```

### Why Static Buffers Eliminate Hot Path Allocations

In hot paths (called millions of times), even small allocations add up:

```
Unpacking 1M splats, creating temporary objects each time:

  Per splat: new Vector3() → 24 bytes + GC tracking overhead
  1M splats: 24MB garbage per frame!
  At 60fps: 1.4GB/sec garbage generation → constant GC

With static buffers:
  Startup: packedCenter = new Vector3() (once)
  Per splat: packedCenter.set(x, y, z) → 0 bytes allocated
  1M splats: 0 bytes garbage
  Result: No GC pressure from unpacking
```

**Caveat**: Caller must copy values if they need to persist:
```typescript
// WRONG: Stores reference to static buffer (will be overwritten!)
results.push(unpackSplat(array, i));

// RIGHT: Copy values before storing
const splat = unpackSplat(array, i);
results.push(splat.center.clone());
```

---

### 9. Shared Static Buffers

```typescript
// Avoid allocations in hot paths by reusing static buffers
const f32buffer = new Float32Array(1);
const u32buffer = new Uint32Array(f32buffer.buffer);  // Shared backing

// Zero-copy float↔uint bit reinterpretation
function floatBitsToUint(f: number): number {
  f32buffer[0] = f;
  return u32buffer[0];
}

function uintBitsToFloat(u: number): number {
  u32buffer[0] = u;
  return f32buffer[0];
}

// Reusable objects for unpacking (never allocate in loop)
const packedCenter = new THREE.Vector3();
const packedScales = new THREE.Vector3();
const packedQuaternion = new THREE.Quaternion();
const packedColor = new THREE.Color();
const tempAxis = new THREE.Vector3();

function unpackSplat(array: Uint32Array, index: number): SplatData {
  // Mutate and return static objects - caller must copy if needed
  // ...
  return { center: packedCenter, scales: packedScales, ... };
}
```

### Why Correspondence Detection Enables Sort Reuse

If the same generators produce splats in the same order, the sort result can be reused:

```
Frame N:   [Generator A: 0-999][Generator B: 1000-1999][Generator C: 2000-2999]
Frame N+1: [Generator A: 0-999][Generator B: 1000-1999][Generator C: 2000-2999]
                    ↑                    ↑                    ↑
               Same layout! Sort indices from Frame N are still valid.

Reuse condition:
  - Same generators in same order
  - Same base indices (where each generator's splats start)
  - Same counts (number of splats per generator)

Cost comparison:
  Re-sort 3M splats: ~36ms
  Check correspondence: ~0.01ms (compare 3 entries)
  Speedup when layout unchanged: 3600×
```

**When correspondence is detected**: Copy previous sort indices directly, skip sorting entirely.

---

### 10. Correspondence Detection for Sort Reuse

```typescript
// Detect when sort order can be reused between frames
class SplatAccumulator {
  hasCorrespondence(other: SplatAccumulator): boolean {
    if (this.mapping.length !== other.mapping.length) return false;

    return this.mapping.every((entry, i) => {
      const otherEntry = other.mapping[i];
      return (
        entry.generator === otherEntry.generator &&
        entry.base === otherEntry.base &&
        entry.count === otherEntry.count
      );
    });
  }
}

// In renderer: skip sorting if layout unchanged
if (newAccumulator.hasCorrespondence(oldAccumulator)) {
  // Reuse previous sort ordering
  newAccumulator.ordering = oldAccumulator.ordering;
}
```

### Why Scissor-Based Partial Updates Save GPU Time

When splat data changes incrementally, only the new portion needs updating:

```
Scenario: Adding 100K splats to existing 1M

Full update:
  Render all 1.1M splats to texture → ~11ms GPU time

Scissor update:
  Render only new 100K splats to texture → ~1ms GPU time
  Speedup: 11×

How it works:
  Texture organized in layers (rows)
  lastCount = 1,000,000 → starts at layer 244
  newCount = 1,100,000 → ends at layer 268
  Scissor rect: layers 244-268 only
```

**GPU fragment shaders** only execute for pixels within the scissor rect, saving fillrate for unchanged regions.

---

### 11. Scissor-Based Partial Rendering

```typescript
// Only render modified texture regions
class PackedSplats {
  generate(renderer: THREE.WebGLRenderer, count: number) {
    // Calculate which layers need updating
    const startLayer = Math.floor(this.lastCount / SPLAT_TEX_LAYER_SIZE);
    const endLayer = Math.ceil(count / SPLAT_TEX_LAYER_SIZE);

    for (let layer = startLayer; layer < endLayer; layer++) {
      const layerStart = layer * SPLAT_TEX_HEIGHT;
      const layerEnd = Math.min((layer + 1) * SPLAT_TEX_HEIGHT, totalHeight);

      // Scissor to modified rows only
      renderer.setScissor(0, layerStart, SPLAT_TEX_WIDTH, layerEnd - layerStart);
      renderer.setScissorTest(true);

      renderer.setRenderTarget(this.target, layer);
      renderer.render(this.scene, this.camera);
    }

    this.lastCount = count;
  }
}
```

### Why WeakMap Caching Prevents Memory Leaks

Regular `Map` prevents garbage collection even when keys are no longer used:

```
Problem: Caching materials by program

With Map:
  const cache = new Map<DynoProgram, Material>();
  cache.set(program, material);
  program = null;  // We're done with it
  // BUT: Map still holds strong reference → program never GC'd!
  // Memory leak: cache grows forever

With WeakMap:
  const cache = new WeakMap<DynoProgram, Material>();
  cache.set(program, material);
  program = null;  // We're done with it
  // WeakMap holds weak reference → program can be GC'd
  // When program GC'd → cache entry automatically removed!
```

**WeakMap behavior**:
- Keys must be objects (not primitives)
- Keys are held weakly: don't prevent GC
- When key is GC'd, entry is automatically removed
- Cannot iterate (no `.size`, `.keys()`, etc.)

**Perfect for**: Caching derived data (materials, textures) keyed by source objects (programs, generators).

---

### 12. Material WeakMap Caching

```typescript
// Automatic GC of unused materials via WeakMap
const programMaterialCache = new WeakMap<DynoProgram, THREE.RawShaderMaterial>();

class DynoProgram {
  getMaterial(): THREE.RawShaderMaterial {
    let material = programMaterialCache.get(this);
    if (!material) {
      material = new THREE.RawShaderMaterial({
        vertexShader: this.compiledVertex,
        fragmentShader: this.compiledFragment,
        uniforms: this.uniforms,
      });
      programMaterialCache.set(this, material);
    }
    return material;
  }
}

// When DynoProgram is GC'd, material is automatically removed from cache
```

### Why Backpressure Prevents Worker Queue Buildup

Without backpressure, rapid camera movements can overwhelm the sort worker:

```
Problem: Camera moving fast, sort requests every frame

Without backpressure:
  Frame 1: Send sort request (50ms to complete)
  Frame 2: Send sort request → queued
  Frame 3: Send sort request → queued
  ...
  Frame 10: Worker still processing Frame 1!
  Result: 9 stale sort requests queued, memory grows, results are outdated

With backpressure:
  Frame 1: Send sort request, sorting=true
  Frame 2: sorting=true → save as pending (replace previous pending)
  Frame 3: sorting=true → save as pending (replace)
  ...
  Frame 10: Worker finishes Frame 1 → send pending request
  Result: Only 2 requests ever active: current + pending
```

**Backpressure pattern**: Track in-flight state. Queue at most ONE pending request. Newest request overwrites previous pending.

---

### 13. Backpressure in Sort Pipeline

```typescript
class SparkViewpoint {
  private sorting = false;
  private pending: SortRequest | null = null;

  requestSort(accumulator: SplatAccumulator, viewMatrix: THREE.Matrix4) {
    // Queue request (overwrites previous pending)
    this.pending = { accumulator, viewMatrix };

    // Process if not already sorting
    this.driveSort();
  }

  private async driveSort() {
    // Backpressure: only one sort at a time
    if (this.sorting || !this.pending) return;

    this.sorting = true;
    const request = this.pending;
    this.pending = null;

    try {
      // GPU readback → Worker sort → Display update
      await this.sortUpdate(request);
    } finally {
      this.sorting = false;
      // Process any pending request that arrived during sort
      if (this.pending) this.driveSort();
    }
  }
}
```

### 14. Deferred Update Scheduling

```typescript
class SparkRenderer {
  private pendingUpdate: { timeoutId: number | null } = { timeoutId: null };

  update() {
    if (this.preUpdate) {
      // Immediate update (blocking)
      this.updateInternal();
    } else {
      // Deferred update (non-blocking)
      if (this.pendingUpdate.timeoutId !== null) return;

      this.pendingUpdate.timeoutId = setTimeout(() => {
        this.pendingUpdate.timeoutId = null;
        this.updateInternal();

        // Flush GPU commands to prevent batching delays
        const gl = this.renderer.getContext();
        gl.flush();
      }, 1);  // 1ms delay allows frame to complete
    }
  }
}
```

### 15. LRU Cache for Network Resources

```typescript
class DataCache {
  private items: Array<{ key: string; data: unknown }> = [];
  private maxItems = 100;

  async get(url: string): Promise<unknown> {
    // Check cache (move to end if found = LRU)
    const index = this.items.findIndex(item => item.key === url);
    if (index >= 0) {
      const item = this.items.splice(index, 1)[0];
      this.items.push(item);  // Most recently used
      return item.data;
    }

    // Fetch and cache
    const data = await fetch(url).then(r => r.arrayBuffer());
    this.items.push({ key: url, data });

    // Evict oldest (LRU)
    while (this.items.length > this.maxItems) {
      this.items.shift();
    }

    return data;
  }
}
```

### 16. WebXR Multi-Eye Support

```typescript
class SparkRenderer {
  // When rendering for WebXR, average camera matrices across eyes
  onBeforeRender(renderer: THREE.WebGLRenderer, scene: THREE.Scene,
                 camera: THREE.Camera, ...) {
    if (renderer.xr.enabled && renderer.xr.isPresenting) {
      const cameras = renderer.xr.getCamera().cameras;
      if (cameras.length > 0) {
        // Compute average view position across all eyes
        this.sortCenter.set(0, 0, 0);
        for (const cam of cameras) {
          this.sortCenter.add(cam.position);
        }
        this.sortCenter.divideScalar(cameras.length);

        // Use averaged position for sorting
        // This prevents popping when switching eyes
      }
    }
  }
}
```

### 17. Supersampling Support

```typescript
class SparkViewpoint {
  superXY = 1;  // 1-4× supersampling factor

  // Maximum supported target: 8192×8192 pixels
  createRenderTarget() {
    const width = Math.min(this.width * this.superXY, 8192);
    const height = Math.min(this.height * this.superXY, 8192);

    return new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType
    });
  }

  // Downsample via simple CPU averaging
  readTarget(): Uint8Array {
    const raw = this.readPixels();  // Full resolution
    if (this.superXY === 1) return raw;

    // Average superXY × superXY blocks
    const result = new Uint8Array(this.width * this.height * 4);
    const scale = this.superXY;
    const scale2 = scale * scale;

    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        let r = 0, g = 0, b = 0, a = 0;
        for (let dy = 0; dy < scale; dy++) {
          for (let dx = 0; dx < scale; dx++) {
            const idx = ((y * scale + dy) * this.width * scale + (x * scale + dx)) * 4;
            r += raw[idx + 0];
            g += raw[idx + 1];
            b += raw[idx + 2];
            a += raw[idx + 3];
          }
        }
        const outIdx = (y * this.width + x) * 4;
        result[outIdx + 0] = r / scale2;
        result[outIdx + 1] = g / scale2;
        result[outIdx + 2] = b / scale2;
        result[outIdx + 3] = a / scale2;
      }
    }
    return result;
  }
}
```

---

## Quality Parameters Reference

Complete list of configurable quality parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Rendering Quality** |
| `falloff` | float | 1.0 | Gaussian kernel decay (0=flat, 1=full) |
| `maxStdDev` | float | 3.0 | Max standard deviations to render |
| `minPixelRadius` | float | 0.5 | Minimum projected splat size |
| `maxPixelRadius` | float | 2048 | Maximum projected splat size |
| `minAlpha` | float | 1/255 | Alpha threshold for culling |
| **Blur & Anti-Aliasing** |
| `preBlurAmount` | float | 0.0 | Pre-projection blur (covariance scaling) |
| `blurAmount` | float | 0.0 | Post-projection blur (ellipse scaling) |
| `focalAdjust` | float | 1.0 | Focal length multiplier |
| **Depth of Field** |
| `focalDistance` | float | 10.0 | Focus plane distance |
| `apertureAngle` | float | 0.0 | Aperture half-angle (radians) |
| **Sorting** |
| `sort32` | bool | true | Use float32 precision (vs float16) |
| `sort360` | bool | false | Disable behind-camera culling |
| `sortDistance` | float | 0.01 | Movement threshold for re-sort |
| `sortCoorient` | float | 0.999 | Orientation threshold (dot product) |
| `sortDepthBias` | float | 1.0 | Depth offset for sort metric |
| **Rendering Mode** |
| `stochastic` | bool | false | Use stochastic depth testing |
| `enable2DGS` | bool | false | 2D Gaussian splatting mode |
| `encodeLinear` | bool | false | Output linear color space |
| **Supersampling** |
| `superXY` | int | 1 | Supersampling factor (1-4) |
| `doubleBuffer` | bool | false | Enable render target swapping |

---

## File Format Support

SparkJS supports 6 different file formats, each converted to the internal 16-byte PackedSplats format:

| Format | Bytes/Splat | Magic/Detection | Notes |
|--------|-------------|-----------------|-------|
| PLY | Variable | `ply\n` header | Standard 3DGS format |
| SPZ | Variable | `0x5053474e` | Niantic compressed |
| SPLAT | 32 bytes | Extension only | Raw floats |
| KSPLAT | 32-44 bytes | Extension only | 3 compression levels |
| PCSOGS | External files | `meta.json` | Separate data files |
| PCSOGSZIP | ZIP archive | `0x04034b50` | Bundled PCSOGS |

### PLY Format

```typescript
// Supported PLY vertex properties for Gsplats
const GSPLAT_PROPERTIES = {
  // Position (required)
  x: 'float', y: 'float', z: 'float',

  // Scale (optional, defaults to 1)
  scale_0: 'float', scale_1: 'float', scale_2: 'float',

  // Rotation (optional, defaults to identity)
  rot_0: 'float', rot_1: 'float', rot_2: 'float', rot_3: 'float',

  // Color (optional)
  red: 'uchar', green: 'uchar', blue: 'uchar',
  f_dc_0: 'float', f_dc_1: 'float', f_dc_2: 'float',  // SH0

  // Opacity (optional)
  opacity: 'float',

  // Spherical Harmonics (optional)
  f_rest_0: 'float', /* ... */ f_rest_44: 'float'  // SH1-3
};
```

### SPLAT Format (AntiSplat)

The `.splat` format uses 32 bytes per splat with raw float values:

```typescript
// 32 bytes per splat layout:
// Bytes 0-11:  Position XYZ (3 × float32)
// Bytes 12-23: Scale XYZ (3 × float32)
// Bytes 24-27: Color RGBA (4 × uint8)
// Bytes 28-31: Quaternion XYZW (4 × int8, bias-encoded)

function decodeAntiSplat(data: Uint8Array, callback: SplatCallback) {
  const numSplats = data.byteLength / 32;
  const view = new DataView(data.buffer);

  for (let i = 0; i < numSplats; i++) {
    const offset = i * 32;

    const x = view.getFloat32(offset + 0, true);
    const y = view.getFloat32(offset + 4, true);
    const z = view.getFloat32(offset + 8, true);

    const scaleX = view.getFloat32(offset + 12, true);
    const scaleY = view.getFloat32(offset + 16, true);
    const scaleZ = view.getFloat32(offset + 20, true);

    const r = data[offset + 24] / 255;
    const g = data[offset + 25] / 255;
    const b = data[offset + 26] / 255;
    const opacity = data[offset + 27] / 255;

    // Quaternion: bias-encoded (byte - 128) / 128
    const qx = (data[offset + 28] - 128) / 128;
    const qy = (data[offset + 29] - 128) / 128;
    const qz = (data[offset + 30] - 128) / 128;
    const qw = (data[offset + 31] - 128) / 128;

    callback(x, y, z, scaleX, scaleY, scaleZ, r, g, b, opacity, qx, qy, qz, qw);
  }
}
```

### SPZ Format (Niantic Compressed)

```typescript
interface SpzHeader {
  magic: 0x5053474e;  // "NGSP"
  version: 1 | 2 | 3;
  numSplats: number;
  shDegree: 0 | 1 | 2 | 3;
  fractionalBits: number;  // For fixed-point positions (default 12)
  flags: number;           // Bit 0: antialiased
}

// Version differences:
// v1: float16 positions, 24-bit quaternion
// v2: 24-bit fixed-point positions
// v3: "Smallest three" quaternion compression (32-bit)
```

### KSPLAT Format

Three compression levels with bucket-based spatial organization:

```typescript
// Level 0: Uncompressed (44 bytes/splat)
// - Centers: 12 bytes (3 × float32)
// - Scales: 12 bytes (3 × float32)
// - Rotation: 16 bytes (4 × float32)
// - Color: 4 bytes (RGBA8)

// Level 1: Half-precision (26 bytes/splat)
// - Centers: 6 bytes (3 × float16)
// - Scales: 6 bytes (3 × float16)
// - Rotation: 8 bytes (4 × float16)
// - Color: 4 bytes (RGBA8)
// - SH: 2 bytes per component

// Level 2: Maximum compression
// - Same as Level 1 but SH: 1 byte per component
```

### Format Detection

```typescript
function detectFormat(bytes: Uint8Array, fileName?: string): FileType {
  const view = new DataView(bytes.buffer);

  // PLY: "ply\n" (0x706c79)
  if (bytes[0] === 0x70 && bytes[1] === 0x6c && bytes[2] === 0x79) {
    return 'ply';
  }

  // SPZ: "NGSP" (0x5053474e)
  if (view.getUint32(0, true) === 0x5053474e) {
    return 'spz';
  }

  // PKZip: (0x04034b50)
  if (view.getUint32(0, true) === 0x04034b50) {
    return 'pcsogszip';
  }

  // Gzip: 0x1f8b
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const decompressed = gunzip(bytes);
    return detectFormat(decompressed, fileName);
  }

  // Fall back to extension
  if (fileName?.endsWith('.splat')) return 'splat';
  if (fileName?.endsWith('.ksplat')) return 'ksplat';
  if (fileName?.endsWith('.sog')) return 'pcsogs';

  throw new Error('Unknown file format');
}
```

---

## Advanced Features

### Skeletal Animation (Dual Quaternion Skinning)

SparkJS uses **dual quaternion** skinning for volume-preserving bone deformations. This avoids the "candy wrapper" and joint collapse artifacts of linear blend skinning.

```typescript
class SplatSkinning {
  private boneTexture: THREE.DataTexture;  // 16 floats per bone
  private skinTexture: THREE.DataTexture;  // 4 uint16 per splat
  private maxBones = 256;  // Maximum supported bones

  // Bone texture layout (16 floats per bone):
  // [0-3]:   Rest quaternion (x, y, z, w)
  // [4-6]:   Rest position (x, y, z)
  // [7]:     Padding
  // [8-11]:  Relative quaternion (current vs rest)
  // [12-15]: Dual quaternion part

  setRestQuatPos(boneIdx: number, quat: THREE.Quaternion, pos: THREE.Vector3) {
    const offset = boneIdx * 16;
    // Store rest pose
    this.boneData[offset + 0] = quat.x;
    this.boneData[offset + 1] = quat.y;
    this.boneData[offset + 2] = quat.z;
    this.boneData[offset + 3] = quat.w;
    this.boneData[offset + 4] = pos.x;
    this.boneData[offset + 5] = pos.y;
    this.boneData[offset + 6] = pos.z;
  }

  setBoneQuatPos(boneIdx: number, quat: THREE.Quaternion, pos: THREE.Vector3) {
    const offset = boneIdx * 16;

    // Read rest pose
    const restQuat = new THREE.Quaternion(
      this.boneData[offset + 0],
      this.boneData[offset + 1],
      this.boneData[offset + 2],
      this.boneData[offset + 3]
    );
    const restPos = new THREE.Vector3(
      this.boneData[offset + 4],
      this.boneData[offset + 5],
      this.boneData[offset + 6]
    );

    // Compute relative transformation:
    // relQuat = inverse(restQuat) × currentQuat
    const invRestQuat = restQuat.clone().invert();
    const relQuat = invRestQuat.clone().multiply(quat);

    // relPos = (currentPos - restPos) rotated by inverse(restQuat)
    const relPos = pos.clone().sub(restPos).applyQuaternion(invRestQuat);

    // Store relative quaternion
    this.boneData[offset + 8] = relQuat.x;
    this.boneData[offset + 9] = relQuat.y;
    this.boneData[offset + 10] = relQuat.z;
    this.boneData[offset + 11] = relQuat.w;

    // Compute and store dual quaternion part:
    // dualQuat = 0.5 × translation_as_pure_quat × relQuat
    const dq = new THREE.Quaternion(
      0.5 * (relPos.x * relQuat.w + relPos.y * relQuat.z - relPos.z * relQuat.y),
      0.5 * (-relPos.x * relQuat.z + relPos.y * relQuat.w + relPos.z * relQuat.x),
      0.5 * (relPos.x * relQuat.y - relPos.y * relQuat.x + relPos.z * relQuat.w),
      0.5 * (-relPos.x * relQuat.x - relPos.y * relQuat.y - relPos.z * relQuat.z)
    );
    this.boneData[offset + 12] = dq.x;
    this.boneData[offset + 13] = dq.y;
    this.boneData[offset + 14] = dq.z;
    this.boneData[offset + 15] = dq.w;

    this.boneTexture.needsUpdate = true;
  }

  // Assign up to 4 bones per splat with weights
  setSplatBones(splatIdx: number, bones: number[], weights: number[]) {
    // Pack: bone index (high byte) + weight (low byte, 0-255)
    const offset = splatIdx * 4;
    for (let i = 0; i < 4; i++) {
      this.skinData[offset + i] =
        ((bones[i] || 0) << 8) | (Math.round((weights[i] || 0) * 255));
    }
    this.skinTexture.needsUpdate = true;
  }
}

// Shader: Blend dual quaternions with sign correction
// to prevent "path ambiguity" (choosing wrong rotation direction)
vec4 blendDualQuaternions(vec4 dq0, vec4 dq1, vec4 dq2, vec4 dq3,
                          float w0, float w1, float w2, float w3) {
  // Ensure all quaternions are on same hemisphere
  if (dot(dq0, dq1) < 0.0) { dq1 = -dq1; w1 = -w1; }
  if (dot(dq0, dq2) < 0.0) { dq2 = -dq2; w2 = -w2; }
  if (dot(dq0, dq3) < 0.0) { dq3 = -dq3; w3 = -w3; }

  // Weighted sum
  vec4 result = w0 * dq0 + w1 * dq1 + w2 * dq2 + w3 * dq3;

  // Normalize
  return result / length(result);
}
```

**Key advantage**: Dual quaternion formulation assumes mass/volume is conserved through transformations, avoiding linear blend skinning artifacts like joint collapse or volume loss.

### SDF-Based Editing

```typescript
// SplatEdit: Apply SDF-based modifications
class SplatEdit extends THREE.Object3D {
  type: 'SPHERE' | 'BOX' | 'ELLIPSOID' | 'CYLINDER' | 'CAPSULE';
  rgbaBlendMode: 'MULTIPLY' | 'SET_RGB' | 'ADD_RGBA';

  // Soft boundary
  sdfSmooth = 0.1;  // Blend radius
  softEdge = 0.05;  // Falloff

  // Effect
  opacity = 1.0;
  color = new THREE.Color(1, 1, 1);
  displace = new THREE.Vector3(0, 0, 0);
}

// Shader: Commutative softmax blending
float evaluateSdfArray(vec3 pos, SdfArray sdfs) {
  float sumExp = 0.0;
  float sumDist = 0.0;

  for (int i = 0; i < sdfs.count; i++) {
    float d = evaluateSdf(pos, sdfs.shapes[i]);
    float w = exp(-d / sdfs.smooth);
    sumExp += w;
    sumDist += w * d;
  }

  return sumDist / sumExp;  // Smooth union
}
```

### Raycasting

```typescript
// Wasm-accelerated ray-splat intersection
class SplatMesh {
  raycast(raycaster: THREE.Raycaster, intersects: Intersection[]) {
    if (!this.isInitialized) return;

    // Transform ray to object space
    const invMatrix = this.matrixWorld.clone().invert();
    const origin = raycaster.ray.origin.clone().applyMatrix4(invMatrix);
    const direction = raycaster.ray.direction.clone()
      .transformDirection(invMatrix);

    // Call Wasm function
    const hits = raycast_splats(
      this.packedSplats.packedArray,
      this.numSplats,
      [origin.x, origin.y, origin.z],
      [direction.x, direction.y, direction.z]
    );

    // Convert hits to Three.js format
    for (const hit of hits) {
      intersects.push({
        distance: hit.t,
        point: raycaster.ray.at(hit.t, new THREE.Vector3()),
        object: this,
        splatIndex: hit.index
      });
    }
  }
}
```

### Environment Map Rendering

```typescript
// Render splats to cubemap for IBL
async renderEnvMap(scene: THREE.Scene, options: {
  position: THREE.Vector3,
  resolution: number,
  hideObjects?: THREE.Object3D[]
}): Promise<THREE.Texture> {
  // Create cube camera
  const cubeCamera = new THREE.CubeCamera(0.1, 1000,
    new THREE.WebGLCubeRenderTarget(options.resolution, {
      generateMipmaps: true,
      minFilter: THREE.LinearMipmapLinearFilter
    })
  );
  cubeCamera.position.copy(options.position);

  // Hide specified objects
  const visibility = options.hideObjects?.map(o => o.visible);
  options.hideObjects?.forEach(o => o.visible = false);

  // Render 6 faces
  cubeCamera.update(this.renderer, scene);

  // Restore visibility
  options.hideObjects?.forEach((o, i) => o.visible = visibility![i]);

  // Pre-filter for PBR
  const pmrem = new THREE.PMREMGenerator(this.renderer);
  return pmrem.fromCubemap(cubeCamera.renderTarget.texture).texture;
}
```

### Procedural Particle Animation (Snow Generator)

SparkJS includes a clever stateless animation technique for particle effects:

```typescript
// Key insight: Deterministic animation without per-particle state
function snowGenerator(box: Box3, options: SnowOptions) {
  const dynoTime = dynoFloat(0);
  const dynoMinY = dynoFloat(box.min.y);

  return new SplatGenerator({
    numSplats: options.count,

    generator: dynoBlock({
      inputs: { index: 'int' },
      outputs: { gsplat: 'Gsplat' },
    }, ({ index }) => {
      // 1. Deterministic position from index hash (consistent across frames)
      const basePos = hashVec3(index);

      // 2. Per-particle time offset for desynchronized motion
      const timeOffset = hashFloat(add(index, 12345));

      // 3. Layered motion:
      //    - Primary: constant fall velocity
      //    - Secondary: sinusoidal wander
      const fallOffset = mul(dynoTime, options.fallVelocity);
      const wander = mul(
        sin(add(vec3(timeOffset), mul(dynoTime, options.wanderFreq))),
        options.wanderScale
      );

      // 4. Combine and clamp to ground
      const position = max(
        add(basePos, fallOffset, wander),
        dynoMinY  // Fake ground collision
      );

      // 5. Output splat
      return {
        gsplat: combineGsplat({
          center: position,
          scales: vec3(options.size),
          quaternion: vec4(0, 0, 0, 1),
          rgba: options.color
        })
      };
    }),

    update: (dt) => {
      dynoTime.value += dt * options.fallVelocity;
      // Wrap time to prevent float precision issues
      if (dynoTime.value > 1000) dynoTime.value -= 1000;
    }
  });
}
```

**Key Techniques**:
1. **Stateless animation**: No per-particle state, all derived from index + time
2. **Hash-based determinism**: Same visual result if replayed
3. **Layered motion**: Combine multiple simple functions for complex behavior
4. **Time wrapping**: Prevent float precision degradation over time

### Normal-to-Color Modifier

Visualize surface normals as RGB colors:

```typescript
function makeNormalColorModifier(context: SplatMeshContext): GsplatModifier {
  return dynoBlock({
    inputs: { gsplat: 'Gsplat' },
    outputs: { gsplat: 'Gsplat' }
  }, ({ gsplat }) => {
    // Extract normal from splat orientation
    const normal = gsplatNormal(gsplat);

    // Transform to view space
    const viewCenter = transformPos(
      splitGsplat(gsplat).center,
      { transform: context.viewToObject }
    );

    // Flip normal if facing away from camera
    const facing = dot(viewCenter, normal);
    const correctedNormal = select(
      lessThan(facing, float(0)),
      neg(normal),
      normal
    );

    // Map [-1, 1] → [0, 1] for RGB display
    const color = add(mul(correctedNormal, 0.5), 0.5);

    return {
      gsplat: combineGsplat({
        ...splitGsplat(gsplat),
        rgba: vec4(color, 1.0)
      })
    };
  });
}
```

---

## Applicability to Luxar nD

### What Can Be Directly Reused

1. **Accumulator Pattern**: Pool accumulators for multiple splat sources
2. **Worker Sorting**: Background thread radix sort
3. **Version-Based Caching**: Skip unchanged generators
4. **Instanced Geometry**: Single draw call for all splats
5. **Threshold-Based Updates**: Avoid sorting when camera moves slightly

### What Needs Adaptation for nD

1. **Data Format**: Must support variable dimensions
   - SparkJS: Fixed 3D (16 bytes)
   - Luxar: Variable (36 bytes 2D, 56 bytes 3D, 80 bytes 4D)

2. **Covariance Representation**:
   - SparkJS: Quaternion + scales (axis-aligned then rotated)
   - Luxar: Full Cholesky factors (arbitrary covariance)

3. **Distance Metric for Sorting**:
   - SparkJS: 3D Euclidean distance
   - Luxar: nD hyperbolic distance with slice intersection

4. **Sharpness Parameter**:
   - SparkJS: None (standard Gaussian only)
   - Luxar: Per-splat sharpness for super-Gaussian

5. **Projection Shader**:
   - SparkJS: 3D → 2D screen projection
   - Luxar: nD → 3D slice → 2D screen projection

### Recommended Architecture for Luxar

```
┌─────────────────────────────────────────────────────────────────┐
│                    LuxarSplatRenderer                           │
│  - Extends THREE.Mesh (like SparkRenderer)                     │
│  - Manages nD splat accumulation                               │
│  - Dimension-aware sorting                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ GSplatData    │ │ GSplatData    │ │ GSplatData    │
    │ from .zarr    │ │ procedural    │ │ from Python   │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ PackedSplatsND  │
                    │ - Variable size │
                    │ - Cholesky      │
                    │ - Sharpness     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ NDSliceViewport │
                    │ - nD → 3D slice │
                    │ - Hyperbolic    │
                    │   distance sort │
                    └─────────────────┘
```

### Key Implementation Notes

1. **Use Float32 for nD positions**: Float16 precision is insufficient for higher dimensions

2. **Cholesky unpacking in shader**: Need efficient triangular solve in GLSL

3. **Sharpness falloff**: Replace `exp(-0.5 * d²)` with `exp(-0.5 * d^s)`

4. **Slice intersection**: Compute visibility based on nD hypersphere intersection with slice plane

5. **Consider WASM**: SparkJS uses WASM for raycasting; consider for nD distance computation

---

## References

- [SparkJS GitHub](https://github.com/sparkjsdev/spark)
- [SparkJS Documentation](https://sparkjs.dev/docs/)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Octahedral Encoding](https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/)
- [Dual Quaternion Skinning](https://users.cs.utah.edu/~ladislav/dq/index.html)
