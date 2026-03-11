/**
 * Line Material for Luxar
 *
 * Specialized THREE.ShaderMaterial for rendering thick lines using instanced quads.
 * Uses the semicircle kernel model for mathematically correct joints with additive blending.
 *
 * Key features:
 * - Instanced quad geometry (4 vertices per segment)
 * - World-space line width
 * - Parabolic intensity falloff: (1 - p²)^sharpness
 * - Cap factor for seamless joints (0.5 at endpoints, 1.0 in body)
 * - Per-vertex attributes (color, width, sharpness)
 *
 * @module rendering/line-material
 */

import * as THREE from 'three';
import { materialManager } from './material-manager';

/**
 * Configuration for line material creation
 */
export interface LineMaterialConfig {
  /** Opacity multiplier (0.0 to 1.0) */
  opacity?: number;
  /** Gamma correction (0.1 to 10.0, default 1.0) */
  gamma?: number;
  /** Intensity (linear color multiplier / gain), default 1.0 */
  intensity?: number;
  /** Offset (additive brightness shift / black level), default 0.0 */
  offset?: number;
  /** Blending mode */
  blendingMode?: 'additive' | 'normal' | 'max' | 'opaque' | 'luminous';
  /** Whether material is transparent (default true) */
  transparent?: boolean;
  /** Whether to test against depth buffer (default true; additive sets false) */
  depthTest?: boolean;
}

/**
 * Line material uniforms interface
 */
export interface LineMaterialUniforms {
  /** Field of view in radians */
  uFOV: { value: number };
  /** Viewport resolution [width, height] */
  uResolution: { value: THREE.Vector2 };
  /** Opacity multiplier */
  uOpacity: { value: number };
  /** Pre-computed 1/gamma for performance */
  uInvGamma: { value: number };
}

/**
 * Line material using instanced quads with semicircle kernel rendering.
 *
 * The semicircle kernel produces parabolic intensity profiles that sum correctly
 * at joints when using additive blending:
 * - Body intensity: (1 - p²)^sharpness where p = perpendicular distance
 * - Endpoint cap factor: 0.5 (half intensity at true endpoints)
 * - Joint rendering: 0.5 + 0.5 = 1.0 (seamless sum)
 */
export class LineMaterial extends THREE.ShaderMaterial {
  /**
   * Vertex shader with screen-space expansion and cap factor calculation.
   *
   * Instance attributes (per segment):
   * - aStartPos, aEndPos: 3D segment endpoints
   * - aStartColor, aEndColor: Per-vertex colors (HDR)
   * - aStartWidth, aEndWidth: Per-vertex half-widths (world units)
   * - aStartSharpness, aEndSharpness: Per-vertex sharpness
   * - aSegmentLength: 3D segment length (for cap factor)
   * - aStartClipped, aEndClipped: Whether endpoints were clipped (force capFactor=1.0)
   */
  // GLSL ES 3.0 for consistency with other materials
  // Note: varyings need smooth interpolation (not flat) as they vary across the quad
  private static readonly VERTEX_SHADER = /* glsl */ `
    precision highp float;

    // Static geometry attribute (per quad vertex)
    in vec2 aQuadCorner;  // (-1,-1), (1,-1), (-1,1), (1,1)

    // Instanced attributes (per segment)
    in vec3 aStartPos;
    in vec3 aEndPos;
    in vec3 aStartColor;
    in vec3 aEndColor;
    in float aStartWidth;
    in float aEndWidth;
    in float aStartSharpness;
    in float aEndSharpness;
    in float aSegmentLength;
    in float aStartClipped;
    in float aEndClipped;

    // Uniforms
    uniform float uFOV;
    uniform vec2 uResolution;

    // Varyings to fragment shader (smooth interpolation needed)
    out vec3 vColor;
    out float vSharpness;
    out float vPerpNorm;  // Signed: -1 at bottom edge, +1 at top edge
    out float vCapFactor; // 0.5 at true endpoints, 1.0 in body
    out float vPixelWidth; // Line width in pixels (for anti-aliasing)

    void main() {
      // Position along segment: 0 = start, 1 = end
      float t = aQuadCorner.x > 0.0 ? 1.0 : 0.0;

      // Interpolate attributes along segment
      vec3 worldPos = mix(aStartPos, aEndPos, t);
      vColor = mix(aStartColor, aEndColor, t);
      float width = mix(aStartWidth, aEndWidth, t);
      vSharpness = mix(aStartSharpness, aEndSharpness, t);

      // Project to clip space
      vec4 clipStart = projectionMatrix * modelViewMatrix * vec4(aStartPos, 1.0);
      vec4 clipEnd = projectionMatrix * modelViewMatrix * vec4(aEndPos, 1.0);
      vec4 clipPos = projectionMatrix * modelViewMatrix * vec4(worldPos, 1.0);

      // Convert clip-space endpoints to pixel coordinates for correct aspect ratio handling
      // NDC to pixels: ndc * resolution / 2 (NDC range -1 to +1, pixels range 0 to resolution)
      vec2 ndcStart = clipStart.xy / clipStart.w;
      vec2 ndcEnd = clipEnd.xy / clipEnd.w;
      vec2 pixelStart = (ndcStart * 0.5 + 0.5) * uResolution;
      vec2 pixelEnd = (ndcEnd * 0.5 + 0.5) * uResolution;

      // Compute line direction and perpendicular in pixel space (aspect-ratio correct)
      vec2 pixelDir = pixelEnd - pixelStart;
      float pixelLen = length(pixelDir);

      // Handle degenerate segments (zero length in screen space)
      vec2 lineDir = pixelLen > 0.0001 ? pixelDir / pixelLen : vec2(1.0, 0.0);
      vec2 perpendicular = vec2(-lineDir.y, lineDir.x);  // Unit vector in pixel space

      // World-space to pixel conversion (perspective-correct)
      float dist = length((modelViewMatrix * vec4(worldPos, 1.0)).xyz);
      float tanHalfFov = tan(uFOV * 0.5);
      float rawPixelWidth = width * uResolution.y / (dist * tanHalfFov);

      // Enforce minimum pixel width to prevent sub-pixel rendering artifacts
      // Lines thinner than ~1.5 pixels cause severe aliasing due to rasterization gaps
      float minPixelWidth = 1.5;
      float pixelWidth = max(rawPixelWidth, minPixelWidth);

      // Pass raw pixel width to fragment shader for intensity scaling
      // This allows thin lines to render at minimum width but with reduced intensity
      vPixelWidth = rawPixelWidth;

      // Perpendicular position: -1 at bottom edge, +1 at top edge
      // GPU interpolates this across the quad, giving 0 at centerline
      vPerpNorm = aQuadCorner.y;

      // Expand quad by perpendicular offset in pixel space, then convert to clip space
      // pixelOffset is in pixels, convert to NDC then to clip space
      vec2 pixelOffset = perpendicular * aQuadCorner.y * pixelWidth;
      vec2 ndcOffset = pixelOffset / uResolution * 2.0;
      clipPos.xy += ndcOffset * clipPos.w;

      // Cap factor calculation with clipping awareness
      // Normal cap factor: 0.5 at true endpoints, 1.0 in body
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

      gl_Position = clipPos;
    }
  `;

  /**
   * Fragment shader with parabolic falloff and cap factor.
   *
   * The parabolic profile (1 - p²)^sharpness results from convolving
   * a semicircle kernel with the line path. This ensures:
   * - Smooth intensity falloff from centerline to edge
   * - Correct half-intensity at endpoints for seamless joints
   */
  // GLSL ES 3.0 for consistency with other materials
  private static readonly FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    uniform float uOpacity;
    uniform float uInvGamma; // Pre-computed 1/gamma for performance
    uniform float uIntensity; // Per-node linear color multiplier (gain)
    uniform float uOffset; // Per-node additive brightness shift (black level)

    in vec3 vColor;
    in float vSharpness;
    in float vPerpNorm;  // Interpolated: 0 at centerline, ±1 at edges
    in float vCapFactor; // 0.5 at endpoints, 1.0 in body
    in float vPixelWidth; // Raw line width in pixels (before minimum clamping)

    out vec4 fragColor;

    void main() {
      // Compute distance from centerline (0 to 1)
      float p = abs(vPerpNorm);

      // Discard pixels clearly outside the line width
      if (p >= 1.0) discard;

      // Parabolic falloff from semicircle kernel convolution
      // Base: (1 - p²) where p = distance from centerline
      // With per-vertex sharpness: (1 - p²)^sharpness
      float perpFalloff = pow(1.0 - p * p, vSharpness);

      // Anti-aliasing: smooth falloff at edges
      // The AA region is ~1 pixel wide in the rendered quad
      // Since we enforce minimum 1.5px width, use that as reference
      float minPixelWidth = 1.5;
      float renderedWidth = max(vPixelWidth, minPixelWidth);
      float aaWidth = 1.0 / renderedWidth;  // ~1 pixel in normalized coords
      float edgeAA = 1.0 - smoothstep(1.0 - aaWidth, 1.0, p);

      // Intensity scaling for sub-pixel lines
      // When a line is rendered wider than intended, reduce intensity proportionally
      // This preserves the visual "weight" of thin lines
      float widthScale = min(vPixelWidth / minPixelWidth, 1.0);

      // Apply cap factor for correct joint intensity
      float intensity = vCapFactor * perpFalloff * edgeAA * widthScale;

      // Per-node GOG (Gain-Offset-Gamma) color adjustment
      vec3 adjusted = vColor * uIntensity + uOffset;
      adjusted = max(adjusted, vec3(0.0));

      // Early discard for zero-contribution fragments after offset
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;

      vec3 gammaColor = pow(adjusted, vec3(uInvGamma));

      // Output color with alpha for AdditiveBlending (SrcAlpha, One)
      vec3 finalColor = gammaColor * intensity;
      fragColor = vec4(finalColor, intensity * uOpacity);
    }
  `;

  /**
   * Create a new LineMaterial with the specified configuration.
   *
   * @param config - Material configuration options
   */
  constructor(materialConfig: LineMaterialConfig = {}) {
    const blendingMode = materialConfig.blendingMode ?? 'additive';
    const isOpaque = blendingMode === 'opaque';
    const isAdditive = blendingMode === 'additive';
    const gammaValue = Math.max(0.001, materialConfig.gamma ?? 1.0); // Prevent division by zero

    // Determine THREE.js blending mode
    // 'additive' and 'luminous' both use AdditiveBlending - only depthTest differs
    let blending: THREE.Blending;
    if (isOpaque || blendingMode === 'normal') {
      blending = THREE.NormalBlending;
    } else if (blendingMode === 'additive' || blendingMode === 'luminous') {
      blending = THREE.AdditiveBlending; // Classic additive: SrcAlpha, One
    } else if (blendingMode === 'max') {
      blending = THREE.CustomBlending;
    } else {
      blending = THREE.NormalBlending;
    }

    super({
      uniforms: {
        uFOV: { value: (60 * Math.PI) / 180 }, // Default 60° FOV
        uResolution: { value: new THREE.Vector2(1, 1) },
        uOpacity: { value: materialConfig.opacity ?? 1.0 },
        uInvGamma: { value: 1.0 / gammaValue }, // Pre-computed inverse for performance
        uIntensity: { value: materialConfig.intensity ?? 1.0 },
        uOffset: { value: materialConfig.offset ?? 0.0 },
      },

      vertexShader: LineMaterial.VERTEX_SHADER,
      fragmentShader: LineMaterial.FRAGMENT_SHADER,

      // GLSL ES 3.0 for consistency with other materials
      glslVersion: THREE.GLSL3,

      transparent: materialConfig.transparent ?? !isOpaque,
      depthWrite:
        isOpaque || (blendingMode === 'normal' && (materialConfig.opacity ?? 1.0) >= 0.99),
      // Additive ignores depth (renders on top), luminous respects depth occlusion
      depthTest: materialConfig.depthTest ?? !isAdditive,
      toneMapped: false, // HDR values pass through to post-processing
      blending: blending,
      side: THREE.DoubleSide, // Lines visible from both sides
    });

    // Configure custom blending for max mode
    if (blendingMode === 'max') {
      this.blendEquation = THREE.MaxEquation; // Max(source, destination)
      this.blendSrc = THREE.OneFactor;
      this.blendDst = THREE.OneFactor;
    }

    // Store blendingMode and gamma in userData for clone()
    this.userData.blendingMode = blendingMode;
    this.userData.gamma = gammaValue;
    this.userData.depthTest = materialConfig.depthTest ?? !isAdditive;
  }

  /**
   * Update camera parameters for world-space line sizing.
   *
   * @param fov - Field of view in radians
   * @param resolution - Viewport resolution
   */
  updateCameraParams(fov: number, resolution: THREE.Vector2): void {
    this.uniforms.uFOV.value = fov;
    this.uniforms.uResolution.value.copy(resolution);
  }

  /**
   * Update opacity.
   */
  updateOpacity(opacity: number): void {
    this.uniforms.uOpacity.value = opacity;
  }

  /**
   * Update gamma correction.
   * Only invGamma is used in shader; gamma value stored in userData for clone()
   */
  updateGamma(gamma: number): void {
    const safeGamma = Math.max(0.001, gamma); // Prevent division by zero
    this.userData.gamma = safeGamma;
    this.uniforms.uInvGamma.value = 1.0 / safeGamma;
  }

  /**
   * Update intensity (linear color multiplier)
   */
  updateIntensity(intensity: number): void {
    this.uniforms.uIntensity.value = intensity;
  }

  /**
   * Update offset (additive brightness shift)
   */
  updateOffset(offset: number): void {
    this.uniforms.uOffset.value = offset;
  }

  /**
   * Clone this material.
   */
  clone(): this {
    const cloned = new LineMaterial({
      opacity: this.uniforms.uOpacity.value,
      gamma: this.userData.gamma ?? 1.0,
      intensity: this.uniforms.uIntensity.value,
      offset: this.uniforms.uOffset.value,
      blendingMode: this.userData.blendingMode ?? 'additive',
      transparent: this.transparent,
      depthTest: this.userData.depthTest ?? true,
    });

    // Copy blend equation settings for custom blending (max mode)
    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
    }

    cloned.uniforms.uFOV.value = this.uniforms.uFOV.value;
    cloned.uniforms.uResolution.value.copy(this.uniforms.uResolution.value);
    cloned.uniforms.uInvGamma.value = this.uniforms.uInvGamma.value;

    return cloned as this;
  }

  /**
   * Dispose this material and unregister from MaterialManager.
   */
  dispose(): void {
    materialManager.unregister(this);
    super.dispose();
  }
}

// ============================================================================
// Instanced Geometry Creation
// ============================================================================

/**
 * Create the base quad geometry for line instances.
 *
 * Each line segment is rendered as a quad with 4 vertices:
 * - (-1, -1): Start, bottom edge
 * - ( 1, -1): End, bottom edge
 * - (-1,  1): Start, top edge
 * - ( 1,  1): End, top edge
 *
 * The vertex shader expands these in screen space based on line width.
 *
 * @returns THREE.BufferGeometry for instanced rendering
 */
export function createLineQuadGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();

  // Quad corners: x determines position along segment, y determines edge
  const quadCorners = new Float32Array([
    -1,
    -1, // Start, bottom
    1,
    -1, // End, bottom
    -1,
    1, // Start, top
    1,
    1, // End, top
  ]);

  // Triangle indices for the quad
  const indices = new Uint16Array([
    0,
    1,
    2, // First triangle
    2,
    1,
    3, // Second triangle
  ]);

  geometry.setAttribute('aQuadCorner', new THREE.BufferAttribute(quadCorners, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  return geometry;
}

/**
 * Configuration for instanced lines mesh
 */
export interface InstancedLinesMeshConfig {
  /** Segment start positions (segmentCount * 3) */
  startPositions: Float32Array;
  /** Segment end positions (segmentCount * 3) */
  endPositions: Float32Array;
  /** Start colors (segmentCount * 3) */
  startColors: Float32Array;
  /** End colors (segmentCount * 3) */
  endColors: Float32Array;
  /** Start widths (segmentCount) */
  startWidths: Float32Array;
  /** End widths (segmentCount) */
  endWidths: Float32Array;
  /** Start sharpness (segmentCount) */
  startSharpness: Float32Array;
  /** End sharpness (segmentCount) */
  endSharpness: Float32Array;
  /** Segment lengths (segmentCount) */
  segmentLengths: Float32Array;
  /** Whether start was clipped (segmentCount) */
  startClipped: Uint8Array;
  /** Whether end was clipped (segmentCount) */
  endClipped: Uint8Array;
  /** Number of segments */
  segmentCount: number;
}

/**
 * Create an instanced mesh for lines rendering.
 *
 * Sets up the instanced geometry with all per-segment attributes.
 *
 * Note: We use THREE.Mesh instead of THREE.InstancedMesh because:
 * - InstancedMesh adds instanceMatrix (mat4 = 4 attribute locations)
 * - Our shader computes positions from custom attributes, not matrices
 * - This avoids exceeding WebGL's 16 attribute location limit
 * - InstancedBufferGeometry with Mesh still uses instanced drawing
 *
/**
 * Compute bounding box and sphere from line segment start/end positions.
 * Uses a direct min/max pass without temporary geometry or array allocations.
 */
function computeLineBounds(
  geometry: THREE.InstancedBufferGeometry,
  meshConfig: InstancedLinesMeshConfig
): void {
  const box = new THREE.Box3(
    new THREE.Vector3(Infinity, Infinity, Infinity),
    new THREE.Vector3(-Infinity, -Infinity, -Infinity)
  );
  const v = new THREE.Vector3();

  for (let i = 0; i < meshConfig.segmentCount; i++) {
    const si = i * 3;
    v.set(
      meshConfig.startPositions[si],
      meshConfig.startPositions[si + 1],
      meshConfig.startPositions[si + 2]
    );
    box.expandByPoint(v);
    v.set(
      meshConfig.endPositions[si],
      meshConfig.endPositions[si + 1],
      meshConfig.endPositions[si + 2]
    );
    box.expandByPoint(v);
  }

  geometry.boundingBox = box;
  geometry.boundingSphere = new THREE.Sphere();
  box.getBoundingSphere(geometry.boundingSphere);
}

/**
 * @param meshConfig - Configuration with all segment data
 * @param material - LineMaterial to use for rendering
 * @returns THREE.Mesh with InstancedBufferGeometry ready for scene addition
 */
export function createInstancedLinesMesh(
  meshConfig: InstancedLinesMeshConfig,
  material: LineMaterial
): THREE.Mesh {
  const baseGeometry = createLineQuadGeometry();

  // Create instanced buffer geometry
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = baseGeometry.index;
  geometry.setAttribute('aQuadCorner', baseGeometry.getAttribute('aQuadCorner'));

  // Set instanced attributes
  geometry.setAttribute(
    'aStartPos',
    new THREE.InstancedBufferAttribute(meshConfig.startPositions, 3)
  );
  geometry.setAttribute('aEndPos', new THREE.InstancedBufferAttribute(meshConfig.endPositions, 3));
  geometry.setAttribute(
    'aStartColor',
    new THREE.InstancedBufferAttribute(meshConfig.startColors, 3)
  );
  geometry.setAttribute('aEndColor', new THREE.InstancedBufferAttribute(meshConfig.endColors, 3));
  geometry.setAttribute(
    'aStartWidth',
    new THREE.InstancedBufferAttribute(meshConfig.startWidths, 1)
  );
  geometry.setAttribute('aEndWidth', new THREE.InstancedBufferAttribute(meshConfig.endWidths, 1));
  geometry.setAttribute(
    'aStartSharpness',
    new THREE.InstancedBufferAttribute(meshConfig.startSharpness, 1)
  );
  geometry.setAttribute(
    'aEndSharpness',
    new THREE.InstancedBufferAttribute(meshConfig.endSharpness, 1)
  );
  geometry.setAttribute(
    'aSegmentLength',
    new THREE.InstancedBufferAttribute(meshConfig.segmentLengths, 1)
  );

  // Convert Uint8Array to Float32Array for clipped flags (shader expects float)
  const startClippedFloat = new Float32Array(meshConfig.startClipped);
  const endClippedFloat = new Float32Array(meshConfig.endClipped);

  geometry.setAttribute('aStartClipped', new THREE.InstancedBufferAttribute(startClippedFloat, 1));
  geometry.setAttribute('aEndClipped', new THREE.InstancedBufferAttribute(endClippedFloat, 1));

  // Set instance count
  geometry.instanceCount = meshConfig.segmentCount;

  // Compute bounding box from segment positions (direct min/max pass, no temp allocations)
  computeLineBounds(geometry, meshConfig);

  // Create mesh with instanced geometry
  // Using THREE.Mesh instead of THREE.InstancedMesh avoids the instanceMatrix attribute
  // which would push us over WebGL's 16 attribute location limit
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = true;

  return mesh;
}

/**
 * Update an existing instanced lines mesh with new segment data.
 *
 * Mirrors the pattern in `updateInstancedGSplatsMesh` (gsplat-material.ts):
 * - Same count: in-place `.set()` on existing attributes (zero GPU allocation)
 * - Different count: `setAttribute` with new InstancedBufferAttribute + `_maxInstanceCount` fix
 * - Always: recompute bounding box/sphere from segment positions
 *
 * @param mesh - Existing mesh to update (must have InstancedBufferGeometry)
 * @param meshConfig - New segment data
 */
export function updateInstancedLinesMesh(
  mesh: THREE.Mesh,
  meshConfig: InstancedLinesMeshConfig
): void {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
  const currentCount = geometry.instanceCount;

  // Attribute layout: [name, source data, components per instance]
  const attrSpecs: Array<[string, Float32Array | Uint8Array, number, boolean]> = [
    ['aStartPos', meshConfig.startPositions, 3, false],
    ['aEndPos', meshConfig.endPositions, 3, false],
    ['aStartColor', meshConfig.startColors, 3, false],
    ['aEndColor', meshConfig.endColors, 3, false],
    ['aStartWidth', meshConfig.startWidths, 1, false],
    ['aEndWidth', meshConfig.endWidths, 1, false],
    ['aStartSharpness', meshConfig.startSharpness, 1, false],
    ['aEndSharpness', meshConfig.endSharpness, 1, false],
    ['aSegmentLength', meshConfig.segmentLengths, 1, false],
    ['aStartClipped', meshConfig.startClipped, 1, true], // Uint8 → Float32
    ['aEndClipped', meshConfig.endClipped, 1, true], // Uint8 → Float32
  ];

  if (meshConfig.segmentCount !== currentCount) {
    // Size changed: recreate attributes
    for (const [name, data, size, needsFloat32Convert] of attrSpecs) {
      const arrayData = needsFloat32Convert ? new Float32Array(data) : (data as Float32Array);
      geometry.setAttribute(name, new THREE.InstancedBufferAttribute(arrayData, size));
    }
    geometry.instanceCount = meshConfig.segmentCount;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
    // Same issue as gsplats: meshes created with 0 instances cache _maxInstanceCount=0.
    // (THREE.js r163+ internal property)
    delete (geometry as any)._maxInstanceCount;
  } else {
    // Same size: update in place (zero GPU allocation)
    for (const [name, data, , needsFloat32Convert] of attrSpecs) {
      const attr = geometry.getAttribute(name) as THREE.InstancedBufferAttribute;
      const arrayData = needsFloat32Convert ? new Float32Array(data) : data;
      attr.set(arrayData);
      attr.needsUpdate = true;
    }
  }

  // Recompute bounding box from segment positions (direct min/max pass, no temp allocations)
  computeLineBounds(geometry, meshConfig);
}
