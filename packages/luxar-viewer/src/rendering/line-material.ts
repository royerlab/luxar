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
import { config } from '../config';
import { materialManager } from './material-manager';

/**
 * Configuration for line material creation
 */
export interface LineMaterialConfig {
  /** Opacity multiplier (0.0 to 1.0) */
  opacity?: number;
  /** Blending mode ('additive' | 'normal' | 'max') */
  blendingMode?: 'additive' | 'normal' | 'max';
  /** HDR intensity multiplier */
  hdrMultiplier?: number;
}

/**
 * Line material uniforms interface
 */
export interface LineMaterialUniforms {
  /** Field of view in radians */
  uFOV: { value: number };
  /** Viewport resolution [width, height] */
  uResolution: { value: THREE.Vector2 };
  /** HDR intensity multiplier */
  uHDRMultiplier: { value: number };
  /** Opacity multiplier */
  uOpacity: { value: number };
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

    uniform float uHDRMultiplier;
    uniform float uOpacity;

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

      // HDR output (gamma correction in post-processing)
      vec3 finalColor = vColor * intensity * uHDRMultiplier;

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
    const blending =
      blendingMode === 'additive'
        ? THREE.AdditiveBlending
        : blendingMode === 'max'
          ? THREE.CustomBlending
          : THREE.NormalBlending;

    super({
      uniforms: {
        uFOV: { value: (60 * Math.PI) / 180 }, // Default 60° FOV
        uResolution: { value: new THREE.Vector2(1, 1) },
        uHDRMultiplier: {
          value: materialConfig.hdrMultiplier ?? config.shader.points.hdrMultiplier,
        },
        uOpacity: { value: materialConfig.opacity ?? 1.0 },
      },

      vertexShader: LineMaterial.VERTEX_SHADER,
      fragmentShader: LineMaterial.FRAGMENT_SHADER,

      // GLSL ES 3.0 for consistency with other materials
      glslVersion: THREE.GLSL3,

      transparent: true,
      depthWrite: blendingMode !== 'additive' && blendingMode !== 'max', // No depth write for additive/max
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
   * Update HDR multiplier.
   */
  updateHDRMultiplier(multiplier: number): void {
    this.uniforms.uHDRMultiplier.value = multiplier;
  }

  /**
   * Update opacity.
   */
  updateOpacity(opacity: number): void {
    this.uniforms.uOpacity.value = opacity;
  }

  /**
   * Clone this material.
   */
  clone(): this {
    const cloned = new LineMaterial({
      opacity: this.uniforms.uOpacity.value,
      hdrMultiplier: this.uniforms.uHDRMultiplier.value,
      blendingMode:
        this.blending === THREE.AdditiveBlending
          ? 'additive'
          : this.blending === THREE.CustomBlending && this.blendEquation === THREE.MaxEquation
            ? 'max'
            : 'normal',
    });

    cloned.uniforms.uFOV.value = this.uniforms.uFOV.value;
    cloned.uniforms.uResolution.value.copy(this.uniforms.uResolution.value);

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

  // Compute bounding box from segment positions
  const positions = new Float32Array(meshConfig.segmentCount * 6);
  for (let i = 0; i < meshConfig.segmentCount; i++) {
    positions[i * 6 + 0] = meshConfig.startPositions[i * 3 + 0];
    positions[i * 6 + 1] = meshConfig.startPositions[i * 3 + 1];
    positions[i * 6 + 2] = meshConfig.startPositions[i * 3 + 2];
    positions[i * 6 + 3] = meshConfig.endPositions[i * 3 + 0];
    positions[i * 6 + 4] = meshConfig.endPositions[i * 3 + 1];
    positions[i * 6 + 5] = meshConfig.endPositions[i * 3 + 2];
  }

  const tempGeometry = new THREE.BufferGeometry();
  tempGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  tempGeometry.computeBoundingBox();
  tempGeometry.computeBoundingSphere();

  if (tempGeometry.boundingBox) {
    geometry.boundingBox = tempGeometry.boundingBox.clone();
  }
  if (tempGeometry.boundingSphere) {
    geometry.boundingSphere = tempGeometry.boundingSphere.clone();
  }

  tempGeometry.dispose();

  // Create mesh with instanced geometry
  // Using THREE.Mesh instead of THREE.InstancedMesh avoids the instanceMatrix attribute
  // which would push us over WebGL's 16 attribute location limit
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = true;

  return mesh;
}
