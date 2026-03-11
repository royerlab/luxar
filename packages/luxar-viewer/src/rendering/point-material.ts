/**
 * Point Material for Luxar
 *
 * Specialized THREE.ShaderMaterial for physically accurate points rendering.
 * Features world-space sizing, HDR colors, and per-point sharpness control.
 */

import * as THREE from 'three';
import { config } from '../config';
import { materialManager } from './material-manager';

/**
 * Configuration for point material creation
 */
export interface PointMaterialConfig {
  opacity?: number;
  gamma?: number;
  intensity?: number; // Linear color multiplier (gain), default 1.0
  offset?: number; // Additive brightness shift (black level), default 0.0
  blending?: THREE.Blending;
  depthWrite?: boolean;
  depthTest?: boolean; // Whether to test against depth buffer (default true)
  transparent?: boolean; // Whether material is transparent (default true)
  radiusScale?: number; // Scale factor for radius normalization (e.g., 1/255 for uint8)
  sharpnessScale?: number; // Scale factor for sharpness normalization (e.g., 1/255 for uint8)
}

/**
 * Points material with physically accurate world-space sizing.
 * Extends THREE.ShaderMaterial to provide specialized point rendering.
 */
export class PointMaterial extends THREE.ShaderMaterial {
  // Static vertex shader with CORRECT world-space sizing formula
  // GLSL ES 3.0 for consistency with other materials
  // OPTIMIZATIONS:
  // - inversesqrt() instead of length() + divide (native GPU instruction)
  // - Pre-computed pointSizeFactor uniform (2.0 * resolution.y / tanHalfFov)
  private static readonly VERTEX_SHADER = /* glsl */ `
    precision highp float;

    in float radius;
    in float sharpness;
    uniform float pointSizeFactor; // Pre-computed: 2.0 * resolution.y / tanHalfFov
    uniform float maxPointSize;    // Pre-computed: resolution.y * 0.5
    uniform float radiusScale;
    uniform float sharpnessScale;

    out mediump vec3 vColor;
    out mediump float vSharpness;
    out highp float vRadius; // Pass radius to fragment for zero-check (needs precision)

    void main() {
      // Pass vertex color to fragment shader
      vColor = color;

      // Apply sharpness scale for dtype normalization and use 2.0 as default
      float normalizedSharpness = sharpness * sharpnessScale;
      vSharpness = normalizedSharpness > 0.0 ? normalizedSharpness : 2.0;

      // Transform vertex position from world space to view space
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;

      // Apply radius scale for dtype normalization (e.g., uint8 needs 1/255 scale)
      float normalizedRadius = radius * radiusScale;
      vRadius = normalizedRadius; // Pass to fragment shader

      // OPTIMIZED world-space point sizing:
      // - inversesqrt is a native GPU instruction (faster than sqrt + divide)
      // - pointSizeFactor pre-computed in JS: 2.0 * resolution.y / tanHalfFov
      float invDistance = inversesqrt(dot(mvPosition.xyz, mvPosition.xyz));
      float basePointSize = normalizedRadius * pointSizeFactor * invDistance;

      // Sharpness compensation based on visibility threshold
      // For falloff function f(r) = (1-r)^s, the visible radius where intensity drops to 1% is:
      // r_vis = 1 - 0.01^(1/s)
      // We need to scale the point size by 1/r_vis to maintain consistent visible size
      // Using approximation: compensation = 1.0 + (s - 1.0) * 0.15
      // This gives: s=1→1.0, s=2→1.15, s=4→1.45, s=8→2.05
      float sharpnessCompensation = 1.0 + (vSharpness - 1.0) * 0.15;
      float pointSize = basePointSize * sharpnessCompensation;

      // Clamp to hardware limits, with minimum of 1.0 to avoid undefined behavior
      // Zero-radius filtering happens in fragment shader
      gl_PointSize = max(1.0, min(pointSize, maxPointSize));
    }
  `;

  // Optimized fragment shader
  // GLSL ES 3.0 for consistency with other materials
  // OPTIMIZATIONS:
  // - mediump precision for color/falloff (reduces register pressure)
  // - sqrt(4.0 * r2) instead of sqrt(r2) * 2.0 (one fewer multiply)
  // - Removed unused gamma uniform (only invGamma is used)
  private static readonly FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    uniform mediump float opacity;
    uniform mediump float baseAlpha;
    uniform mediump float invGamma; // Pre-computed 1/gamma for performance
    uniform mediump float uIntensity; // Per-node linear color multiplier (gain)
    uniform mediump float uOffset; // Per-node additive brightness shift (black level)

    in mediump vec3 vColor;
    in mediump float vSharpness;
    in highp float vRadius; // Radius from vertex shader (needs precision for zero-check)

    out vec4 fragColor;

    void main() {
      // Discard zero-radius points (from nD slicing where points don't intersect hyperplane)
      if (vRadius < 0.0001) {
        discard;
      }

      // OPTIMIZATION: Use dot product for squared distance calculation
      vec2 centered = gl_PointCoord - 0.5;
      float r2 = dot(centered, centered);

      // OPTIMIZATION: Compare squared distances to avoid sqrt in discard check
      if (r2 > 0.25) {
        discard;
      }

      // OPTIMIZATION: sqrt(4.0 * r2) combines sqrt and multiply into one operation
      // normalizedR is in 0-1 range (gl_PointCoord is 0-1, centered is -0.5 to 0.5)
      mediump float normalizedR = sqrt(4.0 * r2);

      // Simple power function for falloff - modern GPUs optimize pow() well
      mediump float falloff = pow(max(1.0 - normalizedR, 0.0), vSharpness);

      // Per-node GOG (Gain-Offset-Gamma) color adjustment
      mediump vec3 adjusted = vColor * uIntensity + uOffset;
      adjusted = max(adjusted, vec3(0.0));

      // Early discard for zero-contribution fragments after offset
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;

      mediump vec3 finalColor = pow(adjusted, vec3(invGamma));

      // Calculate alpha (intensity) for additive blending
      mediump float alpha = baseAlpha * falloff * opacity;

      // Output final color with alpha for AdditiveBlending (SrcAlpha, One)
      fragColor = vec4(finalColor, alpha);
    }
  `;

  /**
   * Create a new PointMaterial with the specified configuration
   */
  constructor(materialConfig: PointMaterialConfig = {}) {
    const gammaValue = Math.max(0.001, materialConfig.gamma ?? 1.0); // Prevent division by zero
    // Default values for initial computation
    const defaultFov = (60 * Math.PI) / 180;
    const defaultResolutionY = 1080;
    const defaultTanHalfFov = Math.tan(defaultFov / 2);

    super({
      uniforms: {
        // Color uniforms
        baseAlpha: { value: config.shader.points.baseAlpha },
        opacity: { value: materialConfig.opacity ?? 1.0 },
        invGamma: { value: 1.0 / gammaValue }, // Pre-computed inverse for performance
        uIntensity: { value: materialConfig.intensity ?? 1.0 },
        uOffset: { value: materialConfig.offset ?? 0.0 },

        // OPTIMIZED camera uniforms - pre-computed for shader performance
        // pointSizeFactor = 2.0 * resolution.y / tan(fov/2)
        pointSizeFactor: { value: (2.0 * defaultResolutionY) / defaultTanHalfFov },
        maxPointSize: { value: defaultResolutionY * 0.5 }, // resolution.y * 0.5

        // Radius and sharpness scaling for dtype normalization
        radiusScale: { value: materialConfig.radiusScale ?? 1.0 }, // Default 1.0 (no scaling)
        sharpnessScale: { value: materialConfig.sharpnessScale ?? 1.0 }, // Default 1.0 (no scaling)
      },

      // Shader source
      vertexShader: PointMaterial.VERTEX_SHADER,
      fragmentShader: PointMaterial.FRAGMENT_SHADER,

      // GLSL ES 3.0 for consistency with other materials
      glslVersion: THREE.GLSL3,

      // Material properties
      vertexColors: true, // Enable per-vertex colors
      transparent: materialConfig.transparent ?? true, // Enable transparency for blending (false for opaque)
      depthWrite: materialConfig.depthWrite ?? false, // Usually false for additive blending
      depthTest: materialConfig.depthTest ?? true, // Default true; additive mode sets false
      toneMapped: false, // HDR values pass through to post-processing
      blending: materialConfig.blending ?? THREE.AdditiveBlending,
    });

    // Store gamma in userData for clone() method
    this.userData.gamma = gammaValue;
    this.userData.depthTest = materialConfig.depthTest ?? true;
  }

  /**
   * Update camera parameters for world-space point sizing
   * Pre-computes pointSizeFactor and maxPointSize for shader performance
   */
  updateCameraParams(fov: number, resolution: THREE.Vector2): void {
    const tanHalfFov = Math.tan(fov / 2);
    // Pre-compute values that were previously computed per-vertex in shader
    // pointSizeFactor = 2.0 * resolution.y / tan(fov/2)
    this.uniforms.pointSizeFactor.value = (2.0 * resolution.y) / tanHalfFov;
    // maxPointSize = resolution.y * 0.5 (hardware limit)
    this.uniforms.maxPointSize.value = resolution.y * 0.5;
  }

  /**
   * Update opacity
   */
  updateOpacity(opacity: number): void {
    this.uniforms.opacity.value = opacity;
  }

  /**
   * Update gamma correction
   * Only invGamma is used in shader; gamma value stored in userData for clone()
   */
  updateGamma(gamma: number): void {
    const safeGamma = Math.max(0.001, gamma); // Prevent division by zero
    this.userData.gamma = safeGamma; // Store for clone() method
    this.uniforms.invGamma.value = 1.0 / safeGamma;
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
   * Update radius scale for dtype normalization
   * Use 1/255 for uint8 radii, 1.0 for float radii
   */
  updateRadiusScale(scale: number): void {
    this.uniforms.radiusScale.value = scale;
  }

  /**
   * Update sharpness scale for dtype normalization
   * Use 1/255 for uint8 sharpness, 1.0 for float sharpness
   */
  updateSharpnessScale(scale: number): void {
    this.uniforms.sharpnessScale.value = scale;
  }

  /**
   * Clone this material with optional config overrides
   * Override base class clone to return PointMaterial type
   */
  clone(): this {
    const cloned = new PointMaterial({
      opacity: this.uniforms.opacity.value,
      gamma: this.userData.gamma ?? 1.0, // gamma stored in userData, not uniforms
      intensity: this.uniforms.uIntensity.value,
      offset: this.uniforms.uOffset.value,
      blending: this.blending,
      depthWrite: this.depthWrite,
      depthTest: this.userData.depthTest ?? true, // depthTest stored in userData
      transparent: this.transparent,
    });

    // Copy blend equation settings for custom blending (max mode)
    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
    }

    // Copy current uniform values
    cloned.uniforms.baseAlpha.value = this.uniforms.baseAlpha.value;
    cloned.uniforms.pointSizeFactor.value = this.uniforms.pointSizeFactor.value;
    cloned.uniforms.maxPointSize.value = this.uniforms.maxPointSize.value;
    cloned.uniforms.invGamma.value = this.uniforms.invGamma.value;

    return cloned as this;
  }

  /**
   * Dispose this material and unregister from MaterialManager
   * This prevents memory leaks by removing the material from global update lists
   */
  dispose(): void {
    // Unregister from material manager to prevent memory leaks
    // This removes the material from global update lists and cache
    materialManager.unregister(this);

    // Call parent dispose to free GPU resources (shaders, uniforms)
    super.dispose();
  }
}
