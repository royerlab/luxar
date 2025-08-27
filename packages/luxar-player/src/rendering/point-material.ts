/**
 * Point Material for Luxar
 *
 * Specialized THREE.ShaderMaterial for physically accurate point cloud rendering.
 * Features world-space sizing, HDR colors, and per-point sharpness control.
 */

import * as THREE from 'three';
import { config } from '../config';

/**
 * Configuration for point material creation
 */
export interface PointMaterialConfig {
  opacity?: number;
  gamma?: number;
  blending?: THREE.Blending;
  depthWrite?: boolean;
  radiusScale?: number; // Scale factor for radius normalization (e.g., 1/255 for uint8)
  sharpnessScale?: number; // Scale factor for sharpness normalization (e.g., 1/255 for uint8)
}

/**
 * Point cloud material with physically accurate world-space sizing.
 * Extends THREE.ShaderMaterial to provide specialized point rendering.
 */
export class PointMaterial extends THREE.ShaderMaterial {
  // Static vertex shader with CORRECT world-space sizing formula
  private static readonly VERTEX_SHADER = /* glsl */ `
    attribute float radius;
    attribute float sharpness;
    uniform float fov;
    uniform vec2 resolution;
    uniform float radiusScale;
    uniform float sharpnessScale;
    
    varying vec3 vColor;
    varying float vSharpness;
    varying float vRadius; // Pass radius to fragment for zero-check
    
    void main() {
      // Pass vertex color to fragment shader
      vColor = color;
      
      // Apply sharpness scale for dtype normalization and use 2.0 as default
      float normalizedSharpness = sharpness * sharpnessScale;
      vSharpness = normalizedSharpness > 0.0 ? normalizedSharpness : 2.0;
      
      // Transform vertex position from world space to view space
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      
      // CORRECT world-space point sizing formula
      // This ensures two points with radius r at distance 2r will just touch
      float distance = length(mvPosition.xyz);
      // Apply radius scale for dtype normalization (e.g., uint8 needs 1/255 scale)
      float normalizedRadius = radius * radiusScale;
      vRadius = normalizedRadius; // Pass to fragment shader
      
      // Calculate base point size
      float basePointSize = 2.0 * normalizedRadius * resolution.y / (distance * tan(fov * 0.5));
      
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
      gl_PointSize = max(1.0, min(pointSize, resolution.y * 0.5));
    }
  `;

  // Optimized fragment shader with simple, effective optimizations
  private static readonly FRAGMENT_SHADER = /* glsl */ `
    uniform float hdrMultiplier;
    uniform float opacity;
    uniform float gamma;
    uniform float baseAlpha;
    uniform float invGamma; // Pre-computed 1/gamma for performance
    varying vec3 vColor;
    varying float vSharpness;
    varying float vRadius; // Radius from vertex shader
    
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
      
      // Calculate actual radius for falloff (single sqrt operation)
      float r = sqrt(r2);
      float normalizedR = r * 2.0; // Normalize to 0-1 range
      
      // Simple power function for falloff - modern GPUs optimize pow() well
      float falloff = pow(max(1.0 - normalizedR, 0.0), vSharpness);
      
      // Apply HDR multiplier
      vec3 hdrColor = vColor * hdrMultiplier;
      
      // Apply gamma correction using pre-computed inverse
      vec3 finalColor = pow(hdrColor, vec3(invGamma));
      
      // Calculate final alpha
      float alpha = baseAlpha * falloff * opacity;
      
      // Output final color with alpha
      gl_FragColor = vec4(finalColor, alpha);
    }
  `;

  /**
   * Create a new PointMaterial with the specified configuration
   */
  constructor(materialConfig: PointMaterialConfig = {}) {
    const gammaValue = Math.max(0.001, materialConfig.gamma ?? 1.0); // Prevent division by zero
    super({
      uniforms: {
        // HDR and color uniforms
        hdrMultiplier: { value: config.shader.points.hdrMultiplier },
        baseAlpha: { value: config.shader.points.baseAlpha },
        opacity: { value: materialConfig.opacity ?? 1.0 },
        gamma: { value: gammaValue },
        invGamma: { value: 1.0 / gammaValue }, // Pre-computed inverse for performance

        // Camera uniforms for world-space sizing
        fov: { value: (60 * Math.PI) / 180 }, // Default 60 degrees in radians
        resolution: { value: new THREE.Vector2(1, 1) }, // Will be updated immediately

        // Radius and sharpness scaling for dtype normalization
        radiusScale: { value: materialConfig.radiusScale ?? 1.0 }, // Default 1.0 (no scaling)
        sharpnessScale: { value: materialConfig.sharpnessScale ?? 1.0 }, // Default 1.0 (no scaling)
      },

      // Shader source
      vertexShader: PointMaterial.VERTEX_SHADER,
      fragmentShader: PointMaterial.FRAGMENT_SHADER,

      // Material properties
      vertexColors: true, // Enable per-vertex colors
      transparent: true, // Enable transparency for blending
      depthWrite: materialConfig.depthWrite ?? false, // Usually false for additive blending
      toneMapped: false, // HDR values pass through to post-processing
      blending: materialConfig.blending ?? THREE.AdditiveBlending,
    });
  }

  /**
   * Update camera parameters for world-space point sizing
   */
  updateCameraParams(fov: number, resolution: THREE.Vector2): void {
    this.uniforms.fov.value = fov;
    // Copy values to avoid reference issues
    this.uniforms.resolution.value.copy(resolution);
  }

  /**
   * Update HDR multiplier
   */
  updateHDRMultiplier(multiplier: number): void {
    this.uniforms.hdrMultiplier.value = multiplier;
  }

  /**
   * Update opacity
   */
  updateOpacity(opacity: number): void {
    this.uniforms.opacity.value = opacity;
  }

  /**
   * Update gamma correction
   */
  updateGamma(gamma: number): void {
    const safeGamma = Math.max(0.001, gamma); // Prevent division by zero
    this.uniforms.gamma.value = safeGamma;
    this.uniforms.invGamma.value = 1.0 / safeGamma;
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
      gamma: this.uniforms.gamma.value,
      blending: this.blending,
      depthWrite: this.depthWrite,
    });

    // Copy current uniform values
    cloned.uniforms.hdrMultiplier.value = this.uniforms.hdrMultiplier.value;
    cloned.uniforms.baseAlpha.value = this.uniforms.baseAlpha.value;
    cloned.uniforms.fov.value = this.uniforms.fov.value;
    cloned.uniforms.resolution.value.copy(this.uniforms.resolution.value);
    cloned.uniforms.invGamma.value = this.uniforms.invGamma.value;

    return cloned as this;
  }
}
