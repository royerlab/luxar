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
    
    varying vec3 vColor;
    varying float vSharpness;
    
    void main() {
      // Pass vertex color to fragment shader
      vColor = color;
      
      // Use 2.0 as default sharpness if not provided or zero
      vSharpness = sharpness > 0.0 ? sharpness : 2.0;
      
      // Transform vertex position from world space to view space
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      
      // CORRECT world-space point sizing formula
      // This ensures two points with radius r at distance 2r will just touch
      float distance = length(mvPosition.xyz);
      float basePointSize = 2.0 * radius * resolution.y / (distance * tan(fov * 0.5));
      
      // Sharpness compensation based on visibility threshold
      // For falloff function f(r) = (1-r)^s, the visible radius where intensity drops to 1% is:
      // r_vis = 1 - 0.01^(1/s)
      // We need to scale the point size by 1/r_vis to maintain consistent visible size
      // Using approximation: compensation = 1.0 + (s - 1.0) * 0.15
      // This gives: s=1→1.0, s=2→1.15, s=4→1.45, s=8→2.05
      float sharpnessCompensation = 1.0 + (vSharpness - 1.0) * 0.15;
      float pointSize = basePointSize * sharpnessCompensation;
      
      // Clamp to hardware limits
      gl_PointSize = clamp(pointSize, 1.0, resolution.y * 0.5);
    }
  `;
  
  // Static fragment shader with proper HDR handling
  private static readonly FRAGMENT_SHADER = /* glsl */ `
    uniform float hdrMultiplier;
    uniform float opacity;
    uniform float gamma;
    uniform float baseAlpha;
    varying vec3 vColor;
    varying float vSharpness;
    
    void main() {
      // Calculate distance from center of point sprite (0.0 to 0.5)
      float r = length(gl_PointCoord - 0.5);
      
      // Discard pixels outside the circular area
      if (r > 0.5) {
        discard;
      }
      
      // Normalize radius to 0-1 range for the visible circle
      float normalizedR = r * 2.0;  // Now 0.0 at center, 1.0 at edge
      
      // Variable falloff controlled by per-point sharpness
      float falloff = pow(1.0 - normalizedR, vSharpness);
      
      // Apply HDR multiplier FIRST (before any other color transformations)
      vec3 hdrColor = vColor * hdrMultiplier;
      
      // Then apply gamma correction
      vec3 finalColor = pow(hdrColor, vec3(1.0 / gamma));
      
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
    super({
      uniforms: {
        // HDR and color uniforms
        hdrMultiplier: { value: config.shader.points.hdrMultiplier },
        baseAlpha: { value: config.shader.points.baseAlpha },
        opacity: { value: materialConfig.opacity ?? 1.0 },
        gamma: { value: materialConfig.gamma ?? 1.0 },
        
        // Camera uniforms for world-space sizing
        fov: { value: (60 * Math.PI) / 180 }, // Default 60 degrees in radians
        resolution: { value: new THREE.Vector2(1, 1) }, // Will be updated
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
    this.uniforms.gamma.value = gamma;
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
    
    // Copy other properties if they exist
    if ('renderOrder' in this) {
      (cloned as any).renderOrder = (this as any).renderOrder;
    }
    
    return cloned as this;
  }
}