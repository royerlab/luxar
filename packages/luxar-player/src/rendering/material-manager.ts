/**
 * Material Manager for Luxar
 *
 * This module manages materials with different rendering properties,
 * implementing a caching strategy to minimize material switching overhead.
 */

import * as THREE from 'three';
import { SHADER_CONFIG } from './shader-manager';

// Supported blending modes
export type BlendingMode = 'normal' | 'additive' | 'multiply' | 'minimum' | 'maximum';

// Material properties for caching
interface MaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  gamma: number;
}

// Vertex shader with support for per-vertex attributes
const VERTEX_SHADER = /* glsl */ `
  attribute float radius;
  attribute float sharpness;
  varying vec3 vColor;
  varying float vSharpness;
  
  void main() {
    // Pass vertex color to fragment shader for per-point coloring
    vColor = color;
    
    // Pass sharpness to fragment shader for per-point falloff control
    vSharpness = sharpness;
    
    // Transform vertex position from world space to screen space
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // Calculate point size based on radius and distance from camera
    float baseSize = ${SHADER_CONFIG.POINTS.size.toFixed(1)};
    float perspectiveScale = baseSize / -mvPosition.z;
    
    // Compensate for sharpness effect on apparent size
    float sizeCompensation = sqrt(vSharpness / 2.0);
    
    // Use the per-point radius attribute to scale the point size
    gl_PointSize = radius * perspectiveScale * 100.0 * sizeCompensation;
  }
`;

// Fragment shader with opacity and gamma support
const FRAGMENT_SHADER = /* glsl */ `
  uniform float hdrMultiplier;
  uniform float opacity;
  uniform float gamma;
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
    
    // Apply base alpha
    float alpha = ${SHADER_CONFIG.POINTS.baseAlpha.toFixed(3)} * falloff * opacity;
    
    // Apply gamma correction to color
    vec3 gammaCorrected = pow(vColor, vec3(1.0 / gamma));
    
    // Multiply color by HDR multiplier to drive bloom effects
    vec3 hdr = gammaCorrected * hdrMultiplier;
    
    // Output HDR color with computed alpha
    gl_FragColor = vec4(hdr, alpha);
  }
`;

/**
 * Manages Three.js materials with caching for performance
 */
export class MaterialManager {
  private materialCache: Map<string, THREE.ShaderMaterial> = new Map();

  /**
   * Get or create a material with specified properties
   */
  getMaterial(props: MaterialProperties): THREE.ShaderMaterial {
    // Create cache key from properties
    const key = `${props.blendingMode}_${props.opacity.toFixed(2)}_${props.gamma.toFixed(2)}`;

    // Check cache first
    let material = this.materialCache.get(key);
    if (material) {
      return material;
    }

    // Create new material
    material = new THREE.ShaderMaterial({
      uniforms: {
        hdrMultiplier: { value: SHADER_CONFIG.POINTS.hdrMultiplier },
        opacity: { value: props.opacity },
        gamma: { value: props.gamma },
      },
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,

      // Enable per-vertex colors from geometry
      vertexColors: true,

      // Enable transparency for smooth point blending
      transparent: true,

      // Disable depth writing for transparent objects to avoid artifacts
      depthWrite: props.blendingMode === 'normal' && props.opacity >= 0.99,

      // Don't apply tone mapping to this material - handled in post-processing
      toneMapped: false,

      // Set blending mode based on property
      blending: this.getThreeBlending(props.blendingMode),
    });

    // Store render order in userData for later application to mesh/points
    material.userData.renderOrder = this.getRenderOrder(props.blendingMode, props.opacity);

    // Cache the material
    this.materialCache.set(key, material);

    console.log(`Created new material: ${key}`);
    return material;
  }

  /**
   * Convert our blending mode to Three.js blending constant
   */
  private getThreeBlending(mode: BlendingMode): THREE.Blending {
    switch (mode) {
      case 'normal':
        return THREE.NormalBlending;
      case 'additive':
        return THREE.AdditiveBlending;
      case 'multiply':
        return THREE.MultiplyBlending;
      case 'minimum':
        // Three.js doesn't have minimum blending, use subtractive as approximation
        return THREE.SubtractiveBlending;
      case 'maximum':
        // Three.js doesn't have maximum blending, use additive as approximation
        return THREE.AdditiveBlending;
      default:
        console.warn(`Unknown blending mode: ${mode}, using normal`);
        return THREE.NormalBlending;
    }
  }

  /**
   * Determine render order based on blending mode and opacity
   */
  private getRenderOrder(mode: BlendingMode, opacity: number): number {
    // Opaque objects render first (order 0)
    if (mode === 'normal' && opacity >= 0.99) {
      return 0;
    }

    // Transparent and special blend modes render later
    // Higher values render later
    switch (mode) {
      case 'normal':
        return 100; // Transparent normal blending
      case 'multiply':
        return 200; // Multiply needs to see what's behind
      case 'additive':
        return 300; // Additive on top
      case 'minimum':
      case 'maximum':
        return 400; // Special modes last
      default:
        return 100;
    }
  }

  /**
   * Update HDR multiplier for all cached materials
   */
  updateHDRMultiplier(multiplier: number): void {
    this.materialCache.forEach((material) => {
      material.uniforms.hdrMultiplier.value = multiplier;
    });
  }

  /**
   * Dispose all cached materials
   */
  dispose(): void {
    this.materialCache.forEach((material) => {
      material.dispose();
    });
    this.materialCache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.materialCache.size,
      keys: Array.from(this.materialCache.keys()),
    };
  }
}

// Global material manager instance
export const materialManager = new MaterialManager();
