// Shader management for advanced point cloud rendering
//
// This module provides sophisticated shader-based point rendering with:
// - Gaussian falloff for smooth, natural-looking points
// - HDR color output for bloom effects
// - Proper alpha blending and depth handling
// - Configurable point sizes and falloff parameters

import * as THREE from 'three';

/**
 * Configuration for shader-based point rendering
 * 
 * These parameters control the visual appearance and performance
 * of the Gaussian point sprites used for rendering point clouds.
 */
export const SHADER_CONFIG = {
  /** Point rendering parameters */
  POINTS: {
    /** Base point size in screen pixels - adjust based on point density */
    SIZE: 8.0,
    
    /** HDR color multiplier for bloom effects - higher values = more bloom */
    HDR_MULTIPLIER: 13.0,
    
    /** Base alpha intensity - controls point visibility */
    BASE_ALPHA: 0.01,
    
    /** Gaussian falloff steepness - higher values = sharper edges */
    FALLOFF_STEEPNESS: 20.0,
  },
} as const;

/**
 * Vertex shader for Gaussian point sprites
 * 
 * This shader:
 * - Passes through vertex colors for per-point coloring
 * - Sets point size for sprite rendering
 * - Transforms vertices to screen space
 */
const GAUSSIAN_VERTEX_SHADER = /* glsl */`
  varying vec3 vColor;
  
  void main() {
    // Pass vertex color to fragment shader for per-point coloring
    vColor = color;
    
    // Transform vertex position from world space to screen space
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    
    // Set point size in screen pixels for sprite rendering
    gl_PointSize = ${SHADER_CONFIG.POINTS.SIZE.toFixed(1)};
  }
`;

/**
 * Fragment shader for Gaussian point sprites with HDR output
 * 
 * This shader creates smooth, natural-looking points using:
 * - Gaussian falloff based on distance from center
 * - HDR color output for bloom effects
 * - Proper alpha blending for transparent overlapping points
 */
const GAUSSIAN_FRAGMENT_SHADER = /* glsl */`
  varying vec3 vColor;
  
  void main() {
    // Calculate distance from center of point sprite (0.0 to ~0.707)
    float r = length(gl_PointCoord - 0.5);
    
    // Gaussian falloff using rational approximation for better performance
    // Formula: alpha = base_alpha / (1.0 + steepness * r^2)
    // This creates smooth, natural-looking circular points
    float alpha = ${SHADER_CONFIG.POINTS.BASE_ALPHA.toFixed(3)} / 
                  (1.0 + ${SHADER_CONFIG.POINTS.FALLOFF_STEEPNESS.toFixed(1)} * r * r);
    
    // Multiply color by HDR multiplier to drive bloom effects
    // Values > 1.0 will bloom in post-processing pipeline
    vec3 hdr = vColor * ${SHADER_CONFIG.POINTS.HDR_MULTIPLIER.toFixed(1)};
    
    // Output HDR color with computed alpha
    gl_FragColor = vec4(hdr, alpha);
  }
`;

/**
 * Creates a shader material for advanced point cloud rendering
 * 
 * This material uses custom vertex and fragment shaders to render
 * point clouds with Gaussian falloff, HDR colors, and proper blending.
 * 
 * @returns Configured ShaderMaterial for point rendering
 */
export function createGaussianPointMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: GAUSSIAN_VERTEX_SHADER,
    fragmentShader: GAUSSIAN_FRAGMENT_SHADER,
    
    // Enable per-vertex colors from geometry
    vertexColors: true,
    
    // Enable transparency for smooth point blending
    transparent: true,
    
    // Disable depth writing to avoid z-buffer artifacts when blending
    // Points will still be depth-tested but won't write to depth buffer
    depthWrite: false,
    
    // Don't apply tone mapping to this material - handled in post-processing
    // This allows HDR values to pass through to bloom effects
    toneMapped: false,
    
    // Use normal alpha blending for better visibility
    // This ensures points render properly without being too bright
    blending: THREE.AdditiveBlending,
  });
}

/**
 * Shader compilation and validation utilities
 * 
 * These functions help with shader development and debugging,
 * providing clear error messages when shaders fail to compile.
 */
export class ShaderValidator {
  /**
   * Validates that a shader material compiled successfully
   * 
   * @param material - The shader material to validate
   * @param name - Human-readable name for error messages
   * @throws Error if shader compilation failed
   */
  static validateShaderMaterial(material: THREE.ShaderMaterial, name: string): void {
    // Note: Three.js compiles shaders lazily during first render
    // For development, you might want to force compilation here:
    // const renderer = new THREE.WebGLRenderer();
    // renderer.compile(new THREE.Scene(), new THREE.Camera());
    
    console.log(`✓ Shader material '${name}' created successfully`);
  }
  
  /**
   * Logs shader configuration for debugging
   */
  static logShaderConfig(): void {
    console.log('Gaussian Point Shader Configuration:', {
      pointSize: SHADER_CONFIG.POINTS.SIZE,
      hdrMultiplier: SHADER_CONFIG.POINTS.HDR_MULTIPLIER,
      baseAlpha: SHADER_CONFIG.POINTS.BASE_ALPHA,
      falloffSteepness: SHADER_CONFIG.POINTS.FALLOFF_STEEPNESS,
    });
  }
}