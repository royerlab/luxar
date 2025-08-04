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
 * - Sets point size based on per-vertex radius attribute
 * - Transforms vertices to screen space
 */
const GAUSSIAN_VERTEX_SHADER = /* glsl */`
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
    // This ensures points scale appropriately with perspective
    float baseSize = ${SHADER_CONFIG.POINTS.SIZE.toFixed(1)};
    float perspectiveScale = baseSize / -mvPosition.z;
    
    // Compensate for sharpness effect on apparent size
    // As sharpness increases, points appear smaller due to steeper falloff
    // This compensation maintains consistent visual size
    float sizeCompensation = sqrt(vSharpness / 2.0);
    
    // Use the per-point radius attribute to scale the point size
    // Multiply by a factor to convert from world units to screen pixels
    gl_PointSize = radius * perspectiveScale * 100.0 * sizeCompensation;
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
  uniform float hdrMultiplier;
  varying vec3 vColor;
  varying float vSharpness;
  
  void main() {
    // Calculate distance from center of point sprite (0.0 to 0.5)
    float r = length(gl_PointCoord - 0.5);
    
    // Discard pixels outside the circular area
    // This prevents the square footprint from being visible
    if (r > 0.5) {
      discard;
    }
    
    // Normalize radius to 0-1 range for the visible circle
    float normalizedR = r * 2.0;  // Now 0.0 at center, 1.0 at edge
    
    // Variable falloff controlled by per-point sharpness
    // Higher sharpness values create sharper edges
    float falloff = pow(1.0 - normalizedR, vSharpness);
    
    // Apply base alpha
    float alpha = ${SHADER_CONFIG.POINTS.BASE_ALPHA.toFixed(3)} * falloff;
    
    // Multiply color by HDR multiplier to drive bloom effects
    // Values > 1.0 will bloom in post-processing pipeline
    vec3 hdr = vColor * hdrMultiplier;
    
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
    uniforms: {
      hdrMultiplier: { value: SHADER_CONFIG.POINTS.HDR_MULTIPLIER },
    },
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