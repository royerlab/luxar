/**
 * Point Material for Luxar
 *
 * Specialized THREE.ShaderMaterial for physically accurate points rendering.
 * Features world-space sizing, HDR colors, and per-point sharpness control.
 */

import * as THREE from 'three';
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
  colormapTexture?: THREE.DataTexture; // Colormap LUT texture (256x1 RGB)
  scalarRange?: [number, number]; // Scalar data range [min, max] for normalization
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
  //
  // IMPORTANT: Do NOT declare `in vec3 color;` here! When vertexColors is true,
  // THREE.js automatically injects `in vec3 color;` into the shader. Declaring it
  // manually causes a "'color' : redefinition" shader compilation error that silently
  // breaks all point rendering. The vertexColors flag in the constructor controls this.
  private static readonly VERTEX_SHADER = /* glsl */ `
    precision highp float;

    in float radius;
    in float sharpness;
    // NOTE: "in vec3 color" is auto-injected by THREE.js when vertexColors=true (see constructor).
    // In colormap mode, vertexColors=false so "color" is not available — use scalar + LUT instead.
    #ifdef USE_COLORMAP
    in float scalar;              // Per-point scalar for colormap lookup
    uniform sampler2D uColormapTex;   // 256x1 LUT texture
    uniform float uScalarMin;         // Scalar range minimum
    uniform float uScalarScale;       // 1.0 / (max - min)
    #endif
    uniform float pointSizeFactor; // Pre-computed: 2.0 * resolution.y / tanHalfFov (or resolution.y / frustumHeight for ortho)
    uniform float maxPointSize;    // Pre-computed: resolution.y * 0.5
    uniform float radiusScale;
    uniform float sharpnessScale;
    uniform int uIsOrtho;          // 0 = perspective, 1 = orthographic

    out mediump vec3 vColor;
    out mediump float vSharpness;
    out highp float vRadius; // Pass radius to fragment for zero-check (needs precision)

    void main() {
      // Pass vertex color — either from attribute or colormap LUT
      #ifdef USE_COLORMAP
      float t = clamp((scalar - uScalarMin) * uScalarScale, 0.0, 1.0);
      vColor = texture(uColormapTex, vec2(t, 0.5)).rgb;
      #else
      vColor = color;
      #endif

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
      float invDistance = (uIsOrtho == 1) ? 1.0 : inversesqrt(dot(mvPosition.xyz, mvPosition.xyz));
      float basePointSize = normalizedRadius * pointSizeFactor * invDistance;

      // Sharpness compensation based on visibility threshold
      // For falloff function f(r) = (1-r)^s, the visible radius where intensity drops to 1% is:
      // r_vis = 1 - 0.01^(1/s)
      // We need to scale the point size by 1/r_vis to maintain consistent visible size
      // Exact formula: compensation = 1 / (1 - 0.01^(1/s)), guarded against s=0
      float sharpnessCompensation = 1.0 / (1.0 - pow(0.01, 1.0 / max(vSharpness, 0.01)));
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
      mediump float alpha = falloff * opacity;

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

        // Projection mode
        uIsOrtho: { value: 0 }, // 0 = perspective, 1 = orthographic

        // Colormap uniforms (only active when USE_COLORMAP define is set)
        ...(materialConfig.colormapTexture
          ? {
              uColormapTex: { value: materialConfig.colormapTexture },
              uScalarMin: { value: materialConfig.scalarRange?.[0] ?? 0.0 },
              uScalarScale: {
                value: materialConfig.scalarRange
                  ? 1.0 /
                    Math.max(1e-10, materialConfig.scalarRange[1] - materialConfig.scalarRange[0])
                  : 1.0,
              },
            }
          : {}),
      },

      // Shader source
      vertexShader: PointMaterial.VERTEX_SHADER,
      fragmentShader: PointMaterial.FRAGMENT_SHADER,

      // Preprocessor defines — USE_COLORMAP enables scalar attribute + LUT lookup
      defines: {
        ...(materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {}),
      },

      // GLSL ES 3.0 for consistency with other materials
      glslVersion: THREE.GLSL3,

      // Material properties
      vertexColors: !materialConfig.colormapTexture, // THREE.js injects `in vec3 color` when true; disable for colormap mode
      transparent: materialConfig.transparent ?? true, // Enable transparency for blending (false for opaque)
      depthWrite: materialConfig.depthWrite ?? false, // Usually false for additive blending
      depthTest: materialConfig.depthTest ?? true, // Default true; additive mode sets false
      toneMapped: false, // HDR values pass through to post-processing
      blending: materialConfig.blending ?? THREE.AdditiveBlending,
    });

    // Store in userData for clone() method
    this.userData.gamma = gammaValue;
    this.userData.depthTest = materialConfig.depthTest ?? true;
    this.userData.scalarRange = materialConfig.scalarRange;
  }

  /**
   * Update camera parameters for world-space point sizing
   * Pre-computes pointSizeFactor and maxPointSize for shader performance
   */
  updateCameraParams(fov: number, resolution: THREE.Vector2, isOrtho: boolean = false): void {
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    if (isOrtho) {
      // fov carries frustumHeight in world units for ortho
      this.uniforms.pointSizeFactor.value = resolution.y / fov;
    } else {
      const tanHalfFov = Math.tan(fov / 2);
      // pointSizeFactor = 2.0 * resolution.y / tan(fov/2)
      this.uniforms.pointSizeFactor.value = (2.0 * resolution.y) / tanHalfFov;
    }
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
   * Update the colormap texture and enable/disable colormap mode.
   */
  updateColormapTexture(texture: THREE.DataTexture | null): void {
    const wasEnabled = 'USE_COLORMAP' in this.defines;
    const nowEnabled = !!texture;

    if (nowEnabled) {
      this.defines.USE_COLORMAP = '';
      if (!this.uniforms.uColormapTex) {
        this.uniforms.uColormapTex = { value: texture };
        this.uniforms.uScalarMin = { value: 0.0 };
        this.uniforms.uScalarScale = { value: 1.0 };
      } else {
        this.uniforms.uColormapTex.value = texture;
      }
    } else {
      delete this.defines.USE_COLORMAP;
    }

    if (wasEnabled !== nowEnabled) {
      // vertexColors controls whether THREE.js injects `in vec3 color` into the shader.
      // Must be disabled for colormap mode (uses scalar + LUT instead of color attribute).
      this.vertexColors = !nowEnabled;
      this.needsUpdate = true; // Triggers shader recompilation
    }
  }

  /**
   * Set the scalar data range for colormap normalization.
   */
  updateScalarRange(min: number, max: number): void {
    if (this.uniforms.uScalarMin) {
      this.uniforms.uScalarMin.value = min;
    }
    if (this.uniforms.uScalarScale) {
      this.uniforms.uScalarScale.value = 1.0 / Math.max(1e-10, max - min);
    }
    this.userData.scalarRange = [min, max];
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
      colormapTexture: this.uniforms.uColormapTex?.value ?? undefined,
      scalarRange: this.userData.scalarRange ?? undefined,
    });

    // Copy blend equation settings for custom blending (max mode)
    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
    }

    // Copy current uniform values
    cloned.uniforms.pointSizeFactor.value = this.uniforms.pointSizeFactor.value;
    cloned.uniforms.maxPointSize.value = this.uniforms.maxPointSize.value;
    cloned.uniforms.invGamma.value = this.uniforms.invGamma.value;
    cloned.uniforms.radiusScale.value = this.uniforms.radiusScale.value;
    cloned.uniforms.sharpnessScale.value = this.uniforms.sharpnessScale.value;

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
