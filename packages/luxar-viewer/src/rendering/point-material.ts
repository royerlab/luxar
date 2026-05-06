/**
 * Point Material for Luxar
 *
 * Specialized THREE.ShaderMaterial for physically accurate points rendering.
 * Features world-space sizing, HDR colors, and per-point sharpness control.
 */

import * as THREE from 'three';
import { POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER } from './shaders/point-shaders';
import type { CameraAwareMaterial } from './camera-aware-material';

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
export class PointMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
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
      vertexShader: POINT_VERTEX_SHADER,
      fragmentShader: POINT_FRAGMENT_SHADER,

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
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    _nearCull?: number
  ): void {
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    if (isOrtho) {
      // fov carries frustumHeight in world units for ortho.
      // Perspective precomputes 2*res.y / tan(fov/2), then shader divides by distance.
      // tan(fov/2) = frustumHeight / (2*distance), so the effective factor at the
      // matching distance is 2*res.y / (frustumHeight/2) = 4*res.y / frustumHeight.
      // Since ortho has invDistance=1, we bake that full factor here.
      const halfFrustum = fov * 0.5;
      this.uniforms.pointSizeFactor.value = (2.0 * resolution.y) / halfFrustum;
    } else {
      const tanHalfFov = Math.tan(fov / 2);
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

  // Note: `dispose()` is inherited from THREE.ShaderMaterial.
  // MaterialManager subscribes to the synchronous `dispose` event the
  // base class fires, so the material is removed from the registry +
  // cache automatically. No explicit unregister callback needed here,
  // which keeps this file out of the manager's import graph.
}
