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
import { LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER } from './shaders/line-shaders';
import type { CameraAwareMaterial } from './camera-aware-material';
import {
  applyColormapTextureToMaterial,
  applyScalarRangeToMaterial,
} from './material-colormap-helpers';
import {
  isAdditiveMode,
  isLuminousMode,
  isMaxMode,
  isNormalMode,
  isOpaqueMode,
} from './blending-state';

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
  /** Colormap texture for scalar-to-color mapping (256x1 RGB) */
  colormapTexture?: THREE.DataTexture;
  /** Scalar data range [min, max] for normalization before LUT lookup */
  scalarRange?: [number, number];
}

/**
 * Line material uniforms interface
 *
 * @internal — reserved extension shape; no current consumer.
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
export class LineMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  /**
   * Create a new LineMaterial with the specified configuration.
   *
   * @param materialConfig - Material configuration options
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
        uFOV: { value: (60 * Math.PI) / 180 }, // Default 60° FOV (or frustumHeight for ortho)
        uResolution: { value: new THREE.Vector2(1, 1) },
        uIsOrtho: { value: 0 }, // 0 = perspective, 1 = orthographic
        uOpacity: { value: materialConfig.opacity ?? 1.0 },
        uInvGamma: { value: 1.0 / gammaValue }, // Pre-computed inverse for performance
        uIntensity: { value: materialConfig.intensity ?? 1.0 },
        uOffset: { value: materialConfig.offset ?? 0.0 },
        // near-plane safety + max-pixel-width clamp uniforms.
        uNearCull: { value: 0.05 },
        uMaxLinePixelWidth: { value: 540 }, // ≈ resolution.y * 0.5 default; updated in updateCameraParams
        // Colormap uniforms (only when USE_COLORMAP define is set)
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

      vertexShader: LINE_VERTEX_SHADER,
      fragmentShader: LINE_FRAGMENT_SHADER,

      // Preprocessor defines
      defines: {
        ...(materialConfig.colormapTexture ? { USE_COLORMAP: '' } : {}),
      },

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

    // Apply mode-specific blending state via the canonical method —
    // same path used by live mode updates from the layers panel.
    this.applyBlendingMode(blendingMode);

    // Honor explicit overrides from config after mode-derived defaults.
    if (materialConfig.transparent !== undefined) {
      this.transparent = materialConfig.transparent;
    }
    if (materialConfig.depthTest !== undefined) {
      this.depthTest = materialConfig.depthTest;
    }

    // gamma + scalarRange in userData for clone(); blendingMode and
    // depthTest are already set by applyBlendingMode.
    this.userData.gamma = gammaValue;
    this.userData.depthTest = materialConfig.depthTest ?? !isAdditive;
    this.userData.scalarRange = materialConfig.scalarRange;
  }

  /**
   * Update camera parameters for world-space line sizing.
   *
   * @param fov - Field of view in radians
   * @param resolution - Viewport resolution
   */
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.uniforms.uFOV.value = fov; // FOV in radians (perspective) or frustumHeight (ortho)
    this.uniforms.uResolution.value.copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    // Apply the near-plane safety distance when provided.
    if (nearCull !== undefined && nearCull > 0) {
      this.uniforms.uNearCull.value = nearCull;
    }
    // clamp screen-space line width to half the viewport height so a
    // near-camera segment can't paint the entire screen.
    this.uniforms.uMaxLinePixelWidth.value = Math.max(2, resolution.y * 0.5);
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
   * Update the colormap texture and enable/disable colormap mode.
   */
  updateColormapTexture(texture: THREE.DataTexture | null): void {
    const { wasEnabled, nowEnabled } = applyColormapTextureToMaterial(this, texture);
    if (wasEnabled !== nowEnabled) {
      this.needsUpdate = true;
    }
  }

  /**
   * Set the scalar data range for colormap normalization.
   */
  updateScalarRange(min: number, max: number): void {
    applyScalarRangeToMaterial(this, min, max);
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
      colormapTexture: this.uniforms.uColormapTex?.value ?? undefined,
      scalarRange: this.userData.scalarRange ?? undefined,
    });

    // Copy blend equation settings for custom blending (max mode)
    if (this.blending === THREE.CustomBlending) {
      cloned.blendEquation = this.blendEquation;
      cloned.blendSrc = this.blendSrc;
      cloned.blendDst = this.blendDst;
    }

    cloned.uniforms.uFOV.value = this.uniforms.uFOV.value;
    cloned.uniforms.uResolution.value.copy(this.uniforms.uResolution.value);
    // Preserve orthographic state and the near-plane / max-pixel-width
    // clamp uniforms.
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uMaxLinePixelWidth.value = this.uniforms.uMaxLinePixelWidth.value;
    cloned.uniforms.uInvGamma.value = this.uniforms.uInvGamma.value;

    return cloned as this;
  }

  // Dispose is inherited from THREE.ShaderMaterial. The MaterialManager
  // subscribes to the synchronous `dispose` event THREE fires from
  // super.dispose(), so registry cleanup happens automatically without
  // this file needing to import the manager (which would create a cycle).

  /**
   * Apply a blending mode to this material in-place.
   *
   * Lines use `THREE.AdditiveBlending` (SrcAlpha factors) for
   * additive/luminous because the per-pixel intensity-squaring concern
   * that GSplats face doesn't apply to thin line segments. Only `max`
   * mode goes through CustomBlending. After this returns,
   * `userData.blendingMode` reflects the live mode so subsequent
   * `clone()` calls preserve it.
   */
  applyBlendingMode(mode: 'additive' | 'normal' | 'max' | 'opaque' | 'luminous'): void {
    const previousMode = this.userData.blendingMode as
      | 'additive'
      | 'normal'
      | 'max'
      | 'opaque'
      | 'luminous'
      | undefined;
    // F.2: predicates over BlendingMode replace inline string comparisons.
    const isOpaque = isOpaqueMode(mode);
    const isAdditive = isAdditiveMode(mode);
    const isMax = isMaxMode(mode);
    const opacity = (this.uniforms.uOpacity?.value as number | undefined) ?? 1.0;

    // Defensive: THREE may leave defines undefined when none were
    // passed at construction.
    if (!this.defines) this.defines = {};

    if (isOpaque || isNormalMode(mode)) {
      this.blending = THREE.NormalBlending;
    } else if (isAdditive || isLuminousMode(mode)) {
      this.blending = THREE.AdditiveBlending;
    } else if (isMax) {
      this.blending = THREE.CustomBlending;
    } else {
      this.blending = THREE.NormalBlending;
    }

    this.transparent = !isOpaque;
    this.depthWrite = isOpaque || (isNormalMode(mode) && opacity >= 0.99);
    this.depthTest = !isAdditive;

    // gate fragment LUXAR_MAX_RGB_CONTRIBUTION on max mode so the
    // shader premultiplies RGB by intensity*opacity (necessary for
    // OneFactor blend factors to capture contribution-weighted max).
    const wantsContrib = isMax;
    const hasContrib = 'LUXAR_MAX_RGB_CONTRIBUTION' in this.defines;
    let definesChanged = false;
    if (wantsContrib && !hasContrib) {
      this.defines.LUXAR_MAX_RGB_CONTRIBUTION = '';
      definesChanged = true;
    } else if (!wantsContrib && hasContrib) {
      delete this.defines.LUXAR_MAX_RGB_CONTRIBUTION;
      definesChanged = true;
    }

    if (isMax) {
      this.blendEquation = THREE.MaxEquation;
      this.blendSrc = THREE.OneFactor;
      this.blendDst = THREE.OneFactor;
    } else {
      // Reset CustomBlending state so a switch out of max doesn't strand
      // MaxEquation. AdditiveBlending and NormalBlending ignore these.
      this.blendEquation = THREE.AddEquation;
      this.blendSrc = THREE.SrcAlphaFactor;
      this.blendDst = THREE.OneMinusSrcAlphaFactor;
    }

    this.userData.blendingMode = mode;
    this.userData.depthTest = this.depthTest;

    // Only mark needsUpdate when something changed that the GPU side
    // actually cares about.
    if (definesChanged || previousMode !== mode) {
      this.needsUpdate = true;
    }
  }
}
