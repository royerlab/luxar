/**
 * LuxarToneMappingEffect — Vendored tone mapping with Exposure-Offset-Gamma (EOG)
 *
 * This is a copy of pmndrs/postprocessing ToneMappingEffect (v6.38.0) with
 * three additional uniforms for global color adjustment applied BEFORE tone mapping:
 *
 *   1. Exposure (log2 stops): color * 2^exposure
 *   2. Offset (additive): color + offset
 *   3. Gamma (power curve): pow(color, 1/gamma)
 *
 * By vendoring the effect and injecting EOG into the same shader pass, we avoid
 * an additional full-screen render pass (zero extra bandwidth cost).
 *
 * @module rendering/luxar-tone-mapping-effect
 */

import {
  Effect,
  BlendFunction,
  ToneMappingMode,
  LuminancePass,
  AdaptiveLuminancePass,
  AdaptiveLuminanceMaterial,
} from 'postprocessing';
import { LinearMipmapLinearFilter, REVISION, Uniform, WebGLRenderTarget } from 'three';
import type { WebGLRenderer } from 'three';

// ============================================================================
// Vendored GLSL fragment shader (from pmndrs/postprocessing v6.38.0)
// Modified: EOG block injected before mainImage
// ============================================================================

const LUXAR_TONE_MAPPING_SHADER = /* glsl */ `
#include <tonemapping_pars_fragment>

uniform float whitePoint;
uniform float uExposure;
uniform float uGlobalOffset;
uniform float uGlobalGamma;

#if TONE_MAPPING_MODE == 2 || TONE_MAPPING_MODE == 3

uniform float middleGrey;

#if TONE_MAPPING_MODE == 3
uniform lowp sampler2D luminanceBuffer;
#else
uniform float averageLuminance;
#endif

vec3 Reinhard2ToneMapping(vec3 color) {
  color *= toneMappingExposure;
  float l = luminance(color);
#if TONE_MAPPING_MODE == 3
  float lumAvg = unpackRGBAToFloat(texture2D(luminanceBuffer, vec2(0.5)));
#else
  float lumAvg = averageLuminance;
#endif
  float lumScaled = (l * middleGrey) / max(lumAvg, 1e-6);
  float lumCompressed = lumScaled * (1.0 + lumScaled / (whitePoint * whitePoint));
  lumCompressed /= (1.0 + lumScaled);
  return clamp(lumCompressed * color, 0.0, 1.0);
}

#elif TONE_MAPPING_MODE == 4

#define A 0.15
#define B 0.50
#define C 0.10
#define D 0.20
#define E 0.02
#define F 0.30

vec3 Uncharted2Helper(const in vec3 x) {
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 Uncharted2ToneMapping(vec3 color) {
  color *= toneMappingExposure;
  return clamp(Uncharted2Helper(color) / Uncharted2Helper(vec3(whitePoint)), 0.0, 1.0);
}

#endif

void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  // === Luxar EOG: Exposure-Offset-Gamma (injected before tone mapping) ===
  vec3 color = inputColor.rgb;
  color = color * exp2(uExposure);
  color = max(color + uGlobalOffset, vec3(0.0));
  color = pow(color, vec3(1.0 / uGlobalGamma));

  // === Original pmndrs tone mapping ===
#if TONE_MAPPING_MODE == 2 || TONE_MAPPING_MODE == 3
  outputColor = vec4(Reinhard2ToneMapping(color), inputColor.a);
#elif TONE_MAPPING_MODE == 4
  outputColor = vec4(Uncharted2ToneMapping(color), inputColor.a);
#else
  outputColor = vec4(toneMapping(color), inputColor.a);
#endif
}
`;

// ============================================================================
// LuxarToneMappingEffect class (vendored from pmndrs ToneMappingEffect)
// ============================================================================

/**
 * Configuration for LuxarToneMappingEffect
 */
export interface LuxarToneMappingConfig {
  blendFunction?: BlendFunction;
  mode?: ToneMappingMode;
  resolution?: number;
  whitePoint?: number;
  middleGrey?: number;
  minLuminance?: number;
  averageLuminance?: number;
  adaptationRate?: number;
  /** Global exposure in log2 stops (default 0.0) */
  exposure?: number;
  /** Global additive offset (default 0.0) */
  globalOffset?: number;
  /** Global gamma correction (default 1.0) */
  globalGamma?: number;
}

/**
 * Vendored ToneMappingEffect with Exposure-Offset-Gamma controls.
 *
 * Extends the pmndrs ToneMappingEffect by injecting EOG adjustments before
 * tone mapping in the same shader pass. This avoids an extra full-screen
 * render pass that a separate BrightnessGamma effect would require.
 */
export class LuxarToneMappingEffect extends Effect {
  /** @internal Luminance pass for adaptive tone mapping */
  readonly luminancePass: LuminancePass;
  /** @internal Adaptive luminance pass */
  private adaptiveLuminancePass: AdaptiveLuminancePass;

  constructor(config: LuxarToneMappingConfig = {}) {
    const {
      blendFunction = BlendFunction.SRC,
      mode = ToneMappingMode.AGX,
      resolution = 256,
      whitePoint = 4.0,
      middleGrey = 0.6,
      minLuminance = 0.01,
      averageLuminance = 1.0,
      adaptationRate = 1.0,
      exposure = 0.0,
      globalOffset = 0.0,
      globalGamma = 1.0,
    } = config;

    super('LuxarToneMappingEffect', LUXAR_TONE_MAPPING_SHADER, {
      blendFunction,
      uniforms: new Map<string, Uniform<unknown>>([
        ['luminanceBuffer', new Uniform(null)],
        ['maxLuminance', new Uniform(whitePoint)], // Legacy alias
        ['whitePoint', new Uniform(whitePoint)],
        ['middleGrey', new Uniform(middleGrey)],
        ['averageLuminance', new Uniform(averageLuminance)],
        // Luxar EOG uniforms
        ['uExposure', new Uniform(exposure)],
        ['uGlobalOffset', new Uniform(globalOffset)],
        ['uGlobalGamma', new Uniform(Math.max(0.001, globalGamma))],
      ]),
    });

    // Luminance passes for adaptive mode (same as original pmndrs)
    this.renderTargetLuminance = new WebGLRenderTarget(1, 1, {
      minFilter: LinearMipmapLinearFilter,
      depthBuffer: false,
    });
    this.renderTargetLuminance.texture.generateMipmaps = true;
    this.renderTargetLuminance.texture.name = 'Luminance';

    this.luminancePass = new LuminancePass({
      renderTarget: this.renderTargetLuminance,
    });

    this.adaptiveLuminancePass = new AdaptiveLuminancePass(this.luminancePass.texture, {
      minLuminance,
      adaptationRate,
    });

    this.uniforms.get('luminanceBuffer')!.value = this.adaptiveLuminancePass.texture;
    this.resolution = resolution;
    this.mode = mode;
  }

  /** @internal Render target for luminance */
  private renderTargetLuminance: WebGLRenderTarget;

  // ======================================================================
  // Tone mapping mode (vendored from pmndrs)
  // ======================================================================

  get mode(): ToneMappingMode {
    return Number(this.defines.get('TONE_MAPPING_MODE')) as ToneMappingMode;
  }

  set mode(value: ToneMappingMode) {
    if (this.mode === value) return;

    const revision = REVISION.replace(/\D+/g, '');
    const cineonToneMapping =
      Number(revision) >= 168 ? 'CineonToneMapping(texel)' : 'OptimizedCineonToneMapping(texel)';

    this.defines.clear();
    this.defines.set('TONE_MAPPING_MODE', value.toFixed(0));

    switch (value) {
      case ToneMappingMode.LINEAR:
        this.defines.set('toneMapping(texel)', 'LinearToneMapping(texel)');
        break;
      case ToneMappingMode.REINHARD:
        this.defines.set('toneMapping(texel)', 'ReinhardToneMapping(texel)');
        break;
      case ToneMappingMode.CINEON:
      case ToneMappingMode.OPTIMIZED_CINEON:
        this.defines.set('toneMapping(texel)', cineonToneMapping);
        break;
      case ToneMappingMode.ACES_FILMIC:
        this.defines.set('toneMapping(texel)', 'ACESFilmicToneMapping(texel)');
        break;
      case ToneMappingMode.AGX:
        this.defines.set('toneMapping(texel)', 'AgXToneMapping(texel)');
        break;
      case ToneMappingMode.NEUTRAL:
        this.defines.set('toneMapping(texel)', 'NeutralToneMapping(texel)');
        break;
      default:
        this.defines.set('toneMapping(texel)', 'texel');
        break;
    }

    this.adaptiveLuminancePass.enabled = value === ToneMappingMode.REINHARD2_ADAPTIVE;
    this.setChanged();
  }

  // ======================================================================
  // Original pmndrs properties
  // ======================================================================

  get whitePoint(): number {
    return this.uniforms.get('whitePoint')!.value;
  }
  set whitePoint(value: number) {
    this.uniforms.get('whitePoint')!.value = value;
  }

  get middleGrey(): number {
    return this.uniforms.get('middleGrey')!.value;
  }
  set middleGrey(value: number) {
    this.uniforms.get('middleGrey')!.value = value;
  }

  get averageLuminance(): number {
    return this.uniforms.get('averageLuminance')!.value;
  }
  set averageLuminance(value: number) {
    this.uniforms.get('averageLuminance')!.value = value;
  }

  get adaptiveLuminanceMaterial(): AdaptiveLuminanceMaterial {
    return this.adaptiveLuminancePass.fullscreenMaterial as AdaptiveLuminanceMaterial;
  }

  get resolution(): number {
    return this.luminancePass.resolution.width;
  }
  set resolution(value: number) {
    const exponent = Math.max(0, Math.ceil(Math.log2(value)));
    const size = Math.pow(2, exponent);
    this.luminancePass.resolution.setPreferredSize(size, size);
    this.adaptiveLuminanceMaterial.mipLevel1x1 = exponent;
  }

  get adaptive(): boolean {
    return this.mode === ToneMappingMode.REINHARD2_ADAPTIVE;
  }
  set adaptive(value: boolean) {
    this.mode = value ? ToneMappingMode.REINHARD2_ADAPTIVE : ToneMappingMode.REINHARD2;
  }

  get adaptationRate(): number {
    return this.adaptiveLuminanceMaterial.adaptationRate;
  }
  set adaptationRate(value: number) {
    this.adaptiveLuminanceMaterial.adaptationRate = value;
  }

  // ======================================================================
  // Luxar EOG properties (new)
  // ======================================================================

  /** Global exposure in log2 stops. 0 = neutral, +1 = 2x brighter, -1 = half. */
  get exposure(): number {
    return this.uniforms.get('uExposure')!.value;
  }
  set exposure(value: number) {
    this.uniforms.get('uExposure')!.value = value;
  }

  /** Global additive offset. Lifts or lowers the entire composited image. */
  get globalOffset(): number {
    return this.uniforms.get('uGlobalOffset')!.value;
  }
  set globalOffset(value: number) {
    this.uniforms.get('uGlobalOffset')!.value = value;
  }

  /** Global gamma correction. Reshapes midtones globally. */
  get globalGamma(): number {
    return this.uniforms.get('uGlobalGamma')!.value;
  }
  set globalGamma(value: number) {
    this.uniforms.get('uGlobalGamma')!.value = Math.max(0.001, value);
  }

  // ======================================================================
  // Lifecycle (vendored from pmndrs)
  // ======================================================================

  update(renderer: WebGLRenderer, inputBuffer: WebGLRenderTarget, deltaTime?: number): void {
    if (this.adaptiveLuminancePass.enabled) {
      this.luminancePass.render(renderer, inputBuffer, this.renderTargetLuminance);
      this.adaptiveLuminancePass.render(renderer, null, null, deltaTime);
    }
  }

  initialize(renderer: WebGLRenderer, alpha: boolean, frameBufferType: number): void {
    this.adaptiveLuminancePass.initialize(renderer, alpha, frameBufferType);
  }
}
