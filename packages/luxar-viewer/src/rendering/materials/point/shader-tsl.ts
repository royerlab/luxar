/**
 * Point material TSL factory — NodeMaterial counterpart to the
 * GLSL3 pair in `shaders/point-shaders.ts`.
 *
 * Renders one soft Gaussian-falloff sprite per point with:
 *   - world-space sizing (perspective + orthographic)
 *   - sharpness compensation (visibility-threshold normalisation)
 *   - per-node Gain/Offset/Gamma colour adjustment
 *   - zero-radius nD-slicing discard
 *
 * The mesh layout is a 4-vertex unit-quad base plus
 * InstancedBufferAttribute per-instance data
 * (aCenter, aRadius, aSharpness, aColor, aScalar). The vertex stage
 * projects aCenter to clip space and expands the unit quad by the
 * per-instance pointSize; the fragment stage discards outside the
 * inscribed circle and computes Gaussian falloff.
 *
 * Feature toggles map to {@link PointTSLConfig}, mirroring the GLSL3
 * `#define` semantics where flipping a flag triggers a recompile:
 *   - `useColormap` → reads `aScalar` + samples `uColormapTex`
 *     instead of `aColor`.
 *   - `useMaxRGBContribution` → premultiplies output RGB by alpha
 *     for the CustomBlending + MaxEquation rendering mode.
 *
 * @module rendering/materials/point/shader-tsl
 */

import * as THREE from 'three';
import {
  Fn,
  uniform,
  attribute,
  varying,
  vec2,
  vec3,
  vec4,
  float,
  int,
  max,
  clamp,
  length,
  dot,
  pow,
  Discard,
  texture,
  modelViewMatrix,
  cameraProjectionMatrix,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { sanitizeNonNegative, sanitizePositive, type TSLNode } from '../_shared/tsl-helpers';
import { applyBlendingStateToMaterial, getCompleteBlendingState } from '../../blending-state';
import type { BlendingMode } from '../../material-manager';

export interface PointTSLConfig {
  readonly useColormap?: boolean;
  /**
   * When `true` (gamma == 1.0), the per-fragment / pre-LUT gamma
   * `pow()` is skipped — `pow(x, 1) == x`. Mirrors the GLSL3
   * `LUXAR_GAMMA_ONE` define. The material wrapper sets this from the
   * presence of that define and toggles it on `updateGamma`.
   */
  readonly gammaOne?: boolean;
  /**
   * When undefined, derived from `blendingMode === 'max'`. Explicit
   * config still wins so callers can decouple shader output from
   * framebuffer blending (rare but supported).
   */
  readonly useMaxRGBContribution?: boolean;
  /**
   * Luxar blending mode. The factory configures the matching THREE
   * state via {@link getCompleteBlendingState} +
   * {@link applyBlendingStateToMaterial}. Defaults to `'additive'` to
   * match the GLSL wrapper class.
   */
  readonly blendingMode?: BlendingMode;
}

/**
 * Point-material TSL factory.
 *
 * The `uniforms` table must include the full set the GLSL3 shader
 * reads. Missing fields are tolerated only on optional branches
 * (colormap uniforms when `useColormap === false`).
 *
 * Pass `outMaterial` to configure an existing NodeMaterial subclass
 * (e.g. `PointTSLMaterial`) rather than allocating a new one — the
 * subclass owns the IUniforms table, so the factory just attaches
 * `vertexNode` / `colorNode` / blending state. When omitted, a
 * fresh NodeMaterial is allocated (the common standalone case used
 * by `tsl-shader-parity.spec.ts`).
 */
export function pointWebGPUFactory(
  uniforms: Record<string, THREE.IUniform>,
  config: PointTSLConfig = {},
  outMaterial?: NodeMaterial
): NodeMaterial {
  // Per-vertex (4 corners, ±1).
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  // Per-instance.
  const aCenter: TSLNode = attribute<'vec3'>('aCenter', 'vec3');
  const aRadius: TSLNode = attribute<'float'>('aRadius', 'float');
  const aSharpness: TSLNode = attribute<'float'>('aSharpness', 'float');
  const aColor: TSLNode = attribute<'vec3'>('aColor', 'vec3');
  const aScalar: TSLNode = config.useColormap ? attribute<'float'>('aScalar', 'float') : null;

  // Uniforms — vertex stage. Each uniform binds via `.onUpdate(() =>
  // iuniform.value)` so the TSL node tracks the host's IUniform table
  // by reference. This is the property that lets a wrapper class
  // (PointTSLMaterial) mutate `this.uniforms.X.value` and have the
  // change propagate to the shader — exactly the same pattern the
  // GLSL ShaderMaterial wrapper relies on. The initial value handed
  // to `uniform(...)` matches the IUniform's first read so the
  // generated shader's constant-folding is identical to the GLSL3
  // path.
  const uPointSizeFactor = uniform((uniforms.pointSizeFactor.value as number) ?? 1.0).onUpdate(
    () => (uniforms.pointSizeFactor.value as number) ?? 1.0,
    'render'
  );
  const uMaxPointSize = uniform((uniforms.maxPointSize.value as number) ?? 1.0).onUpdate(
    () => (uniforms.maxPointSize.value as number) ?? 1.0,
    'render'
  );
  const uRadiusScale = uniform((uniforms.radiusScale.value as number) ?? 1.0).onUpdate(
    () => (uniforms.radiusScale.value as number) ?? 1.0,
    'render'
  );
  const uSharpnessScale = uniform((uniforms.sharpnessScale.value as number) ?? 1.0).onUpdate(
    () => (uniforms.sharpnessScale.value as number) ?? 1.0,
    'render'
  );
  const uIsOrtho = uniform((uniforms.uIsOrtho.value as number) ?? 0).onUpdate(
    () => (uniforms.uIsOrtho.value as number) ?? 0,
    'render'
  );
  // Vector2 is passed by reference — the wrapper mutates the same
  // Vector2 (via .set) and the TSL node picks up the change without
  // needing onUpdate. The fallback below only fires when the IUniform
  // value is missing at graph-build time.
  const uResolution = uniform(
    (uniforms.uResolution.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );

  // Colormap (optional). Texture references propagate via the same
  // shared-reference rule as Vector2; if `uColormapTex.value` is
  // reassigned to a different Texture, we need `onUpdate` to swap
  // it (texture nodes hold the THREE.Texture by reference).
  const uColormapTex =
    config.useColormap && uniforms.uColormapTex
      ? texture((uniforms.uColormapTex.value as THREE.Texture | null) ?? new THREE.Texture())
      : null;
  const uScalarMin =
    config.useColormap && uniforms.uScalarMin
      ? uniform((uniforms.uScalarMin.value as number) ?? 0.0).onUpdate(
          () => (uniforms.uScalarMin?.value as number) ?? 0.0,
          'render'
        )
      : null;
  const uScalarScale =
    config.useColormap && uniforms.uScalarScale
      ? uniform((uniforms.uScalarScale.value as number) ?? 1.0).onUpdate(
          () => (uniforms.uScalarScale?.value as number) ?? 1.0,
          'render'
        )
      : null;

  // Uniforms — fragment stage.
  const uOpacity = uniform((uniforms.opacity.value as number) ?? 1.0).onUpdate(
    () => (uniforms.opacity.value as number) ?? 1.0,
    'render'
  );
  const uInvGamma = uniform((uniforms.invGamma.value as number) ?? 1.0).onUpdate(
    () => (uniforms.invGamma.value as number) ?? 1.0,
    'render'
  );
  const uIntensity = uniform((uniforms.uIntensity.value as number) ?? 1.0).onUpdate(
    () => (uniforms.uIntensity.value as number) ?? 1.0,
    'render'
  );
  const uOffset = uniform((uniforms.uOffset.value as number) ?? 0.0).onUpdate(
    () => (uniforms.uOffset.value as number) ?? 0.0,
    'render'
  );

  // RGB premultiplication is driven by the blending mode: `max` mode
  // routes through CustomBlending + MaxEquation which needs RGB to
  // already include the soft-kernel contribution. Explicit
  // `useMaxRGBContribution` still wins for callers that want to
  // decouple shader output from framebuffer blending.
  const premultiplyRGB =
    config.useMaxRGBContribution !== undefined
      ? config.useMaxRGBContribution
      : config.blendingMode === 'max';

  // ---- Vertex computation ----

  // Sanitise per-instance attributes (NaN/Inf-safe).
  const normalizedSharpness: TSLNode = sanitizePositive(
    aSharpness.mul(uSharpnessScale),
    float(2.0)
  );
  const normalizedRadius: TSLNode = sanitizeNonNegative(aRadius.mul(uRadiusScale), float(0.0));

  // Per-instance colour from LUT or attribute. In colormap mode the
  // display range (uScalarMin/uScalarScale) and gamma operate on the
  // scalar VALUE before the LUT lookup, not on the resulting color —
  // mirrors the GLSL3 USE_COLORMAP path.
  let perPointColor: TSLNode;
  if (config.useColormap && aScalar && uColormapTex && uScalarMin && uScalarScale) {
    const t0 = clamp(aScalar.sub(uScalarMin).mul(uScalarScale), 0.0, 1.0);
    // gammaOne skips the pre-LUT pow() when gamma == 1.0.
    const t = config.gammaOne ? t0 : t0.pow(uInvGamma); // gamma on the value, pre-LUT
    perPointColor = uColormapTex.sample(vec2(t, 0.5)).rgb;
  } else {
    perPointColor = aColor;
  }

  // Project per-instance centre to view + clip space.
  const mvPos: TSLNode = modelViewMatrix.mul(vec4(aCenter, 1.0));
  const projCenter: TSLNode = cameraProjectionMatrix.mul(mvPos);

  // World-space size: perspective gets 1/length(view-z) attenuation;
  // ortho stays constant.
  const invDistance: TSLNode = int(uIsOrtho)
    .equal(int(1))
    .select(float(1.0), length(vec3(mvPos)).reciprocal());
  const basePointSize: TSLNode = normalizedRadius.mul(uPointSizeFactor).mul(invDistance);

  // Sharpness compensation: keep visible disk extent constant.
  const sharpnessCompRaw: TSLNode = float(1.0).div(
    float(1.0).sub(pow(float(0.01), float(1.0).div(max(normalizedSharpness, float(0.01)))))
  );
  const sharpnessCompFinite = sharpnessCompRaw
    .lessThan(1e30)
    .and(sharpnessCompRaw.greaterThan(-1e30));
  const sharpnessComp: TSLNode = sharpnessCompFinite.select(sharpnessCompRaw, float(1.0));
  const computedPointSize: TSLNode = basePointSize.mul(sharpnessComp);
  const pointSize: TSLNode = max(float(1.0), clamp(computedPointSize, float(1.0), uMaxPointSize));

  // Expand the unit quad to a sprite in clip space.
  const offsetClip: TSLNode = aQuadCorner.mul(pointSize.div(uResolution)).mul(projCenter.w);
  const clipPos: TSLNode = projCenter.add(vec4(offsetClip, 0.0, 0.0));

  // Sprite UV (replaces gl_PointCoord). Computed per-vertex,
  // interpolated to the fragment via the `varying()` wrapper —
  // matches `vSpriteCoord = (aQuadCorner + 1.0) * 0.5` from GLSL.
  const vSpriteCoord: TSLNode = varying(aQuadCorner.add(1.0).mul(0.5));
  // Per-instance vRadius and vSharpness are constant within a quad
  // (4 verts share the same instance) so `varying()` interpolation
  // is a no-op but the wrapper is what gets TSL to pass them to the
  // fragment stage.
  const vRadius: TSLNode = varying(normalizedRadius);
  const vSharpness: TSLNode = varying(normalizedSharpness);
  const vColor: TSLNode = varying(perPointColor);

  // ---- Fragment computation ----

  const colorNode = Fn(() => {
    // Zero-radius nD-slicing discard.
    Discard(vRadius.lessThan(0.0001));

    const centered: TSLNode = vec2(vSpriteCoord.sub(0.5));
    const r2: TSLNode = dot(centered, centered);
    Discard(r2.greaterThan(0.25));

    const normalizedR: TSLNode = r2.mul(4.0).sqrt();
    const falloff: TSLNode = float(1.0).sub(normalizedR).max(float(0.0)).pow(vSharpness);

    // GOG: colour × intensity + offset, clamped, then gamma.
    // Colormap mode bypasses color GOG — gamma + display-range shaped the
    // scalar VALUE pre-LUT (vertex stage), matching the GLSL3 path.
    const adjusted: TSLNode = config.useColormap
      ? max(vColor, vec3(0.0))
      : max(vColor.mul(uIntensity).add(uOffset), vec3(0.0));
    Discard(max(adjusted.r, max(adjusted.g, adjusted.b)).lessThan(1e-4));
    // Colormap mode (gamma applied pre-LUT) OR gammaOne both skip the pow().
    const finalColor: TSLNode =
      config.useColormap || config.gammaOne ? adjusted : adjusted.pow(vec3(uInvGamma));

    const alpha: TSLNode = falloff.mul(uOpacity);

    if (premultiplyRGB) {
      // RGB premultiplied by alpha — CustomBlending + MaxEquation.
      return vec4(finalColor.mul(alpha), alpha);
    }
    return vec4(finalColor, alpha);
  });

  const material = outMaterial ?? new NodeMaterial();
  // Override the vertex output entirely — we project + expand the
  // sprite ourselves. NodeMaterial.vertexNode replaces the default
  // modelViewProjection chain.
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.toneMapped = false;

  // Wire blending state from the shared helper. The shader-output
  // shape (premultiplied RGB vs alpha-weighted) is derived from the
  // blending mode unless the caller passed an explicit override.
  const blendingMode: BlendingMode = config.blendingMode ?? 'additive';
  const opacityValue = (uniforms.opacity?.value as number | undefined) ?? 1.0;
  const blendingState = getCompleteBlendingState(blendingMode, opacityValue);
  applyBlendingStateToMaterial(material, blendingState);
  return material;
}
