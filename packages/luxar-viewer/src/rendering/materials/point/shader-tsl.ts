/**
 * Point material TSL factory — NodeMaterial counterpart to the
 * GLSL3 pair in `shader-glsl.ts`.
 *
 * Renders one soft super-Gaussian-falloff sprite per point with:
 *   - world-space sizing (perspective + orthographic)
 *   - sharpness -> super-Gaussian exponent beta mapping (beta = 2^(6s-2))
 *   - per-node Gain/Offset/Gamma colour adjustment
 *   - zero-radius nD-slicing discard
 *
 * The mesh layout is a 4-vertex unit-quad base plus
 * InstancedBufferAttribute per-instance data
 * (aCenter, aRadius, aSharpness, aColor, aScalar). The vertex stage
 * projects aCenter to clip space and expands the unit quad by the
 * per-instance pointSize; the fragment stage discards outside the
 * inscribed circle and computes the shifted-truncated super-Gaussian falloff.
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
  min,
  clamp,
  dot,
  exp,
  Discard,
  texture,
  modelViewMatrix,
  cameraProjectionMatrix,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { sanitizeNonNegative, perspectiveNearFadeTSL, type TSLNode } from '../_shared/tsl-helpers';
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
 * Pre-created TSL leaf nodes supplied by a wrapper class. Same
 * pattern as `LineTSLNodes` / `PointPickTSLNodes`: consumers own the
 * `UniformNode`s and the factory references them directly, which
 * avoids the `.onUpdate('render')` callback bridge that the old
 * uniform-record-based path used. Mutations on the wrapper's
 * `material.uniforms.X.value` (proxied via `proxyIUniform`) land
 * directly on `node.value`.
 *
 * Keys match the public `PointMaterial.uniforms` names (some are
 * historically un-prefixed — `opacity`, `invGamma`, …) so the wrapper
 * proxy table maps 1:1.
 *
 * Colormap nodes are optional and bound only when the consumer is
 * built with `config.useColormap === true`. The factory throws if
 * the config says yes but the colormap nodes are missing.
 */
export interface PointTSLNodes {
  readonly pointSizeFactor: TSLNode;
  readonly maxPointSize: TSLNode;
  readonly radiusScale: TSLNode;
  readonly uIsOrtho: TSLNode;
  readonly uNearCull: TSLNode;
  readonly uResolution: TSLNode;
  readonly opacity: TSLNode;
  readonly invGamma: TSLNode;
  readonly uIntensity: TSLNode;
  readonly uOffset: TSLNode;
  /** Set only when colormap mode is active. */
  readonly uColormapTex?: TSLNode;
  readonly uScalarMin?: TSLNode;
  readonly uScalarScale?: TSLNode;
}

/**
 * Point-material TSL factory.
 *
 * Consumes pre-created `UniformNode` references via `nodes`; the
 * wrapper class (`PointTSLMaterial`) owns those nodes and exposes
 * them through `material.uniforms` as `IUniform`-shaped
 * getter/setter proxies. The harness / `POINT_SOURCE` ShaderSource
 * registry constructs the nodes from a plain `uniforms` record via
 * {@link buildPointTSLNodesFromUniforms}.
 *
 * Pass `outMaterial` to configure an existing NodeMaterial subclass
 * (e.g. `PointTSLMaterial`) rather than allocating a new one — the
 * subclass owns the nodes table, so the factory just attaches
 * `vertexNode` / `colorNode` / blending state. When omitted, a
 * fresh NodeMaterial is allocated (the common standalone case used
 * by `tsl-shader-parity.spec.ts`).
 */
export function pointWebGPUFactory(
  nodes: PointTSLNodes,
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

  // Bind directly to the persistent `UniformNode`s owned by the
  // wrapper class (or by `buildPointTSLNodesFromUniforms` for the
  // harness path). Mutations on `material.uniforms.X.value` go via
  // `proxyIUniform` straight to `node.value` — no per-render
  // `.onUpdate` callbacks needed. Unlike lines, `uIsOrtho` stays a
  // RUNTIME uniform here (the graph selects the ortho branch per
  // vertex), so no rebuild is needed on camera-mode flips.
  const uPointSizeFactor = nodes.pointSizeFactor;
  const uMaxPointSize = nodes.maxPointSize;
  const uRadiusScale = nodes.radiusScale;
  const uIsOrtho = nodes.uIsOrtho;
  const uNearCull = nodes.uNearCull;
  const uResolution = nodes.uResolution;
  const uOpacity = nodes.opacity;
  const uInvGamma = nodes.invGamma;
  const uIntensity = nodes.uIntensity;
  const uOffset = nodes.uOffset;
  if (config.useColormap) {
    if (!nodes.uColormapTex || !nodes.uScalarMin || !nodes.uScalarScale) {
      throw new Error(
        'pointWebGPUFactory: config.useColormap=true but nodes.uColormapTex / uScalarMin / uScalarScale are not bound.'
      );
    }
  }
  const uColormapTex = config.useColormap ? nodes.uColormapTex! : null;
  const uScalarMin = config.useColormap ? nodes.uScalarMin! : null;
  const uScalarScale = config.useColormap ? nodes.uScalarScale! : null;

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
  // Sharpness is authored in [0, 1] -> super-Gaussian exponent
  // beta = 2^(6s - 2) (s=0.5 -> beta=2, a true Gaussian). sanitizeNonNegative
  // keeps a valid s=0 (-> beta=0.25) and routes NaN/Inf/negative to the 0.5
  // default; clamp bounds the [0, 1] range. Mirrors the GLSL3 path exactly.
  const sClamped: TSLNode = clamp(sanitizeNonNegative(aSharpness, float(0.5)), 0.0, 1.0);
  const beta: TSLNode = float(2.0).pow(sClamped.mul(6.0).sub(2.0));
  const normalizedRadius: TSLNode = sanitizeNonNegative(aRadius.mul(uRadiusScale), float(0.0));

  // Per-instance colour from LUT or attribute. In colormap mode the
  // display range (uScalarMin/uScalarScale) and gamma shape the scalar
  // VALUE before the LUT lookup, not the resulting color; intensity/
  // offset apply POST-LUT in the fragment stage (matching the gsplat
  // shader) — mirrors the GLSL3 USE_COLORMAP path.
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

  // World-space size from VIEW-SPACE DEPTH (-mvPos.z), matching the
  // line + gsplat shaders (Euclidean distance shrank edge-of-screen
  // points by cos(theta)); ortho stays constant. 1e-4 floor mirrors
  // the line shader's nearCull floor.
  const invDistance: TSLNode = int(uIsOrtho)
    .equal(int(1))
    .select(float(1.0), mvPos.z.negate().max(float(1e-4)).reciprocal());
  const basePointSize: TSLNode = normalizedRadius.mul(uPointSizeFactor).mul(invDistance);

  // No size compensation: the shifted-truncated super-Gaussian truncates at
  // the sprite edge (rho = 1), so basePointSize already IS the visible extent.
  // Minimum sprite size 1.5px (matches the LINE shader — thinner quads
  // cause rasterization gaps); sub-pixel energy is preserved by the
  // fragment's sizeScale^2 compensation via vPointSize.
  const pointSize: TSLNode = clamp(basePointSize, float(1.5), uMaxPointSize);

  // Expand the unit quad to a sprite in clip space.
  const offsetClip: TSLNode = aQuadCorner.mul(pointSize.div(uResolution)).mul(projCenter.w);
  // Unified near handling (matches line + gsplat shaders and the GLSL
  // twin): behind-camera fades to 0 (projCenter.w <= 0 there would flip
  // the sprite), the near-plane approach fades across
  // [nearCull, 2*nearCull], ortho passes through (NDC clipping is the
  // authority). Reject below 0.01, multiply the survivor into alpha.
  const depthFade: TSLNode = perspectiveNearFadeTSL(
    uIsOrtho,
    mvPos.z,
    max(uNearCull, float(1e-4))
  ).toVar();
  const clipPos: TSLNode = depthFade
    .lessThan(0.01)
    .select(vec4(0.0, 0.0, -2.0, 1.0), projCenter.add(vec4(offsetClip, 0.0, 0.0)));

  // Sprite UV (replaces gl_PointCoord). Computed per-vertex,
  // interpolated to the fragment via the `varying()` wrapper —
  // matches `vSpriteCoord = (aQuadCorner + 1.0) * 0.5` from GLSL.
  const vSpriteCoord: TSLNode = varying(aQuadCorner.add(1.0).mul(0.5));
  // Per-instance vRadius and vBeta are constant within a quad
  // (4 verts share the same instance) so `varying()` interpolation
  // is a no-op but the wrapper is what gets TSL to pass them to the
  // fragment stage.
  const vRadius: TSLNode = varying(normalizedRadius);
  const vBeta: TSLNode = varying(beta);
  const vColor: TSLNode = varying(perPointColor);
  const vPointSize: TSLNode = varying(basePointSize);
  const vNearFade: TSLNode = varying(depthFade);

  // ---- Fragment computation ----

  const colorNode = Fn(() => {
    // Zero-radius nD-slicing discard.
    Discard(vRadius.lessThan(0.0001));

    const centered: TSLNode = vec2(vSpriteCoord.sub(0.5));
    const r2: TSLNode = dot(centered, centered);
    Discard(r2.greaterThan(0.25));

    const normalizedR: TSLNode = r2.mul(4.0).sqrt();
    // Shifted-truncated super-Gaussian: max(exp(-K * rho^beta) - C, 0) / (1 - C),
    // C0-continuous at the sprite edge. beta=2 reproduces the gsplat Gaussian.
    // K = ln(1/floor), floor = 0.01. Mirrors the GLSL3 fragment exactly.
    const K = 4.6051702; // ln(100)
    const C = 0.01; // exp(-K) = floor
    const invOneMinusC = 1.0 / (1.0 - C);
    const falloff: TSLNode = exp(normalizedR.pow(vBeta).mul(-K))
      .sub(C)
      .max(float(0.0))
      .mul(invOneMinusC);

    // GOG. uIntensity (gain) + uOffset apply in BOTH modes so the layer
    // intensity/offset controls work for a colormapped point too (GLSL
    // parity, matching the gsplat shader). Colormap mode: gamma +
    // display-range shaped the scalar VALUE pre-LUT (vertex stage), so
    // only gain/offset apply post-LUT (no extra gamma).
    const adjusted: TSLNode = max(vColor.mul(uIntensity).add(uOffset), vec3(0.0));
    Discard(max(adjusted.r, max(adjusted.g, adjusted.b)).lessThan(1e-4));
    // Colormap mode (gamma applied pre-LUT) OR gammaOne both skip the pow().
    const finalColor: TSLNode =
      config.useColormap || config.gammaOne ? adjusted : adjusted.pow(vec3(uInvGamma));

    // Sub-pixel intensity compensation (mirrors the line shader's
    // widthScale, SQUARED: both sprite dimensions clamp, energy ∝ area).
    const sizeScale: TSLNode = min(vPointSize.div(float(1.5)), float(1.0));
    const alpha: TSLNode = falloff.mul(uOpacity).mul(sizeScale.mul(sizeScale)).mul(vNearFade);

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
  const opacityValue = (nodes.opacity.value as number | undefined) ?? 1.0;
  const blendingState = getCompleteBlendingState(blendingMode, opacityValue);
  applyBlendingStateToMaterial(material, blendingState);
  return material;
}

/**
 * Build a `PointTSLNodes` set from a plain `IUniform` record. Used by
 * the test harness and the `POINT_SOURCE` ShaderSource factory in
 * `shader-glsl.ts` — callers that don't own persistent
 * wrapper-side `UniformNode`s. Mirrors
 * `buildLineTSLNodesFromUniforms`.
 *
 * Note: the resulting nodes capture the current `iuniform.value` at
 * build time. Mutations to the host `IUniform`'s `.value` after this
 * function returns will NOT propagate — appropriate for the harness
 * (which builds once and renders once) but not for live wrappers
 * (which must use `proxyIUniform` against persistent nodes).
 */
export function buildPointTSLNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>,
  config: PointTSLConfig = {}
): PointTSLNodes {
  const base: PointTSLNodes = {
    pointSizeFactor: uniform((uniforms.pointSizeFactor?.value as number) ?? 1.0),
    maxPointSize: uniform((uniforms.maxPointSize?.value as number) ?? 1.0),
    radiusScale: uniform((uniforms.radiusScale?.value as number) ?? 1.0),
    uIsOrtho: uniform((uniforms.uIsOrtho?.value as number) ?? 0),
    uNearCull: uniform((uniforms.uNearCull?.value as number) ?? 0.1),
    uResolution: uniform(
      (uniforms.uResolution?.value as THREE.Vector2 | undefined) ?? new THREE.Vector2(1, 1)
    ),
    opacity: uniform((uniforms.opacity?.value as number) ?? 1.0),
    invGamma: uniform((uniforms.invGamma?.value as number) ?? 1.0),
    uIntensity: uniform((uniforms.uIntensity?.value as number) ?? 1.0),
    uOffset: uniform((uniforms.uOffset?.value as number) ?? 0.0),
  };
  if (!config.useColormap) return base;
  return {
    ...base,
    uColormapTex: texture(
      (uniforms.uColormapTex?.value as THREE.Texture | null) ?? new THREE.Texture()
    ),
    uScalarMin: uniform((uniforms.uScalarMin?.value as number) ?? 0.0),
    uScalarScale: uniform((uniforms.uScalarScale?.value as number) ?? 1.0),
  };
}
