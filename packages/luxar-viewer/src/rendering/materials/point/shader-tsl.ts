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
 * The mesh layout is a 4-vertex unit-quad base plus one per-instance
 * attribute:
 *   - aSortedIndex (uint) — draw-slot → storage-slot mapping
 *     (identity in Phase 1; the sort worker permutes it in Phase 2+)
 * Per-point data comes from the RGBA32F point texture (`uPointTex`,
 * 3 texels/point — layout in `rendering/element-texture-layout.ts`),
 * fetched in the vertex stage via `textureLoad`. The vertex stage
 * projects the fetched centre to clip space and expands the unit quad
 * by the per-instance pointSize; the fragment stage discards outside
 * the inscribed circle and computes the shifted-truncated
 * super-Gaussian falloff.
 *
 * Feature toggles map to {@link PointTSLConfig}, mirroring the GLSL3
 * `#define` semantics where flipping a flag triggers a recompile:
 *   - `useColormap` → fetches the texel2.x scalar + samples
 *     `uColormapTex` instead of the texel1 color.
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
  vec2 as _vec2,
  vec3 as _vec3,
  vec4 as _vec4,
  ivec2 as _ivec2,
  float,
  int,
  textureSize,
  max,
  min,
  mix,
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
import {
  ALPHA_CLAMP,
  VOLUMETRIC_SERIES_C1,
  VOLUMETRIC_SERIES_C2_DIVISOR,
  VOLUMETRIC_SERIES_TAU_THRESHOLD,
  VOLUMETRIC_TAU_EPS,
} from '../_shared/volumetric';
import { POINT_CHORD_SCALE } from './math';
import { getPlaceholderElementTexture } from '../../element-texture-layout';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  effectiveGeometryMode,
  isVolumetricMode,
} from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';

// Type-erased constructor aliases — same rationale as the gsplat TSL
// factory (shader-tsl.ts there): TSL's typed `vec*` overloads reject
// many valid combinations of intermediate `Node<…>` results. Re-export
// each as TSLNode-typed to sidestep overload-mismatch errors without
// affecting the generated GLSL/WGSL.
const vec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _vec2 as TSLNode;
const vec3: (a?: TSLNode, b?: TSLNode, c?: TSLNode) => TSLNode = _vec3 as TSLNode;
const vec4: (a?: TSLNode, b?: TSLNode, c?: TSLNode, d?: TSLNode) => TSLNode = _vec4 as TSLNode;
const ivec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _ivec2 as TSLNode;

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
 *
 * The colormap AND point texture nodes are factory-time bound (TSL
 * `texture(...)` captures the THREE.Texture at call time); the
 * wrapper rebuilds the graph when either texture's identity changes
 * (see `setColormapTexture` / `updatePointTexture` in the wrapper).
 */
export interface PointTSLNodes {
  /**
   * Point data texture node (RGBA32F, 3 texels/point). Every point
   * material has one; the pool commit rebinds it per node via
   * `updatePointTexture`.
   */
  readonly uPointTex: TSLNode;
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
  /**
   * κ — composed node absorption (volumetric mode). Read only when the
   * graph was built with `blendingMode: 'volumetric'`; a plain runtime
   * uniform otherwise (mirrors the gsplat factory).
   */
  readonly uAbsorption: TSLNode;
  /**
   * 1.0 when the committed colors carry a real alpha column (RGBA), 0
   * otherwise. Gates ONLY the volumetric w(a) optical-depth map — the
   * identity alpha 1.0 written for RGB data must not map to w ≈ 6.24.
   * Deliberately a uniform, not a config flag: toggling it never
   * rebuilds the graph.
   */
  readonly uHasElementAlpha: TSLNode;
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
  // The only per-instance attribute: point data itself lives in the
  // point texture; `aSortedIndex` maps the draw slot to a storage slot
  // (identity in Phase 1, permuted by the sort worker in Phase 2+).
  const aSortedIndex: TSLNode = attribute<'uint'>('aSortedIndex', 'uint');

  // Bind directly to the persistent `UniformNode`s owned by the
  // wrapper class (or by `buildPointTSLNodesFromUniforms` for the
  // harness path). Mutations on `material.uniforms.X.value` go via
  // `proxyIUniform` straight to `node.value` — no per-render
  // `.onUpdate` callbacks needed. Unlike lines, `uIsOrtho` stays a
  // RUNTIME uniform here (the graph selects the ortho branch per
  // vertex), so no rebuild is needed on camera-mode flips.
  const uPointTex = nodes.uPointTex;
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
  const uAbsorption = nodes.uAbsorption;
  const uHasElementAlpha = nodes.uHasElementAlpha;
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

  // The volumetric (emission–absorption) output branch is chosen at
  // GRAPH BUILD time — a JS conditional, exactly like the gsplat
  // factory (TSL `.select()` is avoided for structural branches). The
  // wrapper's `applyBlendingMode` rebuilds the graph on any
  // volumetric crossing via the LUXAR_VOLUMETRIC define. Routed
  // through `effectiveGeometryMode` so the line fallback policy stays
  // centralized (identity for points since volumetric phase 3).
  const volumetricGraph = isVolumetricMode(
    effectiveGeometryMode(config.blendingMode ?? 'additive', 'point')
  );

  // ---- Vertex computation ----
  //
  // The ENTIRE vertex stage is traced inside a single Fn() body with
  // explicit `.toVar()` statements — the same load-bearing structure
  // as the gsplat factory (materials/gsplat/shader-tsl.ts): as a free
  // expression tree, TSL materializes a shared subexpression at its
  // FIRST traversal use, which can land inside a `.select()` branch
  // and read uninitialized on the other path. Inside Fn(), statements
  // emit in trace order, unconditionally.

  // Varyings are declared up front and `.assign()`ed inside the vertex
  // body (the TSL pattern for Fn-traced vertex stages). Per-instance
  // varyings are constant within a quad (4 verts share the instance)
  // so interpolation is a no-op, but the wrapper is what gets TSL to
  // pass them to the fragment stage.
  const vSpriteCoord: TSLNode = varying(vec2(float(0.0), float(0.0)));
  const vRadius: TSLNode = varying(float(0.0));
  const vBeta: TSLNode = varying(float(0.0));
  const vColor: TSLNode = varying(vec3(float(0.0), float(0.0), float(0.0)));
  const vPointSize: TSLNode = varying(float(0.0));
  const vNearFade: TSLNode = varying(float(1.0));
  // Smooth (non-flat) on the TSL side vs the GLSL twin's `flat out` —
  // deliberately equivalent, not drift: all 4 quad vertices of an
  // instance write the same per-instance value, so interpolation is a
  // no-op (TSL's varying() has no flat qualifier on the WebGL fallback).
  const vAlpha: TSLNode = varying(float(1.0));

  const vertexBody = Fn(() => {
    // === Point-texture fetch prologue ===
    // textureLoad reads reconstruct the per-point values into the exact
    // local names the math below has always used — zero changes
    // downstream of this block. Every value is a `.toVar()` STATEMENT
    // (the Fn house rule; see the block comment above). The texture
    // width is a multiple of 3 (element-texture-layout.ts), so a
    // point's 3 texels share one row and only x advances. texel2
    // carries the colormap scalar (.x) and the per-point alpha (.y,
    // written unconditionally by the texel writer — 1.0 for RGB data)
    // — mirrors the GLSL twin's unconditional fetch.
    const pointBase: TSLNode = int(aSortedIndex).mul(int(3)).toVar();
    // int() wrap is LOAD-BEARING: TSL types textureSize() as uint (the
    // WGSL textureDimensions convention), but the WebGL2 fallback emits
    // GLSL textureSize() which returns int -- without the explicit
    // conversion the generated `uint nodeVar = textureSize(...).x;`
    // fails to compile on the forceWebGL backend.
    const pointTexW: TSLNode = int(
      (textureSize(uPointTex, int(0)) as unknown as TSLNode).x
    ).toVar();
    const texelX: TSLNode = pointBase.mod(pointTexW).toVar();
    const texelY: TSLNode = pointBase.div(pointTexW).toVar();
    const pointT0: TSLNode = uPointTex.load(ivec2(texelX, texelY)).toVar();
    const pointT1: TSLNode = uPointTex.load(ivec2(texelX.add(int(1)), texelY)).toVar();
    const aCenter: TSLNode = vec3(pointT0).toVar(); // world-space centre
    const aRadius: TSLNode = pointT0.w.toVar();
    const aColor: TSLNode = vec3(pointT1).toVar();
    const aSharpness: TSLNode = pointT1.w.toVar();
    const pointT2: TSLNode = uPointTex.load(ivec2(texelX.add(int(2)), texelY)).toVar();
    const aScalar: TSLNode | null = config.useColormap ? pointT2.x.toVar() : null;

    // Sanitise per-point values (NaN/Inf-safe).
    // Sharpness is authored in [0, 1] -> super-Gaussian exponent
    // beta = 2^(6s - 2) (s=0.5 -> beta=2, a true Gaussian). sanitizeNonNegative
    // keeps a valid s=0 (-> beta=0.25) and routes NaN/Inf/negative to the 0.5
    // default; clamp bounds the [0, 1] range. Mirrors the GLSL3 path exactly.
    const sClamped: TSLNode = clamp(sanitizeNonNegative(aSharpness, float(0.5)), 0.0, 1.0);
    const beta: TSLNode = float(2.0).pow(sClamped.mul(6.0).sub(2.0)).toVar();
    const normalizedRadius: TSLNode = sanitizeNonNegative(
      aRadius.mul(uRadiusScale),
      float(0.0)
    ).toVar();

    // Per-point colour from LUT or the texel1 color. In colormap mode
    // the display range (uScalarMin/uScalarScale) and gamma shape the
    // scalar VALUE before the LUT lookup, not the resulting color;
    // intensity/offset apply POST-LUT in the fragment stage (matching
    // the gsplat shader) — mirrors the GLSL3 USE_COLORMAP path.
    let perPointColor: TSLNode;
    if (config.useColormap && aScalar && uColormapTex && uScalarMin && uScalarScale) {
      const t0 = clamp(aScalar.sub(uScalarMin).mul(uScalarScale), 0.0, 1.0);
      // gammaOne skips the pre-LUT pow() when gamma == 1.0.
      const t = config.gammaOne ? t0 : t0.pow(uInvGamma); // gamma on the value, pre-LUT
      perPointColor = uColormapTex.sample(vec2(t, 0.5)).rgb;
    } else {
      perPointColor = aColor;
    }

    // Project per-point centre to view + clip space.
    const mvPos: TSLNode = modelViewMatrix.mul(vec4(aCenter, 1.0)).toVar();
    const projCenter: TSLNode = cameraProjectionMatrix.mul(mvPos).toVar();

    // World-space size from VIEW-SPACE DEPTH (-mvPos.z), matching the
    // line + gsplat shaders (Euclidean distance shrank edge-of-screen
    // points by cos(theta)); ortho stays constant. The 1e-20 floor is a
    // pure divide-by-zero guard, NOT a scale floor: the near-fade
    // reject below already bounds surviving depths at ~uNearCull
    // (scene-relative) and the size clamp bounds the output. The old
    // absolute 1e-4 clamped VALID depths on tiny-unit scenes
    // (-z ~ 1e-6), shrinking every sprite ~100×. GLSL twin:
    // shader-glsl.ts.
    const invDistance: TSLNode = int(uIsOrtho)
      .equal(int(1))
      .select(float(1.0), mvPos.z.negate().max(float(1e-20)).reciprocal());
    const basePointSize: TSLNode = normalizedRadius.mul(uPointSizeFactor).mul(invDistance).toVar();

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
    // uNearCull is scene-bounds-scaled; the 1e-20 floor only guards the
    // degenerate smoothstep when uNearCull == 0 — an absolute 1e-4
    // floor overrode the scene-relative value on tiny-unit scenes and
    // faded out the whole scene. GLSL twin: shader-glsl.ts.
    const depthFade: TSLNode = perspectiveNearFadeTSL(
      uIsOrtho,
      mvPos.z,
      max(uNearCull, float(1e-20))
    ).toVar();
    const clipPos: TSLNode = depthFade
      .lessThan(0.01)
      .select(vec4(0.0, 0.0, -2.0, 1.0), projCenter.add(vec4(offsetClip, 0.0, 0.0)));

    // Assign varyings (declared outside the Fn; see above). Sprite UV
    // replaces gl_PointCoord — matches `vSpriteCoord =
    // (aQuadCorner + 1.0) * 0.5` from GLSL.
    vSpriteCoord.assign(aQuadCorner.add(1.0).mul(0.5));
    vRadius.assign(normalizedRadius);
    vBeta.assign(beta);
    vColor.assign(perPointColor);
    vPointSize.assign(basePointSize);
    vNearFade.assign(depthFade);
    vAlpha.assign(pointT2.y);

    return clipPos;
  });

  const clipPos: TSLNode = vertexBody();

  // ---- Fragment computation ----

  const colorNode = Fn(() => {
    // Zero-radius nD-slicing discard.
    // Exact-zero only — see the GLSL twin's comment.
    Discard(vRadius.lessThanEqual(0.0));

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
    const adjusted: TSLNode = max(vColor.mul(uIntensity).add(uOffset), vec3(0.0)).toVar();
    const maxAdjusted: TSLNode = max(adjusted.r, max(adjusted.g, adjusted.b));

    // Sub-pixel intensity compensation (mirrors the line shader's
    // widthScale, SQUARED: both sprite dimensions clamp, energy ∝ area).
    const sizeScale: TSLNode = min(vPointSize.div(float(1.5)), float(1.0));
    // Screen density of this fragment — falloff scaled by every "how
    // much of this point is there" factor. This is the additive alpha.
    const alphaBase: TSLNode = falloff
      .mul(uOpacity)
      .mul(sizeScale.mul(sizeScale))
      .mul(vNearFade)
      .toVar();

    // Per-point alpha (texel2.y): linear contribution scale in every
    // non-volumetric mode (identity 1.0 for RGB data); volumetric maps
    // it into optical depth w(a) = −ln(1 − a), gated by uHasElementAlpha
    // so the RGB identity 1.0 never maps to w ≈ 6.24 (GLSL twin; clamp
    // = ALPHA_CLAMP from ../_shared/volumetric).
    const alpha: TSLNode = (
      volumetricGraph
        ? alphaBase.mul(
            mix(
              float(1.0),
              min(vAlpha, float(ALPHA_CLAMP)).oneMinus().log().negate(),
              uHasElementAlpha
            )
          )
        : alphaBase.mul(vAlpha)
    ).toVar();

    // Volumetric optical depth: the isotropic special case of the
    // gsplat ray integral — rayMass = density × through-thickness of
    // the Gaussian-profile ball (R·√(π/K), materials/point/math.ts).
    const tau: TSLNode | null = volumetricGraph
      ? uAbsorption.mul(alpha).mul(vRadius).mul(float(POINT_CHORD_SCALE)).toVar()
      : null;
    if (volumetricGraph && tau) {
      // Discard only when color AND τ are both negligible — a black
      // point still absorbs (pure-ink occluders keep their optical depth).
      Discard(maxAdjusted.lessThan(1e-4).and(tau.lessThan(1e-4)));
    } else {
      Discard(maxAdjusted.lessThan(1e-4));
    }

    // Colormap mode (gamma applied pre-LUT) OR gammaOne both skip the pow().
    const finalColor: TSLNode =
      config.useColormap || config.gammaOne ? adjusted : adjusted.pow(vec3(uInvGamma));

    if (volumetricGraph && tau) {
      // 'volumetric' output branch: emission–absorption (Max 1995).
      // finalColor·alpha is exactly what additive adds to the
      // framebuffer, screened by S(τ) = (1−e^(−τ))/τ (series below
      // τ = 1e-3 keeps S(0) = 1 exact — the κ=0 additive limit); alpha
      // out is the physical absorption 1 − e^(−τ) for the
      // One / OneMinusSrcAlpha state. Mirrors the gsplat factory.
      const volAlpha: TSLNode = float(1.0).sub(exp(tau.negate()));
      const series: TSLNode = float(1.0)
        .sub(tau.mul(VOLUMETRIC_SERIES_C1))
        .add(tau.mul(tau).div(VOLUMETRIC_SERIES_C2_DIVISOR));
      const screen: TSLNode = tau
        .lessThan(VOLUMETRIC_SERIES_TAU_THRESHOLD)
        .select(series, volAlpha.div(max(tau, VOLUMETRIC_TAU_EPS)));
      return vec4(finalColor.mul(alpha).mul(screen), volAlpha);
    }
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
  // This factory tail is the ONLY state writer at TSL construction
  // (the ctor never calls applyBlendingMode, unlike the GLSL twin) AND
  // re-runs on every rebuildGraph — so it must judge the mode through
  // the same effectiveGeometryMode policy chokepoint as the wrapper
  // (identity for points since volumetric phase 3; kept so the policy
  // stays centralized in blending-state.ts).
  const requestedMode: BlendingMode = config.blendingMode ?? 'additive';
  const blendingMode: BlendingMode = effectiveGeometryMode(requestedMode, 'point');
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
    // Point data texture — bound from the caller's uniform when
    // present (harness / material paths), else the shared placeholder
    // so codegen-only consumers still build a valid graph.
    uPointTex: texture(
      (uniforms.uPointTex?.value as THREE.Texture | null) ?? getPlaceholderElementTexture()
    ),
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
    uAbsorption: uniform((uniforms.uAbsorption?.value as number) ?? 1.0),
    uHasElementAlpha: uniform((uniforms.uHasElementAlpha?.value as number) ?? 0),
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
