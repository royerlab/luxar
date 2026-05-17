/**
 * Line material TSL factory — NodeMaterial counterpart to the GLSL3
 * shaders in `shaders/line-shaders.ts`.
 *
 * Renders each line segment as a screen-space-thick instanced quad.
 * Per-vertex `aQuadCorner` (±1) determines:
 *   - x ∈ {-1, +1}: position along the segment (interpolation t).
 *   - y ∈ {-1, +1}: perpendicular offset (used for width expansion).
 *
 * Per-segment InstancedBufferAttribute data:
 *   - aStartPos / aEndPos (vec3 world-space endpoints)
 *   - aStartColor / aEndColor (vec3 RGB; replaced by aStartScalar /
 *     aEndScalar + LUT under USE_COLORMAP)
 *   - aStartWidth / aEndWidth (float world-space half-width-ish)
 *   - aStartSharpness / aEndSharpness (float)
 *   - aSegmentLength (float, used by the cap-ramp fragment math)
 *   - aStartClipped / aEndClipped (float ∈ {0, 1})
 *
 * Vertex stage projects both endpoints to view/clip space, computes
 * a per-segment pixel-width with ortho/perspective scaling, clamps
 * to `[minPixelWidth, uMaxLinePixelWidth]` with an intensity-fading
 * "vWidthFade" for near-camera degenerate cases, then expands the
 * quad by perpendicular×pixelWidth in NDC.
 *
 * Fragment stage produces a soft parabolic line: (1 - p²)^sharpness
 * × edgeAA × widthScale × widthFade × capFactor (capFactor ramps to
 * full intensity inside the body but is 1.0 at clipped endpoints).
 *
 * @module rendering/line.tsl
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
  mix,
  length,
  step,
  smoothstep,
  tan,
  texture,
  modelViewMatrix,
  cameraProjectionMatrix,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { sanitizeNonNegative, sanitizePositive, type TSLNode } from './tsl-helpers';
import { applyBlendingStateToMaterial, getCompleteBlendingState } from './blending-state';
import type { BlendingMode } from './material-manager';

export interface LineTSLConfig {
  readonly useColormap?: boolean;
  /**
   * When undefined, derived from `blendingMode === 'max'`. Explicit
   * config still wins.
   */
  readonly useMaxRGBContribution?: boolean;
  /**
   * Luxar blending mode. Defaults to `'additive'` to match the
   * GLSL wrapper class.
   */
  readonly blendingMode?: BlendingMode;
}

/**
 * Line-material TSL factory.
 *
 * The `uniforms` table must include the full set the GLSL3 shader
 * reads: `uFOV`, `uResolution`, `uIsOrtho`, `uNearCull`,
 * `uMaxLinePixelWidth`, `uOpacity`, `uInvGamma`, `uIntensity`,
 * `uOffset`. Colormap uniforms (`uColormapTex`, `uScalarMin`,
 * `uScalarScale`) are required only when `config.useColormap === true`.
 */
export function lineWebGPUFactory(
  uniforms: Record<string, THREE.IUniform>,
  config: LineTSLConfig = {},
  outMaterial?: NodeMaterial
): NodeMaterial {
  // Per-vertex.
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  // Per-instance — endpoint pairs.
  const aStartPos: TSLNode = attribute<'vec3'>('aStartPos', 'vec3');
  const aEndPos: TSLNode = attribute<'vec3'>('aEndPos', 'vec3');
  const aStartColor: TSLNode = attribute<'vec3'>('aStartColor', 'vec3');
  const aEndColor: TSLNode = attribute<'vec3'>('aEndColor', 'vec3');
  const aStartWidth: TSLNode = attribute<'float'>('aStartWidth', 'float');
  const aEndWidth: TSLNode = attribute<'float'>('aEndWidth', 'float');
  const aStartSharpness: TSLNode = attribute<'float'>('aStartSharpness', 'float');
  const aEndSharpness: TSLNode = attribute<'float'>('aEndSharpness', 'float');
  const aSegmentLength: TSLNode = attribute<'float'>('aSegmentLength', 'float');
  const aStartClipped: TSLNode = attribute<'float'>('aStartClipped', 'float');
  const aEndClipped: TSLNode = attribute<'float'>('aEndClipped', 'float');
  // Colormap (per-endpoint scalars).
  const aStartScalar: TSLNode = config.useColormap
    ? attribute<'float'>('aStartScalar', 'float')
    : null;
  const aEndScalar: TSLNode = config.useColormap
    ? attribute<'float'>('aEndScalar', 'float')
    : null;

  // Uniforms — each primitive bound via `.onUpdate(() => iuniform.value)`
  // so a wrapper class's mutations to `this.uniforms.X.value`
  // propagate to the GPU. Vector2 / Texture uniforms share the same
  // host object by reference, so they don't need onUpdate.
  const uFOV = uniform((uniforms.uFOV.value as number) ?? 1.0).onUpdate(
    () => (uniforms.uFOV.value as number) ?? 1.0,
    'render'
  );
  const uResolution = uniform(
    (uniforms.uResolution.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );
  const uIsOrtho = uniform((uniforms.uIsOrtho.value as number) ?? 0).onUpdate(
    () => (uniforms.uIsOrtho.value as number) ?? 0,
    'render'
  );
  const uNearCull = uniform((uniforms.uNearCull.value as number) ?? 1e-4).onUpdate(
    () => (uniforms.uNearCull.value as number) ?? 1e-4,
    'render'
  );
  const uMaxLinePixelWidth = uniform(
    (uniforms.uMaxLinePixelWidth.value as number) ?? 1.0
  ).onUpdate(() => (uniforms.uMaxLinePixelWidth.value as number) ?? 1.0, 'render');
  const uOpacity = uniform((uniforms.uOpacity.value as number) ?? 1.0).onUpdate(
    () => (uniforms.uOpacity.value as number) ?? 1.0,
    'render'
  );
  const uInvGamma = uniform((uniforms.uInvGamma.value as number) ?? 1.0).onUpdate(
    () => (uniforms.uInvGamma.value as number) ?? 1.0,
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

  // RGB premultiplication is driven by the blending mode: `max` mode
  // routes through CustomBlending + MaxEquation which needs RGB to
  // already include the soft-kernel contribution. Explicit
  // `useMaxRGBContribution` still wins.
  const premultiplyRGB =
    config.useMaxRGBContribution !== undefined
      ? config.useMaxRGBContribution
      : config.blendingMode === 'max';

  // ---- Vertex computation ----

  // t ∈ {0, 1} — position along the segment. Branchless because
  // aQuadCorner.x ∈ {-1, +1} by construction.
  const t: TSLNode = aQuadCorner.x.mul(0.5).add(0.5);

  // Per-endpoint colour or LUT lookup.
  let perPointColor: TSLNode;
  if (config.useColormap && aStartScalar && aEndScalar && uColormapTex && uScalarMin && uScalarScale) {
    const s = mix(aStartScalar, aEndScalar, t);
    const st = clamp(s.sub(uScalarMin).mul(uScalarScale), 0.0, 1.0);
    perPointColor = uColormapTex.sample(vec2(st, 0.5)).rgb;
  } else {
    perPointColor = mix(aStartColor, aEndColor, t);
  }

  // Sanitised widths / sharpness, interpolated.
  const startW: TSLNode = sanitizeNonNegative(aStartWidth, float(0.0));
  const endW: TSLNode = sanitizeNonNegative(aEndWidth, float(0.0));
  const startS: TSLNode = sanitizePositive(aStartSharpness, float(2.0));
  const endS: TSLNode = sanitizePositive(aEndSharpness, float(2.0));
  const width: TSLNode = mix(startW, endW, t);
  const vSharpnessVal: TSLNode = mix(startS, endS, t);

  // Project endpoints to view + clip space.
  const mvStart: TSLNode = modelViewMatrix.mul(vec4(aStartPos, 1.0));
  const mvEnd: TSLNode = modelViewMatrix.mul(vec4(aEndPos, 1.0));
  const mvPos: TSLNode = mix(mvStart, mvEnd, t);

  // Near-plane / behind-camera safety. View-space depth = -z.
  const nearCull: TSLNode = max(uNearCull, float(1e-4));
  const startDepth: TSLNode = mvStart.z.negate();
  const endDepth: TSLNode = mvEnd.z.negate();
  const bothBehind: TSLNode = startDepth.lessThan(nearCull).and(endDepth.lessThan(nearCull));

  const clipStart: TSLNode = cameraProjectionMatrix.mul(mvStart);
  const clipEnd: TSLNode = cameraProjectionMatrix.mul(mvEnd);
  // projection is linear, so proj * mix(a,b,t) == mix(proj*a, proj*b, t).
  const clipPosBase: TSLNode = mix(clipStart, clipEnd, t);

  // Convert clip endpoints to pixel space for aspect-correct
  // perpendicular expansion. Guard tiny .w (near-plane crossings).
  const wStart: TSLNode = max(clipStart.w, float(1e-4));
  const wEnd: TSLNode = max(clipEnd.w, float(1e-4));
  const ndcStart: TSLNode = vec2(clipStart.xy.div(wStart));
  const ndcEnd: TSLNode = vec2(clipEnd.xy.div(wEnd));

  // Direction + perpendicular (pixel space, aspect-correct). The +0.5
  // in (ndc*0.5+0.5)*resolution cancels under subtraction, so the
  // pixel-space direction is just (ndcEnd-ndcStart)*(0.5*resolution).
  const pixelDir: TSLNode = vec2(ndcEnd.sub(ndcStart).mul(uResolution.mul(0.5)));
  const pixelLen: TSLNode = length(pixelDir);
  const lineDir: TSLNode = pixelLen
    .greaterThan(0.0001)
    .select(vec2(pixelDir.div(pixelLen)).toVar(), vec2(1.0, 0.0));
  const perpendicular: TSLNode = vec2(lineDir.y.negate(), lineDir.x);

  // World-space → pixel conversion. Ortho mode uses uFOV as
  // frustumHeight; perspective uses tan(fov/2) + view depth.
  const tanHalfFov: TSLNode = tan(uFOV.mul(0.5));
  const distView: TSLNode = max(length(vec3(mvPos)), nearCull);
  const rawPixelWidthOrtho: TSLNode = width.mul(2.0).mul(uResolution.y).div(uFOV);
  const rawPixelWidthPersp: TSLNode = width.mul(uResolution.y).div(distView.mul(tanHalfFov));
  // TSL select() can return zero when both branches are chained
  // expressions rather than materialised values. Wrapping the
  // branches in `.toVar()` forces the builder to evaluate each side
  // explicitly so the select picks the right concrete result.
  const rawPixelWidth: TSLNode = int(uIsOrtho)
    .equal(int(1))
    .select(rawPixelWidthOrtho.toVar(), rawPixelWidthPersp.toVar());

  const minPixelWidth = float(1.5);
  const maxPW: TSLNode = max(uMaxLinePixelWidth, minPixelWidth.add(1.0));
  const clampedPixelWidth: TSLNode = clamp(rawPixelWidth, minPixelWidth, maxPW);

  // Width fade for clamped extreme cases. `.toVar()` on the
  // expression branch so select() picks the right concrete value.
  const vWidthFadeVal: TSLNode = rawPixelWidth
    .lessThanEqual(maxPW)
    .select(float(1.0), maxPW.div(max(rawPixelWidth, float(1e-4))).toVar());

  // Pathological-segment cull: both endpoints inside near-cull margin
  // AND rawPixelWidth blows past clamp by 2× → degenerate quad.
  const pathological: TSLNode = startDepth
    .lessThan(nearCull.mul(2.0))
    .and(endDepth.lessThan(nearCull.mul(2.0)))
    .and(rawPixelWidth.greaterThan(maxPW.mul(2.0)));

  // Final clip-space position with perpendicular expansion.
  // pixelOffset = perpendicular × aQuadCorner.y × clampedPixelWidth
  // ndcOffset = pixelOffset / uResolution × 2
  // clipPos.xy += ndcOffset × clipPos.w
  const pixelOffset: TSLNode = perpendicular.mul(aQuadCorner.y).mul(clampedPixelWidth);
  const ndcOffset: TSLNode = pixelOffset.div(uResolution).mul(2.0);
  // vec4(vec2, scalar, scalar) — vec4(vec2, vec2) isn't a supported
  // TSL overload. Pass clipPosBase.z and .w as individual scalars.
  const expandedClip: TSLNode = vec4(
    clipPosBase.xy.add(ndcOffset.mul(clipPosBase.w)),
    clipPosBase.z,
    clipPosBase.w
  );

  // Discard culled / pathological segments by routing to off-screen.
  // vec4(2, 2, 2, 1) is outside NDC clip cube — rasterizer produces
  // no fragments.
  // Discard culled / pathological segments by routing to off-screen.
  // vec4(2, 2, 2, 1) is outside NDC clip cube — rasterizer produces
  // no fragments. Branches wrapped in `.toVar()` for the same reason
  // as rawPixelWidth above: TSL's select() needs materialised values
  // on each side to dispatch correctly.
  const offscreen: TSLNode = vec4(2.0, 2.0, 2.0, 1.0);
  const culled: TSLNode = bothBehind.or(pathological);
  const clipPos: TSLNode = culled.select(offscreen.toVar(), expandedClip.toVar());

  // Varyings to the fragment stage.
  const vColor: TSLNode = varying(perPointColor);
  const vSharpness: TSLNode = varying(vSharpnessVal);
  const vPerpNorm: TSLNode = varying(aQuadCorner.y);
  const vT: TSLNode = varying(t);
  const vSegmentLength: TSLNode = varying(aSegmentLength);
  const vWidthAtT: TSLNode = varying(width);
  const vPixelWidth: TSLNode = varying(rawPixelWidth);
  const vWidthFade: TSLNode = varying(vWidthFadeVal);
  // Clipped flags are per-instance — same across all 4 quad verts.
  const vClippedStart: TSLNode = varying(aStartClipped);
  const vClippedEnd: TSLNode = varying(aEndClipped);

  // ---- Fragment computation ----

  const colorNode = Fn(() => {
    const p: TSLNode = vPerpNorm.abs();
    Discard(p.greaterThanEqual(1.0));

    // Parabolic falloff (1 - p²)^sharpness.
    const perpFalloff: TSLNode = max(float(1.0).sub(p.mul(p)), float(0.0)).pow(
      max(vSharpness, float(0.0001))
    );

    // Edge AA: smoothstep over ~1 pixel.
    const minPW = float(1.5);
    const renderedWidth: TSLNode = max(vPixelWidth, minPW);
    const aaWidth: TSLNode = float(1.0).div(renderedWidth);
    const edgeAA: TSLNode = float(1.0).sub(smoothstep(float(1.0).sub(aaWidth), float(1.0), p));

    const widthScale: TSLNode = min(vPixelWidth.div(minPW), float(1.0));

    // Cap factor — ramps to 1 inside body, 0.5 at endpoints; full at
    // clipped endpoints. Mirrors the GLSL implementation.
    const distFromStart: TSLNode = vT.mul(vSegmentLength);
    const distFromEnd: TSLNode = float(1.0).sub(vT).mul(vSegmentLength);
    const distToNearest: TSLNode = min(distFromStart, distFromEnd);
    const capRamp: TSLNode = vWidthAtT
      .greaterThan(float(1e-4))
      // `.toVar()` on the chained branch — see the vertex-stage
      // rawPixelWidth select for why this is needed.
      .select(clamp(distToNearest.div(vWidthAtT), 0.0, 1.0).toVar(), float(1.0));
    const baseCap: TSLNode = float(0.5).add(capRamp.mul(0.5));
    // nearestIsStart = step(distFromStart, distFromEnd): 1 when
    // distFromEnd ≥ distFromStart → start is nearest.
    const nearestIsStart: TSLNode = step(distFromStart, distFromEnd);
    const nearestClipped: TSLNode = mix(vClippedEnd, vClippedStart, nearestIsStart);
    const capFactor: TSLNode = mix(baseCap, float(1.0), nearestClipped);

    const intensity: TSLNode = capFactor
      .mul(perpFalloff)
      .mul(edgeAA)
      .mul(widthScale)
      .mul(vWidthFade);

    // GOG.
    const adjusted: TSLNode = max(vColor.mul(uIntensity).add(uOffset), vec3(0.0));
    Discard(max(adjusted.r, max(adjusted.g, adjusted.b)).lessThan(1e-4));
    const gammaColor: TSLNode = adjusted.pow(vec3(uInvGamma));

    const alpha: TSLNode = intensity.mul(uOpacity);
    if (premultiplyRGB) {
      return vec4(gammaColor.mul(alpha), alpha);
    }
    return vec4(gammaColor, alpha);
  });

  const material = outMaterial ?? new NodeMaterial();
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.toneMapped = false;

  const blendingMode: BlendingMode = config.blendingMode ?? 'additive';
  const opacityValue = (uniforms.uOpacity?.value as number | undefined) ?? 1.0;
  const blendingState = getCompleteBlendingState(blendingMode, opacityValue);
  applyBlendingStateToMaterial(material, blendingState);
  return material;
}
