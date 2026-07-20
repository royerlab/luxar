/**
 * Line material TSL factory — NodeMaterial counterpart to the GLSL3
 * shaders in `shader-glsl.ts`.
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
 * Fragment stage produces a soft line: a shifted-truncated super-Gaussian
 * perpendicular cross-section `max(exp(-K·p^beta) - C, 0)/(1-C)`
 * (beta = 2^(6s - 2), beta=2 is a truncated Gaussian) × edgeAA × widthScale
 * × widthFade × capFactor (capFactor ramps to full intensity inside the body
 * but is 1.0 at clipped endpoints).
 *
 * @module rendering/materials/line/shader-tsl
 */

import * as THREE from 'three';
import {
  Fn,
  If,
  uniform,
  attribute,
  varying,
  vec2,
  vec3,
  vec4,
  float,
  max,
  min,
  clamp,
  mix,
  length,
  step,
  smoothstep,
  exp,
  texture,
  modelViewMatrix,
  cameraProjectionMatrix,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import {
  perspectiveNearFadeStaticTSL,
  sanitizeNonNegative,
  type TSLNode,
} from '../_shared/tsl-helpers';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  effectiveGeometryMode,
} from '../../blending-state';
import type { BlendingMode } from '../../material-manager';

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
  /**
   * Fast path: skip the `pow(adjusted, vec3(uInvGamma))` call when
   * the wrapper knows gamma == 1.0. Saves 3 fragment-stage pow()
   * calls in the default-gamma case (the common case).
   */
  readonly gammaOne?: boolean;
  /**
   * Fast path: skip the `vColor * uIntensity + uOffset` GOG chain
   * (and its `max(..., vec3(0))` clamp) when the wrapper knows
   * intensity == 1 && offset == 0 — the default and most common
   * configuration. Saves 1 vec3 multiply, 1 vec3 add, and 1 vec3
   * max per fragment.
   */
  readonly noGOG?: boolean;
  /**
   * Camera projection mode at build time. When `true` (orthographic),
   * the factory emits only the ortho pixel-width branch; when `false`
   * or undefined (perspective), only the perspective branch is
   * emitted. Eliminates the runtime `int(uIsOrtho).select(...)` and
   * its `.toVar()` materialisation of the unused branch. Wrapper
   * triggers `rebuildGraph()` whenever the camera mode flips.
   */
  readonly isOrtho?: boolean;
}

/**
 * Pre-created TSL leaf nodes supplied by a wrapper class. Same
 * pattern as `LinePickTSLNodes` / `PointPickTSLNodes`: consumers
 * own the `UniformNode`s and the factory references them directly,
 * which avoids the `.onUpdate('render')` callback bridge that the
 * old uniform-record-based path used. Mutations on the wrapper's
 * `material.uniforms.X.value` (proxied via `proxyIUniform`) land
 * directly on `node.value`.
 *
 * Colormap nodes are optional and bound only when the consumer is
 * built with `config.useColormap === true`. The factory throws if
 * the config says yes but the colormap nodes are missing.
 */
export interface LineTSLNodes {
  readonly uResolution: TSLNode;
  readonly uIsOrtho: TSLNode;
  readonly uNearCull: TSLNode;
  readonly uMaxLinePixelWidth: TSLNode;
  readonly uPerspectiveLineScale: TSLNode;
  readonly uOrthoLineScale: TSLNode;
  readonly uOpacity: TSLNode;
  readonly uInvGamma: TSLNode;
  readonly uIntensity: TSLNode;
  readonly uOffset: TSLNode;
  /** Set only when colormap mode is active. */
  readonly uColormapTex?: TSLNode;
  readonly uScalarMin?: TSLNode;
  readonly uScalarScale?: TSLNode;
}

/**
 * Line-material TSL factory.
 *
 * Consumes pre-created `UniformNode` references via `nodes`; the
 * wrapper class (`LineTSLMaterial`) owns those nodes and exposes
 * them through `material.uniforms` as `IUniform`-shaped
 * getter/setter proxies. The harness / `LINE_SOURCE` ShaderSource
 * registry constructs the nodes from a plain `uniforms` record via
 * {@link buildLineTSLNodesFromUniforms}.
 */
export function lineWebGPUFactory(
  nodes: LineTSLNodes,
  config: LineTSLConfig = {},
  outMaterial?: NodeMaterial
): NodeMaterial {
  // Per-vertex.
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  // Per-instance — endpoint pairs.
  const aStartPos: TSLNode = attribute<'vec3'>('aStartPos', 'vec3');
  const aEndPos: TSLNode = attribute<'vec3'>('aEndPos', 'vec3');
  // Per-vertex colours are only read in the non-colormap branch. Under
  // colormap mode the colour comes from the LUT, so these attributes are
  // omitted entirely — parity with the GLSL backend, which gates them under
  // `#ifndef USE_COLORMAP` to keep the active-attribute count within
  // GL_MAX_VERTEX_ATTRIBS (16) alongside the scalar pair.
  const aStartColor: TSLNode | null = config.useColormap
    ? null
    : attribute<'vec3'>('aStartColor', 'vec3');
  const aEndColor: TSLNode | null = config.useColormap
    ? null
    : attribute<'vec3'>('aEndColor', 'vec3');
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
  const aEndScalar: TSLNode = config.useColormap ? attribute<'float'>('aEndScalar', 'float') : null;

  // Bind directly to the persistent `UniformNode`s owned by the
  // wrapper class (or by `buildLineTSLNodesFromUniforms` for the
  // harness path). Mutations on `material.uniforms.X.value` go via
  // `proxyIUniform` straight to `node.value` — no per-render
  // `.onUpdate` callbacks needed.
  // No FOV uniform exists: the TSL graph reads the CPU-precomputed
  // `uPerspectiveLineScale` / `uOrthoLineScale` instead.
  // uIsOrtho is also intentionally absent — projection mode is a
  // JS-level config branch (`config.isOrtho`), not a runtime uniform.
  const uResolution = nodes.uResolution;
  const uNearCull = nodes.uNearCull;
  const uMaxLinePixelWidth = nodes.uMaxLinePixelWidth;
  const uPerspectiveLineScale = nodes.uPerspectiveLineScale;
  const uOrthoLineScale = nodes.uOrthoLineScale;
  const uOpacity = nodes.uOpacity;
  const uInvGamma = nodes.uInvGamma;
  const uIntensity = nodes.uIntensity;
  const uOffset = nodes.uOffset;
  if (config.useColormap) {
    if (!nodes.uColormapTex || !nodes.uScalarMin || !nodes.uScalarScale) {
      throw new Error(
        'lineWebGPUFactory: config.useColormap=true but nodes.uColormapTex / uScalarMin / uScalarScale are not bound.'
      );
    }
  }
  const uColormapTex = config.useColormap ? nodes.uColormapTex! : null;
  const uScalarMin = config.useColormap ? nodes.uScalarMin! : null;
  const uScalarScale = config.useColormap ? nodes.uScalarScale! : null;

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

  // Per-endpoint colour or LUT lookup. Branch on `config.useColormap` so
  // the attribute sets are mutually exclusive (scalars XOR colours),
  // matching the GLSL `#ifndef USE_COLORMAP` split. Non-null assertions are
  // safe: under colormap the scalar attributes are bound (same gate) and the
  // colormap uniforms are validated by the throw above; otherwise the colour
  // attributes are bound.
  let perPointColor: TSLNode;
  if (config.useColormap) {
    // Colormap mode: display range (uScalarMin/uScalarScale) and gamma
    // shape the scalar VALUE before the LUT lookup, not the resulting
    // color; intensity/offset apply POST-LUT in the fragment stage
    // (matching the gsplat shader). gammaOne skips the pow() when
    // gamma == 1.0.
    const s: TSLNode = mix(aStartScalar!, aEndScalar!, t);
    const st0: TSLNode = clamp(s.sub(uScalarMin!).mul(uScalarScale!), 0.0, 1.0);
    const st: TSLNode = config.gammaOne ? st0 : st0.pow(uInvGamma);
    perPointColor = uColormapTex!.sample(vec2(st, 0.5)).rgb;
  } else {
    perPointColor = mix(aStartColor!, aEndColor!, t);
  }

  // Sanitised widths / sharpness, interpolated. Sharpness is authored in
  // [0, 1] and maps (in the fragment) to the super-Gaussian exponent
  // beta = 2^(6s - 2). sanitizeNonNegative keeps a valid s=0 (-> beta=0.25)
  // and routes NaN/Inf/negative to the 0.5 default; clamp bounds [0, 1].
  // (NOT sanitizePositive — that would wrongly reject s=0.) Mirrors GLSL.
  const startW: TSLNode = sanitizeNonNegative(aStartWidth, float(0.0));
  const endW: TSLNode = sanitizeNonNegative(aEndWidth, float(0.0));
  const startS: TSLNode = clamp(sanitizeNonNegative(aStartSharpness, float(0.5)), 0.0, 1.0);
  const endS: TSLNode = clamp(sanitizeNonNegative(aEndSharpness, float(0.5)), 0.0, 1.0);
  const width: TSLNode = mix(startW, endW, t);
  // Interpolated [0, 1] sharpness KNOB; beta computed in the fragment.
  const vSharpnessVal: TSLNode = mix(startS, endS, t);

  // Project endpoints to view + clip space.
  const mvStart: TSLNode = modelViewMatrix.mul(vec4(aStartPos, 1.0));
  const mvEnd: TSLNode = modelViewMatrix.mul(vec4(aEndPos, 1.0));
  const mvPos: TSLNode = mix(mvStart, mvEnd, t);

  // Near-plane / behind-camera safety — PERSPECTIVE ONLY (compile-time
  // graph variant: ortho graphs carry no cull/fade code at all; under
  // ortho NDC clipping is the sole authority and the previous ungated
  // cull wrongly hid in-frustum lines in the near slab). View-space
  // depth = -z.
  const nearCull: TSLNode = max(uNearCull, float(1e-4));
  const startDepth: TSLNode = mvStart.z.negate();
  const endDepth: TSLNode = mvEnd.z.negate();
  const bothBehind: TSLNode | null = config.isOrtho
    ? null
    : startDepth.lessThan(nearCull).and(endDepth.lessThan(nearCull));

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

  // World-space → pixel conversion. Each camera projection mode is a
  // separate graph variant (`config.isOrtho`) so the unused branch
  // never materialises into generated code. The wrapper calls
  // `rebuildGraph()` whenever the camera mode flips. Perspective uses
  // view-space depth (-mvPos.z) — drops a sqrt and is more
  // projection-correct (screen size scales with view-z, not Euclidean
  // distance from the camera position).
  let rawPixelWidth: TSLNode;
  if (config.isOrtho) {
    rawPixelWidth = width.mul(uOrthoLineScale);
  } else {
    const distView: TSLNode = max(mvPos.z.negate(), nearCull);
    rawPixelWidth = width.mul(uPerspectiveLineScale).div(distView);
  }

  const minPixelWidth = float(1.5);
  const maxPW: TSLNode = max(uMaxLinePixelWidth, minPixelWidth.add(1.0));
  const clampedPixelWidth: TSLNode = clamp(rawPixelWidth, minPixelWidth, maxPW);

  // Width fade for clamped extreme cases. `.toVar()` on the
  // expression branch so select() picks the right concrete value.
  const vWidthFadeVal: TSLNode = rawPixelWidth
    .lessThanEqual(maxPW)
    .select(float(1.0), maxPW.div(max(rawPixelWidth, float(1e-4))).toVar());

  // Pathological-segment cull (perspective only — ortho width is
  // depth-independent, a depth gate there is meaningless): both
  // endpoints inside near-cull margin AND rawPixelWidth blows past
  // clamp by 2× → degenerate quad.
  const pathological: TSLNode | null = config.isOrtho
    ? null
    : startDepth
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

  // Route culled / pathological segments to off-screen via real
  // TSL control flow. `If(predicate, () => { ... })` emits actual
  // `if` blocks in the generated WGSL/GLSL so only one branch runs
  // per vertex — unlike `select(...)` which evaluates both. Ortho
  // graphs (culls null) skip the wrapper entirely — dead code drops
  // from the ortho codegen, consistent with the config.isOrtho
  // graph-variant design above. Sentinel vec4(0,0,-2,1) matches the
  // point/gsplat reject convention.
  const culled: TSLNode | null =
    bothBehind && pathological ? bothBehind.or(pathological) : (bothBehind ?? pathological);
  const clipPos: TSLNode = culled
    ? Fn(() => {
        const out = vec4(0.0, 0.0, -2.0, 1.0).toVar('clipPos');
        If(culled.not(), () => {
          out.assign(expandedClip);
        });
        return out;
      })()
    : expandedClip;

  // Varyings to the fragment stage. Per-segment-constant values use
  // `flat` interpolation so the rasterizer skips the perspective
  // divide — matches the GLSL3 `flat` qualifier on the same fields.
  const vColor: TSLNode = varying(perPointColor);
  const vSharpness: TSLNode = varying(vSharpnessVal);
  const vPerpNorm: TSLNode = varying(aQuadCorner.y);
  const vT: TSLNode = varying(t);
  const vSegmentLength: TSLNode = varying(aSegmentLength).setInterpolation('flat');
  const vWidthAtT: TSLNode = varying(width);
  const vPixelWidth: TSLNode = varying(rawPixelWidth);
  const vWidthFade: TSLNode = varying(vWidthFadeVal);
  // View-space z travels to the FRAGMENT, which computes the near fade
  // per-fragment — interpolating the FADE itself is wrong on long
  // segments (fade(lerp(z)) != lerp(fade(z)); one endpoint at the
  // camera plane would dim mid-segment fragments far outside the fade
  // band). Ortho graphs skip the varying entirely (compile-time
  // variant; fade is identically 1).
  const vViewZ: TSLNode | null = config.isOrtho ? null : varying(mvPos.z);
  // Clipped flags are per-instance — same across all 4 quad verts.
  const vClippedStart: TSLNode = varying(aStartClipped).setInterpolation('flat');
  const vClippedEnd: TSLNode = varying(aEndClipped).setInterpolation('flat');

  // ---- Fragment computation ----

  const colorNode = Fn(() => {
    const p: TSLNode = vPerpNorm.abs();
    Discard(p.greaterThanEqual(1.0));

    // Shifted-truncated super-Gaussian perpendicular cross-section:
    // max(exp(-K * p^beta) - C, 0) / (1 - C), C0-continuous at the line
    // edge. The [0, 1] sharpness KNOB maps to beta = 2^(6s - 2) (s=0.5 ->
    // beta=2, a truncated Gaussian). K = ln(1/floor), floor = 0.01.
    // Mirrors the GLSL3 fragment exactly.
    const K = 4.6051702; // ln(100)
    const C = 0.01; // exp(-K) = floor
    const invOneMinusC = 1.0 / (1.0 - C);
    const beta: TSLNode = float(2.0).pow(vSharpness.mul(6.0).sub(2.0));
    const perpFalloff: TSLNode = exp(p.pow(beta).mul(-K)).sub(C).max(float(0.0)).mul(invOneMinusC);

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
      .mul(vWidthFade)
      .mul(vViewZ ? perspectiveNearFadeStaticTSL(false, vViewZ, nearCull) : float(1.0));

    // GOG. uIntensity (gain) + uOffset apply in BOTH modes so the layer
    // intensity/offset controls work for a colormapped line too (GLSL
    // parity, matching the gsplat shader). Colormap mode: gamma +
    // display-range shaped the scalar VALUE pre-LUT (vertex stage), so
    // only gain/offset apply post-LUT (no extra gamma). Fast path: when
    // the wrapper knows intensity==1 && offset==0, the mul/add/clamp
    // chain is identity for non-negative vColor (noGOG).
    const adjusted: TSLNode = config.noGOG
      ? vColor
      : max(vColor.mul(uIntensity).add(uOffset), vec3(0.0));
    Discard(max(adjusted.r, max(adjusted.g, adjusted.b)).lessThan(1e-4));
    // Gamma fast path: when the wrapper knows gamma==1.0 the pow() is
    // identity. JS-level branch so the generated WGSL/GLSL omits the
    // pow entirely when not needed. Colormap mode also skips it (gamma
    // is applied to the value, not the color).
    const gammaColor: TSLNode =
      config.useColormap || config.gammaOne ? adjusted : adjusted.pow(vec3(uInvGamma));

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

  // Phase-1 volumetric fallback: this factory tail is the ONLY state
  // writer at TSL construction (the ctor never calls applyBlendingMode,
  // unlike the GLSL twin) AND re-runs on every rebuildGraph — so it must
  // apply the same volumetric→additive interception as the wrapper, or a
  // volumetric lines node would pair the premultiplied One/
  // OneMinusSrcAlpha state with this alpha-weighted shader (full-strength
  // RGB that DARKENS what's behind it — the opposite of the κ=0 limit).
  const requestedMode: BlendingMode = config.blendingMode ?? 'additive';
  const blendingMode: BlendingMode = effectiveGeometryMode(requestedMode, 'line');
  const opacityValue = (nodes.uOpacity.value as number | undefined) ?? 1.0;
  const blendingState = getCompleteBlendingState(blendingMode, opacityValue);
  applyBlendingStateToMaterial(material, blendingState);
  return material;
}

/**
 * Build a `LineTSLNodes` set from a plain `IUniform` record. Used by
 * the test harness and the `LINE_SOURCE` ShaderSource factory in
 * `shader-glsl.ts` — callers that don't own persistent
 * wrapper-side `UniformNode`s. Mirrors
 * `buildLinePickTSLNodesFromUniforms`.
 *
 * Note: the resulting nodes capture the current `iuniform.value` at
 * build time. Mutations to the host `IUniform`'s `.value` after this
 * function returns will NOT propagate — appropriate for the harness
 * (which builds once and renders once) but not for live wrappers
 * (which must use `proxyIUniform` against persistent nodes).
 */
export function buildLineTSLNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>,
  config: LineTSLConfig = {}
): LineTSLNodes {
  const base: LineTSLNodes = {
    uResolution: uniform(
      (uniforms.uResolution?.value as THREE.Vector2 | undefined) ?? new THREE.Vector2(1, 1)
    ),
    uIsOrtho: uniform((uniforms.uIsOrtho?.value as number) ?? 0),
    uNearCull: uniform((uniforms.uNearCull?.value as number) ?? 1e-4),
    uMaxLinePixelWidth: uniform((uniforms.uMaxLinePixelWidth?.value as number) ?? 1.0),
    uPerspectiveLineScale: uniform((uniforms.uPerspectiveLineScale?.value as number) ?? 1.0),
    uOrthoLineScale: uniform((uniforms.uOrthoLineScale?.value as number) ?? 1.0),
    uOpacity: uniform((uniforms.uOpacity?.value as number) ?? 1.0),
    uInvGamma: uniform((uniforms.uInvGamma?.value as number) ?? 1.0),
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
