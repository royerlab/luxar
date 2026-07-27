/**
 * Line material TSL factory — NodeMaterial counterpart to the GLSL3
 * shaders in `shader-glsl.ts`.
 *
 * Renders each line segment as a screen-space-thick instanced quad.
 * Per-vertex `aQuadCorner` (±1) determines:
 *   - x ∈ {-1, +1}: position along the segment (interpolation t).
 *   - y ∈ {-1, +1}: perpendicular offset (used for width expansion).
 *
 * Per-segment data comes from the RGBA32F line texture (`uLineTex`,
 * 6 texels/segment — layout in `rendering/line-geometry.ts` /
 * `rendering/element-texture-layout.ts`), fetched in the vertex stage
 * via `textureLoad` and indexed by the only per-instance attribute:
 *   - aSortedIndex (uint) — draw-slot → storage-slot mapping
 *     (identity after a fresh commit; permuted by the sort worker)
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
 * but is 1.0 at clipped endpoints). `blendingMode: 'volumetric'` selects
 * the emission–absorption output branch at graph build time (transverse
 * chord integral through the width profile — materials/line/math.ts).
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
  int,
  ivec2,
  max,
  min,
  clamp,
  mix,
  length,
  step,
  smoothstep,
  exp,
  texture,
  textureSize,
  modelViewMatrix,
  cameraProjectionMatrix,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { getPlaceholderElementTexture } from '../../element-texture-layout';
import {
  perspectiveNearFadeStaticTSL,
  sanitizeAlpha,
  sanitizeNonNegative,
  type TSLNode,
} from '../_shared/tsl-helpers';
import {
  ALPHA_CLAMP,
  VOLUMETRIC_SERIES_C1,
  VOLUMETRIC_SERIES_C2_DIVISOR,
  VOLUMETRIC_SERIES_TAU_THRESHOLD,
  VOLUMETRIC_TAU_EPS,
} from '../_shared/volumetric';
import { LINE_CHORD_SCALE } from './math';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  isVolumetricMode,
} from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';

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
 *
 * The colormap AND line texture nodes are factory-time bound (TSL
 * `texture(...)` captures the THREE.Texture at call time); the
 * wrapper rebuilds the graph when either texture's identity changes
 * (see `setColormapTexture` / `updateLineTexture` in the wrapper).
 */
export interface LineTSLNodes {
  /**
   * Line data texture node (RGBA32F, 6 texels/segment). Every line
   * material has one; the pool commit rebinds it per node via
   * `updateLineTexture`.
   */
  readonly uLineTex: TSLNode;
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
  /**
   * κ — composed node absorption (volumetric mode). Read only when the
   * graph was built with `blendingMode: 'volumetric'`; a plain runtime
   * uniform otherwise (mirrors the point/gsplat factories).
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
  // The only per-instance attribute: segment data itself lives in the
  // line texture; `aSortedIndex` maps the draw slot to a storage slot
  // (identity after a fresh commit, permuted by the sort worker).
  const aSortedIndex: TSLNode = attribute<'uint'>('aSortedIndex', 'uint');

  // Bind directly to the persistent `UniformNode`s owned by the
  // wrapper class (or by `buildLineTSLNodesFromUniforms` for the
  // harness path). Mutations on `material.uniforms.X.value` go via
  // `proxyIUniform` straight to `node.value` — no per-render
  // `.onUpdate` callbacks needed.
  // No FOV uniform exists: the TSL graph reads the CPU-precomputed
  // `uPerspectiveLineScale` / `uOrthoLineScale` instead.
  // uIsOrtho is also intentionally absent — projection mode is a
  // JS-level config branch (`config.isOrtho`), not a runtime uniform.
  const uLineTex = nodes.uLineTex;
  const uResolution = nodes.uResolution;
  const uNearCull = nodes.uNearCull;
  const uMaxLinePixelWidth = nodes.uMaxLinePixelWidth;
  const uPerspectiveLineScale = nodes.uPerspectiveLineScale;
  const uOrthoLineScale = nodes.uOrthoLineScale;
  const uOpacity = nodes.uOpacity;
  const uInvGamma = nodes.uInvGamma;
  const uIntensity = nodes.uIntensity;
  const uOffset = nodes.uOffset;
  const uAbsorption = nodes.uAbsorption;
  const uHasElementAlpha = nodes.uHasElementAlpha;
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

  // The volumetric (emission–absorption) output branch is chosen at
  // GRAPH BUILD time — a JS conditional, exactly like the point/gsplat
  // factories (TSL `.select()` is avoided for structural branches).
  // The wrapper's `applyBlendingMode` rebuilds the graph on any
  // volumetric crossing via the LUXAR_VOLUMETRIC define.
  const volumetricGraph = isVolumetricMode(config.blendingMode ?? 'additive');

  // ---- Vertex computation ----
  //
  // The ENTIRE vertex stage is traced inside a single Fn() body with
  // explicit `.toVar()` statements — the same load-bearing structure
  // as the point/gsplat factories (materials/point/shader-tsl.ts): as
  // a free expression tree, TSL materializes a shared subexpression at
  // its FIRST traversal use, which can land inside a `.select()`
  // branch and read uninitialized on the other path. Inside Fn(),
  // statements emit in trace order, unconditionally.

  // Varyings are declared up front and `.assign()`ed inside the vertex
  // body (the TSL pattern for Fn-traced vertex stages).
  // Per-segment-constant values use `flat` interpolation so the
  // rasterizer skips the perspective divide — matches the GLSL3 `flat`
  // qualifier on the same fields.
  const vColor: TSLNode = varying(vec3(float(0.0), float(0.0), float(0.0)));
  const vSharpness: TSLNode = varying(float(0.0));
  const vPerpNorm: TSLNode = varying(float(0.0));
  const vT: TSLNode = varying(float(0.0));
  const vSegmentLength: TSLNode = varying(float(0.0)).setInterpolation('flat');
  const vWidthAtT: TSLNode = varying(float(0.0));
  const vPixelWidth: TSLNode = varying(float(0.0));
  const vWidthFade: TSLNode = varying(float(0.0));
  // View-space z travels to the FRAGMENT, which computes the near fade
  // per-fragment — interpolating the FADE itself is wrong on long
  // segments (fade(lerp(z)) != lerp(fade(z)); one endpoint at the
  // camera plane would dim mid-segment fragments far outside the fade
  // band). Ortho graphs skip the varying entirely (compile-time
  // variant; fade is identically 1).
  const vViewZ: TSLNode | null = config.isOrtho ? null : varying(float(0.0));
  // Clipped flags are per-instance — same across all 4 quad verts.
  const vCapSuppressStart: TSLNode = varying(float(0.0)).setInterpolation('flat');
  const vCapSuppressEnd: TSLNode = varying(float(0.0)).setInterpolation('flat');
  // Per-endpoint opacity, interpolated along the segment (deliberately
  // NON-flat: the start/end alphas differ, matching the GLSL twin's
  // smooth `out float vAlpha`).
  const vAlpha: TSLNode = varying(float(1.0));

  const nearCull: TSLNode = max(uNearCull, float(1e-20));

  const vertexBody = Fn(() => {
    // === Line-texture fetch prologue ===
    // textureLoad reads reconstruct the per-segment values into the
    // exact local names the math below has always used — zero changes
    // downstream of this block. Every value is a `.toVar()` STATEMENT
    // (the Fn house rule; see the block comment above). The texture
    // width is a multiple of 6 (element-texture-layout.ts), so a
    // segment's 6 texels share one row and only x advances. texel5
    // carries the colormap scalars (.xy, read under useColormap) and
    // the per-endpoint alphas (.zw, written unconditionally by the
    // texel writer — 1.0 for RGB data), so it is fetched in every
    // mode. Unlike the GLSL twin, texels 2/3/5 are NOT deferred past
    // the bothBehind cull — the Fn trace-order house rule emits
    // statements unconditionally, so the TSL backend pays the extra
    // loads per culled vertex (accepted asymmetry, output-identical;
    // same trade as the point factory).
    const lineBase: TSLNode = int(aSortedIndex).mul(int(6)).toVar();
    // int() wrap is LOAD-BEARING: TSL types textureSize() as uint (the
    // WGSL textureDimensions convention), but the WebGL2 fallback emits
    // GLSL textureSize() which returns int -- without the explicit
    // conversion the generated `uint nodeVar = textureSize(...).x;`
    // fails to compile on the forceWebGL backend.
    const lineTexW: TSLNode = int((textureSize(uLineTex, int(0)) as unknown as TSLNode).x).toVar();
    const texelX: TSLNode = lineBase.mod(lineTexW).toVar();
    const texelY: TSLNode = lineBase.div(lineTexW).toVar();
    const lineT0: TSLNode = uLineTex.load(ivec2(texelX, texelY)).toVar();
    const lineT1: TSLNode = uLineTex.load(ivec2(texelX.add(int(1)), texelY)).toVar();
    const lineT2: TSLNode = uLineTex.load(ivec2(texelX.add(int(2)), texelY)).toVar();
    const lineT3: TSLNode = uLineTex.load(ivec2(texelX.add(int(3)), texelY)).toVar();
    const lineT4: TSLNode = uLineTex.load(ivec2(texelX.add(int(4)), texelY)).toVar();
    const lineT5: TSLNode = uLineTex.load(ivec2(texelX.add(int(5)), texelY)).toVar();
    const aStartPos: TSLNode = vec3(lineT0).toVar();
    const aStartWidth: TSLNode = lineT0.w.toVar();
    const aEndPos: TSLNode = vec3(lineT1).toVar();
    const aEndWidth: TSLNode = lineT1.w.toVar();
    const aStartSharpness: TSLNode = lineT2.w.toVar();
    const aEndSharpness: TSLNode = lineT3.w.toVar();
    const aSegmentLength: TSLNode = lineT4.x.toVar();
    const aStartCapSuppress: TSLNode = lineT4.y.toVar();
    const aEndCapSuppress: TSLNode = lineT4.z.toVar();

    // t ∈ {0, 1} — position along the segment. Branchless because
    // aQuadCorner.x ∈ {-1, +1} by construction.
    const t: TSLNode = aQuadCorner.x.mul(0.5).add(0.5).toVar();

    // Per-endpoint colour or LUT lookup. Branch on `config.useColormap`
    // (JS-level graph variant, matching the GLSL `#ifdef USE_COLORMAP`
    // split — colors from texels 2/3 XOR scalars from texel5).
    let perPointColor: TSLNode;
    if (config.useColormap) {
      // Colormap mode: display range (uScalarMin/uScalarScale) and gamma
      // shape the scalar VALUE before the LUT lookup, not the resulting
      // color; intensity/offset apply POST-LUT in the fragment stage
      // (matching the gsplat shader). gammaOne skips the pow() when
      // gamma == 1.0.
      const s: TSLNode = mix(lineT5.x, lineT5.y, t);
      const st0: TSLNode = clamp(s.sub(uScalarMin!).mul(uScalarScale!), 0.0, 1.0);
      const st: TSLNode = config.gammaOne ? st0 : st0.pow(uInvGamma);
      perPointColor = uColormapTex!.sample(vec2(st, 0.5)).rgb;
    } else {
      perPointColor = mix(vec3(lineT2), vec3(lineT3), t);
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
    const width: TSLNode = mix(startW, endW, t).toVar();
    // Interpolated [0, 1] sharpness KNOB; beta computed in the fragment.
    const vSharpnessVal: TSLNode = mix(startS, endS, t);

    // Project endpoints to view + clip space.
    const mvStart: TSLNode = modelViewMatrix.mul(vec4(aStartPos, 1.0)).toVar();
    const mvEnd: TSLNode = modelViewMatrix.mul(vec4(aEndPos, 1.0)).toVar();
    const mvPos: TSLNode = mix(mvStart, mvEnd, t).toVar();

    // Near-plane / behind-camera safety — PERSPECTIVE ONLY (compile-time
    // graph variant: ortho graphs carry no cull/fade code at all; under
    // ortho NDC clipping is the sole authority and the previous ungated
    // cull wrongly hid in-frustum lines in the near slab). View-space
    // depth = -z.
    // uNearCull is scene-bounds-scaled; the 1e-20 floor only guards
    // uNearCull == 0 (degenerate smoothstep / division). An absolute
    // 1e-4 floor overrode the scene-relative value on tiny-unit scenes —
    // every segment sat inside the "both behind" margin and was culled.
    // GLSL twin: shader-glsl.ts.
    const startDepth: TSLNode = mvStart.z.negate().toVar();
    const endDepth: TSLNode = mvEnd.z.negate().toVar();
    const bothBehind: TSLNode | null = config.isOrtho
      ? null
      : startDepth.lessThan(nearCull).and(endDepth.lessThan(nearCull));

    const clipStart: TSLNode = cameraProjectionMatrix.mul(mvStart).toVar();
    const clipEnd: TSLNode = cameraProjectionMatrix.mul(mvEnd).toVar();
    // projection is linear, so proj * mix(a,b,t) == mix(proj*a, proj*b, t).
    const clipPosBase: TSLNode = mix(clipStart, clipEnd, t).toVar();

    // Convert clip endpoints to pixel space for aspect-correct
    // perpendicular expansion. Guard tiny .w (near-plane crossings) with
    // the SCENE-RELATIVE nearCull (w == -viewZ under perspective), not an
    // absolute epsilon: 1e-4 clamped VALID w on tiny-unit scenes and
    // scrambled quad directions, while a raw 1e-20 floor could overflow
    // float32 in the pixel-length math for behind-camera endpoints.
    // Ortho graphs: w == 1 exactly, guard 1.0 is inert (compile-time
    // variant). GLSL twin: shader-glsl.ts.
    const wGuard: TSLNode = config.isOrtho ? float(1.0) : nearCull;
    const wStart: TSLNode = max(clipStart.w, wGuard);
    const wEnd: TSLNode = max(clipEnd.w, wGuard);
    const ndcStart: TSLNode = vec2(clipStart.xy.div(wStart)).toVar();
    const ndcEnd: TSLNode = vec2(clipEnd.xy.div(wEnd)).toVar();

    // Direction + perpendicular (pixel space, aspect-correct). The +0.5
    // in (ndc*0.5+0.5)*resolution cancels under subtraction, so the
    // pixel-space direction is just (ndcEnd-ndcStart)*(0.5*resolution).
    const pixelDir: TSLNode = vec2(ndcEnd.sub(ndcStart).mul(uResolution.mul(0.5))).toVar();
    const pixelLen: TSLNode = length(pixelDir).toVar();
    const lineDir: TSLNode = pixelLen
      .greaterThan(0.0001)
      .select(vec2(pixelDir.div(pixelLen)).toVar(), vec2(1.0, 0.0))
      .toVar();
    const perpendicular: TSLNode = vec2(lineDir.y.negate(), lineDir.x).toVar();

    // World-space → pixel conversion. Each camera projection mode is a
    // separate graph variant (`config.isOrtho`) so the unused branch
    // never materialises into generated code. The wrapper calls
    // `rebuildGraph()` whenever the camera mode flips. Perspective uses
    // view-space depth (-mvPos.z) — drops a sqrt and is more
    // projection-correct (screen size scales with view-z, not Euclidean
    // distance from the camera position).
    let rawPixelWidth: TSLNode;
    if (config.isOrtho) {
      rawPixelWidth = width.mul(uOrthoLineScale).toVar();
    } else {
      const distView: TSLNode = max(mvPos.z.negate(), nearCull);
      rawPixelWidth = width.mul(uPerspectiveLineScale).div(distView).toVar();
    }

    const minPixelWidth = float(1.5);
    const maxPW: TSLNode = max(uMaxLinePixelWidth, minPixelWidth.add(1.0)).toVar();
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
    ).toVar();

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
    const clipPosOut: TSLNode = vec4(0.0, 0.0, -2.0, 1.0).toVar('clipPos');
    if (culled) {
      If(culled.not(), () => {
        clipPosOut.assign(expandedClip);
      });
    } else {
      clipPosOut.assign(expandedClip);
    }

    // Assign varyings (declared outside the Fn; see above).
    vColor.assign(perPointColor);
    vSharpness.assign(vSharpnessVal);
    vPerpNorm.assign(aQuadCorner.y);
    vT.assign(t);
    vSegmentLength.assign(aSegmentLength);
    vWidthAtT.assign(width);
    vPixelWidth.assign(rawPixelWidth);
    vWidthFade.assign(vWidthFadeVal);
    if (vViewZ) vViewZ.assign(mvPos.z);
    vCapSuppressStart.assign(aStartCapSuppress);
    vCapSuppressEnd.assign(aEndCapSuppress);
    // Each texel read sanitized BEFORE the mix: NaN/Inf route to the
    // 1.0 opaque identity (loud), finite values clamp to [0, 1] (alpha
    // is load-bearing in every mode and feeds optical depth under
    // volumetric). The guarantee is "NaN never reaches τ/pixels" — a
    // NaN source vertex already poisons BOTH texel alphas upstream in
    // the worker's lerp kernel, so the whole segment renders
    // loud-opaque. Mirrors the GLSL twin.
    vAlpha.assign(mix(sanitizeAlpha(lineT5.z), sanitizeAlpha(lineT5.w), t));

    return clipPosOut;
  });

  const clipPos: TSLNode = vertexBody();

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

    // Cap factor — ramps to 1 inside body, 0.5 at FREE endpoints; lifted
    // back to 1 by the per-endpoint suppression scalar (slice-clipped
    // endpoints and straight-through interior joints). Mirrors GLSL.
    const distFromStart: TSLNode = vT.mul(vSegmentLength);
    const distFromEnd: TSLNode = float(1.0).sub(vT).mul(vSegmentLength);
    const distToNearest: TSLNode = min(distFromStart, distFromEnd);
    // distToNearest / vWidthAtT is a scale-free ratio (both world
    // units) — the guard is a pure div-by-zero threshold at 1e-20 (an
    // absolute 1e-4 skipped the ramp for valid sub-1e-4-unit widths).
    const capRamp: TSLNode = vWidthAtT
      .greaterThan(float(1e-20))
      // `.toVar()` on the chained branch — see the vertex-stage
      // rawPixelWidth select for why this is needed.
      .select(clamp(distToNearest.div(vWidthAtT), 0.0, 1.0).toVar(), float(1.0));
    const baseCap: TSLNode = float(0.5).add(capRamp.mul(0.5));
    // nearestIsStart = step(distFromStart, distFromEnd): 1 when
    // distFromEnd ≥ distFromStart → start is nearest.
    const nearestIsStart: TSLNode = step(distFromStart, distFromEnd);
    const nearestSuppress: TSLNode = mix(vCapSuppressEnd, vCapSuppressStart, nearestIsStart);
    const capFactor: TSLNode = mix(baseCap, float(1.0), nearestSuppress);

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
      : max(vColor.mul(uIntensity).add(uOffset), vec3(0.0)).toVar();
    const maxAdjusted: TSLNode = max(adjusted.r, max(adjusted.g, adjusted.b));

    // Screen density of this fragment — the intensity chain already
    // carries every "how much of this line is there" factor; node
    // opacity folds in here. This is the additive-mode alpha.
    const alphaBase: TSLNode = intensity.mul(uOpacity).toVar();

    // Per-endpoint alpha (texel5.zw, interpolated): a plain linear
    // contribution scale in every non-volumetric mode (identity 1.0 for
    // RGB data); volumetric maps it into optical depth
    // w(a) = −ln(1 − a), gated by uHasElementAlpha so the RGB identity
    // 1.0 never maps to w ≈ 6.24 (GLSL twin; clamp = ALPHA_CLAMP from
    // ../_shared/volumetric).
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

    // Volumetric optical depth: the TRANSVERSE special case of the
    // gsplat ray integral — rayMass = density × through-thickness of
    // the Gaussian-profile ribbon (width·√(π/K), materials/line/math.ts).
    const tau: TSLNode | null = volumetricGraph
      ? uAbsorption.mul(alpha).mul(vWidthAtT).mul(float(LINE_CHORD_SCALE)).toVar()
      : null;
    if (volumetricGraph && tau) {
      // Discard only when color AND τ are both negligible — a black
      // line still absorbs (pure-ink occluders keep their optical depth).
      Discard(maxAdjusted.lessThan(1e-4).and(tau.lessThan(1e-4)));
    } else {
      Discard(maxAdjusted.lessThan(1e-4));
    }

    // Gamma fast path: when the wrapper knows gamma==1.0 the pow() is
    // identity. JS-level branch so the generated WGSL/GLSL omits the
    // pow entirely when not needed. Colormap mode also skips it (gamma
    // is applied to the value, not the color).
    const gammaColor: TSLNode =
      config.useColormap || config.gammaOne ? adjusted : adjusted.pow(vec3(uInvGamma));

    if (volumetricGraph && tau) {
      // 'volumetric' output branch: emission–absorption (Max 1995).
      // gammaColor·alpha is exactly what additive adds to the
      // framebuffer, screened by S(τ) = (1−e^(−τ))/τ (series below
      // τ = 1e-3 keeps S(0) = 1 exact — the κ=0 additive limit); alpha
      // out is the physical absorption 1 − e^(−τ) for the
      // One / OneMinusSrcAlpha state. Mirrors the point/gsplat factories.
      const volAlpha: TSLNode = float(1.0).sub(exp(tau.negate()));
      const series: TSLNode = float(1.0)
        .sub(tau.mul(VOLUMETRIC_SERIES_C1))
        .add(tau.mul(tau).div(VOLUMETRIC_SERIES_C2_DIVISOR));
      const screen: TSLNode = tau
        .lessThan(VOLUMETRIC_SERIES_TAU_THRESHOLD)
        .select(series, volAlpha.div(max(tau, VOLUMETRIC_TAU_EPS)));
      return vec4(gammaColor.mul(alpha).mul(screen), volAlpha);
    }
    if (premultiplyRGB) {
      return vec4(gammaColor.mul(alpha), alpha);
    }
    return vec4(gammaColor, alpha);
  });

  const material = outMaterial ?? new NodeMaterial();
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.toneMapped = false;

  // Wire blending state from the shared helper. This factory tail is
  // the ONLY state writer at TSL construction (the ctor never calls
  // applyBlendingMode, unlike the GLSL twin) AND re-runs on every
  // rebuildGraph — so it must derive the state from the same mode the
  // output branch above used.
  const blendingMode: BlendingMode = config.blendingMode ?? 'additive';
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
    // Line data texture — bound from the caller's uniform when present
    // (harness / material paths), else the shared placeholder so
    // codegen-only consumers still build a valid graph.
    uLineTex: texture(
      (uniforms.uLineTex?.value as THREE.Texture | null) ?? getPlaceholderElementTexture()
    ),
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
