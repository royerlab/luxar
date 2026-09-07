/**
 * Mesh material TSL factory — NodeMaterial counterpart to the GLSL3 pair in
 * `shader-glsl.ts`.
 *
 * Draws a shaded indexed triangle surface:
 *   - view-anchored wrapped diffuse and specular terms, no scene light (spec §6.2)
 *   - stored view-space normals OR a screen-space-derivative flat normal, chosen at
 *     GRAPH BUILD time (the `flatNormal` config flag — the twin of the GLSL
 *     `LUXAR_MESH_FLAT_NORMAL` define)
 *   - per-node Gain/Offset/Gamma, with the same `gammaOne` / `noGOG` fast paths the
 *     sibling factories carry
 *   - per-vertex alpha as the sole coverage term, emitted per blending mode
 *   - the shared perspective near fade, evaluated PER FRAGMENT off `vViewPos`
 *     (a triangle spans depth, so a per-vertex value would smear the ramp across
 *     the face) — the twin of the GLSL fragment stage's `perspectiveNearFade`
 *
 * Unlike the three sibling factories there is no element texture, no
 * `aSortedIndex` indirection and no quad expansion: `position`/`normal` are
 * three's own attributes and the draw is an ordinary indexed one. The vertex stage
 * is still traced inside a single `Fn()` with explicit `.toVar()` statements —
 * the house rule from the depth-sorting spec's remediation, because as a free
 * expression tree TSL materializes a shared subexpression at its FIRST traversal
 * use, which can land inside a `.select()` branch and read uninitialized on the
 * other path.
 *
 * The mesh fragment reads no fragcoord, which sidesteps the bottom-left
 * reconstruction trap (`vec2(x, screenSize.y - y)`) entirely.
 *
 * @module rendering/materials/mesh/shader-tsl
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
  float,
  clamp as _clamp,
  cross as _cross,
  dFdx as _dFdx,
  dFdy as _dFdy,
  dot as _dot,
  max as _max,
  mix as _mix,
  normalize as _normalize,
  texture,
  frontFacing,
  modelViewMatrix,
  modelNormalMatrix,
  cameraViewMatrix,
  cameraProjectionMatrix,
  positionGeometry,
  normalGeometry,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import {
  clipDepthVarying,
  glassPartitionGuardTSL,
  glassPartitionNodesFromUniforms,
  type GlassPartitionTSLNodes,
} from '../_shared/glass-partition-tsl';
import { sanitizeAlpha, perspectiveNearFadeTSL, type TSLNode } from '../_shared/tsl-helpers';
import { applyBlendingStateToMaterial, getCompleteBlendingState } from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';
import {
  MESH_DEFAULTS,
  MESH_LIGHT_DIRECTION,
  MESH_NORMAL_EPS_SQ,
  resolveMeshBlendingMode,
  resolveMeshOutput,
  type MeshShadingMode,
} from './appearance';

// Type-erased constructor aliases — same rationale as the point/gsplat TSL
// factories: TSL's typed `vec*` overloads reject many valid combinations of
// intermediate `Node<…>` results. Re-export each as TSLNode-typed to sidestep
// overload-mismatch errors without affecting the generated GLSL/WGSL.
const vec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _vec2 as TSLNode;
const vec3: (a?: TSLNode, b?: TSLNode, c?: TSLNode) => TSLNode = _vec3 as TSLNode;
const vec4: (a?: TSLNode, b?: TSLNode, c?: TSLNode, d?: TSLNode) => TSLNode = _vec4 as TSLNode;
// The same erasure for the vector-math builders. `TSLNode` is `any`, so overload
// resolution on a TSLNode argument picks the FIRST signature — `Node<"float">` for
// `dFdx` — and `cross()` then rejects it as not-a-vec3. The generated GLSL/WGSL is
// unaffected; only the compile-time overload pick is being sidestepped.
const clamp: (v: TSLNode, lo: TSLNode, hi: TSLNode) => TSLNode = _clamp as TSLNode;
const cross: (a: TSLNode, b: TSLNode) => TSLNode = _cross as TSLNode;
const dFdx: (v: TSLNode) => TSLNode = _dFdx as TSLNode;
const dFdy: (v: TSLNode) => TSLNode = _dFdy as TSLNode;
const dot: (a: TSLNode, b: TSLNode) => TSLNode = _dot as TSLNode;
const max: (a: TSLNode, b: TSLNode) => TSLNode = _max as TSLNode;
const mix: (a: TSLNode, b: TSLNode, t: TSLNode) => TSLNode = _mix as TSLNode;
const normalize: (v: TSLNode) => TSLNode = _normalize as TSLNode;

export interface MeshTSLConfig {
  /** Read `aScalar` + the LUT instead of the `color` attribute. */
  readonly useColormap?: boolean;
  /**
   * When `true` (gamma == 1.0), the gamma `pow()` is skipped — `pow(x, 1) == x`.
   * Mirrors the GLSL3 `LUXAR_GAMMA_ONE` define.
   */
  readonly gammaOne?: boolean;
  /**
   * Skip the `vColor * uIntensity + uOffset` GOG chain when the wrapper knows
   * intensity == 1 && offset == 0. Mirrors `LUXAR_NO_GOG`.
   */
  readonly noGOG?: boolean;
  /**
   * How normals are obtained: stored (`'smooth'`), screen-space derivatives
   * (`'flat'`), or not at all (`'none'`, unlit).
   *
   * Mirrors `LUXAR_MESH_FLAT_NORMAL` / `LUXAR_MESH_NO_SHADING`, and like them this
   * is a BUILD-time choice, not a runtime branch: a declared-but-unbound `normal`
   * reads `(0, 0, 0, 1)` rather than "absent", so there is no runtime value meaning
   * "no normals". `createMeshNode` computes it once per node from the authored
   * `shading` plus the stored normals' validity for the active `displayDims`, and
   * hands the same answer to both backends (§6.2).
   *
   * ONE enum rather than two booleans, because `flatNormal && noShading` has no
   * meaning and an enum keeps it from existing.
   */
  readonly shading?: MeshShadingMode;
  /**
   * Sample the base colour from `uBaseColorTex` through the `uv` attribute.
   * Mirrors `LUXAR_MESH_BASE_COLOR_TEX`.
   *
   * Build-time for the same reason `useColormap` is: it decides whether the `uv`
   * attribute and the `vUv` varying exist in the vertex layout at all. WebGPU bakes
   * attribute identity into the pipeline, so this cannot become a runtime branch.
   */
  readonly useBaseColorTexture?: boolean;
  /**
   * Whether the base-colour texture is single-channel, so red is replicated to RGB.
   * Mirrors `LUXAR_MESH_TEX_LUMINANCE`.
   */
  readonly baseColorTextureLuminance?: boolean;
  /**
   * Luxar blending mode. Drives the fragment's emission shape (cutout /
   * premultiplied / alpha-weighted) and the THREE framebuffer state. Defaults to
   * `'opaque'` — the MESH default, unlike the siblings' `'additive'` (§6.3).
   */
  readonly blendingMode?: BlendingMode;
  /**
   * Explicit override for the max-mode RGB premultiply. When undefined it is
   * derived from the blending mode, exactly as in the sibling factories.
   */
  readonly useMaxRGBContribution?: boolean;
}

/**
 * Pre-created TSL leaf nodes supplied by a wrapper class, mirroring
 * `PointTSLNodes` / `LineTSLNodes`. Keys match the public `MeshMaterial.uniforms`
 * names 1:1 so the wrapper's `proxyIUniform` table maps straight across.
 *
 * The colormap nodes are optional and bound only when the consumer is built with
 * `config.useColormap === true`; the factory throws if the config says yes but they
 * are missing. `texture()` captures the `THREE.Texture` at call time, so a swap
 * needs a fresh node plus a graph rebuild (see the wrapper).
 */
export interface MeshTSLNodes {
  readonly uOpacity: TSLNode;
  readonly uInvGamma: TSLNode;
  readonly uIntensity: TSLNode;
  readonly uOffset: TSLNode;
  /** Wrapped-diffuse floor (`1.0` removes the diffuse gradient). */
  readonly uAmbient: TSLNode;
  /** Wrap-term contrast exponent. */
  readonly uShadeExponent: TSLNode;
  /** Additive specular strength. */
  readonly uSpecular: TSLNode;
  /** Specular highlight exponent. */
  readonly uShininess: TSLNode;
  /** Cutout threshold; read only when the graph was built in `opaque` mode. */
  readonly uAlphaCutoff: TSLNode;
  /**
   * Projection selector for the near fade: 0 = perspective, 1 = orthographic
   * (where the fade is the identity). A runtime UNIFORM, not a build flag —
   * which is why this graph uses `perspectiveNearFadeTSL` and not the
   * compile-time-ortho `…StaticTSL` variant the line graphs take.
   */
  readonly uIsOrtho: TSLNode;
  /** Near-fade start distance, world units (scene-relative; see the fragment). */
  readonly uNearCull: TSLNode;
  /** Refraction split (glass-partition-tsl.ts): mode + shared glass depth texture. */
  readonly uGlassPartition: GlassPartitionTSLNodes['uGlassPartition'];
  readonly uGlassDepth: GlassPartitionTSLNodes['uGlassDepth'];
  /** Set only when `config.useBaseColorTexture` is active. */
  readonly uBaseColorTex?: TSLNode;
  /** Set only when colormap mode is active. */
  readonly uColormapTex?: TSLNode;
  readonly uScalarMin?: TSLNode;
  readonly uScalarScale?: TSLNode;
}

/**
 * Mesh-material TSL factory.
 *
 * Pass `outMaterial` to configure an existing `NodeMaterial` subclass (the
 * `MeshTSLMaterial` wrapper, which owns the nodes table) rather than allocating a
 * new one. When omitted a fresh `NodeMaterial` is allocated — the standalone case
 * used by the codegen/parity harness.
 */
export function meshWebGPUFactory(
  nodes: MeshTSLNodes,
  config: MeshTSLConfig = {},
  outMaterial?: NodeMaterial
): NodeMaterial {
  const uOpacity = nodes.uOpacity;
  const uInvGamma = nodes.uInvGamma;
  const uIntensity = nodes.uIntensity;
  const uOffset = nodes.uOffset;
  const uAmbient = nodes.uAmbient;
  const uShadeExponent = nodes.uShadeExponent;
  const uSpecular = nodes.uSpecular;
  const uShininess = nodes.uShininess;
  const uAlphaCutoff = nodes.uAlphaCutoff;
  const uIsOrtho = nodes.uIsOrtho;
  const uNearCull = nodes.uNearCull;
  if (config.useColormap) {
    if (!nodes.uColormapTex || !nodes.uScalarMin || !nodes.uScalarScale) {
      throw new Error(
        'meshWebGPUFactory: config.useColormap=true but nodes.uColormapTex / uScalarMin / uScalarScale are not bound.'
      );
    }
  }
  if (config.useBaseColorTexture && !nodes.uBaseColorTex) {
    throw new Error(
      'meshWebGPUFactory: config.useBaseColorTexture=true but nodes.uBaseColorTex is not bound.'
    );
  }
  const uBaseColorTex = config.useBaseColorTexture ? nodes.uBaseColorTex! : null;
  const uColormapTex = config.useColormap ? nodes.uColormapTex! : null;
  const uScalarMin = config.useColormap ? nodes.uScalarMin! : null;
  const uScalarScale = config.useColormap ? nodes.uScalarScale! : null;

  const blendingMode: BlendingMode = config.blendingMode ?? 'opaque';
  // The emission shape is a BUILD-time branch (a JS conditional, as in the gsplat
  // factory — TSL `.select()` is avoided for structural branches). The wrapper
  // rebuilds the graph on any mode change that crosses one of these boundaries.
  const shading: MeshShadingMode = config.shading ?? 'smooth';
  const unlit = shading === 'none';
  const output = resolveMeshOutput(blendingMode);
  const premultiplyRGB = config.useMaxRGBContribution ?? output === 'rgb-contribution';
  const alphaCutout = output === 'opaque';

  // The `color` attribute is read as a vec4 whatever its component count: GL and
  // WebGPU both fill a size-3 attribute's missing components with (0, 0, 0, 1), so
  // RGB data supplies the opaque `w = 1.0` for free. The 8/16-bit family is padded
  // to 4 components CPU-side because three r184's WebGPU backend has no valid
  // 3-component unorm8/unorm16 format at all (mesh-geometry.ts, spec §6.1.1).
  const aColor: TSLNode = attribute<'vec4'>('color', 'vec4');
  const aScalar: TSLNode | null = config.useColormap
    ? attribute<'float'>('aScalar', 'float')
    : null;
  // Only referenced on the textured build, so on every other build `uv` stays out
  // of the vertex layout entirely — which matters more here than it would in GLSL:
  // WebGPU bakes the attribute set into the pipeline, so an unused-but-declared
  // attribute is a real difference between variants, not dead code the compiler
  // drops.
  const aUv: TSLNode | null = config.useBaseColorTexture ? attribute<'vec2'>('uv', 'vec2') : null;

  // ---- Varyings ----
  // Declared up front and `.assign()`ed inside the vertex body — the TSL pattern
  // for Fn-traced vertex stages. All of these genuinely interpolate (a mesh vertex
  // is not an instance constant), unlike the sibling factories' per-instance
  // varyings where interpolation happens to be a no-op.
  const vColor: TSLNode = varying(vec3(float(0.0), float(0.0), float(0.0)));
  const vAlpha: TSLNode = varying(float(1.0));
  const vClipZW: TSLNode = clipDepthVarying();
  const vViewPos: TSLNode = varying(vec3(float(0.0), float(0.0), float(0.0)));
  // Only declared for the stored-normal build. Under `flatNormal` the `normal`
  // attribute is never referenced, so it stays out of the vertex layout entirely.
  const vNormal: TSLNode | null =
    shading === 'smooth' ? varying(vec3(float(0.0), float(0.0), float(0.0))) : null;
  const vUv: TSLNode | null = aUv ? varying(vec2(float(0.0), float(0.0))) : null;

  const vertexBody = Fn(() => {
    // View space first: the shade term's every other input is view-space, and the
    // fragment differentiates this position for the flat-normal fallback.
    const mvPos: TSLNode = modelViewMatrix.mul(vec4(positionGeometry, 1.0)).toVar();

    let perVertexColor: TSLNode;
    if (config.useColormap && aScalar && uColormapTex && uScalarMin && uScalarScale) {
      // The display-range window and gamma shape the scalar VALUE pre-LUT, not the
      // resulting colour — mirrors the GLSL USE_COLORMAP path and the siblings.
      const t0: TSLNode = clamp(aScalar.sub(uScalarMin).mul(uScalarScale), 0.0, 1.0);
      const t: TSLNode = config.gammaOne ? t0 : t0.pow(uInvGamma);
      perVertexColor = uColormapTex.sample(vec2(t, 0.5)).rgb;
    } else if (config.useBaseColorTexture) {
      // White: the base colour is a PER-FRAGMENT texture fetch, so there is nothing
      // per-vertex to carry. Not skipped altogether because the fragment multiplies
      // by `vColor` unconditionally — keeping the GOG/gamma tail identical across
      // all three colour sources rather than forking it three ways, and white is
      // the multiplicative identity. GLSL twin does the same.
      perVertexColor = vec3(1.0, 1.0, 1.0);
    } else {
      perVertexColor = aColor.rgb;
    }

    vColor.assign(perVertexColor);
    if (vUv && aUv) vUv.assign(aUv);
    // Read unconditionally, including under colormap mode: the scalar replaces the
    // colour, not the opacity. Sanitized like the GLSL twin — for a mesh this alpha
    // is the WHOLE coverage term, and a NaN would survive into the cutout
    // comparison as a fragment that never discards.
    vAlpha.assign(sanitizeAlpha(aColor.w));
    vViewPos.assign(mvPos.xyz);
    if (vNormal) {
      // The twin of the GLSL `normalMatrix * normal` — `modelNormalMatrix` is the
      // inverse-transpose of the model matrix and the view matrix is orthonormal, so
      // the product equals the inverse-transpose of the model-view matrix that
      // GLSL's built-in `normalMatrix` supplies. The inverse-transpose is
      // load-bearing rather than pedantic: anisotropic voxel spacing (z != xy) is
      // routine in this domain, and under it the plain model-view linear map skews
      // normals off perpendicular.
      //
      // Spelled out rather than using three's `transformNormalToView`, which looks
      // like the obvious helper and is subtly wrong here: it routes through
      // `transformDirection`, which NORMALIZES. That normalize is not a harmless
      // extra op — the writer accepts zero-length normals with a warning (§3.5), and
      // `normalize(vec3(0))` is NaN, which then interpolates across every triangle
      // touching that vertex and flat-shades all of them. Leaving the normal
      // unnormalized (the fragment renormalizes anyway, absorbing the length change)
      // keeps the GLSL contract §3.5 states: shading near a degenerate vertex is
      // LOCALLY distorted, not a whole flat triangle.
      const viewMatrix: TSLNode = cameraViewMatrix;
      const normalMatrix: TSLNode = modelNormalMatrix;
      vNormal.assign(viewMatrix.mul(vec4(normalMatrix.mul(normalGeometry), 0.0)).xyz);
    }

    const clipOut: TSLNode = cameraProjectionMatrix.mul(mvPos).toVar();
    vClipZW.assign(clipOut.zw);
    return clipOut;
  });

  const clipPos: TSLNode = vertexBody();

  const colorNode = Fn(() => {
    // Refraction split partition FIRST: a fragment on the wrong side of the glass
    // costs nothing further (glass-partition-tsl.ts; GLSL twin at the top of main()).
    glassPartitionGuardTSL(nodes, vClipZW);
    // Computed UNCONDITIONALLY, before any guard-dependent branch: the epsilon
    // guard below reads an interpolated varying, so branching on it is non-uniform
    // control flow — where derivatives are undefined, since normal validity can
    // differ between fragments of the same 2x2 quad. Orientation comes from the
    // rasterized fragment rather than the winding, so this always faces the viewer.
    // Base colour first, so the unlit path below can skip every normal term. The
    // texture is sampled PER FRAGMENT, unlike the colormap LUT, which is a
    // vertex-stage lookup: a LUT maps one scalar per vertex, so interpolating the
    // resulting colour approximates interpolating the scalar, whereas an image has
    // structure BETWEEN vertices and a per-vertex fetch would resolve exactly one
    // texel per vertex — reproducing the point-cloud limitation this feature exists
    // to remove.
    let baseColor: TSLNode = vColor;
    let texAlpha: TSLNode | null = null;
    if (uBaseColorTex && vUv) {
      const texel: TSLNode = uBaseColorTex.sample(vUv).toVar();
      // A single-channel texture uploads as RedFormat and samples as (r, 0, 0, 1),
      // so without this swizzle a greyscale basemap renders pure red.
      baseColor = vColor.mul(
        config.baseColorTextureLuminance ? vec3(texel.r, texel.r, texel.r) : texel.rgb
      );
      // Texture alpha MULTIPLIES coverage, so an RGBA basemap gets real cutout holes
      // under `opaque`. A 3-channel texture is expanded to RGBA with alpha 1 at
      // upload and a 1-channel one samples alpha 1, so this is a free no-op for both.
      texAlpha = texel.a;
    }

    const rawDerivative: TSLNode | null = unlit
      ? null
      : normalize(cross(dFdx(vViewPos), dFdy(vViewPos))).toVar();
    // Forced viewer-facing, not assumed so, which makes the fallback
    // convention-INDEPENDENT. `cross(dFdx, dFdy)` carries the sign of the
    // fragment-space y axis, and the specs differ: WGSL's `dpdy` is TOP-DOWN where
    // GLSL's `dFdy` is bottom-up, so under OPPOSITE conventions the same surface yields
    // +z on one backend and -z on the other, and an unforced flat variant would
    // collapse toward `uAmbient` on one of them. V is the fixed view axis (0, 0, 1), so
    // "faces the viewer" is exactly `z >= 0`.
    //
    // MEASURED rather than assumed: on Chrome + Apple Silicon a real-WebGPU A/B with
    // this flip REMOVED renders a face-on flat quad IDENTICALLY to WebGL, so the two
    // conventions coincide there and the flip is inert on that platform. Kept anyway —
    // one instruction, correct under either convention, and neither spec promises they
    // agree. Insurance, not a fix for an observed bug. GLSL twin: shader-glsl.ts.
    const derivativeNormal: TSLNode | null = rawDerivative
      ? rawDerivative.z.lessThan(0.0).select(rawDerivative.negate(), rawDerivative).toVar()
      : null;

    // Stays null under `unlit`, and that is the point rather than a default: no
    // normal is computed at all — not a normal computed and then multiplied by zero.
    // The `normal` attribute stays out of the vertex layout and the derivative pair
    // is never evaluated.
    let N: TSLNode | null = null;
    if (unlit) {
      // Nothing to do — see above. Kept as an explicit arm so the three shading
      // modes read as three branches here, matching `MeshShadingMode`.
    } else if (!vNormal || !derivativeNormal) {
      N = derivativeNormal;
    } else {
      const nn: TSLNode = dot(vNormal, vNormal).toVar();
      // AFFIRMATIVE on purpose — the stored normal is used only when the length test
      // is positively TRUE. NaN fails every comparison, so a corrupt store lands in
      // the fallback, whereas an `nn < eps` test would pass it through to normalize
      // into NaN shading (spec §3.5's `!(dot(N, N) >= eps)` phrasing).
      //
      // TWO-SIDED, unlike that phrasing: an INFINITE normal component gives
      // `nn == inf`, which satisfies `>= eps`, and `inf * inverseSqrt(inf)` is
      // `inf * 0` = NaN — the guard's own failure mode arriving from the other end.
      // `1e30` matches the finite-range convention in `_shared/tsl-helpers.ts`.
      // GLSL twin: shader-glsl.ts.
      const storedUsable: TSLNode = nn.greaterThanEqual(MESH_NORMAL_EPS_SQ).and(nn.lessThan(1e30));
      const storedNormal: TSLNode = vNormal
        .mul(max(nn, float(MESH_NORMAL_EPS_SQ)).inverseSqrt())
        // Flip the STORED normal only. Without it, the stored back normal shades
        // on the wrong side of its gradient, producing an inverted result that can
        // collapse toward uAmbient —
        // immediately visible because `double_sided` defaults true and the
        // whole-triangle cull exposes a sliced isosurface's interior faces. The
        // derivative fallback must NOT be flipped: it already faces the viewer, so
        // flipping would reintroduce exactly that inverted shade, worst precisely
        // at the degenerate vertices the guard exists to rescue. Hence the flip
        // sits inside this branch, per FRAGMENT, not around the select.
        .mul(frontFacing.select(float(1.0), float(-1.0)))
        .toVar();
      N = storedUsable.select(storedNormal, derivativeNormal).toVar();
    }

    // View-anchored lighting: L is offset above-left, while V remains the fixed
    // view-space axis (0, 0, 1), making the Blinn-Phong half-vector constant.
    let shade: TSLNode | null = null;
    let spec: TSLNode | null = null;
    if (N) {
      const lightDirection: TSLNode = normalize(
        vec3(MESH_LIGHT_DIRECTION[0], MESH_LIGHT_DIRECTION[1], MESH_LIGHT_DIRECTION[2])
      );
      const halfVector: TSLNode = normalize(lightDirection.add(vec3(0.0, 0.0, 1.0)));
      const wrap: TSLNode = clamp(dot(N, lightDirection).mul(0.5).add(0.5), 0.0, 1.0);
      shade = mix(uAmbient, float(1.0), wrap.pow(uShadeExponent)).toVar();
      spec = uSpecular.mul(max(dot(N, halfVector), float(0.0)).pow(uShininess));
    }

    const adjusted: TSLNode = config.noGOG
      ? baseColor
      : max(baseColor.mul(uIntensity).add(uOffset), vec3(0.0)).toVar();

    // Colormap mode applied gamma pre-LUT; gammaOne means pow(x, 1) == x.
    const finalColor: TSLNode =
      config.useColormap || config.gammaOne ? adjusted : adjusted.pow(vec3(uInvGamma));

    // The shade factor is a LIGHTING term: RGB only, never the coverage — or a
    // silhouette fragment would also turn transparent (and dissolve under cutout).
    // Unlit means the base colour reaches the screen unmodulated — what every other
    // Luxar geometry type does (the other three are purely emissive) and what a data
    // basemap needs, since a view-anchored key would make a colour-coded surface
    // read differently as the camera moved.
    const shadedColor: TSLNode =
      shade && spec ? finalColor.mul(shade).add(vec3(spec)).toVar() : finalColor.toVar();

    // Mesh has no per-element intensity/amplitude/falloff scalar (§2.2), so
    // coverage is just per-vertex alpha times node opacity.
    const a: TSLNode = (
      texAlpha ? vAlpha.mul(uOpacity).mul(texAlpha) : vAlpha.mul(uOpacity)
    ).toVar();

    // Perspective near fade, PER FRAGMENT — a triangle spans depth, so a
    // per-vertex value would interpolate the RAMP across the face and smear the
    // smoothstep over a large triangle. The runtime-uniform variant, matching
    // point/gsplat: mesh's ortho flag is a uniform, not a build flag. The 1e-20
    // floor only guards the degenerate smoothstep when uNearCull is exactly 0;
    // an absolute floor would override the scene-relative value on a tiny-unit
    // scene. GLSL twin: shader-glsl.ts.
    const nearFade: TSLNode = perspectiveNearFadeTSL(
      uIsOrtho,
      vViewPos.z,
      max(uNearCull, float(1e-20))
    ).toVar();
    // Rejected in EVERY mode at the siblings' 0.01 threshold. Not optional in the
    // depth-writing ones: 'opaque' always writes depth and 'normal' does at opacity
    // >= 0.99 (normalModeDepthWrite), so a fully-faded but still-rasterized fragment
    // would occlude everything behind it. Unconditional rather than gated on that
    // predicate, which would buy a runtime uniform to save a discard. The cost is
    // early-z, which the four non-cutout variants kept until now — any discard
    // forfeits it. Paid knowingly, same as the GLSL twin.
    Discard(nearFade.lessThan(0.01));

    if (alphaCutout) {
      // 'opaque' (the mesh default): a hard, ORDER-INDEPENDENT cutout. Smooth
      // partial transparency is self-contradictory in a depth-writing mode drawn
      // without per-triangle sorting (§6.3), so authored alpha means masks/holes.
      //
      // The cutout compares the UNFADED coverage — the fade is a distance effect,
      // not an authored mask, and letting it move the comparison would dissolve
      // the holes open as the camera approached. This mode emits alpha 1.0, so
      // there is no alpha left to fade: the fade ramps the SHADED RGB instead —
      // i.e. the surface darkens toward black over the band rather than
      // dissolving, and only the reject above removes it (see README.md).
      Discard(a.lessThan(uAlphaCutoff));
      return vec4(shadedColor.mul(nearFade), float(1.0));
    }
    // Every non-cutout mode folds the fade into COVERAGE, exactly as the
    // point/line graphs do — which is also how 'max' picks it up in its RGB
    // premultiply for free.
    const faded: TSLNode = a.mul(nearFade).toVar();
    if (premultiplyRGB) {
      // 'max': MaxEquation + OneFactor/OneFactor does not weight source RGB by
      // alpha at composite, so premultiply by coverage here.
      return vec4(shadedColor.mul(faded), faded);
    }
    return vec4(shadedColor, faded);
  });

  const material = outMaterial ?? new NodeMaterial();
  // Override the vertex output entirely so the graph — not NodeMaterial's default
  // modelViewProjection chain — owns the view-space position the fragment reads.
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.toneMapped = false;

  // This factory tail is the ONLY blending-state writer at TSL construction (the
  // wrapper's ctor never calls applyBlendingMode, unlike the GLSL twin) AND it
  // re-runs on every rebuildGraph, so it must derive the state from the same mode
  // the emission branch above used.
  applyBlendingStateToMaterial(
    material,
    getCompleteBlendingState(
      resolveMeshBlendingMode(blendingMode),
      (uOpacity.value as number | undefined) ?? 1.0
    )
  );
  return material;
}

/**
 * Build a `MeshTSLNodes` set from a plain `IUniform` record — for callers that
 * don't own persistent wrapper-side `UniformNode`s (the codegen/parity harness and
 * the `MESH_SOURCE` ShaderSource factory). Mirrors
 * `buildPointTSLNodesFromUniforms`.
 *
 * The resulting nodes capture the current `iuniform.value` at build time;
 * mutations to the host `IUniform` afterwards do NOT propagate. Appropriate for
 * the harness (build once, render once), never for a live wrapper.
 */
export function buildMeshTSLNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>,
  config: MeshTSLConfig = {}
): MeshTSLNodes {
  const base: MeshTSLNodes = {
    uOpacity: uniform((uniforms.uOpacity?.value as number) ?? 1.0),
    uInvGamma: uniform((uniforms.uInvGamma?.value as number) ?? 1.0),
    uIntensity: uniform((uniforms.uIntensity?.value as number) ?? 1.0),
    uOffset: uniform((uniforms.uOffset?.value as number) ?? 0.0),
    uAmbient: uniform((uniforms.uAmbient?.value as number) ?? MESH_DEFAULTS.ambient),
    uShadeExponent: uniform(
      (uniforms.uShadeExponent?.value as number) ?? MESH_DEFAULTS.shadeExponent
    ),
    uSpecular: uniform((uniforms.uSpecular?.value as number) ?? MESH_DEFAULTS.specular),
    uShininess: uniform((uniforms.uShininess?.value as number) ?? MESH_DEFAULTS.shininess),
    uAlphaCutoff: uniform((uniforms.uAlphaCutoff?.value as number) ?? MESH_DEFAULTS.alphaCutoff),
    // 0 = perspective, and 0.1 is the same near-cull default the sibling
    // materials construct with (overridden per scene by updateCameraParams).
    uIsOrtho: uniform((uniforms.uIsOrtho?.value as number) ?? 0),
    uNearCull: uniform((uniforms.uNearCull?.value as number) ?? 0.1),
    ...glassPartitionNodesFromUniforms(uniforms),
  };
  // Both optional groups are added independently, because the two colour sources
  // are mutually exclusive at the DEFINE level and this function must not encode a
  // second, weaker version of that rule. An early `return base` before the texture
  // group is how the codegen harness would silently build the untextured variant
  // while asking for the textured one.
  return {
    ...base,
    ...(config.useBaseColorTexture
      ? {
          uBaseColorTex: texture(
            (uniforms.uBaseColorTex?.value as THREE.Texture | null) ?? new THREE.Texture()
          ),
        }
      : {}),
    ...(config.useColormap
      ? {
          uColormapTex: texture(
            (uniforms.uColormapTex?.value as THREE.Texture | null) ?? new THREE.Texture()
          ),
          uScalarMin: uniform((uniforms.uScalarMin?.value as number) ?? 0.0),
          uScalarScale: uniform((uniforms.uScalarScale?.value as number) ?? 1.0),
        }
      : {}),
  };
}
