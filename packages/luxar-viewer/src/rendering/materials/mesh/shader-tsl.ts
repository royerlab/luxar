/**
 * Mesh material TSL factory — NodeMaterial counterpart to the GLSL3 pair in
 * `shader-glsl.ts`.
 *
 * Draws a shaded indexed triangle surface:
 *   - view-anchored headlight shade term, no scene light (spec §6.2)
 *   - stored view-space normals OR a screen-space-derivative flat normal, chosen at
 *     GRAPH BUILD time (the `flatNormal` config flag — the twin of the GLSL
 *     `LUXAR_MESH_FLAT_NORMAL` define)
 *   - per-node Gain/Offset/Gamma, with the same `gammaOne` / `noGOG` fast paths the
 *     sibling factories carry
 *   - per-vertex alpha as the sole coverage term, emitted per blending mode
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
import { sanitizeAlpha, type TSLNode } from '../_shared/tsl-helpers';
import { applyBlendingStateToMaterial, getCompleteBlendingState } from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';
import {
  MESH_DEFAULTS,
  MESH_NORMAL_EPS_SQ,
  resolveMeshBlendingMode,
  resolveMeshOutput,
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
   * Shade from screen-space derivatives instead of the stored `normal` attribute.
   * Mirrors `LUXAR_MESH_FLAT_NORMAL`, and like it this is a BUILD-time choice, not a
   * runtime branch: a declared-but-unbound `normal` reads `(0, 0, 0, 1)` rather than
   * "absent", so there is no runtime value meaning "no normals". `createMeshNode`
   * computes it once per node from `shading` + the stored normals' validity for the
   * active `displayDims`, and hands the same answer to both backends (§6.2).
   */
  readonly flatNormal?: boolean;
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
  /** Shade floor at the silhouette (1.0 reproduces the emissive siblings' look). */
  readonly uAmbient: TSLNode;
  /** Wrap-term contrast exponent. */
  readonly uShadeExponent: TSLNode;
  /** Cutout threshold; read only when the graph was built in `opaque` mode. */
  readonly uAlphaCutoff: TSLNode;
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
  const uAlphaCutoff = nodes.uAlphaCutoff;
  if (config.useColormap) {
    if (!nodes.uColormapTex || !nodes.uScalarMin || !nodes.uScalarScale) {
      throw new Error(
        'meshWebGPUFactory: config.useColormap=true but nodes.uColormapTex / uScalarMin / uScalarScale are not bound.'
      );
    }
  }
  const uColormapTex = config.useColormap ? nodes.uColormapTex! : null;
  const uScalarMin = config.useColormap ? nodes.uScalarMin! : null;
  const uScalarScale = config.useColormap ? nodes.uScalarScale! : null;

  const blendingMode: BlendingMode = config.blendingMode ?? 'opaque';
  // The emission shape is a BUILD-time branch (a JS conditional, as in the gsplat
  // factory — TSL `.select()` is avoided for structural branches). The wrapper
  // rebuilds the graph on any mode change that crosses one of these boundaries.
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

  // ---- Varyings ----
  // Declared up front and `.assign()`ed inside the vertex body — the TSL pattern
  // for Fn-traced vertex stages. All of these genuinely interpolate (a mesh vertex
  // is not an instance constant), unlike the sibling factories' per-instance
  // varyings where interpolation happens to be a no-op.
  const vColor: TSLNode = varying(vec3(float(0.0), float(0.0), float(0.0)));
  const vAlpha: TSLNode = varying(float(1.0));
  const vViewPos: TSLNode = varying(vec3(float(0.0), float(0.0), float(0.0)));
  // Only declared for the stored-normal build. Under `flatNormal` the `normal`
  // attribute is never referenced, so it stays out of the vertex layout entirely.
  const vNormal: TSLNode | null = config.flatNormal
    ? null
    : varying(vec3(float(0.0), float(0.0), float(0.0)));

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
    } else {
      perVertexColor = aColor.rgb;
    }

    vColor.assign(perVertexColor);
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

    return cameraProjectionMatrix.mul(mvPos);
  });

  const clipPos: TSLNode = vertexBody();

  const colorNode = Fn(() => {
    // Computed UNCONDITIONALLY, before any guard-dependent branch: the epsilon
    // guard below reads an interpolated varying, so branching on it is non-uniform
    // control flow — where derivatives are undefined, since normal validity can
    // differ between fragments of the same 2x2 quad. Orientation comes from the
    // rasterized fragment rather than the winding, so this always faces the viewer.
    const rawDerivative: TSLNode = normalize(cross(dFdx(vViewPos), dFdy(vViewPos))).toVar();
    // Forced viewer-facing, not assumed so — and this is the one place the two
    // backends would otherwise diverge outright. `cross(dFdx, dFdy)` carries the sign
    // of the fragment-space y axis, and WGSL's `dpdy` is TOP-DOWN where GLSL's
    // `dFdy` is bottom-up, so the same surface yields +z under WebGL and -z under
    // WebGPU: the flat variant would collapse to `uAmbient` everywhere on one backend
    // only. The headlight's V is the fixed view axis (0, 0, 1), so "faces the viewer"
    // is exactly `z >= 0` and one sign flip settles it. GLSL twin: shader-glsl.ts.
    const derivativeNormal: TSLNode = rawDerivative.z
      .lessThan(0.0)
      .select(rawDerivative.negate(), rawDerivative)
      .toVar();

    let N: TSLNode;
    if (config.flatNormal || !vNormal) {
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
        // Flip the STORED normal only. Without it a back fragment has
        // dot(N, V) < 0, so the wrap term lands in [0, 0.5) and the back side
        // shades with a dimmed inverted gradient collapsing toward uAmbient —
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

    // View-anchored headlight: V is the fixed view-space axis (0, 0, 1), so
    // dot(N, V) reduces to N.z. The wrap term keeps the silhouette readable.
    const wrap: TSLNode = clamp(N.z.mul(0.5).add(0.5), 0.0, 1.0);
    const shade: TSLNode = mix(uAmbient, float(1.0), wrap.pow(uShadeExponent)).toVar();

    const adjusted: TSLNode = config.noGOG
      ? vColor
      : max(vColor.mul(uIntensity).add(uOffset), vec3(0.0)).toVar();

    // Colormap mode applied gamma pre-LUT; gammaOne means pow(x, 1) == x.
    const finalColor: TSLNode =
      config.useColormap || config.gammaOne ? adjusted : adjusted.pow(vec3(uInvGamma));

    // The shade factor is a LIGHTING term: RGB only, never the coverage — or a
    // silhouette fragment would also turn transparent (and dissolve under cutout).
    const shadedColor: TSLNode = finalColor.mul(shade).toVar();

    // Mesh has no per-element intensity/amplitude/falloff scalar (§2.2), so
    // coverage is just per-vertex alpha times node opacity.
    const a: TSLNode = vAlpha.mul(uOpacity).toVar();

    if (alphaCutout) {
      // 'opaque' (the mesh default): a hard, ORDER-INDEPENDENT cutout. Smooth
      // partial transparency is self-contradictory in a depth-writing mode drawn
      // without per-triangle sorting (§6.3), so authored alpha means masks/holes.
      Discard(a.lessThan(uAlphaCutoff));
      return vec4(shadedColor, float(1.0));
    }
    if (premultiplyRGB) {
      // 'max': MaxEquation + OneFactor/OneFactor does not weight source RGB by
      // alpha at composite, so premultiply by coverage here.
      return vec4(shadedColor.mul(a), a);
    }
    return vec4(shadedColor, a);
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
    uAlphaCutoff: uniform((uniforms.uAlphaCutoff?.value as number) ?? MESH_DEFAULTS.alphaCutoff),
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
