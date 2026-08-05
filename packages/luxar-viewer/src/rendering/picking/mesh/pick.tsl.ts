/**
 * Mesh picking material TSL factory — NodeMaterial counterpart to
 * `MESH_PICK_SOURCE` in `shaders.ts`.
 *
 * Fragment output (the shared RGBA32F pick encoding):
 *   - R: nodeId (uniform)
 *   - G: elementId LOW 16 bits (= the VERTEX ordinal — mesh has no ordering
 *     attribute to indirect through, §6.5)
 *   - B: brightness = the coverage `vAlpha * uOpacity`, or 1.0 for a cutout
 *     survivor (which the visual shader draws fully opaque)
 *   - A: the same elementId's HIGH 16 bits (one f32 channel cannot carry the
 *     whole index exactly — see `luxarElementIdSplit`)
 *
 * Depth = `1 - brightness` (brightest wins), or the real fragment depth when
 * `uSurfaceDepth == 1` (the depth-ordered `opaque`/`normal` surface modes:
 * front-most wins). Both selectors are runtime uniforms, so a layers-panel
 * blending-mode switch never rebuilds this graph.
 *
 * ## The flat-interpolation requirement is not optional here
 *
 * `vElementId` and `vNodeId` are declared `.setInterpolation('flat')`, following
 * the LINE pick precedent rather than the point pick. The point pick omits `flat`
 * and gets away with it only because a single-instance quad's four corners all
 * carry the same id, making interpolation the identity. A mesh is an indexed,
 * shared-vertex draw where `vertexIndex` differs at every corner: interpolated, the
 * id would arrive FRACTIONAL and the readback's `Math.round` would resolve to
 * arbitrary unrelated vertices.
 *
 * WGSL's `@interpolate(flat)` samples the FIRST vertex of the primitive where
 * OpenGL ES fixes it to the LAST, so the exact corner reported differs by backend
 * unless `WEBGL_provoking_vertex` is available to align them (see `./material.ts`).
 * That is a documented §6.5 exception: at vertex granularity the contract is
 * "a corner of the front-most triangle", since the cursor is over the face.
 *
 * @module rendering/picking/mesh/pick.tsl
 */

import * as THREE from 'three';
import {
  Fn,
  uniform,
  attribute,
  varying,
  vec2 as _vec2,
  vec4 as _vec4,
  float,
  int,
  clamp as _clamp,
  depth,
  vertexIndex,
  modelViewMatrix,
  cameraProjectionMatrix,
  positionGeometry,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { sanitizeAlpha, type TSLNode } from '../../materials/_shared/tsl-helpers';
import { MESH_DEFAULTS } from '../../materials/mesh/appearance';

// Type-erased builder aliases — same rationale as the visual mesh factory: TSLNode
// is `any`, so overload resolution on a TSLNode argument picks the first typed
// signature and rejects otherwise-valid combinations. The generated GLSL/WGSL is
// unaffected.
const vec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _vec2 as TSLNode;
const vec4: (a?: TSLNode, b?: TSLNode, c?: TSLNode, d?: TSLNode) => TSLNode = _vec4 as TSLNode;
const clamp: (v: TSLNode, lo: TSLNode, hi: TSLNode) => TSLNode = _clamp as TSLNode;

/**
 * Pre-created TSL leaf nodes supplied by the wrapper class. Same pattern as
 * `LinePickTSLNodes` / `GSplatPickTSLNodes` — and far shorter, because a mesh has
 * no screen-space footprint to size and therefore no camera uniforms at all.
 */
export interface MeshPickTSLNodes {
  readonly uNodeId: TSLNode;
  /** Node opacity — the second half of the coverage term. */
  readonly uOpacity: TSLNode;
  /** Cutout threshold, read only when `uAlphaCutout == 1`. */
  readonly uAlphaCutoff: TSLNode;
  /** 1 = `opaque`: apply the visual shader's identical hard cutout. */
  readonly uAlphaCutout: TSLNode;
  /**
   * Pick depth convention selector: 0 = brightness-as-depth (brightest wins;
   * the commutative modes), 1 = real fragment depth (front-most wins; the
   * `opaque`/`normal` surface modes). Mirrors the GLSL `uSurfaceDepth`.
   */
  readonly uSurfaceDepth: TSLNode;
}

/**
 * Mesh picking material TSL factory.
 *
 * Consumes pre-created `UniformNode` references; the wrapper
 * (`MeshPickingTSLMaterial`) owns those nodes and exposes them via
 * `material.uniforms` as `IUniform`-shaped getter/setter proxies.
 */
export function meshPickWebGPUFactory(
  nodes: MeshPickTSLNodes,
  outMaterial?: NodeMaterial
): NodeMaterial {
  // The only attribute beyond `position`: the element id is the `vertexIndex`
  // built-in, and normals/scalars are shading inputs with no bearing on which
  // vertex was clicked. Read as vec4 for the same reason the visual factory does —
  // a size-3 attribute supplies the opaque `w = 1.0` for free.
  const aColor: TSLNode = attribute<'vec4'>('color', 'vec4');

  const uNodeId = nodes.uNodeId;
  const uOpacity = nodes.uOpacity;
  const uAlphaCutoff = nodes.uAlphaCutoff;
  const uAlphaCutout = nodes.uAlphaCutout;
  const uSurfaceDepth = nodes.uSurfaceDepth;

  // ---- Varyings ----
  // Flat for the two ids (see the module doc — mandatory, not stylistic); smooth
  // for the coverage, exactly as in the visual pair, so the cutout hole in the
  // pick pass has the same shape as the one on screen.
  const vNodeId: TSLNode = varying(uNodeId).setInterpolation('flat');
  // Vertex ordinal split into two 16-bit halves — the TSL twin of
  // `luxarElementIdSplit` in glsl-lib.ts. Integer div/sub rather than bit ops so
  // the graph lowers the same way on both backends (line-pick precedent).
  const elementIdInt: TSLNode = int(vertexIndex);
  const elementIdHi: TSLNode = elementIdInt.div(int(65536));
  const elementIdLo: TSLNode = elementIdInt.sub(elementIdHi.mul(int(65536)));
  const vElementId: TSLNode = varying(
    vec2(float(elementIdLo), float(elementIdHi))
  ).setInterpolation('flat');
  const vAlpha: TSLNode = varying(float(1.0));

  const vertexBody = Fn(() => {
    // Sanitized identically to the visual pair: alpha is the whole coverage term
    // for a mesh, and a NaN would survive into the cutout comparison as a fragment
    // that never discards — pickable where the visual has a hole.
    vAlpha.assign(sanitizeAlpha(aColor.w));
    return cameraProjectionMatrix.mul(modelViewMatrix.mul(vec4(positionGeometry, 1.0)));
  });

  const clipPos: TSLNode = vertexBody();

  // `colorNode` and `depthNode` are INDEPENDENT stage entry points, so anything
  // both read is hoisted into one `.toVar()` here — the same structure the line and
  // gsplat pick factories use, and the reason the coverage chain is not inlined
  // twice into the generated WGSL/GLSL.
  //
  // Identical coverage to the visual shader (§6.2): a mesh has no per-element
  // intensity/amplitude, so coverage is per-vertex alpha times node opacity.
  const coverageShared = Fn(() => vAlpha.mul(uOpacity)).once();
  const coverage: TSLNode = coverageShared().toVar('meshPickCoverage');
  const cutoutOn: TSLNode = int(uAlphaCutout).equal(int(1));
  // Survivors of the cutout are FULLY OPAQUE on screen (the visual shader emits
  // `vec4(rgb, 1.0)` for them), so their pick brightness must be 1.0 too. Carrying
  // the pre-cutout coverage through instead would under-weight a solid mesh in the
  // cross-node brightness vote purely because its author wrote 0.6 into a channel
  // the visual output ignores.
  const brightness: TSLNode = clamp(cutoutOn.select(float(1.0), coverage), 0.0, 1.0).toVar(
    'meshPickBrightness'
  );

  const colorNode = Fn(() => {
    // The cutout is a runtime-uniform branch, not a build flag — one `mesh-pick`
    // variant covers every blending mode. It lives in `colorNode` rather than in
    // the shared chain, following the line/gsplat pick precedent; a discarded
    // fragment writes neither colour nor depth, so the placement is safe either
    // way, but keeping the shared value a pure expression is not.
    Discard(cutoutOn.and(coverage.lessThan(uAlphaCutoff)));
    return vec4(vNodeId, vElementId.x, brightness, vElementId.y);
  });

  const depthNode = Fn(() =>
    int(uSurfaceDepth)
      .equal(int(1))
      .select(depth as unknown as TSLNode, float(1.0).sub(brightness))
  );

  const material = outMaterial ?? new NodeMaterial();
  // Own the vertex output outright rather than leaving NodeMaterial's default
  // modelViewProjection chain to produce it — matching the visual mesh factory.
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.depthNode = depthNode();
  material.toneMapped = false;
  material.depthTest = true;
  material.depthWrite = true;
  material.transparent = false;
  // The element index's HIGH half rides in alpha, and THREE's NodeMaterial appends
  // `DiffuseColor.w *= material.opacity` to every generated fragment. NoBlending
  // does not suppress that shader-side multiply, so any opacity other than exactly
  // 1 would scale the high half and decode a WRONG element id — on the TSL path
  // only, since the GLSL twin has no such tail. Pin it so the multiply is provably
  // identity, including when a caller injects `outMaterial`.
  //
  // Note this is the MATERIAL's opacity, which is a different quantity from the
  // node opacity in `uOpacity`: the latter is a uniform this graph reads
  // explicitly into the coverage term, exactly as the GLSL twin does.
  material.opacity = 1;
  // Picking output is an opaque ID buffer; any blending would smear
  // nodeId / elementId across overlapping picks. Matches the GLSL material.
  material.blending = THREE.NoBlending;
  return material;
}

/**
 * Build a `MeshPickTSLNodes` set from a plain `IUniform` record — for callers that
 * don't own persistent wrapper-side `UniformNode`s (the codegen/parity harness and
 * the `MESH_PICK_SOURCE` ShaderSource factory). Symmetric with
 * `buildLinePickTSLNodesFromUniforms` / `buildGSplatPickTSLNodesFromUniforms`.
 */
export function buildMeshPickTSLNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>
): MeshPickTSLNodes {
  return {
    uNodeId: uniform((uniforms.uNodeId?.value as number) ?? 0),
    uOpacity: uniform((uniforms.uOpacity?.value as number) ?? 1.0),
    uAlphaCutoff: uniform((uniforms.uAlphaCutoff?.value as number) ?? MESH_DEFAULTS.alphaCutoff),
    // Both selectors default to the ON state, unlike the gsplat pick's `?? 0`,
    // because the mesh default blending mode is `opaque` (§6.3) — which is both a
    // cutout mode and a depth-ordered surface mode. A fallback of 0 here would give
    // a codegen-only or test-constructed material the behaviour of a mode the node
    // is not in.
    uAlphaCutout: uniform((uniforms.uAlphaCutout?.value as number) ?? 1),
    uSurfaceDepth: uniform((uniforms.uSurfaceDepth?.value as number) ?? 1),
  };
}
