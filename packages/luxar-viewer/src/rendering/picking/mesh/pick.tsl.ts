/**
 * Mesh picking material TSL factory — NodeMaterial counterpart to
 * `MESH_PICK_SOURCE` in `shaders.ts`.
 *
 * Fragment output (the shared RGBA32F pick encoding):
 *   - R: nodeId (uniform)
 *   - G: elementId LOW 16 bits (= the VERTEX ordinal — mesh has no ordering
 *     attribute to indirect through, §6.5)
 *   - B: brightness = the coverage `vAlpha * uOpacity`, or 1.0 for a cutout
 *     survivor (which the visual shader draws fully opaque), times the shared
 *     perspective near fade — so pick salience tracks visible salience right up
 *     to the near plane, and a fragment below the 0.01 reject is discarded
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
  texture,
  uniform,
  attribute,
  varying,
  vec2 as _vec2,
  vec4 as _vec4,
  float,
  int,
  bool,
  clamp as _clamp,
  max as _max,
  depth,
  vertexIndex,
  modelViewMatrix,
  cameraProjectionMatrix,
  positionGeometry,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import {
  isOrthoProjectionTSL,
  sanitizeAlpha,
  perspectiveNearFadeTSL,
  type TSLNode,
} from '../../materials/_shared/tsl-helpers';
import { MESH_DEFAULTS } from '../../materials/mesh/appearance';

// Type-erased builder aliases — same rationale as the visual mesh factory: TSLNode
// is `any`, so overload resolution on a TSLNode argument picks the first typed
// signature and rejects otherwise-valid combinations. The generated GLSL/WGSL is
// unaffected.
const vec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _vec2 as TSLNode;
const vec4: (a?: TSLNode, b?: TSLNode, c?: TSLNode, d?: TSLNode) => TSLNode = _vec4 as TSLNode;
const clamp: (v: TSLNode, lo: TSLNode, hi: TSLNode) => TSLNode = _clamp as TSLNode;
const max: (a: TSLNode, b: TSLNode) => TSLNode = _max as TSLNode;

/**
 * Pre-created TSL leaf nodes supplied by the wrapper class. Same pattern as
 * `LinePickTSLNodes` / `GSplatPickTSLNodes` — and shorter, because a mesh has no
 * screen-space footprint to size: of the camera inputs it takes only the two the
 * near fade needs, and no resolution or focal length.
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
  /** Near-fade start distance, world units (scene-relative). */
  readonly uNearCull: TSLNode;
  /**
   * The visual material's base-colour texture, when the node has one.
   *
   * Sampled for its ALPHA only — the pick pass has no colour output — because
   * texture alpha multiplies coverage in the visual shader, so an RGBA basemap's
   * cutout holes are real holes on screen and must not stay pickable or
   * depth-occluding.
   */
  readonly uBaseColorTex?: TSLNode;
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
  // Build-time, keyed on whether the node HAS a texture — a per-node constant, so
  // unlike `uAlphaCutout` (a layers-panel-mutable mode) this never needs to be a
  // runtime branch, and it decides whether `uv` enters the vertex layout at all.
  const uBaseColorTex = nodes.uBaseColorTex ?? null;
  // The only attribute beyond `position`: the element id is the `vertexIndex`
  // built-in, and normals/scalars are shading inputs with no bearing on which
  // vertex was clicked. Read as vec4 for the same reason the visual factory does —
  // a size-3 attribute supplies the opaque `w = 1.0` for free.
  const aColor: TSLNode = attribute<'vec4'>('color', 'vec4');
  const aUv: TSLNode | null = uBaseColorTex ? attribute<'vec2'>('uv', 'vec2') : null;

  const uNodeId = nodes.uNodeId;
  const uOpacity = nodes.uOpacity;
  const uAlphaCutoff = nodes.uAlphaCutoff;
  const uAlphaCutout = nodes.uAlphaCutout;
  const uSurfaceDepth = nodes.uSurfaceDepth;
  const uNearCull = nodes.uNearCull;

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
  // View-space depth for the fragment-stage near fade. Just the z, not the whole
  // view position the visual factory carries — that one is differentiated for the
  // flat-normal fallback, and the pick pass has no shading to do.
  const vViewZ: TSLNode = varying(float(0.0));
  const vUv: TSLNode | null = aUv ? varying(vec2(float(0.0), float(0.0))) : null;

  const vertexBody = Fn(() => {
    // Sanitized identically to the visual pair: alpha is the whole coverage term
    // for a mesh, and a NaN would survive into the cutout comparison as a fragment
    // that never discards — pickable where the visual has a hole.
    vAlpha.assign(sanitizeAlpha(aColor.w));
    if (vUv && aUv) vUv.assign(aUv);
    const mvPos: TSLNode = modelViewMatrix.mul(vec4(positionGeometry, 1.0)).toVar();
    vViewZ.assign(mvPos.z);
    return cameraProjectionMatrix.mul(mvPos);
  });

  const clipPos: TSLNode = vertexBody();

  // `colorNode` and `depthNode` are INDEPENDENT stage entry points, and the order in
  // which three builds them is not part of its API: r184 emitted the colour flow
  // first, r185 emits the depth flow first. A value materialised with `.toVar()` is
  // assigned wherever three first BUILDS it, and a branch-scoped assignment is only
  // re-hoisted for a later reader that is itself inside a block
  // (`NodeBuilder.addFlowCodeHierarchy`, gated on
  // `builder.context.nodeBlock !== undefined`) — never for one at the top level of a
  // flow. There are TWO branches here: `depthNode` selects on `uSurfaceDepth`, and the
  // brightness select on `uAlphaCutout` nests one level deeper inside it. Three lowers
  // each to a real `if / else`, so a free-standing shared chain lands inside an arm
  // while the top-level readers (the two `Discard` conditions, the output vec4) still
  // read the variable — unassigned, i.e. 0. That is a pick buffer with no pixels in
  // it, and a cutout that discards the wrong fragments.
  //
  // So every value both entry points read is declared up front and ASSIGNED in
  // `fragmentPrologue`, a `'void'`-typed `Fn` invoked as the FIRST statement of BOTH
  // of them. A void `Fn` call is a stack STATEMENT — a non-void one is wrapped in an
  // intent var that three skips, which would leave the call to build at its
  // consumption site, inside the arm again — so the assignments are emitted in trace
  // order in unconditional top-level flow, whichever entry point three builds first.
  // Trace order is also why the inner `cutoutOn.select(...)` below can no longer be
  // the first build site of `coverage`.
  //
  // What `.once()` does and does not buy: the prologue's code is emitted where it is
  // FIRST built, and because both entry points call it as their first statement that
  // site is the top level of whichever flow three emits first. `.once()` then lets the
  // second call reuse the already-traced result instead of emitting the chain twice.
  // Its cache lives on the NodeBuilder (so per material build) and is keyed on shader
  // stage `'any'`, so calling this same prologue from another shader STAGE would
  // silently reuse the first stage's nodes — it is fragment-only for that reason. A
  // cache MISS would merely duplicate the chain, which stays correct.
  //
  // One visible consequence of the r185 flip: with the depth flow emitted first,
  // `gl_FragDepth` is written ABOVE the `Discard`s in source order (the GLSL twins
  // discard first). Still correct — a discarded fragment writes no buffer at all,
  // depth included.
  const coverage: TSLNode = float(0.0).toVar('meshPickCoverage');
  const nearFade: TSLNode = float(0.0).toVar('meshPickNearFade');
  // Read by BOTH entry points — the brightness select in the prologue below and the
  // cutout `Discard` in `colorNode` — so it is an explicit var assigned in the
  // prologue like the rest. As a free-standing comparison it was materialised anyway
  // (three gives a select's condition its own `bool`), so it survived r184 only
  // because the colour flow built first and the cutout `Discard` put that assignment
  // at top level there — the same build-order accident the prologue exists to stop
  // depending on. Depth-first, it lands in the arm and the discard reads `false`.
  const cutoutOn: TSLNode = bool(false).toVar('meshPickCutout');
  const brightness: TSLNode = float(0.0).toVar('meshPickBrightness');

  const fragmentPrologue = Fn(() => {
    // Identical coverage to the visual shader (§6.2): a mesh has no per-element
    // intensity/amplitude, so coverage is per-vertex alpha times node opacity.
    // Texture alpha folded in BEFORE the cutout comparison, matching the visual
    // graph's ordering exactly — comparing a different quantity is how the two
    // passes would disagree about where the holes are. No luminance swizzle: a
    // 1-channel texture samples alpha 1.0, so the visual graph's `.rrr` fix for
    // `.rgb` has no analogue here.
    coverage.assign(
      uBaseColorTex && vUv
        ? vAlpha.mul(uOpacity).mul(uBaseColorTex.sample(vUv).a)
        : vAlpha.mul(uOpacity)
    );
    // Same fade, same 1e-20 degenerate-smoothstep floor and same 0.01 reject as the
    // visual graph — pick coverage must keep matching visible coverage as the camera
    // flies into the surface. Per FRAGMENT, because a triangle spans depth.
    nearFade.assign(
      perspectiveNearFadeTSL(isOrthoProjectionTSL(), vViewZ, max(uNearCull, float(1e-20)))
    );
    // Assigned before `brightness`, whose select reads it, and before the cutout
    // `Discard` in `colorNode` reads it.
    cutoutOn.assign(int(uAlphaCutout).equal(int(1)));
    // Survivors of the cutout are FULLY OPAQUE on screen (the visual shader emits
    // `vec4(rgb, 1.0)` for them), so their pick brightness must be 1.0 too. Carrying
    // the pre-cutout coverage through instead would under-weight a solid mesh in the
    // cross-node brightness vote purely because its author wrote 0.6 into a channel
    // the visual output ignores.
    //
    // The fade multiplies AFTER that select, so it reaches both arms once: the cutout
    // arm's visual twin ramps its shaded RGB by the same factor, and the commutative
    // arm's carries it in the coverage. Folding it into `coverage` instead would move
    // the cutout comparison below and dissolve the holes open as the camera neared.
    brightness.assign(clamp(cutoutOn.select(float(1.0), coverage).mul(nearFade), 0.0, 1.0));
    // Returned only so `.once()` has a result to cache — a body with no result
    // re-traces on the second call and emits the whole chain twice. The entry points
    // read the vars above, not this value.
    return brightness;
  }, 'void').once();

  const colorNode = Fn(() => {
    fragmentPrologue();
    // The cutout is a runtime-uniform branch, not a build flag — one `mesh-pick`
    // variant covers every blending mode. The discards live in `colorNode` rather
    // than in the shared prologue, following the line/gsplat pick precedent; a
    // discarded fragment writes neither colour nor depth, so either placement is
    // equivalent on screen — but a discard inside the prologue would execute in
    // whichever flow three happens to build first, which is precisely the ordering
    // the prologue exists to stop depending on.
    //
    // The near reject is ordered FIRST, matching the GLSL twin, so both backends
    // decline a faded fragment for the same reason; either way it writes neither
    // the id nor depth.
    Discard(nearFade.lessThan(0.01));
    Discard(cutoutOn.and(coverage.lessThan(uAlphaCutoff)));
    return vec4(vNodeId, vElementId.x, brightness, vElementId.y);
  });

  const depthNode = Fn(() => {
    fragmentPrologue();
    return int(uSurfaceDepth)
      .equal(int(1))
      .select(depth as unknown as TSLNode, float(1.0).sub(brightness));
  });

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
    // 0.1 is the near-cull default every wrapper constructs with (overridden
    // per scene by updateCameraParams).
    uNearCull: uniform((uniforms.uNearCull?.value as number) ?? 0.1),
    // PRESENCE-keyed, not defaulted: an absent uniform means the node has no
    // texture, and binding a blank one would build the sampling variant for a node
    // whose geometry has no `uv` attribute — a bound-but-unfilled attribute reads as
    // (0, 0, 0, 1) rather than "absent", so every fragment would sample texel 0 and
    // the whole mesh could vanish under the cutout.
    ...(uniforms.uBaseColorTex?.value
      ? { uBaseColorTex: texture(uniforms.uBaseColorTex.value as THREE.Texture) }
      : {}),
  };
}
