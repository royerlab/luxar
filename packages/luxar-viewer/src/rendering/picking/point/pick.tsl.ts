/**
 * Point picking material TSL factory — NodeMaterial counterpart to
 * the GLSL3 picking shader in `shaders.ts::POINT_PICK_SOURCE`.
 *
 * Renders one tight sprite per point with output:
 *   - R: nodeId (set as uniform)
 *   - G: elementId LOW 16 bits (= `aSortedIndex`, the STORAGE slot —
 *     identical to the draw slot under Phase-1 identity ordering, and
 *     stays the id the rest of the pipeline addresses points by once
 *     the sort worker permutes draw order in Phase 2+)
 *   - B: brightness (super-Gaussian falloff at the fragment position)
 *   - A: the same elementId's HIGH 16 bits (one f32 channel cannot
 *     carry the whole index exactly — see `luxarElementIdParts`)
 *
 * Per-point data comes from the RGBA32F point texture (`uPointTex`,
 * 3 texels/point; picking needs texels 0-1 only — center/radius/
 * sharpness), fetched in the vertex stage via `textureLoad` and
 * indexed by `aSortedIndex` (visual-factory parity, shader-tsl.ts).
 *
 * Depth is set to `1.0 - brightness` (brightness-as-depth) so the
 * picking system's tie-breaking prefers the brightest hit. Matches
 * the GLSL `gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0)` path.
 *
 * The pick sprite is 80% of the radius of the visual sprite
 * (the 0.8 factor below) — slightly tighter than the visible disc so
 * overlapping points still resolve to the one whose core you're over,
 * but forgiving enough that sparse points don't need pixel-perfect aim.
 * Keep in sync with shaders.ts.
 *
 * @module rendering/picking/point/pick.tsl
 */

import * as THREE from 'three';
import {
  Fn,
  uniform,
  attribute,
  varying,
  texture,
  vec2 as _vec2,
  vec3 as _vec3,
  vec4 as _vec4,
  ivec2 as _ivec2,
  float,
  int,
  max,
  min,
  clamp,
  dot,
  exp,
  Discard,
  modelViewMatrix,
  cameraProjectionMatrix,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import {
  FALLOFF_FLOOR,
  FALLOFF_K,
  INV_ONE_MINUS_FALLOFF_FLOOR,
} from '../../materials/_shared/falloff';
import {
  perspectiveNearFadeTSL,
  sanitizeNonNegative,
  type TSLNode,
  sortedIndexNode,
  densityDroppedNode,
} from '../../materials/_shared/tsl-helpers';
import {
  getPlaceholderElementTexture,
  resolveElementTextureWidth,
  POINT_TEXTURE_LAYOUT,
} from '../../element-texture-layout';

// Type-erased constructor aliases — same rationale as the gsplat TSL
// factories (see materials/gsplat/shader-tsl.ts).
const vec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _vec2 as TSLNode;
const vec3: (a?: TSLNode, b?: TSLNode, c?: TSLNode) => TSLNode = _vec3 as TSLNode;
const vec4: (a?: TSLNode, b?: TSLNode, c?: TSLNode, d?: TSLNode) => TSLNode = _vec4 as TSLNode;
const ivec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _ivec2 as TSLNode;

/**
 * Pre-created TSL leaf nodes supplied by the wrapper class. See
 * `LineTSLNodes` / `PointTSLNodes` / `GSplatTSLNodes` for the
 * rationale: avoids `.onUpdate('render')` callback churn by using
 * the wrapper-owned `UniformNode` references directly.
 */
export interface PointPickTSLNodes {
  /**
   * Point data texture node (RGBA32F, 3 texels/point) — shared with
   * the visual material's storage; rebound per node by the commit's
   * material sync.
   */
  readonly uPointTex: TSLNode;
  readonly pointSizeFactor: TSLNode;
  readonly maxPointSize: TSLNode;
  readonly radiusScale: TSLNode;
  readonly uIsOrtho: TSLNode;
  /** Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB. */
  readonly uSortedIndexSlot: TSLNode;
  readonly uDensityDrop: TSLNode;
  readonly uNearCull: TSLNode;
  readonly uPixelRatio: TSLNode;
  readonly uNodeId: TSLNode;
  readonly uResolution: TSLNode;
}

/**
 * Point picking material TSL factory.
 *
 * Consumes pre-created `UniformNode` references; the wrapper
 * (`PointPickingTSLMaterial`) owns those nodes and exposes them via
 * `material.uniforms` as `IUniform`-shaped getter/setter proxies.
 */
export function pointPickWebGPUFactory(
  nodes: PointPickTSLNodes,
  outMaterial?: NodeMaterial
): NodeMaterial {
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  // Draw-slot -> storage-slot mapping; point data comes from the point
  // texture (visual-factory parity, shader-tsl.ts).
  const aSortedIndex: TSLNode = sortedIndexNode(nodes.uSortedIndexSlot);
  const densityDropped: TSLNode = densityDroppedNode(nodes.uDensityDrop, aSortedIndex);

  const uPointTex = nodes.uPointTex;
  const uPointSizeFactor = nodes.pointSizeFactor;
  const uMaxPointSize = nodes.maxPointSize;
  const uRadiusScale = nodes.radiusScale;
  const uIsOrtho = nodes.uIsOrtho;
  const uNearCull = nodes.uNearCull;
  const uPixelRatio = nodes.uPixelRatio;
  const uNodeId = nodes.uNodeId;
  const uResolution = nodes.uResolution;

  // ---- Vertex ----
  //
  // Traced inside a single Fn() body with explicit `.toVar()`
  // statements — same load-bearing structure as the visual point
  // factory (materials/point/shader-tsl.ts) and both gsplat factories:
  // inside Fn(), statements emit in trace order, unconditionally, so a
  // shared subexpression can never be first-materialized inside a
  // `.select()` branch.

  // Varyings are declared up front and `.assign()`ed inside the vertex
  // body. nodeId and elementId are flat in the GLSL path; TSL's
  // `varying()` wraps with per-vertex linear interpolation by default —
  // for a single-instance quad all 4 corners carry the same value, so
  // interpolation is the identity (same numeric result).
  const vSpriteCoord: TSLNode = varying(vec2(float(0.0), float(0.0)));
  const vRadius: TSLNode = varying(float(0.0));
  const vBeta: TSLNode = varying(float(0.0));
  const vNearFade: TSLNode = varying(float(1.0));
  const vPickSize: TSLNode = varying(float(0.0));
  const vNodeId: TSLNode = varying(uNodeId);
  // Storage slot, NOT instanceIndex (the draw slot): identical under
  // Phase-1 identity ordering, and stays the id the rest of the
  // pipeline addresses points by once the sort worker permutes draw
  // order (Phase 2+). Mirrors the GLSL pick shader.
  // Storage index split into two 16-bit halves — the TSL twin of
  // `luxarElementIdParts` in glsl-lib.ts. The pick pass carries the index
  // through an RGBA32F buffer and float32 has a 24-bit mantissa, so one
  // channel cannot represent consecutive indices past 16,777,216 while a
  // node's capacity reaches 2^25 on a 32768-texel device. Split in INT
  // space — a float split would already have lost the bit it preserves —
  // and both halves are <= 65535, hence exact. Integer div/sub rather than
  // bit ops so the graph lowers the same way on both backends.
  const elementIdInt: TSLNode = int(aSortedIndex);
  const elementIdHi: TSLNode = elementIdInt.div(int(65536));
  const elementIdLo: TSLNode = elementIdInt.sub(elementIdHi.mul(int(65536)));
  const vElementId: TSLNode = varying(vec2(float(elementIdLo), float(elementIdHi)));

  const vertexBody = Fn(() => {
    // === Point-texture fetch prologue (visual-factory parity) ===
    // Picking needs texels 0-1 only (center/radius/sharpness); color
    // and scalar are not fetched. Every value is a `.toVar()` STATEMENT
    // (Fn house rule). Width is a multiple of 3 -> one row per point.
    const pointBase: TSLNode = int(aSortedIndex).mul(int(3)).toVar();
    // The width is baked as a LITERAL, not read via textureSize(): a
    // compile-time constant lets the shader compiler strength-reduce
    // the per-vertex %/int-div addressing below (measured -7% on the
    // quad's whole GPU pass; a uniform recovered almost none of it).
    // Safe because the width is a per-layout session constant, capped
    // at 4096 on every device (element-texture-layout.ts).
    const pointTexW: TSLNode = int(
      resolveElementTextureWidth(
        POINT_TEXTURE_LAYOUT,
        (nodes.uPointTex as unknown as { value?: { image?: { width?: number } } }).value ?? null
      )
    ).toVar();
    const texelX: TSLNode = pointBase.mod(pointTexW).toVar();
    const texelY: TSLNode = pointBase.div(pointTexW).toVar();
    const pointT0: TSLNode = uPointTex.load(ivec2(texelX, texelY)).toVar();
    const pointT1: TSLNode = uPointTex.load(ivec2(texelX.add(int(1)), texelY)).toVar();
    const aCenter: TSLNode = vec3(pointT0).toVar();
    const aRadius: TSLNode = pointT0.w.toVar();
    const aSharpness: TSLNode = pointT1.w.toVar();

    // Per-point sanitisation — mirrors visual shader-tsl.ts and the GLSL
    // picking shader: sharpness in [0, 1] -> super-Gaussian exponent
    // beta = 2^(6s - 2). sanitizeNonNegative keeps a valid s=0 and routes
    // NaN/Inf/negative to the 0.5 default so the pick footprint can't diverge
    // from the visible footprint.
    const sClamped: TSLNode = clamp(sanitizeNonNegative(aSharpness, float(0.5)), 0.0, 1.0);
    const beta: TSLNode = float(2.0).pow(sClamped.mul(6.0).sub(2.0)).toVar();
    const normalizedRadius: TSLNode = sanitizeNonNegative(
      aRadius.mul(uRadiusScale),
      float(0.0)
    ).toVar();

    // Vertex transform.
    const mvPos: TSLNode = modelViewMatrix.mul(vec4(aCenter, 1.0)).toVar();
    const projCenter: TSLNode = cameraProjectionMatrix.mul(mvPos).toVar();

    // View-space depth, matching the visual point shader (B9a) so the
    // pick footprint stays congruent with the visible sprite. 1e-20 =
    // pure INF guard (near-fade reject bounds surviving depths at the
    // scene-relative ~uNearCull; the size clamp bounds the output).
    const invDistance: TSLNode = int(uIsOrtho)
      .equal(int(1))
      .select(float(1.0), mvPos.z.negate().max(float(1e-20)).reciprocal());
    const basePointSize: TSLNode = normalizedRadius.mul(uPointSizeFactor).mul(invDistance).toVar();

    // Picking footprint: × 0.8 vs the visual material (keep the 0.8 in sync
    // with shaders.ts). No sharpness size compensation — the shifted-truncated
    // super-Gaussian truncates at the sprite edge, so basePointSize IS the
    // visible extent (matches shader-tsl.ts).
    // 1.5px floor tracks the VISUAL sprite floor (the drawn outer ring
    // stays pickable); keep in sync with shaders.ts.
    const rawPickSize: TSLNode = basePointSize.mul(0.8).toVar();
    const appearancePixelRatio: TSLNode = uPixelRatio.max(float(1.0));
    const pickPointSize: TSLNode = clamp(rawPickSize, appearancePixelRatio.mul(1.5), uMaxPointSize);

    const offsetClip: TSLNode = aQuadCorner.mul(pickPointSize.div(uResolution)).mul(projCenter.w);
    // Reject points behind the camera (perspective only; camera looks down -Z).
    // Unified near handling — keep in sync with the visual point shader
    // and the line/gsplat pick guards: pickability tracks visibility
    // (behind-camera fade 0 — projCenter.w <= 0 there would flip the
    // sprite; smooth [nearCull, 2*nearCull] fade; ortho = 1, NDC clip
    // authority). 1e-20 floor = degenerate-smoothstep guard only;
    // uNearCull is scene-bounds-scaled (see the visual point shader).
    const depthFade: TSLNode = perspectiveNearFadeTSL(
      uIsOrtho,
      mvPos.z,
      max(uNearCull, float(1e-20))
    ).toVar();
    const clipPos: TSLNode = depthFade
      .lessThan(0.01)
      .or(densityDropped)
      .select(vec4(0.0, 0.0, -2.0, 1.0), projCenter.add(vec4(offsetClip, 0.0, 0.0)));

    // Assign varyings (declared outside the Fn; see above).
    vSpriteCoord.assign(aQuadCorner.add(1.0).mul(0.5));
    vRadius.assign(normalizedRadius);
    vBeta.assign(beta);
    vNearFade.assign(depthFade);
    vPickSize.assign(rawPickSize);

    return clipPos;
  });

  const clipPos: TSLNode = vertexBody();

  // ---- Fragment ----
  //
  // Compute the super-Gaussian brightness ONCE, materialised via
  // `.toVar()` so both color and depth fragment outputs reference the
  // same computation instead of each rebuilding the pow + exp chain —
  // the same "compile once, reference twice" pattern as the gsplat
  // pick factory (picking/gsplat/pick.tsl.ts).
  const centered: TSLNode = vec2(vSpriteCoord.sub(0.5));
  const r2: TSLNode = dot(centered, centered).toVar();
  const normalizedR: TSLNode = r2.mul(4.0).sqrt();
  // Shifted-truncated super-Gaussian (matches shader-tsl.ts).
  const K = FALLOFF_K; // ln(100)
  const C = FALLOFF_FLOOR; // exp(-K) = floor
  const invOneMinusC = INV_ONE_MINUS_FALLOFF_FLOOR;
  // nearFade folded into brightness (matches gsplat pick).
  const brightness: TSLNode = exp(normalizedR.pow(vBeta).mul(-K))
    .sub(C)
    .max(float(0.0))
    .mul(invOneMinusC)
    .mul(vNearFade)
    // Sub-pixel compensation (sizeScale², matching the VISUAL point and
    // the line pick's widthScale) — pick salience tracks visual salience.
    .mul(min(vPickSize.div(uPixelRatio.max(float(1.0)).mul(1.5)), float(1.0)).pow(2.0))
    .toVar();

  const colorNode = Fn(() => {
    // Exact-zero only — see the GLSL twin's comment.
    Discard(vRadius.lessThanEqual(0.0));
    Discard(r2.greaterThan(0.25));
    Discard(brightness.lessThan(1e-4));

    return vec4(vNodeId, vElementId.x, brightness, vElementId.y);
  });

  // Depth = 1.0 - brightness (the brightest hit takes precedence).
  // Discard-gating happens via colorNode → depth is only written when
  // colorNode also writes.
  //
  // BRANCHLESS by construction, and that is load-bearing: `brightness`
  // is a factory-scope `.toVar()` shared with `colorNode`, so it is
  // assigned wherever three first BUILDS it — which is unconditional
  // top-level flow in either entry point only while this body contains
  // no `if`. Adding a branch here (a `uSurfaceDepth`-style select) would
  // bury that assignment in one arm and leave `colorNode`'s top-level
  // readers with 0; it needs the same unconditional fragment prologue
  // the gsplat/mesh pick factories use.
  const depthNode = Fn(() => float(1.0).sub(clamp(brightness, 0.0, 1.0)));

  const material = outMaterial ?? new NodeMaterial();
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.depthNode = depthNode();
  material.toneMapped = false;
  material.depthTest = true;
  material.depthWrite = true;
  material.transparent = false;
  // The element index's HIGH half rides in alpha, and THREE's NodeMaterial
  // appends `DiffuseColor.w *= material.opacity` to every generated fragment
  // (see the codegen snapshots, and `tsl-opacity-tail.test.ts` for the same
  // tail on the visual materials). NoBlending does not suppress that
  // shader-side multiply, so any opacity other than exactly 1 would scale the
  // high half and decode a WRONG element id — on the TSL path only, since the
  // GLSL twins have no such tail. Pin it so the multiply is provably identity,
  // including when a caller injects `outMaterial`.
  material.opacity = 1;
  // Picking output is an opaque ID buffer; any blending would
  // smear nodeId / elementId values across overlapping picks and
  // produce nonsense readbacks. Matches the GLSL picking material.
  material.blending = THREE.NoBlending;
  return material;
}

/**
 * Snapshot adapter: build a `PointPickTSLNodes` set from a flat
 * `IUniform` record. Used by callers that don't own persistent nodes.
 * See `buildLineTSLNodesFromUniforms` for the rationale.
 */
export function buildPointPickTSLNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>
): PointPickTSLNodes {
  return {
    // Point data texture -- bound from the caller's uniform when present,
    // else the shared placeholder (codegen-only consumers).
    uPointTex: texture(
      (uniforms.uPointTex?.value as THREE.Texture | null) ?? getPlaceholderElementTexture()
    ),
    pointSizeFactor: uniform((uniforms.pointSizeFactor?.value as number) ?? 1.0),
    maxPointSize: uniform((uniforms.maxPointSize?.value as number) ?? 1.0),
    radiusScale: uniform((uniforms.radiusScale?.value as number) ?? 1.0),
    uIsOrtho: uniform((uniforms.uIsOrtho?.value as number) ?? 0),
    uSortedIndexSlot: uniform((uniforms.uSortedIndexSlot?.value as number) ?? 0),
    uDensityDrop: uniform((uniforms.uDensityDrop?.value as number) ?? 0),
    uNearCull: uniform((uniforms.uNearCull?.value as number) ?? 0.1),
    uPixelRatio: uniform((uniforms.uPixelRatio?.value as number) ?? 1),
    uNodeId: uniform((uniforms.uNodeId?.value as number) ?? 0),
    uResolution: uniform(
      (uniforms.uResolution?.value as THREE.Vector2 | undefined) ?? new THREE.Vector2(1, 1)
    ),
  };
}
