/**
 * Line picking material TSL factory — NodeMaterial counterpart to
 * `LINE_PICK_SOURCE` in `shaders.ts`.
 *
 * Mirrors the visual factory's sprite-expansion math, but the fragment
 * outputs:
 *   - R: nodeId (set via uniform)
 *   - G: elementId LOW 16 bits (= `aSortedIndex`, the STORAGE slot —
 *     identical to the draw slot under identity ordering, and stays
 *     correct once the sort worker permutes draw order)
 *   - B: brightness (perpendicular falloff × cap × widthScale × widthFade)
 *   - A: the same elementId's HIGH 16 bits (one f32 channel cannot
 *     carry the whole index exactly — see `luxarElementIdParts`)
 *
 * Per-segment data comes from the RGBA32F line texture (`uLineTex`,
 * 6 texels/segment — layout in `rendering/line-geometry.ts`), fetched
 * in the vertex stage and indexed by `aSortedIndex` (visual-factory
 * parity, shader-tsl.ts).
 *
 * Depth = 1.0 - brightness (brightness-as-depth tie-breaking).
 *
 * Lines use FULL width for picking (no half-radius like points) —
 * thin lines would be impossible to pick otherwise.
 *
 * @module rendering/picking/line/pick.tsl
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
  exp,
  texture,
  textureSize,
  modelViewMatrix,
  cameraProjectionMatrix,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import {
  FALLOFF_FLOOR,
  FALLOFF_K,
  INV_ONE_MINUS_FALLOFF_FLOOR,
} from '../../materials/_shared/falloff';
import {
  perspectiveNearFadeStaticTSL,
  sanitizeNonNegative,
  type TSLNode,
  sortedIndexNode,
  tslLineJointCapSuppression,
} from '../../materials/_shared/tsl-helpers';
import { getPlaceholderElementTexture } from '../../element-texture-layout';

/**
 * Pre-created TSL leaf nodes supplied by the wrapper class. Same
 * pattern as `PointPickTSLNodes` / `GSplatPickTSLNodes`: consumers
 * own the `UniformNode`s and the factory references them directly,
 * avoiding the `.onUpdate('render')` callback churn.
 */
export interface LinePickTSLNodes {
  /**
   * Line data texture node (RGBA32F, 6 texels/segment). Rebound per
   * node by the wrapper's `updateLineTexture` (fresh node + rebuild).
   */
  readonly uLineTex: TSLNode;
  readonly uResolution: TSLNode;
  readonly uIsOrtho: TSLNode;
  /** Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB. */
  readonly uSortedIndexSlot: TSLNode;
  readonly uNodeId: TSLNode;
  readonly uNearCull: TSLNode;
  readonly uMaxLinePixelWidth: TSLNode;
  /** = resolution.y / tan(fov * 0.5), precomputed by the wrapper. */
  readonly uPerspectiveLineScale: TSLNode;
  /** = 2 * resolution.y / frustumHeight, precomputed by the wrapper. */
  readonly uOrthoLineScale: TSLNode;
}

/**
 * Line picking material TSL factory.
 *
 * Consumes pre-created `UniformNode` references via `nodes`; the
 * wrapper class (`LinePickingTSLMaterial`) owns those nodes and
 * exposes them through `material.uniforms` as `IUniform`-shaped
 * getter/setter proxies. The harness / `shaders.ts` ShaderSource
 * registry constructs the nodes from a plain `uniforms` record via
 * {@link buildLinePickTSLNodesFromUniforms}.
 */
/**
 * Per-build configuration for the line-pick factory.
 */
export interface LinePickTSLConfig {
  /**
   * Camera projection mode at build time — mirrors `LineTSLConfig`.
   * When true, only the ortho pixel-width branch is emitted; when
   * false or undefined, only the perspective branch.
   */
  readonly isOrtho?: boolean;
}

export function linePickWebGPUFactory(
  nodes: LinePickTSLNodes,
  config: LinePickTSLConfig = {},
  outMaterial?: NodeMaterial
): NodeMaterial {
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  // The ordering attributes (visual-factory parity): segment
  // data lives in the line texture; `aSortedIndex` maps the draw slot
  // to a storage slot.
  const aSortedIndex: TSLNode = sortedIndexNode(nodes.uSortedIndexSlot);

  // Pixel-width math consumes the CPU-precomputed
  // uPerspectiveLineScale / uOrthoLineScale (no FOV uniform exists).
  // uIsOrtho is unbound — projection mode is a JS-level config
  // branch (`config.isOrtho`), not a runtime uniform.
  const uLineTex = nodes.uLineTex;
  const uResolution = nodes.uResolution;
  const uNodeId = nodes.uNodeId;
  const uNearCull = nodes.uNearCull;
  const uMaxLinePixelWidth = nodes.uMaxLinePixelWidth;
  const uPerspectiveLineScale = nodes.uPerspectiveLineScale;
  const uOrthoLineScale = nodes.uOrthoLineScale;

  // ---- Vertex computation (mirrors line shader-tsl.ts exactly) ----
  //
  // Traced inside a single Fn() body with explicit `.toVar()`
  // statements — same load-bearing structure as the visual factory
  // (see shader-tsl.ts): inside Fn(), statements emit in trace order,
  // unconditionally, so texel fetches can't land inside a select()
  // branch and read uninitialized on the other path.

  const nearCull: TSLNode = max(uNearCull, float(1e-20));

  // Varyings declared up front, `.assign()`ed inside the vertex body.
  // Per-segment-constant values (segment length, cap suppression, node
  // id, element id) use `flat` interpolation — matches the GLSL3
  // `flat` qualifier on the same fields.
  const vSharpness: TSLNode = varying(float(0.0));
  const vPerpNorm: TSLNode = varying(float(0.0));
  const vT: TSLNode = varying(float(0.0));
  const vSegmentLength: TSLNode = varying(float(0.0)).setInterpolation('flat');
  const vWidthAtT: TSLNode = varying(float(0.0));
  const vPixelWidth: TSLNode = varying(float(0.0));
  const vWidthFade: TSLNode = varying(float(0.0));
  // View-space z to the fragment (fade computed per-fragment; see the
  // visual line TSL). Ortho graphs skip it.
  const vViewZ: TSLNode | null = config.isOrtho ? null : varying(float(0.0));
  const vCapSuppressStart: TSLNode = varying(float(0.0)).setInterpolation('flat');
  const vCapSuppressEnd: TSLNode = varying(float(0.0)).setInterpolation('flat');
  const vNodeId: TSLNode = varying(uNodeId).setInterpolation('flat');
  // Storage slot, NOT instanceIndex (the draw slot): identical under
  // identity ordering, and stays correct once the sort worker permutes
  // draw order.
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
  const vElementId: TSLNode = varying(
    vec2(float(elementIdLo), float(elementIdHi))
  ).setInterpolation('flat');

  const vertexBody = Fn(() => {
    // === Line-texture fetch prologue (visual-shader parity) ===
    // Width is a multiple of 6, so a segment's texels share one row.
    // Colors (texels 2/3 .rgb) and scalars (texel 5) are not needed
    // for picking; only the .w sharpness of texels 2/3 is read.
    const lineBase: TSLNode = int(aSortedIndex).mul(int(6)).toVar();
    // int() wrap is LOAD-BEARING — see the visual factory
    // (shader-tsl.ts) for the WebGL2-fallback rationale.
    const lineTexW: TSLNode = int((textureSize(uLineTex, int(0)) as unknown as TSLNode).x).toVar();
    const texelX: TSLNode = lineBase.mod(lineTexW).toVar();
    const texelY: TSLNode = lineBase.div(lineTexW).toVar();
    const lineT0: TSLNode = uLineTex.load(ivec2(texelX, texelY)).toVar();
    const lineT1: TSLNode = uLineTex.load(ivec2(texelX.add(int(1)), texelY)).toVar();
    const lineT2: TSLNode = uLineTex.load(ivec2(texelX.add(int(2)), texelY)).toVar();
    const lineT3: TSLNode = uLineTex.load(ivec2(texelX.add(int(3)), texelY)).toVar();
    const lineT4: TSLNode = uLineTex.load(ivec2(texelX.add(int(4)), texelY)).toVar();
    const aStartPos: TSLNode = vec3(lineT0).toVar();
    const aStartWidth: TSLNode = lineT0.w.toVar();
    const aEndPos: TSLNode = vec3(lineT1).toVar();
    const aEndWidth: TSLNode = lineT1.w.toVar();
    const aStartSharpness: TSLNode = lineT2.w.toVar();
    const aEndSharpness: TSLNode = lineT3.w.toVar();
    const aSegmentLength: TSLNode = lineT4.x.toVar();
    const aStartJointCode: TSLNode = lineT4.y.toVar();
    const aEndJointCode: TSLNode = lineT4.z.toVar();

    // Branchless: aQuadCorner.x ∈ {-1, +1} by construction.
    const t: TSLNode = aQuadCorner.x.mul(0.5).add(0.5).toVar();
    const startW: TSLNode = sanitizeNonNegative(aStartWidth, float(0.0));
    const endW: TSLNode = sanitizeNonNegative(aEndWidth, float(0.0));
    // Sharpness is a [0, 1] knob -> super-Gaussian exponent beta = 2^(6s - 2)
    // (computed in the fragment). A valid s=0 must NOT be rejected, so clamp a
    // non-negative-sanitised value into [0, 1] with the 0.5 default. Mirrors
    // the visual shaders.
    const startS: TSLNode = clamp(sanitizeNonNegative(aStartSharpness, float(0.5)), 0.0, 1.0);
    const endS: TSLNode = clamp(sanitizeNonNegative(aEndSharpness, float(0.5)), 0.0, 1.0);

    const mvStart: TSLNode = modelViewMatrix.mul(vec4(aStartPos, 1.0)).toVar();
    const mvEnd: TSLNode = modelViewMatrix.mul(vec4(aEndPos, 1.0)).toVar();
    const startDepth: TSLNode = mvStart.z.negate().toVar();
    const endDepth: TSLNode = mvEnd.z.negate().toVar();

    // Near-plane SEGMENT clipping — visual-shader parity (see the
    // visual factory / shader-glsl.ts for the full rationale: a
    // behind-camera endpoint has clip w <= 0, which flips the
    // clip-space expansion and rasterizes the quad as a twisted bowtie
    // whose near-clip boundary cuts through the pick footprint). Keeps
    // every vertex at viewZ >= nearCull and remaps t (tEff) so the cap
    // math and per-endpoint attributes keep the original
    // parameterization. select() evaluates both branches, so each
    // denominator gets a benign 1.0 only in its UNTAKEN lane. The taken lane
    // keeps the exact positive depth delta, matching the GLSL division.
    let tA: TSLNode = float(0.0);
    let tB: TSLNode = float(1.0);
    if (!config.isOrtho) {
      const startNear: TSLNode = startDepth
        .lessThan(nearCull)
        .and(endDepth.greaterThanEqual(nearCull));
      const endNear: TSLNode = endDepth
        .lessThan(nearCull)
        .and(startDepth.greaterThanEqual(nearCull));
      const startDenominator: TSLNode = startNear
        .select(endDepth.sub(startDepth), float(1.0))
        .toVar();
      const endDenominator: TSLNode = endNear.select(startDepth.sub(endDepth), float(1.0)).toVar();
      tA = startNear
        .select(nearCull.sub(startDepth).div(startDenominator).toVar(), float(0.0))
        .toVar();
      tB = endNear.select(startDepth.sub(nearCull).div(endDenominator).toVar(), float(1.0)).toVar();
      const mvStartClipped: TSLNode = mix(mvStart, mvEnd, tA).toVar();
      const mvEndClipped: TSLNode = mix(mvStart, mvEnd, tB).toVar();
      mvStart.assign(mvStartClipped);
      mvEnd.assign(mvEndClipped);
    }
    const tEff: TSLNode = mix(tA, tB, t).toVar();

    const width: TSLNode = mix(startW, endW, tEff).toVar();
    const vSharpnessVal: TSLNode = mix(startS, endS, tEff);
    const mvPos: TSLNode = mix(mvStart, mvEnd, t).toVar();

    // PERSPECTIVE ONLY (compile-time graph variant; see the visual line
    // TSL): ortho graphs carry no cull/fade code — NDC clipping is the
    // sole cull authority there. Reads the ORIGINAL depths (computed
    // before segment clipping above).
    // 1e-20 floor = uNearCull == 0 guard only; uNearCull is
    // scene-bounds-scaled (see the visual line shader — an absolute 1e-4
    // floor culled every segment of a tiny-unit scene).
    const bothBehind: TSLNode | null = config.isOrtho
      ? null
      : startDepth.lessThan(nearCull).and(endDepth.lessThan(nearCull));

    const clipStart: TSLNode = cameraProjectionMatrix.mul(mvStart).toVar();
    const clipEnd: TSLNode = cameraProjectionMatrix.mul(mvEnd).toVar();
    // projection is linear, so proj * mix(a,b,t) == mix(proj*a, proj*b, t).
    const clipPosBase: TSLNode = mix(clipStart, clipEnd, t).toVar();

    // Scene-relative w guard (w == -viewZ under perspective; ortho
    // graphs use the inert 1.0 — compile-time variant). See the visual
    // line shader for the scale-free rationale.
    const wGuard: TSLNode = config.isOrtho ? float(1.0) : nearCull;
    const wStart: TSLNode = max(clipStart.w, wGuard);
    const wEnd: TSLNode = max(clipEnd.w, wGuard);
    const ndcStart: TSLNode = vec2(clipStart.xy.div(wStart)).toVar();
    const ndcEnd: TSLNode = vec2(clipEnd.xy.div(wEnd)).toVar();

    // The +0.5 in (ndc*0.5+0.5)*resolution cancels under subtraction.
    const pixelDir: TSLNode = vec2(ndcEnd.sub(ndcStart).mul(uResolution.mul(0.5))).toVar();
    const pixelLen: TSLNode = length(pixelDir).toVar();
    // `.toVar()` on the chained branch keeps the sharp edge stable.
    const lineDir: TSLNode = pixelLen
      .greaterThan(0.0001)
      .select(vec2(pixelDir.div(pixelLen)).toVar(), vec2(1.0, 0.0))
      .toVar();
    const perpendicular: TSLNode = vec2(lineDir.y.negate(), lineDir.x).toVar();

    // Each camera projection mode is a separate graph variant so the
    // unused branch never materialises. Wrapper rebuilds when isOrtho
    // flips. View-space depth (-mvPos.z) matches the visual shader.
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
    const vWidthFadeVal: TSLNode = rawPixelWidth
      .lessThanEqual(maxPW)
      .select(float(1.0), maxPW.div(max(rawPixelWidth, float(1e-4))).toVar());

    // Pathological-segment cull (visual-shader parity): both endpoints
    // inside near-cull margin AND the pixel width blows past the clamp by
    // 2× → degenerate to off-screen. Otherwise picking still rasterizes
    // the half-viewport quad the visual pass already culled.
    // Segment-constant, not per-quad-vertex: the per-vertex rawPixelWidth
    // term differs between the t=0 and t=1 corners, so gating on it would
    // sentinel only half the quad and leave a visible wedge (issue #849).
    // Gate on the MAX of the pixel width at both clipped endpoints so all
    // four vertices take the same branch.
    let pathological: TSLNode | null = null;
    if (!config.isOrtho) {
      const startPixelWidth: TSLNode = mix(startW, endW, tA)
        .mul(uPerspectiveLineScale)
        .div(max(mvStart.z.negate(), nearCull));
      const endPixelWidth: TSLNode = mix(startW, endW, tB)
        .mul(uPerspectiveLineScale)
        .div(max(mvEnd.z.negate(), nearCull));
      const segMaxPixelWidth: TSLNode = max(startPixelWidth, endPixelWidth).toVar();
      pathological = startDepth
        .lessThan(nearCull.mul(2.0))
        .and(endDepth.lessThan(nearCull.mul(2.0)))
        .and(segMaxPixelWidth.greaterThan(maxPW.mul(2.0)));
    }

    const pixelOffset: TSLNode = perpendicular.mul(aQuadCorner.y).mul(clampedPixelWidth);
    const ndcOffset: TSLNode = pixelOffset.div(uResolution).mul(2.0);
    const expandedClip: TSLNode = vec4(
      clipPosBase.xy.add(ndcOffset.mul(clipPosBase.w)),
      clipPosBase.z,
      clipPosBase.w
    ).toVar();

    // Real TSL control flow — see the visual factory for the rationale
    // (one branch per draw instead of evaluating both via select()).
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
    vSharpness.assign(vSharpnessVal);
    vPerpNorm.assign(aQuadCorner.y);
    vT.assign(tEff);
    vSegmentLength.assign(aSegmentLength);
    vWidthAtT.assign(width);
    vPixelWidth.assign(rawPixelWidth);
    vWidthFade.assign(vWidthFadeVal);
    if (vViewZ) vViewZ.assign(mvPos.z);
    // texel4.yz hold a per-endpoint joint CODE, not a [0, 1] scalar. Reading
    // it as one let capFactor scale with the partner's slot index (200.5 for a
    // segment joining slot 399, -0.5 for a hub, 0.0 for a slice-clipped end).
    // Decode it exactly as the GLSL twin does, via the shared helper, so the
    // two backends agree; neither TSL path carries join geometry yet, so both
    // stop at the code-implied cap.
    vCapSuppressStart.assign(tslLineJointCapSuppression(aStartJointCode));
    vCapSuppressEnd.assign(tslLineJointCapSuppression(aEndJointCode));

    return clipPosOut;
  });

  const clipPos: TSLNode = vertexBody();

  // ---- Fragment: brightness output + brightness-as-depth ----

  // Compute brightness once and materialise it into a fragment-local
  // variable via `.toVar()`. Both `colorNode` and `depthNode` then
  // reference the same variable so the pow/cap/widthScale chain
  // doesn't get inlined twice into the generated WGSL/GLSL. The
  // `.once()` on the underlying `Fn(...)` ensures the function body
  // is emitted exactly once even if multiple call sites elsewhere
  // bind to it.
  const brightnessShared = Fn(() => {
    const p: TSLNode = vPerpNorm.abs();
    // Shifted-truncated super-Gaussian perpendicular cross-section —
    // visual-shader parity. beta = 2^(6s - 2) from the [0, 1] knob.
    // K = ln(100), C = exp(-K).
    const K = FALLOFF_K;
    const C = FALLOFF_FLOOR;
    const invOneMinusC = INV_ONE_MINUS_FALLOFF_FLOOR;
    const beta: TSLNode = float(2.0).pow(vSharpness.mul(6.0).sub(2.0));
    const perpFalloff: TSLNode = exp(p.pow(beta).mul(-K)).sub(C).max(float(0.0)).mul(invOneMinusC);
    const minPW = float(1.5);
    const widthScale: TSLNode = min(vPixelWidth.div(minPW), float(1.0));

    const distFromStart: TSLNode = vT.mul(vSegmentLength);
    const distFromEnd: TSLNode = float(1.0).sub(vT).mul(vSegmentLength);
    // Scale-free ratio; 1e-20 = pure div-by-zero guard (visual twin).
    const startRamp: TSLNode = vWidthAtT
      .greaterThan(float(1e-20))
      .select(clamp(distFromStart.div(vWidthAtT), 0.0, 1.0).toVar(), float(1.0));
    const endRamp: TSLNode = vWidthAtT
      .greaterThan(float(1e-20))
      .select(clamp(distFromEnd.div(vWidthAtT), 0.0, 1.0).toVar(), float(1.0));
    // Per-endpoint cap lifted by its own suppression, combined with
    // min() — removes the intra-segment midpoint jump (visual twin;
    // a residual sub-width joint-seam step is documented there).
    const startCap: TSLNode = mix(
      float(0.5).add(startRamp.mul(0.5)),
      float(1.0),
      vCapSuppressStart
    );
    const endCap: TSLNode = mix(float(0.5).add(endRamp.mul(0.5)), float(1.0), vCapSuppressEnd);
    const capFactor: TSLNode = min(startCap, endCap);

    return capFactor
      .mul(perpFalloff)
      .mul(widthScale)
      .mul(vWidthFade)
      .mul(vViewZ ? perspectiveNearFadeStaticTSL(false, vViewZ, nearCull) : float(1.0));
  }).once();
  const brightness: TSLNode = brightnessShared().toVar('lineBrightness');

  const colorNode = Fn(() => {
    const p: TSLNode = vPerpNorm.abs();
    Discard(p.greaterThanEqual(1.0));
    Discard(brightness.lessThan(1e-4));
    return vec4(vNodeId, vElementId.x, brightness, vElementId.y);
  });

  const depthNode = Fn(() => {
    return float(1.0).sub(clamp(brightness, 0.0, 1.0));
  });

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
  // Picking output is an opaque ID buffer; any blending would smear
  // nodeId / elementId across overlapping picks. Matches the GLSL
  // picking material.
  material.blending = THREE.NoBlending;
  return material;
}

/**
 * Build a `LinePickTSLNodes` set from a plain `IUniform` record. Used
 * by the harness and the `LINE_PICK_SOURCE` ShaderSource factory in
 * `shaders.ts` — callers that don't own persistent
 * wrapper-side `UniformNode`s. Symmetric with
 * `buildPointPickTSLNodesFromUniforms` /
 * `buildGSplatPickTSLNodesFromUniforms`.
 */
export function buildLinePickTSLNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>
): LinePickTSLNodes {
  return {
    // Line data texture — bound from the caller's uniform when present,
    // else the shared placeholder so codegen-only consumers still build
    // a valid graph.
    uLineTex: texture(
      (uniforms.uLineTex?.value as THREE.Texture | null) ?? getPlaceholderElementTexture()
    ),
    uResolution: uniform(
      (uniforms.uResolution?.value as THREE.Vector2 | undefined) ?? new THREE.Vector2(1, 1)
    ),
    uIsOrtho: uniform((uniforms.uIsOrtho?.value as number) ?? 0),
    uSortedIndexSlot: uniform((uniforms.uSortedIndexSlot?.value as number) ?? 0),
    uNodeId: uniform((uniforms.uNodeId?.value as number) ?? 0),
    uNearCull: uniform((uniforms.uNearCull?.value as number) ?? 1e-4),
    uMaxLinePixelWidth: uniform((uniforms.uMaxLinePixelWidth?.value as number) ?? 1.0),
    uPerspectiveLineScale: uniform((uniforms.uPerspectiveLineScale?.value as number) ?? 1.0),
    uOrthoLineScale: uniform((uniforms.uOrthoLineScale?.value as number) ?? 1.0),
  };
}
