/**
 * Volumetric line primitive — TSL / NodeMaterial twin of
 * `shader-glsl-volumetric.ts` (issue #1352, behind
 * `?linePrimitive=volumetric`).
 *
 * Same model, same lanes, same constants: the segment ⊛ isotropic-Gaussian
 * density, integrated (sum-family blending) or maximised (peak-family) along
 * the per-fragment camera ray, with bisector-cut interior joints and
 * erf-capped soft ends. The lane math is the quadrature-validated CPU
 * reference in `_shared/line-volumetric.ts` — edit THERE first, prove it,
 * then mirror here and in the GLSL twin. Value-level parity between the two
 * backends is enforced by the `line-volprim-*` fixtures in the
 * tsl-shader-parity suite (`line-volumetric-*` is the unrelated
 * volumetric-BLENDING fixture family on the screen-space quad).
 *
 * Structural notes (the TSL house rules, shared with `shader-tsl.ts`):
 * - the whole vertex stage is ONE `Fn()` body of `.toVar()` statements;
 *   varyings are declared outside and `.assign()`ed inside; per-segment
 *   constants are `flat` (written identically on every vertex — WGSL
 *   provokes first-vertex, WebGL last-vertex).
 * - build-time JS branches replace GLSL defines: `config.isOrtho` (ray
 *   setup), `usesPeakProjection(config.blendingMode)` (peak vs sum body),
 *   `config.useColormap`. Runtime lane selection inside the fragment uses
 *   `If/ElseIf/Else` with lane results `.assign()`ed to a shared var.
 * - every GLSL lane discard is mirrored EXACTLY (per-lane `killed` flag,
 *   one `Discard` after the lane chain, plus the early radial reject):
 *   under a no-blend target a discarded fragment preserves the pixel while
 *   a written zero stomps it, so discard-vs-zero is observable wherever
 *   segments overlap.
 * - `screenCoordinate` is TOP-LEFT-origin on both backends; un-flip with
 *   `screenSize` (NOT uResolution — the bound target may be SSAA-scaled;
 *   see the gsplat TSL fragment for the full note).
 */
import {
  Fn,
  If,
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
  smoothstep,
  exp,
  sqrt,
  inverseSqrt,
  dot,
  abs,
  textureSize,
  modelViewMatrix,
  cameraProjectionMatrix,
  screenCoordinate,
  screenSize,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import {
  FALLOFF_FLOOR,
  FALLOFF_K,
  GAUSSIAN_EQUIVALENT_TRUNCATION,
  INV_ONE_MINUS_FALLOFF_FLOOR,
} from '../_shared/falloff';
import { erfAsTSL, erfPolyTSL } from '../_shared/erf';
import {
  LINE_PARALLEL_LANE_THRESHOLD,
  LINE_SIGMA_PER_WIDTH,
  LINE_STENCIL_DILATION,
} from '../_shared/line-volumetric';
import {
  perspectiveNearFadeStaticTSL,
  sanitizeAlpha,
  sanitizeNonNegative,
  type TSLNode,
  sortedIndexNode,
} from '../_shared/tsl-helpers';
import {
  ALPHA_CLAMP,
  VOLUMETRIC_SERIES_C1,
  VOLUMETRIC_SERIES_C2_DIVISOR,
  VOLUMETRIC_SERIES_TAU_THRESHOLD,
  VOLUMETRIC_TAU_EPS,
} from '../_shared/volumetric';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  isVolumetricMode,
  usesPeakProjection,
} from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';
import type { LineTSLConfig, LineTSLNodes } from './shader-tsl';

const T = GAUSSIAN_EQUIVALENT_TRUNCATION;
const T_SQ = T * T;
const T_SQ_DILATION = T * T * LINE_STENCIL_DILATION;
const INV_SQRT2 = Math.SQRT1_2;
const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);
const INV_SQRT_PI = 1 / Math.sqrt(Math.PI);
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);
const QUARTER_SQRT2 = 1 / (2 * Math.SQRT2);

/**
 * Ψ(x) = ∫ₓ^∞ ½(1 − erf(u)) du — the axial cap remainder for the
 * structurally-parallel mixed lane. Twin of `luxarErfCapRemainder`
 * (GLSL) / `erfCapRemainder` (CPU reference); the ±6 saturation guards
 * are load-bearing (an unbounded plane crossing feeds ±1e30 here).
 * Value-level `select` with `.toVar()`ed branches — both sides are
 * finite for any finite input, so unconditional evaluation is safe.
 */
function erfCapRemainderTSL(x: TSLNode): TSLNode {
  const body = float(0.5)
    .mul(
      exp(x.mul(x).negate())
        .mul(INV_SQRT_PI)
        .sub(x.mul(float(1.0).sub(erfAsTSL(x))))
    )
    .toVar();
  return x
    .greaterThan(6.0)
    .select(float(0.0), x.lessThan(-6.0).select(x.negate().toVar(), body))
    .toVar();
}

/**
 * Volumetric-line TSL factory. Same `nodes`/`config` contract as
 * `lineWebGPUFactory` (the wrapper class and the parity harness feed
 * both from the same records); `config.join` is ignored — the volumetric
 * primitive has no join geometry, interior joints are bisector cuts.
 */
export function volumetricLineWebGPUFactory(
  nodes: LineTSLNodes,
  config: LineTSLConfig = {},
  outMaterial?: NodeMaterial
): NodeMaterial {
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  const aSortedIndex: TSLNode = sortedIndexNode(nodes.uSortedIndexSlot);

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
        'volumetricLineWebGPUFactory: config.useColormap=true but nodes.uColormapTex / uScalarMin / uScalarScale are not bound.'
      );
    }
  }
  const uColormapTex = config.useColormap ? nodes.uColormapTex! : null;
  const uScalarMin = config.useColormap ? nodes.uScalarMin! : null;
  const uScalarScale = config.useColormap ? nodes.uScalarScale! : null;

  const blendingMode: BlendingMode = config.blendingMode ?? 'additive';
  const premultiplyRGB =
    config.useMaxRGBContribution !== undefined
      ? config.useMaxRGBContribution
      : blendingMode === 'max';
  const volumetricGraph = isVolumetricMode(blendingMode);
  // Peak (max/normal/opaque) vs sum (additive/luminous/volumetric) ray
  // projection — the GLSL twin's LUXAR_PEAK_PROJECTION define, here a
  // build-time graph variant (the wrapper rebuilds on projection change).
  const peakGraph = usesPeakProjection(blendingMode);
  const isOrtho = config.isOrtho === true;

  // ---- Varyings (all per-segment constants → flat) ----
  const vSegA: TSLNode = varying(vec4(0.0, 0.0, 0.0, 0.0)).setInterpolation('flat');
  const vSegW: TSLNode = varying(vec4(1.0, 0.0, 0.0, 0.0)).setInterpolation('flat');
  const vEnds: TSLNode = varying(vec4(0.0, 0.0, 0.5, 0.5)).setInterpolation('flat');
  const vMisc: TSLNode = varying(vec4(1.0, 1.0, 0.0, 0.0)).setInterpolation('flat');
  const vCutA: TSLNode = varying(vec4(0.0, 0.0, 0.0, 0.0)).setInterpolation('flat');
  const vCutB: TSLNode = varying(vec4(0.0, 0.0, 0.0, 0.0)).setInterpolation('flat');
  // Colormap scalars vs endpoint colors: compile-time variant (elided
  // varying pattern — same as vViewZ in the screen-space factory).
  const vScalars: TSLNode | null = config.useColormap
    ? varying(vec2(0.0, 0.0)).setInterpolation('flat')
    : null;
  const vColor0: TSLNode | null = config.useColormap
    ? null
    : varying(vec3(0.0, 0.0, 0.0)).setInterpolation('flat');
  const vColor1: TSLNode | null = config.useColormap
    ? null
    : varying(vec3(0.0, 0.0, 0.0)).setInterpolation('flat');

  const nearCull: TSLNode = max(uNearCull, float(1e-20));

  const vertexBody = Fn(() => {
    // === Line-texture fetch (6 texels/segment, one row) ===
    const lineBase: TSLNode = int(aSortedIndex).mul(int(6)).toVar();
    // int() wrap is LOAD-BEARING (WebGL2 textureSize returns int) — see
    // the screen-space factory's note.
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
    const aEndPos: TSLNode = vec3(lineT1).toVar();
    const aStartJointCode: TSLNode = lineT4.y.toVar();
    const aEndJointCode: TSLNode = lineT4.z.toVar();

    const t: TSLNode = aQuadCorner.x.mul(0.5).add(0.5).toVar();

    const mvStart: TSLNode = modelViewMatrix.mul(vec4(aStartPos, 1.0)).toVar();
    const mvEnd: TSLNode = modelViewMatrix.mul(vec4(aEndPos, 1.0)).toVar();

    // TRUE endpoints for the fragment: near-plane clipping below reshapes
    // only the stencil.
    const trueA: TSLNode = vec3(mvStart).toVar();
    const trueB: TSLNode = vec3(mvEnd).toVar();

    const startDepth: TSLNode = mvStart.z.negate().toVar();
    const endDepth: TSLNode = mvEnd.z.negate().toVar();
    const culled: TSLNode = (
      isOrtho
        ? float(0.0).greaterThan(1.0)
        : startDepth.lessThan(nearCull).and(endDepth.lessThan(nearCull))
    ).toVar();

    // Near-plane SEGMENT clipping (stencil only; perspective only). The
    // clip parameters tA/tB also drive the per-end width evaluation below.
    const tA: TSLNode = float(0.0).toVar();
    const tB: TSLNode = float(1.0).toVar();
    const mvStartC: TSLNode = vec4(mvStart).toVar();
    const mvEndC: TSLNode = vec4(mvEnd).toVar();
    if (!isOrtho) {
      If(startDepth.lessThan(nearCull).and(endDepth.greaterThanEqual(nearCull)), () => {
        tA.assign(nearCull.sub(startDepth).div(endDepth.sub(startDepth)));
      }).ElseIf(endDepth.lessThan(nearCull).and(startDepth.greaterThanEqual(nearCull)), () => {
        tB.assign(startDepth.sub(nearCull).div(startDepth.sub(endDepth)));
      });
      mvStartC.assign(mix(mvStart, mvEnd, tA));
      mvEndC.assign(mix(mvStart, mvEnd, tB));
    }

    const w0: TSLNode = sanitizeNonNegative(lineT0.w, float(0.0)).toVar();
    const w1: TSLNode = sanitizeNonNegative(lineT1.w, float(0.0)).toVar();
    const s0: TSLNode = clamp(sanitizeNonNegative(lineT2.w, float(0.5)), 0.0, 1.0).toVar();
    const s1: TSLNode = clamp(sanitizeNonNegative(lineT3.w, float(0.5)), 0.0, 1.0).toVar();

    const seg: TSLNode = trueB.sub(trueA).toVar();
    const segLen: TSLNode = length(seg).toVar();
    const axisW: TSLNode = segLen
      .greaterThan(1e-20)
      .select(seg.div(max(segLen, float(1e-30))).toVar(), vec3(1.0, 0.0, 0.0))
      .toVar();

    // === Bisector cut planes at interior joints ===
    // Partner direction per end, from the joint-code partner slot. All
    // loads run unconditionally (Fn trace order); non-interior codes mask
    // to zero. Slot clamped ≥ 0 so a garbage code still loads in-bounds.
    const partnerDir = (code: TSLNode): TSLNode => {
      const interior: TSLNode = code.greaterThan(0.5).or(code.lessThan(-2.5)).toVar();
      const slotPos: TSLNode = int(code.add(0.5)).sub(int(1)).toVar();
      const slotNeg: TSLNode = int(code.negate().add(0.5)).sub(int(3)).toVar();
      const slotRaw: TSLNode = int(code.greaterThan(0.0).select(slotPos, slotNeg)).toVar();
      const slot: TSLNode = slotRaw.lessThan(int(0)).select(int(0), slotRaw).toVar();
      const pBase: TSLNode = slot.mul(int(6)).toVar();
      const pX: TSLNode = pBase.mod(lineTexW).toVar();
      const pY: TSLNode = pBase.div(lineTexW).toVar();
      const pStart: TSLNode = vec3(uLineTex.load(ivec2(pX, pY))).toVar();
      const pEnd: TSLNode = vec3(uLineTex.load(ivec2(pX.add(int(1)), pY))).toVar();
      const qRaw: TSLNode = code
        .greaterThan(0.0)
        .select(pEnd.sub(pStart).toVar(), pStart.sub(pEnd).toVar())
        .toVar();
      const qLen: TSLNode = length(qRaw).toVar();
      // Normalized in CAMERA space (see the GLSL twin: model scaling would
      // otherwise skew the bisector normal and tanHalf).
      const qCamRaw: TSLNode = vec3(modelViewMatrix.mul(vec4(qRaw, 0.0))).toVar();
      const qCamLen: TSLNode = length(qCamRaw).toVar();
      const qCam: TSLNode = qCamRaw.div(max(qCamLen, float(1e-30))).toVar();
      const valid: TSLNode = interior
        .and(qLen.greaterThan(1e-20))
        .and(qCamLen.greaterThan(1e-20))
        .toVar();
      return vec4(qCam, valid.select(float(1.0), float(0.0))).toVar();
    };
    const qA: TSLNode = partnerDir(aStartJointCode);
    const qB: TSLNode = partnerDir(aEndJointCode);

    const cutA: TSLNode = vec4(0.0, 0.0, 0.0, 0.0).toVar();
    const cutB: TSLNode = vec4(0.0, 0.0, 0.0, 0.0).toVar();
    const tanHalfA: TSLNode = float(0.0).toVar();
    const tanHalfB: TSLNode = float(0.0).toVar();
    If(qA.w.greaterThan(0.5).and(segLen.greaterThan(1e-20)), () => {
      const nRaw: TSLNode = vec3(qA).sub(axisW).toVar(); // m = +w at end A
      const nLen: TSLNode = length(nRaw).toVar();
      const d: TSLNode = dot(axisW, vec3(qA)).toVar();
      If(nLen.greaterThan(1e-6), () => {
        cutA.assign(vec4(nRaw.div(nLen), 1.0));
        tanHalfA.assign(sqrt(max(d.add(1.0), 0.0).div(max(float(1.0).sub(d), 1e-6))));
      });
    });
    If(qB.w.greaterThan(0.5).and(segLen.greaterThan(1e-20)), () => {
      const nRaw: TSLNode = vec3(qB).add(axisW).toVar(); // m = -w at end B
      const nLen: TSLNode = length(nRaw).toVar();
      const d: TSLNode = dot(axisW.negate(), vec3(qB)).toVar();
      If(nLen.greaterThan(1e-6), () => {
        cutB.assign(vec4(nRaw.div(nLen), 1.0));
        tanHalfB.assign(sqrt(max(d.add(1.0), 0.0).div(max(float(1.0).sub(d), 1e-6))));
      });
    });

    // === Stencil (projected stadium with per-end axial extension) ===
    const clipStart: TSLNode = cameraProjectionMatrix.mul(mvStartC).toVar();
    const clipEnd: TSLNode = cameraProjectionMatrix.mul(mvEndC).toVar();
    const clipPosBase: TSLNode = mix(clipStart, clipEnd, t).toVar();

    const wGuard: TSLNode = (isOrtho ? float(1.0) : nearCull).toVar();
    const wStart: TSLNode = max(clipStart.w, wGuard).toVar();
    const wEnd: TSLNode = max(clipEnd.w, wGuard).toVar();
    const ndcStart: TSLNode = vec2(clipStart).div(wStart).toVar();
    const ndcEnd: TSLNode = vec2(clipEnd).div(wEnd).toVar();

    const pixelDir: TSLNode = ndcEnd.sub(ndcStart).mul(uResolution.mul(0.5)).toVar();
    const pixelLen: TSLNode = length(pixelDir).toVar();
    // Degenerate projected direction is HARMLESS here (the stencil tends
    // to a square around a disc; the fragment never reads the stencil
    // orientation).
    const lineDir: TSLNode = pixelLen
      .greaterThan(0.0001)
      .select(pixelDir.div(max(pixelLen, float(1e-30))).toVar(), vec2(1.0, 0.0))
      .toVar();
    const perpendicular: TSLNode = vec2(lineDir.y.negate(), lineDir.x).toVar();

    // Per-end drawn half-width in pixels, at the CLIPPED endpoints, with
    // the T² · dilation variance floor folded in.
    const wAtA: TSLNode = mix(w0, w1, tA).toVar();
    const wAtB: TSLNode = mix(w0, w1, tB).toVar();
    const rawA: TSLNode = (
      isOrtho
        ? wAtA.mul(uOrthoLineScale)
        : wAtA.mul(uPerspectiveLineScale).div(max(mvStartC.z.negate(), nearCull))
    ).toVar();
    const rawB: TSLNode = (
      isOrtho
        ? wAtB.mul(uOrthoLineScale)
        : wAtB.mul(uPerspectiveLineScale).div(max(mvEndC.z.negate(), nearCull))
    ).toVar();
    const RA: TSLNode = sqrt(rawA.mul(rawA).add(T_SQ_DILATION)).toVar();
    const RB: TSLNode = sqrt(rawB.mul(rawB).add(T_SQ_DILATION)).toVar();

    // Coverage fade + hard extent clamp (gsplat pattern).
    const maxExtent: TSLNode = max(uMaxLinePixelWidth, float(2.5)).toVar();
    const Rmax: TSLNode = max(RA, RB).toVar();
    const coverageFade: TSLNode = float(1.0)
      .sub(smoothstep(maxExtent.mul(0.5), maxExtent, Rmax))
      .toVar();
    culled.assign(culled.or(coverageFade.lessThan(0.01)));
    const RAc: TSLNode = min(RA, maxExtent).toVar();
    const RBc: TSLNode = min(RB, maxExtent).toVar();

    // Stadium corner: perpendicular half-width plus per-end, PER-CORNER
    // axial extension (soft cap 0.77R; bisector cut min(R·tanθ/2, R) on
    // the OUTER side of the bend only — the fill-explosion reclaim).
    const isEndB: TSLNode = aQuadCorner.x.greaterThan(0.0).toVar();
    const R: TSLNode = isEndB.select(RBc, RAc).toVar();
    const hardEnd: TSLNode = isEndB.select(cutB.w, cutA.w).toVar();
    const tanHalf: TSLNode = isEndB.select(tanHalfB, tanHalfA).toVar();
    const out2: TSLNode = isEndB
      .select(
        vec2(axisW.x, axisW.y).sub(vec2(qB.x, qB.y)).toVar(),
        vec2(axisW.x, axisW.y).add(vec2(qA.x, qA.y)).negate().toVar()
      )
      .toVar();
    const side: TSLNode = dot(out2, perpendicular).toVar();
    const cornerOuter: TSLNode = aQuadCorner.y.mul(side).greaterThan(0.0).toVar();
    const overhang: TSLNode = min(R.mul(tanHalf), R).toVar();
    // Depth-tilt disc reach — see the GLSL twin: the endpoint's 3D disc
    // (radius R ⊥ axis) projects past the endpoint by R·|cos(axis, view)|
    // when the axis tilts into depth; without it, joints of depth-tilted
    // thick polylines rasterise a dark hairline. Zero face-on.
    const endPos: TSLNode = isEndB.select(vec3(mvEndC).toVar(), vec3(mvStartC).toVar()).toVar();
    const viewDir: TSLNode = (
      isOrtho
        ? vec3(0.0, 0.0, -1.0)
        : endPos.mul(inverseSqrt(max(dot(endPos, endPos), float(1e-20))))
    ).toVar();
    const discReach: TSLNode = R.mul(abs(dot(axisW, viewDir))).toVar();
    const axialExtend: TSLNode = hardEnd
      .greaterThan(0.5)
      .select(
        cornerOuter
          .select(overhang, min(overhang.mul(0.25), float(4.0)).toVar())
          .add(discReach)
          .add(1.5)
          .toVar(),
        R.mul(0.77).add(discReach).add(1.5).toVar()
      )
      .toVar();
    const pixelOffset: TSLNode = perpendicular
      .mul(aQuadCorner.y.mul(R))
      .add(lineDir.mul(aQuadCorner.x.mul(axialExtend)))
      .toVar();
    const ndcOffset: TSLNode = pixelOffset.div(uResolution).mul(2.0).toVar();
    const expandedClip: TSLNode = vec4(
      clipPosBase.xy.add(ndcOffset.mul(clipPosBase.w)),
      clipPosBase.z,
      clipPosBase.w
    ).toVar();

    // Sentinel position for culled segments; varyings carry benign values.
    const clipPosOut: TSLNode = vec4(0.0, 0.0, -2.0, 1.0).toVar();
    If(culled.not(), () => {
      clipPosOut.assign(expandedClip);
    });

    // Assign varyings — identical on every vertex.
    vSegA.assign(vec4(trueA, segLen));
    vSegW.assign(vec4(axisW, culled.select(float(0.0), coverageFade)));
    vEnds.assign(vec4(w0, w1, s0, s1));
    vMisc.assign(vec4(sanitizeAlpha(lineT5.z), sanitizeAlpha(lineT5.w), 0.0, 0.0));
    vCutA.assign(cutA);
    vCutB.assign(cutB);
    if (vScalars) vScalars.assign(vec2(lineT5.x, lineT5.y));
    if (vColor0) vColor0.assign(vec3(lineT2));
    if (vColor1) vColor1.assign(vec3(lineT3));

    return clipPosOut;
  });

  const clipPos: TSLNode = vertexBody();

  // ---- Fragment ----

  const colorNode = Fn(() => {
    // Per-fragment view ray in camera space, UNNORMALIZED (the solver
    // folds |dRaw| in analytically). Focal length in pixels =
    // 0.5 · uPerspectiveLineScale, isotropic. screenCoordinate is
    // top-left-origin on both backends: un-flip with screenSize.
    const fragCoordBL: TSLNode = vec2(
      screenCoordinate.x,
      screenSize.y.sub(screenCoordinate.y)
    ).toVar();
    const rayO: TSLNode = (
      isOrtho
        ? vec3(fragCoordBL.sub(uResolution.mul(0.5)).mul(float(2.0).div(uOrthoLineScale)), 0.0)
        : vec3(0.0, 0.0, 0.0)
    ).toVar();
    const dRaw: TSLNode = (
      isOrtho
        ? vec3(0.0, 0.0, -1.0)
        : vec3(
            fragCoordBL.sub(uResolution.mul(0.5)).mul(float(2.0).div(uPerspectiveLineScale)),
            -1.0
          )
    ).toVar();

    // Ray-segment solver, midpoint-relative (see the GLSL twin).
    const L: TSLNode = vSegA.w.toVar();
    const w: TSLNode = vec3(vSegW).toVar();
    const M: TSLNode = vec3(vSegA)
      .add(w.mul(L.mul(0.5)))
      .toVar();
    const b: TSLNode = rayO.sub(M).toVar();
    const n2: TSLNode = dot(dRaw, dRaw).toVar();
    const rn: TSLNode = sqrt(n2).toVar();
    const dw: TSLNode = dot(dRaw, w).toVar();
    const bdr: TSLNode = dot(b, dRaw).toVar();
    const bw: TSLNode = dot(b, w).toVar();
    const A: TSLNode = n2.sub(dw.mul(dw)).toVar();
    const parallel: TSLNode = A.lessThan(n2.mul(LINE_PARALLEL_LANE_THRESHOLD)).toVar();

    const sM: TSLNode = float(0.0).toVar();
    const D2: TSLNode = float(0.0).toVar();
    const tCenter: TSLNode = float(0.0).toVar();
    If(parallel, () => {
      const invN2: TSLNode = float(1.0).div(n2).toVar();
      const r0: TSLNode = b.sub(dRaw.mul(bdr.mul(invN2))).toVar();
      const rw: TSLNode = dot(r0, w).toVar();
      D2.assign(max(dot(r0, r0).sub(rw.mul(rw)), 0.0));
      tCenter.assign(bdr.negate().mul(invN2));
    }).Else(() => {
      const invA: TSLNode = float(1.0).div(A).toVar();
      sM.assign(bw.mul(n2).sub(dw.mul(bdr)).mul(invA));
      tCenter.assign(dw.mul(bw).sub(bdr).mul(invA));
      const pv: TSLNode = b.add(dRaw.mul(tCenter)).sub(w.mul(sM)).toVar();
      D2.assign(dot(pv, pv));
    });
    const camZ: TSLNode = rayO.z.add(dRaw.z.mul(tCenter)).toVar();

    // Attributes at the clamped closest-approach axial coordinate.
    const sHat: TSLNode = clamp(sM.div(max(L, float(1e-20))).add(0.5), 0.0, 1.0).toVar();
    const width: TSLNode = mix(vEnds.x, vEnds.y, sHat).toVar();
    const sharp: TSLNode = mix(vEnds.z, vEnds.w, sHat).toVar();
    const alphaEl: TSLNode = mix(vMisc.x, vMisc.y, sHat).toVar();

    // σ with the screen-variance dilation floor at the hit depth; aaComp
    // is the matching energy compensation.
    const pxSize: TSLNode = (
      isOrtho
        ? float(2.0).div(uOrthoLineScale)
        : max(camZ.negate(), nearCull).mul(2.0).div(uPerspectiveLineScale)
    ).toVar();
    const sigma: TSLNode = width.mul(LINE_SIGMA_PER_WIDTH).toVar();
    const invSE: TSLNode = inverseSqrt(
      sigma.mul(sigma).add(pxSize.mul(pxSize).mul(LINE_STENCIL_DILATION)).add(1e-30)
    ).toVar();
    const aaComp: TSLNode = sigma.mul(invSE).toVar();

    const r2n: TSLNode = D2.mul(invSE).mul(invSE).toVar();

    const I: TSLNode = float(0.0).toVar();
    if (peakGraph) {
      // PEAK family: Gaussian-shoulder capsule (exact for any sharpness β).
      const sC: TSLNode = clamp(sM, L.mul(-0.5), L.mul(0.5)).toVar();
      const qv: TSLNode = M.add(w.mul(sC)).sub(rayO).toVar();
      const proj: TSLNode = dot(qv, dRaw).toVar();
      const dist2: TSLNode = max(dot(qv, qv).sub(proj.mul(proj).div(n2)), 0.0).toVar();
      const qn2: TSLNode = dist2
        .mul(invSE)
        .mul(invSE)
        .mul(1.0 / T_SQ)
        .toVar();
      Discard(qn2.greaterThanEqual(1.0));
      const beta: TSLNode = float(2.0).pow(sharp.mul(6.0).sub(2.0)).toVar();
      const qn: TSLNode = sqrt(qn2).toVar();
      I.assign(
        max(exp(qn.pow(beta).mul(-FALLOFF_K)).sub(FALLOFF_FLOOR), 0.0).mul(
          INV_ONE_MINUS_FALLOFF_FLOOR
        )
      );
      camZ.assign(rayO.z.add(dRaw.z.mul(proj.div(n2))));
    } else {
      // SUM family: normalized closed-form ray integral, four lanes —
      // quadrature-validated in _shared/line-volumetric.ts; keep the three
      // implementations (reference / GLSL / TSL) in lockstep.
      Discard(r2n.greaterThanEqual(T_SQ));
      const radial: TSLNode = max(exp(r2n.mul(-0.5)).sub(FALLOFF_FLOOR), 0.0)
        .mul(INV_ONE_MINUS_FALLOFF_FLOOR)
        .toVar();
      const hardA: TSLNode = vCutA.w.greaterThan(0.5).toVar();
      const hardB: TSLNode = vCutB.w.greaterThan(0.5).toVar();
      const sAtCenter: TSLNode = sM.add(L.mul(0.5)).toVar();
      const Bp: TSLNode = vec3(vSegA).add(w.mul(L)).toVar();
      const F: TSLNode = float(0.0).toVar();
      // Mirrors the GLSL twin's LANE discards exactly (set in each branch,
      // one Discard after the chain — TSL Discard is safest at statement
      // level). Not just a perf nicety: under a no-blend target (the
      // parity harness) a discarded fragment PRESERVES the pixel while a
      // written zero stomps it, so a fragment beyond a bisector cut must
      // discard or it erases the partner segment's coverage.
      const killed: TSLNode = float(0.0).toVar();

      If(parallel, () => {
        // STRUCTURAL PARALLEL: axial closed forms (constant radial factor).
        const c: TSLNode = invSE.mul(INV_SQRT2).toVar();
        const sLo: TSLNode = float(-1e30).toVar();
        const sHi: TSLNode = float(1e30).toVar();
        const dead: TSLNode = float(0.0).toVar();
        const applyPlane = (normal: TSLNode, through: TSLNode, present: TSLNode) => {
          If(present, () => {
            const dn: TSLNode = dot(dRaw, normal).toVar();
            const sn: TSLNode = dot(through.sub(rayO), normal).toVar();
            If(abs(dn).lessThanEqual(rn.mul(1e-7)), () => {
              If(sn.lessThan(0.0), () => {
                dead.assign(1.0);
              });
            }).Else(() => {
              const sX: TSLNode = sAtCenter.add(sn.div(dn).sub(tCenter).mul(dw)).toVar();
              // (dn>0)==(dw>0) ⇔ dn·dw > 0 for the nonzero values that
              // reach this branch (the |dn| guard above excludes dn≈0,
              // and dw = ±rn in the parallel lane).
              If(dn.mul(dw).greaterThan(0.0), () => {
                sHi.assign(min(sHi, sX));
              }).Else(() => {
                sLo.assign(max(sLo, sX));
              });
            });
          });
        };
        applyPlane(vec3(vCutA), vec3(vSegA), hardA);
        applyPlane(vec3(vCutB), Bp, hardB);
        if (!isOrtho) {
          // Near-plane ray-domain clip: one more s-bound (rayO.z = 0 and
          // dRaw.z = −1 put the crossing at exactly t = nearCull; |dw| is
          // ~rn in this lane, so the mapping through s(t) is well-posed).
          const sX: TSLNode = sAtCenter.add(nearCull.sub(tCenter).mul(dw)).toVar();
          If(dw.greaterThan(0.0), () => {
            sLo.assign(max(sLo, sX));
          }).Else(() => {
            sHi.assign(min(sHi, sX));
          });
        }

        const Glen: TSLNode = float(0.0).toVar();
        If(dead.lessThan(0.5).and(sHi.greaterThan(sLo)), () => {
          If(hardA.not().and(hardB.not()), () => {
            If(sLo.greaterThan(-1e29).or(sHi.lessThan(1e29)), () => {
              // Near-bound clips the exact-L window: H(x) = ∫ₓ^∞ W ds =
              // (Ψ((x−L)c) − Ψ(x·c))/c, bounds clamped into the window's
              // support so the Ψ difference stays well-conditioned.
              const lo: TSLNode = max(sLo, float(-7.0).div(c)).toVar();
              const hi: TSLNode = min(sHi, L.add(float(7.0).div(c))).toVar();
              Glen.assign(
                erfCapRemainderTSL(lo.sub(L).mul(c))
                  .sub(erfCapRemainderTSL(lo.mul(c)))
                  .sub(erfCapRemainderTSL(hi.sub(L).mul(c)))
                  .add(erfCapRemainderTSL(hi.mul(c)))
                  .div(c)
              );
            }).Else(() => {
              Glen.assign(L); // the soft window integrates to exactly L
            });
          })
            .ElseIf(hardA.and(hardB), () => {
              Glen.assign(max(sHi.sub(sLo), 0.0));
            })
            .ElseIf(hardA, () => {
              // Soft cap at B: ∫ ½(1−erf((s−L)c)) over [sLo, sHi].
              Glen.assign(
                erfCapRemainderTSL(sLo.sub(L).mul(c))
                  .sub(erfCapRemainderTSL(sHi.sub(L).mul(c)))
                  .div(c)
              );
            })
            .Else(() => {
              // Soft cap at A: mirror.
              Glen.assign(
                erfCapRemainderTSL(sHi.negate().mul(c))
                  .sub(erfCapRemainderTSL(sLo.negate().mul(c)))
                  .div(c)
              );
            });
        });
        If(Glen.lessThanEqual(0.0), () => {
          killed.assign(1.0); // GLSL twin: `if (Glen <= 0.0) discard;`
        });
        F.assign(max(Glen, 0.0).mul(invSE).mul(INV_SQRT_2PI));
      })
        .ElseIf(hardA.or(hardB), () => {
          // GENERAL PLANE LANE: hard/hard exact; mixed via the
          // sign-selected inclusion–exclusion (A&S erf — pref amplifies
          // erf error unboundedly near-axial; population is only the two
          // chain-end segments per polyline).
          const xiLo: TSLNode = float(-4.0).toVar();
          const xiHi: TSLNode = float(4.0).toVar();
          const kxi: TSLNode = sqrt(A).mul(invSE).mul(INV_SQRT2).toVar();
          const dead: TSLNode = float(0.0).toVar();
          const dnHard: TSLNode = float(0.0).toVar();
          const applyPlane = (normal: TSLNode, through: TSLNode, present: TSLNode) => {
            If(present, () => {
              const dn: TSLNode = dot(dRaw, normal).toVar();
              const sn: TSLNode = dot(through.sub(rayO), normal).toVar();
              dnHard.assign(dn); // last writer wins — single plane in mixed
              If(abs(dn).greaterThan(rn.mul(1e-7)), () => {
                const xi: TSLNode = clamp(sn.div(dn).sub(tCenter).mul(kxi), -4.0, 4.0).toVar();
                If(dn.greaterThan(0.0), () => {
                  xiHi.assign(min(xiHi, xi));
                }).Else(() => {
                  xiLo.assign(max(xiLo, xi));
                });
              }).ElseIf(sn.lessThan(0.0), () => {
                dead.assign(1.0);
              });
            });
          };
          applyPlane(vec3(vCutA), vec3(vSegA), hardA);
          applyPlane(vec3(vCutB), Bp, hardB);
          if (!isOrtho) {
            // Near-plane ray-domain clip: ξ increases with t (kxi > 0), so
            // material at t > nearCull tightens the LOWER bracket edge.
            xiLo.assign(max(xiLo, clamp(nearCull.sub(tCenter).mul(kxi), -4.0, 4.0)));
          }

          If(dead.greaterThan(0.5).or(xiLo.greaterThanEqual(xiHi)), () => {
            killed.assign(1.0); // GLSL twin: `if (dead || xiLo >= xiHi) discard;`
          });
          If(dead.lessThan(0.5).and(xiLo.lessThan(xiHi)), () => {
            const xim: TSLNode = xiHi.add(xiLo).mul(0.5).toVar();
            const dxi: TSLNode = xiHi.sub(xiLo).toVar();
            const pref: TSLNode = rn
              .mul(0.5)
              .mul(inverseSqrt(max(A, n2.mul(1e-12))))
              .toVar();
            const narrow: TSLNode = dxi.lessThan(0.5).toVar();
            const xim2: TSLNode = min(xim.mul(xim), 80.0).toVar();
            const bracketTaylor: TSLNode = exp(xim2.negate())
              .mul(TWO_OVER_SQRT_PI)
              .mul(
                dxi
                  .mul(dxi)
                  .mul(xim2.mul(4.0).sub(2.0))
                  .mul(1.0 / 24.0)
                  .add(1.0)
              )
              .mul(dxi)
              .toVar();
            If(hardA.and(hardB), () => {
              const bracket: TSLNode = narrow
                .select(bracketTaylor, erfPolyTSL(xiHi).sub(erfPolyTSL(xiLo)).toVar())
                .toVar();
              F.assign(pref.mul(max(bracket, 0.0)));
            }).Else(() => {
              // MIXED — see the GLSL twin / CPU reference for the split
              // selection rationale (J0 saturated-cap shortcuts + the
              // sign-selected J1/J2 inclusion–exclusion).
              const kk: TSLNode = kxi.div(rn).toVar();
              const axialDominant: TSLNode = dw.mul(dw).greaterThan(A).toVar();
              const excludedTowardMinusS: TSLNode = dnHard.mul(dw).lessThan(0.0).toVar();
              const complementOnExcludedSide: TSLNode = hardA
                .select(excludedTowardMinusS.not(), excludedTowardMinusS)
                .toVar();
              const xCapB: TSLNode = sAtCenter.sub(L).mul(kk).toVar();
              const xCapA: TSLNode = sAtCenter.mul(kk).toVar();
              const bracketAS: TSLNode = narrow
                .select(bracketTaylor, erfAsTSL(xiHi).sub(erfAsTSL(xiLo)).toVar())
                .toVar();
              // Saturated-cap shortcuts (the J0 split) — computed only
              // under axialDominant, where |dw| > √A keeps the division
              // well-posed (a perpendicular-dominant dw ≈ 0 would feed the
              // comparisons backend-dependent garbage).
              const capSaturated: TSLNode = float(0.0).toVar();
              const capDead: TSLNode = float(0.0).toVar();
              If(axialDominant, () => {
                const sEdge: TSLNode = hardA.select(L, float(0.0)).toVar();
                const xiEdge: TSLNode = sEdge.sub(sAtCenter).mul(kxi).div(dw).toVar();
                const halfRamp: TSLNode = sqrt(A).mul(3.0).div(abs(dw)).toVar();
                // GLSL twin: `(!hardA) == (dw > 0.0)` (bool equality
                // spelled with and/or — TSL-safe).
                const dwPos: TSLNode = dw.greaterThan(0.0).toVar();
                const satHigh: TSLNode = hardA.not().and(dwPos).or(hardA.and(dwPos.not())).toVar();
                If(
                  satHigh.select(
                    xiEdge.add(halfRamp).lessThanEqual(xiLo),
                    xiEdge.sub(halfRamp).greaterThanEqual(xiHi)
                  ),
                  () => {
                    capSaturated.assign(1.0);
                  }
                );
                If(
                  satHigh.select(
                    xiEdge.sub(halfRamp).greaterThanEqual(xiHi),
                    xiEdge.add(halfRamp).lessThanEqual(xiLo)
                  ),
                  () => {
                    capDead.assign(1.0);
                  }
                );
              });
              If(capSaturated.greaterThan(0.5), () => {
                F.assign(pref.mul(max(bracketAS, 0.0)));
              })
                .ElseIf(capDead.greaterThan(0.5), () => {
                  killed.assign(1.0); // GLSL twin: capDead discard
                })
                .ElseIf(axialDominant.and(complementOnExcludedSide), () => {
                  // Cap-only split, min()ed with the bracket: both are
                  // upper bounds of the exact clipped integral.
                  const capOnly: TSLNode = hardA
                    .select(
                      float(1.0).sub(erfAsTSL(xCapB)).toVar(),
                      float(1.0).add(erfAsTSL(xCapA)).toVar()
                    )
                    .toVar();
                  F.assign(pref.mul(max(min(capOnly, bracketAS), 0.0)));
                })
                .Else(() => {
                  const capTerm: TSLNode = hardA
                    .select(
                      float(1.0).add(erfAsTSL(xCapB)).toVar(),
                      float(1.0).sub(erfAsTSL(xCapA)).toVar()
                    )
                    .toVar();
                  F.assign(pref.mul(max(bracketAS.sub(capTerm), 0.0)));
                });
            });
          });
        })
        .Else(() => {
          // SOFT/SOFT: the erf-cap closed form.
          const kk: TSLNode = sqrt(A).mul(inverseSqrt(n2)).mul(invSE).mul(INV_SQRT2).toVar();
          const xm: TSLNode = sM.mul(kk).toVar();
          const dx: TSLNode = L.mul(kk).toVar();
          const x0: TSLNode = xm.sub(dx.mul(0.5)).toVar();
          const x1: TSLNode = xm.add(dx.mul(0.5)).toVar();
          If(x0.greaterThan(3.0).or(x1.lessThan(-3.0)), () => {
            killed.assign(1.0); // GLSL twin: axial-reject discard
          });
          If(x0.lessThanEqual(3.0).and(x1.greaterThanEqual(-3.0)), () => {
            const xm2: TSLNode = min(xm.mul(xm), 80.0).toVar();
            const taylor: TSLNode = exp(xm2.negate())
              .mul(TWO_OVER_SQRT_PI)
              .mul(
                dx
                  .mul(dx)
                  .mul(xm2.mul(4.0).sub(2.0))
                  .mul(1.0 / 24.0)
                  .add(1.0)
              )
              .toVar();
            const E: TSLNode = dx
              .lessThan(0.5)
              .select(
                taylor,
                erfPolyTSL(x1)
                  .sub(erfPolyTSL(x0))
                  .div(max(dx, float(1e-30)))
                  .toVar()
              )
              .toVar();
            F.assign(max(E, 0.0).mul(L).mul(invSE).mul(QUARTER_SQRT2));
          });
        });
      Discard(killed.greaterThan(0.5));
      I.assign(radial.mul(max(F, 0.0)));
    }

    const nearFade: TSLNode = isOrtho
      ? float(1.0)
      : perspectiveNearFadeStaticTSL(false, camZ, nearCull);
    const intensity: TSLNode = I.mul(aaComp).mul(vSegW.w).mul(nearFade).toVar();

    // Color at the clamped closest-approach coordinate.
    let colorExpr: TSLNode;
    if (config.useColormap && uColormapTex && uScalarMin && uScalarScale && vScalars) {
      const s: TSLNode = mix(vScalars.x, vScalars.y, sHat).toVar();
      const stRaw: TSLNode = clamp(s.sub(uScalarMin).mul(uScalarScale), 0.0, 1.0).toVar();
      const st: TSLNode = config.gammaOne ? stRaw : stRaw.pow(uInvGamma).toVar();
      colorExpr = uColormapTex.sample(vec2(st, 0.5)).rgb.toVar();
    } else {
      colorExpr = mix(vColor0!, vColor1!, sHat).toVar();
    }
    const adjusted: TSLNode = config.noGOG
      ? colorExpr
      : max(colorExpr.mul(uIntensity).add(uOffset), vec3(0.0)).toVar();
    const maxAdjusted: TSLNode = max(adjusted.r, max(adjusted.g, adjusted.b)).toVar();

    const gammaColor: TSLNode =
      config.useColormap || config.gammaOne ? adjusted : adjusted.pow(vec3(uInvGamma));

    if (volumetricGraph) {
      const alpha: TSLNode = intensity
        .mul(uOpacity)
        .mul(
          mix(
            float(1.0),
            min(alphaEl, float(ALPHA_CLAMP)).oneMinus().log().negate(),
            uHasElementAlpha
          )
        )
        .toVar();
      const tau: TSLNode = uAbsorption.mul(alpha).toVar();
      Discard(maxAdjusted.lessThan(1e-4).and(tau.lessThan(1e-4)));
      const volAlpha: TSLNode = float(1.0).sub(exp(tau.negate())).toVar();
      const series: TSLNode = float(1.0)
        .sub(tau.mul(VOLUMETRIC_SERIES_C1))
        .add(tau.mul(tau).div(VOLUMETRIC_SERIES_C2_DIVISOR))
        .toVar();
      const screen: TSLNode = tau
        .lessThan(VOLUMETRIC_SERIES_TAU_THRESHOLD)
        .select(series, volAlpha.div(max(tau, VOLUMETRIC_TAU_EPS)).toVar())
        .toVar();
      return vec4(gammaColor.mul(alpha).mul(screen), volAlpha);
    }
    const intensityScaled: TSLNode = intensity.mul(alphaEl).toVar();
    Discard(maxAdjusted.lessThan(1e-4));
    if (premultiplyRGB) {
      const a: TSLNode = intensityScaled.mul(uOpacity).toVar();
      return vec4(gammaColor.mul(a), a);
    }
    return vec4(gammaColor, intensityScaled.mul(uOpacity));
  });

  const material = outMaterial ?? new NodeMaterial();
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.toneMapped = false;

  // Blending state from the shared helper — factory tail is the only
  // state writer at TSL construction, same as the screen-space factory.
  const opacityValue = (nodes.uOpacity.value as number | undefined) ?? 1.0;
  const blendingState = getCompleteBlendingState(blendingMode, opacityValue);
  applyBlendingStateToMaterial(material, blendingState);
  return material;
}
