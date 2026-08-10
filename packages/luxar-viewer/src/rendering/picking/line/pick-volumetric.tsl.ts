/**
 * Volumetric line picking TSL factory — NodeMaterial counterpart to
 * `VOLUMETRIC_LINE_PICK_SOURCE` in `shaders-volumetric.ts` (issue #1352,
 * behind `?linePrimitive=volumetric`).
 *
 * The vertex stage is the visual volumetric factory's stadium stencil
 * (`materials/line/shader-tsl-volumetric.ts`, kept in lockstep) with
 * colors / scalars / element alpha stripped and the pick IDs added; the
 * fragment stage is the visual PEAK capsule lane, used UNCONDITIONALLY —
 * picking wants the hotspot on the centerline regardless of the visual
 * blending mode, and the peak formulation is exact for any sharpness β
 * with none of the sum lanes' integral machinery. See the GLSL twin's
 * module header for the full rationale.
 *
 * Output contract (screen-space pick parity):
 *   `vec4(nodeId, elementId-low16, brightness, elementId-high16)`,
 *   depth = 1 − brightness (brightness-as-depth tie-breaking).
 *
 * @module rendering/picking/line/pick-volumetric.tsl
 */

import * as THREE from 'three';
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
  exp,
  sqrt,
  abs,
  dot,
  smoothstep,
  inverseSqrt,
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
} from '../../materials/_shared/falloff';
import {
  LINE_PARALLEL_LANE_THRESHOLD,
  LINE_SIGMA_PER_WIDTH,
  LINE_STENCIL_DILATION,
} from '../../materials/_shared/line-volumetric';
import {
  perspectiveNearFadeStaticTSL,
  sanitizeNonNegative,
  sortedIndexNode,
  type TSLNode,
} from '../../materials/_shared/tsl-helpers';
import type { LinePickTSLNodes } from './pick.tsl';

const T = GAUSSIAN_EQUIVALENT_TRUNCATION;
const T_SQ = T * T;
const T_SQ_DILATION = T * T * LINE_STENCIL_DILATION;

/**
 * Per-build configuration. The volumetric pick pass has no join-style
 * variant — interior joints are bisector cuts derived from the joint
 * codes, exactly as in the visual volumetric factory.
 */
export interface VolumetricLinePickTSLConfig {
  /** Camera projection mode at build time — mirrors `LinePickTSLConfig`. */
  readonly isOrtho?: boolean;
}

export function volumetricLinePickWebGPUFactory(
  nodes: LinePickTSLNodes,
  config: VolumetricLinePickTSLConfig = {},
  outMaterial?: NodeMaterial
): NodeMaterial {
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  const aSortedIndex: TSLNode = sortedIndexNode(nodes.uSortedIndexSlot);

  const uLineTex = nodes.uLineTex;
  const uResolution = nodes.uResolution;
  const uNodeId = nodes.uNodeId;
  const uNearCull = nodes.uNearCull;
  const uMaxLinePixelWidth = nodes.uMaxLinePixelWidth;
  const uPerspectiveLineScale = nodes.uPerspectiveLineScale;
  const uOrthoLineScale = nodes.uOrthoLineScale;

  const isOrtho = config.isOrtho === true;

  // ---- Varyings (all per-segment constants → flat) ----
  const vSegA: TSLNode = varying(vec4(0.0, 0.0, 0.0, 0.0)).setInterpolation('flat');
  const vSegW: TSLNode = varying(vec4(1.0, 0.0, 0.0, 0.0)).setInterpolation('flat');
  const vEnds: TSLNode = varying(vec4(0.0, 0.0, 0.5, 0.5)).setInterpolation('flat');
  const vCutA: TSLNode = varying(vec4(0.0, 0.0, 0.0, 0.0)).setInterpolation('flat');
  const vCutB: TSLNode = varying(vec4(0.0, 0.0, 0.0, 0.0)).setInterpolation('flat');
  const vNodeId: TSLNode = varying(uNodeId).setInterpolation('flat');
  // Storage index split into two 16-bit halves — see the screen-space
  // pick factory (`pick.tsl.ts`) for the float32-mantissa rationale.
  const elementIdInt: TSLNode = int(aSortedIndex);
  const elementIdHi: TSLNode = elementIdInt.div(int(65536));
  const elementIdLo: TSLNode = elementIdInt.sub(elementIdHi.mul(int(65536)));
  const vElementId: TSLNode = varying(
    vec2(float(elementIdLo), float(elementIdHi))
  ).setInterpolation('flat');

  const nearCull: TSLNode = max(uNearCull, float(1e-20));

  const vertexBody = Fn(() => {
    // === Line-texture fetch (6 texels/segment, one row) — visual
    // volumetric factory parity; colors / scalars / alpha not read. ===
    const lineBase: TSLNode = int(aSortedIndex).mul(int(6)).toVar();
    // int() wrap is LOAD-BEARING (WebGL2 textureSize returns int).
    const lineTexW: TSLNode = int((textureSize(uLineTex, int(0)) as unknown as TSLNode).x).toVar();
    const texelX: TSLNode = lineBase.mod(lineTexW).toVar();
    const texelY: TSLNode = lineBase.div(lineTexW).toVar();
    const lineT0: TSLNode = uLineTex.load(ivec2(texelX, texelY)).toVar();
    const lineT1: TSLNode = uLineTex.load(ivec2(texelX.add(int(1)), texelY)).toVar();
    const lineT2: TSLNode = uLineTex.load(ivec2(texelX.add(int(2)), texelY)).toVar();
    const lineT3: TSLNode = uLineTex.load(ivec2(texelX.add(int(3)), texelY)).toVar();
    const lineT4: TSLNode = uLineTex.load(ivec2(texelX.add(int(4)), texelY)).toVar();
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

    // Near-plane SEGMENT clipping (stencil only; perspective only).
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

    // === Bisector cut planes at interior joints — visual volumetric
    // factory parity (all loads unconditional; camera-space
    // normalization is load-bearing under model scale). ===
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
    const lineDir: TSLNode = pixelLen
      .greaterThan(0.0001)
      .select(pixelDir.div(max(pixelLen, float(1e-30))).toVar(), vec2(1.0, 0.0))
      .toVar();
    const perpendicular: TSLNode = vec2(lineDir.y.negate(), lineDir.x).toVar();

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

    // Coverage fade + hard extent clamp (visual parity: a segment the
    // visual pass faded out must not stay pickable at full strength).
    const maxExtent: TSLNode = max(uMaxLinePixelWidth, float(2.5)).toVar();
    const Rmax: TSLNode = max(RA, RB).toVar();
    const coverageFade: TSLNode = float(1.0)
      .sub(smoothstep(maxExtent.mul(0.5), maxExtent, Rmax))
      .toVar();
    culled.assign(culled.or(coverageFade.lessThan(0.01)));
    const RAc: TSLNode = min(RA, maxExtent).toVar();
    const RBc: TSLNode = min(RB, maxExtent).toVar();

    // Stadium corner — visual volumetric factory, verbatim.
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
    vCutA.assign(cutA);
    vCutB.assign(cutB);

    return clipPosOut;
  });

  const clipPos: TSLNode = vertexBody();

  // ---- Fragment: PEAK capsule brightness + brightness-as-depth ----

  // Computed once, shared by colorNode and depthNode (same `.once()`
  // pattern as the screen-space pick factory).
  const brightnessShared = Fn(() => {
    // Per-fragment view ray, UNNORMALIZED (visual volumetric fragment,
    // verbatim). screenCoordinate is top-left-origin on both backends:
    // un-flip with screenSize.
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

    // The peak path needs no parallel-specific distance math
    // (point-to-rod distance is s-independent for parallel geometry;
    // a clamped garbage sM still yields the right distance) — only a
    // finite sM for the attribute lookup.
    const sM: TSLNode = float(0.0).toVar();
    const camZ: TSLNode = float(0.0).toVar();
    If(parallel, () => {
      camZ.assign(rayO.z.add(dRaw.z.mul(bdr.negate().div(n2))));
    }).Else(() => {
      const invA: TSLNode = float(1.0).div(A).toVar();
      sM.assign(bw.mul(n2).sub(dw.mul(bdr)).mul(invA));
      camZ.assign(rayO.z.add(dRaw.z.mul(dw.mul(bw).sub(bdr).mul(invA))));
    });

    // Width / sharpness at the clamped closest-approach axial coord.
    const sHat: TSLNode = clamp(sM.div(max(L, float(1e-20))).add(0.5), 0.0, 1.0).toVar();
    const width: TSLNode = mix(vEnds.x, vEnds.y, sHat).toVar();
    const sharp: TSLNode = mix(vEnds.z, vEnds.w, sHat).toVar();

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

    // PEAK capsule — the visual factory's peak lane, verbatim.
    const hardA: TSLNode = vCutA.w.greaterThan(0.5).toVar();
    const hardB: TSLNode = vCutB.w.greaterThan(0.5).toVar();
    const BpPeak: TSLNode = vec3(vSegA).add(w.mul(L)).toVar();
    const tLo: TSLNode = float(-1e30).toVar();
    const tHi: TSLNode = float(1e30).toVar();
    const deadPeak: TSLNode = float(0.0).toVar();
    const applyCut = (normal: TSLNode, through: TSLNode, present: TSLNode) => {
      If(present, () => {
        const dn: TSLNode = dot(dRaw, normal).toVar();
        const sn: TSLNode = dot(through.sub(rayO), normal).toVar();
        If(abs(dn).lessThanEqual(rn.mul(1e-7)), () => {
          If(sn.lessThan(0.0), () => {
            deadPeak.assign(1.0);
          });
        }).Else(() => {
          const tX: TSLNode = sn.div(dn).toVar();
          If(dn.greaterThan(0.0), () => {
            tHi.assign(min(tHi, tX));
          }).Else(() => {
            tLo.assign(max(tLo, tX));
          });
        });
      });
    };
    applyCut(vec3(vCutA), vec3(vSegA), hardA);
    applyCut(vec3(vCutB), BpPeak, hardB);
    Discard(deadPeak.greaterThan(0.5).or(tHi.lessThan(tLo)));
    const sLoC: TSLNode = hardA.select(float(-1e30), L.mul(-0.5)).toVar();
    const sHiC: TSLNode = hardB.select(float(1e30), L.mul(0.5)).toVar();
    const sC: TSLNode = clamp(sM, sLoC, sHiC).toVar();
    const qv: TSLNode = M.add(w.mul(sC)).sub(rayO).toVar();
    const tHit: TSLNode = clamp(dot(qv, dRaw).div(n2), tLo, tHi).toVar();
    const pRay: TSLNode = rayO.add(dRaw.mul(tHit)).toVar();
    const dv: TSLNode = pRay.sub(M.add(w.mul(clamp(dot(pRay.sub(M), w), sLoC, sHiC)))).toVar();
    const dist2: TSLNode = dot(dv, dv).toVar();
    const qn2: TSLNode = dist2
      .mul(invSE)
      .mul(invSE)
      .mul(1.0 / T_SQ)
      .toVar();
    Discard(qn2.greaterThanEqual(1.0));
    const beta: TSLNode = float(2.0).pow(sharp.mul(6.0).sub(2.0)).toVar();
    const qn: TSLNode = sqrt(qn2).toVar();
    const I: TSLNode = max(exp(qn.pow(beta).mul(-FALLOFF_K)).sub(FALLOFF_FLOOR), 0.0)
      .mul(INV_ONE_MINUS_FALLOFF_FLOOR)
      .toVar();
    const hitZ: TSLNode = rayO.z.add(dRaw.z.mul(tHit)).toVar();

    const nearFade: TSLNode = isOrtho
      ? float(1.0)
      : perspectiveNearFadeStaticTSL(false, hitZ, nearCull);
    return I.mul(aaComp).mul(vSegW.w).mul(nearFade);
  }).once();
  const brightness: TSLNode = brightnessShared().toVar('lineVolPickBrightness');

  const colorNode = Fn(() => {
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
  // Pin opacity to exactly 1 — the NodeMaterial fragment tail multiplies
  // alpha by material.opacity, and alpha carries the element id's HIGH
  // half (see the screen-space pick factory's note).
  material.opacity = 1;
  material.blending = THREE.NoBlending;
  return material;
}
