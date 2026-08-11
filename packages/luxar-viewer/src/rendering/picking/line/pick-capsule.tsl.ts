/**
 * Capsule line PICKING — TSL / NodeMaterial twin of `shaders-capsule.ts`
 * (issue #1352, behind `?linePrimitive=capsule`).
 *
 * Same stencil, half-disc joints and quartic profile as the visual
 * capsule TSL factory (`materials/line/shader-tsl-capsule.ts`); the pick
 * output contract is the shared one:
 *   `vec4(nodeId, elementId-low16, brightness, elementId-high16)`,
 * with brightness-as-depth via `material.depthNode`. Per-element alpha and
 * node opacity are ignored, matching the other pick variants.
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
  exp2,
  pow,
  dot,
  abs,
  textureSize,
  modelViewMatrix,
  cameraProjectionMatrix,
  Discard,
} from 'three/tsl';
import * as THREE from 'three';
import { NodeMaterial } from 'three/webgpu';
import {
  CAPSULE_JOINT_DEFICIT_GATE,
  CAPSULE_MIN_RADIUS_PX,
  CAPSULE_RADIUS_PER_QUAD_HALFWIDTH,
  CAPSULE_STENCIL_APRON_PX,
} from '../../materials/_shared/line-capsule';
import {
  perspectiveNearFadeStaticTSL,
  sanitizeNonNegative,
  tslLineJointCapSuppression,
  type TSLNode,
  sortedIndexNode,
} from '../../materials/_shared/tsl-helpers';
import type { LinePickTSLNodes } from './pick.tsl';

/** Build-time configuration (projection mode picks the graph variant). */
export interface CapsuleLinePickTSLConfig {
  isOrtho?: boolean;
}

export function capsuleLinePickWebGPUFactory(
  nodes: LinePickTSLNodes,
  config: CapsuleLinePickTSLConfig = {},
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

  // ---- Varyings ----
  const vLocal: TSLNode = varying(vec2(0.0, 0.0));
  const vMeta: TSLNode = varying(vec3(1.0, 0.0, 0.0)).setInterpolation('flat');
  const vCutA2: TSLNode = varying(vec2(-1.0, 0.0)).setInterpolation('flat');
  const vCutB2: TSLNode = varying(vec2(1.0, 0.0)).setInterpolation('flat');
  // Joint packets for the DEFICIT rule (see the fragment): partner axis
  // direction in my local frame (.xy), projected length px (.z; 0 = no
  // usable partner -> hard cut), far radius px (.w); vREnd = my end radii.
  const vJointA: TSLNode = varying(vec4(0.0, 0.0, 0.0, 0.0)).setInterpolation('flat');
  const vJointB: TSLNode = varying(vec4(0.0, 0.0, 0.0, 0.0)).setInterpolation('flat');
  const vREnd: TSLNode = varying(vec2(1.0, 1.0)).setInterpolation('flat');
  const vR: TSLNode = varying(float(1.0));
  const vW: TSLNode = varying(float(1.0));
  const vFade: TSLNode = varying(float(1.0));
  const vSharp: TSLNode = varying(float(0.5));
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
    const lineBase: TSLNode = int(aSortedIndex).mul(int(6)).toVar();
    const lineTexW: TSLNode = int((textureSize(uLineTex, int(0)) as unknown as TSLNode).x).toVar();
    const texelX: TSLNode = lineBase.mod(lineTexW).toVar();
    const texelY: TSLNode = lineBase.div(lineTexW).toVar();
    const lineT0: TSLNode = uLineTex.load(ivec2(texelX, texelY)).toVar();
    const lineT1: TSLNode = uLineTex.load(ivec2(texelX.add(int(1)), texelY)).toVar();
    const lineT2: TSLNode = uLineTex.load(ivec2(texelX.add(int(2)), texelY)).toVar();
    const lineT3: TSLNode = uLineTex.load(ivec2(texelX.add(int(3)), texelY)).toVar();
    const lineT4: TSLNode = uLineTex.load(ivec2(texelX.add(int(4)), texelY)).toVar();

    const mvStart: TSLNode = modelViewMatrix.mul(vec4(lineT0.xyz, 1.0)).toVar();
    const mvEnd: TSLNode = modelViewMatrix.mul(vec4(lineT1.xyz, 1.0)).toVar();

    const startDepth: TSLNode = mvStart.z.negate().toVar();
    const endDepth: TSLNode = mvEnd.z.negate().toVar();
    const culled: TSLNode = isOrtho
      ? float(0.0).greaterThan(1.0).toVar()
      : startDepth.lessThan(nearCull).and(endDepth.lessThan(nearCull)).toVar();

    const mvA: TSLNode = mvStart.toVar();
    const mvB: TSLNode = mvEnd.toVar();
    // Clip flags survive the block (mirrors the visual capsule exactly).
    const clippedA: TSLNode = float(0.0).toVar();
    const clippedB: TSLNode = float(0.0).toVar();
    const tA: TSLNode = float(0.0).toVar();
    const tB: TSLNode = float(1.0).toVar();
    if (!isOrtho) {
      If(startDepth.lessThan(nearCull).and(endDepth.greaterThanEqual(nearCull)), () => {
        tA.assign(nearCull.sub(startDepth).div(endDepth.sub(startDepth)));
        mvA.assign(mix(mvStart, mvEnd, tA));
        clippedA.assign(1.0);
      }).ElseIf(endDepth.lessThan(nearCull).and(startDepth.greaterThanEqual(nearCull)), () => {
        tB.assign(startDepth.sub(nearCull).div(startDepth.sub(endDepth)));
        mvB.assign(mix(mvStart, mvEnd, tB));
        clippedB.assign(1.0);
      });
    }

    const w0: TSLNode = sanitizeNonNegative(lineT0.w, 0.0);
    const w1: TSLNode = sanitizeNonNegative(lineT1.w, 0.0);
    // Endpoint attributes at the CLIPPED span (mirrors the visual twin).
    const wEffA: TSLNode = mix(w0, w1, tA).toVar();
    const wEffB: TSLNode = mix(w0, w1, tB).toVar();
    const s0: TSLNode = clamp(sanitizeNonNegative(lineT2.w, 0.5), 0.0, 1.0).toVar();
    const s1: TSLNode = clamp(sanitizeNonNegative(lineT3.w, 0.5), 0.0, 1.0).toVar();

    const clipA: TSLNode = cameraProjectionMatrix.mul(mvA).toVar();
    const clipB: TSLNode = cameraProjectionMatrix.mul(mvB).toVar();
    const wA: TSLNode = max(clipA.w, float(1e-6)).toVar();
    const wB: TSLNode = max(clipB.w, float(1e-6)).toVar();
    const pA: TSLNode = clipA.xy.div(wA).mul(0.5).add(0.5).mul(uResolution).toVar();
    const pB: TSLNode = clipB.xy.div(wB).mul(0.5).add(0.5).mul(uResolution).toVar();

    const rawA: TSLNode = (
      isOrtho
        ? wEffA.mul(uOrthoLineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
        : wEffA
            .mul(uPerspectiveLineScale)
            .mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
            .div(max(mvA.z.negate(), nearCull))
    ).toVar();
    const rawB: TSLNode = (
      isOrtho
        ? wEffB.mul(uOrthoLineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
        : wEffB
            .mul(uPerspectiveLineScale)
            .mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
            .div(max(mvB.z.negate(), nearCull))
    ).toVar();
    const rA: TSLNode = clamp(rawA, CAPSULE_MIN_RADIUS_PX, uMaxLinePixelWidth).toVar();
    const rB: TSLNode = clamp(rawB, CAPSULE_MIN_RADIUS_PX, uMaxLinePixelWidth).toVar();

    // Which ends cut rather than cap — the shared joint-code rule, same as
    // the visual twin (a free end and a degree->=3 hub keep the round cap).
    const interiorA: TSLNode = tslLineJointCapSuppression(lineT4.y).toVar();
    const interiorB: TSLNode = tslLineJointCapSuppression(lineT4.z).toVar();

    const ab: TSLNode = pB.sub(pA).toVar();
    const abLen: TSLNode = length(ab).toVar();
    const u: TSLNode = abLen
      .greaterThan(1e-4)
      .select(ab.div(abLen).toVar(), vec2(1.0, 0.0))
      .toVar();
    const v: TSLNode = vec2(u.y.negate(), u.x).toVar();
    const rMax: TSLNode = max(rA, rB).add(CAPSULE_STENCIL_APRON_PX).toVar();

    const cutA: TSLNode = vec2(-1.0, 0.0).toVar();
    const cutB: TSLNode = vec2(1.0, 0.0).toVar();
    const extA: TSLNode = rMax.toVar();
    const extB: TSLNode = rMax.toVar();
    const capA: TSLNode = interiorA.toVar();
    const capB: TSLNode = interiorB.toVar();

    const pFarWidth: TSLNode = float(0.0).toVar();
    const pFarDepth: TSLNode = float(1.0).toVar();
    const partnerFarPx = (code: TSLNode, mvJoint: TSLNode, jointDepth: TSLNode): TSLNode => {
      // Near-clips the partner's far endpoint toward the joint vertex
      // before projecting (mirrors the visual capsule exactly).
      const interior: TSLNode = code.greaterThan(0.5).or(code.lessThan(-2.5)).toVar();
      const slot: TSLNode = code
        .greaterThan(0.0)
        .select(int(code.add(0.5)).sub(int(1)), int(code.negate().add(0.5)).sub(int(3)))
        .toVar();
      const pBase: TSLNode = slot.mul(int(6)).toVar();
      const pt0: TSLNode = ivec2(pBase.mod(lineTexW), pBase.div(lineTexW)).toVar();
      const pStart: TSLNode = uLineTex.load(pt0).toVar();
      const pEnd: TSLNode = uLineTex.load(ivec2(pt0.x.add(int(1)), pt0.y)).toVar();
      const farTexel: TSLNode = code.greaterThan(0.0).select(pEnd, pStart).toVar();
      const farObj: TSLNode = farTexel.xyz.toVar();
      pFarWidth.assign(sanitizeNonNegative(farTexel.w, 0.0));
      const mvFar: TSLNode = modelViewMatrix.mul(vec4(farObj, 1.0)).toVar();
      if (!isOrtho) {
        const farDepth: TSLNode = mvFar.z.negate().toVar();
        If(farDepth.lessThan(nearCull), () => {
          const tF: TSLNode = clamp(
            jointDepth.sub(nearCull).div(max(jointDepth.sub(farDepth), float(1e-20))),
            0.0,
            1.0
          ).toVar();
          mvFar.assign(mix(mvJoint, mvFar, tF));
        });
      }
      pFarDepth.assign(max(mvFar.z.negate(), nearCull));
      const cl: TSLNode = cameraProjectionMatrix.mul(mvFar).toVar();
      const px: TSLNode = cl.xy
        .div(max(cl.w, float(1e-6)))
        .mul(0.5)
        .add(0.5)
        .mul(uResolution)
        .toVar();
      return vec3(px, interior.select(float(1.0), float(0.0))).toVar();
    };

    If(interiorA.greaterThan(0.5), () => {
      extA.assign(CAPSULE_STENCIL_APRON_PX);
      const far: TSLNode = partnerFarPx(lineT4.y, mvStart, startDepth);
      If(clippedA.lessThan(0.5).and(far.z.greaterThan(0.5)), () => {
        const qq: TSLNode = far.xy.sub(pA).toVar();
        const ql: TSLNode = length(qq).toVar();
        If(ql.greaterThan(1e-4), () => {
          const qhat: TSLNode = qq.div(ql).toVar();
          const nRaw: TSLNode = qhat.sub(u).toVar(); // q − m, m = +u at A
          const nl: TSLNode = length(nRaw).toVar();
          If(nl.greaterThan(1e-3), () => {
            const n2: TSLNode = nRaw.div(nl).toVar();
            const nLoc: TSLNode = vec2(dot(n2, u), dot(n2, v)).toVar();
            If(nLoc.x.lessThan(-1e-3), () => {
              cutA.assign(nLoc);
              const rpFarA: TSLNode = clamp(
                isOrtho
                  ? pFarWidth.mul(uOrthoLineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
                  : pFarWidth
                      .mul(uPerspectiveLineScale)
                      .mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
                      .div(pFarDepth),
                CAPSULE_MIN_RADIUS_PX,
                uMaxLinePixelWidth
              ).toVar();
              const deficitA: TSLNode = clamp(
                float(1.0).sub(rpFarA.div(max(rA, float(1e-4)))),
                0.0,
                1.0
              ).toVar();
              // Congruence gate (see _shared/line-capsule.ts).
              If(deficitA.greaterThan(CAPSULE_JOINT_DEFICIT_GATE), () => {
                vJointA.assign(vec4(dot(qhat, u), dot(qhat, v), ql, rpFarA));
              });
              extA.assign(max(abs(nLoc.y), deficitA).mul(rMax).add(CAPSULE_STENCIL_APRON_PX));
            });
          }).Else(() => {
            // Near-hairpin: the bisector is degenerate — plain round cap.
            capA.assign(0.0);
            extA.assign(rMax);
          });
        });
      });
    });
    If(interiorB.greaterThan(0.5), () => {
      extB.assign(CAPSULE_STENCIL_APRON_PX);
      const far: TSLNode = partnerFarPx(lineT4.z, mvEnd, endDepth);
      If(clippedB.lessThan(0.5).and(far.z.greaterThan(0.5)), () => {
        const qq: TSLNode = far.xy.sub(pB).toVar();
        const ql: TSLNode = length(qq).toVar();
        If(ql.greaterThan(1e-4), () => {
          const qhat: TSLNode = qq.div(ql).toVar();
          const nRaw: TSLNode = qhat.add(u).toVar(); // q − m, m = −u at B
          const nl: TSLNode = length(nRaw).toVar();
          If(nl.greaterThan(1e-3), () => {
            const n2: TSLNode = nRaw.div(nl).toVar();
            const nLoc: TSLNode = vec2(dot(n2, u), dot(n2, v)).toVar();
            If(nLoc.x.greaterThan(1e-3), () => {
              cutB.assign(nLoc);
              const rpFarB: TSLNode = clamp(
                isOrtho
                  ? pFarWidth.mul(uOrthoLineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
                  : pFarWidth
                      .mul(uPerspectiveLineScale)
                      .mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
                      .div(pFarDepth),
                CAPSULE_MIN_RADIUS_PX,
                uMaxLinePixelWidth
              ).toVar();
              const deficitB: TSLNode = clamp(
                float(1.0).sub(rpFarB.div(max(rB, float(1e-4)))),
                0.0,
                1.0
              ).toVar();
              If(deficitB.greaterThan(CAPSULE_JOINT_DEFICIT_GATE), () => {
                vJointB.assign(vec4(dot(qhat, u), dot(qhat, v), ql, rpFarB));
              });
              extB.assign(max(abs(nLoc.y), deficitB).mul(rMax).add(CAPSULE_STENCIL_APRON_PX));
            });
          }).Else(() => {
            // Near-hairpin (see end A).
            capB.assign(0.0);
            extB.assign(rMax);
          });
        });
      });
    });

    const lx: TSLNode = aQuadCorner.x
      .greaterThan(0.0)
      .select(abLen.add(extB), extA.negate())
      .toVar();
    const ly: TSLNode = aQuadCorner.y.mul(rMax).toVar();
    const corner: TSLNode = pA.add(u.mul(lx)).add(v.mul(ly)).toVar();

    const tc: TSLNode = abLen
      .greaterThan(1e-4)
      .select(clamp(lx.div(abLen), 0.0, 1.0), float(0.5))
      .toVar();
    const tOrig: TSLNode = mix(tA, tB, tc).toVar();
    const rC: TSLNode = mix(rA, rB, tc).toVar();
    const rawC: TSLNode = mix(rawA, rawB, tc).toVar();
    const widthScale: TSLNode = min(rawC.div(CAPSULE_MIN_RADIUS_PX), 1.0).toVar();
    const fade: TSLNode = isOrtho
      ? float(1.0).toVar()
      : perspectiveNearFadeStaticTSL(false, mix(mvA.z, mvB.z, tc), nearCull).toVar();

    vMeta.assign(vec3(abLen, capA, capB));
    vREnd.assign(vec2(rA, rB));
    vCutA2.assign(cutA);
    vCutB2.assign(cutB);
    vFade.assign(fade.mul(widthScale).mul(culled.select(float(0.0), float(1.0))));
    vSharp.assign(mix(s0, s1, tOrig));

    const clipMix: TSLNode = mix(clipA, clipB, tc).toVar();
    const wMix: TSLNode = max(clipMix.w, float(1e-6)).toVar();
    // Screen-linear geometry varyings (see the visual twin).
    vLocal.assign(vec2(lx, ly).mul(wMix));
    vR.assign(rC.mul(wMix));
    vW.assign(wMix);
    const ndc: TSLNode = corner.div(uResolution).mul(2.0).sub(1.0).toVar();
    const clipPosOut: TSLNode = vec4(0.0, 0.0, -2.0, 1.0).toVar();
    If(culled.not(), () => {
      clipPosOut.assign(vec4(ndc.mul(wMix), clipMix.z, wMix));
    });
    return clipPosOut;
  });

  const clipPos: TSLNode = vertexBody();

  // ---- Fragment: shared brightness, contract output + brightness-depth ----
  const brightnessShared = Fn(() => {
    const invW: TSLNode = float(1.0)
      .div(max(vW, float(1e-9)))
      .toVar();
    const x: TSLNode = vLocal.x.mul(invW).toVar();
    const y: TSLNode = vLocal.y.mul(invW).toVar();
    const rPx: TSLNode = max(vR.mul(invW), float(1e-4)).toVar();
    // TRUE point-to-segment distance: every end is capped (a free end
    // keeps the whole disc, a cut end its half of the joint disc).
    const oxA: TSLNode = x.negate().toVar();
    const oxB: TSLNode = x.sub(vMeta.x).toVar();
    const ox: TSLNode = max(max(oxA, oxB), 0.0).toVar();
    const q: TSLNode = y.mul(y).add(ox.mul(ox)).div(rPx.mul(rPx)).toVar();
    const w: TSLNode = float(1.0).sub(q).toVar();
    Discard(w.lessThanEqual(0.0));
    const profile: TSLNode = abs(vSharp.sub(0.5))
      .lessThan(1e-3)
      .select(w.mul(w).toVar(), pow(w, exp2(float(3.0).sub(vSharp.mul(4.0)))).toVar())
      .toVar();

    // Joint DEFICIT rule (see the GLSL twin's note): on the partner's side
    // of the joint bisector I render max(mine − partner, 0), so the pair
    // composes to max(mine, partner) — exact partition for congruent legs,
    // and a fat vertex's disc no longer loses its far half to a thin
    // neighbour. Packet length 0 = no usable partner: hard cut.
    const partnerProfile = (rel: TSLNode, joint: TSLNode, rEnd: TSLNode): TSLNode => {
      const xp: TSLNode = dot(rel, joint.xy).toVar();
      const yp2: TSLNode = max(dot(rel, rel).sub(xp.mul(xp)), 0.0).toVar();
      const tp: TSLNode = clamp(xp.div(max(joint.z, float(1e-4))), 0.0, 1.0).toVar();
      const rp: TSLNode = max(mix(rEnd, joint.w, tp), float(1e-4)).toVar();
      const op: TSLNode = max(max(xp.negate(), xp.sub(joint.z)), 0.0).toVar();
      const qp: TSLNode = yp2.add(op.mul(op)).div(rp.mul(rp)).toVar();
      const wp: TSLNode = max(float(1.0).sub(qp), 0.0).toVar();
      return abs(vSharp.sub(0.5))
        .lessThan(1e-3)
        .select(wp.mul(wp), pow(wp, exp2(float(3.0).sub(vSharp.mul(4.0)))))
        .toVar();
    };
    If(vMeta.y.greaterThan(0.5).and(vCutA2.x.mul(x).add(vCutA2.y.mul(y)).greaterThan(0.0)), () => {
      Discard(vJointA.z.lessThan(0.5));
      profile.subAssign(partnerProfile(vec2(x, y), vJointA, vREnd.x));
      Discard(profile.lessThanEqual(0.0));
    });
    If(
      vMeta.z
        .greaterThan(0.5)
        .and(vCutB2.x.mul(x.sub(vMeta.x)).add(vCutB2.y.mul(y)).greaterThan(0.0)),
      () => {
        Discard(vJointB.z.lessThan(0.5));
        profile.subAssign(partnerProfile(vec2(x.sub(vMeta.x), y), vJointB, vREnd.y));
        Discard(profile.lessThanEqual(0.0));
      }
    );
    return profile.mul(vFade);
  }).once();
  const brightness: TSLNode = brightnessShared().toVar('lineCapsulePickBrightness');

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
