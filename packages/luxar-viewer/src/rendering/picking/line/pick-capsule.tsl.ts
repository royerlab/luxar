/**
 * Capsule line PICKING — TSL / NodeMaterial twin of `shaders-capsule.ts`
 * (issue #1352, the DEFAULT line primitive (`?linePrimitive=capsule`)).
 *
 * Same stencil, half-disc joints and quartic profile as the visual
 * capsule TSL factory (`materials/line/shader-tsl-capsule.ts`); the pick
 * output contract is the shared one:
 *   `vec4(nodeId, elementId-low16, brightness, elementId-high16)`,
 * with brightness-as-depth via `material.depthNode`. Per-element alpha and
 * node opacity are ignored, matching the other pick variants.
 */
import {
  abs,
  attribute,
  cameraProjectionMatrix,
  clamp,
  Discard,
  dot,
  exp2,
  float,
  Fn,
  If,
  int,
  ivec2,
  length,
  max,
  min,
  mix,
  mod,
  modelViewMatrix,
  packHalf2x16,
  pow,
  unpackHalf2x16,
  uvec4,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import * as THREE from 'three';
import { NodeMaterial } from 'three/webgpu';
import { resolveElementTextureWidth, LINE_TEXTURE_LAYOUT } from '../../element-texture-layout';
import {
  CAPSULE_JOINT_DEFICIT_GATE,
  CAPSULE_JOINT_PACKET_MIN_RADIUS_PX,
  CAPSULE_MIN_RADIUS_PX,
  CAPSULE_RADIUS_PER_QUAD_HALFWIDTH,
  CAPSULE_STENCIL_APRON_PX,
} from '../../materials/_shared/line-capsule';
import {
  projectionSizeScaleTSL,
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
  const uPixelRatio = nodes.uPixelRatio;
  const uNodeId = nodes.uNodeId;
  const uNearCull = nodes.uNearCull;
  const uMaxLinePixelWidth = nodes.uMaxLinePixelWidth;
  // Pixels per view unit at unit depth: resY * |P11|, read from the
  // projection this draw uses (GLSL twin: luxarProjectionSizeScale). It is
  // the historical uPerspectiveLineScale AND uOrthoLineScale.
  const lineScale: TSLNode = uResolution.y.mul(projectionSizeScaleTSL());

  const isOrtho = config.isOrtho === true;

  // ---- Varyings ----
  const vLocal: TSLNode = varying(vec2(0.0, 0.0));
  // PACKED joint state — mirrors the visual TSL twin exactly (normals
  // full precision per #1502; halves for packets/radii/invLen/flags).
  const vCutN: TSLNode = varying(vec4(-1.0, 0.0, 1.0, 0.0)).setInterpolation('flat');
  const vPack: TSLNode = varying(uvec4(0, 0, 0, 0)).setInterpolation('flat');
  const vAbLen: TSLNode = varying(float(1.0)).setInterpolation('flat');
  // The packed packet lanes carry the partner's radius gradient (px/px,
  // either sign) + its projected length, which doubles as packet validity
  // (0 = hard cut) — the gate opens for every deficit source (#1495).
  // (tc clamps per vertex; see #1494 and the GLSL twin).
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
    const lineTexW: TSLNode = int(
      resolveElementTextureWidth(
        LINE_TEXTURE_LAYOUT,
        (nodes.uLineTex as unknown as { value?: { image?: { width?: number } } }).value ?? null
      )
    ).toVar();
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
        ? wEffA.mul(lineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
        : wEffA
            .mul(lineScale)
            .mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
            .div(max(mvA.z.negate(), nearCull))
    ).toVar();
    const rawB: TSLNode = (
      isOrtho
        ? wEffB.mul(lineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
        : wEffB
            .mul(lineScale)
            .mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
            .div(max(mvB.z.negate(), nearCull))
    ).toVar();
    const appearancePixelRatio: TSLNode = uPixelRatio.max(float(1.0));
    const minRadius: TSLNode = appearancePixelRatio.mul(CAPSULE_MIN_RADIUS_PX).toVar();
    const packetMinRadius: TSLNode = appearancePixelRatio
      .mul(CAPSULE_JOINT_PACKET_MIN_RADIUS_PX)
      .toVar();
    const rA: TSLNode = clamp(rawA, minRadius, uMaxLinePixelWidth).toVar();
    const rB: TSLNode = clamp(rawB, minRadius, uMaxLinePixelWidth).toVar();

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
    // The joint partition is NOT width-gated — the pick shape must stay
    // the visual shape (see the visual twin's note).

    const cutA: TSLNode = vec4(-1.0, 0.0, 0.0, 0.0).toVar();
    const cutB: TSLNode = vec4(1.0, 0.0, 0.0, 0.0).toVar();
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
            // Cut normal: keep full precision — no per-leg quantisation
            // (see `_shared/line-capsule.ts`'s note; #1502).
            const n2: TSLNode = nRaw.div(nl).toVar();
            const nLoc: TSLNode = vec2(dot(n2, u), dot(n2, v)).toVar();
            If(nLoc.x.lessThan(-1e-3), () => {
              cutA.assign(vec4(nLoc, 0.0, 0.0));
              // Width gate + its floored sharp-turn exception (#1495),
              // exactly as the visual twin (or hover desyncs from pixels).
              If(
                rMax
                  .greaterThan(packetMinRadius)
                  .or(
                    dot(qhat, u).greaterThan(0.5).and(min(rawA, rawB).greaterThanEqual(minRadius))
                  ),
                () => {
                  const rpFarA: TSLNode = clamp(
                    isOrtho
                      ? pFarWidth.mul(lineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
                      : pFarWidth
                          .mul(lineScale)
                          .mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
                          .div(pFarDepth),
                    minRadius,
                    uMaxLinePixelWidth
                  ).toVar();
                  // Packet gate (#1495, #1501; see the GLSL twin's note).
                  const needPacketA: TSLNode = abs(float(1.0).sub(rpFarA.div(max(rA, float(1e-4)))))
                    .greaterThan(CAPSULE_JOINT_DEFICIT_GATE)
                    .or(rB.greaterThan(rA.mul(float(1.0).add(CAPSULE_JOINT_DEFICIT_GATE))))
                    .or(ql.lessThan(rA.mul(2.0)))
                    .or(dot(qhat, u).greaterThan(0.5))
                    .toVar();
                  If(needPacketA, () => {
                    cutA.z.assign(rpFarA.sub(rA).div(ql));
                    cutA.w.assign(ql);
                    // Full-disc reach: the deficit term ≤ my own profile (#1488).
                    extA.assign(rMax.add(CAPSULE_STENCIL_APRON_PX));
                  }).Else(() => {
                    extA.assign(abs(nLoc.y).mul(rMax).add(CAPSULE_STENCIL_APRON_PX));
                  });
                }
              ).Else(() => {
                extA.assign(abs(nLoc.y).mul(rMax).add(CAPSULE_STENCIL_APRON_PX));
              });
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
              cutB.assign(vec4(nLoc, 0.0, 0.0));
              // Width gate + its floored sharp-turn exception (see end A).
              If(
                rMax
                  .greaterThan(packetMinRadius)
                  .or(dot(qhat, u).lessThan(-0.5).and(min(rawA, rawB).greaterThanEqual(minRadius))),
                () => {
                  const rpFarB: TSLNode = clamp(
                    isOrtho
                      ? pFarWidth.mul(lineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
                      : pFarWidth
                          .mul(lineScale)
                          .mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
                          .div(pFarDepth),
                    minRadius,
                    uMaxLinePixelWidth
                  ).toVar();
                  // Packet gate (#1495, #1501; see the GLSL twin's note).
                  const needPacketB: TSLNode = abs(float(1.0).sub(rpFarB.div(max(rB, float(1e-4)))))
                    .greaterThan(CAPSULE_JOINT_DEFICIT_GATE)
                    .or(rA.greaterThan(rB.mul(float(1.0).add(CAPSULE_JOINT_DEFICIT_GATE))))
                    .or(ql.lessThan(rB.mul(2.0)))
                    .or(dot(qhat, u).lessThan(-0.5))
                    .toVar();
                  If(needPacketB, () => {
                    cutB.z.assign(rpFarB.sub(rB).div(ql));
                    cutB.w.assign(ql);
                    // Full-disc reach: the deficit term ≤ my own profile (#1488).
                    extB.assign(rMax.add(CAPSULE_STENCIL_APRON_PX));
                  }).Else(() => {
                    extB.assign(abs(nLoc.y).mul(rMax).add(CAPSULE_STENCIL_APRON_PX));
                  });
                }
              ).Else(() => {
                extB.assign(abs(nLoc.y).mul(rMax).add(CAPSULE_STENCIL_APRON_PX));
              });
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
    const rawC: TSLNode = mix(rawA, rawB, tc).toVar();
    const widthScale: TSLNode = min(rawC.div(minRadius), 1.0).toVar();
    const fade: TSLNode = isOrtho
      ? float(1.0).toVar()
      : perspectiveNearFadeStaticTSL(false, mix(mvA.z, mvB.z, tc), nearCull).toVar();

    vCutN.assign(vec4(cutA.xy, cutB.xy));
    vPack.assign(
      uvec4(
        packHalf2x16(cutA.zw) as unknown as TSLNode,
        packHalf2x16(cutB.zw) as unknown as TSLNode,
        packHalf2x16(vec2(rA, rB)) as unknown as TSLNode,
        packHalf2x16(
          vec2(float(1.0).div(max(abLen, float(1e-4))), capA.add(capB.mul(2.0)))
        ) as unknown as TSLNode
      )
    );
    vAbLen.assign(abLen);
    vFade.assign(fade.mul(widthScale).mul(culled.select(float(0.0), float(1.0))));
    vSharp.assign(mix(s0, s1, tOrig));

    const clipMix: TSLNode = mix(clipA, clipB, tc).toVar();
    const wMix: TSLNode = max(clipMix.w, float(1e-6)).toVar();
    // Screen-linear geometry varyings (see the visual twin).
    vLocal.assign(vec2(lx, ly).mul(wMix));
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
    // Unpack the flat joint state (see the visual twin).
    const pkA: TSLNode = (unpackHalf2x16(vPack.x) as unknown as TSLNode).toVar();
    const pkB: TSLNode = (unpackHalf2x16(vPack.y) as unknown as TSLNode).toVar();
    const pkR: TSLNode = (unpackHalf2x16(vPack.z) as unknown as TSLNode).toVar();
    const pkM: TSLNode = (unpackHalf2x16(vPack.w) as unknown as TSLNode).toVar();
    // EXACT per-fragment radius from the endpoint radii (see the GLSL
    // twin's note — a linear varying cannot represent this).
    const cutFlagA: TSLNode = mod(pkM.y, 2.0).toVar();
    const cutFlagB: TSLNode = pkM.y.greaterThanEqual(2.0).select(float(1.0), float(0.0)).toVar();
    // Div-free exact radius (see the GLSL twin); constant-width fast path.
    const rPx: TSLNode = float(0.0).toVar();
    If(pkR.x.equal(pkR.y), () => {
      rPx.assign(max(pkR.x, float(1e-4)));
    }).Else(() => {
      rPx.assign(max(mix(pkR.x, pkR.y, clamp(x.mul(pkM.x), 0.0, 1.0)), float(1e-4)));
    });
    // TRUE point-to-segment distance: every end is capped (a free end
    // keeps the whole disc, a cut end its half of the joint disc).
    const oxA: TSLNode = x.negate().toVar();
    const oxB: TSLNode = x.sub(vAbLen).toVar();
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
    const partnerProfile = (cut: TSLNode, rel: TSLNode, mSign: number, rEnd: TSLNode): TSLNode => {
      // Partner axis = my inward axis reflected across the cut plane
      // (exact). Radius from the SHARED VERTEX radius (rEnd — the caller's
      // pkR.x / pkR.y, from the packed vPack.z lane, #1494), tapered by
      // the packed gradient, FROZEN past the far end;
      // the far cap term closes the rod there (#1490).
      const nx: TSLNode = cut.x.toVar();
      const ny: TSLNode = cut.y.toVar();
      const proj: TSLNode = nx.mul(mSign).toVar();
      const qdir: TSLNode = vec2(
        float(mSign).sub(proj.mul(nx).mul(2.0)),
        proj.mul(ny).mul(-2.0)
      ).toVar();
      const xp: TSLNode = dot(rel, qdir).toVar();
      const yp2: TSLNode = max(dot(rel, rel).sub(xp.mul(xp)), 0.0).toVar();
      const rp: TSLNode = max(rEnd.add(cut.z.mul(clamp(xp, 0.0, cut.w))), float(1e-4)).toVar();
      const op: TSLNode = max(max(xp.negate(), xp.sub(cut.w)), 0.0).toVar();
      const qp: TSLNode = yp2.add(op.mul(op)).div(rp.mul(rp)).toVar();
      const wp: TSLNode = max(float(1.0).sub(qp), 0.0).toVar();
      return abs(vSharp.sub(0.5))
        .lessThan(1e-3)
        .select(wp.mul(wp), pow(wp, exp2(float(3.0).sub(vSharp.mul(4.0)))))
        .toVar();
    };
    // 1 px AA ramp on the cut + deficit blend (see the GLSL twin's note).
    If(cutFlagA.greaterThan(0.5), () => {
      const sideA: TSLNode = vCutN.x.mul(x).add(vCutN.y.mul(y)).toVar();
      If(sideA.greaterThan(-0.5), () => {
        const coverA: TSLNode = clamp(float(0.5).sub(sideA), 0.0, 1.0).toVar();
        const defA: TSLNode = float(0.0).toVar();
        If(pkA.y.greaterThan(0.0), () => {
          defA.assign(
            max(profile.sub(partnerProfile(vec4(vCutN.xy, pkA), vec2(x, y), 1.0, pkR.x)), 0.0)
          );
        });
        profile.assign(profile.mul(coverA).add(defA.mul(float(1.0).sub(coverA))));
        Discard(profile.lessThanEqual(0.0));
      });
    });
    If(cutFlagB.greaterThan(0.5), () => {
      const sideB: TSLNode = vCutN.z.mul(x.sub(vAbLen)).add(vCutN.w.mul(y)).toVar();
      If(sideB.greaterThan(-0.5), () => {
        const coverB: TSLNode = clamp(float(0.5).sub(sideB), 0.0, 1.0).toVar();
        const defB: TSLNode = float(0.0).toVar();
        If(pkB.y.greaterThan(0.0), () => {
          defB.assign(
            max(
              profile.sub(partnerProfile(vec4(vCutN.zw, pkB), vec2(x.sub(vAbLen), y), -1.0, pkR.y)),
              0.0
            )
          );
        });
        profile.assign(profile.mul(coverB).add(defB.mul(float(1.0).sub(coverB))));
        Discard(profile.lessThanEqual(0.0));
      });
    });
    return profile.mul(vFade);
  }).once();
  const brightness: TSLNode = brightnessShared().toVar('lineCapsulePickBrightness');

  const colorNode = Fn(() => {
    Discard(brightness.lessThan(1e-4));
    return vec4(vNodeId, vElementId.x, brightness, vElementId.y);
  });

  const depthNode = Fn(() => {
    // BRANCHLESS by construction, and that is load-bearing: `brightness` is a
    // factory-scope `.toVar()` shared with `colorNode`, so it is assigned wherever
    // three first BUILDS it — which is unconditional top-level flow in either entry
    // point only while this body contains no `if`. Adding a branch here (a
    // `uSurfaceDepth`-style select) would bury that assignment in one arm and leave
    // `colorNode`'s top-level readers with 0; it needs the same unconditional fragment
    // prologue the gsplat/mesh pick factories use.
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
