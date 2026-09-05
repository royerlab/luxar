/**
 * Capsule line primitive — TSL / NodeMaterial twin of
 * `shader-glsl-capsule.ts` (issue #1352, the DEFAULT line primitive (`?linePrimitive=capsule`)).
 *
 * Same model, same constants: a gaussian-like quartic profile of the 2D
 * point-to-segment distance, evaluated on stencil-LOCAL interpolated
 * varyings, with half-disc bisector interior joints — see
 * `_shared/line-capsule.ts` for the model and the
 * exactness relaxations. Value-level parity with the GLSL twin is pinned
 * by the `line-capsule-*` fixtures in the tsl-shader-parity suite.
 *
 * Structural notes (the TSL house rules, shared with `shader-tsl.ts`):
 * - the whole vertex stage is ONE `Fn()` body of `.toVar()` statements;
 *   varyings are declared outside and `.assign()`ed inside; per-segment
 *   constants are `flat` (written identically on every vertex).
 * - build-time JS branches replace GLSL defines: `config.isOrtho`,
 *   `config.useColormap`, `config.gammaOne`, `config.noGOG`, and the
 *   blending mode select the graph variant.
 * - every GLSL discard is mirrored exactly (cut sides, zero support,
 *   zero-contribution colour).
 * - unlike the quad fragment, the capsule fragment reads NO
 *   screen coordinate at all — the stencil-local varyings carry the
 *   geometry — so there is no y-flip hazard here.
 */
import {
  abs,
  attribute,
  cameraProjectionMatrix,
  clamp,
  Discard,
  dot,
  exp,
  exp2,
  float,
  Fn,
  If,
  int,
  ivec2,
  length,
  log,
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
import { NodeMaterial } from 'three/webgpu';
import { resolveElementTextureWidth, LINE_TEXTURE_LAYOUT } from '../../element-texture-layout';
import {
  CAPSULE_JOINT_DEFICIT_GATE,
  CAPSULE_JOINT_PACKET_MIN_RADIUS_PX,
  CAPSULE_MIN_RADIUS_PX,
  CAPSULE_RADIUS_PER_QUAD_HALFWIDTH,
  CAPSULE_STENCIL_APRON_PX,
} from '../_shared/line-capsule';
import {
  ALPHA_CLAMP,
  VOLUMETRIC_SERIES_C1,
  VOLUMETRIC_SERIES_C2_DIVISOR,
  VOLUMETRIC_SERIES_TAU_THRESHOLD,
  VOLUMETRIC_TAU_EPS,
} from '../_shared/volumetric';
import {
  perspectiveNearFadeStaticTSL,
  sanitizeAlpha,
  sanitizeNonNegative,
  tslLineJointCapSuppression,
  type TSLNode,
  sortedIndexNode,
  densityDroppedNode,
} from '../_shared/tsl-helpers';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  isVolumetricMode,
} from '../../blending-state';
import type { LineTSLConfig, LineTSLNodes } from './shader-tsl';

/**
 * Capsule-line TSL factory. Same `nodes`/`config` contract as
 * `lineWebGPUFactory` (the wrapper class and the parity harness feed both
 * from the same records); `config.join` is ignored — the capsule has no
 * miter geometry, interior joints are 2D bisector cuts.
 */
export function capsuleLineWebGPUFactory(
  nodes: LineTSLNodes,
  config: LineTSLConfig = {},
  outMaterial?: NodeMaterial
): NodeMaterial {
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  const aSortedIndex: TSLNode = sortedIndexNode(nodes.uSortedIndexSlot);
  const densityDropped: TSLNode = densityDroppedNode(nodes.uDensityDrop, aSortedIndex);

  const uLineTex = nodes.uLineTex;
  const uResolution = nodes.uResolution;
  const uPixelRatio = nodes.uPixelRatio;
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
        'capsuleLineWebGPUFactory: config.useColormap=true but nodes.uColormapTex / uScalarMin / uScalarScale are not bound.'
      );
    }
  }
  const uColormapTex = config.useColormap ? nodes.uColormapTex! : null;
  const uScalarMin = config.useColormap ? nodes.uScalarMin! : null;
  const uScalarScale = config.useColormap ? nodes.uScalarScale! : null;

  const blendingMode = config.blendingMode ?? 'additive';
  const volumetric = isVolumetricMode(blendingMode);
  const opaque = blendingMode === 'opaque';
  const premultiplyRGB =
    config.useMaxRGBContribution !== undefined
      ? config.useMaxRGBContribution
      : blendingMode === 'max';
  const isOrtho = config.isOrtho === true;

  // ---- Varyings (stencil-local geometry interpolated; cuts flat) ----
  // Geometry varyings are screen-space quantities pre-multiplied by the
  // corner's clip w and divided by vW in the fragment — screen-linear
  // (default interpolation is perspective-correct = hyperbolic in screen
  // space; see the GLSL twin's declaration note).
  const vLocal: TSLNode = varying(vec2(0.0, 0.0));
  // PACKED joint state — the exact mirror of the GLSL twin's layout (see
  // shader-glsl-capsule.ts: register pressure is the measured cost; cut
  // normals stay FULL precision per #1502).
  // vCutN = (nAx, nAy, nBx, nBy); vPack halves = (gradA,qlA) (gradB,qlB)
  // (rA,rB) (1/abLen, capFlags); vAbLen = full-precision length.
  const vCutN: TSLNode = varying(vec4(-1.0, 0.0, 1.0, 0.0)).setInterpolation('flat');
  const vPack: TSLNode = varying(uvec4(0, 0, 0, 0)).setInterpolation('flat');
  const vAbLen: TSLNode = varying(float(1.0)).setInterpolation('flat');
  // The packed packet lanes carry the partner's radius gradient (px/px,
  // either sign) + its projected length, which doubles as packet validity
  // (0 = hard cut) — the gate opens for every deficit source (#1495).
  // (tc clamps per vertex; see #1494 and the GLSL twin).
  const vW: TSLNode = varying(float(1.0));
  const vFade: TSLNode = varying(float(1.0));
  const vAlphaV: TSLNode = varying(float(1.0));
  const vSharp: TSLNode = varying(float(0.5));
  const vScalar: TSLNode | null = config.useColormap ? varying(float(0.0)) : null;
  const vColor: TSLNode | null = config.useColormap ? null : varying(vec3(0.0, 0.0, 0.0));

  const nearCull: TSLNode = max(uNearCull, float(1e-20));

  const vertexBody = Fn(() => {
    // === Line-texture fetch (6 texels/segment) ===
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
    const lineT5: TSLNode = uLineTex.load(ivec2(texelX.add(int(5)), texelY)).toVar();

    const mvStart: TSLNode = modelViewMatrix.mul(vec4(lineT0.xyz, 1.0)).toVar();
    const mvEnd: TSLNode = modelViewMatrix.mul(vec4(lineT1.xyz, 1.0)).toVar();

    const startDepth: TSLNode = mvStart.z.negate().toVar();
    const endDepth: TSLNode = mvEnd.z.negate().toVar();
    // Projected-density thinning rides the both-behind cull (GLSL twin does the
    // same), so a dropped segment takes the zero-varyings path.
    const bothBehind: TSLNode = isOrtho
      ? float(0.0).greaterThan(1.0)
      : startDepth.lessThan(nearCull).and(endDepth.lessThan(nearCull));
    const culled: TSLNode = bothBehind.or(densityDropped).toVar();

    // Near-plane segment clip (perspective only) — stencil AND domain.
    const mvA: TSLNode = mvStart.toVar();
    const mvB: TSLNode = mvEnd.toVar();
    // Clip flags survive the block: a clipped end means the JOINT VERTEX
    // is behind the near plane, which the cut construction must know.
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
    // Endpoint attributes at the CLIPPED span (see the GLSL twin).
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

    // Per-end pixel radius: 2σ-trimmed fraction of the quad half-width.
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
    const appearancePixelRatio: TSLNode = uPixelRatio.max(float(1.0));
    const minRadius: TSLNode = appearancePixelRatio.mul(CAPSULE_MIN_RADIUS_PX).toVar();
    const packetMinRadius: TSLNode = appearancePixelRatio
      .mul(CAPSULE_JOINT_PACKET_MIN_RADIUS_PX)
      .toVar();
    const rA: TSLNode = clamp(rawA, minRadius, uMaxLinePixelWidth).toVar();
    const rB: TSLNode = clamp(rawB, minRadius, uMaxLinePixelWidth).toVar();

    // Which ends CUT rather than cap — the shared joint-code rule (a free
    // end and a degree->=3 hub keep the whole round cap; a slot-bearing
    // code cuts at the bisector, a slice-clipped end at the butt).
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
    // The joint partition is NOT width-gated (see the GLSL twin's note:
    // the drawn radius is floored at the AA minimum, so a gated hairline
    // joint would overlap two full caps — +1.00 of peak, never sub-pixel).

    // Half-disc bisector joints per end (mirrors the GLSL twin).
    const cutA: TSLNode = vec4(-1.0, 0.0, 0.0, 0.0).toVar();
    const cutB: TSLNode = vec4(1.0, 0.0, 0.0, 0.0).toVar();
    const extA: TSLNode = rMax.toVar();
    const extB: TSLNode = rMax.toVar();
    const capA: TSLNode = interiorA.toVar();
    const capB: TSLNode = interiorB.toVar();

    const pFarWidth: TSLNode = float(0.0).toVar();
    const pFarDepth: TSLNode = float(1.0).toVar();
    const partnerFarPx = (code: TSLNode, mvJoint: TSLNode, jointDepth: TSLNode): TSLNode => {
      // Decode the joint code to the partner's FAR endpoint, projected to
      // pixels; .z carries validity (0 when the code is a sentinel). The
      // far endpoint is near-clipped toward the joint vertex BEFORE
      // projecting — a behind-eye projection flips, and the garbage
      // direction poisons the fold decision and the cut normal.
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
      // Joint vertex behind the near plane: the joint region is invisible
      // and the partner's projection meaningless — keep the butt cut.
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
              // Width gate + its floored sharp-turn exception (#1495; see
              // the GLSL twin's note for both halves).
              If(
                rMax
                  .greaterThan(packetMinRadius)
                  .or(
                    dot(qhat, u).greaterThan(0.5).and(min(rawA, rawB).greaterThanEqual(minRadius))
                  ),
                () => {
                  const rpFarA: TSLNode = clamp(
                    isOrtho
                      ? pFarWidth.mul(uOrthoLineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
                      : pFarWidth
                          .mul(uPerspectiveLineScale)
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
      // See end A: behind-near joint keeps the butt cut.
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
                      ? pFarWidth.mul(uOrthoLineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
                      : pFarWidth
                          .mul(uPerspectiveLineScale)
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

    // Corner in stencil-local coordinates.
    const lx: TSLNode = aQuadCorner.x
      .greaterThan(0.0)
      .select(abLen.add(extB), extA.negate())
      .toVar();
    const ly: TSLNode = aQuadCorner.y.mul(rMax).toVar();
    const corner: TSLNode = pA.add(u.mul(lx)).add(v.mul(ly)).toVar();

    // Attributes at the clamped corner position.
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
    vAlphaV.assign(mix(sanitizeAlpha(lineT5.z), sanitizeAlpha(lineT5.w), tOrig));
    vSharp.assign(mix(s0, s1, tOrig));
    if (vScalar) vScalar.assign(mix(lineT5.x, lineT5.y, tOrig));
    if (vColor) vColor.assign(mix(vec3(lineT2), vec3(lineT3), tOrig));

    // Depth interpolates along the segment; culled segments collapse.
    const clipMix: TSLNode = mix(clipA, clipB, tc).toVar();
    const wMix: TSLNode = max(clipMix.w, float(1e-6)).toVar();
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

  // ---- Fragment ----
  const colorNode = Fn(() => {
    // Undo the w-premultiplication: screen-linear local coordinates.
    const invW: TSLNode = float(1.0)
      .div(max(vW, float(1e-9)))
      .toVar();
    const x: TSLNode = vLocal.x.mul(invW).toVar();
    const y: TSLNode = vLocal.y.mul(invW).toVar();
    // Unpack the flat joint state (see the GLSL twin).
    const pkA: TSLNode = (unpackHalf2x16(vPack.x) as unknown as TSLNode).toVar();
    const pkB: TSLNode = (unpackHalf2x16(vPack.y) as unknown as TSLNode).toVar();
    const pkR: TSLNode = (unpackHalf2x16(vPack.z) as unknown as TSLNode).toVar();
    const pkM: TSLNode = (unpackHalf2x16(vPack.w) as unknown as TSLNode).toVar();
    // Bisector-cut sides (my side negative); straight joints = butt.
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
    // Squared distance in the local frame — the TRUE point-to-segment
    // distance, cap term included at BOTH ends: every end is capped (a free
    // end keeps the whole disc, a cut end its half of the joint disc).
    const oxA: TSLNode = x.negate().toVar();
    const oxB: TSLNode = x.sub(vAbLen).toVar();
    const ox: TSLNode = max(max(oxA, oxB), 0.0).toVar();
    const q: TSLNode = y.mul(y).add(ox.mul(ox)).div(rPx.mul(rPx)).toVar();
    const w: TSLNode = float(1.0).sub(q).toVar();
    Discard(w.lessThanEqual(0.0));

    // Quartic bump; the sharpness knob bends it off the default only.
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
    const intensity: TSLNode = profile.mul(vFade).toVar();

    let color: TSLNode;
    if (config.useColormap) {
      let st: TSLNode = clamp(vScalar!.sub(uScalarMin!).mul(uScalarScale!), 0.0, 1.0).toVar();
      if (!config.gammaOne) {
        st = pow(st, uInvGamma).toVar();
      }
      color = uColormapTex!.sample(vec2(st, 0.5)).rgb.toVar();
    } else {
      color = vColor!.toVar();
    }

    const adjusted: TSLNode = config.noGOG
      ? color
      : max(color.mul(uIntensity).add(uOffset), vec3(0.0)).toVar();
    const maxAdjusted: TSLNode = max(adjusted.r, max(adjusted.g, adjusted.b)).toVar();

    const gammaColor: TSLNode =
      config.useColormap || config.gammaOne ? adjusted : pow(adjusted, vec3(uInvGamma)).toVar();

    if (volumetric) {
      // Emission–absorption tail, identical to the quad twin.
      const alphaBase: TSLNode = intensity.mul(uOpacity).toVar();
      const optical: TSLNode = log(float(1.0).sub(min(vAlphaV, ALPHA_CLAMP)))
        .negate()
        .toVar();
      const alpha: TSLNode = alphaBase.mul(mix(float(1.0), optical, uHasElementAlpha)).toVar();
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
    const intensityScaled: TSLNode = intensity.mul(vAlphaV).toVar();
    if (opaque) {
      Discard(maxAdjusted.mul(intensityScaled).mul(uOpacity).lessThan(1e-4));
    } else {
      Discard(maxAdjusted.lessThan(1e-4));
    }
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

  const opacityValue = (nodes.uOpacity.value as number | undefined) ?? 1.0;
  const blendingState = getCompleteBlendingState(blendingMode, opacityValue);
  applyBlendingStateToMaterial(material, blendingState);
  return material;
}
