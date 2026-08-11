/**
 * Capsule line primitive — TSL / NodeMaterial twin of
 * `shader-glsl-capsule.ts` (issue #1352, behind `?linePrimitive=capsule`).
 *
 * Same model, same constants: a gaussian-like quartic profile of the 2D
 * point-to-segment distance, evaluated on stencil-LOCAL interpolated
 * varyings, with 2D bisector-cut interior joints and the width-gated
 * fold-cap rule — see `_shared/line-capsule.ts` for the model and the
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
 * - unlike the quad/volumetric fragments, the capsule fragment reads NO
 *   screen coordinate at all — the stencil-local varyings carry the
 *   geometry — so there is no y-flip hazard here.
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
  exp,
  exp2,
  log,
  pow,
  dot,
  abs,
  textureSize,
  modelViewMatrix,
  cameraProjectionMatrix,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import {
  CAPSULE_FOLD_CAP_MAX_COS,
  CAPSULE_FOLD_CAP_MIN_RADIUS_PX,
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
  type TSLNode,
  sortedIndexNode,
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
        'capsuleLineWebGPUFactory: config.useColormap=true but nodes.uColormapTex / uScalarMin / uScalarScale are not bound.'
      );
    }
  }
  const uColormapTex = config.useColormap ? nodes.uColormapTex! : null;
  const uScalarMin = config.useColormap ? nodes.uScalarMin! : null;
  const uScalarScale = config.useColormap ? nodes.uScalarScale! : null;

  const blendingMode = config.blendingMode ?? 'additive';
  const volumetric = isVolumetricMode(blendingMode);
  const premultiplyRGB =
    config.useMaxRGBContribution !== undefined
      ? config.useMaxRGBContribution
      : blendingMode === 'max';
  const isOrtho = config.isOrtho === true;

  // ---- Varyings (stencil-local geometry interpolated; cuts flat) ----
  const vLocal: TSLNode = varying(vec2(0.0, 0.0));
  const vMeta: TSLNode = varying(vec3(1.0, 0.0, 0.0)).setInterpolation('flat');
  const vCutA2: TSLNode = varying(vec2(-1.0, 0.0)).setInterpolation('flat');
  const vCutB2: TSLNode = varying(vec2(1.0, 0.0)).setInterpolation('flat');
  const vInvR2: TSLNode = varying(float(1.0));
  const vFade: TSLNode = varying(float(1.0));
  const vAlphaV: TSLNode = varying(float(1.0));
  const vSharp: TSLNode = varying(float(0.5));
  const vScalar: TSLNode | null = config.useColormap ? varying(float(0.0)) : null;
  const vColor: TSLNode | null = config.useColormap ? null : varying(vec3(0.0, 0.0, 0.0));

  const nearCull: TSLNode = max(uNearCull, float(1e-20));

  const vertexBody = Fn(() => {
    // === Line-texture fetch (6 texels/segment) ===
    const lineBase: TSLNode = int(aSortedIndex).mul(int(6)).toVar();
    const lineTexW: TSLNode = int((textureSize(uLineTex, int(0)) as unknown as TSLNode).x).toVar();
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
    const culled: TSLNode = isOrtho
      ? float(0.0).greaterThan(1.0).toVar()
      : startDepth.lessThan(nearCull).and(endDepth.lessThan(nearCull)).toVar();

    // Near-plane segment clip (perspective only) — stencil AND domain.
    const mvA: TSLNode = mvStart.toVar();
    const mvB: TSLNode = mvEnd.toVar();
    if (!isOrtho) {
      If(startDepth.lessThan(nearCull).and(endDepth.greaterThanEqual(nearCull)), () => {
        const tA: TSLNode = nearCull.sub(startDepth).div(endDepth.sub(startDepth)).toVar();
        mvA.assign(mix(mvStart, mvEnd, tA));
      }).ElseIf(endDepth.lessThan(nearCull).and(startDepth.greaterThanEqual(nearCull)), () => {
        const tB: TSLNode = startDepth.sub(nearCull).div(startDepth.sub(endDepth)).toVar();
        mvB.assign(mix(mvStart, mvEnd, tB));
      });
    }

    const w0: TSLNode = sanitizeNonNegative(lineT0.w, 0.0);
    const w1: TSLNode = sanitizeNonNegative(lineT1.w, 0.0);
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
        ? w0.mul(uOrthoLineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
        : w0
            .mul(uPerspectiveLineScale)
            .mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
            .div(max(mvA.z.negate(), nearCull))
    ).toVar();
    const rawB: TSLNode = (
      isOrtho
        ? w1.mul(uOrthoLineScale).mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
        : w1
            .mul(uPerspectiveLineScale)
            .mul(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH)
            .div(max(mvB.z.negate(), nearCull))
    ).toVar();
    const rA: TSLNode = clamp(rawA, CAPSULE_MIN_RADIUS_PX, uMaxLinePixelWidth).toVar();
    const rB: TSLNode = clamp(rawB, CAPSULE_MIN_RADIUS_PX, uMaxLinePixelWidth).toVar();

    const interiorA: TSLNode = abs(lineT4.y)
      .greaterThan(0.5)
      .select(float(1.0), float(0.0))
      .toVar();
    const interiorB: TSLNode = abs(lineT4.z)
      .greaterThan(0.5)
      .select(float(1.0), float(0.0))
      .toVar();

    const ab: TSLNode = pB.sub(pA).toVar();
    const abLen: TSLNode = length(ab).toVar();
    const u: TSLNode = abLen
      .greaterThan(1e-4)
      .select(ab.div(abLen).toVar(), vec2(1.0, 0.0))
      .toVar();
    const v: TSLNode = vec2(u.y.negate(), u.x).toVar();
    const rMax: TSLNode = max(rA, rB).add(CAPSULE_STENCIL_APRON_PX).toVar();

    // Bisector cuts + fold-cap rule per end (mirrors the GLSL twin).
    const cutA: TSLNode = vec2(-1.0, 0.0).toVar();
    const cutB: TSLNode = vec2(1.0, 0.0).toVar();
    const extA: TSLNode = rMax.toVar();
    const extB: TSLNode = rMax.toVar();
    const capA: TSLNode = interiorA.toVar();
    const capB: TSLNode = interiorB.toVar();

    const partnerFarPx = (code: TSLNode): TSLNode => {
      // Decode the joint code to the partner's FAR endpoint, projected to
      // pixels; .z carries validity (0 when the code is a sentinel).
      const interior: TSLNode = code.greaterThan(0.5).or(code.lessThan(-2.5)).toVar();
      const slot: TSLNode = code
        .greaterThan(0.0)
        .select(int(code.add(0.5)).sub(int(1)), int(code.negate().add(0.5)).sub(int(3)))
        .toVar();
      const pBase: TSLNode = slot.mul(int(6)).toVar();
      const pt0: TSLNode = ivec2(pBase.mod(lineTexW), pBase.div(lineTexW)).toVar();
      const pStart: TSLNode = uLineTex.load(pt0).xyz.toVar();
      const pEnd: TSLNode = uLineTex.load(ivec2(pt0.x.add(int(1)), pt0.y)).xyz.toVar();
      const farObj: TSLNode = code.greaterThan(0.0).select(pEnd, pStart).toVar();
      const cl: TSLNode = cameraProjectionMatrix
        .mul(modelViewMatrix.mul(vec4(farObj, 1.0)))
        .toVar();
      const px: TSLNode = cl.xy
        .div(max(cl.w, float(1e-6)))
        .mul(0.5)
        .add(0.5)
        .mul(uResolution)
        .toVar();
      return vec3(px, interior.select(float(1.0), float(0.0))).toVar();
    };

    If(interiorA.greaterThan(0.5), () => {
      extA.assign(1.0);
      const far: TSLNode = partnerFarPx(lineT4.y);
      If(far.z.greaterThan(0.5), () => {
        const qq: TSLNode = far.xy.sub(pA).toVar();
        const ql: TSLNode = length(qq).toVar();
        If(ql.greaterThan(1e-4), () => {
          const qhat: TSLNode = qq.div(ql).toVar();
          If(
            dot(qhat, u)
              .greaterThan(CAPSULE_FOLD_CAP_MAX_COS)
              .and(rA.greaterThan(CAPSULE_FOLD_CAP_MIN_RADIUS_PX)),
            () => {
              // Fold-cap rule: round cap at a visibly-wide sharp fold.
              capA.assign(0.0);
              extA.assign(rMax);
            }
          ).Else(() => {
            const nRaw: TSLNode = qhat.sub(u).toVar(); // q − m, m = +u at A
            const nl: TSLNode = length(nRaw).toVar();
            If(nl.greaterThan(1e-3), () => {
              const n2: TSLNode = nRaw.div(nl).toVar();
              const nLoc: TSLNode = vec2(dot(n2, u), dot(n2, v)).toVar();
              If(nLoc.x.lessThan(-1e-3), () => {
                cutA.assign(nLoc);
                extA.assign(
                  clamp(abs(nLoc.y.div(nLoc.x)).mul(rMax), 0.0, rMax).add(CAPSULE_STENCIL_APRON_PX)
                );
              });
            });
          });
        });
      });
    });
    If(interiorB.greaterThan(0.5), () => {
      extB.assign(1.0);
      const far: TSLNode = partnerFarPx(lineT4.z);
      If(far.z.greaterThan(0.5), () => {
        const qq: TSLNode = far.xy.sub(pB).toVar();
        const ql: TSLNode = length(qq).toVar();
        If(ql.greaterThan(1e-4), () => {
          const qhat: TSLNode = qq.div(ql).toVar();
          If(
            dot(qhat, u)
              .lessThan(-CAPSULE_FOLD_CAP_MAX_COS)
              .and(rB.greaterThan(CAPSULE_FOLD_CAP_MIN_RADIUS_PX)),
            () => {
              capB.assign(0.0);
              extB.assign(rMax);
            }
          ).Else(() => {
            const nRaw: TSLNode = qhat.add(u).toVar(); // q − m, m = −u at B
            const nl: TSLNode = length(nRaw).toVar();
            If(nl.greaterThan(1e-3), () => {
              const n2: TSLNode = nRaw.div(nl).toVar();
              const nLoc: TSLNode = vec2(dot(n2, u), dot(n2, v)).toVar();
              If(nLoc.x.greaterThan(1e-3), () => {
                cutB.assign(nLoc);
                extB.assign(
                  clamp(abs(nLoc.y.div(nLoc.x)).mul(rMax), 0.0, rMax).add(CAPSULE_STENCIL_APRON_PX)
                );
              });
            });
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
    const rC: TSLNode = mix(rA, rB, tc).toVar();
    const rawC: TSLNode = mix(rawA, rawB, tc).toVar();
    const widthScale: TSLNode = min(rawC.div(CAPSULE_MIN_RADIUS_PX), 1.0).toVar();
    const fade: TSLNode = isOrtho
      ? float(1.0).toVar()
      : perspectiveNearFadeStaticTSL(false, mix(mvA.z, mvB.z, tc), nearCull).toVar();

    vLocal.assign(vec2(lx, ly));
    vMeta.assign(vec3(abLen, capA, capB));
    vCutA2.assign(cutA);
    vCutB2.assign(cutB);
    vInvR2.assign(float(1.0).div(rC.mul(rC)));
    vFade.assign(fade.mul(widthScale).mul(culled.select(float(0.0), float(1.0))));
    vAlphaV.assign(mix(sanitizeAlpha(lineT5.z), sanitizeAlpha(lineT5.w), tc));
    vSharp.assign(mix(s0, s1, tc));
    if (vScalar) vScalar.assign(mix(lineT5.x, lineT5.y, tc));
    if (vColor) vColor.assign(mix(vec3(lineT2), vec3(lineT3), tc));

    // Depth interpolates along the segment; culled segments collapse.
    const clipMix: TSLNode = mix(clipA, clipB, tc).toVar();
    const wMix: TSLNode = max(clipMix.w, float(1e-6)).toVar();
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
    const x: TSLNode = vLocal.x.toVar();
    const y: TSLNode = vLocal.y.toVar();
    // Bisector-cut sides (my side negative); straight joints = butt.
    Discard(vMeta.y.greaterThan(0.5).and(vCutA2.x.mul(x).add(vCutA2.y.mul(y)).greaterThan(0.0)));
    Discard(
      vMeta.z
        .greaterThan(0.5)
        .and(vCutB2.x.mul(x.sub(vMeta.x)).add(vCutB2.y.mul(y)).greaterThan(0.0))
    );
    // Squared distance in the local frame; cut ends are rods (no cap term).
    const oxA: TSLNode = vMeta.y.greaterThan(0.5).select(float(0.0), x.negate()).toVar();
    const oxB: TSLNode = vMeta.z.greaterThan(0.5).select(float(0.0), x.sub(vMeta.x)).toVar();
    const ox: TSLNode = max(max(oxA, oxB), 0.0).toVar();
    const q: TSLNode = y.mul(y).add(ox.mul(ox)).mul(vInvR2).toVar();
    const w: TSLNode = float(1.0).sub(q).toVar();
    Discard(w.lessThanEqual(0.0));

    // Quartic bump; the sharpness knob bends it off the default only.
    const profile: TSLNode = abs(vSharp.sub(0.5))
      .lessThan(1e-3)
      .select(w.mul(w).toVar(), pow(w, exp2(float(3.0).sub(vSharp.mul(4.0)))).toVar())
      .toVar();
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

  const opacityValue = (nodes.uOpacity.value as number | undefined) ?? 1.0;
  const blendingState = getCompleteBlendingState(blendingMode, opacityValue);
  applyBlendingStateToMaterial(material, blendingState);
  return material;
}
