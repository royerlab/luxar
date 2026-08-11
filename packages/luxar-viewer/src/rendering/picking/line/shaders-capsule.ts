/**
 * Capsule line PICKING shaders (GLSL) — issue #1352, behind
 * `?linePrimitive=capsule`.
 *
 * The pick pass rasterizes the SAME stencil the visual capsule draws
 * (half-disc bisector joints, near-plane clip) and shades it
 * with the SAME quartic profile, so the hover hotspot matches the pixels
 * exactly. Per the shared pick contract it drops the colour machinery and
 * adds the `uNodeId` uniform and `vNodeId` / `vElementId` varyings,
 * emitting `(nodeId, elementId-low16, brightness, elementId-high16)` with
 * brightness-as-depth (`gl_FragDepth = 1 − brightness`), identical to the
 * screen-space and volumetric pick variants. Like those, the pick pass
 * ignores per-element alpha and node opacity — faint-but-hoverable stays
 * consistent across primitives.
 *
 * Model + constants: `_shared/line-capsule.ts` (the visual twin's header
 * documents the exactness relaxations; they apply here identically).
 */
import {
  GLSL_LINE_JOINT_CODE,
  GLSL_NEAR_FADE_FUNCTIONS,
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_SORTED_INDEX,
} from '../../materials/_shared/glsl-lib';
import type { ShaderSource } from '../../materials/_shared/shader-source';
import { buildLinePickTSLNodesFromUniforms } from './pick.tsl';
import { capsuleLinePickWebGPUFactory } from './pick-capsule.tsl';
import {
  CAPSULE_MIN_RADIUS_PX,
  CAPSULE_RADIUS_PER_QUAD_HALFWIDTH,
  CAPSULE_STENCIL_APRON_PX,
} from '../../materials/_shared/line-capsule';

const G = {
  RADIUS_FACTOR: CAPSULE_RADIUS_PER_QUAD_HALFWIDTH.toFixed(7),
  MIN_RADIUS: CAPSULE_MIN_RADIUS_PX.toFixed(1),
  APRON: CAPSULE_STENCIL_APRON_PX.toFixed(1),
};

export const CAPSULE_LINE_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}
    ${GLSL_LINE_JOINT_CODE}

    in vec2 aQuadCorner;

    ${GLSL_SORTED_INDEX}

    uniform highp sampler2D uLineTex;
    uniform vec2 uResolution;
    uniform int uIsOrtho;
    uniform float uNodeId;
    uniform float uNearCull;
    uniform float uMaxLinePixelWidth;
    uniform float uPerspectiveLineScale;
    uniform float uOrthoLineScale;

    // Geometry varyings are screen-space quantities pre-multiplied by the
    // corner's clip w and divided by vW in the fragment (screen-linear;
    // see the visual twin's declaration note).
    out vec2 vLocal;
    flat out vec3 vMeta;
    flat out vec2 vCutA2;
    flat out vec2 vCutB2;
    flat out vec4 vJointA;
    flat out vec4 vJointB;
    flat out vec2 vREnd;
    out float vR;
    out float vW;
    out float vFade;
    out float vSharp;
    flat out highp float vNodeId;
    flat out highp vec2 vElementId;

    vec4 luxarPartnerFar(float code, int lineTexW) {
      bool interior = (code > 0.5) || (code < -2.5);
      if (!interior) return vec4(0.0, 0.0, 0.0, -1.0);
      int slot = (code > 0.0) ? int(code + 0.5) - 1 : int(-code + 0.5) - 3;
      int pBase = slot * 6;
      ivec2 pt0 = ivec2(pBase % lineTexW, pBase / lineTexW);
      vec4 pStart = texelFetch(uLineTex, pt0, 0);
      vec4 pEnd = texelFetch(uLineTex, ivec2(pt0.x + 1, pt0.y), 0);
      vec4 far = (code > 0.0) ? pEnd : pStart;
      return vec4(far.xyz, sanitizeNonNegative(far.w, 0.0));
    }

    // Project a VIEW-space point to pixel coordinates. Callers must
    // near-clip the point first — a behind-eye w flips the projection.
    vec2 luxarViewToPx(vec4 mv) {
      vec4 cl = projectionMatrix * mv;
      return (cl.xy / max(cl.w, 1e-6) * 0.5 + 0.5) * uResolution;
    }

    void main() {
      vNodeId = uNodeId;
      vElementId = luxarElementIdParts();

      int lineBase = int(luxarSortedIndex()) * 6;
      int lineTexW = textureSize(uLineTex, 0).x;
      ivec2 texel0 = ivec2(lineBase % lineTexW, lineBase / lineTexW);
      vec4 lineT0 = texelFetch(uLineTex, texel0, 0);
      vec4 lineT1 = texelFetch(uLineTex, ivec2(texel0.x + 1, texel0.y), 0);
      vec4 lineT4 = texelFetch(uLineTex, ivec2(texel0.x + 4, texel0.y), 0);

      vec4 mvStart = modelViewMatrix * vec4(lineT0.xyz, 1.0);
      vec4 mvEnd = modelViewMatrix * vec4(lineT1.xyz, 1.0);

      float nearCull = max(uNearCull, 1e-20);
      float startDepth = -mvStart.z;
      float endDepth = -mvEnd.z;
      if ((uIsOrtho == 0) && startDepth < nearCull && endDepth < nearCull) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        vLocal = vec2(0.0); vMeta = vec3(1.0, 0.0, 0.0);
        vCutA2 = vec2(-1.0, 0.0); vCutB2 = vec2(1.0, 0.0);
        vJointA = vec4(0.0); vJointB = vec4(0.0); vREnd = vec2(1.0);
        vR = 1.0; vW = 1.0; vFade = 0.0; vSharp = 0.5;
        return;
      }
      float tA = 0.0;
      float tB = 1.0;
      if (uIsOrtho == 0) {
        if (startDepth < nearCull && endDepth >= nearCull) {
          tA = (nearCull - startDepth) / (endDepth - startDepth);
        } else if (endDepth < nearCull && startDepth >= nearCull) {
          tB = (startDepth - nearCull) / (startDepth - endDepth);
        }
        vec4 a = mix(mvStart, mvEnd, tA);
        vec4 b = mix(mvStart, mvEnd, tB);
        mvStart = a; mvEnd = b;
      }

      vec4 lineT2 = texelFetch(uLineTex, ivec2(texel0.x + 2, texel0.y), 0);
      vec4 lineT3 = texelFetch(uLineTex, ivec2(texel0.x + 3, texel0.y), 0);

      float w0 = sanitizeNonNegative(lineT0.w, 0.0);
      float w1 = sanitizeNonNegative(lineT1.w, 0.0);
      // Endpoint attributes at the CLIPPED span (mirrors the visual twin).
      float wEffA = mix(w0, w1, tA);
      float wEffB = mix(w0, w1, tB);
      float s0 = clamp(sanitizeNonNegative(lineT2.w, 0.5), 0.0, 1.0);
      float s1 = clamp(sanitizeNonNegative(lineT3.w, 0.5), 0.0, 1.0);

      vec4 clipA = projectionMatrix * mvStart;
      vec4 clipB = projectionMatrix * mvEnd;
      float wA = max(clipA.w, 1e-6);
      float wB = max(clipB.w, 1e-6);
      vec2 pA = (clipA.xy / wA * 0.5 + 0.5) * uResolution;
      vec2 pB = (clipB.xy / wB * 0.5 + 0.5) * uResolution;

      float rawA;
      float rawB;
      if (uIsOrtho == 1) {
        rawA = wEffA * uOrthoLineScale * ${G.RADIUS_FACTOR};
        rawB = wEffB * uOrthoLineScale * ${G.RADIUS_FACTOR};
      } else {
        rawA = wEffA * uPerspectiveLineScale * ${G.RADIUS_FACTOR} / max(-mvStart.z, nearCull);
        rawB = wEffB * uPerspectiveLineScale * ${G.RADIUS_FACTOR} / max(-mvEnd.z, nearCull);
      }
      float rA = clamp(rawA, ${G.MIN_RADIUS}, uMaxLinePixelWidth);
      float rB = clamp(rawB, ${G.MIN_RADIUS}, uMaxLinePixelWidth);

      // Which ends cut rather than cap — the shared joint-code rule, same
      // as the visual twin (a free end and a degree->=3 hub keep the whole
      // round cap; everything else cuts).
      float interiorA = luxarLineJointCapSuppression(lineT4.y);
      float interiorB = luxarLineJointCapSuppression(lineT4.z);

      vec2 ab = pB - pA;
      float abLen = length(ab);
      vec2 u = abLen > 1e-4 ? ab / abLen : vec2(1.0, 0.0);
      vec2 v = vec2(-u.y, u.x);
      float rMax = max(rA, rB) + ${G.APRON};
      vec2 cutA = vec2(-1.0, 0.0);
      vec2 cutB = vec2(1.0, 0.0);
      float extA = rMax;
      float extB = rMax;
      vJointA = vec4(0.0);
      vJointB = vec4(0.0);
      vREnd = vec2(rA, rB);
      if (interiorA > 0.5) {
        extA = ${G.APRON};
        // Mirrors the visual capsule exactly (or hover desyncs from
        // pixels): behind-near joint keeps the butt; the partner's far
        // endpoint is near-clipped toward the joint vertex first.
        vec4 farA = tA > 0.0 ? vec4(0.0, 0.0, 0.0, -1.0) : luxarPartnerFar(lineT4.y, lineTexW);
        if (farA.w >= 0.0) {
          vec4 mvFarA = modelViewMatrix * vec4(farA.xyz, 1.0);
          float farDepthA = -mvFarA.z;
          if (uIsOrtho == 0 && farDepthA < nearCull) {
            float tF = (startDepth - nearCull) / max(startDepth - farDepthA, 1e-20);
            mvFarA = mix(mvStart, mvFarA, clamp(tF, 0.0, 1.0));
          }
          vec2 qq = luxarViewToPx(mvFarA) - pA;
          float ql = length(qq);
          if (ql > 1e-4) {
            vec2 nRaw = qq / ql - u;   // q − m, m = +u at A
            float nl = length(nRaw);
            if (nl > 1e-3) {
              vec2 n2 = nRaw / nl;
              vec2 nLoc = vec2(dot(n2, u), dot(n2, v));
              if (nLoc.x < -1e-3) {
                cutA = nLoc;
                float wFarA = farA.w;
                float rpFarA;
                if (uIsOrtho == 1) {
                  rpFarA = wFarA * uOrthoLineScale * ${G.RADIUS_FACTOR};
                } else {
                  rpFarA = wFarA * uPerspectiveLineScale * ${G.RADIUS_FACTOR} / max(-mvFarA.z, nearCull);
                }
                rpFarA = clamp(rpFarA, ${G.MIN_RADIUS}, uMaxLinePixelWidth);
                vJointA = vec4(dot(qq / ql, u), dot(qq / ql, v), ql, rpFarA);
                float deficitA = clamp(1.0 - rpFarA / max(rA, 1e-4), 0.0, 1.0);
                extA = max(abs(nLoc.y), deficitA) * rMax + ${G.APRON};
              }
            } else {
              // Near-hairpin: the bisector is degenerate — plain round cap
              // (the partner nearly coincides; overlap is unavoidable).
              interiorA = 0.0;
              extA = rMax;
            }
          }
        }
      }
      if (interiorB > 0.5) {
        extB = ${G.APRON};
        vec4 farB = tB < 1.0 ? vec4(0.0, 0.0, 0.0, -1.0) : luxarPartnerFar(lineT4.z, lineTexW);
        if (farB.w >= 0.0) {
          vec4 mvFarB = modelViewMatrix * vec4(farB.xyz, 1.0);
          float farDepthB = -mvFarB.z;
          if (uIsOrtho == 0 && farDepthB < nearCull) {
            float tF = (endDepth - nearCull) / max(endDepth - farDepthB, 1e-20);
            mvFarB = mix(mvEnd, mvFarB, clamp(tF, 0.0, 1.0));
          }
          vec2 qq = luxarViewToPx(mvFarB) - pB;
          float ql = length(qq);
          if (ql > 1e-4) {
            vec2 nRaw = qq / ql + u;   // q − m, m = −u at B
            float nl = length(nRaw);
            if (nl > 1e-3) {
              vec2 n2 = nRaw / nl;
              vec2 nLoc = vec2(dot(n2, u), dot(n2, v));
              if (nLoc.x > 1e-3) {
                cutB = nLoc;
                float wFarB = farB.w;
                float rpFarB;
                if (uIsOrtho == 1) {
                  rpFarB = wFarB * uOrthoLineScale * ${G.RADIUS_FACTOR};
                } else {
                  rpFarB = wFarB * uPerspectiveLineScale * ${G.RADIUS_FACTOR} / max(-mvFarB.z, nearCull);
                }
                rpFarB = clamp(rpFarB, ${G.MIN_RADIUS}, uMaxLinePixelWidth);
                vJointB = vec4(dot(qq / ql, u), dot(qq / ql, v), ql, rpFarB);
                float deficitB = clamp(1.0 - rpFarB / max(rB, 1e-4), 0.0, 1.0);
                extB = max(abs(nLoc.y), deficitB) * rMax + ${G.APRON};
              }
            } else {
              // Near-hairpin (see end A).
              interiorB = 0.0;
              extB = rMax;
            }
          }
        }
      }
      vCutA2 = cutA;
      vCutB2 = cutB;
      vMeta = vec3(abLen, interiorA, interiorB);

      float lx = aQuadCorner.x > 0.0 ? abLen + extB : -extA;
      float ly = aQuadCorner.y * rMax;
      vec2 corner = pA + u * lx + v * ly;
      float tc = abLen > 1e-4 ? clamp(lx / abLen, 0.0, 1.0) : 0.5;
      float tOrig = mix(tA, tB, tc);
      // The RADIUS interpolates linearly in screen space (1/depth is
      // perspective-linear, so a constant-width tube's pixel radius is
      // exactly linear in screen x). Interpolating 1/r² instead bends the
      // rim quadratically inward — a hard concave silhouette at strong
      // taper (the zoomed near-axial case).
      vR = mix(rA, rB, tc);
      float rawC = mix(rawA, rawB, tc);
      float widthScale = min(rawC / ${G.MIN_RADIUS}, 1.0);
      vFade = perspectiveNearFade(uIsOrtho, mix(mvStart.z, mvEnd.z, tc), nearCull) * widthScale;
      vSharp = mix(s0, s1, tOrig);

      vec4 clipMix = mix(clipA, clipB, tc);
      float wMix = max(clipMix.w, 1e-6);
      vLocal = vec2(lx, ly) * wMix;
      vR = vR * wMix;
      vW = wMix;
      vec2 ndc = corner / uResolution * 2.0 - 1.0;
      gl_Position = vec4(ndc * wMix, clipMix.z, wMix);
    }
`;

export const CAPSULE_LINE_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    in vec2 vLocal;
    flat in vec3 vMeta;
    flat in vec2 vCutA2;
    flat in vec2 vCutB2;
    flat in vec4 vJointA;
    flat in vec4 vJointB;
    flat in vec2 vREnd;
    in float vR;
    in float vW;
    in float vFade;
    in float vSharp;
    flat in highp float vNodeId;
    flat in highp vec2 vElementId;

    out vec4 fragColor;

    // The PARTNER leg's profile at a pixel offset rel from the shared
    // vertex — mirrors the visual capsule exactly (hover must track pixels).
    float luxarPartnerProfile(vec2 rel, vec4 joint, float rEnd, float sharp) {
      float xp = dot(rel, joint.xy);
      float yp2 = max(dot(rel, rel) - xp * xp, 0.0);
      float tp = clamp(xp / max(joint.z, 1e-4), 0.0, 1.0);
      float rp = max(mix(rEnd, joint.w, tp), 1e-4);
      float op = max(max(-xp, xp - joint.z), 0.0);
      float qp = (yp2 + op * op) / (rp * rp);
      float wp = 1.0 - qp;
      if (wp <= 0.0) return 0.0;
      return (abs(sharp - 0.5) < 1e-3) ? wp * wp : pow(wp, exp2(3.0 - 4.0 * sharp));
    }

    void main() {
      float invW = 1.0 / max(vW, 1e-9);
      float x = vLocal.x * invW;
      float y = vLocal.y * invW;
      float rPx = max(vR * invW, 1e-4);
      // TRUE point-to-segment distance: every end is capped (a free end
      // keeps the whole disc, a cut end its half of the joint disc).
      float ox = max(max(-x, x - vMeta.x), 0.0);
      float q = (y * y + ox * ox) / (rPx * rPx);
      float w = 1.0 - q;
      if (w <= 0.0) discard;

      float profile = (abs(vSharp - 0.5) < 1e-3)
        ? w * w
        : pow(w, exp2(3.0 - 4.0 * vSharp));

      // Joint DEFICIT rule (mirrors the visual capsule; see its note).
      if (vMeta.y > 0.5 && (vCutA2.x * x + vCutA2.y * y) > 0.0) {
        if (vJointA.z < 0.5) discard;
        profile -= luxarPartnerProfile(vec2(x, y), vJointA, vREnd.x, vSharp);
        if (profile <= 0.0) discard;
      }
      if (vMeta.z > 0.5 && (vCutB2.x * (x - vMeta.x) + vCutB2.y * y) > 0.0) {
        if (vJointB.z < 0.5) discard;
        profile -= luxarPartnerProfile(vec2(x - vMeta.x, y), vJointB, vREnd.y, vSharp);
        if (profile <= 0.0) discard;
      }
      float brightness = profile * vFade;
      if (brightness < 1e-4) discard;

      fragColor = vec4(vNodeId, vElementId.x, brightness, vElementId.y);
      gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0);
    }
`;

export const CAPSULE_LINE_PICK_SOURCE: ShaderSource = {
  name: 'line-pick-capsule',
  webgl: {
    vertex: CAPSULE_LINE_PICK_VERTEX_SHADER,
    fragment: CAPSULE_LINE_PICK_FRAGMENT_SHADER,
  },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    const isOrtho = ((u.uIsOrtho?.value as number) ?? 0) === 1;
    return capsuleLinePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(u), { isOrtho });
  },
};
