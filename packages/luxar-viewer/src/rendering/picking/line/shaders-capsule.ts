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
  CAPSULE_JOINT_DEFICIT_GATE,
  CAPSULE_JOINT_PACKET_MIN_RADIUS_PX,
  CAPSULE_MIN_RADIUS_PX,
  CAPSULE_RADIUS_PER_QUAD_HALFWIDTH,
  CAPSULE_STENCIL_APRON_PX,
} from '../../materials/_shared/line-capsule';

const G = {
  RADIUS_FACTOR: CAPSULE_RADIUS_PER_QUAD_HALFWIDTH.toFixed(7),
  MIN_RADIUS: CAPSULE_MIN_RADIUS_PX.toFixed(1),
  DEFICIT_GATE: CAPSULE_JOINT_DEFICIT_GATE.toFixed(2),
  PACKET_MIN_R: CAPSULE_JOINT_PACKET_MIN_RADIUS_PX.toFixed(1),
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
    out vec3 vLocal;
    // (abLen px, capFlags = interiorA + 2·interiorB, rA px, rB px)
    flat out vec4 vMeta;
    // .xy = bisector-cut normal; .z = partner radius gradient packet
    // (mirrors the visual capsule exactly).
    flat out vec4 vCutA2;
    flat out vec4 vCutB2;
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
        vLocal = vec3(0.0); vMeta = vec4(1.0, 0.0, 1.0, 1.0);
        vCutA2 = vec4(-1.0, 0.0, 0.0, 0.0); vCutB2 = vec4(1.0, 0.0, 0.0, 0.0);
        vW = 1.0; vFade = 0.0; vSharp = 0.5;
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
      vec4 cutA = vec4(-1.0, 0.0, 0.0, 0.0);
      vec4 cutB = vec4(1.0, 0.0, 0.0, 0.0);
      float extA = rMax;
      float extB = rMax;
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
              // Snap the cut normal to a 1/1024 grid. NOTE the honest
              // rationale: each leg snaps in its OWN (u, v) basis, so the
              // two planes still disagree by up to ~1e-3 rad — the AA ramp
              // is what actually kills the boundary speckle; the snap just
              // keeps the residual plane disagreement ≲0.05 px of the 1 px
              // ramp.
              nLoc = round(nLoc * 1024.0) / 1024.0;
              if (nLoc.x < -1e-3) {
                cutA = vec4(nLoc, 0.0, 0.0);
                if (rMax > ${G.PACKET_MIN_R}) {
                  float wFarA = farA.w;
                  float rpFarA;
                  if (uIsOrtho == 1) {
                    rpFarA = wFarA * uOrthoLineScale * ${G.RADIUS_FACTOR};
                  } else {
                    rpFarA = wFarA * uPerspectiveLineScale * ${G.RADIUS_FACTOR} / max(-mvFarA.z, nearCull);
                  }
                  rpFarA = clamp(rpFarA, ${G.MIN_RADIUS}, uMaxLinePixelWidth);
                  float deficitA = clamp(1.0 - rpFarA / max(rA, 1e-4), 0.0, 1.0);
                  if (deficitA > ${G.DEFICIT_GATE}) {
                    cutA.z = (rpFarA - rA) / ql;
                    cutA.w = ql;
                    // The deficit term's support is bounded by MY OWN
                    // capsule (it renders max(mine − partner, 0) ≤ mine),
                    // so the full disc reach covers it at any partner
                    // length or taper (#1488).
                    extA = rMax + ${G.APRON};
                  } else {
                    extA = abs(nLoc.y) * rMax + ${G.APRON};
                  }
                } else {
                  extA = abs(nLoc.y) * rMax + ${G.APRON};
                }
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
              nLoc = round(nLoc * 1024.0) / 1024.0;
              if (nLoc.x > 1e-3) {
                cutB = vec4(nLoc, 0.0, 0.0);
                if (rMax > ${G.PACKET_MIN_R}) {
                  float wFarB = farB.w;
                  float rpFarB;
                  if (uIsOrtho == 1) {
                    rpFarB = wFarB * uOrthoLineScale * ${G.RADIUS_FACTOR};
                  } else {
                    rpFarB = wFarB * uPerspectiveLineScale * ${G.RADIUS_FACTOR} / max(-mvFarB.z, nearCull);
                  }
                  rpFarB = clamp(rpFarB, ${G.MIN_RADIUS}, uMaxLinePixelWidth);
                  float deficitB = clamp(1.0 - rpFarB / max(rB, 1e-4), 0.0, 1.0);
                  if (deficitB > ${G.DEFICIT_GATE}) {
                    cutB.z = (rpFarB - rB) / ql;
                    cutB.w = ql;
                    // The deficit term's support is bounded by MY OWN
                    // capsule (it renders max(mine − partner, 0) ≤ mine),
                    // so the full disc reach covers it at any partner
                    // length or taper (#1488).
                    extB = rMax + ${G.APRON};
                  } else {
                    extB = abs(nLoc.y) * rMax + ${G.APRON};
                  }
                } else {
                  extB = abs(nLoc.y) * rMax + ${G.APRON};
                }
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
      vMeta = vec4(abLen, interiorA + 2.0 * interiorB, rA, rB);

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
      float rawC = mix(rawA, rawB, tc);
      float widthScale = min(rawC / ${G.MIN_RADIUS}, 1.0);
      vFade = perspectiveNearFade(uIsOrtho, mix(mvStart.z, mvEnd.z, tc), nearCull) * widthScale;
      vSharp = mix(s0, s1, tOrig);

      vec4 clipMix = mix(clipA, clipB, tc);
      float wMix = max(clipMix.w, 1e-6);
      vLocal = vec3(lx, ly, 1.0 / max(abLen, 1e-4)) * wMix;
      vW = wMix;
      vec2 ndc = corner / uResolution * 2.0 - 1.0;
      gl_Position = vec4(ndc * wMix, clipMix.z, wMix);
    }
`;

export const CAPSULE_LINE_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    in vec3 vLocal;
    flat in vec4 vMeta;
    flat in vec4 vCutA2;
    flat in vec4 vCutB2;
    in float vW;
    in float vFade;
    in float vSharp;
    flat in highp float vNodeId;
    flat in highp vec2 vElementId;

    out vec4 fragColor;

    // The PARTNER leg's profile at a pixel offset rel from the shared
    // vertex — mirrors the visual capsule exactly (hover must track pixels).
    float luxarPartnerProfile(vec4 cut, vec2 rel, float mSign, float rEnd, float sharp) {
      vec2 n = cut.xy;
      // Partner axis = my inward axis reflected across the cut plane
      // (q = m − 2(m·n)n — exact; both cut normals are normalize(q − m)
      // up to sign).
      vec2 qdir = vec2(mSign - 2.0 * (mSign * n.x) * n.x, -2.0 * (mSign * n.x) * n.y);
      float xp = dot(rel, qdir);
      float yp2 = max(dot(rel, rel) - xp * xp, 0.0);
      // Radius from the SHARED VERTEX radius (rEnd, #1494) tapered by
      // the packed gradient, FROZEN past the partner's far end; the far
      // cap term closes the rod there (#1490).
      float rp = max(rEnd + cut.z * clamp(xp, 0.0, cut.w), 1e-4);
      float op = max(max(-xp, xp - cut.w), 0.0);
      float qp = (yp2 + op * op) / (rp * rp);
      float wp = 1.0 - qp;
      if (wp <= 0.0) return 0.0;
      return (abs(sharp - 0.5) < 1e-3) ? wp * wp : pow(wp, exp2(3.0 - 4.0 * sharp));
    }

    void main() {
      float invW = 1.0 / max(vW, 1e-9);
      float x = vLocal.x * invW;
      float y = vLocal.y * invW;
      // EXACT per-fragment radius: mix of the endpoint radii clamped to
      // the segment span — a varying cannot represent this (its linear
      // interpolation spans the cap extensions, so a short stub's drawn
      // radius at its own endpoint drifts; the deeper root of #1494).
      float cutFlagA = mod(vMeta.y, 2.0);
      float cutFlagB = vMeta.y >= 2.0 ? 1.0 : 0.0;
      // EXACT per-fragment radius without a divide: 1/abLen rides the
      // vLocal lane, and constant-width segments (the fill/thin-heavy
      // cases) take a mix-free fast path.
      float rPx;
      if (vMeta.z == vMeta.w) {
        rPx = max(vMeta.z, 1e-4);
      } else {
        rPx = max(mix(vMeta.z, vMeta.w, clamp(x * (vLocal.z * invW), 0.0, 1.0)), 1e-4);
      }
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
      // The cut is a 1 px AA RAMP, not a hard step: each leg's fragment
      // evaluates the plane in its OWN local frame, so pixels within float
      // noise of the line would otherwise flip independently — sprinkling
      // black (both discard) and double-bright (both keep) speckles along
      // every joint. Complementary ramps sum to exactly 1 instead, and
      // anti-alias the cut for free. The DEFICIT term blends in over the
      // same ramp: full profile on my side, max(mine − partner, 0) beyond.
      if (cutFlagA > 0.5) {
        float sideA = vCutA2.x * x + vCutA2.y * y;
        if (sideA > -0.5) {
          float coverA = clamp(0.5 - sideA, 0.0, 1.0);
          float defA = 0.0;
          if (vCutA2.w > 0.0) {
            defA = max(
              profile - luxarPartnerProfile(vCutA2, vec2(x, y), 1.0, vMeta.z, vSharp),
              0.0
            );
          }
          profile = profile * coverA + defA * (1.0 - coverA);
          if (profile <= 0.0) discard;
        }
      }
      if (cutFlagB > 0.5) {
        float sideB = vCutB2.x * (x - vMeta.x) + vCutB2.y * y;
        if (sideB > -0.5) {
          float coverB = clamp(0.5 - sideB, 0.0, 1.0);
          float defB = 0.0;
          if (vCutB2.w > 0.0) {
            defB = max(
              profile - luxarPartnerProfile(vCutB2, vec2(x - vMeta.x, y), -1.0, vMeta.w, vSharp),
              0.0
            );
          }
          profile = profile * coverB + defB * (1.0 - coverB);
          if (profile <= 0.0) discard;
        }
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
