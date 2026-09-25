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
 * screen-space pick variant. Like it, the pick pass
 * ignores per-element alpha and node opacity — faint-but-hoverable stays
 * consistent across primitives.
 *
 * Model + constants: `_shared/line-capsule.ts` (the visual twin's header
 * documents the exactness relaxations; they apply here identically).
 */
import {
  GLSL_LINE_JOINT_CODE,
  GLSL_NEAR_FADE_FUNCTIONS,
  GLSL_PROJECTION_FUNCTIONS,
  GLSL_LINE_SCALE,
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_SORTED_INDEX,
} from '../../materials/_shared/glsl-lib';
import type { ShaderSource } from '../../materials/_shared/shader-source';
import {
  CAPSULE_JOINT_DEFICIT_GATE,
  CAPSULE_JOINT_PACKET_MIN_RADIUS_PX,
  CAPSULE_MIN_RADIUS_PX,
  CAPSULE_RADIUS_PER_QUAD_HALFWIDTH,
  CAPSULE_STENCIL_APRON_PX,
} from '../../materials/_shared/line-capsule';
import { requireTslMaterials } from '../../tsl/slot';

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
    ${GLSL_PROJECTION_FUNCTIONS}
    ${GLSL_LINE_SCALE}
    ${GLSL_LINE_JOINT_CODE}

    in vec2 aQuadCorner;

    ${GLSL_SORTED_INDEX}

    uniform highp sampler2D uLineTex;
    uniform vec2 uResolution;
    uniform float uPixelRatio;
    uniform int uIsOrtho;
    uniform float uNodeId;
    uniform float uNearCull;
    uniform float uMaxLinePixelWidth;
    uniform float uPerspectiveLineScale;
    uniform float uOrthoLineScale;

    // Geometry varyings are screen-space quantities pre-multiplied by the
    // corner's clip w and multiplied by gl_FragCoord.w (= 1/w) in the
    // fragment (screen-linear; see the visual twin's declaration note).
    out vec2 vLocal;
    // PACKED joint state — mirrors the visual capsule exactly (see
    // shader-glsl-capsule.ts: normals full precision per #1502, halves
    // for packets/radii/invLen/flags).
    flat out vec4 vCutN;    // (nAx, nAy, nBx, nBy)
    flat out uvec4 vPack;   // (gradA,qlA) (gradB,qlB) (rA,rB) (1/abLen,flags)
    flat out float vAbLen;
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
      // Line pixel-width scale for this draw: resY * |P11| (glsl-lib GLSL_LINE_SCALE).
      luxarLineScale = uResolution.y * luxarProjectionSizeScale();
      vNodeId = uNodeId;
      vElementId = luxarElementIdParts();

      int lineBase = int(luxarSortedIndex()) * 6;
      int lineTexW = LUXAR_LINE_TEX_W;
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
        vLocal = vec2(0.0);
        vCutN = vec4(-1.0, 0.0, 1.0, 0.0);
        vPack = uvec4(0u, 0u, packHalf2x16(vec2(1.0, 1.0)), packHalf2x16(vec2(1.0, 0.0)));
        vAbLen = 1.0;
        vFade = 0.0; vSharp = 0.5;
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
        rawA = wEffA * luxarLineScale * ${G.RADIUS_FACTOR};
        rawB = wEffB * luxarLineScale * ${G.RADIUS_FACTOR};
      } else {
        rawA = wEffA * luxarLineScale * ${G.RADIUS_FACTOR} / max(-mvStart.z, nearCull);
        rawB = wEffB * luxarLineScale * ${G.RADIUS_FACTOR} / max(-mvEnd.z, nearCull);
      }
      float appearancePixelRatio = max(uPixelRatio, 1.0);
      float minRadius = ${G.MIN_RADIUS} * appearancePixelRatio;
      float packetMinRadius = ${G.PACKET_MIN_R} * appearancePixelRatio;
      float rA = clamp(rawA, minRadius, uMaxLinePixelWidth);
      float rB = clamp(rawB, minRadius, uMaxLinePixelWidth);

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
      // The joint partition is NOT width-gated — the pick shape must stay
      // the visual shape, and the drawn radius is floored at the AA
      // minimum (see the visual twin's note).
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
              // Cut normal: keep full precision — no per-leg quantisation
              // (see _shared/line-capsule.ts's note; #1502).
              vec2 n2 = nRaw / nl;
              vec2 nLoc = vec2(dot(n2, u), dot(n2, v));
              if (nLoc.x < -1e-3) {
                cutA = vec4(nLoc, 0.0, 0.0);
                // Width gate + its floored sharp-turn exception (#1495),
                // exactly as the visual twin (or hover desyncs from pixels;
                // the floor conjunct is why — see that note).
                if (rMax > packetMinRadius ||
                    (dot(qq / ql, u) > 0.5 && min(rawA, rawB) >= minRadius)) {
                  float wFarA = farA.w;
                  float rpFarA;
                  if (uIsOrtho == 1) {
                    rpFarA = wFarA * luxarLineScale * ${G.RADIUS_FACTOR};
                  } else {
                    rpFarA = wFarA * luxarLineScale * ${G.RADIUS_FACTOR} / max(-mvFarA.z, nearCull);
                  }
                  rpFarA = clamp(rpFarA, minRadius, uMaxLinePixelWidth);
                  // Packet gate (#1495, #1501): a hard cut is only exact
                  // when the partner actually covers my foreign side —
                  // which fails whenever EITHER leg tapers (both
                  // directions), the partner is short relative to the
                  // joint disc, or the turn nears a hairpin: there the
                  // bisector tilts toward my axis and splits my rod
                  // LENGTHWISE, so a partner longer than the disc but
                  // shorter than my leg refills only part of the cut
                  // half (measured to −0.92 of peak in the 2r–3r
                  // partner band without the angle clause).
                  bool needPacketA =
                    abs(1.0 - rpFarA / max(rA, 1e-4)) > ${G.DEFICIT_GATE} ||
                    rB > rA * (1.0 + ${G.DEFICIT_GATE}) ||
                    ql < 2.0 * rA ||
                    dot(qq / ql, u) > 0.5;
                  if (needPacketA) {
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
              if (nLoc.x > 1e-3) {
                cutB = vec4(nLoc, 0.0, 0.0);
                // Width gate + its floored sharp-turn exception (see end A).
                if (rMax > packetMinRadius ||
                    (dot(qq / ql, u) < -0.5 && min(rawA, rawB) >= minRadius)) {
                  float wFarB = farB.w;
                  float rpFarB;
                  if (uIsOrtho == 1) {
                    rpFarB = wFarB * luxarLineScale * ${G.RADIUS_FACTOR};
                  } else {
                    rpFarB = wFarB * luxarLineScale * ${G.RADIUS_FACTOR} / max(-mvFarB.z, nearCull);
                  }
                  rpFarB = clamp(rpFarB, minRadius, uMaxLinePixelWidth);
                  // Packet gate (#1495, #1501): a hard cut is only exact
                  // when the partner actually covers my foreign side —
                  // which fails whenever EITHER leg tapers (both
                  // directions), the partner is short relative to the
                  // joint disc, or the turn nears a hairpin: there the
                  // bisector tilts toward my axis and splits my rod
                  // LENGTHWISE, so a partner longer than the disc but
                  // shorter than my leg refills only part of the cut
                  // half (measured to −0.92 of peak in the 2r–3r
                  // partner band without the angle clause).
                  bool needPacketB =
                    abs(1.0 - rpFarB / max(rB, 1e-4)) > ${G.DEFICIT_GATE} ||
                    rA > rB * (1.0 + ${G.DEFICIT_GATE}) ||
                    ql < 2.0 * rB ||
                    dot(qq / ql, u) < -0.5;
                  if (needPacketB) {
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
      vCutN = vec4(cutA.xy, cutB.xy);
      vPack = uvec4(
        packHalf2x16(cutA.zw),
        packHalf2x16(cutB.zw),
        packHalf2x16(vec2(rA, rB)),
        packHalf2x16(vec2(1.0 / max(abLen, 1e-4), interiorA + 2.0 * interiorB))
      );
      vAbLen = abLen;

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
      float widthScale = min(rawC / minRadius, 1.0);
      vFade = perspectiveNearFade(uIsOrtho, mix(mvStart.z, mvEnd.z, tc), nearCull) * widthScale;
      vSharp = mix(s0, s1, tOrig);

      vec4 clipMix = mix(clipA, clipB, tc);
      float wMix = max(clipMix.w, 1e-6);
      vLocal = vec2(lx, ly) * wMix;
      vec2 ndc = corner / uResolution * 2.0 - 1.0;
      gl_Position = vec4(ndc * wMix, clipMix.z, wMix);
    }
`;

export const CAPSULE_LINE_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    in vec2 vLocal;
    flat in vec4 vCutN;
    flat in uvec4 vPack;
    flat in float vAbLen;
    in float vFade;
    in float vSharp;
    flat in highp float vNodeId;
    flat in highp vec2 vElementId;

    out vec4 fragColor;

    // The PARTNER leg's tapered-capsule field at a pixel offset rel from
    // the shared vertex (my local frame). Its axis is my inward axis
    // reflected across the cut plane (q = m − 2(m·n)n — exact); its
    // radius starts at rEnd, the SHARED-VERTEX radius handed in by the
    // caller from the packed vPack.z lane (#1494); it tapers by the
    // packed gradient, freezes past its far end,
    // and the far cap term closes the rod there (#1490). Sharpness is
    // taken from OUR fragment — the joint region is local.
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
      // gl_FragCoord.w IS the perspective-interpolated 1/w (spec identity
      // — PC-interp of a per-vertex w equals 1/gl_FragCoord.w exactly), so
      // the vW varying and its per-fragment divide are both unnecessary.
      float invW = gl_FragCoord.w;
      float x = vLocal.x * invW;
      float y = vLocal.y * invW;
      // Unpack the flat joint state (see the visual twin).
      vec2 pkA = unpackHalf2x16(vPack.x);
      vec2 pkB = unpackHalf2x16(vPack.y);
      vec2 pkR = unpackHalf2x16(vPack.z);
      vec2 pkM = unpackHalf2x16(vPack.w);
      float cutFlagA = mod(pkM.y, 2.0);
      float cutFlagB = pkM.y >= 2.0 ? 1.0 : 0.0;
      float rPx;
      if (pkR.x == pkR.y) {
        rPx = max(pkR.x, 1e-4);
      } else {
        rPx = max(mix(pkR.x, pkR.y, clamp(x * pkM.x, 0.0, 1.0)), 1e-4);
      }
      // TRUE point-to-segment distance: every end is capped (a free end
      // keeps the whole disc, a cut end its half of the joint disc).
      float ox = max(max(-x, x - vAbLen), 0.0);
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
        float sideA = vCutN.x * x + vCutN.y * y;
        if (sideA > -0.5) {
          float coverA = clamp(0.5 - sideA, 0.0, 1.0);
          float defA = 0.0;
          if (pkA.y > 0.0) {
            defA = max(
              profile - luxarPartnerProfile(vec4(vCutN.xy, pkA), vec2(x, y), 1.0, pkR.x, vSharp),
              0.0
            );
          }
          profile = profile * coverA + defA * (1.0 - coverA);
          if (profile <= 0.0) discard;
        }
      }
      if (cutFlagB > 0.5) {
        float sideB = vCutN.z * (x - vAbLen) + vCutN.w * y;
        if (sideB > -0.5) {
          float coverB = clamp(0.5 - sideB, 0.0, 1.0);
          float defB = 0.0;
          if (pkB.y > 0.0) {
            defB = max(
              profile - luxarPartnerProfile(vec4(vCutN.zw, pkB), vec2(x - vAbLen, y), -1.0, pkR.y, vSharp),
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
    const { capsuleLinePickWebGPUFactory, buildLinePickTSLNodesFromUniforms } =
      requireTslMaterials().factories.pickCapsuleLine;
    return capsuleLinePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(u), { isOrtho });
  },
};
