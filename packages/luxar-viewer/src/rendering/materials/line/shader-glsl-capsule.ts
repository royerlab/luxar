/**
 * Capsule line primitive — GLSL backend (issue #1352, behind
 * `?linePrimitive=capsule`).
 *
 * The screen-space quad's near-axial behaviour is pathological (direction
 * instability — the #790 sliver class) exactly where a distance-based
 * profile is stable by construction: this primitive evaluates a
 * gaussian-like profile of the 2D POINT-TO-SEGMENT DISTANCE in pixel
 * space, on stencil-LOCAL interpolated coordinates. Side-on it is the
 * familiar ribbon; end-on the segment projects to a point and the profile
 * becomes a clean radial disc; the transition is continuous because
 * point-to-segment distance is continuous in the endpoints. No 3D solver,
 * no ray integral — quad-class fragment cost.
 *
 * Model, calibration, the three deliberate exactness relaxations
 * (quartic profile, 2σ support, interpolated attributes) and the joint
 * rules (half-disc bisector joints — every end a round cap) are
 * documented in `_shared/line-capsule.ts`, the single constants source
 * this file and the TSL twin (`shader-tsl-capsule.ts`) fold from.
 *
 * The fragment's per-pixel work at the default knob: two cut-plane
 * multiply-adds, the overshoot max, one squared distance, the quartic
 * `w·w`, and the mode tail — no transcendentals.
 */
import {
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_NEAR_FADE_FUNCTIONS,
  GLSL_LINE_JOINT_CODE,
  GLSL_SORTED_INDEX,
} from '../_shared/glsl-lib';
import type { ShaderSource } from '../_shared/shader-source';
import { buildLineTSLNodesFromUniforms } from './shader-tsl';
import { capsuleLineWebGPUFactory } from './shader-tsl-capsule';
import {
  ALPHA_CLAMP,
  VOLUMETRIC_SERIES_C1,
  VOLUMETRIC_SERIES_C2_DIVISOR,
  VOLUMETRIC_SERIES_TAU_THRESHOLD,
  VOLUMETRIC_TAU_EPS,
} from '../_shared/volumetric';
import {
  CAPSULE_JOINT_DEFICIT_GATE,
  CAPSULE_JOINT_PACKET_MIN_RADIUS_PX,
  CAPSULE_MIN_RADIUS_PX,
  CAPSULE_RADIUS_PER_QUAD_HALFWIDTH,
  CAPSULE_STENCIL_APRON_PX,
} from '../_shared/line-capsule';

// All literals toFixed(7) — float32-exact and snapshot-stable.
const G = {
  RADIUS_FACTOR: CAPSULE_RADIUS_PER_QUAD_HALFWIDTH.toFixed(7), // 0.6590102
  MIN_RADIUS: CAPSULE_MIN_RADIUS_PX.toFixed(1),
  DEFICIT_GATE: CAPSULE_JOINT_DEFICIT_GATE.toFixed(2),
  PACKET_MIN_R: CAPSULE_JOINT_PACKET_MIN_RADIUS_PX.toFixed(1),
  APRON: CAPSULE_STENCIL_APRON_PX.toFixed(1),
};

export const CAPSULE_LINE_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}
    ${GLSL_LINE_JOINT_CODE}

    in vec2 aQuadCorner;

    ${GLSL_SORTED_INDEX}

    uniform highp sampler2D uLineTex;
    uniform vec2 uResolution;
    uniform int uIsOrtho;
    uniform float uNearCull;
    uniform float uMaxLinePixelWidth;
    uniform float uPerspectiveLineScale;
    uniform float uOrthoLineScale;

    // Stencil-LOCAL coordinates + attributes as INTERPOLATED varyings —
    // the fragment reads everything pre-blended by the rasterizer instead
    // of projecting/mixing per pixel (relaxation 3 in _shared/line-capsule).
    //
    // GEOMETRY varyings (vLocal) are SCREEN-SPACE quantities, but
    // default varying interpolation is perspective-correct — hyperbolic in
    // screen space whenever the two ends have different clip w (and with a
    // triangle-diagonal kink). GLSL ES 3.0 has no noperspective qualifier,
    // so they are pre-multiplied by the corner's clip w here and divided
    // by the interpolated w (vW) in the fragment: PC-interp(a·w)/PC(w) is
    // exactly screen-linear. Ortho (w = 1) reduces to a no-op.
    out vec2 vLocal;        // (x: axial px from A, y: perp px) × clip w
    // (abLen px, capFlags = interiorA + 2·interiorB, rA px, rB px)
    flat out vec4 vMeta;
    // 2D bisector-cut normals at interior joints, in stencil-local
    // coordinates. My side is NEGATIVE; a straight joint degrades to the
    // perpendicular butt ((-1,0) at A, (1,0) at B).
    // .xy = the 2D bisector-cut normal (my side negative); .z = the
    // DEFICIT packet: the partner leg's radius gradient in px/px along
    // its axis, stored ONLY when negative beyond the congruence gate
    // (a thinning partner is the one case whose missing light I must
    // render — see the fragment's deficit rule). 0 = hard cut.
    flat out vec4 vCutA2;
    flat out vec4 vCutB2;
    out float vW;           // clip w (divides vLocal in the fragment)
    out float vFade;        // nearFade × thin-width energy compensation
    out float vAlpha;       // per-element alpha (raw — volumetric gates it)
    out float vSharp;
    #ifdef USE_COLORMAP
    out float vScalar;
    #else
    out vec3 vColor;
    #endif

    // For an interior joint code, return the partner's FAR endpoint
    // (object space, .xyz) and its WIDTH there (.w — the same texels carry
    // both). Invalid codes return .w = -1 (widths are sanitized >= 0).
    // Same decode as the volumetric twin's luxarPartnerDir; codes land at
    // texel4.y/.z — see line-geometry.ts.
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
        vLocal = vec2(0.0); vMeta = vec4(1.0, 0.0, 1.0, 1.0);
        vCutA2 = vec4(-1.0, 0.0, 0.0, 0.0); vCutB2 = vec4(1.0, 0.0, 0.0, 0.0);
        vW = 1.0; vFade = 0.0; vAlpha = 1.0; vSharp = 0.5;
        #ifdef USE_COLORMAP
        vScalar = 0.0;
        #else
        vColor = vec3(0.0);
        #endif
        return;
      }
      // Near-plane segment clip (stencil AND profile domain — the 2D
      // formulation has no behind-eye notion, so clip the endpoints).
      // tA/tB survive the block: a clipped end means the JOINT VERTEX is
      // behind the near plane, which the cut construction must know.
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
      vec4 lineT5 = texelFetch(uLineTex, ivec2(texel0.x + 5, texel0.y), 0);

      // Endpoint attributes are re-evaluated at the CLIPPED span: a
      // near-clipped end must carry the values interpolated at tA/tB, not
      // the behind-camera endpoint's (taper/colour would otherwise jump).
      float w0 = sanitizeNonNegative(lineT0.w, 0.0);
      float w1 = sanitizeNonNegative(lineT1.w, 0.0);
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

      // Per-end pixel radius: the 2σ-trimmed fraction of the legacy quad
      // half-width, floored for AA and clamped like the quad.
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

      // Which ends CUT rather than cap, from the shared joint-code rule
      // (glsl-lib.ts): a free end (code 0) and a degree->=3 hub (code -2)
      // keep the whole round cap — several legs already stack at a hub and
      // its cut has no single partner to tile against. Everything else
      // cuts: a slot-bearing code at the bisector below, a slice-clipped
      // end (-1) at the perpendicular butt (nothing beyond the slice plane
      // may draw).
      float interiorA = luxarLineJointCapSuppression(lineT4.y);
      float interiorB = luxarLineJointCapSuppression(lineT4.z);

      vec2 ab = pB - pA;
      float abLen = length(ab);
      vec2 u = abLen > 1e-4 ? ab / abLen : vec2(1.0, 0.0);
      vec2 v = vec2(-u.y, u.x);
      float rMax = max(rA, rB) + ${G.APRON};
      // Interior ends get a 2D BISECTOR cut (adjacent capsules tile
      // EVERY end is a round cap. A free end keeps the whole disc; an
      // interior end keeps its HALF of the joint disc — the bisector cut
      // partitions the disc exactly between the two legs, so joints are
      // seamless, notch-free, and double-bright-free at any bend angle
      // and any zoom. Stencil reach at a cut end is |ny|·rMax (the kept
      // half-disc's axial extent), so joints cost LESS fill than caps.
      vec4 cutA = vec4(-1.0, 0.0, 0.0, 0.0);
      vec4 cutB = vec4(1.0, 0.0, 0.0, 0.0);
      float extA = rMax;
      float extB = rMax;
      if (interiorA > 0.5) {
        extA = ${G.APRON};
        // Joint vertex behind the near plane (my A end was clipped): the
        // joint region is invisible and the partner's projection is
        // meaningless — keep the perpendicular butt at the clip line.
        vec4 farA = tA > 0.0 ? vec4(0.0, 0.0, 0.0, -1.0) : luxarPartnerFar(lineT4.y, lineTexW);
        if (farA.w >= 0.0) {
          // Near-clip the PARTNER's far endpoint toward the joint vertex
          // before projecting — a behind-eye projection flips, and the
          // garbage direction poisons both the fold decision and the cut
          // normal (razor seams when zoomed into a joint).
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
                // Width gate (see _shared/line-capsule.ts): hairline
                // joints skip the packet — a deficit there is sub-pixel.
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
                  // Congruence gate: only a thinning partner (negative
                  // gradient beyond the gate) needs the deficit rule.
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
        // See end-A: behind-near joint keeps the butt; the partner's far
        // endpoint is near-clipped toward the joint vertex first.
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
                // Width gate (see _shared/line-capsule.ts): hairline
                // joints skip the packet — a deficit there is sub-pixel.
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
                  // Congruence gate: only a thinning partner (negative
                  // gradient beyond the gate) needs the deficit rule.
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

      // Corner in stencil-local coordinates (x from A along the axis).
      float lx = aQuadCorner.x > 0.0 ? abLen + extB : -extA;
      float ly = aQuadCorner.y * rMax;
      vec2 corner = pA + u * lx + v * ly;

      // Attribute values AT the corner's clamped axial position; the
      // rasterizer blends them per fragment (the blend spans the cap
      // extensions too — sub-quantization stretch, accepted).
      float tc = abLen > 1e-4 ? clamp(lx / abLen, 0.0, 1.0) : 0.5;
      // Texel-attribute mixes address the ORIGINAL endpoint values, so a
      // corner's parameter maps through the clipped span.
      float tOrig = mix(tA, tB, tc);
      // The RADIUS interpolates linearly in screen space (1/depth is
      // perspective-linear, so a constant-width tube's pixel radius is
      // exactly linear in screen x). Interpolating 1/r² instead bends the
      // rim quadratically inward — a hard concave silhouette at strong
      // taper (the zoomed near-axial case).
      // nearFade at the segment's own depth × the thin-width energy
      // compensation (the AA radius floor fattens sub-1.5px lines; scale
      // intensity down so additive totals stay width-linear — the quad's
      // widthScale rule).
      float rawC = mix(rawA, rawB, tc);
      float widthScale = min(rawC / ${G.MIN_RADIUS}, 1.0);
      float fade = perspectiveNearFade(uIsOrtho, mix(mvStart.z, mvEnd.z, tc), nearCull);
      vFade = fade * widthScale;
      vAlpha = mix(sanitizeAlpha(lineT5.z), sanitizeAlpha(lineT5.w), tOrig);
      vSharp = mix(s0, s1, tOrig);
      #ifdef USE_COLORMAP
      vScalar = mix(lineT5.x, lineT5.y, tOrig);
      #else
      vColor = mix(lineT2.rgb, lineT3.rgb, tOrig);
      #endif

      // Depth interpolates along the segment (clamped to the nearer end
      // across cap extensions) so depth-tested modes compose correctly.
      vec4 clipMix = mix(clipA, clipB, tc);
      float wMix = max(clipMix.w, 1e-6);
      // Screen-linear geometry varyings (see the declaration note).
      vLocal = vec2(lx, ly) * wMix;
      vW = wMix;
      vec2 ndc = corner / uResolution * 2.0 - 1.0;
      gl_Position = vec4(ndc * wMix, clipMix.z, wMix);
    }
`;

export const CAPSULE_LINE_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    uniform float uOpacity;
    uniform float uInvGamma;
    uniform float uIntensity;
    uniform float uOffset;
    uniform float uAbsorption;
    uniform float uHasElementAlpha;
    #ifdef USE_COLORMAP
    uniform sampler2D uColormapTex;
    uniform float uScalarMin;
    uniform float uScalarScale;
    #endif

    in vec2 vLocal;
    flat in vec4 vMeta;
    flat in vec4 vCutA2;
    flat in vec4 vCutB2;
    in float vW;
    in float vFade;
    in float vAlpha;
    in float vSharp;
    #ifdef USE_COLORMAP
    in float vScalar;
    #else
    in vec3 vColor;
    #endif

    out vec4 fragColor;

    // The PARTNER leg's profile at a pixel offset rel from the shared
    // vertex (my local frame), rebuilt from just the cut normal and the
    // partner's radius GRADIENT: the partner's axis is the reflection of
    // my inward axis mDir across the cut plane (q = m − 2(m·n)n — exact,
    // both normals being normalize(q − m) up to sign), its radius at the
    // cap region, where the deficit rule fires), and it is treated as an
    // unbounded tapering rod (its far cap only matters where both
    // profiles are ~0). Same profile family as ours; the sharpness knob
    // comes from our fragment — the joint region is local.
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
      // Undo the w-premultiplication: screen-linear local coordinates.
      float invW = 1.0 / max(vW, 1e-9);
      float x = vLocal.x * invW;
      float y = vLocal.y * invW;
      // Interior joints: the CAP region (beyond the endpoint) is cut along
      // the joint BISECTOR (my side negative) — the two legs' half-discs
      // tile the joint disc exactly. The cut applies ONLY beyond the
      // endpoint: where the two rod BODIES genuinely overlap (the inner
      // corner of a bend) both render, matching the physical union the
      // volumetric primitive integrates (and additive sums it, correctly).
      // EXACT per-fragment radius: mix of the endpoint radii clamped to
      // the segment span — a varying cannot represent this (its linear
      // interpolation spans the cap extensions, so a short stub's drawn
      // radius at its own endpoint drifts; the deeper root of #1494).
      float cutFlagA = mod(vMeta.y, 2.0);
      float cutFlagB = vMeta.y >= 2.0 ? 1.0 : 0.0;
      float rPx = max(
        mix(vMeta.z, vMeta.w, clamp(x / max(vMeta.x, 1e-4), 0.0, 1.0)),
        1e-4
      );
      // Squared distance in the local frame — the TRUE point-to-segment
      // distance, cap term included at BOTH ends: every end is capped (a
      // free end keeps the whole disc, a cut end its half of the joint
      // disc, carved out of that same cap by the bisector above).
      float ox = max(max(-x, x - vMeta.x), 0.0);
      float q = (y * y + ox * ox) / (rPx * rPx);
      float w = 1.0 - q;
      if (w <= 0.0) discard;

      // Gaussian-like quartic bump (see _shared/line-capsule.ts): exactly 0
      // at the 2σ rim, no exp, no floor constants. The sharpness knob bends
      // the bump via pow only off the default (n = 2^(3−4s); SMALLER
      // exponents are boxier in w-space).
      float profile = (abs(vSharp - 0.5) < 1e-3)
        ? w * w
        : pow(w, exp2(3.0 - 4.0 * vSharp));

      // Joint DEFICIT rule: on the partner's side of the joint bisector I
      // render only what the partner CANNOT — max(mine − partner, 0) — so
      // the additive pair composes to max(mine, partner). For congruent
      // legs this is exactly the hard partition (zero contribution, no
      // double-count, no bead); where the partner tapers away or its
      // apparent radius diverges under perspective, it fills exactly the
      // light the partition used to chop (a fat vertex's disc no longer
      // loses its far half to a thin neighbour). A packet length of 0
      // means no usable partner (slice clip, behind-near joint, decode
      // failure) — hard cut, nothing drawn past the plane.
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
      float intensity = profile * vFade;

      #ifdef USE_COLORMAP
      float st = clamp((vScalar - uScalarMin) * uScalarScale, 0.0, 1.0);
      #ifndef LUXAR_GAMMA_ONE
      st = pow(st, uInvGamma);
      #endif
      vec3 color = texture(uColormapTex, vec2(st, 0.5)).rgb;
      #else
      vec3 color = vColor;
      #endif

      #ifdef LUXAR_NO_GOG
      vec3 adjusted = color;
      #else
      vec3 adjusted = color * uIntensity + uOffset;
      adjusted = max(adjusted, vec3(0.0));
      #endif

      #ifdef LUXAR_VOLUMETRIC
      // Volumetric blending: alpha composes as optical depth via
      // w(a) = −ln(1 − a), gated by uHasElementAlpha (identity 1.0 written
      // for RGB data must NOT map to w ≈ 6.24). Mirrors the quad/point/
      // gsplat shaders; constants from ../_shared/volumetric.
      float alpha = intensity * uOpacity;
      alpha *= mix(1.0, -log(1.0 - min(vAlpha, ${ALPHA_CLAMP})), uHasElementAlpha);
      float tau = uAbsorption * alpha;
      // A black line still absorbs — discard only when colour AND τ vanish.
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4 && tau < 1e-4) discard;
      #else
      // Per-endpoint alpha is a plain linear contribution scale in every
      // non-volumetric mode.
      intensity *= vAlpha;
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;
      #endif

      #if defined(USE_COLORMAP) || defined(LUXAR_GAMMA_ONE)
      vec3 gammaColor = adjusted;
      #else
      vec3 gammaColor = pow(adjusted, vec3(uInvGamma));
      #endif

      #if defined(LUXAR_VOLUMETRIC)
      // Emission–absorption tail (Max 1995) — identical to the quad's.
      float volAlpha = 1.0 - exp(-tau);
      float screen = (tau < ${VOLUMETRIC_SERIES_TAU_THRESHOLD}) ? 1.0 - ${VOLUMETRIC_SERIES_C1} * tau + tau * tau / ${VOLUMETRIC_SERIES_C2_DIVISOR}.0
                                  : volAlpha / max(tau, ${VOLUMETRIC_TAU_EPS});
      fragColor = vec4(gammaColor * alpha * screen, volAlpha);
      #elif defined(LUXAR_MAX_RGB_CONTRIBUTION)
      // max-mode premultiplication (MaxEquation + One/One) — identical to
      // the quad's rationale.
      float a = intensity * uOpacity;
      fragColor = vec4(gammaColor * a, a);
      #else
      fragColor = vec4(gammaColor, intensity * uOpacity);
      #endif
    }
`;

export const CAPSULE_LINE_SOURCE: ShaderSource = {
  name: 'line-capsule',
  webgl: { vertex: CAPSULE_LINE_VERTEX_SHADER, fragment: CAPSULE_LINE_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    // Projection mode at build time (see LINE_SOURCE's note). No join
    // handling: the capsule has no miter geometry. Default config
    // otherwise — harness consumers needing USE_COLORMAP or mode variants
    // call `capsuleLineWebGPUFactory` directly with explicit flags.
    const isOrtho = ((u.uIsOrtho?.value as number) ?? 0) === 1;
    return capsuleLineWebGPUFactory(buildLineTSLNodesFromUniforms(u, {}), { isOrtho });
  },
};
