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
 * rules (2D bisector cuts; width-gated round caps past 120° turns) are
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
  CAPSULE_FOLD_CAP_MAX_COS,
  CAPSULE_FOLD_CAP_MIN_RADIUS_PX,
  CAPSULE_MIN_RADIUS_PX,
  CAPSULE_RADIUS_PER_QUAD_HALFWIDTH,
  CAPSULE_STENCIL_APRON_PX,
} from '../_shared/line-capsule';

// All literals toFixed(7) — float32-exact and snapshot-stable.
const G = {
  RADIUS_FACTOR: CAPSULE_RADIUS_PER_QUAD_HALFWIDTH.toFixed(7), // 0.6590102
  MIN_RADIUS: CAPSULE_MIN_RADIUS_PX.toFixed(1),
  APRON: CAPSULE_STENCIL_APRON_PX.toFixed(1),
  FOLD_COS: CAPSULE_FOLD_CAP_MAX_COS.toFixed(1),
  FOLD_MIN_R: CAPSULE_FOLD_CAP_MIN_RADIUS_PX.toFixed(1),
};

export const CAPSULE_LINE_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}

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
    out vec2 vLocal;        // (x: axial px from A, y: perp px)
    flat out vec3 vMeta;    // abLen px, cutA active, cutB active
    // 2D bisector-cut normals at interior joints, in stencil-local
    // coordinates. My side is NEGATIVE; a straight joint degrades to the
    // perpendicular butt ((-1,0) at A, (1,0) at B).
    flat out vec2 vCutA2;
    flat out vec2 vCutB2;
    out float vInvR2;       // 1/r² (linearized across taper — relaxation 3)
    out float vFade;        // nearFade × thin-width energy compensation
    out float vAlpha;       // per-element alpha (raw — volumetric gates it)
    out float vSharp;
    #ifdef USE_COLORMAP
    out float vScalar;
    #else
    out vec3 vColor;
    #endif

    // For an interior joint code, return the partner's FAR endpoint
    // (object space) + validity (same decode as the volumetric twin's
    // luxarPartnerDir; codes land at texel4.y/.z — see line-geometry.ts).
    vec4 luxarPartnerFar(float code, int lineTexW) {
      bool interior = (code > 0.5) || (code < -2.5);
      if (!interior) return vec4(0.0);
      int slot = (code > 0.0) ? int(code + 0.5) - 1 : int(-code + 0.5) - 3;
      int pBase = slot * 6;
      ivec2 pt0 = ivec2(pBase % lineTexW, pBase / lineTexW);
      vec3 pStart = texelFetch(uLineTex, pt0, 0).xyz;
      vec3 pEnd = texelFetch(uLineTex, ivec2(pt0.x + 1, pt0.y), 0).xyz;
      return vec4((code > 0.0) ? pEnd : pStart, 1.0);
    }

    // Project an object-space point to pixel coordinates.
    vec2 luxarToPx(vec3 objP) {
      vec4 cl = projectionMatrix * (modelViewMatrix * vec4(objP, 1.0));
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
        vLocal = vec2(0.0); vMeta = vec3(1.0, 0.0, 0.0);
        vCutA2 = vec2(-1.0, 0.0); vCutB2 = vec2(1.0, 0.0);
        vInvR2 = 1.0; vFade = 0.0; vAlpha = 1.0; vSharp = 0.5;
        #ifdef USE_COLORMAP
        vScalar = 0.0;
        #else
        vColor = vec3(0.0);
        #endif
        return;
      }
      // Near-plane segment clip (stencil AND profile domain — the 2D
      // formulation has no behind-eye notion, so clip the endpoints).
      if (uIsOrtho == 0) {
        float tA = 0.0;
        float tB = 1.0;
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

      float w0 = sanitizeNonNegative(lineT0.w, 0.0);
      float w1 = sanitizeNonNegative(lineT1.w, 0.0);
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
        rawA = w0 * uOrthoLineScale * ${G.RADIUS_FACTOR};
        rawB = w1 * uOrthoLineScale * ${G.RADIUS_FACTOR};
      } else {
        rawA = w0 * uPerspectiveLineScale * ${G.RADIUS_FACTOR} / max(-mvStart.z, nearCull);
        rawB = w1 * uPerspectiveLineScale * ${G.RADIUS_FACTOR} / max(-mvEnd.z, nearCull);
      }
      float rA = clamp(rawA, ${G.MIN_RADIUS}, uMaxLinePixelWidth);
      float rB = clamp(rawB, ${G.MIN_RADIUS}, uMaxLinePixelWidth);

      float interiorA = abs(lineT4.y) > 0.5 ? 1.0 : 0.0;
      float interiorB = abs(lineT4.z) > 0.5 ? 1.0 : 0.0;

      vec2 ab = pB - pA;
      float abLen = length(ab);
      vec2 u = abLen > 1e-4 ? ab / abLen : vec2(1.0, 0.0);
      vec2 v = vec2(-u.y, u.x);
      float rMax = max(rA, rB) + ${G.APRON};
      // Interior ends get a 2D BISECTOR cut (adjacent capsules tile
      // exactly at any bend); the stencil extends by the clamped miter
      // overhang. Free ends — and interior ends at visibly-wide sharp
      // folds (the fold-cap rule) — keep the round cap.
      vec2 cutA = vec2(-1.0, 0.0);
      vec2 cutB = vec2(1.0, 0.0);
      float extA = rMax;
      float extB = rMax;
      if (interiorA > 0.5) {
        extA = 1.0;
        vec4 farA = luxarPartnerFar(lineT4.y, lineTexW);
        if (farA.w > 0.5) {
          vec2 qq = luxarToPx(farA.xyz) - pA;
          float ql = length(qq);
          if (ql > 1e-4 && dot(qq / ql, u) > ${G.FOLD_COS} && rA > ${G.FOLD_MIN_R}) {
            // Turn sharper than the fold bound AND visibly wide: a clamped
            // cut would chop the fold tip flat — round cap instead. At
            // hairline widths the notch is sub-pixel, so the cut stays.
            interiorA = 0.0;
            extA = rMax;
          } else if (ql > 1e-4) {
            vec2 nRaw = qq / ql - u;   // q − m, m = +u at A
            float nl = length(nRaw);
            if (nl > 1e-3) {
              vec2 n2 = nRaw / nl;
              vec2 nLoc = vec2(dot(n2, u), dot(n2, v));
              if (nLoc.x < -1e-3) {
                cutA = nLoc;
                extA = clamp(abs(nLoc.y / nLoc.x) * rMax, 0.0, rMax) + ${G.APRON};
              }
            }
          }
        }
      }
      if (interiorB > 0.5) {
        extB = 1.0;
        vec4 farB = luxarPartnerFar(lineT4.z, lineTexW);
        if (farB.w > 0.5) {
          vec2 qq = luxarToPx(farB.xyz) - pB;
          float ql = length(qq);
          if (ql > 1e-4 && dot(qq / ql, u) < -${G.FOLD_COS} && rB > ${G.FOLD_MIN_R}) {
            // Fold-cap rule at end B (see end-A note).
            interiorB = 0.0;
            extB = rMax;
          } else if (ql > 1e-4) {
            vec2 nRaw = qq / ql + u;   // q − m, m = −u at B
            float nl = length(nRaw);
            if (nl > 1e-3) {
              vec2 n2 = nRaw / nl;
              vec2 nLoc = vec2(dot(n2, u), dot(n2, v));
              if (nLoc.x > 1e-3) {
                cutB = nLoc;
                extB = clamp(abs(nLoc.y / nLoc.x) * rMax, 0.0, rMax) + ${G.APRON};
              }
            }
          }
        }
      }
      vCutA2 = cutA;
      vCutB2 = cutB;
      vMeta = vec3(abLen, interiorA, interiorB);

      // Corner in stencil-local coordinates (x from A along the axis).
      float lx = aQuadCorner.x > 0.0 ? abLen + extB : -extA;
      float ly = aQuadCorner.y * rMax;
      vec2 corner = pA + u * lx + v * ly;
      vLocal = vec2(lx, ly);

      // Attribute values AT the corner's clamped axial position; the
      // rasterizer blends them per fragment (the blend spans the cap
      // extensions too — sub-quantization stretch, accepted).
      float tc = abLen > 1e-4 ? clamp(lx / abLen, 0.0, 1.0) : 0.5;
      float rC = mix(rA, rB, tc);
      vInvR2 = 1.0 / (rC * rC);
      // nearFade at the segment's own depth × the thin-width energy
      // compensation (the AA radius floor fattens sub-1.5px lines; scale
      // intensity down so additive totals stay width-linear — the quad's
      // widthScale rule).
      float rawC = mix(rawA, rawB, tc);
      float widthScale = min(rawC / ${G.MIN_RADIUS}, 1.0);
      float fade = perspectiveNearFade(uIsOrtho, mix(mvStart.z, mvEnd.z, tc), nearCull);
      vFade = fade * widthScale;
      vAlpha = mix(sanitizeAlpha(lineT5.z), sanitizeAlpha(lineT5.w), tc);
      vSharp = mix(s0, s1, tc);
      #ifdef USE_COLORMAP
      vScalar = mix(lineT5.x, lineT5.y, tc);
      #else
      vColor = mix(lineT2.rgb, lineT3.rgb, tc);
      #endif

      // Depth interpolates along the segment (clamped to the nearer end
      // across cap extensions) so depth-tested modes compose correctly.
      vec4 clipMix = mix(clipA, clipB, tc);
      float wMix = max(clipMix.w, 1e-6);
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
    flat in vec3 vMeta;
    flat in vec2 vCutA2;
    flat in vec2 vCutB2;
    in float vInvR2;
    in float vFade;
    in float vAlpha;
    in float vSharp;
    #ifdef USE_COLORMAP
    in float vScalar;
    #else
    in vec3 vColor;
    #endif

    out vec4 fragColor;

    void main() {
      float x = vLocal.x;
      float y = vLocal.y;
      // Interior joints: cut along the joint BISECTOR (my side negative)
      // — adjacent capsules tile exactly; straight joints degrade to butt.
      if (vMeta.y > 0.5 && (vCutA2.x * x + vCutA2.y * y) > 0.0) discard;
      if (vMeta.z > 0.5 && (vCutB2.x * (x - vMeta.x) + vCutB2.y * y) > 0.0) discard;
      // Squared distance in the local frame; at cut ends the ROD continues
      // to the cut line (no cap overshoot term there).
      float ox = max(max(vMeta.y > 0.5 ? 0.0 : -x, vMeta.z > 0.5 ? 0.0 : x - vMeta.x), 0.0);
      float q = (y * y + ox * ox) * vInvR2;
      float w = 1.0 - q;
      if (w <= 0.0) discard;

      // Gaussian-like quartic bump (see _shared/line-capsule.ts): exactly 0
      // at the 2σ rim, no exp, no floor constants. The sharpness knob bends
      // the bump via pow only off the default (n = 2^(3−4s); SMALLER
      // exponents are boxier in w-space).
      float profile = (abs(vSharp - 0.5) < 1e-3)
        ? w * w
        : pow(w, exp2(3.0 - 4.0 * vSharp));
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
