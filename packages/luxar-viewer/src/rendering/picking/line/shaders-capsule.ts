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
  GLSL_NEAR_FADE_FUNCTIONS,
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_SORTED_INDEX,
} from '../../materials/_shared/glsl-lib';
import type { ShaderSource } from '../../materials/_shared/shader-source';
import { buildLinePickTSLNodesFromUniforms } from './pick.tsl';
import { capsuleLinePickWebGPUFactory } from './pick-capsule.tsl';
import {
  CAPSULE_MIN_RADIUS_PX,
  CAPSULE_CUT_FADE_RADIUS_FRACTION,
  CAPSULE_RADIUS_PER_QUAD_HALFWIDTH,
  CAPSULE_STENCIL_APRON_PX,
} from '../../materials/_shared/line-capsule';

const G = {
  RADIUS_FACTOR: CAPSULE_RADIUS_PER_QUAD_HALFWIDTH.toFixed(7),
  MIN_RADIUS: CAPSULE_MIN_RADIUS_PX.toFixed(1),
  APRON: CAPSULE_STENCIL_APRON_PX.toFixed(1),
  CUT_FADE: CAPSULE_CUT_FADE_RADIUS_FRACTION.toFixed(2),
};

export const CAPSULE_LINE_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}

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

    out vec2 vLocal;
    flat out vec3 vMeta;
    flat out vec2 vCutA2;
    flat out vec2 vCutB2;
    out float vR;
    out float vFade;
    out float vSharp;
    flat out highp float vNodeId;
    flat out highp vec2 vElementId;

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
        vR = 1.0; vFade = 0.0; vSharp = 0.5;
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
      vec2 cutA = vec2(-1.0, 0.0);
      vec2 cutB = vec2(1.0, 0.0);
      float extA = rMax;
      float extB = rMax;
      if (interiorA > 0.5) {
        extA = ${G.APRON};
        // Mirrors the visual capsule exactly (or hover desyncs from
        // pixels): behind-near joint keeps the butt; the partner's far
        // endpoint is near-clipped toward the joint vertex first.
        vec4 farA = tA > 0.0 ? vec4(0.0) : luxarPartnerFar(lineT4.y, lineTexW);
        if (farA.w > 0.5) {
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
                extA = abs(nLoc.y) * rMax + ${G.APRON};
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
        vec4 farB = tB < 1.0 ? vec4(0.0) : luxarPartnerFar(lineT4.z, lineTexW);
        if (farB.w > 0.5) {
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
                extB = abs(nLoc.y) * rMax + ${G.APRON};
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
      vLocal = vec2(lx, ly);

      float tc = abLen > 1e-4 ? clamp(lx / abLen, 0.0, 1.0) : 0.5;
      // The RADIUS interpolates linearly in screen space (1/depth is
      // perspective-linear, so a constant-width tube's pixel radius is
      // exactly linear in screen x). Interpolating 1/r² instead bends the
      // rim quadratically inward — a hard concave silhouette at strong
      // taper (the zoomed near-axial case).
      vR = mix(rA, rB, tc);
      float rawC = mix(rawA, rawB, tc);
      float widthScale = min(rawC / ${G.MIN_RADIUS}, 1.0);
      vFade = perspectiveNearFade(uIsOrtho, mix(mvStart.z, mvEnd.z, tc), nearCull) * widthScale;
      vSharp = mix(s0, s1, tc);

      vec4 clipMix = mix(clipA, clipB, tc);
      float wMix = max(clipMix.w, 1e-6);
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
    in float vR;
    in float vFade;
    in float vSharp;
    flat in highp float vNodeId;
    flat in highp vec2 vElementId;

    out vec4 fragColor;

    void main() {
      float x = vLocal.x;
      float y = vLocal.y;
      // Foreign-side cap fade: my cap region on the PARTNER's side of the
      // joint bisector fades out over a quarter radius of axial overhang
      // instead of a hard cut — C0 with the partner's body at its endpoint
      // line (no chevron edge) and with my own half-disc at the bisector.
      // Sub-pixel at normal widths; smooth and wide when zoomed in.
      float rPx = max(vR, 1e-4);
      float cutFade = 1.0;
      if (vMeta.y > 0.5 && x < 0.0 && (vCutA2.x * x + vCutA2.y * y) > 0.0) {
        cutFade *= clamp(1.0 + x / (${G.CUT_FADE} * rPx), 0.0, 1.0);
      }
      if (vMeta.z > 0.5 && x > vMeta.x && (vCutB2.x * (x - vMeta.x) + vCutB2.y * y) > 0.0) {
        cutFade *= clamp(1.0 - (x - vMeta.x) / (${G.CUT_FADE} * rPx), 0.0, 1.0);
      }
      if (cutFade <= 0.0) discard;
      // TRUE point-to-segment distance: every end is capped (a free end
      // keeps the whole disc, a cut end its half of the joint disc).
      float ox = max(max(-x, x - vMeta.x), 0.0);
      float q = (y * y + ox * ox) / (rPx * rPx);
      float w = 1.0 - q;
      if (w <= 0.0) discard;

      float profile = (abs(vSharp - 0.5) < 1e-3)
        ? w * w
        : pow(w, exp2(3.0 - 4.0 * vSharp));
      float brightness = profile * vFade * cutFade;
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
