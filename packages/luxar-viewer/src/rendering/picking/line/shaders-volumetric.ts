/**
 * GLSL3 picking shader for the VOLUMETRIC line primitive (issue #1352,
 * behind `?linePrimitive=volumetric`) + `ShaderSource` record.
 *
 * Mirrors the visual volumetric shader's stadium stencil vertex stage
 * (`materials/line/shader-glsl-volumetric.ts`) so the pick footprint is
 * rasterized by the SAME stencil the eye sees — bisector-cut overhang,
 * depth-tilt disc reach, coverage fade and all. The fragment stage is the
 * visual shader's PEAK lane (Gaussian-shoulder capsule at the exact
 * ray-to-cell distance), used UNCONDITIONALLY: picking wants the hotspot on
 * the centerline regardless of the visual blending mode, and the peak
 * formulation is exact for any sharpness β with no integral machinery.
 * The sum lanes' closed forms (erf windows, mixed splits, near-plane
 * clip algebra) are deliberately absent — a pick buffer needs a
 * brightness ORDERING, not radiometry, and peak brightness at the hit
 * point gives exactly that (centerline ≥ shoulder everywhere).
 *
 * Output contract (unchanged from the screen-space pick shader):
 * `vec4(nodeId, elementId-low16, brightness, elementId-high16)` with
 * `gl_FragDepth = 1 − brightness` (brightness-as-depth tie-breaking).
 *
 * Colors / scalars / element alpha are stripped (not needed for picking);
 * widths and sharpness are kept — they shape the capsule.
 *
 * Source-of-truth for GLSL3; the WebGPU counterpart lives in
 * `./pick-volumetric.tsl` and is referenced through the
 * `ShaderSource.webgpu` factory below.
 *
 * @module rendering/picking/line/shaders-volumetric
 */

import type { ShaderSource } from '../../materials/_shared/shader-source';
import {
  GLSL_NEAR_FADE_FUNCTIONS,
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_SORTED_INDEX,
} from '../../materials/_shared/glsl-lib';
import { volumetricLinePickWebGPUFactory } from './pick-volumetric.tsl';
import { buildLinePickTSLNodesFromUniforms } from './pick.tsl';
import {
  FALLOFF_FLOOR,
  FALLOFF_K,
  GAUSSIAN_EQUIVALENT_TRUNCATION,
} from '../../materials/_shared/falloff';
import {
  LINE_PARALLEL_LANE_THRESHOLD,
  LINE_SIGMA_PER_WIDTH,
  LINE_STENCIL_DILATION,
} from '../../materials/_shared/line-volumetric';

const T = GAUSSIAN_EQUIVALENT_TRUNCATION;
// All literals toFixed(7) — float32-exact and snapshot-stable (falloff.ts note).
const G = {
  K: FALLOFF_K.toFixed(7),
  C: FALLOFF_FLOOR.toFixed(7),
  INV_ONE_MINUS_C: (1 / (1 - FALLOFF_FLOOR)).toFixed(7),
  T_SQ: (T * T).toFixed(7),
  SIGMA_PER_WIDTH: LINE_SIGMA_PER_WIDTH.toFixed(7),
  DILATION: LINE_STENCIL_DILATION.toFixed(7),
  T_SQ_DILATION: (T * T * LINE_STENCIL_DILATION).toFixed(7),
  PARALLEL_THRESHOLD: LINE_PARALLEL_LANE_THRESHOLD.toExponential(1),
};

/**
 * Volumetric picking vertex shader: the visual volumetric stadium stencil
 * (see `shader-glsl-volumetric.ts` for the geometry rationale, kept in
 * lockstep) with colors / scalars / alpha stripped and the pick IDs added.
 */
export const VOLUMETRIC_LINE_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}

    in vec2 aQuadCorner;  // (-1,-1), (1,-1), (-1,1), (1,1)

    ${GLSL_SORTED_INDEX}

    uniform highp sampler2D uLineTex;
    uniform vec2 uResolution;
    uniform int uIsOrtho;
    uniform float uNearCull;
    uniform float uMaxLinePixelWidth;
    uniform float uPerspectiveLineScale;
    uniform float uOrthoLineScale;
    uniform float uNodeId;

    // All varyings are flat per-segment constants, written identically on
    // every vertex (WGSL provokes first-vertex, WebGL last-vertex).
    flat out vec4 vSegA;   // TRUE camera-space start.xyz (pre near-clip), .w = L
    flat out vec4 vSegW;   // unit axis.xyz, .w = coverageFade amplitude scale
    flat out vec4 vEnds;   // width0, width1, sharpness0, sharpness1 (sanitized)
    // Bisector cut planes at interior joints — see the visual shader.
    flat out vec4 vCutA;
    flat out vec4 vCutB;
    flat out highp float vNodeId;
    flat out highp vec2 vElementId;

    // Partner direction per end — the visual shader's helper, verbatim
    // (camera-space normalization is load-bearing under model scale).
    vec4 luxarPartnerDir(float code, int lineTexW) {
      bool interior = (code > 0.5) || (code < -2.5);
      if (!interior) return vec4(0.0);
      int slot = (code > 0.0) ? int(code + 0.5) - 1 : int(-code + 0.5) - 3;
      int pBase = slot * 6;
      ivec2 pt0 = ivec2(pBase % lineTexW, pBase / lineTexW);
      vec3 pStart = texelFetch(uLineTex, pt0, 0).xyz;
      vec3 pEnd = texelFetch(uLineTex, ivec2(pt0.x + 1, pt0.y), 0).xyz;
      vec3 q = (code > 0.0) ? (pEnd - pStart) : (pStart - pEnd);
      float qLen = length(q);
      if (qLen < 1e-20) return vec4(0.0);
      vec3 qCam = mat3(modelViewMatrix) * q;
      float qCamLen = length(qCam);
      if (qCamLen < 1e-20) return vec4(0.0);
      return vec4(qCam / qCamLen, 1.0);
    }

    void main() {
      int lineBase = int(luxarSortedIndex()) * 6;
      int lineTexW = textureSize(uLineTex, 0).x;
      ivec2 texel0 = ivec2(lineBase % lineTexW, lineBase / lineTexW);
      vec4 lineT0 = texelFetch(uLineTex, texel0, 0);
      vec4 lineT1 = texelFetch(uLineTex, ivec2(texel0.x + 1, texel0.y), 0);
      vec4 lineT4 = texelFetch(uLineTex, ivec2(texel0.x + 4, texel0.y), 0);
      vec3 aStartPos = lineT0.xyz;
      float aStartWidth = lineT0.w;
      vec3 aEndPos = lineT1.xyz;
      float aEndWidth = lineT1.w;
      float aStartJointCode = lineT4.y;
      float aEndJointCode = lineT4.z;

      float t = aQuadCorner.x * 0.5 + 0.5;

      // Pick IDs are unconditional — every exit path below (sentinel or
      // rasterized) leaves them valid for the fragment stage.
      vNodeId = uNodeId;
      vElementId = luxarElementIdParts();

      vec4 mvStart = modelViewMatrix * vec4(aStartPos, 1.0);
      vec4 mvEnd = modelViewMatrix * vec4(aEndPos, 1.0);

      // TRUE endpoints for the fragment: near-plane clipping below reshapes
      // only the stencil.
      vec3 trueA = mvStart.xyz;
      vec3 trueB = mvEnd.xyz;

      float nearCull = max(uNearCull, 1e-20);
      float startDepth = -mvStart.z;
      float endDepth = -mvEnd.z;
      bool bothBehind =
        (uIsOrtho == 0) && (startDepth < nearCull) && (endDepth < nearCull);
      if (bothBehind) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        vSegA = vec4(0.0);
        vSegW = vec4(1.0, 0.0, 0.0, 0.0);
        vEnds = vec4(0.0);
        vCutA = vec4(0.0);
        vCutB = vec4(0.0);
        return;
      }

      // Near-plane SEGMENT clipping (stencil only; perspective only).
      float tA = 0.0;
      float tB = 1.0;
      if (uIsOrtho == 0) {
        if (startDepth < nearCull && endDepth >= nearCull) {
          tA = (nearCull - startDepth) / (endDepth - startDepth);
        } else if (endDepth < nearCull && startDepth >= nearCull) {
          tB = (startDepth - nearCull) / (startDepth - endDepth);
        }
        vec4 mvStartClipped = mix(mvStart, mvEnd, tA);
        vec4 mvEndClipped = mix(mvStart, mvEnd, tB);
        mvStart = mvStartClipped;
        mvEnd = mvEndClipped;
      }

      // Deferred texel fetches past the cheap cull — sharpness only
      // (colors, scalars and alpha are not read by the pick pass).
      float sharpT2 = texelFetch(uLineTex, ivec2(texel0.x + 2, texel0.y), 0).w;
      float sharpT3 = texelFetch(uLineTex, ivec2(texel0.x + 3, texel0.y), 0).w;

      float w0 = sanitizeNonNegative(aStartWidth, 0.0);
      float w1 = sanitizeNonNegative(aEndWidth, 0.0);
      float s0 = clamp(sanitizeNonNegative(sharpT2, 0.5), 0.0, 1.0);
      float s1 = clamp(sanitizeNonNegative(sharpT3, 0.5), 0.0, 1.0);
      vEnds = vec4(w0, w1, s0, s1);

      vec3 seg = trueB - trueA;
      float segLen = length(seg);
      vec3 axisW = segLen > 1e-20 ? seg / segLen : vec3(1.0, 0.0, 0.0);
      vSegA = vec4(trueA, segLen);

      // Bisector cut planes at interior joints — visual shader, verbatim.
      vec4 qA = luxarPartnerDir(aStartJointCode, lineTexW);
      vec4 qB = luxarPartnerDir(aEndJointCode, lineTexW);
      vCutA = vec4(0.0);
      vCutB = vec4(0.0);
      float tanHalfA = 0.0;
      float tanHalfB = 0.0;
      if (qA.w > 0.5 && segLen > 1e-20) {
        vec3 nRaw = qA.xyz - axisW;             // m = +w at end A
        float nLen = length(nRaw);
        float d = dot(axisW, qA.xyz);           // = m·q
        if (nLen > 1e-6) {
          vCutA = vec4(nRaw / nLen, 1.0);
          tanHalfA = sqrt(max(1.0 + d, 0.0) / max(1.0 - d, 1e-6));
        }
      }
      if (qB.w > 0.5 && segLen > 1e-20) {
        vec3 nRaw = qB.xyz + axisW;             // m = -w at end B
        float nLen = length(nRaw);
        float d = dot(-axisW, qB.xyz);
        if (nLen > 1e-6) {
          vCutB = vec4(nRaw / nLen, 1.0);
          tanHalfB = sqrt(max(1.0 + d, 0.0) / max(1.0 - d, 1e-6));
        }
      }

      vec4 clipStart = projectionMatrix * mvStart;
      vec4 clipEnd = projectionMatrix * mvEnd;
      vec4 clipPos = mix(clipStart, clipEnd, t);

      float wGuard = (uIsOrtho == 1) ? 1.0 : nearCull;
      float wStart = max(clipStart.w, wGuard);
      float wEnd = max(clipEnd.w, wGuard);
      vec2 ndcStart = clipStart.xy / wStart;
      vec2 ndcEnd = clipEnd.xy / wEnd;

      vec2 pixelDir = (ndcEnd - ndcStart) * (0.5 * uResolution);
      float pixelLen = length(pixelDir);
      // Degenerate projected direction is HARMLESS (square covers disc;
      // the fragment never reads the stencil orientation).
      vec2 lineDir = pixelLen > 0.0001 ? pixelDir / pixelLen : vec2(1.0, 0.0);
      vec2 perpendicular = vec2(-lineDir.y, lineDir.x);

      // Per-end drawn half-width in pixels, at the CLIPPED endpoints, with
      // the T² · dilation variance floor folded in.
      float wAtA = mix(w0, w1, tA);
      float wAtB = mix(w0, w1, tB);
      float rawA;
      float rawB;
      if (uIsOrtho == 1) {
        rawA = wAtA * uOrthoLineScale;
        rawB = wAtB * uOrthoLineScale;
      } else {
        rawA = wAtA * uPerspectiveLineScale / max(-mvStart.z, nearCull);
        rawB = wAtB * uPerspectiveLineScale / max(-mvEnd.z, nearCull);
      }
      float RA = sqrt(rawA * rawA + ${G.T_SQ_DILATION});
      float RB = sqrt(rawB * rawB + ${G.T_SQ_DILATION});

      // Coverage fade + hard extent clamp (visual parity: a segment the
      // visual pass faded out must not stay pickable at full strength).
      float maxExtent = max(uMaxLinePixelWidth, 2.5);
      float Rmax = max(RA, RB);
      float coverageFade = 1.0 - smoothstep(maxExtent * 0.5, maxExtent, Rmax);
      if (coverageFade < 0.01) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        vSegW = vec4(axisW, 0.0);
        return;
      }
      RA = min(RA, maxExtent);
      RB = min(RB, maxExtent);
      vSegW = vec4(axisW, coverageFade);

      // Stadium corner — visual shader, verbatim (soft cap 0.77·R; bisector
      // cut min(R·tanθ/2, R) on the OUTER side; depth-tilt disc reach).
      float R = (aQuadCorner.x > 0.0) ? RB : RA;
      float hardEnd = (aQuadCorner.x > 0.0) ? vCutB.w : vCutA.w;
      float tanHalf = (aQuadCorner.x > 0.0) ? tanHalfB : tanHalfA;
      vec2 out2 = (aQuadCorner.x > 0.0)
        ? (axisW.xy - qB.xy)
        : (-(axisW.xy + qA.xy));
      float side = dot(out2, perpendicular);
      bool cornerOuter = (aQuadCorner.y * side) > 0.0;
      float overhang = min(R * tanHalf, R);
      vec3 endPos = (aQuadCorner.x > 0.0) ? mvEnd.xyz : mvStart.xyz;
      vec3 viewDir = (uIsOrtho == 1)
        ? vec3(0.0, 0.0, -1.0)
        : endPos * inversesqrt(max(dot(endPos, endPos), 1e-20));
      float discReach = R * abs(dot(axisW, viewDir));
      float axialExtend = (hardEnd > 0.5)
        ? ((cornerOuter ? overhang : min(0.25 * overhang, 4.0)) + discReach + 1.5)
        : (0.77 * R + discReach + 1.5);
      vec2 pixelOffset =
        perpendicular * (aQuadCorner.y * R) + lineDir * (aQuadCorner.x * axialExtend);
      vec2 ndcOffset = pixelOffset / uResolution * 2.0;
      clipPos.xy += ndcOffset * clipPos.w;

      gl_Position = clipPos;
    }
  `;

/**
 * Volumetric picking fragment shader: the visual shader's PEAK capsule
 * lane, unconditionally, with the pick output contract. Brightness is the
 * capsule profile scaled by the same energy compensation, coverage fade
 * and near fade the visual fragment applies — the pick footprint tracks
 * the visible one by construction rather than by parallel re-derivation.
 */
export const VOLUMETRIC_LINE_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;
    ${GLSL_NEAR_FADE_FUNCTIONS}

    uniform int uIsOrtho;
    uniform float uNearCull;
    uniform vec2 uResolution;
    uniform float uPerspectiveLineScale;
    uniform float uOrthoLineScale;

    flat in vec4 vSegA;
    flat in vec4 vSegW;
    flat in vec4 vEnds;
    flat in vec4 vCutA;
    flat in vec4 vCutB;
    flat in highp float vNodeId;
    flat in highp vec2 vElementId;

    out vec4 fragColor;

    void main() {
      // Per-fragment view ray in camera space, UNNORMALIZED — the solver
      // folds |dRaw| in analytically (visual fragment, verbatim).
      vec3 dRaw;
      vec3 rayO;
      if (uIsOrtho == 1) {
        rayO = vec3((gl_FragCoord.xy - 0.5 * uResolution) * (2.0 / uOrthoLineScale), 0.0);
        dRaw = vec3(0.0, 0.0, -1.0);
      } else {
        rayO = vec3(0.0);
        dRaw = vec3((gl_FragCoord.xy - 0.5 * uResolution) * (2.0 / uPerspectiveLineScale), -1.0);
      }

      float L = vSegA.w;
      vec3 w = vSegW.xyz;
      vec3 M = vSegA.xyz + w * (0.5 * L);
      vec3 b = rayO - M;
      float n2 = dot(dRaw, dRaw);
      float rn = sqrt(n2);
      float dw = dot(dRaw, w);
      float bdr = dot(b, dRaw);
      float bw = dot(b, w);
      float A = n2 - dw * dw;
      bool parallel = A < ${G.PARALLEL_THRESHOLD} * n2;
      float sM;
      float camZ;
      if (parallel) {
        // No parallel-specific distance math is needed on the peak path
        // (point-to-rod distance is s-independent for parallel geometry;
        // a clamped garbage sM still yields the right distance) — only a
        // finite sM for the attribute lookup below.
        sM = 0.0;
        camZ = rayO.z + dRaw.z * (-bdr / n2);
      } else {
        float invA = 1.0 / A;
        sM = (bw * n2 - dw * bdr) * invA;
        camZ = rayO.z + dRaw.z * ((dw * bw - bdr) * invA);
      }

      // Width / sharpness at the clamped closest-approach axial coord.
      float sHat = clamp(sM / max(L, 1e-20) + 0.5, 0.0, 1.0);
      float width = mix(vEnds.x, vEnds.y, sHat);
      float sharp = mix(vEnds.z, vEnds.w, sHat);

      float nearCull = max(uNearCull, 1e-20);
      float pxSize = (uIsOrtho == 1)
        ? (2.0 / uOrthoLineScale)
        : (2.0 * max(-camZ, nearCull) / uPerspectiveLineScale);
      float sigma = ${G.SIGMA_PER_WIDTH} * width;
      float invSE = inversesqrt(sigma * sigma + ${G.DILATION} * pxSize * pxSize + 1e-30);
      float aaComp = sigma * invSE;

      // PEAK capsule — the visual fragment's LUXAR_PEAK_PROJECTION lane,
      // verbatim: bisector cuts as a ray-domain interval [tLo, tHi], the
      // convex-in-t clamp-and-reproject exact constrained minimum, and the
      // Gaussian-shoulder profile at the resulting distance.
      bool hardA = vCutA.w > 0.5;
      bool hardB = vCutB.w > 0.5;
      float tLo = -1e30;
      float tHi = 1e30;
      bool dead = false;
      if (hardA) {
        float dn = dot(dRaw, vCutA.xyz);
        float sn = dot(vSegA.xyz - rayO, vCutA.xyz);
        if (abs(dn) <= 1e-7 * rn) {
          dead = dead || (sn < 0.0);
        } else {
          float tX = sn / dn;
          if (dn > 0.0) tHi = min(tHi, tX); else tLo = max(tLo, tX);
        }
      }
      if (hardB) {
        vec3 Bp = vSegA.xyz + w * L;
        float dn = dot(dRaw, vCutB.xyz);
        float sn = dot(Bp - rayO, vCutB.xyz);
        if (abs(dn) <= 1e-7 * rn) {
          dead = dead || (sn < 0.0);
        } else {
          float tX = sn / dn;
          if (dn > 0.0) tHi = min(tHi, tX); else tLo = max(tLo, tX);
        }
      }
      if (dead || tHi < tLo) discard;
      float sLoC = hardA ? -1e30 : -0.5 * L;
      float sHiC = hardB ? 1e30 : 0.5 * L;
      float sC = clamp(sM, sLoC, sHiC);
      vec3 qv = (M + sC * w) - rayO;
      float tHit = clamp(dot(qv, dRaw) / n2, tLo, tHi);
      vec3 pRay = rayO + dRaw * tHit;
      vec3 dv = pRay - (M + clamp(dot(pRay - M, w), sLoC, sHiC) * w);
      float dist2 = dot(dv, dv);
      float qn2 = dist2 * invSE * invSE * (1.0 / ${G.T_SQ});
      if (qn2 >= 1.0) discard;
      float beta = exp2(6.0 * sharp - 2.0);
      float qn = sqrt(qn2);
      float I = max(exp(-${G.K} * pow(qn, beta)) - ${G.C}, 0.0) * ${G.INV_ONE_MINUS_C};
      camZ = rayO.z + dRaw.z * tHit;

      float nearFade = perspectiveNearFade(uIsOrtho, camZ, nearCull);
      float brightness = I * aaComp * vSegW.w * nearFade;
      if (brightness < 1e-4) discard;

      fragColor = vec4(vNodeId, vElementId.x, brightness, vElementId.y);
      gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0);
    }
`;

export const VOLUMETRIC_LINE_PICK_SOURCE: ShaderSource = {
  name: 'line-pick-volumetric',
  webgl: {
    vertex: VOLUMETRIC_LINE_PICK_VERTEX_SHADER,
    fragment: VOLUMETRIC_LINE_PICK_FRAGMENT_SHADER,
  },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    const isOrtho = ((u.uIsOrtho?.value as number) ?? 0) === 1;
    return volumetricLinePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(u), { isOrtho });
  },
};
