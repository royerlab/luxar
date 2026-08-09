/**
 * Volumetric line primitive — GLSL backend (issue #1352, behind
 * `?linePrimitive=volumetric`).
 *
 * Renders each segment as the convolution of the segment with an isotropic
 * 3D Gaussian: ρ(x) = a · G₂D(r) · W(s). The screen quad becomes a pure
 * rasterization stencil (a "stadium": the projected segment extended
 * axially by the cap radius at each end); every shading quantity is
 * computed per-fragment against the TRUE camera-space segment, so end-on
 * viewing is exact rather than degenerate, and joins are seam-free by
 * construction:
 *
 * - SUM-family blending (additive, luminous, volumetric) integrates ρ along
 *   the view ray in closed form. Interior polyline joints use bisector-cut
 *   ends (the infinite-rod density clipped by the plane between the segment
 *   and its partner): reflection across the plane swaps the two axes, so the
 *   pair partitions the rod exactly at any bend angle — no partner blending,
 *   single-covered, C0. (The MATH is exact at any bend; the rasterized
 *   STENCIL clamps the oblique-cut overhang at min(R·tan(θ/2), R) for fill
 *   control, so bends past ~90° truncate the outermost wedge tip beyond one
 *   radius — a measured G0 trade, revisited at the G1 perf/visual gate.)
 *   Free ends / hubs / slice-clips keep the erf cap of
 *   the exact convolution. Chain-end segments (one of each) use the
 *   inclusion–exclusion closed form.
 * - PEAK-family blending (max, normal, opaque) takes the max along the ray:
 *   today's profile at the ray→clamped-segment distance (a Gaussian-shoulder
 *   capsule), exact for any sharpness β and idempotent under gl.MAX.
 *
 * The lane structure, constants, and closed forms are the SHARED ones in
 * `_shared/line-volumetric.ts`, whose CPU reference is quadrature-validated
 * in `line-volumetric-integral.test.ts` — edit the lanes THERE first, prove
 * them, then mirror here and in the TSL twin (`shader-tsl-volumetric.ts`).
 *
 * Calibration: today's fragment profile exp(−K·p²) with p in units of the
 * drawn half-width IS a truncated Gaussian with T = √(2K) and
 * σ = drawnHalfWidth / T; drawn half-width is 2 width-texel units, so
 * σ_world = width · 2/T and the side-on appearance matches the screen-space
 * primitive by construction. Sum output is normalized by σ√2π (side-on core
 * pinned; end-on brightens by path/chord — honest path integration).
 *
 * Projection split: the material stamps `LUXAR_PEAK_PROJECTION` (from
 * `usesPeakProjection(blendingMode)`) the same way it stamps the other
 * blending defines — mode changes already rebuild the program.
 *
 * Sharpness: the sum path is β=2 regardless of the sharpness knob (the
 * closed form exists only for the Gaussian); the peak path honours β
 * exactly. The sum-mode sharpness LUT is #1352 PR-4.
 */
import {
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_NEAR_FADE_FUNCTIONS,
  GLSL_SORTED_INDEX,
} from '../_shared/glsl-lib';
import type { ShaderSource } from '../_shared/shader-source';
import { buildLineTSLNodesFromUniforms } from './shader-tsl';
import { volumetricLineWebGPUFactory } from './shader-tsl-volumetric';
import { GLSL_ERF_AS_FUNCTIONS, GLSL_ERF_FUNCTIONS } from '../_shared/erf';
import {
  ALPHA_CLAMP,
  VOLUMETRIC_SERIES_C1,
  VOLUMETRIC_SERIES_C2_DIVISOR,
  VOLUMETRIC_SERIES_TAU_THRESHOLD,
  VOLUMETRIC_TAU_EPS,
} from '../_shared/volumetric';
import { FALLOFF_FLOOR, FALLOFF_K, GAUSSIAN_EQUIVALENT_TRUNCATION } from '../_shared/falloff';
import {
  LINE_PARALLEL_LANE_THRESHOLD,
  LINE_SIGMA_PER_WIDTH,
  LINE_STENCIL_DILATION,
} from '../_shared/line-volumetric';

const T = GAUSSIAN_EQUIVALENT_TRUNCATION;
// All literals toFixed(7) — float32-exact and snapshot-stable (falloff.ts note).
const G = {
  K: FALLOFF_K.toFixed(7), // 4.6051702
  C: FALLOFF_FLOOR.toFixed(7),
  INV_ONE_MINUS_C: (1 / (1 - FALLOFF_FLOOR)).toFixed(7),
  T_SQ: (T * T).toFixed(7), // 9.2103404 (= 2K)
  SIGMA_PER_WIDTH: LINE_SIGMA_PER_WIDTH.toFixed(7), // world sigma per width-texel unit
  DILATION: LINE_STENCIL_DILATION.toFixed(7),
  // T² · dilation pre-folded for the vertex-stage stencil radius floor.
  T_SQ_DILATION: (T * T * LINE_STENCIL_DILATION).toFixed(7),
  PARALLEL_THRESHOLD: LINE_PARALLEL_LANE_THRESHOLD.toExponential(1),
  INV_SQRT2: Math.SQRT1_2.toFixed(7), // 0.7071068
  TWO_OVER_SQRT_PI: (2 / Math.sqrt(Math.PI)).toFixed(7), // 1.1283792
  INV_SQRT_PI: (1 / Math.sqrt(Math.PI)).toFixed(7), // 0.5641896
  INV_SQRT_2PI: (1 / Math.sqrt(2 * Math.PI)).toFixed(7), // 0.3989423
  QUARTER_SQRT2: (1 / (2 * Math.SQRT2)).toFixed(7), // 0.3535534
};

export const VOLUMETRIC_LINE_VERTEX_SHADER = /* glsl */ `
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

    // All varyings are flat per-segment constants, written identically on
    // every vertex (WGSL provokes first-vertex, WebGL last-vertex — writing
    // segment-level values everywhere satisfies both).
    flat out vec4 vSegA;   // TRUE camera-space start.xyz (pre near-clip), .w = L
    flat out vec4 vSegW;   // unit axis.xyz, .w = coverageFade amplitude scale
    flat out vec4 vEnds;   // width0, width1, sharpness0, sharpness1 (sanitized)
    flat out vec4 vMisc;   // alpha0, alpha1, unused, unused
    // Bisector cut planes at interior joints: .xyz = the unit plane normal in
    // camera space (through the endpoint; my material on the NEGATIVE side),
    // .w = 1.0 for a hard cut / 0.0 for a soft (erf-cap) end. Built from the
    // partner segment's direction, fetched via the joint-code partner slot
    // (the #1342 plumbing — joint codes survive as the partner provider).
    flat out vec4 vCutA;
    flat out vec4 vCutB;
    #ifdef USE_COLORMAP
    flat out vec2 vScalars;
    #else
    flat out vec3 vColor0;
    flat out vec3 vColor1;
    #endif

    // Decode a joint code and, for an interior joint (a code naming a
    // partner slot), fetch the partner's endpoints and return its unit
    // into-partner direction FROM the shared vertex, in camera space.
    // Returns w = 1.0 when a valid partner direction was found.
    vec4 luxarPartnerDir(float code, int lineTexW) {
      bool interior = (code > 0.5) || (code < -2.5);
      if (!interior) return vec4(0.0);
      int slot = (code > 0.0) ? int(code + 0.5) - 1 : int(-code + 0.5) - 3;
      int pBase = slot * 6;
      ivec2 pt0 = ivec2(pBase % lineTexW, pBase / lineTexW);
      vec3 pStart = texelFetch(uLineTex, pt0, 0).xyz;
      vec3 pEnd = texelFetch(uLineTex, ivec2(pt0.x + 1, pt0.y), 0).xyz;
      // code > 0: the partner shares its START (vertex = pStart), so
      // into-partner points toward pEnd; code < -2.5: shares its END.
      vec3 q = (code > 0.0) ? (pEnd - pStart) : (pStart - pEnd);
      float qLen = length(q);
      if (qLen < 1e-20) return vec4(0.0);
      // Normalize in CAMERA space, not object space: the bisector plane is
      // built against the camera-space unit axis, and any model scaling
      // (uniform included) makes the transformed direction non-unit —
      // skewing the bisector normal and tanHalf (PR #1426 review).
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
        vMisc = vec4(1.0, 1.0, 0.0, 0.0);
        vCutA = vec4(0.0);
        vCutB = vec4(0.0);
        #ifdef USE_COLORMAP
        vScalars = vec2(0.0);
        #else
        vColor0 = vec3(0.0);
        vColor1 = vec3(0.0);
        #endif
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

      // Deferred texel fetches past the cheap cull.
      vec4 lineT2 = texelFetch(uLineTex, ivec2(texel0.x + 2, texel0.y), 0);
      vec4 lineT3 = texelFetch(uLineTex, ivec2(texel0.x + 3, texel0.y), 0);
      vec4 lineT5 = texelFetch(uLineTex, ivec2(texel0.x + 5, texel0.y), 0);

      float w0 = sanitizeNonNegative(aStartWidth, 0.0);
      float w1 = sanitizeNonNegative(aEndWidth, 0.0);
      float s0 = clamp(sanitizeNonNegative(lineT2.w, 0.5), 0.0, 1.0);
      float s1 = clamp(sanitizeNonNegative(lineT3.w, 0.5), 0.0, 1.0);
      vEnds = vec4(w0, w1, s0, s1);
      vMisc = vec4(sanitizeAlpha(lineT5.z), sanitizeAlpha(lineT5.w), 0.0, 0.0);
      #ifdef USE_COLORMAP
      vScalars = lineT5.xy;
      #else
      vColor0 = lineT2.rgb;
      vColor1 = lineT3.rgb;
      #endif

      vec3 seg = trueB - trueA;
      float segLen = length(seg);
      vec3 axisW = segLen > 1e-20 ? seg / segLen : vec3(1.0, 0.0, 0.0);
      vSegA = vec4(trueA, segLen);

      // Bisector cut planes at interior joints. m = into-me from the shared
      // vertex (+w at A, -w at B), q = into-partner; the separating plane
      // normal is normalize(q - m) (my side negative). Reflection across it
      // swaps the two axes, so both densities agree ON the plane — the cut
      // is exact at any bend angle. Fold-back (q ≈ m) degenerates to a soft
      // end. tanHalf = tan(theta/2) sizes the oblique-cut stencil overhang.
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
      // Degenerate projected direction is HARMLESS here: the stencil tends
      // to a square around a disc, and the fragment never reads the stencil
      // orientation. Any fallback direction covers the footprint.
      vec2 lineDir = pixelLen > 0.0001 ? pixelDir / pixelLen : vec2(1.0, 0.0);
      vec2 perpendicular = vec2(-lineDir.y, lineDir.x);

      // Per-end drawn half-width in pixels, at the CLIPPED endpoints, with
      // the T² · dilation variance floor folded in (replaces the 1.5px min
      // clamp; minimum stencil radius ~1.66px).
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

      // Coverage fade + hard extent clamp (gsplat pattern) replaces
      // vWidthFade and the pathological near-camera cull.
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

      // Stadium corner: perpendicular half-width plus a per-end, PER-CORNER
      // axial extension. A SOFT end (free / hub / slice-clipped) needs
      // 0.77·R (the erf cap reaches the 1% floor at 2.33·sigma). A
      // bisector-CUT end needs the oblique overhang R·tan(theta/2) only on
      // the OUTER side of the bend — the inner side's cell recedes behind
      // the plane (a small clamped safety covers the screen-side
      // first-order approximation). This is where the (L+2R)/L fill
      // explosion of the naive stadium is reclaimed.
      float R = (aQuadCorner.x > 0.0) ? RB : RA;
      float hardEnd = (aQuadCorner.x > 0.0) ? vCutB.w : vCutA.w;
      float tanHalf = (aQuadCorner.x > 0.0) ? tanHalfB : tanHalfA;
      // Outward (elbow-exterior) direction of this end's bend, first-order
      // in screen space: camera-xy directions map to pixel directions
      // under the isotropic focal length.
      vec2 out2 = (aQuadCorner.x > 0.0)
        ? (axisW.xy - qB.xy)
        : (-(axisW.xy + qA.xy));
      float side = dot(out2, perpendicular);
      bool cornerOuter = (aQuadCorner.y * side) > 0.0;
      float overhang = min(R * tanHalf, R);
      // Depth-tilt disc reach: the cell's cross-section at the endpoint is a
      // 3D disc of radius R perpendicular to the axis, and when the axis
      // tilts into depth that disc PROJECTS past the endpoint by
      // R·|cos(axis, viewDir)| — in ortho and perspective alike. Without
      // this term, rays just past the projected endpoint that still hit the
      // cell in depth never rasterise: a dark hairline at every joint of a
      // depth-tilted thick polyline (found on the flag-on smoke, invisible
      // in face-on views like the G0 spike's). Zero when face-on, so the
      // fill-cost profile of the bisector-cut reclaim is unchanged there.
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

export const VOLUMETRIC_LINE_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;
    ${GLSL_NEAR_FADE_FUNCTIONS}
    ${GLSL_ERF_FUNCTIONS}
    ${GLSL_ERF_AS_FUNCTIONS}

    // Ψ(x) = ∫ₓ^∞ ½(1 − erf(u)) du — the axial cap remainder for the
    // structurally-parallel mixed lane. The saturation guards are
    // load-bearing: an unbounded plane crossing feeds ±1e30 here, and
    // x·(1−erf(x)) must not turn a saturated erf residual into garbage.
    float luxarErfCapRemainder(float x) {
      if (x > 6.0) return 0.0;
      if (x < -6.0) return -x;
      return 0.5 * (exp(-x * x) * ${G.INV_SQRT_PI} - x * (1.0 - luxarErfAS(x)));
    }

    uniform int uIsOrtho;
    uniform float uNearCull;
    uniform vec2 uResolution;
    uniform float uPerspectiveLineScale;
    uniform float uOrthoLineScale;
    uniform float uOpacity;
    uniform float uInvGamma;
    uniform float uIntensity;
    uniform float uOffset;
    uniform highp float uAbsorption;
    uniform lowp float uHasElementAlpha;
    #ifdef USE_COLORMAP
    uniform sampler2D uColormapTex;
    uniform float uScalarMin;
    uniform float uScalarScale;
    #endif

    flat in vec4 vSegA;
    flat in vec4 vSegW;
    flat in vec4 vEnds;
    flat in vec4 vMisc;
    flat in vec4 vCutA;
    flat in vec4 vCutB;
    #ifdef USE_COLORMAP
    flat in vec2 vScalars;
    #else
    flat in vec3 vColor0;
    flat in vec3 vColor1;
    #endif

    out vec4 fragColor;

    void main() {
      // Per-fragment view ray in camera space, UNNORMALIZED (dRaw) — the
      // solver folds |dRaw| in analytically, so there is no normalize().
      // Focal length in pixels = 0.5 * uPerspectiveLineScale, isotropic.
      vec3 dRaw;
      vec3 rayO;
      if (uIsOrtho == 1) {
        rayO = vec3((gl_FragCoord.xy - 0.5 * uResolution) * (2.0 / uOrthoLineScale), 0.0);
        dRaw = vec3(0.0, 0.0, -1.0);
      } else {
        rayO = vec3(0.0);
        dRaw = vec3((gl_FragCoord.xy - 0.5 * uResolution) * (2.0 / uPerspectiveLineScale), -1.0);
      }

      // Ray-segment solver in unnormalized-ray form, midpoint-relative.
      //   sM  = (bw*n2 - dw*bdr) / A      A = n2 - dw^2  (= n2 * sin^2(angle))
      //   tC*d = dRaw * (dw*bw - bdr)/A   (the closest-approach point offset)
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
      // Structural parallel lane (RELATIVE threshold): the general lane's
      // sM is meaningless at A == 0 (0/0), and the mixed lane's erf
      // subtraction loses float32 significance as A → 0 — see
      // LINE_PARALLEL_LANE_THRESHOLD in _shared/line-volumetric.ts.
      bool parallel = A < ${G.PARALLEL_THRESHOLD} * n2;
      float sM;
      float D2;
      float camZ;
      float tCenter;  // raw-ray parameter of the closest approach
      if (parallel) {
        float invN2 = 1.0 / n2;
        vec3 r0 = b - dRaw * (bdr * invN2);
        float rw = dot(r0, w);
        D2 = max(dot(r0, r0) - rw * rw, 0.0);
        sM = 0.0;
        tCenter = -bdr * invN2;
        camZ = rayO.z + dRaw.z * tCenter;
      } else {
        float invA = 1.0 / A;
        sM = (bw * n2 - dw * bdr) * invA;
        tCenter = (dw * bw - bdr) * invA;
        vec3 pv = b + dRaw * tCenter - w * sM;
        D2 = dot(pv, pv);
        camZ = rayO.z + dRaw.z * tCenter;
      }

      // Per-fragment attributes at the clamped closest-approach axial coord.
      float sHat = clamp(sM / max(L, 1e-20) + 0.5, 0.0, 1.0);
      float width = mix(vEnds.x, vEnds.y, sHat);
      float sharp = mix(vEnds.z, vEnds.w, sHat);
      float alphaEl = mix(vMisc.x, vMisc.y, sHat);

      // sigma with the Mip-Splatting-style 3D variance floor at the hit
      // depth; aaComp is the matching energy compensation (replaces the
      // 1.5px clamp + widthScale pair).
      float nearCull = max(uNearCull, 1e-20);
      float pxSize = (uIsOrtho == 1)
        ? (2.0 / uOrthoLineScale)
        : (2.0 * max(-camZ, nearCull) / uPerspectiveLineScale);
      float sigma = ${G.SIGMA_PER_WIDTH} * width;
      float invSE = inversesqrt(sigma * sigma + ${G.DILATION} * pxSize * pxSize + 1e-30);
      float aaComp = sigma * invSE;

      // EARLY radial reject, before any window math (r2n is in sigma^2
      // units; truncation at T^2 matches the stencil radius).
      float r2n = D2 * (invSE * invSE);

      float I;
      #ifdef LUXAR_PEAK_PROJECTION
      // PEAK family (max, normal, opaque): Gaussian-shoulder capsule —
      // today's profile at the ray-to-clamped-segment distance. Exact for
      // any sharpness beta. No parallel lane needed: point-to-ray-line
      // distance is s-independent for parallel geometry, so a clamped
      // garbage sM still yields the right distance.
      float sC = clamp(sM, -0.5 * L, 0.5 * L);
      vec3 qv = (M + sC * w) - rayO;
      float proj = dot(qv, dRaw);
      float dist2 = max(dot(qv, qv) - proj * proj / n2, 0.0);
      float qn2 = dist2 * invSE * invSE * (1.0 / ${G.T_SQ});
      if (qn2 >= 1.0) discard;
      float beta = exp2(6.0 * sharp - 2.0);
      float qn = sqrt(qn2);
      I = max(exp(-${G.K} * pow(qn, beta)) - ${G.C}, 0.0) * ${G.INV_ONE_MINUS_C};
      camZ = rayO.z + dRaw.z * (proj / n2);
      #else
      // SUM family: normalized closed-form ray integral (I = ∫ρ du / σ√2π).
      // Four lanes, quadrature-validated against the CPU reference in
      // _shared/line-volumetric.ts — keep the three implementations in
      // lockstep (reference / GLSL / TSL).
      if (r2n >= ${G.T_SQ}) discard;
      float radial = max(exp(-0.5 * r2n) - ${G.C}, 0.0) * ${G.INV_ONE_MINUS_C};
      bool hardA = vCutA.w > 0.5;
      bool hardB = vCutB.w > 0.5;
      float sAtCenter = sM + 0.5 * L;  // endpoint-coords axial closest approach
      float F;
      if (parallel) {
        // STRUCTURAL PARALLEL: the ray runs along the axis, the radial
        // factor is constant, and the integral reduces to the axial
        // window's length G_len (soft caps via the Ψ remainder, plane cuts
        // as sharp s-bounds). I = radial · G_len · invSE / √2π.
        float c = invSE * ${G.INV_SQRT2};
        float sLo = -1e30;
        float sHi = 1e30;
        bool dead = false;
        if (hardA) {
          float dn = dot(dRaw, vCutA.xyz);
          float sn = dot(vSegA.xyz - rayO, vCutA.xyz);
          if (abs(dn) <= 1e-7 * rn) {
            dead = dead || (sn < 0.0);
          } else {
            float sX = sAtCenter + (sn / dn - tCenter) * dw;
            if ((dn > 0.0) == (dw > 0.0)) sHi = min(sHi, sX); else sLo = max(sLo, sX);
          }
        }
        if (hardB) {
          vec3 Bp = vSegA.xyz + w * L;
          float dn = dot(dRaw, vCutB.xyz);
          float sn = dot(Bp - rayO, vCutB.xyz);
          if (abs(dn) <= 1e-7 * rn) {
            dead = dead || (sn < 0.0);
          } else {
            float sX = sAtCenter + (sn / dn - tCenter) * dw;
            if ((dn > 0.0) == (dw > 0.0)) sHi = min(sHi, sX); else sLo = max(sLo, sX);
          }
        }
        float Glen;
        if (dead || sHi <= sLo) {
          Glen = 0.0;
        } else if (!hardA && !hardB) {
          Glen = L; // the soft window integrates to exactly L over the line
        } else if (hardA && hardB) {
          Glen = max(sHi - sLo, 0.0);
        } else if (hardA) {
          // Soft cap at B: ∫ ½(1−erf((s−L)c)) ds over [sLo, sHi].
          Glen = (luxarErfCapRemainder((sLo - L) * c) - luxarErfCapRemainder((sHi - L) * c)) / c;
        } else {
          // Soft cap at A: mirror (window ½(1+erf(s·c))).
          Glen = (luxarErfCapRemainder(-sHi * c) - luxarErfCapRemainder(-sLo * c)) / c;
        }
        if (Glen <= 0.0) discard;
        F = Glen * invSE * ${G.INV_SQRT_2PI};
      } else if (hardA || hardB) {
        // GENERAL PLANE LANE: fold each present bisector plane into the
        // ξ-interval [xiLo, xiHi]; hard/hard is exact, mixed adds the
        // inclusion–exclusion cap treatment (A&S erf — pref amplifies erf
        // error unboundedly near-axial, and this lane's population is only
        // the two chain-end segments of each polyline).
        float xiLo = -4.0;
        float xiHi = 4.0;
        float kxi = sqrt(A) * invSE * ${G.INV_SQRT2};
        bool dead = false;
        if (hardA) {
          float dn = dot(dRaw, vCutA.xyz);
          float sn = dot(vSegA.xyz - rayO, vCutA.xyz);
          if (abs(dn) > 1e-7 * rn) {
            float xi = clamp((sn / dn - tCenter) * kxi, -4.0, 4.0);
            if (dn > 0.0) xiHi = min(xiHi, xi); else xiLo = max(xiLo, xi);
          } else if (sn < 0.0) {
            dead = true;
          }
        }
        if (hardB) {
          vec3 Bp = vSegA.xyz + w * L;
          float dn = dot(dRaw, vCutB.xyz);
          float sn = dot(Bp - rayO, vCutB.xyz);
          if (abs(dn) > 1e-7 * rn) {
            float xi = clamp((sn / dn - tCenter) * kxi, -4.0, 4.0);
            if (dn > 0.0) xiHi = min(xiHi, xi); else xiLo = max(xiLo, xi);
          } else if (sn < 0.0) {
            dead = true;
          }
        }
        if (dead || xiLo >= xiHi) discard;
        float xim = 0.5 * (xiLo + xiHi);
        float dxi = xiHi - xiLo;
        float pref = 0.5 * rn * inversesqrt(max(A, 1e-12 * n2));
        // erf(ξHi) − erf(ξLo), through the widened-midpoint Taylor lane
        // when the interval is narrow (thin hard/hard wedges):
        //   E = (2/√π) e^(−xm²) [1 + dxi² (4xm² − 2)/24] + O(dxi⁴)
        bool narrow = dxi < 0.5;
        float bracketTaylor = 0.0;
        if (narrow) {
          float xim2 = min(xim * xim, 80.0);
          bracketTaylor =
            ${G.TWO_OVER_SQRT_PI} * exp(-xim2) * (1.0 + dxi * dxi * (4.0 * xim2 - 2.0) * (1.0 / 24.0)) * dxi;
        }
        if (hardA && hardB) {
          float bracket = narrow ? bracketTaylor : (luxarErf(xiHi) - luxarErf(xiLo));
          F = pref * max(bracket, 0.0);
        } else {
          // MIXED (one hard cut, one soft cap). Two inclusion–exclusion
          // splits with COMPLEMENTARY error domains, selected by one sign:
          //   J1 = [plane-clipped rod] − [full-line cap complement]
          //   J2 = [cap only, plane ignored]
          // J1 drops the cap-COMPLEMENT's mass on the ray's excluded side
          // of the plane; J2 drops the CAP's. For axis-dominant rays the
          // excluded side points along ±s and exactly one split is
          // exponentially exact. Residual error survives only when the
          // plane sits within ~3σ of the cap (short chain-end segments,
          // sharp bends) — measured and pinned in the unit tests.
          float kk = kxi / rn;
          vec3 hardN = hardA ? vCutA.xyz : vCutB.xyz;
          float dnH = dot(dRaw, hardN);
          bool axialDominant = dw * dw > A;
          bool excludedTowardMinusS = (dnH * dw) < 0.0;
          bool complementOnExcludedSide = hardA ? !excludedTowardMinusS : excludedTowardMinusS;
          float xCapB = (sAtCenter - L) * kk;  // B-edge in identity coords
          float xCapA = sAtCenter * kk;        // A-edge in identity coords
          if (axialDominant && complementOnExcludedSide) {
            float capOnly = hardA ? (1.0 - luxarErfAS(xCapB)) : (1.0 + luxarErfAS(xCapA));
            F = pref * max(capOnly, 0.0);
          } else {
            float bracket = narrow ? bracketTaylor : (luxarErfAS(xiHi) - luxarErfAS(xiLo));
            float capTerm = hardA ? (1.0 + luxarErfAS(xCapB)) : (1.0 - luxarErfAS(xCapA));
            F = pref * max(bracket - capTerm, 0.0);
          }
        }
      } else {
        // SOFT/SOFT (free ends, hubs, slice-clipped, isolated segments):
        // the erf-cap closed form, full axial stencil support.
        float kk = sqrt(A) * inversesqrt(n2) * invSE * ${G.INV_SQRT2};
        float xm = sM * kk;
        float dx = L * kk;           // = x1 - x0, non-cancelling product form
        float x0 = xm - 0.5 * dx;
        float x1 = xm + 0.5 * dx;
        // EARLY axial reject, in erf-argument units — SAFE near-parallel
        // (kk -> 0 there, so the args -> 0 and this never fires; contrast
        // the raw-sM discard, which would wrongly kill those fragments).
        if (x0 > 3.0 || x1 < -3.0) discard;
        // Widened midpoint lane (dx < 0.5, with the dx^2 correction from
        // the erf Taylor series) bounds the polynomial-difference error:
        //   E = (2/sqrt(pi)) e^(-xm^2) [1 + dx^2 (4 xm^2 - 2)/24] + O(dx^4)
        float E;
        if (dx < 0.5) {
          float xm2 = min(xm * xm, 80.0);
          E = ${G.TWO_OVER_SQRT_PI} * exp(-xm2) * (1.0 + dx * dx * (4.0 * xm2 - 2.0) * (1.0 / 24.0));
        } else {
          E = (luxarErf(x1) - luxarErf(x0)) / dx;
        }
        F = max(E, 0.0) * L * invSE * ${G.QUARTER_SQRT2};
      }
      I = radial * max(F, 0.0);
      #endif

      float nearFade = perspectiveNearFade(uIsOrtho, camZ, nearCull);
      float intensity = I * aaComp * vSegW.w * nearFade;

      #ifdef USE_COLORMAP
      float s = mix(vScalars.x, vScalars.y, sHat);
      float st = clamp((s - uScalarMin) * uScalarScale, 0.0, 1.0);
      #ifndef LUXAR_GAMMA_ONE
      st = pow(st, uInvGamma);
      #endif
      vec3 color = texture(uColormapTex, vec2(st, 0.5)).rgb;
      #else
      vec3 color = mix(vColor0, vColor1, sHat);
      #endif

      #ifdef LUXAR_NO_GOG
      vec3 adjusted = color;
      #else
      vec3 adjusted = color * uIntensity + uOffset;
      adjusted = max(adjusted, vec3(0.0));
      #endif

      #ifdef LUXAR_VOLUMETRIC
      float alpha = intensity * uOpacity;
      alpha *= mix(1.0, -log(1.0 - min(alphaEl, ${ALPHA_CLAMP})), uHasElementAlpha);
      float tau = uAbsorption * alpha;
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4 && tau < 1e-4) discard;
      #else
      intensity *= alphaEl;
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;
      #endif

      #if defined(USE_COLORMAP) || defined(LUXAR_GAMMA_ONE)
      vec3 gammaColor = adjusted;
      #else
      vec3 gammaColor = pow(adjusted, vec3(uInvGamma));
      #endif

      #if defined(LUXAR_VOLUMETRIC)
      float volAlpha = 1.0 - exp(-tau);
      float screen = (tau < ${VOLUMETRIC_SERIES_TAU_THRESHOLD}) ? 1.0 - ${VOLUMETRIC_SERIES_C1} * tau + tau * tau / ${VOLUMETRIC_SERIES_C2_DIVISOR}.0
                                  : volAlpha / max(tau, ${VOLUMETRIC_TAU_EPS});
      fragColor = vec4(gammaColor * alpha * screen, volAlpha);
      #elif defined(LUXAR_MAX_RGB_CONTRIBUTION)
      float a = intensity * uOpacity;
      fragColor = vec4(gammaColor * a, a);
      #else
      fragColor = vec4(gammaColor, intensity * uOpacity);
      #endif
    }
  `;

/**
 * ShaderSource registry entry for the volumetric primitive — the second
 * LINE source (`LINE_SOURCE` in `shader-glsl.ts` is the screen-space
 * quad). The parity harness points fixtures at whichever source matches
 * the primitive under test; the material factories select between the
 * two pairs directly.
 *
 * The webgpu bridge reads `uIsOrtho` from the uniform record at build
 * time (same convention as LINE_SOURCE); blending/colormap toggles are
 * for direct `volumetricLineWebGPUFactory` callers, as the parity
 * fixtures do.
 */
export const VOLUMETRIC_LINE_SOURCE: ShaderSource = {
  name: 'line-volumetric',
  webgl: {
    vertex: VOLUMETRIC_LINE_VERTEX_SHADER,
    fragment: VOLUMETRIC_LINE_FRAGMENT_SHADER,
  },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    const isOrtho = ((u.uIsOrtho?.value as number) ?? 0) === 1;
    return volumetricLineWebGPUFactory(buildLineTSLNodesFromUniforms(u, {}), { isOrtho });
  },
};
