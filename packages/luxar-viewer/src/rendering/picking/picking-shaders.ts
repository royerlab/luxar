/**
 * Picking shader sources for the three first-class geometries.
 *
 * Each shader pair is based on its visual counterpart in
 * `rendering/shaders/{point,line,gsplat}-shaders.ts` with the
 * following additions:
 *   - `uNodeId` uniform + `vNodeId` / `vElementId` varyings, written
 *     into the RGBA32F pick buffer as `(nodeId, elementId, brightness, 1)`.
 *   - Tighter truncation for points (50% radius) and gsplats (1.5σ
 *     instead of 3σ). Lines use full width — they're already narrow.
 *   - Brightness-as-depth so the brightest element at each pixel
 *     wins the depth test (essential for picking through translucent
 *     splats and overlapping line segments).
 *
 * Source-of-truth lives here; the per-material wrappers in
 * `*-picking-material.ts` consume these `ShaderSource` values. When
 * GLSL→TSL porting starts, the `webgpu?` field on each source is
 * filled in alongside.
 *
 * @module rendering/picking/picking-shaders
 */

import type { ShaderSource } from '../shaders/shader-source';
import { GLSL_SANITIZE_FUNCTIONS } from '../shaders/glsl-lib';
import { pointPickWebGPUFactory, buildPointPickTSLNodesFromUniforms } from './point-pick.tsl';
import { linePickWebGPUFactory, buildLinePickTSLNodesFromUniforms } from './line-pick.tsl';
import { gsplatPickWebGPUFactory, buildGSplatPickTSLNodesFromUniforms } from './gsplat-pick.tsl';

// ---------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------

/**
 * Picking vertex shader for points.
 * Adds uNodeId uniform and vNodeId/vElementId outputs.
 */
export const POINT_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}

    // Per-vertex (4 corners shared across all instances)
    in vec2 aQuadCorner;

    // Per-instance (one per point)
    in vec3 aCenter;
    in float aRadius;
    in float aSharpness;

    uniform float pointSizeFactor;
    uniform float maxPointSize;
    uniform float radiusScale;
    uniform float sharpnessScale;
    uniform int uIsOrtho;
    uniform float uNodeId;
    uniform vec2 uResolution;

    out highp float vRadius;
    out mediump float vSharpness;
    out mediump vec2 vSpriteCoord;
    flat out highp float vNodeId;
    flat out highp float vElementId;

    void main() {
      // Mirror visual point shader sanitization (point-shaders.ts:69)
      // so a NaN/Inf sharpness or negative radius can't cause the pick
      // footprint to diverge from the visible footprint.
      float normalizedSharpness = sanitizePositive(aSharpness * sharpnessScale, 2.0);
      vSharpness = normalizedSharpness;

      float normalizedRadius = sanitizeNonNegative(aRadius * radiusScale, 0.0);
      vRadius = normalizedRadius;

      vec4 mvPosition = modelViewMatrix * vec4(aCenter, 1.0);
      vec4 projCenter = projectionMatrix * mvPosition;

      float invDistance = (uIsOrtho == 1) ? 1.0 : inversesqrt(dot(mvPosition.xyz, mvPosition.xyz));
      float basePointSize = normalizedRadius * pointSizeFactor * invDistance;

      // Tighter truncation for picking: 50% of visual radius
      // Use sharpness compensation * 0.5 so we only pick the bright core.
      // Mirror visual point shader's invalid-result guard so a degenerate
      // vSharpness (e.g. ≪0.01 after clamp) can't poison pointSize.
      float sharpnessCompensationRaw = 1.0 / (1.0 - pow(0.01, 1.0 / max(vSharpness, 0.01)));
      float sharpnessCompensation = isInvalidFloat(sharpnessCompensationRaw) ? 1.0 : sharpnessCompensationRaw;
      float pointSize = basePointSize * sharpnessCompensation * 0.5;
      pointSize = max(1.0, min(pointSize, maxPointSize));

      // Instanced quad expansion (matches point-shaders.ts approach).
      vec2 offsetClip = aQuadCorner * (pointSize / uResolution) * projCenter.w;
      gl_Position = projCenter + vec4(offsetClip, 0.0, 0.0);

      vSpriteCoord = (aQuadCorner + 1.0) * 0.5;

      vNodeId = uNodeId;
      // Under instanced rendering, gl_InstanceID is the per-point index
      // (the old THREE.Points path used gl_VertexID which was equivalent).
      vElementId = float(gl_InstanceID);
    }
`;

/**
 * Picking fragment shader for points.
 * Outputs vec4(nodeId, elementId, brightness, 1.0) with brightness-as-depth.
 */
export const POINT_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    in highp float vRadius;
    in mediump float vSharpness;
    in mediump vec2 vSpriteCoord;
    flat in highp float vNodeId;
    flat in highp float vElementId;

    out vec4 fragColor;

    void main() {
      if (vRadius < 0.0001) discard;

      vec2 centered = vSpriteCoord - 0.5;
      float r2 = dot(centered, centered);

      // Full circle discard (gl_PointSize is already halved in vertex shader for tighter picking)
      if (r2 > 0.25) discard;

      float normalizedR = sqrt(4.0 * r2);
      float falloff = pow(max(1.0 - normalizedR, 0.0), vSharpness);

      float brightness = falloff;
      if (brightness < 1e-4) discard;

      fragColor = vec4(vNodeId, vElementId, brightness, 1.0);
      gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0);
    }
`;

export const POINT_PICK_SOURCE: ShaderSource = {
  name: 'point-pick',
  webgl: { vertex: POINT_PICK_VERTEX_SHADER, fragment: POINT_PICK_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    return pointPickWebGPUFactory(buildPointPickTSLNodesFromUniforms(u));
  },
};

// ---------------------------------------------------------------------
// Lines
// ---------------------------------------------------------------------

/**
 * Picking vertex shader for lines.
 * Adds uNodeId uniform and vNodeId/vElementId outputs.
 * Strips colormap and color varyings (not needed for picking).
 * Mirrors the visual line shader's near-plane safety, max-pixel-width
 * clamp, and width/sharpness sanitization for pick/visual parity.
 */
export const LINE_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    in vec2 aQuadCorner;

    // Instanced attributes (per segment)
    in vec3 aStartPos;
    in vec3 aEndPos;
    in float aStartWidth;
    in float aEndWidth;
    in float aStartSharpness;
    in float aEndSharpness;
    in float aSegmentLength;
    in float aStartClipped;
    in float aEndClipped;

    uniform float uFOV;
    uniform vec2 uResolution;
    uniform int uIsOrtho;
    uniform float uNodeId;
    uniform float uNearCull;          // visual-shader parity
    uniform float uMaxLinePixelWidth; // visual-shader parity
    // CPU-precomputed pixel-width scales — visual-shader parity.
    uniform float uPerspectiveLineScale; // = resolution.y / tan(fov * 0.5)
    uniform float uOrthoLineScale;       // = 2 * resolution.y / frustumHeight

    out float vSharpness;
    out float vPerpNorm;
    out float vT;             // fragment-side cap math (parity with visual)
    flat out float vSegmentLength; // per-segment constant — visual-shader parity
    out float vWidthAtT;
    out float vPixelWidth;
    out float vWidthFade;     // visual-shader parity
    flat out float vClippedStart;
    flat out float vClippedEnd;
    flat out highp float vNodeId;
    flat out highp float vElementId;

    void main() {
      // Branchless: aQuadCorner.x ∈ {-1, +1} by construction.
      float t = aQuadCorner.x * 0.5 + 0.5;
      vT = t;
      vSegmentLength = aSegmentLength;
      vClippedStart = aStartClipped;
      vClippedEnd = aEndClipped;

      // sanitize width/sharpness against negative/NaN/Inf.
      float startW = (isnan(aStartWidth) || isinf(aStartWidth) || aStartWidth < 0.0) ? 0.0 : aStartWidth;
      float endW = (isnan(aEndWidth) || isinf(aEndWidth) || aEndWidth < 0.0) ? 0.0 : aEndWidth;
      float startS = (isnan(aStartSharpness) || isinf(aStartSharpness) || aStartSharpness <= 0.0) ? 2.0 : aStartSharpness;
      float endS = (isnan(aEndSharpness) || isinf(aEndSharpness) || aEndSharpness <= 0.0) ? 2.0 : aEndSharpness;

      float width = mix(startW, endW, t);
      vSharpness = mix(startS, endS, t);
      vWidthAtT = width;

      vec4 mvStart = modelViewMatrix * vec4(aStartPos, 1.0);
      vec4 mvEnd = modelViewMatrix * vec4(aEndPos, 1.0);
      vec4 mvPos = mix(mvStart, mvEnd, t);

      // Visual-shader parity: near-plane safety (degenerate quad if both endpoints behind).
      float nearCull = max(uNearCull, 1e-4);
      float startDepth = -mvStart.z;
      float endDepth = -mvEnd.z;
      bool bothBehind = (startDepth < nearCull) && (endDepth < nearCull);
      if (bothBehind) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vPerpNorm = 0.0;
        vPixelWidth = 0.0;
        vWidthFade = 0.0;
        vNodeId = uNodeId;
        vElementId = float(gl_InstanceID);
        return;
      }

      vec4 clipStart = projectionMatrix * mvStart;
      vec4 clipEnd = projectionMatrix * mvEnd;
      // projection is linear, so proj * mix(a,b,t) == mix(proj*a, proj*b, t).
      vec4 clipPos = mix(clipStart, clipEnd, t);

      float wStart = max(clipStart.w, 1e-4);
      float wEnd = max(clipEnd.w, 1e-4);
      vec2 ndcStart = clipStart.xy / wStart;
      vec2 ndcEnd = clipEnd.xy / wEnd;

      // The +0.5 in (ndc*0.5+0.5)*resolution cancels under subtraction.
      vec2 pixelDir = (ndcEnd - ndcStart) * (0.5 * uResolution);
      float pixelLen = length(pixelDir);
      vec2 lineDir = pixelLen > 0.0001 ? pixelDir / pixelLen : vec2(1.0, 0.0);
      vec2 perpendicular = vec2(-lineDir.y, lineDir.x);

      float rawPixelWidth;
      if (uIsOrtho == 1) {
        rawPixelWidth = width * uOrthoLineScale;
      } else {
        // View-space depth: drops a sqrt, more projection-correct.
        // Visual-shader parity — see line-shaders.ts.
        float dist = max(-mvPos.z, nearCull);
        rawPixelWidth = width * uPerspectiveLineScale / dist;
      }

      float minPixelWidth = 1.5;
      float maxPW = max(uMaxLinePixelWidth, minPixelWidth + 1.0);
      // Visual-shader parity: discard pathological near-camera segments
      // (both endpoints inside near-cull margin AND rawPixelWidth blows
      // past clamp by 2×). Without this, picking still rasterizes the
      // half-viewport quad the visual pass already culled.
      if (
        startDepth < nearCull * 2.0 &&
        endDepth < nearCull * 2.0 &&
        rawPixelWidth > maxPW * 2.0
      ) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vPerpNorm = 0.0;
        vPixelWidth = 0.0;
        vWidthFade = 0.0;
        vNodeId = uNodeId;
        vElementId = float(gl_InstanceID);
        return;
      }
      float clampedPixelWidth = clamp(rawPixelWidth, minPixelWidth, maxPW);
      vWidthFade = (rawPixelWidth <= maxPW) ? 1.0 : (maxPW / max(rawPixelWidth, 1e-4));
      vPixelWidth = rawPixelWidth;
      vPerpNorm = aQuadCorner.y;

      vec2 pixelOffset = perpendicular * aQuadCorner.y * clampedPixelWidth;
      vec2 ndcOffset = pixelOffset / uResolution * 2.0;
      clipPos.xy += ndcOffset * clipPos.w;

      gl_Position = clipPos;

      vNodeId = uNodeId;
      vElementId = float(gl_InstanceID);
    }
`;

/**
 * Picking fragment shader for lines.
 * Outputs vec4(nodeId, elementId, brightness, 1.0) with brightness-as-depth.
 *
 * Lines use FULL width for picking (same as visual) — unlike points/splats,
 * lines are already narrow with a sharp parabolic profile. Tighter truncation
 * would make thin lines nearly impossible to pick. The brightness-weighted
 * voting handles overlap correctly (centerline brightness always wins).
 *
 * Cap factor is computed in fragment to match the visual shader.
 */
export const LINE_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    in float vSharpness;
    in float vPerpNorm;
    in float vT;
    flat in float vSegmentLength; // per-segment constant
    in float vWidthAtT;
    in float vPixelWidth;
    in float vWidthFade;
    flat in float vClippedStart;
    flat in float vClippedEnd;
    flat in highp float vNodeId;
    flat in highp float vElementId;

    out vec4 fragColor;

    void main() {
      float p = abs(vPerpNorm);

      // Full width — lines are already narrow, no need for tighter truncation
      if (p >= 1.0) discard;

      // Sharpness fast path — visual-shader parity. See line-shaders.ts.
      float oneMinusPSq = max(1.0 - p * p, 0.0);
      #ifdef LUXAR_SHARPNESS_TWO
      float perpFalloff = oneMinusPSq * oneMinusPSq;
      #else
      float perpFalloff = pow(oneMinusPSq, max(vSharpness, 0.0001));
      #endif

      float minPixelWidth = 1.5;
      float widthScale = min(vPixelWidth / minPixelWidth, 1.0);

      // cap factor in fragment (matches visual shader).
      float distFromStart = vT * vSegmentLength;
      float distFromEnd = (1.0 - vT) * vSegmentLength;
      float distToNearest = min(distFromStart, distFromEnd);
      float capRamp = vWidthAtT > 1e-4
        ? clamp(distToNearest / vWidthAtT, 0.0, 1.0)
        : 1.0;
      float baseCap = 0.5 + 0.5 * capRamp;
      // step(distFromStart, distFromEnd) is 1 when start is closer (distFromEnd >= distFromStart).
      float nearestIsStart = step(distFromStart, distFromEnd);
      float nearestClipped = mix(vClippedEnd, vClippedStart, nearestIsStart);
      float capFactor = mix(baseCap, 1.0, nearestClipped);

      float brightness = capFactor * perpFalloff * widthScale * vWidthFade;
      if (brightness < 1e-4) discard;

      fragColor = vec4(vNodeId, vElementId, brightness, 1.0);
      gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0);
    }
`;

export const LINE_PICK_SOURCE: ShaderSource = {
  name: 'line-pick',
  webgl: { vertex: LINE_PICK_VERTEX_SHADER, fragment: LINE_PICK_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    const isOrtho = ((u.uIsOrtho?.value as number) ?? 0) === 1;
    return linePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(u), { isOrtho });
  },
};

// ---------------------------------------------------------------------
// GSplats
// ---------------------------------------------------------------------

/**
 * Picking vertex shader for gsplats.
 * Adds uNodeId/vNodeId/vElementId, strips colormap.
 * Uses tighter truncation (1.5σ) for more precise picking.
 */
export const GSPLAT_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    in vec2 aQuadCorner;

    in vec3 aCenter;
    in vec2 aCholesky01;
    in vec2 aCholesky23;
    in vec2 aCholesky45;
    in float aAmplitude;

    uniform vec2 uResolution;
    uniform float uFx, uFy;
    uniform float uTruncate;
    uniform float uTruncateSq;
    uniform int uIsOrtho;
    uniform float uNearCull;
    uniform float uMaxExtentFactor;
    uniform float uNodeId;

    flat out mediump float vAmplitude2D;
    flat out highp vec3 vL2D;
    flat out highp vec2 vCenterScreen;
    flat out highp float vNodeId;
    flat out highp float vElementId;

    mat3 unpackCholesky3D() {
        return mat3(
            aCholesky01.x, aCholesky01.y, aCholesky23.y,
            0.0,           aCholesky23.x, aCholesky45.x,
            0.0,           0.0,           aCholesky45.y
        );
    }

    vec3 cholesky2x2(mat2 S) {
        float L00 = sqrt(max(S[0][0], 1e-8));
        float invL00 = 1.0 / L00;
        float L10 = S[1][0] * invL00;
        float L11 = sqrt(max(S[1][1] - L10 * L10, 1e-8));
        float invL11 = 1.0 / L11;
        return vec3(invL00, L10, invL11);
    }

    void main() {
        vec4 centerCam4 = modelViewMatrix * vec4(aCenter, 1.0);
        vec3 centerCam = centerCam4.xyz;

        if (centerCam.z >= 0.0) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }

        mat3 R = mat3(modelViewMatrix);
        mat3 L3D = unpackCholesky3D();
        mat3 L_cam = R * L3D;
        mat3 Sigma_cam = L_cam * transpose(L_cam);

        float zDepth = -centerCam.z;

        float depthFade = 1.0;
        if (uIsOrtho == 0) {
            depthFade = smoothstep(uNearCull, uNearCull * 2.0, zDepth);
            if (depthFade < 0.01) {
                gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
                return;
            }
        }

        float coverageFade = 1.0;
        if (uIsOrtho == 0) {
            float maxLateralVar = max(Sigma_cam[0][0], max(Sigma_cam[1][1], Sigma_cam[2][2]));
            if (maxLateralVar > 0.01) {
                float projectedExtent = uFx * sqrt(maxLateralVar) * uTruncate / zDepth;
                float maxExtent = max(uResolution.x, uResolution.y) * uMaxExtentFactor;
                coverageFade = 1.0 - smoothstep(maxExtent * 0.5, maxExtent, projectedExtent);
                if (coverageFade < 0.01) {
                    gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
                    return;
                }
            }
        }

        float nearFade = min(depthFade, coverageFade);

        float invZ = 1.0 / zDepth;
        float invZ2 = invZ * invZ;

        mat3x2 J;
        if (uIsOrtho == 1) {
            J[0] = vec2(uFx, 0.0);
            J[1] = vec2(0.0, uFy);
            J[2] = vec2(0.0, 0.0);
        } else {
            J[0] = vec2(uFx * invZ, 0.0);
            J[1] = vec2(0.0, uFy * invZ);
            J[2] = vec2(uFx * centerCam.x * invZ2, uFy * centerCam.y * invZ2);
        }

        vec2 JS0 = J[0] * Sigma_cam[0][0] + J[1] * Sigma_cam[0][1] + J[2] * Sigma_cam[0][2];
        vec2 JS1 = J[0] * Sigma_cam[1][0] + J[1] * Sigma_cam[1][1] + J[2] * Sigma_cam[1][2];
        vec2 JS2 = J[0] * Sigma_cam[2][0] + J[1] * Sigma_cam[2][1] + J[2] * Sigma_cam[2][2];

        mat2 Sigma2D;
        Sigma2D[0][0] = JS0.x * J[0].x + JS1.x * J[1].x + JS2.x * J[2].x;
        Sigma2D[1][0] = JS0.x * J[0].y + JS1.x * J[1].y + JS2.x * J[2].y;
        Sigma2D[0][1] = Sigma2D[1][0];
        Sigma2D[1][1] = JS0.y * J[0].y + JS1.y * J[1].y + JS2.y * J[2].y;

        // For picking, always use max projection (no ray integration needed)
        vAmplitude2D = aAmplitude * nearFade;

        vL2D = cholesky2x2(Sigma2D);

        float trace = Sigma2D[0][0] + Sigma2D[1][1];
        float det = Sigma2D[0][0] * Sigma2D[1][1] - Sigma2D[0][1] * Sigma2D[1][0];
        float disc = max(trace * trace - 4.0 * det, 0.0);
        float sqrtDisc = sqrt(disc);
        float lambda1 = max(0.5 * (trace + sqrtDisc), 1e-6);
        float lambda2 = max(0.5 * (trace - sqrtDisc), 1e-6);

        vec2 majorAxis;
        if (abs(Sigma2D[0][1]) > 1e-6) {
            majorAxis = normalize(vec2(lambda1 - Sigma2D[1][1], Sigma2D[0][1]));
        } else {
            // Near-diagonal covariance: pick axis with larger variance
            majorAxis = (Sigma2D[0][0] >= Sigma2D[1][1]) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
        }
        vec2 minorAxis = vec2(-majorAxis.y, majorAxis.x);

        float extent1 = uTruncate * sqrt(lambda1);
        float extent2 = uTruncate * sqrt(lambda2);

        float maxExtentPx = max(uResolution.x, uResolution.y) * uMaxExtentFactor;
        float largestExtent = max(extent1, extent2);
        if (largestExtent > maxExtentPx) {
            float clampScale = maxExtentPx / largestExtent;
            extent1 *= clampScale;
            extent2 *= clampScale;
        }

        if (uIsOrtho == 1) {
            vCenterScreen = vec2(
                uFx * centerCam.x + uResolution.x * 0.5,
                uFy * centerCam.y + uResolution.y * 0.5
            );
        } else {
            vCenterScreen = vec2(
                uFx * centerCam.x * invZ + uResolution.x * 0.5,
                uFy * centerCam.y * invZ + uResolution.y * 0.5
            );
        }

        vec2 quadOffset = aQuadCorner.x * majorAxis * extent1
                        + aQuadCorner.y * minorAxis * extent2;
        vec2 screenPos = vCenterScreen + quadOffset;
        vec2 ndcXY = (screenPos / uResolution) * 2.0 - 1.0;

        vec4 centerClip = projectionMatrix * centerCam4;
        float ndcZ = centerClip.z / centerClip.w;

        gl_Position = vec4(ndcXY, ndcZ, 1.0);

        vNodeId = uNodeId;
        vElementId = float(gl_InstanceID);
    }
`;

/**
 * Picking fragment shader for gsplats.
 * Outputs vec4(nodeId, elementId, brightness, 1.0) with brightness-as-depth.
 * Uses tighter truncation (1.5σ squared = 2.25) for precise picking.
 */
export const GSPLAT_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    flat in mediump float vAmplitude2D;
    flat in highp vec3 vL2D;
    flat in highp vec2 vCenterScreen;
    flat in highp float vNodeId;
    flat in highp float vElementId;

    uniform highp float uShiftC;
    uniform highp float uInvOneMinusC;
    uniform highp float uTruncateSq;

    out vec4 fragColor;

    void main() {
        vec2 d = gl_FragCoord.xy - vCenterScreen;

        float y0 = d.x * vL2D.x;
        float y1 = (d.y - vL2D.y * y0) * vL2D.z;
        float mahalSq = y0 * y0 + y1 * y1;

        if (mahalSq > uTruncateSq) discard;

        float intensity = vAmplitude2D * uInvOneMinusC * max(exp(-0.5 * mahalSq) - uShiftC, 0.0);
        if (intensity < 1e-4) discard;

        float brightness = clamp(intensity, 0.0, 1.0);

        fragColor = vec4(vNodeId, vElementId, brightness, 1.0);
        gl_FragDepth = 1.0 - brightness;
    }
`;

export const GSPLAT_PICK_SOURCE: ShaderSource = {
  name: 'gsplat-pick',
  webgl: { vertex: GSPLAT_PICK_VERTEX_SHADER, fragment: GSPLAT_PICK_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    return gsplatPickWebGPUFactory(buildGSplatPickTSLNodesFromUniforms(u));
  },
};
