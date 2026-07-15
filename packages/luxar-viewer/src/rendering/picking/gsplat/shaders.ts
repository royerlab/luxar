/**
 * GLSL3 picking shader source for Gaussian splats + `ShaderSource` record.
 *
 * Mirrors the visual gsplat shader with picking-specific adjustments:
 *   - `uNodeId` uniform + `vNodeId` / `vElementId` varyings, written
 *     into the RGBA32F pick buffer as `(nodeId, elementId, brightness, 1)`.
 *   - Tighter truncation: 1.5σ (vs 3σ for visual) so the pick footprint
 *     is the bright core only.
 *   - Always max-projection (no ray-integral) — picking only needs the
 *     brightest fragment, not the integrated path.
 *   - Brightness-as-depth so the brightest overlapping fragment wins.
 *
 * Source-of-truth for GLSL3; the WebGPU counterpart lives in `./pick.tsl`
 * and is referenced through the `ShaderSource.webgpu` factory below.
 *
 * @module rendering/picking/gsplat/shaders
 */

import type { ShaderSource } from '../../materials/_shared/shader-source';
import {
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_NEAR_FADE_FUNCTIONS,
} from '../../materials/_shared/glsl-lib';
import { gsplatPickWebGPUFactory, buildGSplatPickTSLNodesFromUniforms } from './pick.tsl';

/**
 * Picking vertex shader for gsplats.
 * Adds uNodeId/vNodeId/vElementId, strips colormap.
 * Uses tighter truncation (1.5σ) for more precise picking.
 */
export const GSPLAT_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}

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

    // Parity with visual shader-glsl.ts invalidCov2D — reject Σ_2D
    // entries that are NaN/Inf so a degenerate splat can't poison the
    // eigendecomposition and produce undefined pick geometry.
    bool invalidCov2D(mat2 S) {
        return isInvalidFloat(S[0][0]) || isInvalidFloat(S[0][1])
            || isInvalidFloat(S[1][0]) || isInvalidFloat(S[1][1]);
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

        // Unified near handling — see the visual gsplat shader: the
        // shared perspectiveNearFade subsumes the old standalone
        // behind-camera reject; ortho falls through to NDC clipping.
        float depthFade = perspectiveNearFade(uIsOrtho, centerCam.z, max(uNearCull, 1e-4));
        if (depthFade < 0.01) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }

        mat3 R = mat3(modelViewMatrix);
        mat3 L3D = unpackCholesky3D();
        mat3 L_cam = R * L3D;
        mat3 Sigma_cam = L_cam * transpose(L_cam);

        float zDepth = -centerCam.z;

        // Coverage fade applies in BOTH projections (matches the visual
        // shader — ortho projected size is depth-independent, divisor 1)
        // so pickability tracks what is actually visible. Computed
        // UNCONDITIONALLY (visual twin updated in lockstep): below
        // maxExtent*0.5 the smoothstep is 0 and the fade is a no-op, so
        // no size gate is needed — the former maxLateralVar > 0.01 gate
        // skipped the fade for sigma < 0.1 world-unit splats.
        float coverageFade;
        {
            float maxLateralVar = max(Sigma_cam[0][0], max(Sigma_cam[1][1], Sigma_cam[2][2]));
            float extentDivisor = (uIsOrtho == 1) ? 1.0 : max(zDepth, 1e-8);
            float projectedExtent = uFx * sqrt(max(maxLateralVar, 1e-8)) * uTruncate / extentDivisor;
            float maxExtent = max(uResolution.x, uResolution.y) * uMaxExtentFactor;
            coverageFade = 1.0 - smoothstep(maxExtent * 0.5, maxExtent, projectedExtent);
            if (coverageFade < 0.01) {
                gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
                return;
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

        // Visual-shader parity (shader-glsl.ts) + TSL-side parity
        // (gsplat-pick.tsl.ts): reject splats with NaN/Inf Σ_2D or
        // amplitude so picking and rendering agree on which elements
        // are pickable across WebGL and WebGPU backends.
        if (invalidCov2D(Sigma2D) || isInvalidFloat(aAmplitude)) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }

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
