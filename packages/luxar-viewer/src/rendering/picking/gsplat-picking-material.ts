/**
 * GSplat Picking Material for GPU object picking.
 *
 * Renders Gaussian splats to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (gl_InstanceID), B = brightness, A = 1.0
 *
 * Vertex shader is based on GSPLAT_VERTEX_SHADER (rendering/shaders/gsplat-shaders.ts)
 * with additions for nodeId/elementId output. Keep in sync with that shader.
 *
 * Uses tighter truncation (1.5 sigma vs 3.0) and brightness-as-depth.
 */

import * as THREE from 'three';
import type { CameraAwareMaterial } from '../camera-aware-material';
import { materialManager } from '../material-manager';

export interface GSplatPickingMaterialConfig {
  nodeId: number;
}

/**
 * Picking vertex shader for gsplats.
 * Based on GSPLAT_VERTEX_SHADER — adds uNodeId/vNodeId/vElementId, strips colormap.
 * Uses tighter truncation (1.5σ) for more precise picking.
 */
const GSPLAT_PICK_VERTEX_SHADER = /* glsl */ `
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
    uniform float uRayIntegralFactor;
    uniform int uProjectionMode;
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
const GSPLAT_PICK_FRAGMENT_SHADER = /* glsl */ `
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

export class GSplatPickingMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  constructor(config: GSplatPickingMaterialConfig) {
    // Tighter truncation: 1.5σ instead of 3.0σ
    const truncate = 1.5;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    // Compute ray integral factor for 1.5σ truncation
    const SQRT_2PI = Math.sqrt(2 * Math.PI);
    const x = truncate / Math.SQRT2;
    const t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
    const erfVal =
      1.0 -
      t *
        (0.254829592 +
          t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) *
        Math.exp(-x * x);
    const erf = x >= 0 ? erfVal : -erfVal;
    const rayIntegralFactor = SQRT_2PI * erf - 2 * truncate * Math.exp(-0.5 * truncate * truncate);

    super({
      uniforms: {
        uResolution: { value: new THREE.Vector2(1, 1) },
        uFx: { value: 500 },
        uFy: { value: 500 },
        uTruncate: { value: truncate },
        uTruncateSq: { value: truncate * truncate },
        uShiftC: { value: shiftC },
        uInvOneMinusC: { value: invOneMinusC },
        uRayIntegralFactor: { value: rayIntegralFactor },
        uProjectionMode: { value: 1 }, // Max projection for picking
        uIsOrtho: { value: 0 },
        uNearCull: { value: 0.1 },
        uMaxExtentFactor: { value: 0.33 },
        uNodeId: { value: config.nodeId },
      },
      vertexShader: GSPLAT_PICK_VERTEX_SHADER,
      fragmentShader: GSPLAT_PICK_FRAGMENT_SHADER,
      glslVersion: THREE.GLSL3,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      blending: THREE.NoBlending,
      toneMapped: false,
      side: THREE.DoubleSide,
    });
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.uniforms.uResolution.value.copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;

    if (isOrtho) {
      const fy = resolution.y / fov;
      this.uniforms.uFx.value = fy;
      this.uniforms.uFy.value = fy;
    } else {
      const tanHalfFov = Math.tan(fov / 2);
      const fy = resolution.y / (2 * tanHalfFov);
      this.uniforms.uFx.value = fy;
      this.uniforms.uFy.value = fy;
    }

    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
  }

  dispose(): void {
    materialManager.unregister(this);
    super.dispose();
  }
}
