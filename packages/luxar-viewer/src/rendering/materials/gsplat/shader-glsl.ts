/**
 * Shared vertex shader for Gaussian splat rendering.
 *
 * Used by both GSplatMaterial (main rendering) and GSplatPickingMaterial (GPU picking).
 * Contains 3D-to-2D covariance projection, perspective Jacobian, amplitude calculation,
 * oriented quad expansion, near-plane fade, and screen-coverage safety.
 */
import { GLSL_SANITIZE_FUNCTIONS } from '../_shared/glsl-lib';
import type { ShaderSource } from '../_shared/shader-source';
import { gsplatWebGPUFactory, buildGSplatTSLNodesFromUniforms } from './shader-tsl';

export const GSPLAT_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}

    // Quad corner attribute (static geometry)
    in vec2 aQuadCorner;  // (-1,-1), (1,-1), (-1,1), (1,1)

    // Per-instance attributes
    in vec3 aCenter;           // 3D center (after nD slicing)
    in vec2 aCholesky01;       // [L00, L10]
    in vec2 aCholesky23;       // [L11, L20]
    in vec2 aCholesky45;       // [L21, L22]
    in float aAmplitude;       // Already attenuated by hidden dims
    in vec3 aColor;

    // Uniforms (modelViewMatrix and projectionMatrix are built-in THREE.js uniforms)
    uniform vec2 uResolution;
    uniform float uFx, uFy;           // Focal lengths in pixels
    uniform float uTruncate;          // Truncation radius (in sigmas)
    uniform float uTruncateSq;        // Truncation radius squared
    uniform float uRayIntegralFactor; // Shifted Gaussian ray integral factor
    uniform int uProjectionMode;      // 0 = sum projection (additive), 1 = max projection (max blending)
    uniform int uIsOrtho;             // 0 = perspective, 1 = orthographic
    uniform float uNearCull;          // Near cull distance (scene-scale-aware)
    uniform float uMaxExtentFactor;   // Max projected extent as fraction of viewport before fade

    // Colormap uniforms (only active when USE_COLORMAP is defined)
    #ifdef USE_COLORMAP
    uniform sampler2D uColormapTex;   // 256x1 LUT texture
    uniform float uScalarMin;         // Scalar range minimum (display-range window)
    uniform float uScalarScale;       // 1.0 / (max - min)
    uniform mediump float uInvGamma;  // Gamma applied to the VALUE, pre-LUT
    #endif

    // Varyings to fragment - all per-instance varyings use "flat" (no interpolation needed)
    // OPTIMIZATION: flat qualifier skips GPU interpolation hardware for constant values
    flat out mediump vec3 vColor;
    flat out mediump float vAmplitude2D;
    // These need highp for screen-space calculations
    // OPTIMIZATION: vL2D stores [1/L00, L10, 1/L11] to replace fragment divisions with multiplications
    flat out highp vec3 vL2D;                // 2D Cholesky packed as [invL00, L10, invL11]
    flat out highp vec2 vCenterScreen;       // Splat center in screen pixels
    flat out int vProjectionMode;            // 0=sum (additive), 1=max

    // Unpack 3D Cholesky to matrix (column-major order for GLSL mat3)
    // Packed order: [L00, L10, L11, L20, L21, L22]
    // aCholesky01 = [L00, L10], aCholesky23 = [L11, L20], aCholesky45 = [L21, L22]
    mat3 unpackCholesky3D() {
        return mat3(
            aCholesky01.x, aCholesky01.y, aCholesky23.y,  // Column 0: [L00, L10, L20]
            0.0,           aCholesky23.x, aCholesky45.x,  // Column 1: [0, L11, L21]
            0.0,           0.0,           aCholesky45.y   // Column 2: [0, 0, L22]
        );
    }

    // invalidFloat is an alias for the shared isInvalidFloat helper in glsl-lib.
    bool invalidFloat(float v) {
        return isInvalidFloat(v);
    }

    bool invalidCov2D(mat2 S) {
        return invalidFloat(S[0][0]) || invalidFloat(S[0][1]) || invalidFloat(S[1][0]) || invalidFloat(S[1][1]);
    }

    // Compute 2D Cholesky from 2D covariance (symmetric positive definite)
    // OPTIMIZATION: Returns [1/L00, L10, 1/L11] for faster fragment shader (MUL instead of DIV)
    vec3 cholesky2x2(mat2 S) {
        float s00 = max(S[0][0], 1e-6);
        float s10 = invalidFloat(S[1][0]) ? 0.0 : S[1][0];
        float s11 = invalidFloat(S[1][1]) ? 1e-6 : S[1][1];
        float L00 = sqrt(s00);
        float invL00 = 1.0 / L00;
        float L10 = s10 * invL00;  // Use reciprocal here too
        float L11 = sqrt(max(s11 - L10 * L10, 1e-6));
        float invL11 = 1.0 / L11;
        return vec3(invL00, L10, invL11);  // Pack reciprocals for fragment shader
    }

    void main() {
        // Transform center to camera space
        vec4 centerCam4 = modelViewMatrix * vec4(aCenter, 1.0);
        vec3 centerCam = centerCam4.xyz;

        // Reject splats behind the camera (camera looks down -Z axis)
        // centerCam.z >= 0 means the splat is on the +Z side (behind camera)
        if (centerCam.z >= 0.0) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);  // Behind camera
            return;
        }

        // Transform Cholesky to camera space (rotation only)
        mat3 R = mat3(modelViewMatrix);
        mat3 L3D = unpackCholesky3D();
        mat3 L_cam = R * L3D;
        mat3 Sigma_cam = L_cam * transpose(L_cam);

        // Positive depth (camera looks down -Z); reused by fades, Jacobian, and projection
        float zDepth = -centerCam.z;

        // === Near-plane depth fade (perspective only; ortho has no 1/z singularity) ===
        // Principled fade to prevent 1/z Jacobian singularity near camera.
        // Scene-scale-aware via uNearCull uniform.
        float depthFade = 1.0;
        if (uIsOrtho == 0) {
            depthFade = smoothstep(uNearCull, uNearCull * 2.0, zDepth);
            if (depthFade < 0.01) {
                gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
                return;
            }
        }

        // === Screen-coverage safety guard (independent of depth fade) ===
        // Prevents GPU overload from splats whose projected quad is too large.
        // Fade starts at 50% of the limit and reaches ~0% AT the limit, so the
        // amplitude is negligible before the extent clamp (below) kicks in.
        // This avoids visible hard edges from clamped quads. Applies in BOTH
        // projections: the extent CLAMP below runs unconditionally, so ortho
        // previously showed hard-edged clamped rectangles with no fade
        // (perspective-only was a leftover from when the clamp was too).
        // Ortho projected size is depth-independent (divisor 1).
        // CAVEAT (inherited, unchanged here): the maxLateralVar > 0.01
        // gate below is an ABSOLUTE world-units² threshold, so splats
        // with spatial sigma < 0.1 world units skip the fade entirely
        // while the extent clamp still applies — deep-zoomed tiny-sigma
        // (or nm-unit-scale) scenes can still show clamped rectangles.
        float coverageFade = 1.0;
        {
            float maxLateralVar = max(Sigma_cam[0][0], max(Sigma_cam[1][1], Sigma_cam[2][2]));
            if (maxLateralVar > 0.01) {
                float extentDivisor = (uIsOrtho == 1) ? 1.0 : zDepth;
                float projectedExtent = uFx * sqrt(maxLateralVar) * uTruncate / extentDivisor;
                float maxExtent = max(uResolution.x, uResolution.y) * uMaxExtentFactor;
                coverageFade = 1.0 - smoothstep(maxExtent * 0.5, maxExtent, projectedExtent);
                if (coverageFade < 0.01) {
                    gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
                    return;
                }
            }
        }

        // Combined fade: most restrictive wins (both use smoothstep → no popping)
        float nearFade = min(depthFade, coverageFade);

        // Precompute depth reciprocals (used by perspective Jacobian and screen projection)
        float invZ = 1.0 / zDepth;
        float invZ2 = invZ * invZ;

        // Projection Jacobian at splat center
        mat3x2 J;
        if (uIsOrtho == 1) {
            // Orthographic: no depth dependence (parallel projection)
            J[0] = vec2(uFx, 0.0);
            J[1] = vec2(0.0, uFy);
            J[2] = vec2(0.0, 0.0);
        } else {
            // Perspective projection Jacobian
            // For camera looking down -Z, with z = -centerCam.z (positive depth):
            // x_s = fx * centerCam.x / z, y_s = fy * centerCam.y / z
            // ∂x_s/∂(centerCam.z) = fx * centerCam.x / z² (since z = -centerCam.z)
            J[0] = vec2(uFx * invZ, 0.0);
            J[1] = vec2(0.0, uFy * invZ);
            J[2] = vec2(uFx * centerCam.x * invZ2, uFy * centerCam.y * invZ2);
        }

        // Project covariance to 2D: Σ_2D = J · Σ_cam · Jᵀ
        // Compute J * Sigma_cam first
        vec2 JS0 = J[0] * Sigma_cam[0][0] + J[1] * Sigma_cam[0][1] + J[2] * Sigma_cam[0][2];
        vec2 JS1 = J[0] * Sigma_cam[1][0] + J[1] * Sigma_cam[1][1] + J[2] * Sigma_cam[1][2];
        vec2 JS2 = J[0] * Sigma_cam[2][0] + J[1] * Sigma_cam[2][1] + J[2] * Sigma_cam[2][2];

        // Sigma2D = (J * Sigma_cam) * J^T
        // For M = J*Sigma (cols JS0, JS1, JS2), compute M * J^T:
        // (M*J^T)[i,j] = sum_k M[i,k] * J[j,k] = sum_k JS_k[i] * J[k][j]
        mat2 Sigma2D;
        Sigma2D[0][0] = JS0.x * J[0].x + JS1.x * J[1].x + JS2.x * J[2].x;
        Sigma2D[1][0] = JS0.x * J[0].y + JS1.x * J[1].y + JS2.x * J[2].y;
        Sigma2D[0][1] = Sigma2D[1][0];  // Symmetric (J*S*J^T preserves symmetry)
        Sigma2D[1][1] = JS0.y * J[0].y + JS1.y * J[1].y + JS2.y * J[2].y;

        if (invalidCov2D(Sigma2D) || invalidFloat(aAmplitude)) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }

        // Projection mode determines amplitude calculation:
        // - Sum projection (uProjectionMode = 0): Integrate Gaussian along ray → ray boost
        // - Max projection (uProjectionMode = 1): Use peak Gaussian value → no boost
        //
        // OPTIMIZATION: Use branch instead of branchless mix() to skip expensive operations
        // (normalize, sqrt, exp) when in max mode. Warps are typically coherent on this uniform.
        float sigmaRay = 1.0;  // Default for max mode (no ray integration)
        if (uProjectionMode == 0) {
            // Sum projection: compute ray-integral standard deviation.
            //
            // The line-integral of an anisotropic Gaussian along ray
            // direction r has 1D std-dev sigma_line = 1 / sqrt(rᵀ Σ⁻¹ r),
            // not sqrt(rᵀ Σ r). The two only agree when r is aligned with
            // a covariance eigenvector or Σ is isotropic.
            //
            // Implementation: compute Σ_cam⁻¹ via the closed-form 3×3
            // inverse and clamp to a minimum determinant. The shader is
            // already paying for a covariance matrix-vector product, so
            // a one-off explicit inverse is a small constant factor.
            vec3 rayDir = (uIsOrtho == 1) ? vec3(0.0, 0.0, -1.0) : normalize(centerCam);
            // Cofactor expansion for 3x3 inverse. Σ_cam is symmetric SPD,
            // so the inverse is symmetric SPD too.
            float a = Sigma_cam[0][0];
            float b = Sigma_cam[0][1];
            float c = Sigma_cam[0][2];
            float d = Sigma_cam[1][1];
            float e = Sigma_cam[1][2];
            float f = Sigma_cam[2][2];
            // det(Σ) for 3x3 symmetric — clamped against numerical singularity.
            float detSigma = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
            float invDet = 1.0 / max(detSigma, 1e-12);
            // Cofactors of the inverse (symmetric).
            float i00 = (d * f - e * e) * invDet;
            float i11 = (a * f - c * c) * invDet;
            float i22 = (a * d - b * b) * invDet;
            float i01 = -(b * f - c * e) * invDet;
            float i02 = (b * e - c * d) * invDet;
            float i12 = -(a * e - b * c) * invDet;
            // r' = Σ⁻¹ r ; precision quadratic = rᵀ Σ⁻¹ r
            float prx = i00 * rayDir.x + i01 * rayDir.y + i02 * rayDir.z;
            float pry = i01 * rayDir.x + i11 * rayDir.y + i12 * rayDir.z;
            float prz = i02 * rayDir.x + i12 * rayDir.y + i22 * rayDir.z;
            float quad = max(rayDir.x * prx + rayDir.y * pry + rayDir.z * prz, 1e-8);
            sigmaRay = inversesqrt(quad);
            // Shifted Gaussian ray integral: sqrt(2π)·erf(T/√2) - 2·T·exp(-0.5·T²)
            // Precomputed in TypeScript as uRayIntegralFactor (≈2.433 for T=3)
            float rayIntegrationBoost = sigmaRay * uRayIntegralFactor;  // voxelSpacing = 1.0
            vAmplitude2D = aAmplitude * rayIntegrationBoost * nearFade;
        } else {
            // Max projection: no boost needed
            vAmplitude2D = aAmplitude * nearFade;
        }

        // Compute 2D Cholesky for fragment shader
        vL2D = cholesky2x2(Sigma2D);

        // Eigenvalues of Σ_2D for quad extents (oriented quad)
        float trace = Sigma2D[0][0] + Sigma2D[1][1];
        float det = Sigma2D[0][0] * Sigma2D[1][1] - Sigma2D[0][1] * Sigma2D[1][0];
        float disc = max(trace * trace - 4.0 * det, 0.0);  // Clamp for numerical stability
        float sqrtDisc = sqrt(disc);
        float lambda1 = max(0.5 * (trace + sqrtDisc), 1e-6);
        float lambda2 = max(0.5 * (trace - sqrtDisc), 1e-6);

        vProjectionMode = uProjectionMode;

        // Eigenvector for major axis (for oriented quad)
        vec2 majorAxis;
        if (abs(Sigma2D[0][1]) > 1e-6) {
            majorAxis = normalize(vec2(lambda1 - Sigma2D[1][1], Sigma2D[0][1]));
        } else {
            // Near-diagonal covariance: pick axis with larger variance
            majorAxis = (Sigma2D[0][0] >= Sigma2D[1][1]) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
        }
        vec2 minorAxis = vec2(-majorAxis.y, majorAxis.x);

        // Quad extents: truncation radius × sqrt(eigenvalue)
        // Shifted Gaussian: effectiveTruncate = uTruncate
        float extent1 = uTruncate * sqrt(lambda1);
        float extent2 = uTruncate * sqrt(lambda2);

        // Clamp quad extents so no splat exceeds uMaxExtentFactor × viewport.
        // The amplitude fade (nearFade) handles the visual transition smoothly;
        // this clamp prevents the rasterizer from shading oversized quads.
        float maxExtentPx = max(uResolution.x, uResolution.y) * uMaxExtentFactor;
        float largestExtent = max(extent1, extent2);
        if (largestExtent > maxExtentPx) {
            float clampScale = maxExtentPx / largestExtent;
            extent1 *= clampScale;
            extent2 *= clampScale;
        }

        // Project center to screen (pixels)
        if (uIsOrtho == 1) {
            // Orthographic: direct linear mapping (no depth division)
            vCenterScreen = vec2(
                uFx * centerCam.x + uResolution.x * 0.5,
                uFy * centerCam.y + uResolution.y * 0.5
            );
        } else {
            // Perspective: reuse invZ from earlier computation
            vCenterScreen = vec2(
                uFx * centerCam.x * invZ + uResolution.x * 0.5,
                uFy * centerCam.y * invZ + uResolution.y * 0.5
            );
        }

        // Expand quad vertex in screen space (oriented)
        vec2 quadOffset = aQuadCorner.x * majorAxis * extent1
                        + aQuadCorner.y * minorAxis * extent2;
        vec2 screenPos = vCenterScreen + quadOffset;

        // Convert screen pixels to NDC (xy only)
        vec2 ndcXY = (screenPos / uResolution) * 2.0 - 1.0;

        // Pass through color — either from vertex attribute or colormap LUT.
        // Colormap mode: display range (uScalarMin/uScalarScale) and gamma
        // operate on the scalar VALUE (here the amplitude) before the LUT
        // lookup, not on the resulting color. See fragment-shader note.
        #ifdef USE_COLORMAP
        float t = clamp((aAmplitude - uScalarMin) * uScalarScale, 0.0, 1.0);
        #ifndef LUXAR_GAMMA_ONE
        t = pow(t, uInvGamma);          // gamma on the value, pre-LUT
        #endif
        vColor = texture(uColormapTex, vec2(t, 0.5)).rgb;
        #else
        vColor = aColor;
        #endif

        // Compute proper clip-space depth using projection matrix
        // This ensures correct depth buffer behavior for overlapping splats
        vec4 centerClip = projectionMatrix * centerCam4;
        float ndcZ = centerClip.z / centerClip.w;

        // Output final clip position
        gl_Position = vec4(ndcXY, ndcZ, 1.0);
    }
  `;

/**
 * Fragment shader for standard Gaussian splat rendering.
 *
 * Uses the 2D Cholesky factor passed from vertex shader to compute
 * Mahalanobis distance, then applies shifted Gaussian falloff.
 * The picking system uses a different fragment shader (see picking/gsplat-picking-material.ts).
 */
export const GSPLAT_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    // All varyings use flat - no interpolation needed (constant per instance)
    // OPTIMIZATION: flat qualifier skips GPU interpolation hardware
    flat in mediump vec3 vColor;
    flat in mediump float vAmplitude2D;
    // These need highp for screen-space calculations
    // OPTIMIZATION: vL2D stores [1/L00, L10, 1/L11] for MUL instead of DIV
    flat in highp vec3 vL2D;          // 2D Cholesky packed as [invL00, L10, invL11]
    flat in highp vec2 vCenterScreen;
    flat in int vProjectionMode;      // 0=sum (additive), 1=max

    uniform mediump float uOpacity;
    uniform mediump float uInvGamma; // Pre-computed 1/gamma for performance
    uniform mediump float uIntensity; // Per-node linear color multiplier (gain)
    uniform mediump float uOffset; // Per-node additive brightness shift (black level)
    uniform highp float uShiftC;       // Shifted Gaussian: exp(-0.5 * T²)
    uniform highp float uInvOneMinusC; // Shifted Gaussian: 1/(1-C)
    uniform highp float uTruncateSq;   // Truncation radius squared (T²)

    // GLSL ES 3.0 requires explicit fragment output declaration
    out vec4 fragColor;

    void main() {
        // Pixel offset from splat center
        vec2 d = gl_FragCoord.xy - vCenterScreen;

        // Forward substitution: solve L · y = d
        // OPTIMIZATION: vL2D contains [invL00, L10, invL11] - use MUL instead of DIV
        float y0 = d.x * vL2D.x;  // d.x * invL00
        float y1 = (d.y - vL2D.y * y0) * vL2D.z;  // (d.y - L10 * y0) * invL11

        // Squared Mahalanobis distance
        float mahalSq = y0 * y0 + y1 * y1;

        // EARLY DISCARD: Skip pixels beyond truncation radius
        if (mahalSq > uTruncateSq) discard;

        // Shifted Gaussian: a·scale·max(0, exp(-½·r²) - C)
        // Ensures C⁰ continuity at truncation boundary (no discontinuity)
        float intensity = vAmplitude2D * uInvOneMinusC * max(exp(-0.5 * mahalSq) - uShiftC, 0.0);

        // Early discard for negligible contribution (raised threshold for performance)
        if (intensity < 1e-4) discard;

        // Per-node GOG (Gain-Offset-Gamma) color adjustment.
        // Colormap (LUT) mode: gamma + display-range already shaped the
        // scalar VALUE (amplitude) before the LUT lookup, so the mapped
        // color passes through untouched. Direct-color mode: GOG on color.
        #ifdef USE_COLORMAP
        vec3 adjusted = max(vColor, vec3(0.0));
        #else
        vec3 adjusted = vColor * uIntensity + uOffset;
        adjusted = max(adjusted, vec3(0.0));
        #endif

        // Early discard for zero-contribution fragments after offset
        if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;

        // LUXAR_GAMMA_ONE (gamma == 1.0) skips the per-fragment pow() —
        // pow(x, 1) == x — same fast path the colormap branch already takes.
        #if defined(USE_COLORMAP) || defined(LUXAR_GAMMA_ONE)
        vec3 gammaColor = adjusted;
        #else
        vec3 gammaColor = pow(adjusted, vec3(uInvGamma));
        #endif

        // HDR color output for linear additive blending
        // With OneFactor blending (additive/luminous/max modes), alpha is ignored,
        // so apply opacity to RGB directly. This gives correct LINEAR sum projection
        // without the intensity-squaring bug that AdditiveBlending (SrcAlpha) would cause.
        vec3 finalColor = gammaColor * intensity * uOpacity;

        #ifdef LUXAR_NORMAL_PREMULT
        // 'normal' mode: premultiplied alpha-over. RGB already carries the
        // full (unclamped, HDR) contribution; alpha carries a CLAMPED
        // coverage term so the One / OneMinusSrcAlpha framebuffer state
        // (see blending-state.ts getGSplatNormalBlendingState) attenuates
        // the destination without ever over-subtracting. Dim splats
        // (intensity·opacity << 1) occlude proportionally little — an
        // emitter-with-occlusion model, deliberate for HDR scientific data.
        float coverage = clamp(intensity * uOpacity, 0.0, 1.0);
        fragColor = vec4(finalColor, coverage);
        #else
        // All other modes keep the alpha=1.0 contract: additive/luminous
        // rely on SrcAlpha being the identity factor (what makes the
        // shared AdditiveBlending state equal the linear One+One sum),
        // and max compares premultiplied RGB contributions directly.
        fragColor = vec4(finalColor, 1.0);
        #endif
    }
  `;

export const GSPLAT_SOURCE: ShaderSource = {
  name: 'gsplat',
  webgl: { vertex: GSPLAT_VERTEX_SHADER, fragment: GSPLAT_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) =>
    gsplatWebGPUFactory(
      buildGSplatTSLNodesFromUniforms(uniforms as Record<string, import('three').IUniform>)
    ),
};
