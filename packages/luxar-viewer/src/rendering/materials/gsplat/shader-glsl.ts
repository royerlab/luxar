/**
 * Shared vertex shader for Gaussian splat rendering.
 *
 * Used by both GSplatMaterial (main rendering) and GSplatPickingMaterial (GPU picking).
 * Contains 3D-to-2D covariance projection, perspective Jacobian, amplitude calculation,
 * oriented quad expansion, near-plane fade, and screen-coverage safety.
 */
import {
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_NEAR_FADE_FUNCTIONS,
  GLSL_SORTED_INDEX,
} from '../_shared/glsl-lib';
import type { ShaderSource } from '../_shared/shader-source';
import {
  ALPHA_CLAMP,
  VOLUMETRIC_SERIES_C1,
  VOLUMETRIC_SERIES_C2_DIVISOR,
  VOLUMETRIC_SERIES_TAU_THRESHOLD,
  VOLUMETRIC_TAU_EPS,
} from '../_shared/volumetric';
import { requireTslMaterials } from '../../tsl/slot';

export const GSPLAT_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}

    // Quad corner attribute (static geometry)
    in vec2 aQuadCorner;  // (-1,-1), (1,-1), (-1,1), (1,1)

    // Draw-slot → storage-slot mapping, double-buffered so a new ordering
    // swaps atomically (declaration + luxarSortedIndex() in glsl-lib).
    // Uint32Array attributes → bound via vertexAttribIPointer, matching
    // the uint declarations.
    ${GLSL_SORTED_INDEX}

    // Splat data texture: RGBA32F, 4 texels/splat (see
    // rendering/element-texture-layout.ts for the texel layout).
    uniform highp sampler2D uSplatTex;

    // Uniforms (modelViewMatrix and projectionMatrix are built-in THREE.js uniforms)
    uniform vec2 uResolution;
    uniform float uFx, uFy;           // Focal lengths in pixels
    uniform float uTruncate;          // Truncation radius (in sigmas)
    uniform float uRayIntegralFactor; // Shifted Gaussian ray integral factor
    uniform int uProjectionMode;      // 0 = sum (ray-integral: additive/luminous/volumetric), 1 = peak (2D-projected surfaces: max/normal/opaque)
    uniform int uIsOrtho;             // 0 = perspective, 1 = orthographic
    uniform float uNearCull;          // Near cull distance (scene-scale-aware)
    uniform float uMaxExtentFactor;   // Max projected extent as fraction of viewport before fade
    uniform float uCov2DDilation;     // 2D-covariance low-pass dilation in CSS px² (3DGS anti-aliasing)
    uniform float uPixelRatio;
    uniform int uLabelColorMode;
    uniform int uLabelFilterIndex;

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
    flat out mediump float vAlpha;           // per-splat opacity (texel3.y; 1.0 for RGB data)
    // These need highp for screen-space calculations
    // OPTIMIZATION: vL2D stores [1/L00, L10, 1/L11] to replace fragment divisions with multiplications
    flat out highp vec3 vL2D;                // 2D Cholesky packed as [invL00, L10, invL11]
    flat out highp vec2 vCenterScreen;       // Splat center in screen pixels

    // Unpack 3D Cholesky to matrix (column-major order for GLSL mat3)
    // Packed order: [L00, L10, L11, L20, L21, L22]
    // c01 = [L00, L10], c23 = [L11, L20], c45 = [L21, L22]
    mat3 unpackCholesky3D(vec2 c01, vec2 c23, vec2 c45) {
        return mat3(
            c01.x, c01.y, c23.y,  // Column 0: [L00, L10, L20]
            0.0,   c23.x, c45.x,  // Column 1: [0, L11, L21]
            0.0,   0.0,   c45.y   // Column 2: [0, 0, L22]
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

    vec3 categoricalColor(float index) {
        return 0.25 + 0.75 * fract(index * vec3(0.61803398875, 0.38196601125, 0.75487766625));
    }

    void main() {
        // === Splat-texture fetch prologue ===
        // Four texelFetch reads reconstruct the per-splat values into
        // the exact local names the math below has always used — zero
        // changes downstream of this block. The width is a multiple of
        // 4 (element-texture-layout.ts), so a splat's 4 texels share one
        // row and only x advances.
        int splatBase = int(luxarSortedIndex()) * 4;
        int splatTexW = LUXAR_SPLAT_TEX_W;
        ivec2 texel0 = ivec2(splatBase % splatTexW, splatBase / splatTexW);
        vec4 splatT0 = texelFetch(uSplatTex, texel0, 0);
        vec4 splatT1 = texelFetch(uSplatTex, ivec2(texel0.x + 1, texel0.y), 0);
        vec4 splatT2 = texelFetch(uSplatTex, ivec2(texel0.x + 2, texel0.y), 0);
        vec4 splatT3 = texelFetch(uSplatTex, ivec2(texel0.x + 3, texel0.y), 0);
        vec3 aCenter = splatT0.xyz;        // 3D center (after nD slicing)
        float aAmplitude = splatT0.w;      // Already attenuated by hidden dims
        vec2 aCholesky01 = splatT1.xy;     // [L00, L10]
        vec2 aCholesky23 = splatT1.zw;     // [L11, L20]
        vec2 aCholesky45 = splatT2.xy;     // [L21, L22]
        vec3 aColor = vec3(splatT2.zw, splatT3.x);
        float aAlpha = splatT3.y;          // per-splat opacity (1.0 when the dataset is RGB)
        float aLabelIndex = splatT3.z;
        if (uLabelFilterIndex > 0 && int(aLabelIndex + 0.5) != uLabelFilterIndex) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }

        // Transform center to camera space
        vec4 centerCam4 = modelViewMatrix * vec4(aCenter, 1.0);
        vec3 centerCam = centerCam4.xyz;

        // === Unified near handling (shared perspectiveNearFade helper;
        // point + line shaders use the same) ===
        // Perspective: behind-camera splats fade to 0 (subsumes the old
        // standalone centerCam.z >= 0 reject) and the near-plane
        // approach fades across [uNearCull, 2*uNearCull] — prevents the
        // 1/z Jacobian singularity. Ortho: fade = 1; a behind-camera
        // splat falls through to NDC clipping, which drops it (ortho
        // near > 0 in this viewer), and no 1/z is consumed on the
        // ortho path.
        // uNearCull is scene-bounds-scaled (diagonal * 0.001); the
        // 1e-20 floor only guards uNearCull == 0 (degenerate
        // smoothstep). An absolute 1e-4 floor overrode the
        // scene-relative value on tiny-unit scenes and faded out the
        // whole scene. (Consequence: surviving zDepth is only bounded
        // by ~uNearCull, so the unguarded 1/zDepth below can get large
        // on a sub-camera-plane splat — J/Sigma2D then go non-finite
        // and the invalidCov2D reject drops the splat safely.)
        float depthFade = perspectiveNearFade(uIsOrtho, centerCam.z, max(uNearCull, 1e-20));
        if (depthFade < 0.01) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }

        // Transform Cholesky to camera space (rotation only)
        mat3 R = mat3(modelViewMatrix);
        mat3 L3D = unpackCholesky3D(aCholesky01, aCholesky23, aCholesky45);
        mat3 L_cam = R * L3D;
        mat3 Sigma_cam = L_cam * transpose(L_cam);

        // Positive depth (camera looks down -Z); reused by fades, Jacobian, and projection
        float zDepth = -centerCam.z;

        // === Screen-coverage safety guard (independent of depth fade) ===
        // Prevents GPU overload from splats whose projected quad is too large.
        // Fade starts at 50% of the limit and reaches ~0% AT the limit, so the
        // amplitude is negligible before the extent clamp (below) kicks in.
        // This avoids visible hard edges from clamped quads. Applies in BOTH
        // projections (ortho projected size is depth-independent, divisor 1)
        // and is computed UNCONDITIONALLY: below maxExtent*0.5 the smoothstep
        // is 0 and the fade is a no-op, so no size gate is needed. (The former
        // absolute maxLateralVar > 0.01 gate — a perf leftover from the
        // pre-#51 two-stage near cull — skipped the fade for splats with
        // spatial sigma < 0.1 world units while the extent clamp still
        // applied, leaving hard-edged clamped rectangles on deep-zoomed
        // tiny-sigma / nm-unit-scale scenes.) The 1e-20 floors match
        // the TSL twin's expressions exactly and are pure
        // div-by-zero/sqrt guards, NOT scale floors: maxLateralVar is a
        // WORLD-unit² variance, so the old absolute 1e-8 floor inflated
        // valid tiny-unit variances (sigma ~ 1e-7 => var ~ 1e-14) up to
        // sqrt(1e-8) = 1e-4 world units — projectedExtent exploded and
        // coverageFade culled EVERY splat in the scene. zDepth is
        // likewise bounded below by the scene-relative near fade.
        float coverageFade;
        {
            float maxLateralVar = max(Sigma_cam[0][0], max(Sigma_cam[1][1], Sigma_cam[2][2]));
            float extentDivisor = (uIsOrtho == 1) ? 1.0 : max(zDepth, 1e-20);
            float projectedExtent = uFx * sqrt(max(maxLateralVar, 1e-20)) * uTruncate / extentDivisor;
            float maxExtent = max(uResolution.x, uResolution.y) * uMaxExtentFactor;
            coverageFade = 1.0 - smoothstep(maxExtent * 0.5, maxExtent, projectedExtent);
            if (coverageFade < 0.01) {
                gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
                return;
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

        // 2D low-pass dilation (standard 3DGS anti-aliasing): widen the diagonal
        // so every splat covers at least ~1px. Guarantees near-degenerate
        // (edge-on/flat) splats render as a soft ellipse instead of a razor-thin
        // sub-pixel spike. Applied before the Cholesky + eigen extent below so
        // the fragment footprint and the quad stay consistent. Diagonal only —
        // adding to the off-diagonal would rotate/shear the ellipse.
        //
        // ENERGY COMPENSATION (Mip-Splatting): widening the footprint without
        // touching the peak CREATES light — a 2D Gaussian's screen-integrated
        // brightness is 2*pi*peak*sqrt(det Sigma2D), so dilation inflates it by
        // sqrt(detDilated/detRaw), i.e. (sigma_px^2 + d)/sigma_px^2 for an
        // isotropic splat; the compensation multiplier below is the reciprocal,
        // sqrt(detRaw/detDilated). The inflation diverges as the splat shrinks on screen
        // (measured 1.43x at 0.84 px, 3.75x at 0.33 px), so a scene silently
        // brightened as the camera pulled back, and a lifted points->gsplat LOD
        // ladder could not match its Points level in ANY mode. Points already
        // compensate their own sub-pixel widening (the sizeScale^2 term in
        // materials/point/shader-glsl.ts); gsplats now do too.
        float detRaw2D = Sigma2D[0][0] * Sigma2D[1][1] - Sigma2D[0][1] * Sigma2D[1][0];
        float cov2DDilation = uCov2DDilation * uPixelRatio * uPixelRatio;
        Sigma2D[0][0] += cov2DDilation;
        Sigma2D[1][1] += cov2DDilation;
        float detDilated2D = Sigma2D[0][0] * Sigma2D[1][1] - Sigma2D[0][1] * Sigma2D[1][0];
        float dilationCompensation = sqrt(max(detRaw2D, 0.0) / max(detDilated2D, 1e-12));

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
            // SCALE-FREE inversion: normalize Σ_cam by its mean diagonal
            // variance s = trace/3 first. det(Σ) is world-units⁶ — on a
            // tiny-unit scene (sigma ~ 1e-7 => det ~ 1e-42) it
            // underflows float32 (GPUs flush denormals to zero) and the
            // absolute 1e-12 clamp turned Σ⁻¹ into garbage (sum-mode
            // brightness off by many orders of magnitude); huge-unit
            // scenes overflow the same way. With Σn = Σ/s the
            // determinant and ray quadratic are O(1) at ANY scene
            // scale, so the 1e-12 / 1e-8 floors below act as pure
            // scale-free CONDITION-NUMBER guards. Σ⁻¹ = Σn⁻¹ / s, so
            // sigmaRay = sqrt(s / quadN) restores the world-unit
            // result exactly. The 1e-30 floor on s only guards an
            // all-zero (degenerate) covariance.
            float sTrace = max((Sigma_cam[0][0] + Sigma_cam[1][1] + Sigma_cam[2][2]) * (1.0 / 3.0), 1e-30);
            float invS = 1.0 / sTrace;
            float a = Sigma_cam[0][0] * invS;
            float b = Sigma_cam[0][1] * invS;
            float c = Sigma_cam[0][2] * invS;
            float d = Sigma_cam[1][1] * invS;
            float e = Sigma_cam[1][2] * invS;
            float f = Sigma_cam[2][2] * invS;
            // det(Σn) for 3x3 symmetric — clamped against numerical singularity.
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
            // quad is rᵀ Σn⁻¹ r (normalized space, O(1) for a
            // well-conditioned splat at any scale); un-normalize via
            // sqrt(sTrace): sigmaRay = 1/sqrt(rᵀ Σ⁻¹ r) = sqrt(s/quadN).
            float quad = max(rayDir.x * prx + rayDir.y * pry + rayDir.z * prz, 1e-8);
            sigmaRay = inversesqrt(quad) * sqrt(sTrace);
            // Shifted Gaussian ray integral: sqrt(2π)·erf(T/√2) - 2·T·exp(-0.5·T²)
            // Precomputed in TypeScript as uRayIntegralFactor (≈2.433 for T=3)
            float rayIntegrationBoost = sigmaRay * uRayIntegralFactor;  // voxelSpacing = 1.0
            // dilationCompensation keeps the screen-integrated light invariant
            // under the 2D low-pass above (see its derivation there). Sum
            // projection only: this branch's quantity IS that integral, so
            // conserving it is exactly right. The peak branch below reports a
            // peak, not an integral, and is left alone.
            vAmplitude2D = aAmplitude * rayIntegrationBoost * nearFade * dilationCompensation;
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
        if (uLabelColorMode == 1 && aLabelIndex > 0.0) {
            vColor = categoricalColor(aLabelIndex);
        } else {
            #ifdef USE_COLORMAP
            float t = clamp((aAmplitude - uScalarMin) * uScalarScale, 0.0, 1.0);
            #ifndef LUXAR_GAMMA_ONE
            t = pow(t, uInvGamma);          // gamma on the value, pre-LUT
            #endif
            vColor = texture(uColormapTex, vec2(t, 0.5)).rgb;
            #else
            vColor = aColor;
            #endif
        }
        // Per-splat opacity rides regardless of color source (in colormap
        // mode an RGBA dataset keeps its alpha; RGB data carries 1.0).
        // Sanitized: alpha is load-bearing in EVERY mode (linear
        // contribution scale) and maps into optical depth under
        // volumetric, where a NaN/Inf poisons τ past the discard into
        // NaN pixels — and a huge finite value would blow out the
        // linear folds (or overflow the mediump varying). Python
        // validation pins alpha to [0, 1] at write; this guards
        // hand-crafted zarr. NaN/Inf → the 1.0 opaque identity (loud);
        // finite values clamp to [0, 1] (a negative epsilon vanishes
        // continuously instead of flipping opaque). The point twin
        // does the same.
        vAlpha = sanitizeAlpha(aAlpha);

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
    flat in mediump float vAlpha;
    // These need highp for screen-space calculations
    // OPTIMIZATION: vL2D stores [1/L00, L10, 1/L11] for MUL instead of DIV
    flat in highp vec3 vL2D;          // 2D Cholesky packed as [invL00, L10, invL11]
    flat in highp vec2 vCenterScreen;

    uniform mediump float uOpacity;
    // Absorption coefficient κ — only read under LUXAR_VOLUMETRIC
    // (τ = κ·opacity·intensity); highp: τ enters an exp().
    uniform highp float uAbsorption;
    // 1.0 when the dataset's colors carry a per-splat alpha (RGBA), else
    // 0.0. Only the volumetric branch needs the gate: it maps alpha into
    // optical depth (w = −ln(1−a)), and the RGB default alpha of 1.0
    // would otherwise map to w ≈ 6.24 instead of the identity.
    uniform lowp float uHasElementAlpha;
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

        // Per-splat opacity (color alpha channel; 1.0 for RGB datasets).
        // Every mode scales its contribution linearly by a; volumetric
        // instead maps a into optical DENSITY, w = −ln(1 − a), so a
        // splat's peak rendered alpha reproduces a exactly (3DGS-faithful;
        // clamp = ALPHA_CLAMP from ../_shared/volumetric, mirrors Python's 1 − 1/512).
        // Dilute limit: w ≈ a, so the modes agree as a → 0; at large a
        // volumetric is intentionally denser (optical-depth semantics —
        // see spec §5.4).
        #ifdef LUXAR_VOLUMETRIC
        intensity *= mix(1.0, -log(1.0 - min(vAlpha, ${ALPHA_CLAMP})), uHasElementAlpha);
        #else
        intensity *= vAlpha;
        #endif

        // Early discard for negligible contribution (raised threshold for
        // performance). Alpha is already folded in, so a ~zero-alpha splat
        // discards here in every mode (it neither emits nor absorbs).
        // GAIN-AWARE: the emitted brightness is intensity * uIntensity *
        // color, so the visibility test must include the gain — a flat
        // 1e-4 gate discarded dim splats that a high gain (dim
        // fluorescence channels) would have lifted well above the ~1/255
        // floor (hard clipped rims + vanishing splats at gain >~ 40).
        // max(uIntensity, 1.0) keeps gain <= 1 EXACTLY at the historical
        // threshold (no overdraw change for default renders).
        if (intensity * max(uIntensity, 1.0) < 1e-4) discard;

        // Per-node GOG (Gain-Offset-Gamma) color adjustment. uIntensity (gain)
        // and uOffset apply in BOTH modes so the layer intensity/offset controls
        // work for a colormapped gsplat too. Colormap (LUT) mode: gamma + the
        // display-range window already shaped the scalar VALUE (amplitude) before
        // the LUT lookup, so only gain/offset apply post-LUT (no extra gamma).
        // Direct-color mode: full GOG on the raw color.
        // When the wrapper knows intensity==1 && offset==0 (the default), the
        // mul/add/clamp chain is identity for the common non-negative vColor
        // range; the wrapper stamps LUXAR_NO_GOG to skip it (mirrors the line
        // shader). The gain-aware discard above KEEPS reading uIntensity —
        // under NO_GOG uIntensity == 1 so max(uIntensity, 1.0) == 1.0 anyway.
        #ifdef LUXAR_NO_GOG
        vec3 adjusted = vColor;
        #else
        vec3 adjusted = max(vColor * uIntensity + uOffset, vec3(0.0));
        #endif

        #ifdef LUXAR_VOLUMETRIC
        // Volumetric optical depth: tau = kappa*opacity*intensity, where
        // intensity is the PRE-GOG density scalar (sum-projected ray
        // mass incl. rayIntegrationBoost + nearFade — a near-fading splat
        // loses emission and absorption together) and opacity scales
        // density (VOLUMETRIC_BLENDING_SPEC.md §3.1). GOG gain/offset/
        // gamma shape emission COLOR only, never tau.
        float tau = uAbsorption * uOpacity * intensity;
        // τ is color-independent — a black splat still absorbs (a
        // pure-ink occluder via gain→0 must keep its optical depth), so
        // the zero-color discard only fires when τ is negligible too.
        // (The earlier intensity<1e-4 discard bounds any lost τ at
        // κ·opacity·1e-4 per fragment — invisible at slider range.)
        if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4 && tau < 1e-4) discard;
        #else
        // Early discard for zero-contribution fragments after offset
        if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;
        #endif

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
        #elif defined(LUXAR_VOLUMETRIC)
        // 'volumetric' mode: emission–absorption (Max 1995). RGB carries
        // the self-screened emission (the splat's front absorbs its own
        // back: S(τ) = (1−e^(−τ))/τ, the exact closed form for emission ∝
        // density — what makes split-splat compositing exact); alpha is
        // the physical absorption 1 − e^(−τ) for the One /
        // OneMinusSrcAlpha state. κ = 0 ⇒ α = 0, S = 1 — bit-identical
        // framebuffer RGB arithmetic to additive (dst-alpha differs;
        // invisible on the alpha:false canvas). Series for τ < 1e-3 keeps
        // S well-conditioned through τ → 0 (rel. err < 1e-10 at cutoff).
        float alpha = 1.0 - exp(-tau);
        // Divisor guarded: GPU ternaries/selects evaluate both lanes, and
        // the TSL twin's .select does too — max() keeps the unselected
        // lane NaN-free at tau = 0 (identical in the selected regime).
        float screen = (tau < ${VOLUMETRIC_SERIES_TAU_THRESHOLD}) ? 1.0 - ${VOLUMETRIC_SERIES_C1} * tau + tau * tau / ${VOLUMETRIC_SERIES_C2_DIVISOR}.0
                                    : alpha / max(tau, ${VOLUMETRIC_TAU_EPS});
        fragColor = vec4(finalColor * screen, alpha);
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
  webgpu: (uniforms: Record<string, unknown>) => {
    const { gsplatWebGPUFactory, buildGSplatTSLNodesFromUniforms } =
      requireTslMaterials().factories.gsplat;
    return gsplatWebGPUFactory(
      buildGSplatTSLNodesFromUniforms(uniforms as Record<string, import('three').IUniform>)
    );
  },
};
