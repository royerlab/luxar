/**
 * GLSL gsplat projection — ONE source for the visual vertex stage
 * (`./shader-glsl.ts`) and the pick vertex stage (`picking/gsplat/shaders.ts`),
 * so the two cannot place, size or cull a splat differently: the camera-space
 * covariance, the screen-coverage fade, the projected 2D covariance (general
 * projection Jacobian), the sum projection's ray sigma, the Σ_2D eigen-axes and
 * the max-extent-clamped quad extents. TSL twins: the `gsplat*TSL` builders in
 * `./shader-tsl.ts`. CPU mirror of the projection terms:
 * `../_shared/projection-math.ts`.
 *
 * The 2D Cholesky of Σ_2D is NOT here on purpose: the visual one sanitizes and
 * floors at 1e-6 (a fragment-stage falloff must never see NaN), the pick one
 * keeps its historical 1e-8 floors; sharing it would move pick footprints.
 *
 * REQUIRED GLOBALS — include AFTER them (GLSL resolves names top-down):
 * `uResolution`, `uMaxExtentFactor`, `modelViewMatrix`, `projectionMatrix`,
 * `luxarProjectionSizeScale()` (GLSL_PROJECTION_FUNCTIONS) and
 * `isInvalidFloat` (GLSL_SANITIZE_FUNCTIONS).
 *
 * @module rendering/materials/gsplat/projection-glsl
 */

export const GLSL_GSPLAT_PROJECTION = /* glsl */ `
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

    // Camera-space covariance: rotate the Cholesky factor by the model-view
    // rotation, Σ_cam = (R·L)(R·L)ᵀ.
    mat3 gsplatSigmaCam(vec2 c01, vec2 c23, vec2 c45) {
        mat3 R = mat3(modelViewMatrix);
        mat3 L3D = unpackCholesky3D(c01, c23, c45);
        mat3 L_cam = R * L3D;
        return L_cam * transpose(L_cam);
    }

    // A Σ_2D entry that is NaN/Inf would poison the eigendecomposition; both
    // stages reject such a splat.
    bool invalidCov2D(mat2 S) {
        return isInvalidFloat(S[0][0]) || isInvalidFloat(S[0][1]) || isInvalidFloat(S[1][0]) || isInvalidFloat(S[1][1]);
    }

    // === Screen-coverage safety fade (independent of the depth fade) ===
    // Prevents GPU overload from splats whose projected quad is too large.
    // Fade starts at 50% of the limit and reaches ~0% AT the limit, so the
    // amplitude is negligible before the extent clamp kicks in. Applies in
    // BOTH projections (ortho projected size is depth-independent, divisor 1)
    // and is computed UNCONDITIONALLY: below maxExtent*0.5 the smoothstep is 0
    // and the fade is a no-op, so no size gate is needed. (A former absolute
    // maxLateralVar > 0.01 gate skipped the fade for spatial sigma < 0.1 world
    // units while the extent clamp still applied, leaving hard-edged clamped
    // rectangles on deep-zoomed tiny-unit scenes.) The 1e-20 floors are pure
    // div-by-zero/sqrt guards, NOT scale floors: maxLateralVar is a WORLD-unit²
    // variance, and an absolute 1e-8 floor once coverage-culled EVERY splat of
    // a tiny-unit scene. zDepth is bounded below by the scene-relative near
    // fade. truncate is the DRAW pass's T (the pick stage passes its
    // uCoverageTruncate, synced from the visual material).
    float gsplatCoverageFade(mat3 Sigma_cam, float zDepth, int isOrtho, float truncate) {
        float halfResY = 0.5 * uResolution.y;
        float maxLateralVar = max(Sigma_cam[0][0], max(Sigma_cam[1][1], Sigma_cam[2][2]));
        float extentDivisor = (isOrtho == 1) ? 1.0 : max(zDepth, 1e-20);
        float projectedExtent = (halfResY * luxarProjectionSizeScale()) * sqrt(max(maxLateralVar, 1e-20)) * truncate / extentDivisor;
        float maxExtent = max(uResolution.x, uResolution.y) * uMaxExtentFactor;
        return 1.0 - smoothstep(maxExtent * 0.5, maxExtent, projectedExtent);
    }

    // Σ_2D = J · Σ_cam · Jᵀ with the projection Jacobian at the splat centre,
    // general form (valid for any P): J[k] = res/2 * (P[k].xy / w -
    // clip.xy * P[k].w / w^2). For three's symmetric perspective P it is the
    // classic [[fx/z, 0], [0, fy/z], [fx*x/z^2, fy*y/z^2]]; for ortho (w = 1,
    // P[k].w = 0) it is [[fx, 0], [0, fy], [0, 0]]. The x terms use P00 (not a
    // shared fx = fy), so a camera aspect that differs from the buffer aspect
    // is honoured instead of assumed away. Before the 2D low-pass dilation.
    mat2 gsplatProjectedCovariance(mat3 Sigma_cam, vec4 centerClip, float invW) {
        vec2 halfRes = 0.5 * uResolution;
        vec2 clipTerm = centerClip.xy * (invW * invW);
        mat3x2 J;
        J[0] = halfRes * (projectionMatrix[0].xy * invW - clipTerm * projectionMatrix[0].w);
        J[1] = halfRes * (projectionMatrix[1].xy * invW - clipTerm * projectionMatrix[1].w);
        J[2] = halfRes * (projectionMatrix[2].xy * invW - clipTerm * projectionMatrix[2].w);

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
        return Sigma2D;
    }

    // Sum projection: the 1D std-dev of the splat along the view ray,
    // sigma_line = 1 / sqrt(rᵀ Σ⁻¹ r) — not sqrt(rᵀ Σ r); the two only agree
    // when r is aligned with a covariance eigenvector or Σ is isotropic.
    // SCALE-FREE inversion: Σ_cam is normalized by its mean diagonal variance
    // s = trace/3 first. det(Σ) is world-units⁶ — on a tiny-unit scene (sigma
    // ~ 1e-7 => det ~ 1e-42) it underflows float32 (GPUs flush denormals to
    // zero) and an absolute 1e-12 clamp turned Σ⁻¹ into garbage; huge-unit
    // scenes overflow the same way. With Σn = Σ/s the determinant and ray
    // quadratic are O(1) at ANY scene scale, so the 1e-12 / 1e-8 floors below
    // act as pure scale-free CONDITION-NUMBER guards. Σ⁻¹ = Σn⁻¹ / s, so
    // sigmaRay = sqrt(s / quadN) restores the world-unit result exactly. The
    // 1e-30 floor on s only guards an all-zero (degenerate) covariance.
    float gsplatRaySigma(mat3 Sigma_cam, vec3 rayDir) {
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
        // quad is rᵀ Σn⁻¹ r (normalized space, O(1) for a well-conditioned
        // splat at any scale); un-normalize via sqrt(sTrace).
        float quad = max(rayDir.x * prx + rayDir.y * pry + rayDir.z * prz, 1e-8);
        return inversesqrt(quad) * sqrt(sTrace);
    }

    // Eigenvalues of the (dilated) Σ_2D, (lambda1, lambda2), for the oriented quad.
    vec2 gsplatEigenvalues(mat2 Sigma2D) {
        float trace = Sigma2D[0][0] + Sigma2D[1][1];
        float det = Sigma2D[0][0] * Sigma2D[1][1] - Sigma2D[0][1] * Sigma2D[1][0];
        float disc = max(trace * trace - 4.0 * det, 0.0);  // Clamp for numerical stability
        float sqrtDisc = sqrt(disc);
        return vec2(max(0.5 * (trace + sqrtDisc), 1e-6), max(0.5 * (trace - sqrtDisc), 1e-6));
    }

    // Eigenvector of the major axis (for the oriented quad).
    vec2 gsplatMajorAxis(mat2 Sigma2D, float lambda1) {
        if (abs(Sigma2D[0][1]) > 1e-6) {
            return normalize(vec2(lambda1 - Sigma2D[1][1], Sigma2D[0][1]));
        }
        // Near-diagonal covariance: pick axis with larger variance
        return (Sigma2D[0][0] >= Sigma2D[1][1]) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    }

    // Quad half extents truncate × sqrt(eigenvalue), clamped so no splat
    // exceeds uMaxExtentFactor × viewport. The amplitude fade (coverage)
    // handles the visual transition smoothly; this clamp prevents the
    // rasterizer from shading oversized quads.
    vec2 gsplatClampedExtents(float truncate, vec2 lambdas) {
        float extent1 = truncate * sqrt(lambdas.x);
        float extent2 = truncate * sqrt(lambdas.y);
        float maxExtentPx = max(uResolution.x, uResolution.y) * uMaxExtentFactor;
        float largestExtent = max(extent1, extent2);
        if (largestExtent > maxExtentPx) {
            float clampScale = maxExtentPx / largestExtent;
            extent1 *= clampScale;
            extent2 *= clampScale;
        }
        return vec2(extent1, extent2);
    }
`;
