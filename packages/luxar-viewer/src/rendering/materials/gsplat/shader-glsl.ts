/**
 * GLSL3 vertex + fragment shaders for Gaussian splat rendering (GSplatMaterial).
 *
 * Contains 3D-to-2D covariance projection, perspective Jacobian, amplitude calculation,
 * oriented quad expansion, near-plane fade, and screen-coverage safety. The GPU pick
 * pass has its own shader pair (picking/gsplat/shaders.ts) that shares the
 * visible-footprint helpers (GLSL_GSPLAT_VISIBLE_FOOTPRINT, ./math.ts) with this one.
 */
import {
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_NEAR_FADE_FUNCTIONS,
  GLSL_PROJECTION_FUNCTIONS,
  GLSL_SORTED_INDEX,
  GLSL_DENSITY_ALPHA,
} from '../_shared/glsl-lib';
import {
  GLSL_GLASS_PARTITION_GUARD,
  GLSL_GLASS_PARTITION_UNIFORMS,
} from '../_shared/glass-partition';
import type { ShaderSource } from '../_shared/shader-source';
import {
  ALPHA_CLAMP,
  VOLUMETRIC_SERIES_C1,
  VOLUMETRIC_SERIES_C2_DIVISOR,
  VOLUMETRIC_SERIES_TAU_THRESHOLD,
  VOLUMETRIC_TAU_EPS,
} from '../_shared/volumetric';
import { requireTslMaterials } from '../../tsl/slot';
import { GLSL_GSPLAT_VISIBLE_FOOTPRINT, GSPLAT_VISIBILITY_FLOOR } from './math';
import { GLSL_GSPLAT_PROJECTION } from './projection-glsl';

export const GSPLAT_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}
    ${GLSL_PROJECTION_FUNCTIONS}
    ${GLSL_GSPLAT_VISIBLE_FOOTPRINT}

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
    uniform float uTruncate;          // Truncation radius (in sigmas)
    uniform float uRayIntegralFactor; // Shifted Gaussian ray integral factor
    uniform int uProjectionMode;      // 0 = sum (ray-integral: additive/luminous/volumetric), 1 = peak (2D-projected surfaces: max/normal/opaque)
    uniform float uNearCull;          // Near cull distance (scene-scale-aware)
    uniform float uMaxExtentFactor;   // Max projected extent as fraction of viewport before fade
    uniform float uCov2DDilation;     // 2D-covariance low-pass dilation in CSS px² (3DGS anti-aliasing)
    uniform float uPixelRatio;
    uniform int uLabelColorMode;
    uniform int uLabelFilterIndex;
    // Fragment-discard inputs, read here to size the quad to the VISIBLE
    // footprint (see the tightening block in main). A uniform declared in
    // both stages must carry the SAME precision (GLSL ES link rule), so
    // these mirror the fragment declarations exactly.
    uniform highp float uShiftC;
    uniform highp float uInvOneMinusC;
    uniform highp float uTruncateSq;
    uniform mediump float uIntensity;
    uniform lowp float uHasElementAlpha;

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

    // Covariance projection shared with the pick stage (unpackCholesky3D,
    // gsplatSigmaCam, invalidCov2D, coverage fade, Jacobian, ray sigma, eigen
    // axes, clamped extents).
    ${GLSL_GSPLAT_PROJECTION}

    // invalidFloat is an alias for the shared isInvalidFloat helper in glsl-lib.
    bool invalidFloat(float v) {
        return isInvalidFloat(v);
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
        // Projected-density thinning (density-guard): drop this instance
        // before any texel fetch; the rasterizer discards a z=-2 vertex.
        if (luxarDensityDropped()) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }
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

        // Clip-space centre through the projection THIS draw uses. The screen
        // centre, the covariance Jacobian, the coverage extent and the ortho
        // branch all come from it and from P, so a splat is placed and sized
        // for whatever camera three draws with: a cube-capture face (fov -90
        // flips P), a zoomed or asymmetric frustum, an embedder's camera.
        // CPU mirror + tests: projection-math.ts.
        vec4 centerClip = projectionMatrix * centerCam4;
        int isOrtho = luxarIsOrthoProjection();
        float invW = 1.0 / centerClip.w;

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
        float depthFade = perspectiveNearFade(isOrtho, centerCam.z, max(uNearCull, 1e-20));
        if (depthFade < 0.01) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }

        // Camera-space covariance (rotation only; projection-glsl.ts).
        mat3 Sigma_cam = gsplatSigmaCam(aCholesky01, aCholesky23, aCholesky45);

        // Positive depth (camera looks down -Z); reused by fades, Jacobian, and projection
        float zDepth = -centerCam.z;

        // === Screen-coverage safety guard (independent of depth fade) ===
        // Rationale and the tiny-unit history: gsplatCoverageFade in
        // projection-glsl.ts (shared with the pick stage, which passes the
        // same node T as uCoverageTruncate).
        float coverageFade = gsplatCoverageFade(Sigma_cam, zDepth, isOrtho, uTruncate);
        if (coverageFade < 0.01) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }

        // Combined fade: most restrictive wins (both use smoothstep → no popping)
        float nearFade = min(depthFade, coverageFade);

        // Project covariance to 2D: Σ_2D = J · Σ_cam · Jᵀ with the general
        // projection Jacobian (projection-glsl.ts).
        mat2 Sigma2D = gsplatProjectedCovariance(Sigma_cam, centerClip, invW);

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
        // Keep the historical framebuffer-pixel low-pass below 1× render scale.
        float dilationPixelRatio = max(uPixelRatio, 1.0);
        float cov2DDilation = uCov2DDilation * dilationPixelRatio * dilationPixelRatio;
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
            // Sum projection: the ray-integral standard deviation
            // sigma_line = 1 / sqrt(rᵀ Σ⁻¹ r), from a scale-free Σ_cam
            // inversion (gsplatRaySigma in projection-glsl.ts carries the
            // derivation).
            vec3 rayDir = (isOrtho == 1) ? vec3(0.0, 0.0, -1.0) : normalize(centerCam);
            sigmaRay = gsplatRaySigma(Sigma_cam, rayDir);
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

        // Eigen-axes of Σ_2D and the max-extent-clamped quad half extents
        // (projection-glsl.ts — shared with the pick stage).
        vec2 lambdas = gsplatEigenvalues(Sigma2D);
        float lambda1 = lambdas.x;
        float lambda2 = lambdas.y;
        vec2 majorAxis = gsplatMajorAxis(Sigma2D, lambda1);
        vec2 minorAxis = vec2(-majorAxis.y, majorAxis.x);
        vec2 extents = gsplatClampedExtents(uTruncate, lambdas);
        float extent1 = extents.x;
        float extent2 = extents.y;

        // === Visible-footprint tightening (#2944 B10) ===
        // The quad above circumscribes the T-sigma ellipse, but the fragment
        // also discards below the visibility floor, which binds FIRST for a
        // dim splat: its surviving fragments fill the smaller ellipse of
        // squared radius gsplatVisibleMahalSq (exact rearrangement of that
        // discard; derivation + margins in math.ts). Shrink each half extent
        // to it (+1 px), never growing the legacy quad, and cull a splat none
        // of whose fragments can pass. The peak scale below is EXACTLY the
        // factor the fragment multiplies its falloff by before the test:
        // vAmplitude2D · 1/(1-C) · alpha factor · max(gain, 1).
        float splatAlpha = sanitizeAlpha(aAlpha);
        #ifdef LUXAR_VOLUMETRIC
        float footprintAlpha = mix(1.0, -log(1.0 - min(splatAlpha, ${ALPHA_CLAMP})), uHasElementAlpha);
        #else
        float footprintAlpha = splatAlpha;
        #endif
        float visibleMahalSq = gsplatVisibleMahalSq(
            gsplatFootprintPeakScale(vAmplitude2D, uInvOneMinusC, footprintAlpha, uIntensity),
            uShiftC, uTruncateSq);
        if (visibleMahalSq < 0.0) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }
        extent1 = gsplatFootprintExtent(extent1, lambda1, visibleMahalSq);
        extent2 = gsplatFootprintExtent(extent2, lambda2, visibleMahalSq);

        // Screen centre in pixels from the clip-space centre (gl_FragCoord
        // convention: origin at the viewport's bottom-left corner).
        vCenterScreen = (centerClip.xy * invW * 0.5 + 0.5) * uResolution;

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

        // Depth through the same projection (the clip-space centre above).
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
 * The picking system uses a different fragment shader (see picking/gsplat/shaders.ts).
 */
export const GSPLAT_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;
    ${GLSL_GLASS_PARTITION_UNIFORMS}
    ${GLSL_DENSITY_ALPHA}

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
      ${GLSL_GLASS_PARTITION_GUARD}
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
        // The vertex stage sizes the quad from this exact test
        // (gsplatVisibleMahalSq) — change both together.
        if (intensity * max(uIntensity, 1.0) < ${GSPLAT_VISIBILITY_FLOOR.toExponential()}) discard;

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
        // Density-guard thinning of an alpha-over node: raise the coverage
        // to 1 − (1 − c)^(1/keep) and the premultiplied RGB with it, so the
        // kept fraction occludes and emits what the whole node did. Skipped
        // (bit-identical) at the identity exponent.
        if (uDensityAlphaExp > 1.0) {
          float thinned = luxarDensityAlpha(coverage);
          finalColor *= thinned / max(coverage, 1e-6);
          coverage = thinned;
        }
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
