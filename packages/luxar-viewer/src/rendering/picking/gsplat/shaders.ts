/**
 * GLSL3 picking shader source for Gaussian splats + `ShaderSource` record.
 *
 * Mirrors the visual gsplat shader with picking-specific adjustments:
 *   - `uNodeId` uniform + `vNodeId` / `vElementId` varyings, written
 *     into the RGBA32F pick buffer as `(nodeId, elementId-low16, brightness, elementId-high16)`.
 *   - Tighter truncation: 1.5σ (vs the node's own T, default 2.75σ, for
 *     the visual) so the pick footprint is the bright core only; the
 *     coverage fade still culls with the node's T (`uCoverageTruncate`).
 *   - The VISUAL amplitude and weight: the node's projection (the
 *     sum-projection ray-integral boost included), per-splat alpha, node
 *     opacity and gain (`../_shared/visibility-glsl.ts`), so a splat is
 *     pickable exactly when it is visible.
 *   - Brightness-as-depth so the brightest overlapping fragment wins —
 *     except in surface ('normal') mode (`uSurfaceDepth == 1`), where the
 *     real projected depth is written so the FRONT-MOST splat wins,
 *     matching the depth-sorted occluding surface the user sees.
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
  GLSL_PROJECTION_FUNCTIONS,
  GLSL_SORTED_INDEX,
} from '../../materials/_shared/glsl-lib';
import { requireTslMaterials } from '../../tsl/slot';
import {
  GLSL_GSPLAT_VISIBLE_FOOTPRINT,
  GSPLAT_VISIBILITY_FLOOR,
} from '../../materials/gsplat/math';
import { GLSL_PICK_VISIBILITY } from '../_shared/visibility-glsl';

/**
 * Picking vertex shader for gsplats.
 * Adds uNodeId/vNodeId/vElementId, strips colormap.
 * Uses tighter truncation (1.5σ) for more precise picking.
 */
export const GSPLAT_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}
    ${GLSL_PROJECTION_FUNCTIONS}
    ${GLSL_GSPLAT_VISIBLE_FOOTPRINT}
    ${GLSL_PICK_VISIBILITY}

    in vec2 aQuadCorner;

    // Draw-slot -> storage-slot mapping (identity in Phase 1; permuted
    // by the sort worker in Phase 2+). Also the pick ELEMENT id: the
    // pick buffer must report the storage slot -- the id the rest of
    // the pipeline (loaders, selection) addresses splats by -- not the
    // transient draw slot.
    ${GLSL_SORTED_INDEX}

    // Splat data texture: RGBA32F, 4 texels/splat (see
    // rendering/element-texture-layout.ts). Picking needs texels 0-2
    // only (center/amplitude/cholesky) -- color is not fetched.
    uniform highp sampler2D uSplatTex;

    uniform vec2 uResolution;
    uniform float uTruncate;
    // The DRAW pass's truncation radius (the node's T), synced from the
    // visual material per pick render: the coverage fade below is a cull
    // rule, not a footprint, and must cull exactly what the draw culls.
    uniform float uCoverageTruncate;
    uniform float uNearCull;
    uniform float uMaxExtentFactor;
    uniform float uCov2DDilation;     // 2D-covariance low-pass dilation in CSS px² (visual-shader parity)
    uniform float uPixelRatio;
    uniform float uNodeId;
    uniform int uLabelFilterIndex;
    // The visual node's projection (synced per pick render): 0 = sum
    // (ray-integral: additive/luminous/volumetric), 1 = peak (surface modes).
    // The amplitude the draw emits — and so whether a splat is visible —
    // differs between the two by the ray-integral boost.
    uniform int uProjectionMode;
    uniform float uRayIntegralFactor; // the visual node's (its own T), not the pick's
    // Fragment-discard inputs, read here to size the quad to the visible
    // footprint (visual-shader parity). Same precision as the fragment
    // declarations (GLSL ES link rule).
    uniform highp float uShiftC;
    uniform highp float uInvOneMinusC;
    uniform highp float uTruncateSq;

    flat out mediump float vAmplitude2D;
    // Visual-pass weight: alpha factor x opacity x max(gain, 1) — the factor
    // the fragment's visibility test multiplies its falloff by.
    flat out mediump float vPickWeight;
    flat out highp vec3 vL2D;
    flat out highp vec2 vCenterScreen;
    flat out highp float vNodeId;
    flat out highp vec2 vElementId;

    mat3 unpackCholesky3D(vec2 c01, vec2 c23, vec2 c45) {
        return mat3(
            c01.x, c01.y, c23.y,
            0.0,   c23.x, c45.x,
            0.0,   0.0,   c45.y
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
        // === Splat-texture fetch prologue (visual-shader parity) ===
        // Width is a multiple of 4, so a splat's texels share one row.
        // Projected-density thinning: a splat the visual pass dropped must not
        // be pickable either (same hash, same uniform value).
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
        vec3 aCenter = splatT0.xyz;
        float aAmplitude = splatT0.w;
        vec2 aCholesky01 = splatT1.xy;
        vec2 aCholesky23 = splatT1.zw;
        vec2 aCholesky45 = splatT2.xy;
        float aAlpha = splatT3.y;          // per-splat opacity (1.0 for RGB data)
        float aLabelIndex = splatT3.z;
        if (uLabelFilterIndex > 0 && int(aLabelIndex + 0.5) != uLabelFilterIndex) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }

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

        // Unified near handling — see the visual gsplat shader: the
        // shared perspectiveNearFade subsumes the old standalone
        // behind-camera reject; ortho falls through to NDC clipping.
        // 1e-20 floor = degenerate-smoothstep guard only; uNearCull is
        // scene-bounds-scaled (an absolute 1e-4 faded out tiny-unit
        // scenes entirely).
        float depthFade = perspectiveNearFade(isOrtho, centerCam.z, max(uNearCull, 1e-20));
        if (depthFade < 0.01) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }

        mat3 R = mat3(modelViewMatrix);
        mat3 L3D = unpackCholesky3D(aCholesky01, aCholesky23, aCholesky45);
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
        float halfResY = 0.5 * uResolution.y;
        float coverageFade;
        {
            // 1e-20 floors = pure div-by-zero/sqrt guards, matching the
            // visual shader: maxLateralVar is world-unit² (an absolute
            // 1e-8 floor coverage-culled every splat of a tiny-unit
            // scene); zDepth is bounded by the scene-relative near fade.
            float maxLateralVar = max(Sigma_cam[0][0], max(Sigma_cam[1][1], Sigma_cam[2][2]));
            float extentDivisor = (isOrtho == 1) ? 1.0 : max(zDepth, 1e-20);
            float projectedExtent = (halfResY * luxarProjectionSizeScale()) * sqrt(max(maxLateralVar, 1e-20)) * uCoverageTruncate / extentDivisor;
            float maxExtent = max(uResolution.x, uResolution.y) * uMaxExtentFactor;
            coverageFade = 1.0 - smoothstep(maxExtent * 0.5, maxExtent, projectedExtent);
            if (coverageFade < 0.01) {
                gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
                return;
            }
        }

        float nearFade = min(depthFade, coverageFade);

        // Projection Jacobian at the splat centre, general form (valid for any
        // P): J[k] = res/2 * (P[k].xy / w - clip.xy * P[k].w / w^2). For
        // three's symmetric perspective P it is the classic
        // [[fx/z, 0], [0, fy/z], [fx*x/z^2, fy*y/z^2]]; for ortho (w = 1,
        // P[k].w = 0) it is [[fx, 0], [0, fy], [0, 0]]. The x terms use P00
        // (not a shared fx = fy), so a camera aspect that differs from the
        // buffer aspect is honoured instead of assumed away.
        vec2 halfRes = 0.5 * uResolution;
        vec2 clipTerm = centerClip.xy * (invW * invW);
        mat3x2 J;
        J[0] = halfRes * (projectionMatrix[0].xy * invW - clipTerm * projectionMatrix[0].w);
        J[1] = halfRes * (projectionMatrix[1].xy * invW - clipTerm * projectionMatrix[1].w);
        J[2] = halfRes * (projectionMatrix[2].xy * invW - clipTerm * projectionMatrix[2].w);

        vec2 JS0 = J[0] * Sigma_cam[0][0] + J[1] * Sigma_cam[0][1] + J[2] * Sigma_cam[0][2];
        vec2 JS1 = J[0] * Sigma_cam[1][0] + J[1] * Sigma_cam[1][1] + J[2] * Sigma_cam[1][2];
        vec2 JS2 = J[0] * Sigma_cam[2][0] + J[1] * Sigma_cam[2][1] + J[2] * Sigma_cam[2][2];

        mat2 Sigma2D;
        Sigma2D[0][0] = JS0.x * J[0].x + JS1.x * J[1].x + JS2.x * J[2].x;
        Sigma2D[1][0] = JS0.x * J[0].y + JS1.x * J[1].y + JS2.x * J[2].y;
        Sigma2D[0][1] = Sigma2D[1][0];
        Sigma2D[1][1] = JS0.y * J[0].y + JS1.y * J[1].y + JS2.y * J[2].y;

        // 2D low-pass dilation — visual-shader parity (shader-glsl.ts). Widens
        // the pickable footprint to match the dilated visual splat, so what you
        // click matches what you see. The energy compensation is the visual
        // one too: the sum projection below reads it.
        float detRaw2D = Sigma2D[0][0] * Sigma2D[1][1] - Sigma2D[0][1] * Sigma2D[1][0];
        float dilationPixelRatio = max(uPixelRatio, 1.0);
        float cov2DDilation = uCov2DDilation * dilationPixelRatio * dilationPixelRatio;
        Sigma2D[0][0] += cov2DDilation;
        Sigma2D[1][1] += cov2DDilation;
        float detDilated2D = Sigma2D[0][0] * Sigma2D[1][1] - Sigma2D[0][1] * Sigma2D[1][0];
        float dilationCompensation = sqrt(max(detRaw2D, 0.0) / max(detDilated2D, 1e-12));

        // Visual-shader parity (shader-glsl.ts) + TSL-side parity
        // (gsplat-pick.tsl.ts): reject splats with NaN/Inf Σ_2D or
        // amplitude so picking and rendering agree on which elements
        // are pickable across WebGL and WebGPU backends.
        if (invalidCov2D(Sigma2D) || isInvalidFloat(aAmplitude)) {
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
            return;
        }

        // The amplitude the VISUAL pass emits, so pickability tracks visibility:
        // the sum projection's ray-integral boost (sigmaRay is in world units, so
        // it can make a splat far brighter or far dimmer than its peak) and the
        // dilation energy compensation; the peak projection's plain amplitude.
        // Same scale-free Σ_cam inversion as the visual shader (shader-glsl.ts
        // carries the derivation).
        if (uProjectionMode == 0) {
            vec3 rayDir = (isOrtho == 1) ? vec3(0.0, 0.0, -1.0) : normalize(centerCam);
            float sTrace = max((Sigma_cam[0][0] + Sigma_cam[1][1] + Sigma_cam[2][2]) * (1.0 / 3.0), 1e-30);
            float invS = 1.0 / sTrace;
            float a = Sigma_cam[0][0] * invS;
            float b = Sigma_cam[0][1] * invS;
            float c = Sigma_cam[0][2] * invS;
            float d = Sigma_cam[1][1] * invS;
            float e = Sigma_cam[1][2] * invS;
            float f = Sigma_cam[2][2] * invS;
            float detSigma = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
            float invDet = 1.0 / max(detSigma, 1e-12);
            float i00 = (d * f - e * e) * invDet;
            float i11 = (a * f - c * c) * invDet;
            float i22 = (a * d - b * b) * invDet;
            float i01 = -(b * f - c * e) * invDet;
            float i02 = (b * e - c * d) * invDet;
            float i12 = -(a * e - b * c) * invDet;
            float prx = i00 * rayDir.x + i01 * rayDir.y + i02 * rayDir.z;
            float pry = i01 * rayDir.x + i11 * rayDir.y + i12 * rayDir.z;
            float prz = i02 * rayDir.x + i12 * rayDir.y + i22 * rayDir.z;
            float quad = max(rayDir.x * prx + rayDir.y * pry + rayDir.z * prz, 1e-8);
            float sigmaRay = inversesqrt(quad) * sqrt(sTrace);
            float rayIntegrationBoost = sigmaRay * uRayIntegralFactor;
            vAmplitude2D = aAmplitude * rayIntegrationBoost * nearFade * dilationCompensation;
        } else {
            vAmplitude2D = aAmplitude * nearFade;
        }

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

        // Visible-footprint tightening — visual-shader parity (shader-glsl.ts;
        // derivation in materials/gsplat/math.ts). The pick fragment's
        // visibility test multiplies its falloff by the visual weight
        // (vPickWeight = this alpha factor x max(gain, 1)), so the quad is
        // sized by the SAME peak-scale function with the same factors; the
        // alpha factor additionally carries the node opacity (a transparent
        // node is invisible, so it must not be pickable either).
        float splatAlpha = sanitizeAlpha(aAlpha);
        float pickAlphaFactor = luxarPickAlphaFactor(splatAlpha) * uOpacity;
        vPickWeight = pickAlphaFactor * max(uIntensity, 1.0);
        float visibleMahalSq = gsplatVisibleMahalSq(
            gsplatFootprintPeakScale(vAmplitude2D, uInvOneMinusC, pickAlphaFactor, uIntensity),
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

        vec2 quadOffset = aQuadCorner.x * majorAxis * extent1
                        + aQuadCorner.y * minorAxis * extent2;
        vec2 screenPos = vCenterScreen + quadOffset;
        vec2 ndcXY = (screenPos / uResolution) * 2.0 - 1.0;

        // Depth through the same projection (the clip-space centre above).
        float ndcZ = centerClip.z / centerClip.w;

        gl_Position = vec4(ndcXY, ndcZ, 1.0);

        vNodeId = uNodeId;
        // Storage slot, NOT gl_InstanceID (the draw slot): identical
        // under Phase-1 identity ordering, and stays correct once the
        // sort worker permutes draw order (Phase 2+).
        vElementId = luxarElementIdParts();
    }
`;

/**
 * Picking fragment shader for gsplats.
 * Outputs vec4(nodeId, elementId-low16, brightness, elementId-high16) with brightness-as-depth
 * (or real projected depth when `uSurfaceDepth == 1` — surface/'normal' mode).
 * Uses tighter truncation (1.5σ squared = 2.25) for precise picking.
 */
export const GSPLAT_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    flat in mediump float vAmplitude2D;
    flat in mediump float vPickWeight;
    flat in highp vec3 vL2D;
    flat in highp vec2 vCenterScreen;
    flat in highp float vNodeId;
    flat in highp vec2 vElementId;

    uniform highp float uShiftC;
    uniform highp float uInvOneMinusC;
    uniform highp float uTruncateSq;
    // 1 = surface ('normal') mode: write real projected depth (front-most
    // wins). 0 = commutative modes: brightness-as-depth (brightest wins).
    uniform int uSurfaceDepth;

    out vec4 fragColor;

    void main() {
        vec2 d = gl_FragCoord.xy - vCenterScreen;

        float y0 = d.x * vL2D.x;
        float y1 = (d.y - vL2D.y * y0) * vL2D.z;
        float mahalSq = y0 * y0 + y1 * y1;

        if (mahalSq > uTruncateSq) discard;

        float intensity = vAmplitude2D * uInvOneMinusC * max(exp(-0.5 * mahalSq) - uShiftC, 0.0);
        // Visual salience: the falloff times what the draw scales it by
        // (alpha factor, opacity) and the gain its discard honours. The vertex
        // stage sizes the quad from this exact test (gsplatVisibleMahalSq) —
        // change both together.
        float salience = intensity * vPickWeight;
        if (salience < ${GSPLAT_VISIBILITY_FLOOR.toExponential()}) discard;

        float brightness = clamp(salience, 0.0, 1.0);

        fragColor = vec4(vNodeId, vElementId.x, brightness, vElementId.y);
        // Pick depth convention (synced from the MAIN material's blending
        // mode by PickingSystem.renderPickBuffer):
        //   - surface ('normal') mode: the user sees a depth-sorted
        //     occluding surface, so write the real projected depth (the
        //     vertex puts the splat-center NDC z in gl_Position.z, so
        //     gl_FragCoord.z is the true depth) — FRONT-MOST wins.
        //   - commutative modes (additive/max/luminous):
        //     brightness-as-depth — BRIGHTEST wins.
        gl_FragDepth = (uSurfaceDepth == 1) ? gl_FragCoord.z : 1.0 - brightness;
    }
`;

export const GSPLAT_PICK_SOURCE: ShaderSource = {
  name: 'gsplat-pick',
  webgl: { vertex: GSPLAT_PICK_VERTEX_SHADER, fragment: GSPLAT_PICK_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    const { gsplatPickWebGPUFactory, buildGSplatPickTSLNodesFromUniforms } =
      requireTslMaterials().factories.pickGsplat;
    return gsplatPickWebGPUFactory(buildGSplatPickTSLNodesFromUniforms(u));
  },
};
