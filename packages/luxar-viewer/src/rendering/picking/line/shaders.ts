/**
 * GLSL3 picking shader source for lines + `ShaderSource` record.
 *
 * Mirrors the visual line shader's near-plane safety, max-pixel-width
 * clamp, and width/sharpness sanitization so the pick footprint stays
 * in lock-step with the visible footprint. Strips colormap and color
 * varyings (not needed for picking) and adds the `uNodeId` uniform and
 * `vNodeId` / `vElementId` varyings written into the RGBA32F pick
 * buffer as `(nodeId, elementId-low16, brightness, elementId-high16)`. Brightness-as-depth
 * keeps overlapping segments correctly resolved.
 *
 * Lines use **full** pick width (unlike points / gsplats which truncate
 * — thin lines are already narrow with a sharp parabolic profile).
 *
 * Source-of-truth for GLSL3; the WebGPU counterpart lives in `./pick.tsl`
 * and is referenced through the `ShaderSource.webgpu` factory below.
 *
 * @module rendering/picking/line/shaders
 */

import type { ShaderSource } from '../../materials/_shared/shader-source';
import {
  GLSL_NEAR_FADE_FUNCTIONS,
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_SORTED_INDEX,
  GLSL_LINE_JOINT_CODE,
  GLSL_LINE_JOIN,
} from '../../materials/_shared/glsl-lib';
import { linePickWebGPUFactory, buildLinePickTSLNodesFromUniforms } from './pick.tsl';
import { lineJoinStyleFromUniform } from '../../../types/line-join';
import { FALLOFF_FLOOR, FALLOFF_K } from '../../materials/_shared/falloff';

/**
 * Picking vertex shader for lines.
 * Adds uNodeId uniform and vNodeId/vElementId outputs.
 * Strips colormap and color varyings (not needed for picking).
 * Mirrors the visual line shader's near-plane safety, max-pixel-width
 * clamp, and width/sharpness sanitization for pick/visual parity.
 */
export const LINE_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;
    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}

    in vec2 aQuadCorner;

    // Draw-slot → storage-slot mapping (visual-shader parity).
    ${GLSL_SORTED_INDEX}
    ${GLSL_LINE_JOINT_CODE}

    // Line data texture: RGBA32F, 6 texels/segment (see
    // rendering/line-geometry.ts for the texel layout).
    uniform highp sampler2D uLineTex;

    uniform vec2 uResolution;
    uniform int uIsOrtho;
    uniform float uNodeId;
    uniform float uNearCull;          // visual-shader parity
    uniform float uMaxLinePixelWidth; // visual-shader parity
    // CPU-precomputed pixel-width scales — visual-shader parity.
    uniform float uPerspectiveLineScale; // = resolution.y / tan(fov * 0.5)
    uniform float uOrthoLineScale;       // = 2 * resolution.y / frustumHeight

    // Screen-space miter join (#790) — same block the visual vertex stage
    // includes, so the pick footprint keeps matching the visible one at a
    // mitred corner. Declares uLineJoin; MUST follow the uniforms above.
    ${GLSL_LINE_JOIN}

    out float vSharpness;
    out float vPerpNorm;
    out float vT;             // fragment-side cap math (parity with visual)
    flat out float vSegmentLength; // per-segment constant — visual-shader parity
    out float vWidthAtT;
    out float vPixelWidth;
    out float vWidthFade;     // visual-shader parity
    out float vViewZ;         // View-space z (fragment computes the near fade)
    flat out float vCapSuppressStart;
    flat out float vCapSuppressEnd;
    flat out highp float vNodeId;
    flat out highp vec2 vElementId;

    void main() {
      // === Line-texture fetch prologue (visual-shader parity) ===
      // Width is a multiple of 6, so a segment's texels share one row.
      // Colors (texels 2/3 .rgb) and scalars (texel 5) are not needed
      // for picking; only the .w sharpness of texels 2/3 is read.
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
      float aStartSharpness = texelFetch(uLineTex, ivec2(texel0.x + 2, texel0.y), 0).w;
      float aEndSharpness = texelFetch(uLineTex, ivec2(texel0.x + 3, texel0.y), 0).w;
      float aSegmentLength = lineT4.x;
      float aStartJointCode = lineT4.y;
      float aEndJointCode = lineT4.z;

      // Branchless: aQuadCorner.x ∈ {-1, +1} by construction.
      float t = aQuadCorner.x * 0.5 + 0.5;
      vT = t;
      vSegmentLength = aSegmentLength;
      // texel4.yz hold a per-endpoint joint CODE, not a [0, 1] scalar — reading
      // it as one made brightness scale with the partner's slot index (a
      // segment joining slot 399 reached capFactor 200.5), which both broke the
      // brightness < 1e-4 discard for clipped endpoints and let high-slot
      // segments win every gl_FragDepth comparison. Decode it the same way the
      // visual vertex stage does. This is the DEFAULT; the join block further
      // down replaces it with the screen-space value wherever it reaches a
      // partner, exactly as in the visual shader.
      vCapSuppressStart = luxarLineJointCapSuppression(aStartJointCode);
      vCapSuppressEnd = luxarLineJointCapSuppression(aEndJointCode);

      // sanitize width/sharpness against negative/NaN/Inf. Sharpness is a
      // [0, 1] knob -> super-Gaussian exponent beta = 2^(6s - 2) (computed
      // in the fragment); a valid s=0 must NOT be rejected, so clamp a
      // non-negative-sanitised value into [0, 1] with the 0.5 default.
      // Mirrors the visual shader (shader-glsl.ts).
      float startW = sanitizeNonNegative(aStartWidth, 0.0);
      float endW = sanitizeNonNegative(aEndWidth, 0.0);
      float startS = clamp(sanitizeNonNegative(aStartSharpness, 0.5), 0.0, 1.0);
      float endS = clamp(sanitizeNonNegative(aEndSharpness, 0.5), 0.0, 1.0);

      vec4 mvStart = modelViewMatrix * vec4(aStartPos, 1.0);
      vec4 mvEnd = modelViewMatrix * vec4(aEndPos, 1.0);

      // Visual-shader parity: near-plane safety (degenerate quad if both endpoints behind).
      // PERSPECTIVE ONLY — see the visual line shader: under ortho NDC
      // clipping is the sole cull authority (the ungated cull wrongly
      // made near-slab lines unpickable while points/gsplats picked).
      // 1e-20 floor = uNearCull == 0 guard only; uNearCull is
      // scene-bounds-scaled (see the visual line shader — an absolute
      // 1e-4 floor culled every segment of a tiny-unit scene).
      float nearCull = max(uNearCull, 1e-20);
      float startDepth = -mvStart.z;
      float endDepth = -mvEnd.z;
      bool bothBehind =
        (uIsOrtho == 0) && (startDepth < nearCull) && (endDepth < nearCull);
      if (bothBehind) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        // Defensive: width/sharpness are computed AFTER the clip
        // (compute-once from tEff, visual-shader parity), so zero the
        // fragment-readable varyings here — the rasterizer drops this
        // segment, but uninitialised out-vars can trip driver validators.
        vSharpness = 0.5;
        vWidthAtT = 0.0;
        vPerpNorm = 0.0;
        vPixelWidth = 0.0;
        vWidthFade = 0.0;
        vViewZ = 0.0;
        vNodeId = uNodeId;
        vElementId = luxarElementIdParts();
        return;
      }

      // Near-plane SEGMENT clipping — visual-shader parity (see
      // shader-glsl.ts for the full rationale: a behind-camera endpoint
      // has clip w <= 0, which flips the clip-space expansion and
      // rasterizes the quad as a twisted bowtie whose near-clip boundary
      // cuts through the pick footprint). Keeps every vertex at
      // viewZ >= nearCull and remaps t (tEff) so the cap math and the
      // per-endpoint attributes keep the original parameterization.
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
      float tEff = mix(tA, tB, t);
      vT = tEff;
      // View-space z to the fragment — the fade is computed per-fragment
      // there (interpolating the fade itself is wrong on long segments;
      // see the visual line shader).
      vec4 mvPos = mix(mvStart, mvEnd, t);
      vViewZ = mvPos.z;
      // Compute-once from the clipped tEff (visual-shader parity); the
      // raw-t values are never read before this point.
      float width = mix(startW, endW, tEff);
      vSharpness = mix(startS, endS, tEff);
      vWidthAtT = width;

      vec4 clipStart = projectionMatrix * mvStart;
      vec4 clipEnd = projectionMatrix * mvEnd;
      // projection is linear, so proj * mix(a,b,t) == mix(proj*a, proj*b, t).
      vec4 clipPos = mix(clipStart, clipEnd, t);

      // Scene-relative w guard (w == -viewZ under perspective; ortho
      // w == 1, guard inert) — see the visual line shader for why an
      // absolute 1e-4 scrambled tiny-unit scenes and a raw 1e-20 could
      // overflow the pixel-length math.
      float wGuard = (uIsOrtho == 1) ? 1.0 : nearCull;
      float wStart = max(clipStart.w, wGuard);
      float wEnd = max(clipEnd.w, wGuard);
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
        // Visual-shader parity — see shader-glsl.ts.
        float dist = max(-mvPos.z, nearCull);
        rawPixelWidth = width * uPerspectiveLineScale / dist;
      }

      float minPixelWidth = 1.5;
      float maxPW = max(uMaxLinePixelWidth, minPixelWidth + 1.0);
      // Visual-shader parity: discard pathological near-camera segments
      // (both endpoints inside near-cull margin AND the pixel width blows
      // past clamp by 2×). Without this, picking still rasterizes the
      // half-viewport quad the visual pass already culled.
      // Segment-constant, not per-quad-vertex: the per-vertex
      // rawPixelWidth term differs between the t=0 and t=1 corners, so
      // gating on it would sentinel only half the quad and leave a
      // visible wedge (issue #849). Gate on the MAX of the pixel width at
      // both clipped endpoints so all four vertices take the same branch.
      float startPixelWidth =
        mix(startW, endW, tA) * uPerspectiveLineScale / max(-mvStart.z, nearCull);
      float endPixelWidth =
        mix(startW, endW, tB) * uPerspectiveLineScale / max(-mvEnd.z, nearCull);
      float segMaxPixelWidth = max(startPixelWidth, endPixelWidth);
      if (
        uIsOrtho == 0 &&
        startDepth < nearCull * 2.0 &&
        endDepth < nearCull * 2.0 &&
        segMaxPixelWidth > maxPW * 2.0
      ) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        vPerpNorm = 0.0;
        vPixelWidth = 0.0;
        vWidthFade = 0.0;
        vViewZ = 0.0;
        vNodeId = uNodeId;
        vElementId = luxarElementIdParts();
        return;
      }
      float clampedPixelWidth = clamp(rawPixelWidth, minPixelWidth, maxPW);
      vWidthFade = (rawPixelWidth <= maxPW) ? 1.0 : (maxPW / max(rawPixelWidth, 1e-4));
      vPixelWidth = rawPixelWidth;
      vPerpNorm = aQuadCorner.y;

      // Join geometry (#790) — visual-shader parity. The pick quad must be the
      // SAME quad the eye sees, or the mitred corner region becomes unpickable
      // (and the plain-perpendicular corner outside it picks a segment that no
      // longer draws there). Same shared helper, same operands, so the two
      // stages agree by construction rather than by two copies of the math
      // staying in sync. Both ends are evaluated on every vertex so the two
      // "flat" cap varyings stay segment-constant — see shader-glsl.ts for why
      // a per-corner write splits the quad along its diagonal.
      // Per-END widths, not this vertex's: the gate inside luxarLineJoin must be
      // segment-constant or the "flat" cap varyings below resolve from whichever
      // corner provokes (see luxarLineEndPixelWidth). Geometrically identical —
      // each equals clampedPixelWidth at the corner that consumes it.
      float startEndPixelWidth = clamp(
        luxarLineEndPixelWidth(mix(startW, endW, tA), mvStart.z, nearCull),
        minPixelWidth, maxPW
      );
      float endEndPixelWidth = clamp(
        luxarLineEndPixelWidth(mix(startW, endW, tB), mvEnd.z, nearCull),
        minPixelWidth, maxPW
      );
      vec3 startJoin = luxarLineJoin(
        false, tA <= 0.0, aStartJointCode, ndcStart,
        lineDir, pixelLen, startEndPixelWidth, nearCull
      );
      vec3 endJoin = luxarLineJoin(
        true, tB >= 1.0, aEndJointCode, ndcEnd,
        lineDir, pixelLen, endEndPixelWidth, nearCull
      );
      if (startJoin.z >= 0.0) vCapSuppressStart = startJoin.z;
      if (endJoin.z >= 0.0) vCapSuppressEnd = endJoin.z;

      vec2 cornerOffset = (aQuadCorner.x > 0.0) ? endJoin.xy : startJoin.xy;
      vec2 pixelOffset = cornerOffset * aQuadCorner.y;
      vec2 ndcOffset = pixelOffset / uResolution * 2.0;
      clipPos.xy += ndcOffset * clipPos.w;

      gl_Position = clipPos;

      vNodeId = uNodeId;
      // Storage slot, NOT gl_InstanceID (the draw slot): identical
      // under identity ordering, and stays correct once the sort
      // worker permutes draw order.
      vElementId = luxarElementIdParts();
    }
`;

/**
 * Picking fragment shader for lines.
 * Outputs vec4(nodeId, elementId-low16, brightness, elementId-high16) with brightness-as-depth.
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
    ${GLSL_NEAR_FADE_FUNCTIONS}

    uniform int uIsOrtho;   // shared with the vertex stage
    uniform float uNearCull;

    in float vSharpness;
    in float vPerpNorm;
    in float vT;
    flat in float vSegmentLength; // per-segment constant
    in float vWidthAtT;
    in float vPixelWidth;
    in float vWidthFade;
    in float vViewZ; // near fade computed here per-fragment
    flat in float vCapSuppressStart;
    flat in float vCapSuppressEnd;
    flat in highp float vNodeId;
    flat in highp vec2 vElementId;

    out vec4 fragColor;

    void main() {
      float p = abs(vPerpNorm);

      // Full width — lines are already narrow, no need for tighter truncation
      if (p >= 1.0) discard;

      // Shifted-truncated super-Gaussian perpendicular cross-section —
      // visual-shader parity. beta = 2^(6s - 2) from the [0, 1] knob.
      // K = ln(100), C = exp(-K). See shader-glsl.ts.
      const float K = ${FALLOFF_K};
      const float C = ${FALLOFF_FLOOR};
      const float INV_ONE_MINUS_C = 1.0 / (1.0 - C);
      float beta = exp2(6.0 * vSharpness - 2.0);
      float perpFalloff = max(exp(-K * pow(p, beta)) - C, 0.0) * INV_ONE_MINUS_C;

      float minPixelWidth = 1.5;
      float widthScale = min(vPixelWidth / minPixelWidth, 1.0);

      // cap factor in fragment (matches visual shader).
      float distFromStart = vT * vSegmentLength;
      float distFromEnd = (1.0 - vT) * vSegmentLength;
      // Scale-free ratio; 1e-20 = pure div-by-zero guard (visual twin).
      float startRamp = vWidthAtT > 1e-20
        ? clamp(distFromStart / vWidthAtT, 0.0, 1.0)
        : 1.0;
      float endRamp = vWidthAtT > 1e-20
        ? clamp(distFromEnd / vWidthAtT, 0.0, 1.0)
        : 1.0;
      // Per-endpoint cap lifted by its own suppression, combined with
      // min() — removes the intra-segment midpoint jump (visual twin;
      // a residual sub-width joint-seam step is documented there).
      float startCap = mix(0.5 + 0.5 * startRamp, 1.0, vCapSuppressStart);
      float endCap = mix(0.5 + 0.5 * endRamp, 1.0, vCapSuppressEnd);
      float capFactor = min(startCap, endCap);

      // 1e-20 floor = degenerate-smoothstep guard only (scene-relative
      // uNearCull; see the vertex-stage nearCull note).
      float nearFade = perspectiveNearFade(uIsOrtho, vViewZ, max(uNearCull, 1e-20));
      float brightness = capFactor * perpFalloff * widthScale * vWidthFade * nearFade;
      if (brightness < 1e-4) discard;

      fragColor = vec4(vNodeId, vElementId.x, brightness, vElementId.y);
      gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0);
    }
`;

export const LINE_PICK_SOURCE: ShaderSource = {
  name: 'line-pick',
  webgl: { vertex: LINE_PICK_VERTEX_SHADER, fragment: LINE_PICK_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    const isOrtho = ((u.uIsOrtho?.value as number) ?? 0) === 1;
    // The join style likewise (see LINE_SOURCE): GLSL carries it as a runtime
    // uniform, TSL as a graph variant, so the record must select the variant.
    const join = lineJoinStyleFromUniform(u.uLineJoin?.value as number | undefined);
    return linePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(u), { isOrtho, join });
  },
};
