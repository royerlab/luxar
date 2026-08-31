/**
 * Shared vertex shader for line rendering.
 *
 * Used by both LineMaterial (main rendering) and LinePickingMaterial (GPU picking).
 * Contains screen-space expansion, cap factor calculation, and colormap support.
 *
 * Shader contracts:
 *   - Cap factor is evaluated in the fragment shader. The vertex shader
 *     passes `vT`, `vSegmentLength`, `vWidthAtT`, `vCapSuppressStart`, and
 *     `vCapSuppressEnd`; the fragment evaluates the documented "0.5 at
 *     FREE endpoints, 1.0 in body" profile per fragment, with the
 *     suppression scalars lifting interior joints and clipped ends to 1.0.
 *   - Near-plane / behind-camera safety rejects segments where both
 *     endpoints are behind/near the camera (clipPos.w → 0/negative
 *     produces invalid NDC and a full-screen quad). When `uNearCull` is
 *     set, segments closer than that view-space depth are degenerated.
 *   - Near-plane SEGMENT clipping: when exactly one endpoint is closer
 *     than `uNearCull` (or behind the camera), it is moved along the
 *     segment onto the nearCull plane before any screen-space math and
 *     `t` is remapped (`tEff`) so attributes / cap math keep the
 *     original parameterization. Without it a behind-camera endpoint
 *     (clip w <= 0) flips the clip-space expansion and rasterizes the
 *     quad as a twisted bowtie whose near-clip boundary cuts a bright
 *     razor edge through the profile at close zoom.
 *   - Max pixel width clamp. `pixelWidth` is clamped to
 *     `uMaxLinePixelWidth` (default `resY * 0.5`); when the clamp
 *     engages, intensity fades proportionally so a single very-near
 *     segment doesn't paint the screen.
 *   - Width and sharpness sanitised against negative/NaN/Inf.
 *
 * Shader defines:
 *   - `USE_COLORMAP` — enables the texel5 scalar fetch + colormap LUT
 *     path. Set in `LineMaterial` when the node carries real colormap
 *     scalars (the `userData.hasScalars` stamp — the fixed 6-texel
 *     layout always has the slot, so presence rides the stamp).
 *   - `LUXAR_MAX_RGB_CONTRIBUTION` — fragment-side define that
 *     premultiplies `rgb *= alpha` before output so the
 *     `MaxEquation` + `OneFactor`/`OneFactor` blend captures
 *     contribution-weighted colour (a bright-but-thin fragment loses
 *     against a dim-but-thick fragment), not flat full-bright. Set by
 *     `LineMaterial.applyBlendingMode('max')` and cleared when
 *     switching back. Without this define, max-mode line rendering
 *     looks "stranded" — every visible fragment paints at full
 *     intensity regardless of opacity, intensity, or fade.
 *   - `LUXAR_VOLUMETRIC` — fragment-side emission–absorption output
 *     branch (Max 1995): τ = κ × the same ray mass every other mode
 *     emits, self-screened emission over the
 *     One/OneMinusSrcAlpha state. Set by
 *     `LineMaterial.applyBlendingMode('volumetric')` (volumetric
 *     phase 4 — VOLUMETRIC_BLENDING_SPEC.md §7).
 */
import {
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_NEAR_FADE_FUNCTIONS,
  GLSL_SORTED_INDEX,
  GLSL_LINE_JOINT_CODE,
  GLSL_LINE_JOIN,
} from '../_shared/glsl-lib';
import {
  ALPHA_CLAMP,
  VOLUMETRIC_SERIES_C1,
  VOLUMETRIC_SERIES_C2_DIVISOR,
  VOLUMETRIC_SERIES_TAU_THRESHOLD,
  VOLUMETRIC_TAU_EPS,
} from '../_shared/volumetric';
import { lineJoinStyleFromUniform } from '../../../types/line-join';
import { FALLOFF_FLOOR, FALLOFF_K } from '../_shared/falloff';
import type { ShaderSource } from '../_shared/shader-source';
import { requireTslMaterials } from '../../tsl/slot';

export const LINE_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}

    // Static geometry attribute (per quad vertex)
    in vec2 aQuadCorner;  // (-1,-1), (1,-1), (-1,1), (1,1)

    // Draw-slot → storage-slot mapping, double-buffered so a new ordering
    // swaps atomically (declaration + luxarSortedIndex() in glsl-lib).
    // Uint32Array attributes → bound via vertexAttribIPointer, matching
    // the uint declarations.
    ${GLSL_SORTED_INDEX}
    ${GLSL_LINE_JOINT_CODE}

    // Line data texture: RGBA32F, 6 texels/segment (see
    // rendering/line-geometry.ts for the texel layout). Replaces the
    // interleaved era's 11–13 instanced attributes — the colormap
    // attribute-set toggle (GL_MAX_VERTEX_ATTRIBS pressure) is gone.
    uniform highp sampler2D uLineTex;

    // Uniforms
    uniform vec2 uResolution;
    uniform float uPixelRatio;
    uniform int uIsOrtho;  // 0 = perspective, 1 = orthographic
    uniform float uNearCull;          // near-plane safety distance (view-space, +z toward camera)
    uniform float uMaxLinePixelWidth; // clamp for screen-space width
    // Precomputed CPU-side line-width scales — kill the per-vertex tan()
    // and one division. See updateCameraParams in line-material.ts.
    uniform float uPerspectiveLineScale; // = resolution.y / tan(fov * 0.5)
    uniform float uOrthoLineScale;       // = 2 * resolution.y / frustumHeight

    // Screen-space miter join (#790) — declares uLineJoin and defines
    // luxarLinePixelPos + luxarLineJoin. MUST follow the uniforms above:
    // it reads uLineTex, uResolution and uIsOrtho.
    ${GLSL_LINE_JOIN}

    // Colormap uniforms (only active when USE_COLORMAP is defined)
    #ifdef USE_COLORMAP
    uniform sampler2D uColormapTex;
    uniform float uScalarMin;       // display-range window minimum
    uniform float uScalarScale;     // 1.0 / (max - min)
    uniform float uInvGamma;        // gamma applied to the VALUE, pre-LUT
    #endif

    // Varyings to fragment shader
    out vec3 vColor;
    out float vSharpness;
    out float vPerpNorm;     // Signed: -1 at bottom edge, +1 at top edge
    out float vT;            // interpolated 0..1 along segment for fragment-side cap math
    flat out float vSegmentLength; // per-segment constant — same on all 4 quad verts
    out float vWidthAtT;      // interpolated world-space width (or half-width)
    out float vPixelWidth;   // Raw line width in pixels (for anti-aliasing)
    out float vWidthFade;    // in [0..1], fades intensity when pixel-width clamped
    out float vViewZ;        // View-space z (fragment computes the near fade)
    flat out float vCapSuppressStart; // flat: same value across all 4 quad vertices
    flat out float vCapSuppressEnd;
    out mediump float vAlpha; // per-endpoint opacity, interpolated along t (texel5.zw; 1.0 for RGB data)

    void main() {
      // === Line-texture fetch prologue ===
      // texelFetch reads reconstruct the per-segment values into the exact
      // local names the math below has always used — zero changes
      // downstream of this block. The width is a multiple of 6
      // (element-texture-layout.ts), so a segment's 6 texels share one row
      // and only x advances. Texels 2/3 (colors + sharpness) and 5
      // (scalars in .xy, per-endpoint alphas in .zw) are fetched only PAST
      // the bothBehind cull below, keeping the cheap-cull ordering the
      // interleaved shader had.
      int lineBase = int(luxarSortedIndex()) * 6;
      int lineTexW = LUXAR_LINE_TEX_W;
      ivec2 texel0 = ivec2(lineBase % lineTexW, lineBase / lineTexW);
      vec4 lineT0 = texelFetch(uLineTex, texel0, 0);
      vec4 lineT1 = texelFetch(uLineTex, ivec2(texel0.x + 1, texel0.y), 0);
      vec4 lineT4 = texelFetch(uLineTex, ivec2(texel0.x + 4, texel0.y), 0);
      vec3 aStartPos = lineT0.xyz;
      float aStartWidth = lineT0.w;
      vec3 aEndPos = lineT1.xyz;
      float aEndWidth = lineT1.w;
      float aSegmentLength = lineT4.x;
      float aStartJointCode = lineT4.y;
      float aEndJointCode = lineT4.z;

      // Position along segment: 0 = start, 1 = end. Branchless because
      // aQuadCorner.x ∈ {-1, +1} by construction.
      float t = aQuadCorner.x * 0.5 + 0.5;
      vT = t;
      vSegmentLength = aSegmentLength;
      // Endpoint cap default, before the join block below may refine it.
      //
      // ONLY a free end (code 0) and a degree->=3 hub (code -2) keep the soft
      // cap. Everything else suppresses it: a slice-clipped endpoint (-1)
      // because no neighbour will ever arrive there, and a slot-bearing code
      // because a neighbouring quad DOES meet it.
      //
      // Defaulting a slot-bearing code to "keep the cap" would be the #780 bead
      // chain all over again, and not only under join style 'none': the join
      // block is also skipped for every line below the rendered-width gate, so
      // thin-line scenes — the million-segment ones — would lose the #785 fix
      // entirely. Measured, an interior joint bottoms out at 0.5 instead of 1.0
      // and a dense polyline (segment length <= width) loses ~40% of its total
      // brightness.
      //
      // The default is exact for the straight and gentle joints that dominate
      // real polyline data, and those are projection-invariant so no camera can
      // change the answer. Where the block DOES run it replaces this with the
      // screen-space value, which is exact at any angle. The residual gap is a
      // SHARP bend on a line too thin to be mitered: it keeps full intensity on
      // both quads over their sub-pixel overlap lens instead of half each. That
      // is a ~1 px speck slightly too bright, against the alternative of dimming
      // every joint in the scene.
      vCapSuppressStart = luxarLineJointCapSuppression(aStartJointCode);
      vCapSuppressEnd = luxarLineJointCapSuppression(aEndJointCode);

      // Project endpoints to view space first — the bothBehind near-cull
      // test reads view-space depth, and culling BEFORE the colormap
      // sample / width sanitisation / colour interpolation skips that
      // wasted work for off-screen segments. The pathological-cull
      // (further down) reads rawPixelWidth which depends on width,
      // so we still have to do the cheap parts of width sanitisation
      // and interpolation; the colormap branch and the full mix /
      // pow chain are the real wins.
      vec4 mvStart = modelViewMatrix * vec4(aStartPos, 1.0);
      vec4 mvEnd = modelViewMatrix * vec4(aEndPos, 1.0);

      // near-plane / behind-camera safety — PERSPECTIVE ONLY. Three.js
      // view space has -z pointing into the scene, so a positive
      // viewDepth means the point is in front of the camera. Reject
      // segments where BOTH endpoints fail the near-cull (degenerate
      // the quad to clip). When only ONE endpoint is behind, the
      // segment is clipped onto the nearCull plane below (before any
      // screen-space math) so the quad stays a true trapezoid instead
      // of a razor-edged bowtie. Under ORTHO there is no 1/z
      // singularity and NDC near/far clipping is the sole cull
      // authority — the previous ungated cull WRONGLY hid in-frustum
      // lines in the near slab (< uNearCull from the camera plane)
      // where points/gsplats still drew. Matches the unified
      // perspectiveNearFade semantics (point + gsplat shaders).
      // uNearCull is scene-bounds-scaled (diagonal * 0.001); the 1e-20
      // floor only guards uNearCull == 0 (degenerate smoothstep /
      // division). An absolute 1e-4 floor overrode the scene-relative
      // value on tiny-unit scenes — every segment sat inside the
      // "both behind" margin and was culled.
      float nearCull = max(uNearCull, 1e-20);
      float startDepth = -mvStart.z;
      float endDepth = -mvEnd.z;
      bool bothBehind =
        (uIsOrtho == 0) && (startDepth < nearCull) && (endDepth < nearCull);
      if (bothBehind) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0); // off-screen → no fragments
        // Defensive: zero the remaining varyings the fragment-stage
        // can read. The rasterizer drops this segment entirely so the
        // values don't actually matter, but uninitialised out-vars can
        // trip driver validators on some platforms.
        vColor = vec3(0.0);
        vSharpness = 0.5;
        vWidthAtT = 0.0;
        vPerpNorm = 0.0;
        vPixelWidth = 0.0;
        vWidthFade = 0.0;
        vViewZ = 0.0;
        vAlpha = 1.0;
        return;
      }

      // Near-plane SEGMENT clipping (perspective only). When exactly one
      // endpoint sits closer than nearCull (or behind the camera), move it
      // along the segment onto the nearCull plane BEFORE any screen-space
      // math. Without this, a behind-camera endpoint has clipPos.w <= 0:
      // its NDC is mirrored across the origin AND the clip-space expansion
      // (ndcOffset * clipPos.w below) flips sign, so the quad rasterizes
      // as a twisted bowtie whose hardware near-clip boundary slices
      // through the MIDDLE of the Gaussian cross-profile — a bright razor
      // edge running along the line's side at close zoom. Clipping keeps
      // every vertex at viewZ >= nearCull, so quads stay true trapezoids,
      // and the cut lands exactly where the per-fragment near fade reaches
      // zero — no visible seam. View-space depth is linear along the
      // segment, so the plane intersection is exact. t is remapped onto
      // the clipped sub-range so ALL per-endpoint attributes (width,
      // color, sharpness, alpha, scalars) and the cap math (vT against
      // the ORIGINAL vSegmentLength) keep their original parameterization.
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

      vec4 mvPos = mix(mvStart, mvEnd, t);
      // View-space z travels to the FRAGMENT, which computes the near
      // fade per-fragment. Interpolating the FADE itself would be wrong
      // on long segments: fade(lerp(z)) != lerp(fade(z)) — one endpoint
      // at the camera plane would dim fragments far outside the
      // [nearCull, 2*nearCull] band (mid-segment at ~50%).
      vViewZ = mvPos.z;

      // === Below here only runs when the segment passed the cheap cull. ===

      // Deferred texel fetches (colors + sharpness + texel5) — skipped
      // entirely for cheap-culled segments. texel5 carries the colormap
      // scalars (.xy, read under USE_COLORMAP) and the per-endpoint
      // alphas (.zw, written unconditionally by the texel writer — 1.0
      // for RGB data), so it is fetched in every mode.
      vec4 lineT2 = texelFetch(uLineTex, ivec2(texel0.x + 2, texel0.y), 0);
      vec4 lineT3 = texelFetch(uLineTex, ivec2(texel0.x + 3, texel0.y), 0);
      vec4 lineT5 = texelFetch(uLineTex, ivec2(texel0.x + 5, texel0.y), 0);
      float aStartSharpness = lineT2.w;
      float aEndSharpness = lineT3.w;

      // Per-endpoint opacity, interpolated along the segment (1.0 for
      // RGB data). Each texel read is sanitized BEFORE the mix: alpha
      // is load-bearing in EVERY mode (linear contribution scale) and
      // maps into optical depth under volumetric, where a NaN/Inf
      // poisons τ past the discard into NaN pixels — and a huge finite
      // value would blow out the linear folds (or overflow the mediump
      // varying). The guarantee is "NaN never reaches τ/pixels", NOT
      // per-endpoint containment: a NaN source vertex already
      // propagates to BOTH of the segment's texel alphas upstream (the
      // worker's lerp kernel), so the whole segment renders loud-opaque
      // — same containment as every other lerped attribute. Python
      // validation pins alpha to [0, 1] at write; this guards
      // hand-crafted zarr. NaN/Inf → the 1.0 opaque identity (loud);
      // finite values clamp to [0, 1]. The point/gsplat twins do the
      // same.
      vAlpha = mix(sanitizeAlpha(lineT5.z), sanitizeAlpha(lineT5.w), tEff);

      // Interpolate attributes along segment.
      // Colormap mode: display range (uScalarMin/uScalarScale) and gamma
      // shape the scalar VALUE before the LUT lookup, not the resulting
      // color; intensity/offset apply POST-LUT to the mapped color
      // (fragment shader, matching the gsplat shader) so the layer
      // gain/offset controls work on colormapped nodes too. The gamma
      // fast path (LUXAR_GAMMA_ONE) skips the pow() when gamma == 1.0.
      #ifdef USE_COLORMAP
      float s = mix(lineT5.x, lineT5.y, tEff);
      float st = clamp((s - uScalarMin) * uScalarScale, 0.0, 1.0);
      #ifndef LUXAR_GAMMA_ONE
      st = pow(st, uInvGamma);          // gamma on the value, pre-LUT
      #endif
      vColor = texture(uColormapTex, vec2(st, 0.5)).rgb;
      #else
      vColor = mix(lineT2.rgb, lineT3.rgb, tEff);
      #endif

      // sanitise width/sharpness against negative/NaN/Inf so a
      // malformed input can't poison gl_Position via pow() or screen-
      // space expansion. Sanitize helpers from glsl-lib.
      float startW = sanitizeNonNegative(aStartWidth, 0.0);
      float endW = sanitizeNonNegative(aEndWidth, 0.0);
      // Sharpness is authored in [0, 1] and maps (in the fragment) to the
      // super-Gaussian exponent beta = 2^(6s - 2): s=0.5 -> beta=2 (a true
      // Gaussian, the default). sanitizeNonNegative keeps a valid s=0
      // (-> beta=0.25) and routes NaN/Inf/negative to the 0.5 default;
      // clamp guards the [0, 1] range. (NOT sanitizePositive — that would
      // wrongly reject s=0.)
      float startS = clamp(sanitizeNonNegative(aStartSharpness, 0.5), 0.0, 1.0);
      float endS = clamp(sanitizeNonNegative(aEndSharpness, 0.5), 0.0, 1.0);

      float width = mix(startW, endW, tEff);
      // Pass the interpolated [0, 1] sharpness KNOB to the fragment; beta is
      // computed there from the interpolated value.
      vSharpness = mix(startS, endS, tEff);
      vWidthAtT = width;

      vec4 clipStart = projectionMatrix * mvStart;
      vec4 clipEnd = projectionMatrix * mvEnd;
      // projection is linear, so proj * mix(a,b,t) == mix(proj*a, proj*b, t).
      vec4 clipPos = mix(clipStart, clipEnd, t);

      // Convert clip-space endpoints to pixel coordinates for correct aspect ratio handling
      // Guard against tiny clipStart.w / clipEnd.w (near-plane crossing) so 1/w
      // doesn't blow up. The guard is the SCENE-RELATIVE nearCull (w == -viewZ
      // under perspective), not an absolute epsilon: the old 1e-4 clamped VALID
      // w on tiny-unit scenes (w ~ 1e-6), collapsing every endpoint's NDC and
      // scrambling quad directions. nearCull scales with the scene, so in-front
      // endpoints are never clamped at any scale while behind/crossing endpoints
      // still get a bounded NDC (clip.xy scales with the scene too, keeping the
      // ratio finite — a raw 1e-20 floor could overflow float32 in the
      // pixel-length math below). Ortho: w == 1 exactly, guard 1.0 is inert.
      float wGuard = (uIsOrtho == 1) ? 1.0 : nearCull;
      float wStart = max(clipStart.w, wGuard);
      float wEnd = max(clipEnd.w, wGuard);
      vec2 ndcStart = clipStart.xy / wStart;
      vec2 ndcEnd = clipEnd.xy / wEnd;

      // Compute line direction in pixel space (aspect-ratio correct).
      // The +0.5 in (ndc*0.5+0.5)*resolution cancels under subtraction.
      vec2 pixelDir = (ndcEnd - ndcStart) * (0.5 * uResolution);
      float pixelLen = length(pixelDir);

      // Handle degenerate segments (zero length in screen space)
      vec2 lineDir = pixelLen > 0.0001 ? pixelDir / pixelLen : vec2(1.0, 0.0);
      vec2 perpendicular = vec2(-lineDir.y, lineDir.x);  // Unit vector in pixel space

      // World-space to pixel conversion. Both branches consume a scale
      // precomputed on the CPU once per camera/resolution change so
      // the shader avoids per-vertex tan() and FOV divisions.
      float rawPixelWidth;
      if (uIsOrtho == 1) {
        // Orthographic: constant screen size regardless of distance.
        rawPixelWidth = width * uOrthoLineScale;
      } else {
        // View-space depth instead of Euclidean distance — drops a
        // sqrt per vertex and is more projection-correct (screen-space
        // size scales with view-z, not distance from camera position).
        // Off-axis segments will be slightly different in apparent
        // width vs the old length(mvPos.xyz) form; this is the
        // intended correctness improvement.
        float dist = max(-mvPos.z, nearCull);
        rawPixelWidth = width * uPerspectiveLineScale / dist;
      }

      // Enforce minimum pixel width to prevent sub-pixel rendering artifacts
      // Lines thinner than ~1.5 pixels cause severe aliasing due to rasterization gaps
      float minPixelWidth = 1.5 * uPixelRatio;
      // clamp to a maximum pixel width so a near-camera segment
      // can't paint the entire screen. Default uMaxLinePixelWidth is
      // resolution.y * 0.5 (set by JS).
      float maxPW = max(uMaxLinePixelWidth, minPixelWidth + 1.0);
      // Degenerate (extremely close to camera AND extreme pixel
      // width) segments expand into a half-viewport quad that the GPU
      // still rasterizes pixel-by-pixel. The pixel-width clamp + fade
      // keeps the visible footprint bounded but doesn't avoid the
      // shading cost — discard the segment entirely when both endpoints
      // are within the near cull margin AND the pixel width blows past
      // the clamp by 2× (a clear pathological case, not a normal
      // close-up).
      // PERSPECTIVE ONLY: under ortho rawPixelWidth is depth-independent
      // (width * uOrthoLineScale), so a depth gate here would make a
      // legitimately wide line vanish only while inside the 2*nearCull
      // slab and pop back one unit deeper — depth-dependent visibility
      // with no physical rationale in a depth-independent projection.
      // The discard MUST be segment-constant, not per-quad-vertex: the
      // per-vertex rawPixelWidth term (both width and dist vary between
      // the t=0 and t=1 corners of the shared quad) would let only 2 of
      // the 4 vertices exceed the clamp, sentinelling half the quad and
      // leaving a visible non-degenerate wedge (issue #849). Evaluate the
      // pixel width at BOTH clipped endpoints and gate on the MAX so all
      // four vertices take the same branch — this reproduces the max of
      // the per-vertex rawPixelWidth over the quad (only t∈{0,1} occur),
      // so it never culls a segment a current vertex wouldn't have.
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
        vColor = vec3(0.0);
        vPerpNorm = 0.0;
        vPixelWidth = 0.0;
        vWidthFade = 0.0;
        vViewZ = 0.0;
        return;
      }
      // Fade intensity in proportion to the clamp so the giant quad
      // doesn't overcontribute. fade=1 when not clamped, →0 as the
      // raw width grows past the clamp by a factor.
      vWidthFade = (rawPixelWidth <= maxPW) ? 1.0 : (maxPW / max(rawPixelWidth, 1e-4));

      // Pass raw pixel width to fragment shader for intensity scaling
      // This allows thin lines to render at minimum width but with reduced intensity
      vPixelWidth = rawPixelWidth;

      // Perpendicular position: -1 at bottom edge, +1 at top edge
      // GPU interpolates this across the quad, giving 0 at centerline
      vPerpNorm = aQuadCorner.y;

      // === Join geometry at degree-2 polyline joints (#790) ===
      //
      // luxarLineJoin (glsl-lib.ts, shared with the pick vertex stage) returns
      // one END's pixel-space corner offset in .xy and, in .z, either the
      // screen-space endpoint cap it derived from the partner's direction or
      // -1.0 meaning "no partner reached — keep the code-implied default".
      // Under join style "none", below the rendered-width gate, or at a free
      // end, .xy is exactly the plain perpendicular half-width this shader has
      // always used.
      //
      // BOTH ends are evaluated on EVERY vertex, and only the offset is then
      // selected per corner. That is not redundancy — vCapSuppressStart/End are
      // "flat" varyings, so a value that differs between the four quad corners
      // is resolved from the provoking vertex alone, and the two triangles of
      // one quad have DIFFERENT provoking vertices (indices [0,1,2, 2,1,3]
      // provoke v2=start and v3=end under WebGL's last-vertex rule). Writing
      // the cap only at the corner it belongs to therefore split the quad
      // diagonally wherever the refined value differed from the default — and
      // WGSL's "@interpolate(flat)" provokes from the FIRST vertex, so the two
      // backends disagreed as well. Evaluating both ends everywhere makes the
      // two caps segment-constant, which is what "flat" requires.
      //
      // Per-END widths, not this vertex's: the gate inside luxarLineJoin must be
      // segment-constant or the "flat" cap varyings below resolve from whichever
      // corner provokes (see luxarLineEndPixelWidth). Geometrically identical —
      // each equals the per-vertex clamped width at the corner that consumes it
      // (at a t=0 vertex tEff == tA and mvPos == mvStart, symmetrically at t=1),
      // so only the DECISIONS become segment-constant. Same #849 reasoning as
      // segMaxPixelWidth above.
      //
      // The trailing depth argument is the segment's FAR endpoint, from the
      // ORIGINAL pre-clipping depths: the near-plane guard inside the helper
      // needs both far endpoints of the joint, so the start call passes the
      // END's depth and the end call the START's.
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
        lineDir, pixelLen, startEndPixelWidth, endDepth, nearCull
      );
      vec3 endJoin = luxarLineJoin(
        true, tB >= 1.0, aEndJointCode, ndcEnd,
        lineDir, pixelLen, endEndPixelWidth, startDepth, nearCull
      );
      if (startJoin.z >= 0.0) vCapSuppressStart = startJoin.z;
      if (endJoin.z >= 0.0) vCapSuppressEnd = endJoin.z;
      vec2 cornerOffset = (aQuadCorner.x > 0.0) ? endJoin.xy : startJoin.xy;

      // Expand quad by the corner offset in pixel space, then convert to clip
      // space. pixelOffset is in pixels, convert to NDC then to clip space
      vec2 pixelOffset = cornerOffset * aQuadCorner.y;
      vec2 ndcOffset = pixelOffset / uResolution * 2.0;
      clipPos.xy += ndcOffset * clipPos.w;

      gl_Position = clipPos;
    }
  `;

/**
 * Fragment shader for standard line rendering.
 *
 * Computes the shifted-truncated super-Gaussian perpendicular
 * cross-section (beta = 2^(6s - 2), beta=2 is a truncated Gaussian) plus
 * fragment-side cap factor so the segment body reaches the documented
 * full intensity. Under `LUXAR_VOLUMETRIC` the output switches to the
 * emission–absorption branch (τ = κ × the same ray mass every other mode
 * emits). The picking system uses a
 * different fragment shader (see picking/line-picking-material.ts).
 */
export const LINE_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;
    ${GLSL_NEAR_FADE_FUNCTIONS}

    uniform int uIsOrtho;   // shared with the vertex stage
    uniform float uNearCull;
    uniform float uPixelRatio;
    uniform float uOpacity;
    uniform float uInvGamma; // Pre-computed 1/gamma for performance
    uniform float uIntensity; // Per-node linear color multiplier (gain)
    uniform float uOffset; // Per-node additive brightness shift (black level)

    // Volumetric (emission–absorption) uniforms — read only under
    // LUXAR_VOLUMETRIC; inert (compiled out) in every other mode.
    uniform highp float uAbsorption;         // κ — composed node absorption
    uniform lowp float uHasElementAlpha;     // 1.0 when colors carry a real alpha column

    in vec3 vColor;
    in float vSharpness;
    in float vPerpNorm;     // Interpolated: 0 at centerline, ±1 at edges
    in float vT;            // interpolated 0..1 along segment
    flat in float vSegmentLength; // per-segment constant
    in float vWidthAtT;     // interpolated world-space width
    in float vPixelWidth;   // Raw line width in pixels (before minimum clamping)
    in float vWidthFade;    // max-pixel-width clamp fade
    in float vViewZ; // View-space z (near fade computed here per-fragment)
    flat in float vCapSuppressStart;
    flat in float vCapSuppressEnd;
    in mediump float vAlpha; // per-endpoint opacity, interpolated along t (1.0 for RGB data)

    out vec4 fragColor;

    void main() {
      // Compute distance from centerline (0 to 1)
      float p = abs(vPerpNorm);

      // Discard pixels clearly outside the line width
      if (p >= 1.0) discard;

      // Shifted-truncated super-Gaussian perpendicular cross-section:
      //   perpFalloff(p) = max(exp(-K * p^beta) - C, 0) / (1 - C)
      // where p = distance from centerline in [0, 1] and the per-vertex
      // sharpness KNOB s in [0, 1] maps to beta = 2^(6s - 2): s=0.5 ->
      // beta=2 (a truncated Gaussian, the default), s=1 -> beta=16 (hard
      // edge), s=0 -> beta=0.25 (cusp). Shifted by C and renormalised so
      // perpFalloff(0)=1 and perpFalloff(1)=0 (C0-continuous truncation at
      // the line edge, no hard ring). K = ln(1/floor), floor = 0.01.
      const float K = ${FALLOFF_K};          // ln(100)
      const float C = ${FALLOFF_FLOOR};               // exp(-K) = floor
      const float INV_ONE_MINUS_C = 1.0 / (1.0 - C);
      float beta = exp2(6.0 * vSharpness - 2.0);
      float perpFalloff = max(exp(-K * pow(p, beta)) - C, 0.0) * INV_ONE_MINUS_C;

      // Anti-aliasing: smooth falloff at edges
      // The AA region is ~1 pixel wide in the rendered quad
      // Since we enforce minimum 1.5px width, use that as reference
      float minPixelWidth = 1.5 * uPixelRatio;
      float renderedWidth = max(vPixelWidth, minPixelWidth);
      float aaWidth = 1.0 / renderedWidth;  // ~1 pixel in normalized coords
      float edgeAA = 1.0 - smoothstep(1.0 - aaWidth, 1.0, p);

      // Intensity scaling for sub-pixel lines
      // When a line is rendered wider than intended, reduce intensity proportionally
      // This preserves the visual "weight" of thin lines
      float widthScale = min(vPixelWidth / minPixelWidth, 1.0);

      // cap factor in fragment. With the 4-vertex quad, vertex-side
      // computation produced 0.5 everywhere. Compute it here so the
      // segment body reaches the documented 1.0.
      // - one ramp per endpoint, from the distance to that endpoint
      //   along the segment (world units)
      // - the per-endpoint suppression scalar (texel4.yz, [0, 1]) lifts
      //   its endpoint's ramp back to 1.0 wherever dimming would be wrong: a
      //   slice-clipped endpoint, or a straight-through interior joint
      //   whose neighbouring quad TILES rather than overlaps (nothing
      //   there to add the missing half back). Computed once per commit
      //   from the joint code in the vertex stage (see compute_joint_codes,
      //   wasm/rust/src/lines_clipping.rs).
      float distFromStart = vT * vSegmentLength;
      float distFromEnd = (1.0 - vT) * vSegmentLength;
      // dist / vWidthAtT is a scale-free ratio (both world units), so
      // the guard is a pure div-by-zero threshold at 1e-20 — an
      // absolute 1e-4 skipped the cap ramp for valid sub-1e-4-unit
      // widths (tiny-unit scenes). clamp() bounds the quotient.
      float startRamp = vWidthAtT > 1e-20
        ? clamp(distFromStart / vWidthAtT, 0.0, 1.0)
        : 1.0;
      float endRamp = vWidthAtT > 1e-20
        ? clamp(distFromEnd / vWidthAtT, 0.0, 1.0)
        : 1.0;

      // Each endpoint's cap ramp is lifted towards full intensity by ITS
      // OWN suppression, then the two are combined with min(). Keying the
      // ramp on the nearest endpoint only was discontinuous at the
      // midpoint of segments shorter than 2*width when the suppressions
      // differ (the first/last segment of every polyline: one free end,
      // one suppressed joint) — though exactly continuous ACROSS the
      // joint seam. min() RELOCATES that discontinuity: continuous within
      // the segment, with a strictly smaller step moved to the seam, and
      // only when the segment is shorter than ONE width (its far-end ramp
      // cannot reach 1.0 before the neighbour takes over). Worst case
      // 0.5*(1 - clamp(L/w)) (far end fully free, joint fully
      // suppressed); the general case scales by (1 - s_far). Always <=
      // the old midpoint jump, zero for L >= width. min() reduces to the
      // old single-ramp behaviour when both suppressions are equal or the
      // two width-sized cap regions do not overlap (the far ramp
      // saturates at 1.0).
      float startCap = mix(0.5 + 0.5 * startRamp, 1.0, vCapSuppressStart);
      float endCap = mix(0.5 + 0.5 * endRamp, 1.0, vCapSuppressEnd);
      float capFactor = min(startCap, endCap);

      // Apply cap factor for correct joint intensity
      // Per-fragment near fade from the interpolated view depth (see
      // the vertex stage note on why the fade itself must not be the
      // varying).
      // 1e-20 floor = degenerate-smoothstep guard only; uNearCull is
      // scene-relative (see the vertex-stage nearCull note).
      float nearFade = perspectiveNearFade(uIsOrtho, vViewZ, max(uNearCull, 1e-20));
      float intensity = capFactor * perpFalloff * edgeAA * widthScale * vWidthFade * nearFade;

      // Per-node GOG (Gain-Offset-Gamma) color adjustment. uIntensity (gain)
      // and uOffset apply in BOTH modes so the layer intensity/offset
      // controls work for a colormapped line too (matching the gsplat
      // shader). Colormap (LUT) mode: gamma + the display-range window
      // already shaped the scalar VALUE before the LUT lookup (vertex
      // shader), so only gain/offset apply post-LUT (no extra gamma).
      // When the wrapper knows intensity==1 && offset==0 (the default),
      // the mul/add/clamp chain is identity for the common non-negative
      // vColor range; the wrapper stamps LUXAR_NO_GOG to skip it.
      #ifdef LUXAR_NO_GOG
      vec3 adjusted = vColor;
      #else
      vec3 adjusted = vColor * uIntensity + uOffset;
      adjusted = max(adjusted, vec3(0.0));
      #endif

      #ifdef LUXAR_VOLUMETRIC
      // Screen density of this fragment — the intensity chain already
      // carries every "how much of this line is there" factor (cap,
      // profile, AA coverage, sub-pixel energy, width-clamp fade, near
      // fade); node opacity folds in here. This is the additive-mode
      // alpha, and volumetric scales it by the per-endpoint alpha's
      // optical-depth map w(a) = −ln(1 − a) so alpha composes as
      // optical depth ("peak rendered alpha = a" is exact when the
      // remaining τ factor is 1; mirrors the point/gsplat shaders;
      // clamp = ALPHA_CLAMP from ../_shared/volumetric). Gated by
      // uHasElementAlpha: the identity 1.0 written for RGB data must
      // NOT map to w ≈ 6.24.
      float alpha = intensity * uOpacity;
      alpha *= mix(1.0, -log(1.0 - min(vAlpha, ${ALPHA_CLAMP})), uHasElementAlpha);
      // 'volumetric' optical depth: kappa x the SAME ray mass every other
      // mode emits (VOLUMETRIC_BLENDING_SPEC.md §3.1 / §7). The alpha above
      // is already the complete ray mass — the intensity chain carries every
      // "how much of this line is there" factor and opacity is a peak SCREEN
      // ALPHA, an integrated quantity. It used to be multiplied by a world
      // thickness (width * sqrt(pi/K)) as though opacity were a volume
      // density still awaiting integration; that read it as a density in
      // this one mode and as a peak alpha in the other five. Mirrors the
      // point twin (materials/point/shader-glsl.ts, full rationale there)
      // and the gsplat rule tau = kappa * opacity * intensity.
      float tau = uAbsorption * alpha;
      // Discard only when color AND τ are both negligible — a black
      // line still absorbs (a pure-ink occluder keeps its optical depth).
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4 && tau < 1e-4) discard;
      #else
      // Per-endpoint alpha is a plain linear contribution scale in
      // every non-volumetric mode (identity 1.0 for RGB data — no gate
      // needed).
      intensity *= vAlpha;
      #ifdef LUXAR_OPAQUE_RGB_CONTRIBUTION
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) * intensity * uOpacity < 1e-4) discard;
      #else
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;
      #endif
      #endif

      // Gamma fast path: when gamma==1 (the default) the pow() is
      // identity. The wrapper class stamps LUXAR_GAMMA_ONE on the
      // material defines whenever gamma transitions to/from 1.0, so
      // this skips three per-fragment pow() calls in the common case.
      // Colormap mode also skips the color pow() — gamma is on the value.
      #if defined(USE_COLORMAP) || defined(LUXAR_GAMMA_ONE)
      vec3 gammaColor = adjusted;
      #else
      vec3 gammaColor = pow(adjusted, vec3(uInvGamma));
      #endif

      #if defined(LUXAR_VOLUMETRIC)
      // 'volumetric' mode: emission–absorption (Max 1995). RGB carries
      // the self-screened emission — gammaColor·alpha is exactly what
      // additive adds to the framebuffer, times S(τ) = (1−e^(−τ))/τ
      // (the front of the ribbon absorbs its own back; the series
      // branch keeps S(0) = 1 exact — the κ=0 additive limit; constants
      // from ../_shared/volumetric, shared with the point/gsplat
      // twins); the output alpha is the physical absorption 1 − e^(−τ)
      // for the One / OneMinusSrcAlpha state.
      float volAlpha = 1.0 - exp(-tau);
      float screen = (tau < ${VOLUMETRIC_SERIES_TAU_THRESHOLD}) ? 1.0 - ${VOLUMETRIC_SERIES_C1} * tau + tau * tau / ${VOLUMETRIC_SERIES_C2_DIVISOR}.0
                                  : volAlpha / max(tau, ${VOLUMETRIC_TAU_EPS});
      fragColor = vec4(gammaColor * alpha * screen, volAlpha);
      #elif defined(LUXAR_MAX_RGB_CONTRIBUTION)
      // max-mode RGB premultiplication. With CustomBlending +
      // MaxEquation + OneFactor/OneFactor the source RGB isn't
      // multiplied by alpha at composite time, so a soft line in max
      // mode would render as a flat full-bright quad. Premultiply by
      // intensity*opacity here so the framebuffer max captures
      // contribution-weighted colour. Other modes keep alpha-weighted
      // output.
      float a = intensity * uOpacity;
      fragColor = vec4(gammaColor * a, a);
      #else
      vec3 finalColor = gammaColor;
      fragColor = vec4(finalColor, intensity * uOpacity);
      #endif
    }
  `;

export const LINE_SOURCE: ShaderSource = {
  name: 'line',
  webgl: { vertex: LINE_VERTEX_SHADER, fragment: LINE_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    // Read "uIsOrtho" from the uniform record at build time so the
    // projection-mode graph variant matches the camera the caller set
    // up. Live ortho/perspective flips on a long-lived material go
    // through "LineTSLMaterial.updateCameraParams", which calls
    // "rebuildGraph()" itself — this short-lived ShaderSource path
    // just needs the right variant at construction.
    // Default config otherwise — no toggles: colormap uniforms in the
    // record are IGNORED here (matching POINT_SOURCE). Consumers
    // needing USE_COLORMAP / LUXAR_MAX_RGB_CONTRIBUTION call
    // `lineWebGPUFactory(buildLineTSLNodesFromUniforms(u, { useColormap }),
    // { ...flags })` directly, as the parity harness does.
    const isOrtho = ((u.uIsOrtho?.value as number) ?? 0) === 1;
    // Same at build time for the join style. The GLSL twin carries it as the
    // runtime "uLineJoin" uniform, so a harness that pins one backend's
    // uniform record gets the matching graph variant out of the other —
    // without this the WebGPU build would silently ignore a pinned
    // "uLineJoin: 0" and draw mitred quads against unmitred GLSL ones.
    const join = lineJoinStyleFromUniform(u.uLineJoin?.value as number | undefined);
    const { lineWebGPUFactory, buildLineTSLNodesFromUniforms } =
      requireTslMaterials().factories.line;
    return lineWebGPUFactory(buildLineTSLNodesFromUniforms(u, {}), { isOrtho, join });
  },
};
