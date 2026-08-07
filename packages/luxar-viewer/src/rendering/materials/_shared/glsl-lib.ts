/**
 * Shared GLSL helper snippets injected into the Points / Lines / GSplats
 * shaders.
 *
 * Keeping the sanitize helpers in one place lets us:
 *   - change the sanitize policy once (e.g., to also reject denormals)
 *     without hunting through three shader source strings;
 *   - test the GLSL-string content with a single regex assertion;
 *   - make the intent explicit at the call site.
 *
 * Helpers:
 *   - `isInvalidFloat(v)`         — true if v is NaN or +/-Inf.
 *   - `sanitizePositive(v, fb)`   — returns v if v > 0 and finite, else fb.
 *   - `sanitizeNonNegative(v, fb)` — returns v if v ≥ 0 and finite, else fb.
 *
 * Inject by prepending `GLSL_SANITIZE_FUNCTIONS` to a shader source
 * string before the main() block:
 *
 * ```ts
 * const vertexShader = `
 *   uniform vec2 uResolution;
 *   ${GLSL_SANITIZE_FUNCTIONS}
 *   void main() { ... }
 * `;
 * ```
 *
 * @module rendering/shaders/glsl-lib
 */

export const GLSL_SANITIZE_FUNCTIONS = `
bool isInvalidFloat(float v) {
  return isnan(v) || isinf(v);
}

float sanitizePositive(float v, float fallback) {
  return (isInvalidFloat(v) || v <= 0.0) ? fallback : v;
}

float sanitizeNonNegative(float v, float fallback) {
  return (isInvalidFloat(v) || v < 0.0) ? fallback : v;
}

// Per-element opacity sanitizer: NaN/Inf route to the 1.0 opaque
// identity (corruption stays LOUD), finite values clamp to [0, 1]
// (alpha is opacity, never HDR — Python pins the range at write; this
// guards hand-crafted zarr). The clamp keeps the zero boundary
// CONTINUOUS (a -1e-4 epsilon vanishes like +0.0 renders, instead of
// jumping to full opacity) and keeps the value mediump-varying-safe.
float sanitizeAlpha(float v) {
  return isInvalidFloat(v) ? 1.0 : clamp(v, 0.0, 1.0);
}
`;

/**
 * Unified perspective near-plane fade, shared by the point / line /
 * gsplat vertex shaders (visual + pick).
 *
 * Perspective: 0.0 behind the camera (viewZ >= 0 — camera looks down
 * -Z, and the quad expansion math flips/degenerates for such
 * vertices), smoothstep fade across [nearCull, 2*nearCull], else 1.0.
 * Ortho: always 1.0 — there is no 1/z singularity, and NDC near/far
 * clipping is the sole cull authority (an explicit vertex-level cull
 * under ortho WRONGLY hid in-frustum content in the near slab).
 *
 * Callers reject the vertex when the result < 0.01 and multiply the
 * surviving amplitude/alpha/brightness by it (no hard pop).
 */
export const GLSL_NEAR_FADE_FUNCTIONS = `
float perspectiveNearFade(int isOrtho, float viewZ, float nearCull) {
  if (isOrtho == 1) return 1.0;
  if (viewZ >= 0.0) return 0.0;
  return smoothstep(nearCull, nearCull * 2.0, -viewZ);
}
`;

/**
 * The pick buffer's 16-bit element-id split, as a standalone function of an
 * arbitrary index.
 *
 * Separate from {@link GLSL_SORTED_INDEX} because mesh needs the split WITHOUT
 * the ordering attributes: its pick id is `gl_VertexID` (mesh has no depth sort
 * and therefore no `aSortedIndex` indirection — spec §6.5), so injecting the
 * whole sorted-index block would declare two attributes the geometry does not
 * carry. Declaring an unbound attribute is not merely wasteful on WebGPU — the
 * vertex-buffer layout is cached from the attribute set at first draw.
 *
 * The split itself must NOT be written twice. Both halves have to agree with
 * `voteWinner`'s `high * 65536 + low` recombination exactly
 * (`picking-system/pick-render.ts`), and a second copy is a place for the shift
 * or the mask to drift where the only symptom is picks resolving to the wrong
 * element past 65,536 — silent, and only on large nodes.
 */
export const GLSL_ELEMENT_ID_SPLIT = `
// Element index split into two 16-bit halves, low in .x and high in .y.
// The pick pass carries the index through an RGBA32F buffer, and float32
// has a 24-bit mantissa — so a single float channel cannot represent
// consecutive indices past 16,777,216, while a node's capacity reaches
// 2^25 on a 32768-texel device. Both halves are <= 65535, hence exact,
// and the pick decoder recombines them (see picking-system/pick-render.ts).
// Kept in INT space: doing the split on a float would already have lost
// the bit it is meant to preserve.
vec2 luxarElementIdSplit(uint i) {
  return vec2(float(i & 0xFFFFu), float(i >> 16u));
}
`;

/**
 * Double-buffered draw-slot → storage-slot mapping (depth-sorting spec
 * §2.1 tier 3). Declares BOTH ordering attributes plus the slot selector,
 * and exposes `luxarSortedIndex()` as the single read point.
 *
 * A permutation must swap ATOMICALLY: a half-applied ordering is not a
 * reordering but a corrupt permutation (elements drawn twice / not at
 * all). So the coordinator streams each new ordering into the INACTIVE
 * attribute across frames and flips `uSortedIndexSlot` only once that
 * buffer holds the whole permutation — the attribute being read is
 * therefore always complete.
 *
 * Both attributes are ALWAYS present on the geometry, as two DISTINCT
 * buffers allocated together by `attachElementStorage` — never aliased
 * onto one and never materialised later. That is load-bearing on
 * WebGPU twice over: `RenderObject` dereferences a graph-referenced
 * attribute before its undefined guard, and the vertex-buffer layout is
 * cached from the attribute set at first draw and never rebuilt, so a
 * set that grows afterwards renders the scene black (`element-storage.ts`).
 *
 * `uSortedIndexSlot` is a RUNTIME uniform, never a define: a flip must
 * not recompile the program.
 *
 * Inject once per vertex shader that indexes an element texture, then
 * read the index via `luxarSortedIndex()`:
 *
 * ```ts
 * const vertexShader = `
 *   ${GLSL_SORTED_INDEX}
 *   void main() { int base = int(luxarSortedIndex()) * 4; ... }
 * `;
 * ```
 */
export const GLSL_SORTED_INDEX = `${GLSL_ELEMENT_ID_SPLIT}
in uint aSortedIndex;
in uint aSortedIndexB;
uniform int uSortedIndexSlot;

uint luxarSortedIndex() {
  return uSortedIndexSlot == 1 ? aSortedIndexB : aSortedIndex;
}

// The STORAGE slot's id parts — the id the rest of the pipeline (loaders,
// selection) addresses elements by, not the transient draw slot. Mesh reads
// the shared split directly instead, off gl_VertexID (spec §6.5).
vec2 luxarElementIdParts() {
  return luxarElementIdSplit(luxarSortedIndex());
}
`;

/**
 * Line joint-code helpers, shared by the visual and picking GLSL vertex stages
 * (both build the same screen-space quad, so both must read the code the same
 * way — see `rendering/line-geometry.ts` for the texel layout and
 * `wasm/rust/src/lines_clipping.rs::compute_joint_codes` for the encoding).
 *
 * `texel4.yz` carry a per-endpoint joint code: `0` free polyline end, `-1`
 * slice-clipped, `-2` degree->=3 hub, `+(slot + 1)` / `-(slot + 3)` naming the
 * partner segment's storage slot and which of its endpoints is the shared one.
 *
 * `luxarLineJointKeepsCap` is true only for the two codes that want the soft
 * endpoint cap kept — a free end and a hub, where several quads already stack.
 * Everything else suppresses it: a slice-clipped endpoint because no neighbour
 * will ever arrive there, and a slot-bearing code because a neighbouring quad
 * does meet it. Defaulting a slot-bearing code the other way is the #780 bead
 * chain (measured: an interior joint bottoms out at 0.5 instead of 1.0, and a
 * dense polyline loses ~40% of its total brightness).
 *
 * Codes are exact small integers out of an RGBA32F texel fetched without
 * filtering; the half-integer midpoints are for defensiveness only.
 */
export const GLSL_LINE_JOINT_CODE = `
bool luxarLineJointKeepsCap(float jointCode) {
  return (jointCode > -0.5 && jointCode < 0.5) || (jointCode < -1.5 && jointCode > -2.5);
}

// The endpoint cap multiplier implied by a joint code, before any join
// geometry refines it from the partner's screen-space direction.
float luxarLineJointCapSuppression(float jointCode) {
  return luxarLineJointKeepsCap(jointCode) ? 0.0 : 1.0;
}
`;

/**
 * Screen-space join geometry at degree-2 polyline joints (#790), shared by the
 * visual and picking GLSL vertex stages.
 *
 * Two segments meeting at a turn of angle theta leave an uncovered circular
 * sector of that angle on the OUTSIDE of the bend and double-cover a lens on
 * the inside: dark ticks along the convex edge of any thick curve, bright ticks
 * along the concave one. No per-endpoint intensity scalar can close the outer
 * wedge, because nothing rasterises there to shade — only geometry can.
 *
 * `luxarLineJoin` rotates this quad's end edge onto the shared miter edge so
 * the two quads TILE. Coverage becomes a partition, so there is nothing to sum
 * and every blending mode is correct by construction, with no axial profile and
 * no cap dimming.
 *
 * The two stages MUST build the same quad or a pick footprint stops matching
 * what the eye sees, so this lives here rather than being written twice.
 *
 * REQUIRED GLOBALS (same implicit-context pattern as `GLSL_SORTED_INDEX`):
 * `uLineTex`, `uResolution`, `uIsOrtho`, `modelViewMatrix`,
 * `projectionMatrix`, and `luxarSortedIndex()`. Include this block AFTER
 * those declarations — GLSL resolves names top-down. `uLineJoin` is declared
 * here, so an including shader must not declare it again.
 */
export const GLSL_LINE_JOIN = `
// Join style at degree-2 polyline joints: 0 none, 1 miter. Runtime uniform
// (not a #define) so the ?lineJoin= override never recompiles a program.
// See types/line-join.ts.
uniform float uLineJoin;

// Project a node-local position into the same pixel space the quad expansion
// works in. Returns xy = pixel coords (offset by a constant that cancels in
// the differences below) and z = view-space depth, so the caller can reject a
// partner that is behind the camera. Mirrors the main path's wGuard logic.
vec3 luxarLinePixelPos(vec3 localPos, float nearCullValue) {
  vec4 mv = modelViewMatrix * vec4(localPos, 1.0);
  vec4 clip = projectionMatrix * mv;
  float wG = (uIsOrtho == 1) ? 1.0 : nearCullValue;
  vec2 ndc = clip.xy / max(clip.w, wG);
  return vec3(ndc * (0.5 * uResolution), -mv.z);
}

// Rendered half-width AT ONE ENDPOINT, in pixels, for either projection.
//
// The main path's clampedPixelWidth is PER-VERTEX: it interpolates the width
// and the view depth at the vertex's own t, so the t=0 and t=1 corners of one
// quad see different values (the same property that forced the pathological
// cull to gate on a segment-constant max — issue #849). Feeding it to
// luxarLineJoin would make the join's width gate per-vertex, and both cap
// varyings are "flat": the two triangles of a quad provoke from different
// vertices, so a segment whose ends straddle the gate would resolve its cap
// from whichever corner happened to provoke, and WebGL and WGSL would disagree
// about which that is.
//
// Evaluated at an END this is segment-constant, and it agrees with
// clampedPixelWidth exactly at the corner that consumes it (t=1 interpolates to
// mvEnd and endW), so the geometry is unchanged. Both segments meeting at a
// joint also read the SAME shared vertex, so they still agree on the gate.
float luxarLineEndPixelWidth(float widthAtEnd, float viewZ, float nearCullValue) {
  return (uIsOrtho == 1)
    ? (widthAtEnd * uOrthoLineScale)
    : (widthAtEnd * uPerspectiveLineScale / max(-viewZ, nearCullValue));
}

// The corner offset and endpoint cap for one end of one segment.
//   .xy  this corner's offset in pixel space, ALWAYS valid: the plain
//        perpendicular half-width whenever no join applies.
//   .z   the endpoint cap suppression, or -1.0 meaning "no partner was
//        reached — keep the code-implied default".
vec3 luxarLineJoin(
  bool atEnd,             // this vertex sits at the segment's END (t == 1)
  bool reachesVertex,     // near-clipping did NOT move this endpoint
  float jointCode,        // texel4.y at the start, texel4.z at the end
  vec2 sharedNdc,         // NDC of the shared vertex (ndcStart / ndcEnd)
  vec2 lineDir,           // this segment's unit direction, pixel space
  float pixelLen,
  float clampedPixelWidth,
  float nearCull
) {
  vec2 perpendicular = vec2(-lineDir.y, lineDir.x);
  vec3 noJoin = vec3(perpendicular * clampedPixelWidth, -1.0);

  // The wedge has area ~theta*R^2/2 pixels, so below a couple of pixels of
  // half-width it is sub-pixel and invisible — and a line that thin already
  // sits on the 1.5 px floor with its intensity faded. Gating on width puts
  // the cost only where the benefit is: million-segment scenes are thin-line
  // scenes and skip this entirely.
  float joinMinHalfWidth = 2.0;
  if (uLineJoin < 0.5 || clampedPixelWidth <= joinMinHalfWidth) return noJoin;
  // A near-clipped endpoint was moved onto the nearCull plane, so it is no
  // longer AT its source vertex and no neighbour meets it there.
  if (!reachesVertex) return noJoin;

  // Sentinels: 0 free end, -1 slice-clipped, -2 degree->=3 hub. Only a
  // slot-bearing code names a partner. The texel writer already clamped any
  // code naming an UNWRITTEN slot back to the free-end sentinel, and the
  // kernel never emits a self-reference.
  //
  // The self-reference is still rejected here rather than assumed away. It
  // costs one integer compare inside a block already gated on rendered width,
  // and it guards the one failure mode this whole change exists to remove: a
  // quad mitered against itself has no partner to tile with, so its rotated
  // end edge rasterises as a flap sticking out of the tube. The guarantee also
  // comes from a DIFFERENT module than the consumer — and a texel writer is
  // not the only producer. The TSL parity harness writes joint codes by hand,
  // and did in fact carry a fixture that decoded to slot 0 (itself) in a
  // single-segment scene until this review re-specified it.
  bool partnerSharesItsStart = jointCode > 0.0;
  int partnerSlot = partnerSharesItsStart ? int(jointCode) - 1 : int(-jointCode) - 3;
  bool namesAPartner =
    (jointCode > 0.5 || jointCode < -2.5) && partnerSlot != int(luxarSortedIndex());
  if (!namesAPartner) return noJoin;

  // Decode: +(slot + 1) => the partner's START is the shared vertex;
  // -(slot + 3) => its END is.
  int lineTexW = textureSize(uLineTex, 0).x;
  int partnerBase = partnerSlot * 6;
  ivec2 pTexel = ivec2(partnerBase % lineTexW, partnerBase / lineTexW);
  // Only the partner's FAR endpoint is needed — the near one is this vertex,
  // already projected. One texelFetch, one projection.
  vec3 partnerFar = partnerSharesItsStart
    ? texelFetch(uLineTex, ivec2(pTexel.x + 1, pTexel.y), 0).xyz
    : texelFetch(uLineTex, pTexel, 0).xyz;
  vec3 farPx = luxarLinePixelPos(partnerFar, nearCull);
  vec2 sharedPx = sharedNdc * (0.5 * uResolution);
  // The partner runs FROM the shared vertex TO its far endpoint when it shares
  // its start, and the other way when it shares its end.
  vec2 partnerDelta = partnerSharesItsStart
    ? (farPx.xy - sharedPx)
    : (sharedPx - farPx.xy);
  float partnerLen = length(partnerDelta);
  bool partnerInFront = (uIsOrtho == 1) || (farPx.z >= nearCull);
  if (!partnerInFront || partnerLen <= 0.0001 || pixelLen <= 0.0001) return noJoin;

  // CANONICAL operand order — incoming edge first, outgoing second — so both
  // segments meeting here evaluate the same expression and, crucially, take
  // the same branch. A branch disagreement leaves one diagonal edge with
  // nothing to tile against, which rasterises as a flap sticking out of the
  // tube.
  vec2 partnerDir = partnerDelta / partnerLen;
  vec2 dirIn = atEnd ? lineDir : partnerDir;
  vec2 dirOut = atEnd ? partnerDir : lineDir;
  float turn = dot(dirIn, dirOut);

  // The endpoint cap, DERIVED here rather than stored. The kernel's old scalar
  // was clamp(-dot(away_a, away_b), 0, 1) with each away vector pointing from
  // the shared vertex back along its segment; away_mine = -lineDir and
  // away_partner = +partnerDir, so it is exactly clamp(dot(lineDir,
  // partnerDir), 0, 1) — the same dot the miter limit needs anyway. Deriving
  // it frees texel4.yz to carry the partner code, and it is measured in SCREEN
  // space, so unlike the stored data-space angle it tracks the camera (#795).
  float suppression = clamp(turn, 0.0, 1.0);

  // Miter limit: |M| / R = 1 / cos(theta/2) = sqrt(2 / (1 + turn)), so bail
  // past 2x (theta > 120 deg). The overshoot guard tests the AXIAL reach,
  // |M . dir| = R * tan(theta/2), NOT |M| (which is ~R always): gating on the
  // magnitude disables the join on every polyline whose segments are shorter
  // than twice the tube radius, i.e. exactly the dense-curve case this issue
  // is about. Both tests read the same operands from either side, so the two
  // quads always agree on whether this joint is mitred.
  float grow = sqrt(2.0 / max(1.0 + turn, 1e-6));
  float axialReach = clampedPixelWidth * sqrt(max(grow * grow - 1.0, 0.0));
  if (grow > 2.0 || axialReach > 0.5 * min(pixelLen, partnerLen)) {
    return vec3(perpendicular * clampedPixelWidth, suppression);
  }

  // Intersection of the two segments' +R offset lines. It reduces to R * perp
  // at a collinear joint, so a straight polyline is unchanged. The miter point
  // lies ON this segment's own +R offset line, which is why the resulting
  // trapezoid keeps vPerpNorm an exact perpendicular coordinate and the
  // super-Gaussian cross-section is untouched. A mitred joint tiles exactly,
  // so nothing may dim it.
  vec2 perpIn = vec2(-dirIn.y, dirIn.x);
  vec2 perpOut = vec2(-dirOut.y, dirOut.x);
  return vec3((perpIn + perpOut) * (clampedPixelWidth / (1.0 + turn)), 1.0);
}
`;
