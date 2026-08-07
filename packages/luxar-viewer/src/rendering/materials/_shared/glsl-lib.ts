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
 * the ordering attributes: its pick id is `gl_VertexID` (mesh IS depth sorted,
 * but its ordering permutes `geometry.index` itself, so there is no
 * `aSortedIndex` indirection to read — spec §6.5), so injecting the whole
 * sorted-index block would declare two attributes the geometry does not
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
